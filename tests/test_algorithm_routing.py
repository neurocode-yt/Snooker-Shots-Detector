"""Algorithm selection cannot replace another algorithm's saved job."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.algorithms import create_analyzer
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.types import AnalysisResult, EditMode, ShotRecord, VideoMetadata


@pytest.fixture
def routing(config, tmp_path, monkeypatch):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    calls = []

    def run(algorithm, job_dir, source, job_id, progress):
        calls.append((algorithm, job_id, job_dir))
        result = AnalysisResult(
            job_id=job_id, source_path=str(source), metadata=VideoMetadata(path=str(source)),
            shots=[ShotRecord(shot_id=1, cue_strike=2, clip_start=0, clip_end=5)],
        )
        JobStore(config).save_analysis(result)
        progress(1.0, "ready_for_review", "Done")
        return result

    def classic(self, source, job_id, *, progress, **kwargs):
        return run("classic", self.job_dir, source, job_id, progress)

    class FakeDL:
        ready = True

        def __init__(self, cfg, job_dir):
            self.job_dir = Path(job_dir)

        @classmethod
        def capabilities(cls, cfg):
            return {"available": cls.ready, "reason": "Ready" if cls.ready else "Training required",
                    "model": "test temporal model"}

        def analyze(self, source, job_id, *, progress, **kwargs):
            return run("dl_algo", self.job_dir, source, job_id, progress)

    fake_pipeline = ModuleType("snooker_ai.dl.pipeline")
    fake_pipeline.DLAnalyzer = FakeDL
    monkeypatch.setitem(sys.modules, "snooker_ai.dl.pipeline", fake_pipeline)
    monkeypatch.setattr("apps.api.main.Analyzer.analyze", classic)
    return config, source, JobStore(config), calls, FakeDL


def test_capabilities_expose_both_algorithms_without_changing_edit_modes(routing):
    cfg, _, _, calls, _ = routing
    with TestClient(create_app(cfg)) as client:
        capabilities = client.get("/api/algorithms").json()["algorithms"]
    assert [item["id"] for item in capabilities] == ["classic", "dl_algo"]
    assert capabilities[1]["label"] == "DL Algo"
    assert capabilities[1]["available"] is True
    assert capabilities[1]["model"] == "test temporal model"
    assert calls == []
    assert list(EditMode) == [EditMode.STRICT]
    assert EditMode.from_string("natural") is EditMode.STRICT


@pytest.mark.parametrize("algorithm", [None, "classic", "dl_algo"])
def test_api_dispatches_only_selected_algorithm(routing, algorithm):
    cfg, source, store, calls, _ = routing
    body = {"source_path": str(source), "mode": "natural", "auto_export": False}
    if algorithm is not None:
        body["algorithm"] = algorithm
    with TestClient(create_app(cfg)) as client:
        response = client.post("/api/jobs", json=body)
    assert response.status_code == 200
    selected = algorithm or "classic"
    job_id = response.json()["job_id"]
    assert response.json()["mode"] == "strict"
    assert response.json()["algorithm"] == selected
    assert calls == [(selected, job_id, store.job_dir(job_id))]
    assert store.get_meta(job_id)["algorithm"] == selected
    assert store.get_meta(job_id)["status"] == "ready_for_review"


def test_old_job_without_algorithm_loads_and_restarts_as_classic(routing):
    cfg, source, store, calls, _ = routing
    job_id = store.create(source)
    meta = store.get_meta(job_id)
    del meta["algorithm"]
    store._write_meta(job_id, meta)
    before = store._meta_path(job_id).read_bytes()
    assert store.get_meta(job_id)["algorithm"] == "classic"
    assert store.list_jobs()[0]["algorithm"] == "classic"
    assert store._meta_path(job_id).read_bytes() == before
    with TestClient(create_app(cfg)) as client:
        response = client.post(f"/api/jobs/{job_id}/restart", json={"auto_export": False})
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert calls[0][0] == "classic"


def test_dl_analysis_uses_existing_exporter_to_create_download(routing, monkeypatch):
    cfg, source, store, calls, _ = routing

    def export(_self, result, out_dir, request):
        assert request.export_clips is False
        output = Path(out_dir) / "highlights.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"DL highlights")
        return ExportResult(joined_path=output)

    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    with TestClient(create_app(cfg)) as client:
        response = client.post("/api/jobs", json={"source_path": str(source), "algorithm": "dl_algo"})
        job_id = response.json()["job_id"]
        progress = client.get(f"/api/jobs/{job_id}/progress").json()
        assert progress["status"] == "completed"
        assert progress["algorithm"] == "dl_algo"
        assert client.get(progress["download_url"]).content == b"DL highlights"
    assert calls == [("dl_algo", job_id, store.job_dir(job_id))]


@pytest.mark.parametrize("endpoint", ["create", "restart"])
@pytest.mark.parametrize("old_algorithm,new_algorithm", [("classic", "dl_algo"), ("dl_algo", "classic")])
def test_algorithm_switch_creates_new_job_and_preserves_original_files(
    routing, endpoint, old_algorithm, new_algorithm,
):
    cfg, source, store, calls, _ = routing
    old_id = store.create(source, algorithm=old_algorithm)
    original = store.job_dir(old_id)
    (original / "export").mkdir()
    for name in ("analysis.json", "corrections.json", "export/highlights.mp4"):
        (original / name).write_bytes(f"original {name}".encode())
    snapshot = {path.relative_to(original): path.read_bytes() for path in original.rglob("*") if path.is_file()}
    with TestClient(create_app(cfg)) as client:
        if endpoint == "create":
            response = client.post("/api/jobs", json={
                "job_id": old_id, "algorithm": new_algorithm, "auto_export": False,
            })
        else:
            response = client.post(f"/api/jobs/{old_id}/restart", json={
                "algorithm": new_algorithm, "auto_export": False,
            })
    assert response.status_code == 200
    new_id = response.json()["job_id"]
    assert new_id != old_id
    assert response.json()["derived_from_job_id"] == old_id
    assert store.get_meta(new_id)["algorithm"] == new_algorithm
    assert calls == [(new_algorithm, new_id, store.job_dir(new_id))]
    assert {path.relative_to(original): path.read_bytes() for path in original.rglob("*") if path.is_file()} == snapshot


def test_restart_preserves_existing_dl_selection_when_algorithm_omitted(routing):
    cfg, source, store, calls, _ = routing
    job_id = store.create(source, algorithm="dl_algo")
    with TestClient(create_app(cfg)) as client:
        response = client.post(f"/api/jobs/{job_id}/restart", json={"auto_export": False})
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert calls[0][0] == "dl_algo"


@pytest.mark.parametrize("endpoint", ["create", "restart"])
@pytest.mark.parametrize("selection,status", [("unsupported", 422), ("dl_algo", 409)])
def test_rejected_algorithm_does_not_mutate_or_create_jobs(routing, endpoint, selection, status):
    cfg, source, store, calls, fake_dl = routing
    fake_dl.ready = False
    old_id = store.create(source)
    before = store._meta_path(old_id).read_bytes()
    with TestClient(create_app(cfg)) as client:
        if endpoint == "create":
            response = client.post("/api/jobs", json={"job_id": old_id, "algorithm": selection})
        else:
            response = client.post(f"/api/jobs/{old_id}/restart", json={"algorithm": selection})
    assert response.status_code == status
    assert len(store.list_jobs()) == 1
    assert store._meta_path(old_id).read_bytes() == before
    assert calls == []


def test_factory_rejects_unknown_algorithm_instead_of_using_classic(routing):
    cfg, _, store, _, _ = routing
    with pytest.raises(ValueError):
        create_analyzer(cfg, store.root / "job", "unknown")
    with pytest.raises(ValueError):
        store.create("input.mp4", algorithm="unknown")
    assert store.list_jobs() == []


@pytest.mark.parametrize("selection", [None, "dl_algo"])
def test_cli_algorithm_selection(routing, monkeypatch, selection):
    from snooker_ai.cli import app

    cfg, source, store, calls, _ = routing
    monkeypatch.setattr("snooker_ai.cli._cfg", lambda _: cfg)
    arguments = ["analyze", str(source)]
    if selection:
        arguments.extend(["--algorithm", selection])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert calls[0][0] == (selection or "classic")
    assert store.list_jobs()[0]["algorithm"] == (selection or "classic")


def test_cli_algorithm_switch_isolates_existing_job(routing, monkeypatch):
    from snooker_ai.cli import app

    cfg, source, store, calls, _ = routing
    monkeypatch.setattr("snooker_ai.cli._cfg", lambda _: cfg)
    old_id = store.create(source)
    original = store.job_dir(old_id) / "corrections.json"
    original.write_text(json.dumps({"edited": True}), encoding="utf-8")
    before = store._meta_path(old_id).read_bytes()
    result = CliRunner().invoke(app, ["analyze", str(source), "--job-id", old_id, "--algorithm", "dl_algo"])
    assert result.exit_code == 0, result.output
    assert len(store.list_jobs()) == 2
    assert calls[0][1] != old_id
    assert store._meta_path(old_id).read_bytes() == before
    assert json.loads(original.read_text(encoding="utf-8")) == {"edited": True}


def test_cli_rejects_output_directory_owned_by_other_algorithm(routing, monkeypatch):
    from snooker_ai.cli import app

    cfg, source, store, calls, _ = routing
    monkeypatch.setattr("snooker_ai.cli._cfg", lambda _: cfg)
    old_id = store.create(source)
    before = store._meta_path(old_id).read_bytes()
    result = CliRunner().invoke(app, [
        "analyze", str(source), "--algorithm", "dl_algo", "--output-dir", str(store.job_dir(old_id)),
    ])
    assert result.exit_code != 0
    assert calls == []
    assert len(store.list_jobs()) == 1
    assert store._meta_path(old_id).read_bytes() == before


def test_workflow_adds_dl_algo_without_removing_existing_options():
    root = Path(__file__).resolve().parents[1]
    html = (root / "apps/web/index.html").read_text(encoding="utf-8")
    for option in ('value="automatic">Automatic video', 'value="classic">Classic editor', 'value="dl_algo">DL Algo'):
        assert option in html
    script = (root / "apps/web/static/app.js").read_text(encoding="utf-8")
    assert 'fetch("/api/algorithms")' in script
    assert 'workflow.value === "dl_algo" ? "dl_algo" : "classic"' in script
