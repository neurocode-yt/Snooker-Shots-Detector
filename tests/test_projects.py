"""Project-family naming and deletion operate only on verified owned resources."""

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.projects import ProjectConflict, ProjectLibrary, families
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.capcut import CapCutInstallation, create_capcut_project
from snooker_ai.types import AnalysisResult, ShotRecord, VideoMetadata


@pytest.fixture
def library(config, tmp_path, monkeypatch):
    config._data["paths"].update(
        data_dir=str(tmp_path),
        jobs_dir=str(tmp_path / "jobs"),
        uploads_dir=str(tmp_path / "uploads"),
        outputs_dir=str(tmp_path / "outputs"),
    )
    config._data["capcut"] = {"projects_dir": str(tmp_path / "drafts")}
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr("snooker_ai.jobs.projects.capcut_running", lambda: False)
    return ProjectLibrary(JobStore(config))


def job(library, source=None, selection="all", algorithm="classic"):
    store = library.store
    source = source or store.uploads / "Match.mp4"
    source.parent.mkdir(exist_ok=True)
    if not source.exists():
        source.write_bytes(b"original source")
    jid = store.create(source, selection=selection, algorithm=algorithm)
    store.update_progress(jid, 1, "completed", "Ready", shots_detected=1)
    result = AnalysisResult(
        job_id=jid,
        source_path=str(source),
        metadata=VideoMetadata(path=str(source), duration=10, fps=25, width=1280, height=720),
        shots=[ShotRecord(shot_id=1, cue_strike=2, clip_start=1, clip_end=4)],
    )
    store.save_analysis(result)
    for name in [
        "export/clips/shot_1.mp4",
        "export/highlights.mp4",
        "proxy/proxy.mp4",
        "shorts/batch/shot.mp4",
    ]:
        resource = store.job_dir(jid) / name
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(b"generated resource")
    return jid, result


def draft(library, result):
    installation = CapCutInstallation(
        Path("CapCut.exe"), Path(library.store.config.get("capcut.projects_dir")), "9.6.0.4097"
    )
    return create_capcut_project(result, library.store.config, installation)


def test_name_persists_across_versions_future_runs_and_does_not_touch_media(library):
    jid, result = job(library)
    other, _ = job(library, Path(result.source_path), "pots_only")
    before = (library.store.job_dir(jid) / "analysis.json").read_bytes()
    library.rename(jid, "  Final <match> & friends  ")
    future, _ = job(library, Path(result.source_path), "pots_and_fouls")
    renamed = ProjectLibrary(library.store).annotate(library.store.list_jobs())
    assert {j["job_id"] for j in renamed} == {jid, other, future}
    assert {j["display_name"] for j in renamed} == {"Final <match> & friends"}
    assert len({j["project_id"] for j in renamed}) == 1
    assert (library.store.job_dir(jid) / "analysis.json").read_bytes() == before
    with pytest.raises(ValueError):
        library.rename(jid, "   ")


def test_family_identity_handles_legacy_replacement_and_windows_case():
    a = {"job_id": "a", "source_path": "G:/Matches/A.mp4"}
    b = {
        **a,
        "job_id": "b",
        "source_fingerprint": {"path": "g:/matches/a.mp4", "size": 10, "mtime_ns": 20},
    }
    c = {**b, "job_id": "c", "source_fingerprint": {**b["source_fingerprint"], "mtime_ns": 21}}
    assert len(families([a, b])) == 1
    assert len(families([a, b, c])) == 3
    assert len(families([b, {**b, "job_id": "d", "algorithm": "dl_algo"}])) == 2


def test_delete_removes_all_versions_resources_repeated_and_renamed_capcut_drafts(library):
    jid, result = job(library)
    version, _ = job(library, Path(result.source_path), "pots_only")
    first, second = draft(library, result), draft(library, result)
    renamed = Path(second["project_path"]).with_name("Renamed in CapCut")
    Path(second["project_path"]).rename(renamed)
    unrelated, unrelated_result = job(library, library.store.uploads / "Other.mp4")
    untouched = draft(library, unrelated_result)
    library.rename(jid, "Named project")
    plan = library.plan([version])
    assert set(plan["job_ids"]) == {jid, version}
    assert len(plan["capcut"]) == 2
    assert plan["project_names"] == ["Named project"]
    outcome = library.delete(plan, plan["token"])
    assert outcome["deleted_projects"] == 1 and outcome["deleted_versions"] == 2
    assert not outcome["cleanup_pending"]
    for path in [
        library.store.job_dir(jid),
        library.store.job_dir(version),
        Path(result.source_path),
        Path(first["project_path"]),
        renamed,
    ]:
        assert not path.exists()
    assert library.store.job_dir(unrelated).is_dir()
    assert Path(untouched["project_path"]).is_dir()
    index = json.loads((Path(untouched["project_path"]).parent / "root_meta_info.json").read_text())
    assert [e["draft_id"] for e in index["all_draft_store"]] == [untouched["project_id"]]
    assert not library.names()


def test_shared_upload_and_external_original_survive_until_last_project(library, tmp_path):
    jid, result = job(library)
    other, _ = job(library, Path(result.source_path), algorithm="dl_algo")
    plan = library.plan([jid])
    assert plan["kept_sources"] == [str(Path(result.source_path))]
    library.delete(plan, plan["token"])
    assert Path(result.source_path).exists()
    plan = library.plan([other])
    library.delete(plan, plan["token"])
    assert not Path(result.source_path).exists()
    external = tmp_path / "external.mp4"
    jid, _ = job(library, external)
    plan = library.plan([jid])
    library.delete(plan, plan["token"])
    assert external.exists()


def test_preprocessed_input_is_preserved_while_other_project_uses_it(library):
    original = library.store.uploads / "Original.mp4"
    original.write_bytes(b"original")
    cleaned = library.store.uploads / "Cleaned.mp4"
    receipt = cleaned.with_suffix(".mp4.source.json")
    receipt.write_text(json.dumps({"source_path": str(original)}))
    jid, _ = job(library, cleaned)
    other, _ = job(library, original)
    plan = library.plan([jid])
    assert str(original) in plan["kept_sources"]
    library.delete(plan, plan["token"])
    assert original.exists() and not cleaned.exists() and not receipt.exists()
    assert library.store.job_dir(other).exists()


def test_unrelated_manual_capcut_project_prevents_removing_shared_media(library):
    jid, result = job(library)
    manual = Path(library.store.config.get("capcut.projects_dir")) / "Manual"
    manual.mkdir(parents=True)
    (manual / "draft_meta_info.json").write_text(
        json.dumps({"draft_materials": [{"value": [{"file_Path": result.source_path}]}]})
    )
    plan = library.plan([jid])
    library.delete(plan, plan["token"])
    assert Path(result.source_path).exists() and manual.is_dir()


def test_active_worker_capcut_and_changed_confirmation_are_rejected(library, monkeypatch):
    jid, result = job(library)
    lease = JobLease(library.store.job_dir(jid))
    assert lease.acquire()
    try:
        plan = library.plan([jid])
        assert plan["issues"]
        with pytest.raises(ProjectConflict):
            library.delete(plan, plan["token"])
    finally:
        lease.release()
    draft(library, result)
    monkeypatch.setattr("snooker_ai.jobs.projects.capcut_running", lambda: True)
    plan = library.plan([jid])
    assert "Close CapCut" in plan["issues"][0]
    monkeypatch.setattr("snooker_ai.jobs.projects.capcut_running", lambda: False)
    before = library.plan([jid])
    job(library, Path(result.source_path), "pots_only")
    with pytest.raises(ProjectConflict):
        library.delete(library.plan([jid]), before["token"])
    assert library.store.job_dir(jid).exists()


def test_staging_error_rolls_back_everything(library, monkeypatch):
    jid, result = job(library)
    project = draft(library, result)
    plan = library.plan([jid])
    original = Path.rename

    def fail(path, target):
        if path == library.store.job_dir(jid):
            raise PermissionError("locked folder")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(PermissionError):
        library.delete(plan, plan["token"])
    assert Path(result.source_path).exists() and Path(project["project_path"]).exists()
    assert library.store.get_meta(jid)
    assert not list(library.store.root.rglob(".snooker-delete-*"))


def test_index_conflict_does_not_overwrite_another_projects_new_entry(library, monkeypatch):
    jid, result = job(library)
    project = draft(library, result)
    index = Path(project["project_path"]).parent / "root_meta_info.json"
    plan = library.plan([jid])
    rename = Path.rename

    def concurrent_registration(path, target):
        moved = rename(path, target)
        if path == library.store.job_dir(jid):
            data = json.loads(index.read_text())
            data["all_draft_store"].append({"draft_id": "another-new-project"})
            index.write_text(json.dumps(data))
        return moved

    monkeypatch.setattr(Path, "rename", concurrent_registration)
    with pytest.raises(ProjectConflict):
        library.delete(plan, plan["token"])
    ids = {entry["draft_id"] for entry in json.loads(index.read_text())["all_draft_store"]}
    assert ids == {project["project_id"], "another-new-project"}
    assert library.store.job_dir(jid).is_dir()


def test_cleanup_failure_is_durable_and_retry_removes_all_resources(library, monkeypatch):
    jid, _ = job(library)
    plan = library.plan([jid])
    with monkeypatch.context() as patch:
        patch.setattr(
            "snooker_ai.jobs.projects.shutil.rmtree",
            lambda _: (_ for _ in ()).throw(PermissionError("locked")),
        )
        assert library.delete(plan, plan["token"])["cleanup_pending"]
    assert library.store.list_jobs() == []
    assert ProjectLibrary(library.store).cleanup() == 0
    assert not list(library.store.root.glob(".deletions/*.json"))
    assert not list(library.store.root.glob(".snooker-delete-*"))


def test_path_escape_and_unverified_capcut_ownership_block_delete(library):
    jid, result = job(library)
    project = draft(library, result)
    marker = Path(project["project_path"]) / "snooker_shots.json"
    marker.write_text(json.dumps({"job_id": jid, "source": "/different.mp4"}))
    with pytest.raises(ProjectConflict):
        library.plan([jid])
    with pytest.raises(ProjectConflict):
        library.plan(["../outside"])


def test_deletion_markers_block_processing_queue_and_shorts_leases(library):
    jid, _ = job(library)
    directory = library.store.job_dir(jid)
    marker = library.store.root / ".deleting" / jid
    marker.parent.mkdir()
    marker.touch()
    assert not JobLease(directory).acquire()
    assert not JobLease(directory, ".queue.lock").acquire()
    assert not JobLease(directory / "shorts", ".shorts.lock").acquire()
    marker.unlink()
    lease = JobLease(directory)
    assert lease.acquire()
    lease.release()


def test_interrupted_staging_restores_project_and_clears_stale_marker(library):
    jid, result = job(library)
    original = library.store.job_dir(jid)
    staged = original.with_name(".snooker-delete-interrupted-" + original.name)
    original.rename(staged)
    marker = library.store.root / ".deleting" / jid
    marker.parent.mkdir()
    marker.touch()
    journal = library.store.root / ".deletions" / "interrupted.json"
    journal.parent.mkdir()
    journal.write_text(
        json.dumps(
            {
                "state": "preparing",
                "moves": [[str(original), str(staged)]],
                "indexes": {},
                "names": {},
                "job_ids": [jid],
            }
        )
    )
    assert library.cleanup() == 0
    assert original.exists() and not staged.exists() and not marker.exists()
    assert Path(result.source_path).exists()


def test_legacy_name_survives_a_later_fingerprint(library):
    jid, result = job(library)
    meta = library.store.get_meta(jid)
    meta.pop("source_fingerprint")
    library.store._write_meta(jid, meta)
    library.rename(jid, "Legacy name")
    child, _ = job(library, Path(result.source_path), "pots_only")
    assert library.info(child)["display_name"] == "Legacy name"


def test_project_api_names_and_confirmed_deletion(library, monkeypatch):
    jid, result = job(library)
    monkeypatch.setattr("apps.api.main.launch_capcut", lambda _: None)
    install = CapCutInstallation(
        Path("CapCut.exe"), Path(library.store.config.get("capcut.projects_dir")), "9.6.0.4097"
    )
    monkeypatch.setattr("apps.api.main.find_capcut", lambda _: install)
    with TestClient(create_app(library.store.config)) as client:
        assert (
            client.patch(
                f"/api/projects/{jid}",
                json={"name": "My_match"},
                headers={"Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert client.patch(f"/api/projects/{jid}", json={"name": "My_match"}).status_code == 200
        assert client.get(f"/api/jobs/{jid}").json()["display_name"] == "My_match"
        assert client.get("/api/jobs").json()[0]["display_name"] == "My_match"
        capcut = client.post(f"/api/jobs/{jid}/open-capcut").json()
        assert capcut["project_name"].startswith("Snooker - My_match - ")
        assert client.post("/api/projects/delete", json={"job_ids": [jid]}).status_code == 400
        preview = client.post("/api/projects/delete-preview", json={"job_ids": [jid]}).json()
        assert preview["project_names"] == ["My_match"]
        response = client.post(
            "/api/projects/delete", json={"job_ids": [jid], "token": preview["token"]}
        )
        assert response.status_code == 200, response.text
        assert response.json()["capcut_projects"] == 1
        assert not Path(result.source_path).exists()
        assert client.get("/api/jobs").json() == []


def test_api_rejects_project_in_use_without_deleting_resources(library):
    from apps.api.main import _locks

    jid, result = job(library)
    with TestClient(create_app(library.store.config)) as client:
        preview = client.post("/api/projects/delete-preview", json={"job_ids": [jid]}).json()
        lock = _locks.setdefault(jid, threading.Lock())
        lock.acquire()
        try:
            response = client.post(
                "/api/projects/delete", json={"job_ids": [jid], "token": preview["token"]}
            )
            assert response.status_code == 409
            assert Path(result.source_path).exists()
        finally:
            lock.release()
