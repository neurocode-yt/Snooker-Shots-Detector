"""Completed-video actions reuse detection and keep pots and foul points."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.pot_detection.selector import PotSelector
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.types import PotStatus, ShotSelection, ShotUpdate
from tests.test_pot_selection import api_setup, fixture_result, readings


@pytest.mark.parametrize("penalty", [4, 5, 6, 7])
def test_points_workflow_keeps_fouls_while_pots_only_stays_unchanged(
    config, tmp_path, monkeypatch, penalty
):
    result = fixture_result(tmp_path)
    selector = PotSelector(config, tmp_path)
    samples = readings((20, 12), end=9.5) + readings((20, 12 + penalty), start=11, active=1)
    monkeypatch.setattr(selector, "scan", lambda *args, **kwargs: samples)
    selector.apply(result)
    assert result.shots[0].pot_status is PotStatus.FOUL
    assert not result.shots[0].included
    selector.apply(result, selection=ShotSelection.POTS_AND_FOULS)
    assert result.shots[0].included
    assert result.pot_selection_summary["foul"] == 1


def test_points_increase_without_turn_marker_is_kept_without_claiming_pot_or_foul(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    selector = PotSelector(config, tmp_path)
    samples = readings((20, 12), active=None, end=9.5) + readings((27, 12), active=None, start=11)
    monkeypatch.setattr(selector, "scan", lambda *args, **kwargs: samples)
    selector.apply(result, selection=ShotSelection.POTS_AND_FOULS)
    assert result.shots[0].pot_status is PotStatus.POINTS
    assert result.shots[0].included
    assert result.pot_selection_summary["points"] == 1


@pytest.mark.parametrize(
    "samples", [[], readings((20, 12)), readings((20, 12), end=9.5) + readings((0, 0), start=11)]
)
def test_points_workflow_excludes_misses_unreadable_scores_and_resets(
    config, tmp_path, monkeypatch, samples
):
    result = fixture_result(tmp_path)
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *args, **kwargs: samples)
    selector.apply(result, selection=ShotSelection.POTS_AND_FOULS)
    assert not result.shots[0].included


@pytest.mark.parametrize("active", [0, None])
def test_corrected_foul_points_are_not_kept(config, tmp_path, monkeypatch, active):
    result = fixture_result(tmp_path)
    selector = PotSelector(config, tmp_path)
    samples = readings((20, 12), end=9.5, active=active)
    samples += readings((20, 16), start=11, end=13, active=active)
    samples += readings((20, 12), start=14, active=active)
    monkeypatch.setattr(selector, "scan", lambda *args, **kwargs: samples)
    selector.apply(result, selection=ShotSelection.POTS_AND_FOULS)
    assert result.shots[0].pot_status is PotStatus.UNKNOWN
    assert not result.shots[0].included
    assert result.shots[0].evidence["pot_confirmation"]["reason"] == "score_corrected_back"


def prepare(config, tmp_path, monkeypatch):
    store, parent, result = api_setup(config, tmp_path)
    result.analysis_signature = "old detector policy"
    store.save_analysis(result)
    store.update_progress(parent, 1.0, "completed", "Original video ready")
    parent_export = store.job_dir(parent) / "export" / "highlights.mp4"
    parent_export.parent.mkdir()
    parent_export.write_bytes(b"original highlights")
    score_cache = store.job_dir(parent) / "pot_score_cache" / "window_0000.json"
    score_cache.parent.mkdir()
    score_cache.write_text(json.dumps({"fixture": "saved score evidence"}), encoding="utf-8")
    cfg_analysis = config._data["analysis"]
    cfg_analysis["refine_fps"] = 12  # old detector settings must not trigger a full rerun
    monkeypatch.setattr(
        "apps.api.main.create_analyzer",
        lambda *args, **kwargs: pytest.fail("completed-video action must skip the detector"),
    )
    scans = []

    def scan(self, analysis, progress=None):
        scans.append(analysis.job_id)
        assert (
            self.job_dir / "pot_score_cache" / "window_0000.json"
        ).read_bytes() == score_cache.read_bytes()
        return readings((20, 12), end=9.5) + readings((20, 16), start=11, active=1)

    monkeypatch.setattr(PotSelector, "scan", scan)
    exports = []

    def export(self, analysis, output_dir, request):
        exports.append(analysis.job_id)
        assert analysis.shots[0].included and analysis.shots[0].pot_status is PotStatus.FOUL
        output = Path(output_dir) / "highlights.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"points highlights")
        return ExportResult(joined_path=output)

    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    return store, parent, scans, exports


def test_completed_video_click_uses_slim_timeline_skips_detection_and_preserves_original(
    config, tmp_path, monkeypatch
):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    original = {
        p.relative_to(store.job_dir(parent)): p.read_bytes()
        for p in store.job_dir(parent).rglob("*")
        if p.is_file() and p.name != ".processing.lock"
    }
    original_load = JobStore.load_analysis

    def forbid_parent_features(self, job_id):
        assert job_id != parent, "parent dense features must not be loaded"
        return original_load(self, job_id)

    monkeypatch.setattr(JobStore, "load_analysis", forbid_parent_features)
    with TestClient(create_app(config)) as client:
        meta = client.get(f"/api/jobs/{parent}/progress").json()
        assert meta["can_create_points"] is True
        response = client.post(f"/api/jobs/{parent}/points-highlights")
        assert response.status_code == 200
        child = response.json()["job_id"]
        assert child != parent
        assert client.app.state.match_queue.wait_idle()
        progress = client.get(f"/api/jobs/{child}/progress").json()
        assert progress["status"] == "completed"
        assert progress["selection"] == "pots_and_fouls"
        assert progress["reuse_saved_detection"] is True
        assert client.get(progress["download_url"]).content == b"points highlights"
    assert scans == exports == [child]
    assert original_load(store, child).features == []
    assert original == {
        p.relative_to(store.job_dir(parent)): p.read_bytes()
        for p in store.job_dir(parent).rglob("*")
        if p.is_file() and p.name != ".processing.lock"
    }


def test_second_click_returns_finished_variant_instead_of_scanning_or_rendering_again(
    config, tmp_path, monkeypatch
):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        first = client.post(f"/api/jobs/{parent}/points-highlights").json()
        assert client.app.state.match_queue.wait_idle()
        second = client.post(f"/api/jobs/{parent}/points-highlights").json()
    assert second["job_id"] == first["job_id"]
    assert second["status"] == "ready" and second["download_url"]
    assert scans == exports == [first["job_id"]]
    assert len(store.list_jobs()) == 2


def test_legacy_completed_job_without_source_fingerprint_reuses_saved_detection(
    config, tmp_path, monkeypatch
):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    meta = store.get_meta(parent)
    meta.pop("source_fingerprint")
    store._write_meta(parent, meta)
    with TestClient(create_app(config)) as client:
        response = client.post(f"/api/jobs/{parent}/points-highlights")
        assert response.status_code == 200
        child = response.json()["job_id"]
        assert client.app.state.match_queue.wait_idle()
        assert client.get(f"/api/jobs/{child}/progress").json()["status"] == "completed"
    assert scans == exports == [child]


def test_upgrade_from_pots_only_restores_fouls_and_preserves_explicit_review_choices(
    config, tmp_path, monkeypatch
):
    store, parent, _, _ = prepare(config, tmp_path, monkeypatch)
    result = store.load_analysis(parent)
    result.selection = ShotSelection.POTS_ONLY
    result.shots[0].included = False
    result.shots[0].evidence["pots_only_base_included"] = True
    store.save_analysis(result)
    first_child = store.create(result.source_path, selection=ShotSelection.POTS_AND_FOULS)
    seeded = store.seed_selection_analysis(first_child, parent)
    assert seeded.shots[0].included
    store.update_shot(parent, 1, ShotUpdate(included=False))
    second_child = store.create(result.source_path, selection=ShotSelection.POTS_AND_FOULS)
    seeded = store.seed_selection_analysis(second_child, parent)
    selector = PotSelector(config, store.job_dir(second_child))
    selector.apply(seeded, selection=ShotSelection.POTS_AND_FOULS)
    assert seeded.shots[0].pot_status is PotStatus.FOUL
    assert not seeded.shots[0].included
    assert seeded.shots[0].evidence["pots_only_manual_override"]


def test_source_replacement_is_rejected_without_starting_work(config, tmp_path, monkeypatch):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    Path(store.get_meta(parent)["source_path"]).write_bytes(b"replacement source")
    with TestClient(create_app(config)) as client:
        response = client.post(f"/api/jobs/{parent}/points-highlights")
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]
    assert not scans and not exports
    assert len(store.list_jobs()) == 1


def test_edited_parent_starts_a_new_variant(config, tmp_path, monkeypatch):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        first = client.post(f"/api/jobs/{parent}/points-highlights").json()
        assert client.app.state.match_queue.wait_idle()
        store.update_shot(parent, 1, ShotUpdate(clip_end=14))
        second = client.post(f"/api/jobs/{parent}/points-highlights").json()
    assert first["job_id"] != second["job_id"]
    assert len(scans) == len(exports) == 2


def test_running_variant_is_not_queued_twice(config, tmp_path, monkeypatch):
    store, parent, _, _ = prepare(config, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("apps.api.main._run_analysis", lambda *args: calls.append(args))
    with TestClient(create_app(config)) as client:
        first = client.post(f"/api/jobs/{parent}/points-highlights").json()
        lease = JobLease(store.job_dir(first["job_id"]))
        assert lease.acquire()
        try:
            second = client.post(f"/api/jobs/{parent}/points-highlights").json()
        finally:
            lease.release()
    assert second == {"job_id": first["job_id"], "status": "running"}
    assert len(calls) == 1


def test_export_retry_continues_slim_variant_without_rescanning_detector(
    config, tmp_path, monkeypatch
):
    store, parent, scans, exports = prepare(config, tmp_path, monkeypatch)
    export_original = __import__("apps.api.main", fromlist=["Exporter"]).Exporter.export
    attempts = []

    def interrupted_export(self, *args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("export interrupted")
        return export_original(self, *args, **kwargs)

    monkeypatch.setattr("apps.api.main.Exporter.export", interrupted_export)
    with TestClient(create_app(config)) as client:
        first = client.post(f"/api/jobs/{parent}/points-highlights").json()
        assert client.app.state.match_queue.wait_idle()
        assert client.get(f"/api/jobs/{first['job_id']}/progress").json()["status"] == "failed"
        second = client.post(f"/api/jobs/{parent}/points-highlights").json()
        assert client.app.state.match_queue.wait_idle()
        assert second["job_id"] == first["job_id"]
        assert client.get(f"/api/jobs/{second['job_id']}/progress").json()["status"] == "completed"
    assert len(attempts) == 2
    assert exports == [first["job_id"]]
