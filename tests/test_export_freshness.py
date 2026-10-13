"""A finished download must reflect the user's saved review changes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.jobs.lease import JobLease
from snooker_ai.types import ShotRecord, ShotSelection, ShotUpdate
from tests.test_pot_selection import api_setup


def test_review_edit_preserves_old_file_but_requires_export_before_download(
    config, tmp_path, monkeypatch
):
    store, job, _ = api_setup(config, tmp_path)
    path = store.job_dir(job) / "export" / "highlights.mp4"
    path.parent.mkdir()
    path.write_bytes(b"previous video")
    store.update_progress(job, 1, "completed", "Ready")

    def export(self, analysis, output_dir, request):
        assert analysis.shots[0].clip_end == 14
        output = Path(output_dir) / "highlights.mp4"
        output.write_bytes(b"updated highlights")
        return ExportResult(joined_path=output)

    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    with TestClient(create_app(config)) as client:
        assert client.get(f"/api/jobs/{job}/progress").json()["download_url"]
        assert client.patch(f"/api/jobs/{job}/shots/1", json={"clip_end": 14}).status_code == 200
        progress = client.get(f"/api/jobs/{job}/progress").json()
        assert progress["export_needs_update"]
        assert progress["download_url"] is None
        assert client.get(f"/api/jobs/{job}/download/highlights").status_code == 409
        assert path.read_bytes() == b"previous video"
        response = client.post(f"/api/jobs/{job}/export", json={"export_clips": False})
        assert response.status_code == 200
        progress = client.get(f"/api/jobs/{job}/progress").json()
        assert not progress["export_needs_update"]
        assert client.get(progress["download_url"]).content == b"updated highlights"


@pytest.mark.parametrize("dirty", [False, True])
def test_individual_clip_export_preserves_the_combined_video_state(
    config, tmp_path, monkeypatch, dirty
):
    store, job, _ = api_setup(config, tmp_path)
    path = store.job_dir(job) / "export" / "highlights.mp4"
    path.parent.mkdir()
    path.write_bytes(b"combined video")
    meta = store.get_meta(job)
    meta.update(auto_export_ready=True, export_needs_update=dirty)
    store._write_meta(job, meta)
    store.update_progress(job, 1, "completed", "Ready")
    monkeypatch.setattr("apps.api.main.Exporter.export", lambda *args: ExportResult())
    with TestClient(create_app(config)) as client:
        response = client.post(
            f"/api/jobs/{job}/export", json={"export_joined": False, "export_clips": True}
        )
        assert response.status_code == 200
        progress = client.get(f"/api/jobs/{job}/progress").json()
        assert progress["auto_export_ready"]
        assert progress["export_needs_update"] is dirty
        assert bool(progress["download_url"]) is not dirty
        assert path.read_bytes() == b"combined video"


@pytest.mark.parametrize(
    "method, suffix, body",
    [
        ("PATCH", "shots/1", {"included": False}),
        ("DELETE", "shots/1", None),
        ("POST", "shots", {"cue_strike": 23, "clip_start": 21, "clip_end": 27}),
        ("POST", "shots/merge", {"shot_ids": [1, 2]}),
        ("POST", "shots/1/split", {"at_time": 12}),
    ],
)
def test_review_writes_wait_for_workers_and_exports(config, tmp_path, method, suffix, body):
    store, job, _ = api_setup(config, tmp_path)
    analysis = (store.job_dir(job) / "analysis.json").read_bytes()
    lease = JobLease(store.job_dir(job))
    assert lease.acquire()
    try:
        with TestClient(create_app(config)) as client:
            response = client.request(method, f"/api/jobs/{job}/{suffix}", json=body)
            assert response.status_code == 409
            assert "processing" in response.json()["detail"]
    finally:
        lease.release()
    assert (store.job_dir(job) / "analysis.json").read_bytes() == analysis


def test_timeline_revision_only_changes_when_saved_shots_change(config, tmp_path):
    store, job, result = api_setup(config, tmp_path)
    store.update_progress(job, 1, "completed", "Ready")
    with TestClient(create_app(config)) as client:
        first = client.get(f"/api/jobs/{job}/progress").json()["timeline_revision"]
        store.update_progress(job, 1, "completed", "Updated message")
        assert client.get(f"/api/jobs/{job}/progress").json()["timeline_revision"] == first
        result.shots[0].included = False
        store.save_analysis(result)
        assert client.get(f"/api/jobs/{job}/progress").json()["timeline_revision"] != first


def test_review_counts_follow_saved_selection_and_added_shots(config, tmp_path):
    store, job, result = api_setup(config, tmp_path)
    result.selection = ShotSelection.POTS_AND_FOULS
    result.pot_selection_summary = {"included": 1, "pot": 1}
    store.save_analysis(result)
    store.update_shot(job, 1, ShotUpdate(included=False))
    assert store.get_meta(job)["pot_selection_summary"]["included"] == 0
    assert store.load_analysis(job).pot_selection_summary["included"] == 0
    store.add_shot(job, ShotRecord(shot_id=0, cue_strike=23, clip_start=21, clip_end=27))
    assert store.get_meta(job)["shots_detected"] == 2
    assert store.get_meta(job)["pot_selection_summary"]["included"] == 1
