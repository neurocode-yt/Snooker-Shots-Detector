"""Review export API behavior for combined and individual outputs."""

from pathlib import Path
import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.types import AnalysisResult, EditMode, ShotRecord, VideoMetadata


def test_review_can_export_combined_video_or_clips_separately(
    config, tmp_path, monkeypatch
):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    store = JobStore(config)
    job_id = store.create(source, mode="strict")
    store.save_analysis(
        AnalysisResult(
            job_id=job_id,
            source_path=str(source),
            metadata=VideoMetadata(
                path=str(source), duration=10.0, width=1280, height=720, fps=30.0
            ),
            shots=[
                ShotRecord(
                    shot_id=1,
                    cue_strike=2.0,
                    clip_start=1.0,
                    clip_end=6.0,
                    included=True,
                )
            ],
            mode=EditMode.STRICT,
            original_duration=10.0,
        )
    )

    requests = []

    def fake_export(_self, _result, output_dir, request):
        requests.append(request)
        output_dir = Path(output_dir)
        return ExportResult(
            joined_path=(output_dir / "highlights.mp4") if request.export_joined else None,
            clip_paths=[output_dir / "clips" / "shot_0001.mp4"]
            if request.export_clips
            else [],
        )

    monkeypatch.setattr("apps.api.main.Exporter.export", fake_export)

    with TestClient(create_app(config)) as client:
        combined = client.post(
            f"/api/jobs/{job_id}/export",
            json={
                "mode": "strict",
                "export_clips": False,
                "export_joined": True,
            },
        )
        clips = client.post(
            f"/api/jobs/{job_id}/export",
            json={
                "mode": "strict",
                "export_clips": True,
                "export_joined": False,
            },
        )
        empty = client.post(
            f"/api/jobs/{job_id}/export",
            json={"export_clips": False, "export_joined": False},
        )

    assert combined.status_code == 200
    assert combined.json()["download_url"] == (
        f"/api/jobs/{job_id}/download/highlights"
    )
    assert requests[0].export_joined is True
    assert requests[0].export_clips is False

    assert clips.status_code == 200
    assert clips.json()["joined"] is None
    assert clips.json()["download_url"] is None
    assert clips.json()["clip_count"] == 1
    assert requests[1].export_joined is False
    assert requests[1].export_clips is True

    assert empty.status_code == 400
    assert len(requests) == 2


def test_open_export_folder_reports_combined_video(config, tmp_path, monkeypatch):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = JobStore(config)
    job_id = store.create(source, mode="strict")
    export_dir = store.job_dir(job_id) / "export"
    export_dir.mkdir(parents=True)
    (export_dir / "highlights.mp4").touch()
    monkeypatch.setattr("apps.api.main._open_folder", lambda _path: True)

    with TestClient(create_app(config)) as client:
        response = client.post(f"/api/jobs/{job_id}/open-export-folder")

    assert response.status_code == 200
    payload = response.json()
    assert payload["combined_exists"] is True
    assert payload["clip_count"] == 0
    assert payload["folder"].endswith("export")


def test_null_lists_do_not_fail_len(config):
    """AnalysisResult deserialization with null lists must sanitize to empty lists."""
    data = {
        "job_id": "test-job",
        "source_path": "test.mp4",
        "metadata": {"path": "test.mp4"},
        "shots": None,
        "features": None,
        "scenes": None,
        "strike_candidates": None,
        "events": None,
    }
    result = AnalysisResult.model_validate(data)
    assert result.shots == []
    assert result.features == []
    assert result.scenes == []
    assert result.strike_candidates == []
    assert result.events == []
    assert len(result.shots) == 0


@pytest.mark.asyncio
async def test_export_keeps_progress_available_and_rejects_duplicate_export(config, tmp_path, monkeypatch):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = JobStore(config)
    job_id = store.create(source)
    store.save_analysis(AnalysisResult(job_id=job_id, source_path=str(source), metadata=VideoMetadata(path=str(source))))
    started, release = threading.Event(), threading.Event()

    def slow_export(*args):
        started.set()
        if not release.wait(3):
            raise RuntimeError("web request loop blocked during export")
        return ExportResult(clip_paths=[])

    monkeypatch.setattr("apps.api.main.Exporter.export", slow_export)
    transport = httpx.ASGITransport(app=create_app(config))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        export_task = asyncio.create_task(client.post(f"/api/jobs/{job_id}/export", json={}))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert not export_task.done()
            progress = await asyncio.wait_for(client.get(f"/api/jobs/{job_id}/progress"), 1)
            assert progress.status_code == 200
            duplicate = await client.post(f"/api/jobs/{job_id}/export", json={})
            assert duplicate.status_code == 409
        finally:
            release.set()
            response = await export_task
        assert response.status_code == 200
