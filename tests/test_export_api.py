"""Review export API behavior for combined and individual outputs."""

from pathlib import Path

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
    job_id = store.create(source, mode="action_only")
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
            mode=EditMode.ACTION_ONLY,
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
                "mode": "action_only",
                "export_clips": False,
                "export_joined": True,
            },
        )
        clips = client.post(
            f"/api/jobs/{job_id}/export",
            json={
                "mode": "action_only",
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
