"""Analysis can create the finished video without a review/export request."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.types import AnalysisResult, ShotRecord, VideoMetadata


@pytest.mark.parametrize("auto_export", [True, False])
def test_analysis_finishes_automatically_even_with_uncertain_diagnostics(
    config, tmp_path, monkeypatch, auto_export,
):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    exports = []

    def analyze(_self, source, job_id, *, progress, **kwargs):
        result = AnalysisResult(
            job_id=job_id, source_path=str(source), metadata=VideoMetadata(path=str(source)),
            shots=[ShotRecord(shot_id=1, cue_strike=2, clip_start=0, clip_end=5, manual_review_required=True)],
        )
        JobStore(config).save_analysis(result)
        progress(1.0, "ready_for_review", "Detected one shot")
        return result

    def export(_self, result, out_dir, request):
        exports.append(request)
        # Progress remains in processing until rendering has actually finished.
        assert JobStore(config).get_meta(result.job_id)["status"] == "exporting"
        output = Path(out_dir) / "highlights.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"rendered video")
        return ExportResult(joined_path=output)

    monkeypatch.setattr("apps.api.main.Analyzer.analyze", analyze)
    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    with TestClient(create_app(config)) as client:
        started = client.post("/api/jobs", json={"source_path": str(source), "auto_export": auto_export})
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        progress = client.get(f"/api/jobs/{job_id}/progress").json()
        assert progress["status"] == ("completed" if auto_export else "ready_for_review")
        if auto_export:
            assert len(exports) == 1
            assert exports[0].export_clips is False
            assert exports[0].export_joined is True
            assert client.get(f"/api/jobs/{job_id}/download/highlights").content == b"rendered video"
        else:
            assert exports == []


def test_no_shots_does_not_offer_an_old_export_as_new_output(config, tmp_path, monkeypatch):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = JobStore(config)
    job_id = store.create(source)
    export_dir = store.job_dir(job_id) / "export"
    export_dir.mkdir()
    (export_dir / "highlights.mp4").write_bytes(b"previous output")

    def analyze(*args, **kwargs):
        return AnalysisResult(job_id=job_id, source_path=str(source), metadata=VideoMetadata(path=str(source)))

    monkeypatch.setattr("apps.api.main.Analyzer.analyze", analyze)
    monkeypatch.setattr("apps.api.main.Exporter.export", lambda *args: pytest.fail("empty analysis cannot export"))
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json={"source_path": str(source), "job_id": job_id})
        assert response.status_code == 200
        progress = client.get(f"/api/jobs/{job_id}/progress").json()
        assert progress["status"] == "completed"
        assert progress["download_url"] is None
        assert client.get(f"/api/jobs/{job_id}/download/highlights").status_code == 404
