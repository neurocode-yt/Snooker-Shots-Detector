"""Review-player media delivery regressions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.types import AnalysisResult, EditMode, VideoMetadata


def test_review_video_prefers_proxy_and_supports_ranges(config, tmp_path):
    jobs = tmp_path / "jobs"
    uploads = tmp_path / "uploads"
    config._data["paths"]["jobs_dir"] = str(jobs)
    config._data["paths"]["uploads_dir"] = str(uploads)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-original")
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy-fast")

    store = JobStore(config)
    job_id = store.create(source, mode="strict")
    store.save_analysis(
        AnalysisResult(
            job_id=job_id,
            source_path=str(source),
            proxy_path=str(proxy),
            metadata=VideoMetadata(
                path=str(source), duration=10.0, width=1280, height=720, fps=30.0
            ),
            mode=EditMode.STRICT,
            original_duration=10.0,
        )
    )

    with TestClient(create_app(config)) as client:
        response = client.get(
            f"/api/jobs/{job_id}/video", headers={"Range": "bytes=0-4"}
        )

    assert response.status_code == 206
    assert response.content == b"proxy"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 0-4/10"
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert "content-disposition" not in response.headers


def test_review_video_falls_back_to_original_when_proxy_is_missing(config, tmp_path):
    jobs = tmp_path / "jobs"
    uploads = tmp_path / "uploads"
    config._data["paths"]["jobs_dir"] = str(jobs)
    config._data["paths"]["uploads_dir"] = str(uploads)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-original")
    store = JobStore(config)
    job_id = store.create(source, mode="strict")

    with TestClient(create_app(config)) as client:
        response = client.get(
            f"/api/jobs/{job_id}/video", headers={"Range": "bytes=0-5"}
        )

    assert response.status_code == 206
    assert response.content == b"source"
