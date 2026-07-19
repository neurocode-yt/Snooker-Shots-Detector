"""Review-player media delivery regressions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.types import AnalysisResult, EditMode, ShotRecord, VideoMetadata


def test_review_shot_list_uses_compact_timeline(config, tmp_path):
    jobs = tmp_path / "jobs"
    uploads = tmp_path / "uploads"
    config._data["paths"]["jobs_dir"] = str(jobs)
    config._data["paths"]["uploads_dir"] = str(uploads)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-original")
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
                )
            ],
            mode=EditMode.STRICT,
            original_duration=10.0,
            edited_duration=5.0,
        )
    )
    # A corrupt/heavy full analysis must not block the lightweight review list.
    (store.job_dir(job_id) / "analysis.json").write_text("not json", encoding="utf-8")

    with TestClient(create_app(config)) as client:
        response = client.get(f"/api/jobs/{job_id}/shots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_duration"] == 10.0
    assert payload["edited_duration"] == 5.0
    assert [shot["shot_id"] for shot in payload["shots"]] == [1]


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
