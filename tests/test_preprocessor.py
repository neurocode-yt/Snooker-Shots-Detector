"""Pre-analysis match-break editor rendering tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.config import load_config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.rendering.preprocessor import (
    KeepRange,
    VideoPreprocessor,
    normalize_keep_ranges,
)
from snooker_ai.types import VideoMetadata
from snooker_ai.utils.ffmpeg import run_command


def test_keep_ranges_are_sorted_merged_and_validated():
    normalized = normalize_keep_ranges(
        [KeepRange(4.0, 7.0), KeepRange(0.0, 2.0), KeepRange(2.0, 4.0)],
        10.0,
    )
    assert normalized == [KeepRange(0.0, 7.0)]

    with pytest.raises(ValueError, match="overlap"):
        normalize_keep_ranges([KeepRange(0.0, 3.0), KeepRange(2.0, 4.0)], 10.0)
    with pytest.raises(ValueError, match="At least one"):
        normalize_keep_ranges([], 10.0)
    with pytest.raises(ValueError, match="outside"):
        normalize_keep_ranges([KeepRange(0.0, 11.0)], 10.0)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)
def test_preprocessor_joins_kept_ranges_with_audio(tmp_path: Path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "cleaned.mp4"
    run_command(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=6",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        timeout=60,
    )
    config = load_config(
        overrides={
            "export": {"video_codec": "libx264"},
            "preprocess": {"preset": "ultrafast", "crf": 28},
        }
    )

    rendered = VideoPreprocessor(config).render(
        source,
        [KeepRange(0.0, 1.5), KeepRange(3.0, 5.0)],
        output,
    )

    metadata = probe_video(rendered)
    assert metadata.duration == pytest.approx(3.5, abs=0.15)
    assert metadata.has_audio is True
    assert source.exists() is True


def test_preprocess_api_only_accepts_uploaded_sources(config, tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    jobs = tmp_path / "jobs"
    uploads.mkdir()
    config._data["paths"]["uploads_dir"] = str(uploads)
    config._data["paths"]["jobs_dir"] = str(jobs)
    source = uploads / "match.mp4"
    source.write_bytes(b"source")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")

    def fake_render(_self, _source, _ranges, output):
        Path(output).write_bytes(b"cleaned")
        return Path(output)

    monkeypatch.setattr("apps.api.main.VideoPreprocessor.render", fake_render)
    monkeypatch.setattr(
        "apps.api.main.probe_video",
        lambda path: VideoMetadata(
            path=str(path), duration=8.0, width=1280, height=720, fps=30.0
        ),
    )

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/api/preprocess",
            json={
                "source_path": str(source),
                "ranges": [{"start": 0.0, "end": 3.0}, {"start": 5.0, "end": 8.0}],
            },
        )
        rejected = client.post(
            "/api/preprocess",
            json={"source_path": str(outside), "ranges": [{"start": 0.0, "end": 1.0}]},
        )

    assert response.status_code == 200
    assert Path(response.json()["path"]).read_bytes() == b"cleaned"
    assert response.json()["original_path"] == str(source.resolve())
    assert rejected.status_code == 400
