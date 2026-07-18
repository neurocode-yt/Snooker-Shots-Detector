"""End-to-end-ish test on synthetic video (requires opencv writer; ffmpeg optional for proxy)."""

from __future__ import annotations

import shutil

import pytest

from snooker_ai.config import load_config
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import EditMode


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe required")
def test_analyze_synthetic(synthetic_video, tmp_path):
    cfg = load_config()
    # Speed up
    cfg._data["analysis"]["sample_fps"] = 5.0
    cfg._data["proxy"]["target_fps"] = 10.0
    cfg._data["proxy"]["max_width"] = 480
    cfg._data["proxy"]["max_height"] = 270

    analyzer = Analyzer(cfg, tmp_path / "job")
    result = analyzer.analyze(synthetic_video, "test-synthetic", mode=EditMode.ACTION_ONLY, resume=False)
    assert result.metadata.duration > 0
    assert isinstance(result.shots, list)
    # Synthetic has motion bursts; may detect 1+ candidates depending on thresholds
    # Honest assertion: pipeline completes and writes analysis
    assert (tmp_path / "job" / "analysis.json").exists()
    assert (tmp_path / "job" / "timeline.json").exists()
