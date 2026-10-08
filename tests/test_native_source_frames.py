"""Contact extraction must retain original frames and consistent geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.utils.timebase import TimeMapper


def video(path: Path, fps: int, size: tuple[int, int], flash: bool):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(fps):
        frame = np.full((size[1], size[0], 3), (30,140,30), np.uint8)
        if flash and i == fps//2:
            frame[40:56,40:56] = (230,230,230)
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize("source_fps", [25, 50])
def test_dense_contact_reads_every_original_frame_at_source_cadence(config, tmp_path, monkeypatch, source_fps):
    original, proxy = tmp_path/"original.mp4", tmp_path/"proxy.mp4"
    video(original, source_fps, (160,128), True)
    video(proxy, 30, (80,64), False)
    config._data["device"] = "cpu"
    analyzer = Analyzer(config, tmp_path/"job")
    analyzer._native_source = original
    seen = []

    def detect(frame, *args, **kwargs):
        seen.append((frame.shape, float(frame[20:28,20:28].mean())))
        return []

    monkeypatch.setattr(analyzer.objects, "detect", detect)
    mapper = TimeMapper(source_duration=1., proxy_duration=1., source_fps=source_fps)
    features, _, _ = analyzer._extract_features(proxy, None, mapper, 1., sample_fps=min(source_fps,30))
    assert len(features) == source_fps
    assert all(f.observation_fps == source_fps for f in features)
    assert all(shape == (64,80,3) for shape, _ in seen)
    assert sum(value > 180 for _, value in seen) == 1

    seen.clear()
    analyzer._extract_features(proxy, None, mapper, 1., sample_fps=2)
    assert sum(value > 180 for _, value in seen) == 0

    # Seeking changes returned coverage, not which original flash is decoded.
    for start in (.20, .24):
        seen.clear()
        features, _, _ = analyzer._extract_features(
            proxy, None, mapper, 1., sample_fps=min(source_fps,30), start_time=start,
        )
        assert features[0].t >= start-1e-6
        assert sum(value > 180 for _, value in seen) == 1
