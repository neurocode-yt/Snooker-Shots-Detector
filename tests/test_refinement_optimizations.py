"""Scheduling and memoization must preserve the measured frame evidence."""
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.config import load_config
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.scene_detection.detector import SceneDetector


class NoReuse(dict):
    """Calculate every retry independently, retaining original multiplicity."""
    def __contains__(self, key):
        return False

    def get(self, key, default=None):
        return default


@pytest.mark.parametrize("occluded", [False, True])
def test_identical_colour_masks_share_contours_without_changing_proposals(monkeypatch, occluded):
    hsv = np.full((220, 320, 3), (60, 180, 160), np.uint8)
    cv2.ellipse(hsv, (160, 90), (40, 18), 0, 0, 360, (173, 22, 250), -1)
    cv2.circle(hsv, (160, 115), 15, (30, 80, 220), -1)
    cv2.circle(hsv, (156, 109), 5, (25, 20, 255), -1)
    if occluded:
        cv2.circle(hsv, (164, 128), 16, (2, 180, 210), -1)
    mask = np.full(hsv.shape[:2], 255, np.uint8)
    detector = ObjectDetector(load_config())
    original = ObjectDetector._warm_mask_contours
    calls = []

    def count(*args, **kwargs):
        calls.append(kwargs["strong_green_cast"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ObjectDetector, "_warm_mask_contours", staticmethod(count))
    expected = detector._compute_warm_cue_spheres(hsv, mask, _contour_cache=NoReuse())
    independent_calls = len(calls)
    calls.clear()
    actual = detector._compute_warm_cue_spheres(hsv, mask, _contour_cache={})
    assert [asdict(p) for p in actual] == [asdict(p) for p in expected]
    assert len(calls) < independent_calls


def test_proxy_metadata_reuse_preserves_native_frames_and_reopens_changed_file(
    synthetic_video, tmp_path, monkeypatch,
):
    import snooker_ai.pipeline.analyzer as module

    config = load_config(overrides={"device": "cpu", "analysis": {"hwaccel_decode": False}})
    analyzer = Analyzer(config, tmp_path / "memoized")
    analyzer._native_source = synthetic_video
    mapper = TimeMapper(7, source_fps=15)
    original = module.open_capture
    opens = []

    def capture(path, **kwargs):
        opens.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(module, "open_capture", capture)

    def extract(owner):
        return owner._extract_features(str(synthetic_video), None, mapper, 7,
            sample_fps=15, start_time=1.1, end_time=1.3)[0]

    first = extract(analyzer)
    assert len(opens) == 2  # proxy header, then original images
    second = extract(analyzer)
    assert len(opens) == 3  # original images only
    assert [f.model_dump() for f in first] == [f.model_dump() for f in second]
    assert len(analyzer._proxy_capture_info) == 1

    import os
    stamp = synthetic_video.stat()
    os.utime(synthetic_video, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    extract(analyzer)
    assert len(opens) == 5  # changed file cannot reuse the old header
    assert len(analyzer._proxy_capture_info) == 1


def test_proxy_sampling_still_opens_a_decoder_for_every_scan(synthetic_video, tmp_path):
    config = load_config(overrides={"device": "cpu", "analysis": {"hwaccel_decode": False}})
    analyzer = Analyzer(config, tmp_path)
    mapper = TimeMapper(7, source_fps=15)
    first = analyzer._extract_features(synthetic_video, None, mapper, 7,
        sample_fps=2, start_time=1, end_time=2)[0]
    second = analyzer._extract_features(synthetic_video, None, mapper, 7,
        sample_fps=2, start_time=1, end_time=2)[0]
    assert [f.model_dump() for f in first] == [f.model_dump() for f in second]


def test_scene_stream_reuses_only_same_frame_evidence_and_preserves_fade_lookahead(config, monkeypatch):
    detector = SceneDetector(config)
    baseline, shared = detector.start_stream(), detector.start_stream()
    images = [np.full((120, 200, 3), value, np.uint8) for value in (30, 120, 180, 200, 50)]
    prepared = [(detector.histogram(f), detector.cut_thumbnail(f), detector.classifier.classify(f))
                for f in images]
    expected = [baseline.observe(f, i*.5) for i, f in enumerate(images)]

    def repeated(*args):
        pytest.fail("An already measured thumbnail/classification must not be recomputed")

    monkeypatch.setattr(detector, "cut_thumbnail", repeated)
    monkeypatch.setattr(detector.classifier, "classify", repeated)
    for i, (f, (hist, thumbnail, classification)) in enumerate(zip(images, prepared)):
        shared.observe(f, i*.5, histogram=hist, thumbnail=thumbnail, classification=classification)
    assert [asdict(o) for o in shared.observations] == [asdict(o) for o in expected]
