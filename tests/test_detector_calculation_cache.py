"""Identical pixel calculations can be reused without altering ball evidence."""

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.object_detection.detector import ObjectDetector


FIXTURES = Path(__file__).parent / "fixtures/classic_hard/saudi"


def source_pixels():
    frame = cv2.imread(str(FIXTURES / "close-red.jpg"))
    assert frame is not None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv, np.full(hsv.shape[:2], 255, np.uint8)


def test_revisited_pixels_reuse_work_but_preserve_merge_inputs(config, monkeypatch):
    detector = ObjectDetector(config)
    hsv, mask = source_pixels()
    calls = []
    compute = detector._compute_warm_cue_spheres

    def counted(*args, **kwargs):
        if not kwargs.get("green_cast_only") and not kwargs.get("strong_green_cast"):
            calls.append(1)
        return compute(*args, **kwargs)

    monkeypatch.setattr(detector, "_compute_warm_cue_spheres", counted)
    first = detector._warm_cue_spheres(hsv, mask)
    expected = [asdict(p) for p in first]
    assert first
    detector._merge_proposals(first, 60)
    # Window resets change tracking/scale, not this pixel-only calculation.
    detector.reset()
    detector._diameter_ema = 110
    second = detector._warm_cue_spheres(hsv, mask)
    assert [asdict(p) for p in second] == expected
    assert len(calls) == 1
    assert all(a is not b for a, b in zip(first, second))


@pytest.mark.parametrize("changed", ["pixels", "mask", "cloth_bounds"])
def test_changed_measurement_cannot_reuse_another_image(config, changed):
    detector = ObjectDetector(config)
    hsv, mask = source_pixels()
    detector._warm_cue_spheres(hsv, mask)
    if changed == "pixels":
        hsv[0, 0, 2] ^= 1
    elif changed == "mask":
        mask[0, 0] = 0
    else:
        detector.cloth_upper[0] -= 1
    detector._warm_cue_spheres(hsv, mask)
    assert len(detector._ivory_cache) == 2


def test_pixel_cache_evicts_old_images_instead_of_growing_with_match(config):
    detector = ObjectDetector(config)
    detector._ivory_cache_limit = 2
    hsv = np.zeros((20, 20, 3), np.uint8)
    mask = np.full((20, 20), 255, np.uint8)
    for value in range(3):
        hsv[0, 0, 2] = value
        detector._warm_cue_spheres(hsv, mask)
    assert len(detector._ivory_cache) == 2


def test_repeated_contours_are_evaluated_once_without_dropping_proposals(config, monkeypatch):
    detector = ObjectDetector(config)
    hsv, mask = source_pixels()
    evaluated = []
    evaluate = detector._ivory_contour_proposals

    def counted(_hsv, _cloth, contour, *, strong_green_cast, _evidence_cache=None):
        evaluated.append((strong_green_cast, contour.shape, contour.tobytes()))
        return evaluate(_hsv, _cloth, contour, strong_green_cast=strong_green_cast,
                        _evidence_cache=_evidence_cache)

    monkeypatch.setattr(detector, "_ivory_contour_proposals", counted)
    cached = detector._compute_warm_cue_spheres(hsv, mask)
    assert len(evaluated) == len(set(evaluated))

    class Uncached(dict):
        def __contains__(self, key):
            return False

    evaluated.clear()
    uncached = detector._compute_warm_cue_spheres(hsv, mask, _contour_cache=Uncached())
    assert len(evaluated) > len(set(evaluated))
    assert [asdict(p) for p in cached] == [asdict(p) for p in uncached]
