"""Native coverage reacquisition keeps a safety contact in the source cut."""

import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.scene_detection.contact_visibility import (
    ContactVisibilityDetector,
    raw_visibility,
)
from snooker_ai.types import FrameFeatures, StrikeCandidate


FIXTURES = Path(__file__).parent / "fixtures/classic_hard"
CASE = json.loads((FIXTURES / "missed-safety-native.json").read_text())


class FixtureCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_MSEC
        requested = value / 1000
        self.index = min(range(len(self.frames)), key=lambda i: abs(self.frames[i][0] - requested))
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        _, frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.frames[self.index - 1][0] * 1000
        raise AssertionError(prop)

    def release(self):
        pass


def test_occluded_safety_white_reacquisition_uses_native_full_table_frames(config, monkeypatch):
    names = ("194.24", "194.28", "194.32", "194.36", "194.40", "194.44")
    frames = [
        (float(name), cv2.imread(str(FIXTURES / "missed_safety" / f"{name}.jpg"))) for name in names
    ]
    assert all(frame is not None for _, frame in frames)
    capture = FixtureCapture(frames)
    monkeypatch.setattr(
        "snooker_ai.scene_detection.contact_visibility.open_capture",
        lambda _source: capture,
    )

    features = [FrameFeatures.model_validate(row) for row in CASE["features"]]
    assert not any(f.table_full_view for f in features if 194.2 <= f.t <= 194.48)
    # The coarse proposal at 194.535 is weak and later rejected; it must not
    # prevent native coverage from recovering the physical contact at 194.36.
    assert abs(CASE["coarse_candidate_timestamp"] - 194.36) < 2
    coarse = StrikeCandidate(
        timestamp=CASE["coarse_candidate_timestamp"],
        confidence=0.55,
        uncertainty_start=194.0,
        uncertainty_end=195.0,
    )
    updated = ContactVisibilityDetector(config).revalidate("source.mp4", features, [coarse])
    confirmed = [f for f in updated if f.native_contact_coverage_revalidated]
    assert [f.t for f in confirmed] == [pytest.approx(194.36)]
    assert confirmed[0].table_full_view

    detector = StrikeDetector(config)
    candidates = detector.detect_candidates(
        detector.score_frames(detector._stabilized_features(updated))
    )
    assert len(candidates) == 1
    lower, upper = CASE["source_contact_bounds"]
    assert lower - 0.25 <= candidates[0].timestamp <= upper + 0.25
    assert candidates[0].evidence["occlusion_inferred"] == 1

    # Cached policy-derived geometry must be remeasured after a policy change.
    cached = raw_visibility(updated)
    assert not any(
        f.native_contact_coverage_revalidated or f.table_full_view
        for f in cached
        if 194.2 <= f.t <= 194.48
    )


def test_distant_proposal_cannot_trigger_visibility_revalidation(config, monkeypatch):
    monkeypatch.setattr(
        "snooker_ai.scene_detection.contact_visibility.open_capture",
        lambda _source: pytest.fail("frames are unnecessary beside an accepted proposal"),
    )
    features = [FrameFeatures.model_validate(row) for row in CASE["features"]]
    distant = StrikeCandidate(
        timestamp=190.0,
        confidence=0.9,
        uncertainty_start=189.9,
        uncertainty_end=190.1,
    )
    unchanged = ContactVisibilityDetector(config).revalidate("source.mp4", features, [distant])
    assert not any(f.native_contact_coverage_revalidated for f in unchanged)
