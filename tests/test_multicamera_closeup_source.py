"""Measured low-camera contact observations, with no source media in Git."""

import json
from pathlib import Path

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate


CASE = json.loads((Path(__file__).parent / "fixtures/multicamera_closeup_brown_contact.json")
                  .read_text(encoding="utf-8"))


def test_shaded_white_launch_in_low_brown_camera_is_automatically_detected(config):
    features = [FrameFeatures.model_validate(f) for f in CASE["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    lo, hi = CASE["expected_contact"]
    assert lo <= candidates[0].timestamp <= hi
    assert candidates[0].evidence.get("impact_occlusion_contact") == 1
    assert candidates[0].evidence.get("cue_ball_motion_confirmed") == 1
    assert candidates[0].confidence >= 0.70


def test_native_confirmation_recovers_low_camera_brown_contact_after_brief_blur(config):
    features = [FrameFeatures.model_validate(f) for f in CASE["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidate = StrikeCandidate(
        timestamp=2139.8, confidence=0.7,
        uncertainty_start=2139.3, uncertainty_end=2140.1,
    )
    detector.refine_boundaries([candidate], features)
    lo, hi = CASE["expected_contact"]
    assert lo <= candidate.timestamp <= hi
    assert candidate.evidence.get("dense_transition_confirmed") == 1
    assert candidate.evidence.get("stable_cue_motion_count", 0) >= 3
    assert candidate.evidence.get("stable_cue_peak_speed", 0) >= 4
    # Cue-ball observations before and after the blur are retained in the same
    # low camera, so a camera-coordinate jump cannot explain this launch.
    before = [f for f in features if 2139.5 <= f.t <= 2139.72 and f.cue_ball_detected]
    after = [f for f in features if 2139.82 <= f.t <= 2140.1 and f.cue_ball_detected]
    assert before and after
    assert before[0].camera_scene_id == after[0].camera_scene_id
    assert all(not f.table_full_view for f in before + after)


def test_low_camera_red_contact_is_kept_when_cue_touches_the_shaded_outline(config):
    case = json.loads((Path(__file__).parent / "fixtures/multicamera_closeup_red_contact.json")
                      .read_text(encoding="utf-8"))
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    lo, hi = case["expected_contact"]
    assert lo <= candidates[0].timestamp <= hi
    candidate = StrikeCandidate(
        timestamp=2870.25, confidence=.70,
        uncertainty_start=2870.10, uncertainty_end=2870.65,
    )
    detector.refine_boundaries([candidate], features)
    assert candidate.evidence.get("dense_transition_confirmed") == 1
    assert lo <= candidate.timestamp <= hi
