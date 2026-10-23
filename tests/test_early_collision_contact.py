"""A real cue launch can hit a nearby red and reverse inside the motion window."""
import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate

FIXTURE = Path(__file__).parent / "fixtures/classic_hard/early-red-collision-native.json"


def recorded():
    data = json.loads(FIXTURE.read_text())
    return data, [FrameFeatures.model_validate(item) for item in data["features"]]


def test_near_red_collision_preserves_measured_initial_launch(config):
    data, features = recorded()
    detector = StrikeDetector(config)
    scored = detector.score_frames(features)
    candidate = StrikeCandidate(timestamp=286, confidence=.7, uncertainty_start=284, uncertainty_end=288)
    detector.refine_boundaries([candidate], scored)
    assert candidate.evidence["dense_transition_confirmed"] == 1
    assert candidate.timestamp == pytest.approx(data["expected_contact"], abs=1e-9)
    found = detector.detect_candidates(scored)
    assert any(abs(item.timestamp - data["expected_contact"]) < 1e-9 for item in found)


@pytest.mark.parametrize("missing_evidence", ["object_motion", "cue_contact", "quiet_white"])
def test_collision_path_cannot_replace_independent_strike_evidence(config, missing_evidence):
    _, features = recorded()
    for item in features:
        if missing_evidence == "object_motion":
            item.moving_ball_count = 0
            item.max_ball_normalized_speed = 0
        elif missing_evidence == "cue_contact":
            item.cue_tip_visible = False
            item.cue_contact_score = 0
            item.cue_approach_speed = 0
            item.cue_forward_motion = 0
        elif item.t < 285.6:
            item.cue_ball_speed = 30
            item.cue_ball_normalized_speed = 3
            item.cue_ball_stable_normalized_speed = 3
    detector = StrikeDetector(config)
    assert detector.detect_candidates(detector.score_frames(features)) == []
