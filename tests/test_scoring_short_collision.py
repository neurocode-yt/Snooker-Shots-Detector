"""Real short white rolls must survive an immediate collision, without relaxing ordinary detection."""

import json
from pathlib import Path

import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pot_detection.recall import scoring_collision_contact
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate

CASES = json.loads((Path(__file__).parent / "fixtures/scoring_short_collisions.json").read_text())[
    "cases"
]


def sample(case):
    gap = case["gap"]
    hint = StrikeCandidate(
        timestamp=gap["end"] - 3,
        confidence=0.5,
        uncertainty_start=gap["start"],
        uncertainty_end=gap["end"],
        evidence={
            "score_assisted_proposal": 1.0,
            "score_delta": gap["points"],
            "score_update_time": gap["score_update_time"],
        },
    )
    return hint, [FrameFeatures.model_validate(f) for f in case["features"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_real_collision_keeps_both_measured_legs_and_bounded_contact(config, case, tmp_path):
    hint, rows = sample(case)
    contact = scoring_collision_contact(hint, rows)
    assert contact is not None and contact.timestamp == pytest.approx(case["expected_contact"])
    assert contact.evidence["collision_launch_travel_diameters"] >= 0.9
    assert contact.evidence["collision_exit_travel_diameters"] >= 0.4
    assert contact.evidence["collision_turn_cosine"] < 0
    assert contact.uncertainty_start < contact.uncertainty_end == contact.timestamp
    owner = Analyzer(config, tmp_path / case["name"])
    recovered = owner._recover_scoring_depth("unused", [hint], [], rows)
    assert len(recovered) == 1  # No extra pixel decode is needed for measured native white motion.
    owner.strike_det.refine_boundaries(recovered, rows)
    assert SegmentBuilder._candidate_supported(recovered[0])
    assert len(owner.segmenter.build(recovered, rows, 500)) == 1


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize(
    "defect",
    [
        "no_score",
        "large_score",
        "replay",
        "handling",
        "foreign",
        "rack",
        "invalid",
        "sparse",
        "gap",
        "scene",
        "cut",
        "camera",
        "no_shape",
        "no_address",
        "still",
        "straight_roll",
        "identity_jump",
        "no_rest",
        "nan_position",
    ],
)
def test_score_hint_never_replaces_native_motion_and_context(case, defect):
    hint, rows = sample(case)
    if defect == "no_score":
        hint.evidence = {}
    if defect == "large_score":
        hint.evidence["score_delta"] = 12
    onset = case["expected_contact"]
    for f in rows:
        if defect == "replay":
            f.broadcast_replay = True
        elif defect == "handling":
            f.table_handling = True
        elif defect == "foreign":
            f.match_context_valid = False
        elif defect == "rack":
            f.rack_idle = True
        elif defect == "invalid":
            f.observation_valid = False
        elif defect == "sparse":
            f.observation_fps = 2
        elif defect == "scene":
            f.camera_scene_id = round(f.t * 1000)
        elif defect == "cut":
            f.scene_cut_score = 0.8
        elif defect == "camera":
            f.camera_motion_magnitude = 5
        elif defect == "no_shape":
            f.cue_ball_quality = False
        elif defect == "no_address":
            f.cue_tip_visible = False
            f.cue_contact_score = 0
        elif defect == "still" and f.t >= onset:
            f.cue_ball_x = 500
            f.cue_ball_y = 300
        elif defect == "straight_roll" and f.t >= onset:
            f.cue_ball_x = 500 + 200 * (f.t - onset)
            f.cue_ball_y = 300
        elif defect == "identity_jump" and f.t >= onset + 0.08:
            f.cue_ball_x = (f.cue_ball_x or 0) + 100
        elif defect == "no_rest" and f.t < onset:
            f.motion_raw = 1
            f.max_ball_normalized_speed = 20
            f.cue_ball_x = 500 + 100 * (f.t - onset)
        elif defect == "nan_position":
            f.cue_ball_x = float("nan")
    if defect == "gap":
        rows = [f for f in rows if not onset + 0.03 < f.t < onset + 0.1]
    assert scoring_collision_contact(hint, rows) is None


@pytest.mark.parametrize(
    "missing",
    [
        "native_scoring_short_collision",
        "score_assisted_proposal",
        "score_supported_object_launch",
        "collision_measured_frames",
        "collision_launch_travel_diameters",
        "collision_exit_travel_diameters",
        "collision_launch_direction",
        "collision_exit_direction",
        "collision_address_score",
        "score_delta",
        "collision_launch_motion_start",
        "object_launch_confirmation_time",
    ],
)
def test_segmentation_requires_complete_collision_proof(missing):
    hint, rows = sample(CASES[0])
    contact = scoring_collision_contact(hint, rows)
    contact.evidence[missing] = 0.0
    assert not SegmentBuilder._candidate_supported(contact)
