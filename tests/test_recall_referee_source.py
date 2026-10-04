"""Full-pipeline referee negatives with intermittently observed stationary white."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures


CASES = json.loads((Path(__file__).parent / "fixtures/recall_referee_windows.json")
                  .read_text())["cases"]


@pytest.mark.parametrize("name", ["walking_before_respot", "blue_respot"])
def test_referee_movement_does_not_turn_missing_white_samples_into_a_strike(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-case["previous_false_timestamp"]))
    onset = detector._ball_onset_metrics(features, index)
    assert onset["cue_missing_after_onset"] == 1
    assert onset["cue_stationary_after_onset"] == 1
    assert not detector._ball_onset_confirmed(onset)
    detector.score_frames(features)
    assert not detector.detect_candidates(features)


@pytest.mark.parametrize("name", ["cut_before_player_walk", "pan_while_chalking"])
def test_camera_discontinuity_cannot_supply_the_missing_launch(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-case["previous_false_timestamp"]))
    if name == "cut_before_player_walk":
        onset = detector._ball_onset_metrics(features, index)
        assert onset["onset_observation_contiguous"] == 0
        assert not detector._ball_onset_confirmed(onset)
    else:
        launch = detector._transition_metrics(features, index)
        assert launch["post_observation_contiguous"] == 0
        assert launch["stable_cue_motion_count"] == 0
        assert not detector._transition_confirmed(launch)
    detector.score_frames(features)
    assert not detector.detect_candidates(features)


def test_brief_invalid_frame_after_an_established_roll_does_not_erase_it(config):
    case = CASES["rolling_before_brief_invalid_frame"]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    assert case["expected_candidate_bounds"][0] <= candidates[0].timestamp <= case["expected_candidate_bounds"][1]
    index = min(range(len(features)), key=lambda i: abs(features[i].t-458.665))
    assert detector._transition_confirmed(detector._transition_metrics(features, index))
    for feature in features:
        if not feature.observation_valid:
            feature.scene_cut_score = .8
    assert not detector._transition_confirmed(detector._transition_metrics(features, index))
