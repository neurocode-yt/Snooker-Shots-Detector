"""Source-verified misses and aiming/referee negatives from a complete interval."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures


CASES = json.loads((Path(__file__).parent / "fixtures/recall_middle_windows.json")
                  .read_text())["cases"]


@pytest.mark.parametrize("name", CASES)
def test_source_contact_and_negative_windows(config, name):
    features = [FrameFeatures.model_validate(f) for f in CASES[name]["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    bounds = CASES[name]["contact_bounds"]
    if bounds is None:
        assert not candidates
    else:
        assert len(candidates) == 1
        assert bounds[0] <= candidates[0].timestamp <= bounds[1]


def test_interleaved_missing_white_does_not_prove_impact_occlusion(config):
    features = [FrameFeatures.model_validate(f)
                for f in CASES["referee_stationary_white"]["features"]]
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-1698.826))
    metrics = detector._ball_onset_metrics(features, index)
    assert metrics["cue_missing_after_onset"] == 1
    assert metrics["cue_stationary_after_onset"] == 1
    assert not detector._ball_onset_confirmed(metrics)


@pytest.mark.parametrize("name,timestamp", [
    ("aiming_identity_switch", 411.832),
    ("aiming_red_identity_switch", 1717.926),
])
def test_filter_velocity_does_not_confirm_a_single_identity_jump(config, name, timestamp):
    features = [FrameFeatures.model_validate(f) for f in CASES[name]["features"]]
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-timestamp))
    metrics = detector._transition_metrics(features, index)
    assert metrics["stable_cue_motion_count"] >= 3
    assert detector._single_step_identity_jump(metrics)
    assert not detector._transition_confirmed(metrics)


def test_fast_collision_with_continuing_object_ball_motion_is_retained(config):
    # The white travels two diameters between frames and stops at a red. The
    # red continues, so its independent speed explains the short white path.
    features = []
    for index in range(31):
        moving = index >= 20
        speed = ([1.2, 15., 6., 3., 1.5, .8, .4, .2][min(index-20, 7)]
                 if moving else 0.)
        features.append(FrameFeatures(
            t=index*.04, observation_fps=25, table_confidence=.9,
            view_type="main_table", ball_diameter_px=20., ball_count=4,
            cue_ball_detected=True, cue_ball_x=100. if not moving else (100.5 if index == 20 else 140.5),
            cue_ball_y=100., cue_ball_track_confidence=.9,
            cue_ball_normalized_speed=speed, cue_ball_stable_normalized_speed=speed,
            max_ball_normalized_speed=max(speed, 8.) if moving else 0.,
            moving_ball_count=1 if moving else 0, motion_raw=.5 if moving else .02,
            cue_tip_visible=True, cue_tip_distance_to_ball=5., cue_contact_score=.9,
        ))
    detector = StrikeDetector(config)
    detector.score_frames(features)
    metrics = detector._transition_metrics(features, 20)
    assert metrics["largest_cue_step_diameters"] >= 2.
    assert metrics["other_cue_step_travel_diameters"] < .1
    assert metrics["independent_object_motion_count"] >= 3
    assert not detector._single_step_identity_jump(metrics)
    assert detector._transition_confirmed(metrics)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(.8)


def test_single_step_guard_does_not_interpret_sparse_sampling_as_identity_failure(config):
    features = [FrameFeatures.model_validate(f)
                for f in CASES["aiming_identity_switch"]["features"]]
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-411.832))
    metrics = detector._transition_metrics(features, index)
    metrics["observation_fps"] = 2.
    assert not detector._single_step_identity_jump(metrics)


@pytest.mark.parametrize("invalid", ["no_departure", "no_return_motion", "distant_return", "handling", "coarse",
                                     "sparse_tail", "missing_images", "missing_quiet_images", "camera_cut"])
def test_interrupted_departure_requires_both_sides_of_the_occlusion(config, invalid):
    features = [FrameFeatures.model_validate(f)
                for f in CASES["departure_hidden_by_red"]["features"]]
    detector = StrikeDetector(config)
    assert len(detector._interrupted_departure_candidates(features)) == 1
    for feature in features:
        if invalid == "no_departure" and feature.t < 1770.12:
            feature.cue_ball_normalized_speed = 0.
            feature.cue_ball_stable_normalized_speed = 0.
        elif invalid == "no_return_motion" and feature.t > 1770.4:
            feature.cue_ball_stable_normalized_speed = 0.
        elif invalid == "distant_return" and feature.t > 1770.4 and feature.cue_ball_x is not None:
            feature.cue_ball_x += 200.
        elif invalid == "handling":
            feature.table_handling = True
        elif invalid == "coarse":
            feature.observation_fps = 2.
        elif invalid == "sparse_tail" and feature.t > 1770.4:
            feature.observation_fps = 2.
        elif invalid == "camera_cut" and feature.t > 1770.2:
            feature.camera_scene_id += 1
    if invalid == "missing_images":
        features = [f for f in features if not 1770.2 < f.t < 1770.5]
    elif invalid == "missing_quiet_images":
        features = [f for f in features if not 1769.65 < f.t < 1769.9]
    assert not detector._interrupted_departure_candidates(features)
