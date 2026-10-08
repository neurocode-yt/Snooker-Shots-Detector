"""Native source regressions for the nine omissions in the Zhao–Trump edit."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import EditMode, FrameFeatures, StrikeCandidate


CASES = json.loads((Path(__file__).parent / "fixtures/recall_zhao_windows.json").read_text())["cases"]


def source_features(name):
    return [FrameFeatures.model_validate(f) for f in CASES[name]["features"]]


@pytest.mark.parametrize("name", CASES)
def test_missing_source_contact_survives_native_confirmation_and_segmentation(config, name):
    features = source_features(name)
    lower, upper = CASES[name]["source_contact_bounds"]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    contacts = detector.detect_candidates(features)
    assert len(contacts) == 1
    assert lower-.08 <= contacts[0].timestamp <= upper+.50
    if contacts[0].evidence.get("occlusion_inferred", 0) >= .5:
        assert contacts[0].uncertainty_start <= upper
        assert contacts[0].uncertainty_end >= lower
    proposal = StrikeCandidate(timestamp=(lower+upper)/2, confidence=.5,
                               uncertainty_start=lower-.3, uncertainty_end=upper+.5,
                               evidence={"sparse_proposal": 1.})
    refined = detector.refine_boundaries([proposal], features)
    assert len(refined) == 1
    assert lower-.08 <= refined[0].timestamp <= upper+.50
    assert (refined[0].evidence.get("dense_transition_confirmed", 0) >= .5
            or refined[0].evidence.get("native_occlusion_confirmed", 0) >= .5)
    shots = SegmentBuilder(config).build(refined, features, 797.386, EditMode.STRICT)
    assert len(shots) == 1 and shots[0].included
    assert shots[0].clip_start <= lower and shots[0].clip_end >= upper


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("invalid", ["replay", "handling", "foreign"])
def test_recovered_source_contacts_still_require_live_untouched_table(config, name, invalid):
    features = source_features(name)
    for f in features:
        if invalid == "replay":
            f.broadcast_replay = True
        elif invalid == "handling":
            f.table_handling = True
        else:
            f.match_context_valid = False
    detector = StrikeDetector(config)
    detector.score_frames(features)
    assert not detector.detect_candidates(features)


@pytest.mark.parametrize("invalid", ["no_quality", "stationary", "identity_jump", "missing_images"])
def test_hidden_ball_roll_needs_measured_ball_shape_and_continuous_travel(config, invalid):
    features = source_features("hidden_safety")
    for f in features:
        if invalid == "no_quality":
            f.cue_ball_quality = False
        elif invalid == "stationary" and f.cue_ball_detected:
            f.cue_ball_x, f.cue_ball_y = 300., 300.
        elif invalid == "identity_jump" and f.t > 59.80 and f.cue_ball_detected:
            f.cue_ball_x, f.cue_ball_y = 800., 300.
    if invalid == "missing_images":
        features = [f for f in features if not 59.86 < f.t < 60.06]
    assert not StrikeDetector(config)._reacquired_roll_candidates(features)


@pytest.mark.parametrize("invalid", ["no_address", "moving_anchor", "stationary_destination",
                                     "extra_cut", "missing_images"])
def test_contact_at_camera_cut_requires_both_views_to_support_one_launch(config, invalid):
    features = source_features("pink_at_cut")
    for f in features:
        if invalid == "no_address" and f.t < 512.85:
            f.cue_tip_visible = False
        elif invalid == "moving_anchor" and f.t < 512.85:
            f.cue_ball_x += (f.t-512.)*300
            f.cue_ball_stable_normalized_speed = 5.
        elif invalid == "stationary_destination" and f.t > 512.85:
            f.cue_ball_x, f.cue_ball_y = 300., 100.
            f.cue_ball_stable_normalized_speed = 0.
        elif invalid == "extra_cut" and 512.99 < f.t < 513.04:
            f.scene_cut_score = .9
            f.camera_scene_id += 1
    if invalid == "missing_images":
        features = [f for f in features if not 512.96 < f.t < 513.08]
    assert not StrikeDetector(config)._cut_launch_candidates(features)


def test_contact_at_cut_never_compares_camera_coordinates(config):
    features = source_features("pink_at_cut")
    detector = StrikeDetector(config)
    original = detector._cut_launch_candidates(features)
    assert len(original) == 1
    for f in features:
        if f.t < 512.85:
            f.cue_ball_x += 2000.
            f.cue_ball_y -= 1500.
    assert detector._cut_launch_candidates(features) == original


@pytest.mark.parametrize("invalid", ["no_colour_departure", "stationary_white", "no_cue", "missing_images"])
def test_touching_ball_contact_needs_independent_colour_motion_and_white_departure(config, invalid):
    features = source_features("touching_red")
    for f in features:
        if invalid == "no_colour_departure":
            f.object_ball_launch_count = 0
        elif invalid == "stationary_white" and f.cue_ball_detected:
            f.cue_ball_x, f.cue_ball_y = 500., 260.
        elif invalid == "no_cue":
            f.cue_tip_visible = False
            f.cue_contact_score = 0.
    if invalid == "missing_images":
        features = [f for f in features if not 557.72 < f.t < 557.96]
    assert not StrikeDetector(config)._micro_contact_candidates(features)


@pytest.mark.parametrize("name", ["aiming", "walking", "handling", "rest_and_blue"])
def test_fresh_other_match_observations_reject_referee_and_rest_identity_events(config, name):
    negative_cases = json.loads((Path(__file__).parent / "fixtures/recall_recovery_negatives.json").read_text())["cases"]
    case = negative_cases[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    contacts = detector.detect_candidates(features)
    bounds = case["source_contact_bounds"]
    if bounds is None:
        assert contacts == []
    else:
        assert len(contacts) == 1
        assert bounds[0] <= contacts[0].timestamp <= bounds[1]+.12
