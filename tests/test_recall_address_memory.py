"""A source-labelled contact hidden for the duration of a new camera view."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


FIXTURE = json.loads((Path(__file__).parent / "fixtures/recall_address_memory.json").read_text())


def source_features():
    return [FrameFeatures.model_validate(f) for f in FIXTURE["features"]]


def test_hidden_white_retains_the_previous_camera_address_and_explicit_uncertainty(config):
    features = source_features()
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.evidence["camera_address_memory"] == 1.
    assert candidate.evidence["reacquired_roll_displacement"] > 1.8
    assert candidate.evidence["reacquired_roll_direction"] > .98
    first_white = next(f for f in features if f.t > 1630.53 and f.cue_ball_detected)
    assert candidate.timestamp == first_white.t
    assert candidate.uncertainty_start < FIXTURE["source_contact_bounds"][0]
    assert candidate.uncertainty_end > FIXTURE["source_contact_bounds"][1]
    assert candidate.uncertainty_end == candidate.timestamp
    assert candidate.timestamp > 1632.7  # The earlier camera cut is not the strike.


def test_address_memory_refinement_retains_hidden_contact_bounds(config):
    detector = StrikeDetector(config)
    proposal = StrikeCandidate(timestamp=1632.8, confidence=.5,
                               uncertainty_start=1632.1, uncertainty_end=1633.3)
    refined = detector.refine_boundaries([proposal], source_features())[0]
    assert refined.evidence["native_occlusion_confirmed"] == 1.
    assert refined.evidence["camera_address_memory"] == 1.
    assert refined.evidence["dense_transition_confirmed"] == 1.
    assert refined.uncertainty_start < 1630.5
    assert 1632.7 < refined.timestamp < 1632.9
    assert refined.uncertainty_end == refined.timestamp


def test_address_memory_never_compares_positions_across_cameras(config):
    detector = StrikeDetector(config)
    features = source_features()
    original = detector._address_memory_candidates(features)
    for frame in features:
        if frame.t < 1630.5 and frame.cue_ball_x is not None:
            frame.cue_ball_x += 2000.
            frame.cue_ball_y -= 1000.
    assert detector._address_memory_candidates(features) == original


@pytest.mark.parametrize("invalid", [
    "no_address", "moving_anchor", "prior_replay", "cut_replay", "hidden_replay",
    "handling", "foreign", "camera_cut", "changed_scene", "missing_images",
    "missing_source_images", "missing_roll_images", "invalid_image", "coarse",
    "busy_destination", "stationary_return", "no_roll", "identity_jump",
    "no_contact", "visible_before_roll",
])
def test_address_memory_cannot_bridge_unobserved_or_contradictory_evidence(config, invalid):
    features = source_features()
    detector = StrikeDetector(config)
    assert len(detector._address_memory_candidates(features)) == 1
    for frame in features:
        before = frame.t < 1630.5
        hidden = 1630.55 < frame.t < 1632.79
        rolling = frame.t > 1632.79
        if invalid == "no_address" and before:
            frame.cue_tip_visible = False
        elif invalid == "moving_anchor" and before:
            frame.cue_ball_stable_normalized_speed = 3.
            frame.cue_ball_x += (frame.t-1629.8)*100
        elif invalid == "prior_replay" and before:
            frame.broadcast_replay = True
        elif invalid == "cut_replay" and frame.scene_cut_score > .5:
            frame.view_type = CameraViewType.REPLAY
        elif invalid == "hidden_replay" and hidden:
            frame.broadcast_replay = True
        elif invalid == "handling" and hidden:
            frame.table_handling = True
        elif invalid == "foreign" and hidden:
            frame.match_context_valid = False
        elif invalid == "camera_cut" and 1631.2 < frame.t < 1631.3:
            frame.scene_cut_score = .8
        elif invalid == "changed_scene" and frame.t > 1631.2:
            frame.camera_scene_id += 1
        elif invalid == "invalid_image" and 1631.2 < frame.t < 1631.3:
            frame.observation_valid = False
        elif invalid == "coarse":
            frame.observation_fps = 2.
        elif invalid == "busy_destination" and hidden:
            frame.max_ball_normalized_speed = 3.
        elif invalid == "stationary_return" and 1632.79 < frame.t < 1632.9:
            frame.cue_ball_stable_normalized_speed = 0.
        elif invalid == "no_roll" and rolling:
            frame.cue_ball_x = 500.
            frame.cue_ball_y = 360.
            frame.cue_ball_stable_normalized_speed = 0.
        elif invalid == "identity_jump" and frame.t > 1632.8:
            frame.cue_ball_x = 900.
            frame.cue_ball_y = 360.
        elif invalid == "no_contact" and rolling:
            frame.cue_contact_score = 0.
        elif invalid == "visible_before_roll" and 1631.8 < frame.t < 1631.9:
            frame.cue_ball_detected = True
            frame.cue_ball_x = 500.
            frame.cue_ball_y = 360.
            frame.cue_ball_track_confidence = .9
    if invalid == "missing_images":
        features = [f for f in features if not 1631.1 < f.t < 1631.3]
    elif invalid == "missing_source_images":
        features = [f for f in features if not 1630.1 < f.t < 1630.3]
    elif invalid == "missing_roll_images":
        features = [f for f in features if not 1633.0 < f.t < 1633.2]
    assert not detector._address_memory_candidates(features)


@pytest.mark.parametrize("filename", ["recall_middle_windows.json", "recall_referee_windows.json"])
def test_address_memory_adds_nothing_to_other_source_fixture_windows(config, filename):
    detector = StrikeDetector(config)
    cases = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())["cases"]
    for case in cases.values():
        features = [FrameFeatures.model_validate(f) for f in case["features"]]
        assert not detector._address_memory_candidates(features)


def test_rejected_refinement_clears_previous_address_memory_confirmation(config):
    detector = StrikeDetector(config)
    candidate = detector._address_memory_candidates(source_features())[0]
    features = source_features()
    for frame in features:
        frame.broadcast_replay = True
    refined = detector.refine_boundaries([candidate], features)[0]
    assert refined.evidence["camera_address_memory"] == 0.
    assert refined.evidence["dense_transition_confirmed"] == 0.
