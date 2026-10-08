"""A rest identity swap must not hide the real stroke a few seconds later."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate


CASE = json.loads((Path(__file__).parent / "fixtures/recall_rest_identity.json").read_text())


def source_features():
    return [FrameFeatures.model_validate(f) for f in CASE["features"]]


def test_stationary_white_contradicts_rest_identity_motion_and_retains_real_contact(config):
    features = source_features()
    detector = StrikeDetector(config)
    times = [f.t for f in features]
    index = min(range(len(features)), key=lambda i: abs(features[i].t-CASE["previous_false_timestamp"]))
    metrics = detector._transition_metrics(features, index, times)
    assert metrics["stable_cue_peak_speed"] > 20
    assert metrics["cue_displacement_diameters"] > .6
    assert metrics["quiet_anchor_contradiction"] == 1
    assert not detector._transition_confirmed(metrics)
    assert not detector._sparse_dense_transition_confirmed(metrics)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    assert CASE["source_contact_bounds"][0]-.05 <= candidates[0].timestamp <= CASE["source_contact_bounds"][1]
    evidence = candidates[0].evidence
    if evidence.get("occlusion_inferred", 0) >= .5:
        assert evidence["ball_onset_run"] >= 3
    else:
        # A measured initial leg can now prove this contact before the
        # collision bends the white's longer trajectory.
        assert evidence["cue_ball_motion_confirmed"] == 1
        assert evidence["initial_launch_displacement"] >= 1.25
        assert evidence["initial_launch_direction"] >= .95


def test_refinement_cannot_confirm_the_earlier_rest_motion(config):
    detector = StrikeDetector(config)
    proposal = StrikeCandidate(timestamp=2817., confidence=.5,
                               uncertainty_start=2816.8, uncertainty_end=2817.4)
    candidate = detector.refine_boundaries([proposal], source_features())[0]
    assert candidate.evidence["dense_transition_confirmed"] == 0
    assert candidate.evidence["sparse_dense_transition"] == 0
    assert candidate.evidence["occlusion_inferred"] == 0


@pytest.mark.parametrize("counterexample", [
    "reliable_departure", "independent_collision", "continued_travel", "missing_images",
    "camera_cut", "invalid_observation", "unreliable_anchor", "sparse",
])
def test_quiet_return_rejection_requires_actual_reliable_stationarity(config, counterexample):
    features = source_features()
    detector = StrikeDetector(config)
    timestamp = CASE["previous_false_timestamp"]
    index = min(range(len(features)), key=lambda i: abs(features[i].t-timestamp))
    times = [f.t for f in features]
    metrics = detector._transition_metrics(features, index, times)
    assert detector._quiet_anchor_contradiction(features, index, times, metrics)
    for frame in features:
        after = timestamp <= frame.t <= timestamp+.6
        if counterexample == "reliable_departure" and 2817.15 < frame.t < 2817.18:
            # A genuinely measured out-and-back trajectory cannot be rejected
            # merely because the white ultimately reaches its original place.
            frame.cue_ball_track_confidence = .9
        elif counterexample == "independent_collision" and timestamp < frame.t < timestamp+.3:
            frame.moving_ball_count = 1
            frame.max_ball_normalized_speed = (frame.cue_ball_stable_normalized_speed or 0.)+8.
        elif counterexample == "continued_travel" and after and frame.cue_ball_x is not None:
            frame.cue_ball_x += (frame.t-timestamp)*100.
        elif counterexample == "camera_cut" and 2817.3 < frame.t < 2817.4:
            frame.scene_cut_score = .8
        elif counterexample == "invalid_observation" and 2817.3 < frame.t < 2817.4:
            frame.observation_valid = False
        elif counterexample == "unreliable_anchor" and frame.t < timestamp:
            frame.cue_ball_track_confidence = .2
        elif counterexample == "sparse":
            frame.observation_fps = 2.
    if counterexample == "missing_images":
        features = [f for f in features if not 2817.25 < f.t < 2817.45]
    times = [f.t for f in features]
    index = min(range(len(features)), key=lambda i: abs(features[i].t-timestamp))
    metrics = detector._transition_metrics(features, index, times)
    assert not detector._quiet_anchor_contradiction(features, index, times, metrics)
