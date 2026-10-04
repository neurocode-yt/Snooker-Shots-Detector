"""Missed broadcast launches, compared with independently inspected contacts."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate


CASES = json.loads((Path(__file__).parent / "fixtures/recall_temporal_windows.json")
                  .read_text())["cases"]


def observations(name):
    return [FrameFeatures.model_validate(f) for f in CASES[name]["features"]]


@pytest.mark.parametrize("name", CASES)
def test_native_recall_and_aiming_negative(config, name):
    detector = StrikeDetector(config)
    features = observations(name)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    bounds = CASES[name]["contact_bounds"]
    if bounds is None:
        assert not candidates
    else:
        assert len(candidates) == 1
        assert bounds[0] <= candidates[0].timestamp <= bounds[1]
        if name == "obscured_red":
            assert candidates[0].evidence["contact_time_upper_bound"] == 1
            assert candidates[0].uncertainty_start < 315.0
            assert candidates[0].uncertainty_end == candidates[0].timestamp


def test_native_refinement_retains_bounded_hidden_contact(config):
    detector = StrikeDetector(config)
    features = observations("obscured_red")
    detector.score_frames(features)
    proposal = StrikeCandidate(timestamp=316., confidence=.6,
                               uncertainty_start=314.5, uncertainty_end=317.)
    refined = detector.refine_boundaries([proposal], features)[0]
    assert 315.32 <= refined.timestamp <= 315.40
    assert refined.evidence["native_occlusion_confirmed"] == 1
    assert refined.evidence["contact_time_upper_bound"] == 1
    assert refined.evidence.get("dense_transition_confirmed", 0) == 0


@pytest.mark.parametrize("invalid", ["foreign", "handling", "no_address", "camera_motion",
                                     "distant_identity", "sparse"])
def test_hidden_contact_requires_the_complete_visual_sequence(config, invalid):
    features = observations("obscured_red")
    for f in features:
        if invalid == "foreign":
            f.match_context_valid = False
        elif invalid == "handling":
            f.table_handling = True
        elif invalid == "no_address":
            f.cue_contact_score = 0
            f.cue_tip_visible = False
        elif invalid == "camera_motion":
            f.cue_ball_stable_normalized_speed = 0
        elif invalid == "distant_identity" and f.t >= 315.35 and f.cue_ball_x is not None:
            f.cue_ball_x += 200
        elif invalid == "sparse":
            f.observation_fps = 2
    assert not StrikeDetector(config)._occluded_launch_candidates(features)


@pytest.mark.parametrize("invalid", ["no_cue", "no_stable_motion", "no_quiet_anchor"])
def test_slow_contact_gate_still_requires_independent_launch_evidence(config, invalid):
    features = observations("slow_contact_in_partial_view")
    detector = StrikeDetector(config)
    times = [f.t for f in features]
    index = min(range(len(features)), key=lambda i: abs(features[i].t-264.0322))
    metrics = detector._transition_metrics(features, index, times)
    assert detector._addressed_slow_launch_confirmed(metrics)
    if invalid == "no_cue":
        metrics["pre_cue_address_score"] = metrics["cue_contact_score"] = 0
    elif invalid == "no_stable_motion":
        metrics["stable_cue_motion_count"] = 0
    else:
        metrics["stationary_ratio"] = .5
    assert not detector._addressed_slow_launch_confirmed(metrics)
