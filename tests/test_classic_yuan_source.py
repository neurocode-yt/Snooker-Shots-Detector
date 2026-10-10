"""Frozen native observations from the first full-pipeline Yuan evaluation."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate


CASES = json.loads(
    (Path(__file__).parent / "fixtures/classic_hard/yuan-native-windows.json").read_text()
)["cases"]


def rows(name):
    return [FrameFeatures.model_validate(f) for f in CASES[name]["features"]]


def test_first_hidden_contact_survives_later_larger_residual_peak(config):
    detector = StrikeDetector(config)
    features = rows("occluded_contact")
    candidates = detector.detect_candidates(detector.score_frames(features))
    assert len(candidates) == 1
    assert 1.68 <= candidates[0].timestamp <= 1.72
    assert candidates[0].evidence["impact_occlusion_contact"] == 1
    proposal = StrikeCandidate(
        timestamp=2.27, confidence=0.5, uncertainty_start=1.2, uncertainty_end=3.0
    )
    refined = detector.refine_boundaries([proposal], features)[0]
    assert 1.68 <= refined.timestamp <= 1.72


def test_lifting_white_cannot_borrow_other_table_motion(config):
    detector = StrikeDetector(config)
    features = rows("referee_reset")
    index = min(range(len(features)), key=lambda i: abs(features[i].t - 111.28))
    metrics = detector._ball_onset_metrics(features, index)
    assert metrics["cue_observation_disrupted"] == 1
    assert metrics["ball_onset_run"] >= 3
    assert not detector._ball_onset_confirmed(metrics)
    assert detector.detect_candidates(detector.score_frames(features)) == []


@pytest.mark.parametrize("proof", ["cue_address", "independent_object_launch"])
def test_hidden_native_contact_retains_independent_contact_evidence(config, proof):
    detector = StrikeDetector(config)
    features = rows("referee_reset")
    index = min(range(len(features)), key=lambda i: abs(features[i].t - 111.28))
    # Exercise the evidence gate separately: these counterfactual changes do
    # not relabel the referee sequence as a real shot.
    metrics = detector._ball_onset_metrics(features, index)
    if proof == "cue_address":
        metrics["onset_cue_address_score"] = 0.8
    else:
        metrics["onset_object_launch_observations"] = 3
    assert detector._ball_onset_confirmed(metrics)
