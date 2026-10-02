"""Measured end-of-frame shots must not retain a minute of break footage."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import FrameFeatures, StrikeCandidate


OBSERVATIONS = json.loads(
    (Path(__file__).parent / "fixtures/multicamera_terminal_windows.json")
    .read_text(encoding="utf-8")
)
CASES = OBSERVATIONS["cases"]


def _candidate(case):
    return StrikeCandidate(
        timestamp=case["candidate_time"], confidence=.95,
        evidence={"dense_transition_confirmed": 1.0, "cue_geometry_confirmed": 1.0},
    )


@pytest.mark.parametrize("name", list(CASES))
def test_native_stop_discovery_bounds_terminal_shot(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    candidate = _candidate(case)
    stop = BallStopDetector(config).detect_stop(candidate, features, 8000)
    assert stop.confirmed
    assert 0 < stop.physical_stop_timestamp - candidate.timestamp <= case[
        "observed_boundary_within_seconds"
    ]
    # This return-to-table boundary follows a hidden interval. It bounds the
    # automatic edit without claiming the physical stop was visible.
    if case["boundary_is_upper_bound"]:
        assert stop.manual_review_required
        assert stop.end_confidence <= .70


@pytest.mark.parametrize("name", list(CASES))
def test_coarse_stop_coverage_requests_native_recovery(config, tmp_path, name):
    case = CASES[name]
    candidate = _candidate(case)
    native = [FrameFeatures.model_validate(f) for f in case["features"]]
    coverage = [f.model_copy() for f in native if f.t <= candidate.timestamp + 2]
    next_sample = candidate.timestamp + 2.5
    for feature in native:
        if feature.t >= next_sample:
            coarse = feature.model_copy()
            coarse.observation_fps = 2
            coverage.append(coarse)
            next_sample = feature.t + .5
    detector = BallStopDetector(config)
    assert not detector.detect_stop(candidate, coverage, 8000).confirmed
    analyzer = Analyzer(config, tmp_path / "job")
    assert analyzer._unresolved_stop_candidates([candidate], coverage, 8000) == [candidate]
    assert analyzer._unresolved_stop_candidates([candidate], native, 8000) == []


def test_sparse_earlier_stillness_revisits_a_borrowed_verified_stop(config, tmp_path):
    case = OBSERVATIONS["borrowed_later_stop"]
    candidate = _candidate(case)
    candidate.evidence.update({
        "refined_stop_timestamp": case["late_declared_stop"],
        "refined_stop_confidence": .95,
    })
    sparse = [FrameFeatures.model_validate(f) for f in case["sparse_features"]]
    native = [FrameFeatures.model_validate(f) for f in case["features"]]
    analyzer = Analyzer(config, tmp_path / "job")
    # A later native boundary cannot hide an earlier gap containing stationary
    # full-table samples. Those sparse samples request decoding, not certainty.
    assert analyzer._unresolved_stop_candidates([candidate], sparse, 8000) == [candidate]
    stop = analyzer.segmenter.ball_stop.detect_stop(candidate, native, 8000)
    assert stop.confirmed
    assert stop.physical_stop_timestamp - candidate.timestamp <= 10
    assert case["late_declared_stop"] - stop.physical_stop_timestamp > 30
    assert analyzer._unresolved_stop_candidates([candidate], native, 8000) == []


def test_bridge_identity_excursion_returns_to_quiet_white_without_a_strike(config):
    features = [FrameFeatures.model_validate(f) for f in OBSERVATIONS["bridge_identity_switch"]["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert not [c for c in candidates if 2131.8 < c.timestamp < 2132.6]
