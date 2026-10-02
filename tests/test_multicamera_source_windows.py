"""Regression observations from the supplied broadcast, without media files."""
import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import FrameFeatures, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper

CASES = json.loads((Path(__file__).parent / "fixtures/multicamera_visual_windows.json")
                   .read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("name", [
    "hidden_impact_then_camera_cut", "safety_launch", "oblique_red_pot", "short_blue_pot",
    "tight_preparation_then_overhead_launch", "bridge_after_referee_then_strike",
    "referee_colour_replacement",
    "stationary_white_during_camera_zoom",
])
def test_observed_broadcast_contacts_and_referee_replacement(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    if case["expected_contact"] is None:
        assert candidates == []
    else:
        assert len(candidates) == 1
        lo, hi = case["expected_contact"]
        assert lo <= candidates[0].timestamp <= hi


def test_observed_red_motion_prevents_white_only_stop(config):
    case = CASES["red_keeps_rolling_after_white_stops"]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=case["candidate_time"], confidence=.95), features, 195.4)
    assert stop.confirmed
    lo, hi = case["expected_stop"]
    assert lo <= stop.physical_stop_timestamp <= hi
    assert stop.physical_stop_timestamp > 193.8  # the white is already still


SPARSE = json.loads((Path(__file__).parent / "fixtures/multicamera_sparse_windows.json")
                    .read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("name", list(SPARSE))
def test_whole_match_discovery_covers_actual_multicamera_contact(config, name):
    case = SPARSE[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    proposed = StrikeDetector(config).detect_sparse_candidates(features)
    covering = [c for c in proposed if c.uncertainty_start <= case["contact"] <= c.uncertainty_end]
    assert covering
    assert all(c.evidence.get("dense_transition_confirmed", 0) == 0 for c in covering)


@pytest.mark.parametrize("flag", ["table_handling", "match_context_valid"])
def test_sparse_cue_address_cannot_override_handling_or_foreign_table(config, flag):
    features = [FrameFeatures.model_validate(f) for f in SPARSE["oblique_zoom"]["features"]]
    for feature in features:
        setattr(feature, flag, flag == "table_handling")
    assert StrikeDetector(config).detect_sparse_candidates(features) == []


@pytest.mark.parametrize("sparse_name,native_name", [
    ("breakoff_camera_cut", "pipeline_breakoff"),
    ("oblique_zoom", "pipeline_oblique_zoom"),
    ("delayed_motion_peak", "pipeline_tight_camera"),
    ("zoom_hidden_contact", "pipeline_zoom_hidden_contact"),
])
def test_actual_sparse_proposal_survives_native_pipeline(
    config, tmp_path, monkeypatch, sparse_name, native_name,
):
    sparse = SPARSE[sparse_name]
    reference = [FrameFeatures.model_validate(f) for f in sparse["features"]]
    native = [FrameFeatures.model_validate(f) for f in CASES[native_name]["features"]]
    analyzer = Analyzer(config, tmp_path / "job")
    proposed = analyzer._visual_proposals(reference)
    covering = [c for c in proposed if c.uncertainty_start <= sparse["contact"] <= c.uncertainty_end]
    candidate = min(covering, key=lambda c: abs(c.timestamp-sparse["contact"]))
    def extract(*args, start_time=0, end_time=0, **kwargs):
        return [f.model_copy() for f in native if start_time <= f.t <= end_time], [], []
    monkeypatch.setattr(analyzer, "_extract_features", extract)
    candidates, dense = analyzer._refine_candidate_windows(
        tmp_path / "source.mp4", None,
        TimeMapper(source_duration=8000, source_fps=25, analysis_fps=30),
        8000, [candidate], reference, resume=False)
    lo, hi = CASES[native_name]["expected_contact"]
    assert lo <= candidates[0].timestamp <= hi
    assert analyzer.segmenter._candidate_supported(candidates[0])
    assert any(f.contact_window for f in dense)
    # Global consolidation must retain a contact already confirmed locally.
    merged = analyzer._merge_feature_layers(reference, dense)
    analyzer.strike_det.score_frames(merged)
    analyzer.strike_det.refine_boundaries(candidates, merged)
    assert analyzer.segmenter._candidate_supported(candidates[0])
    assert lo <= candidates[0].timestamp <= hi


@pytest.mark.parametrize("name", [name for name in CASES
                                  if name.startswith("pipeline_") and "expected_contact" in CASES[name]])
def test_native_windows_from_whole_match_proposals_confirm_contact(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    lo, hi = case["expected_contact"]
    assert any(lo <= c.timestamp <= hi for c in candidates)


def test_native_stop_keeps_red_roll_through_its_collision(config):
    case = CASES["pipeline_slow_red_stop"]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=case["candidate_time"], confidence=.95), features, 195.4)
    assert stop.confirmed
    lo, hi = case["expected_stop"]
    assert lo <= stop.physical_stop_timestamp <= hi


def test_short_blue_impact_retains_later_native_confirmation_after_deduplication(config, tmp_path):
    features = [FrameFeatures.model_validate(f) for f in CASES["pipeline_blue"]["features"]]
    analyzer = Analyzer(config, tmp_path / "job")
    analyzer.strike_det.score_frames(features)
    current = analyzer.strike_det.detect_candidates(features)
    old = StrikeCandidate(timestamp=current[0].timestamp, confidence=.999,
                          uncertainty_start=current[0].timestamp-.05,
                          uncertainty_end=current[0].timestamp)
    merged = analyzer._deduplicate_candidates([old]+current)
    analyzer.strike_det.refine_boundaries(merged, features)
    assert len(merged) == 1
    assert analyzer.segmenter._candidate_supported(merged[0])
    assert abs(merged[0].timestamp-245.42) < .1
