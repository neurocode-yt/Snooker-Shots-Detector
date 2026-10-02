"""Live rolls hidden by the ending replay wipe stay uncertain and automatic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.event_fusion.ball_stop import StopDetection
from snooker_ai.types import FrameFeatures, StrikeCandidate


@pytest.fixture
def replay_return_source():
    raw = json.loads((Path(__file__).parent / "fixtures" / "multicamera_replay_return.json").read_text())
    return (
        raw,
        [StrikeCandidate.model_validate(item) for item in raw["candidates"]],
        [FrameFeatures.model_validate(item) for item in raw["features"]],
        dict(raw["completed_live_stops"]),
    )


def test_source_replay_return_recovers_visible_roll_as_contact_upper_bound(config, replay_return_source):
    raw, candidates, features, stops = replay_return_source
    snapshots = ([c.model_dump() for c in candidates], [f.model_dump() for f in features], dict(stops))
    returned = ReplayDetector(config).returning_live_candidates(candidates, features, stops)
    assert len(returned) == 1
    candidate = returned[0]
    assert candidate.timestamp == pytest.approx(raw["expected_timestamp"])
    assert candidate.uncertainty_start == pytest.approx(raw["expected_uncertainty_start"])
    assert candidate.uncertainty_end == candidate.timestamp
    assert candidate.confidence == .70
    assert not candidate.possible_replay
    assert candidate.evidence["return_from_replay_visible_roll"] == 1
    assert candidate.evidence["contact_time_upper_bound"] == 1
    assert candidate.evidence["cue_ball_motion_confirmed"] == 1
    assert "dense_transition_confirmed" not in candidate.evidence
    assert candidate.evidence["cue_displacement_diameters"] >= .50
    assert candidate.evidence["cue_direction_consistency"] >= .80
    # The initial roll is proven before the source's red collision/rebound.
    assert candidate.evidence["replay_return_proof_timestamp"] < 2797.35
    assert [c.model_dump() for c in candidates] == snapshots[0]
    assert [f.model_dump() for f in features] == snapshots[1]
    assert stops == snapshots[2]


def test_source_return_survives_prior_replay_annotation(config, replay_return_source):
    _, candidates, features, stops = replay_return_source
    detector = ReplayDetector(config)
    detector.mark_candidates(candidates, features)
    assert candidates[1].possible_replay
    returned = detector.returning_live_candidates(candidates, features, stops)
    assert len(returned) == 1
    assert returned[0].timestamp > candidates[1].timestamp
    assert not returned[0].possible_replay


@pytest.mark.parametrize("stop_condition", ["missing", "after_opening", "latest_not_completed"])
def test_prior_live_must_have_completed_before_opening(config, replay_return_source, stop_condition):
    raw, candidates, features, stops = replay_return_source
    if stop_condition == "missing":
        stops = {}
    elif stop_condition == "after_opening":
        stops[candidates[0].timestamp] = 2790.5
    else:
        candidates.append(StrikeCandidate(timestamp=2787, confidence=.9))
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


def test_single_closing_graphic_cannot_recover_a_new_live_roll(config, replay_return_source):
    _, candidates, features, stops = replay_return_source
    for feature in features:
        if feature.t < 2791:
            feature.appearance_signature = []
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


@pytest.mark.parametrize("flag,value", [
    ("table_full_view", False), ("table_observable", False),
    ("match_context_valid", False), ("table_handling", True),
    ("broadcast_replay", True), ("observation_fps", 2),
    ("cue_ball_track_confidence", .69),
])
def test_unreliable_or_ineligible_return_does_not_create_shot(config, replay_return_source, flag, value):
    _, candidates, features, stops = replay_return_source
    for feature in features:
        if feature.t > 2797.1:
            setattr(feature, flag, value)
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


@pytest.mark.parametrize("known_quiet_seed", [False, True])
def test_quiet_address_after_wipe_is_not_a_hidden_contact(config, replay_return_source, known_quiet_seed):
    _, candidates, features, stops = replay_return_source
    if known_quiet_seed:
        first = next(f for f in features if f.t > 2797.1 and f.cue_ball_detected)
        first.observation_valid = True
        first.scene_cut_score = 0
    else:
        next(f for f in features if 2797.25 < f.t < 2797.26).cue_ball_stable_normalized_speed = 0
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


@pytest.mark.parametrize("fault", ["cut", "warmup_identity_jump", "reversal", "no_displacement", "hole"])
def test_motion_proof_cannot_cross_cuts_identity_jumps_or_unobserved_holes(config, replay_return_source, fault):
    _, candidates, features, stops = replay_return_source
    if fault == "cut":
        next(f for f in features if 2797.28 < f.t < 2797.29).camera_scene_id += 1
    elif fault == "warmup_identity_jump":
        next(f for f in features if 2797.22 < f.t < 2797.23).cue_ball_x += 200
    elif fault == "reversal":
        next(f for f in features if 2797.28 < f.t < 2797.29).cue_ball_x -= 20
    elif fault == "no_displacement":
        for feature in features:
            if feature.t > 2797.1 and feature.cue_ball_detected:
                feature.cue_ball_x, feature.cue_ball_y = 695, 424
    else:
        features = [f for f in features if not 2797.25 < f.t < 2797.29]
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


def test_existing_verified_live_contact_deduplicates_recovered_roll(config, replay_return_source):
    raw, candidates, features, stops = replay_return_source
    candidates.append(StrikeCandidate(timestamp=raw["expected_timestamp"]+.12, confidence=.9,
                                      evidence={"dense_transition_confirmed": 1}))
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


def test_first_return_near_deadline_can_complete_native_proof_after_deadline(config, replay_return_source):
    raw, candidates, features, stops = replay_return_source
    for feature in features:
        if feature.t > 2797.1:
            feature.t += .44
    returned = ReplayDetector(config).returning_live_candidates(candidates, features, stops)
    assert len(returned) == 1
    assert returned[0].timestamp == pytest.approx(raw["expected_timestamp"]+.44)
    assert returned[0].evidence["replay_return_proof_timestamp"] > 2797.087744610801+.60


def test_unsorted_inputs_are_not_reordered_and_recovered_proposal_is_idempotent(config, replay_return_source):
    _, candidates, features, stops = replay_return_source
    candidates.reverse()
    features.reverse()
    original_times = [f.t for f in features]
    detector = ReplayDetector(config)
    returned = detector.returning_live_candidates(candidates, features, stops)
    assert len(returned) == 1
    assert [f.t for f in features] == original_times
    assert detector.returning_live_candidates(candidates+returned, features, stops) == []


def test_replay_disabled_disables_return_recovery(config, replay_return_source):
    _, candidates, features, stops = replay_return_source
    config._data["replay"]["enabled"] = False
    assert ReplayDetector(config).returning_live_candidates(candidates, features, stops) == []


def test_pipeline_retains_recovered_live_roll_with_automatic_replay_free_clip(
    config, tmp_path, replay_return_source,
):
    raw, candidates, features, stops = replay_return_source
    prior = candidates[0]
    prior.evidence.update(refined_stop_timestamp=stops[prior.timestamp],
                          refined_stop_confidence=.68, refined_stop_upper_bound=1)
    analyzer = Analyzer(config, tmp_path/'job')
    analyzer.replay_det.mark_candidates(candidates, features)
    recovered = analyzer._recover_replay_returns(candidates, features, 2810)
    returning = [c for c in recovered if c.evidence.get("return_from_replay_visible_roll")]
    assert len(returning) == 1
    shots = analyzer.segmenter.build(returning, features, 2810)
    assert len(shots) == 1 and shots[0].included
    assert shots[0].manual_review_required  # Diagnostic, not approval for export.
    assert shots[0].clip_start >= raw["expected_uncertainty_start"]
    assert shots[0].evidence["usable_source_start_reason"] == "replay_clip_boundary"
    padding = shots[0].evidence["minimum_clip_transition_padding_seconds"]
    assert shots[0].duration()-padding >= 4


def test_pipeline_low_confidence_stale_stop_does_not_confirm_prior_live_end(
    config, tmp_path, monkeypatch, replay_return_source,
):
    _, candidates, features, stops = replay_return_source
    analyzer = Analyzer(config, tmp_path/'job')
    analyzer.replay_det.mark_candidates(candidates, features)
    end = stops[candidates[0].timestamp]
    monkeypatch.setattr(analyzer.segmenter.ball_stop, "detect_stop", lambda *a, **k: StopDetection(
        motion_start=candidates[0].timestamp, last_ball_motion_timestamp=end,
        physical_stop_timestamp=end, stop_confirmation_timestamp=end+.5,
        end_confidence=.55, start_confidence=.9, confirmed=True,
        reason="confirmed_stale_occlusion_override",
    ))
    recovered = analyzer._recover_replay_returns(candidates, features, 2810)
    assert not [c for c in recovered if c.evidence.get("return_from_replay_visible_roll")]
