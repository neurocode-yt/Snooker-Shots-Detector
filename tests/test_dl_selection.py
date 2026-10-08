"""Neural-only event selection, edit estimates, and safe rendering contracts."""

import json

import numpy as np
import pytest

from snooker_ai.dl.selection import predict_event_times, select_highlights
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.rendering.mix import plan_mix
from snooker_ai.types import AnalysisResult, VideoMetadata


def sequence(duration=30.0, cadence=.2):
    times = np.round(np.arange(0, duration + cadence / 2, cadence), 8)
    probabilities = np.full((len(times), 5), .02)
    return times, probabilities


def set_event(times, probabilities, strike, confidence=.9):
    probabilities[np.argmin(np.abs(times - strike)), 0] = confidence


def validated():
    return {"head_validation": {head: {"status": "positive_and_negative_labels"}
                                for head in ("keep", "end", "replay", "handling")}}


def keep_and_end(times, probabilities, start, end):
    probabilities[(times >= start) & (times <= end), 1] = .9
    probabilities[np.argmin(np.abs(times - end)), 2] = .95


def assert_exportable(config, shots, duration):
    Exporter(config)._validate_strict_boundaries(shots, source_duration=duration, source_fps=25)
    for shot in shots:
        assert shot.cue_strike == shot.cue_strike_timestamp
        assert shot.clip_start == shot.clip_start_timestamp
        assert shot.clip_end == shot.clip_end_timestamp
        assert shot.ball_motion_end == shot.physical_stop_timestamp
        assert shot.evidence["stop_confirmed"] is False
        assert shot.evidence["physical_stop_observed"] is False
        assert shot.evidence["boundary_kind"] == "neural_edit_estimate"
        assert shot.evidence["end_before_ball_stop_seconds"] == 0


def test_learned_event_and_edit_heads_create_compatible_estimate(config):
    times, output = sequence()
    set_event(times, output, 5)
    keep_and_end(times, output, 4, 10)
    candidates, shots, diagnostics = select_highlights(times, output, 30, validated(), {"event": .7})
    assert [candidate.timestamp for candidate in candidates] == [5]
    assert [(shot.clip_start, shot.clip_end) for shot in shots] == [(3, 10)]
    assert shots[0].manual_review_required is False
    assert shots[0].evidence["boundary_reason"] == "learned_end_and_keep"
    assert diagnostics["included_shots"] == 1
    assert_exportable(config, shots, 30)


def test_calibrated_event_threshold_changes_selection():
    times, output = sequence()
    set_event(times, output, 5, .65)
    assert not select_highlights(times, output, 30, {}, {"event": .7})[0]
    assert len(select_highlights(times, output, 30, {}, {"event": .6})[0]) == 1


def test_nms_keeps_stronger_duplicate_peak():
    times, output = sequence()
    set_event(times, output, 5, .85)
    set_event(times, output, 5.4, .95)
    candidates, _, _ = select_highlights(times, output, 30, {}, {})
    assert [candidate.timestamp for candidate in candidates] == [5.4]


def test_uncertainty_overlap_suppresses_broad_duplicate_beyond_point_radius():
    times, output = sequence()
    output[(times >= 4) & (times <= 8), 0] = .65
    set_event(times, output, 5, .95)
    set_event(times, output, 6.2, .9)
    candidates, _, _ = select_highlights(times, output, 30, {"max_uncertainty_seconds": 4}, {})
    assert [candidate.timestamp for candidate in candidates] == [5]
    assert candidates[0].uncertainty_start <= 4
    assert candidates[0].uncertainty_end > 6.2


def test_plateau_produces_one_contact_with_uncertainty():
    times, output = sequence()
    output[(times >= 5) & (times <= 5.8), 0] = .9
    candidates, _, _ = select_highlights(times, output, 30, {}, {})
    assert len(candidates) == 1
    assert candidates[0].uncertainty_start < 5
    assert candidates[0].uncertainty_end > 5.8


@pytest.mark.parametrize("status", ["unvalidated", "positive_only_labels", "negative_only_labels"])
def test_unvalidated_endpoint_heads_use_bounded_review_fallback(config, status):
    times, output = sequence()
    set_event(times, output, 5)
    keep_and_end(times, output, 4, 7.5)
    settings = {"head_validation": {"keep": {"status": status}, "end": {"status": status}},
                "fallback_post_seconds": 6}
    _, shots, diagnostics = select_highlights(times, output, 30, settings, {})
    assert shots[0].clip_end == 11
    assert shots[0].manual_review_required
    assert shots[0].evidence["boundary_reason"] == "bounded_uncertainty_fallback"
    assert "end" in diagnostics["unvalidated_heads"]
    assert_exportable(config, shots, 30)


def test_short_keep_dip_does_not_cut_off_continuing_action():
    times, output = sequence()
    set_event(times, output, 5)
    keep_and_end(times, output, 4, 10)
    output[np.argmin(np.abs(times - 8)), 1] = .02
    _, shots, _ = select_highlights(times, output, 30, validated(), {})
    assert shots[0].clip_end == 10


def test_missing_learned_endpoint_does_not_include_unbounded_waiting():
    times, output = sequence(200)
    set_event(times, output, 5)
    output[:, 1] = .99
    _, shots, _ = select_highlights(times, output, 200, validated(), {})
    assert shots[0].clip_end == 13
    assert shots[0].manual_review_required


def test_close_contacts_allocate_nonoverlapping_source_and_keep_visibility(config):
    times, output = sequence()
    for strike in (5, 8, 12):
        set_event(times, output, strike)
    _, shots, _ = select_highlights(times, output, 30, {}, {})
    assert len(shots) == 3
    assert all(left.clip_end <= right.clip_start for left, right in zip(shots, shots[1:]))
    assert all(shot.clip_end - shot.cue_strike >= 2 for shot in shots)
    assert all(shot.duration() >= 4 for shot in shots)
    assert shots[1].evidence["pre_roll_trimmed_seconds"] > 0
    assert_exportable(config, shots, 30)


def test_unavoidable_source_budget_marks_review_and_preserves_both_contacts(config):
    times, output = sequence()
    for strike in (5, 6.2):
        set_event(times, output, strike)
    _, shots, _ = select_highlights(times, output, 30, {}, {})
    assert len(shots) == 2
    assert shots[0].clip_end <= shots[1].clip_start
    assert shots[0].manual_review_required
    assert shots[0].clip_start <= 5 <= shots[0].clip_end
    assert shots[1].clip_start <= 6.2 <= shots[1].clip_end
    assert_exportable(config, shots, 30)
    assert plan_mix(shots, 25, .24).overlaps == [0]


@pytest.mark.parametrize("strike", [0, 29])
def test_source_edges_have_honest_available_viewing_budgets(config, strike):
    times, output = sequence()
    set_event(times, output, strike)
    _, shots, _ = select_highlights(times, output, 30, {}, {})
    assert len(shots) == 1
    assert 0 <= shots[0].clip_start <= strike <= shots[0].clip_end <= 30
    assert_exportable(config, shots, 30)


def test_confident_replay_is_excluded_and_caps_live_windows(config):
    times, output = sequence()
    for strike in (5, 15, 18):
        set_event(times, output, strike, .99)
    output[(times >= 14) & (times <= 16), 3] = .99
    _, shots, diagnostics = select_highlights(times, output, 30, validated(), {})
    replay = next(shot for shot in shots if shot.cue_strike == 15)
    live = [shot for shot in shots if shot.included]
    assert replay.included is False and replay.possible_replay is True
    assert replay.manual_review_required
    assert len(live) == 2
    assert live[0].clip_end <= 13.9
    assert live[1].clip_start >= 16.1
    assert diagnostics["excluded_replays"] == 1
    assert_exportable(config, shots, 30)


@pytest.mark.parametrize("status", ["unvalidated", "positive_only_labels"])
def test_unvalidated_replay_probability_does_not_silently_remove_shot(status):
    times, output = sequence()
    set_event(times, output, 5)
    output[:, 3] = .99
    settings = {"head_validation": {"replay": {"status": status}}}
    _, shots, _ = select_highlights(times, output, 30, settings, {})
    assert shots[0].included
    assert not shots[0].possible_replay
    assert shots[0].manual_review_required
    assert shots[0].evidence["replay_uncertain"]


def test_isolated_replay_spike_is_reviewed_without_exclusion():
    times, output = sequence()
    set_event(times, output, 5)
    keep_and_end(times, output, 4, 10)
    output[np.argmin(np.abs(times - 5)), 3] = .99
    _, shots, _ = select_highlights(times, output, 30, validated(), {})
    assert shots[0].included
    assert shots[0].manual_review_required
    assert shots[0].evidence["replay_uncertain"]


def test_strong_event_is_kept_while_referee_handling_coexists():
    times, output = sequence()
    set_event(times, output, 5, .99)
    output[(times >= 4) & (times <= 6), 4] = .99
    _, shots, _ = select_highlights(times, output, 30, validated(), {"event": .7})
    assert shots[0].included
    assert shots[0].manual_review_required
    assert shots[0].evidence["handling_event_conflict"]


def test_weak_handling_event_is_an_excluded_review_record():
    times, output = sequence()
    set_event(times, output, 5, .6)
    output[(times >= 4) & (times <= 6), 4] = .99
    _, shots, diagnostics = select_highlights(times, output, 30, validated(), {})
    assert not shots[0].included
    assert shots[0].manual_review_required
    assert diagnostics["excluded_handling"] == 1


def test_positive_only_handling_labels_never_veto_a_contact():
    times, output = sequence()
    set_event(times, output, 5, .6)
    output[:, 4] = .99
    settings = {"head_validation": {"handling": {"status": "positive_only_labels"}}}
    _, shots, _ = select_highlights(times, output, 30, settings, {})
    assert shots[0].included
    assert shots[0].evidence["handling_uncertain"]


def test_mix_does_not_cover_event_or_post_impact_uncertainty(config):
    times, output = sequence()
    for strike in (5, 20):
        set_event(times, output, strike)
        keep_and_end(times, output, strike - 1, strike + 8)
    _, shots, _ = select_highlights(times, output, 30, validated(), {})
    mix = plan_mix(shots, 25, .24)
    for index, shot in enumerate(shots):
        incoming = mix.overlaps[index - 1] if index else 0
        outgoing = mix.overlaps[index] if index < len(mix.overlaps) else 0
        assert shot.clip_start + incoming < shot.evidence["event_uncertainty_start"]
        assert shot.clip_start + mix.durations[index] - outgoing >= (
            shot.evidence["event_uncertainty_end"] + 2 - .04
        )
    assert_exportable(config, shots, 30)


def test_gaps_do_not_make_separate_replay_spikes_a_confident_span():
    times = np.array([0, .2, .4, 5, 10, 10.2, 10.4, 20, 20.2])
    output = np.full((len(times), 5), .02)
    output[times == 5, 0] = .99
    output[(times == 5) | (times == 10), 3] = .99
    _, shots, diagnostics = select_highlights(times, output, 30, validated(), {})
    assert shots[0].included
    assert diagnostics["replay_spans"] == []
    assert shots[0].manual_review_required
    assert shots[0].evidence["extraction_gap"]


def test_one_coarse_prediction_does_not_confirm_a_replay_interval():
    times, output = sequence(cadence=1)
    set_event(times, output, 5)
    output[times == 5, 3] = .99
    _, shots, _ = select_highlights(times, output, 30, validated(), {})
    assert shots[0].included
    assert shots[0].evidence["replay_uncertain"]


def test_ambiguous_neural_sequences_preserve_export_and_nonoverlap_contracts(config):
    rng = np.random.default_rng(43)
    for _ in range(20):
        times, output = sequence(50)
        output[:] = rng.uniform(0, .25, output.shape)
        for index in rng.choice(len(times), size=15, replace=False):
            output[index, 0] = rng.uniform(.55, .99)
            end = min(len(times), index + int(rng.integers(5, 50)))
            output[index:end, 1] = .9
            output[end - 1, 2] = .95
        for head in (3, 4):
            first = int(rng.integers(0, len(times) - 20))
            output[first:first + 20, head] = .99
        _, shots, _ = select_highlights(times, output, 50, validated(), {})
        live = [shot for shot in shots if shot.included]
        assert all(left.clip_end <= right.clip_start for left, right in zip(live, live[1:]))
        assert_exportable(config, shots, 50)


def test_empty_probabilities_are_a_valid_empty_result():
    candidates, shots, diagnostics = select_highlights(np.array([]), np.empty((0, 5)), 0, {}, {})
    assert candidates == shots == []
    assert diagnostics["included_shots"] == 0


def test_selected_probability_evidence_serializes_in_saved_analysis():
    times, output = sequence(12)
    set_event(times, output, 5)
    # Numpy comparisons must not leak numpy.bool_ into Pydantic's Any evidence.
    output[times == 5, 3] = .95
    candidates, shots, diagnostics = select_highlights(times, output, np.float64(12), {}, {})
    result = AnalysisResult(
        job_id="neural-serialization", source_path="source.mp4",
        metadata=VideoMetadata(path="source.mp4", duration=12),
        shots=shots, strike_candidates=candidates,
    )
    encoded = json.loads(result.model_dump_json())
    evidence = encoded["shots"][0]["evidence"]
    assert evidence["replay_uncertain"] is True
    assert evidence["handling_uncertain"] is False
    assert evidence["stop_confirmed"] is False
    assert encoded["shots"][0]["clip_end"] == 12
    assert json.loads(json.dumps(diagnostics))["stop_confirmed"] is False


def test_event_calibration_api_matches_decoded_candidates():
    times, output = sequence()
    output[(times >= 5) & (times <= 5.8), 0] = .9
    set_event(times, output, 12, .85)
    set_event(times, output, 12.4, .95)
    settings = {"nms_seconds": 1.0}
    candidates, _, _ = select_highlights(times, output, 30, settings, {"event": .7})
    expected = [candidate.timestamp for candidate in candidates]
    assert predict_event_times(times, output, .7, settings, duration=30) == expected
    assert predict_event_times(times, output, .7, settings) == expected
    assert predict_event_times(np.array([]), np.empty((0, 5)), .7) == []


@pytest.mark.parametrize("values,expected", [
    ([.6, .8, .9, .7], [2.2, 100]),
    ([.9, .9, .9, .9], [2, 100]),
])
def test_event_peaks_on_both_sides_of_gap_are_independent(values, expected):
    times = np.array([2, 2.2, 100, 100.2])
    output = np.full((len(times), 5), .02)
    output[:, 0] = values
    assert predict_event_times(times, output, .5) == expected
    candidates, _, _ = select_highlights(times, output, 110, {}, {"event": .5})
    assert [candidate.timestamp for candidate in candidates] == expected


def test_isolated_reviewed_frames_are_not_one_plateau_or_nms_window():
    times = np.array([2, 100, 1000])
    output = np.full((len(times), 5), .02)
    output[:, 0] = [.9, .9, .95]
    assert predict_event_times(times, output, .5) == [2, 100, 1000]
    candidates, _, _ = select_highlights(times, output, 1010, {}, {})
    assert len(candidates) == 3
    assert all(candidate.uncertainty_end - candidate.uncertainty_start <= 1 for candidate in candidates)


@pytest.mark.parametrize("probabilities", [[.9, .95, .02], [.9, .9, .9]])
def test_learned_end_peak_is_not_suppressed_by_a_later_observation_block(probabilities):
    times = np.array([0, .2, 5, 5.2, 5.4, 8, 20, 20.2, 20.4, 30])
    output = np.full((len(times), 5), .02)
    output[times == 5, 0] = .9
    output[(times == 8) | (times == 20) | (times == 20.2), 2] = probabilities
    _, shots, _ = select_highlights(times, output, 30, {"head_validation": {"end": True}}, {})
    assert shots[0].clip_end == 8
    assert shots[0].evidence["boundary_reason"] == "learned_end"
    assert shots[0].manual_review_required


def test_keep_fall_across_unobserved_gap_is_not_a_learned_endpoint():
    times = np.array([0, .2, 5, 5.2, 5.4, 20, 20.2, 20.4, 30])
    output = np.full((len(times), 5), .02)
    output[times == 5, 0] = .9
    output[(times >= 5) & (times <= 5.4), 1] = .9
    _, shots, _ = select_highlights(times, output, 30, {"head_validation": {"keep": True}}, {})
    assert shots[0].clip_end == 13
    assert shots[0].evidence["boundary_reason"] == "bounded_uncertainty_fallback"


@pytest.mark.parametrize("threshold", [np.nan, -1, 1.1])
def test_event_calibration_api_rejects_invalid_thresholds(threshold):
    times, output = sequence()
    with pytest.raises(ValueError):
        predict_event_times(times, output, threshold)


@pytest.mark.parametrize("fault", ["shape", "nan", "range", "order", "negative", "beyond", "threshold"])
def test_invalid_model_arrays_fail_explicitly(fault):
    times, output = sequence()
    thresholds = {}
    if fault == "shape":
        output = output[:, :4]
    elif fault == "nan":
        output[0, 0] = np.nan
    elif fault == "range":
        output[0, 0] = 1.1
    elif fault == "order":
        times[1] = times[0]
    elif fault == "negative":
        times[0] = -1
    elif fault == "beyond":
        times[-1] = 31
    elif fault == "threshold":
        thresholds["event"] = -1
    with pytest.raises(ValueError):
        select_highlights(times, output, 30, {}, thresholds)
    with pytest.raises(ValueError):
        predict_event_times(times, output, thresholds.get("event", .5), duration=30)
