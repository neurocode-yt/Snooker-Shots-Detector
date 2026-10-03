import pytest

from snooker_ai.evaluation.recall import evaluate_broadcast_recall
from snooker_ai.types import ExportRequest


def reference(contacts):
    return {"windows": [{"name": "complete", "start": 10., "end": 30., "contacts": contacts}]}


def test_recall_counts_misses_duplicates_and_excludes_unlabelled_time():
    labels = reference([{"lower": 12., "upper": 12.1}, {"lower": 25., "upper": 25.2}])
    shots = [{"cue_strike": t} for t in [2., 12.05, 12.2, 20.]]
    result = evaluate_broadcast_recall(shots, labels)
    assert (result["matched"], result["missed"], result["false_positive"]) == (1, 1, 2)
    assert result["recall"] == .5
    assert result["precision"] == pytest.approx(1/3)


def test_recall_respects_excluded_and_replay_clips():
    shots = [{"cue_strike": 15., "included": False},
             {"cue_strike": 20., "possible_replay": True}]
    result = evaluate_broadcast_recall(shots, reference([]))
    assert result["false_positive"] == 0
    assert result["recall"] is None


def test_hidden_contact_bounds_are_not_an_exact_timestamp():
    labels = reference([{"lower": 12., "upper": 12.4}])
    result = evaluate_broadcast_recall([{"cue_strike": 12.39}], labels, tolerance=0.)
    assert result["recall"] == 1
    assert result["windows"][0]["matched"][0]["error_outside_bounds_seconds"] == 0


def test_recall_rejects_double_counted_label_windows():
    labels = reference([])
    labels["windows"] *= 2
    with pytest.raises(ValueError, match="overlap"):
        evaluate_broadcast_recall([], labels)


def test_ambiguous_pairs_preserve_maximum_one_to_one_match_count():
    labels = reference([{"lower": 12., "upper": 12.}, {"lower": 12.8, "upper": 12.8}])
    result = evaluate_broadcast_recall([{"cue_strike": 11.6}, {"cue_strike": 12.3}], labels, .51)
    assert (result["matched"], result["missed"], result["false_positive"]) == (2, 0, 0)


def test_cardinality_tie_prefers_lower_total_timing_error():
    labels = reference([{"lower": 12., "upper": 12.}, {"lower": 12.8, "upper": 12.8}])
    result = evaluate_broadcast_recall([{"cue_strike": 12.1}, {"cue_strike": 12.7}], labels, 1)
    assert result["matched"] == 2
    assert sum(p["error_outside_bounds_seconds"] for p in result["windows"][0]["matched"]) == pytest.approx(.2)


@pytest.mark.parametrize("timestamp", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_predictions_are_rejected_even_outside_labelled_windows(timestamp):
    with pytest.raises(ValueError, match="timestamps must be finite"):
        evaluate_broadcast_recall([{"cue_strike": timestamp}], reference([]))


@pytest.mark.parametrize("replay", [
    {"possible_replay": True}, {"linked_live_shot_id": 4},
    {"camera_views": ["replay"]}, {"camera_views": ["slow_motion_replay"]},
])
def test_analysis_uses_renderer_replay_selection_but_delivered_false_clips_still_count(replay):
    clip = {"cue_strike": 15., **replay}
    assert evaluate_broadcast_recall([clip], reference([]))["false_positive"] == 0
    assert evaluate_broadcast_recall([clip], reference([]), delivered=True)["false_positive"] == 1
    assert evaluate_broadcast_recall([clip], reference([]),
                                    export_request=ExportRequest(include_replays=True))["false_positive"] == 1


def test_replay_re_admitted_by_export_counts_even_when_builder_excluded_it():
    clip = {"cue_strike": 15., "possible_replay": True, "included": False}
    assert evaluate_broadcast_recall([clip], reference([]),
                                    export_request=ExportRequest(include_replays=True))["false_positive"] == 1
    assert evaluate_broadcast_recall([clip], reference([]), delivered=True)["false_positive"] == 1


def test_explicit_export_thresholds_filter_live_records():
    clips = [{"cue_strike": 15., "included": False, "shot_confidence": .9, "importance": .7},
             {"cue_strike": 20., "shot_confidence": .3, "importance": .7},
             {"cue_strike": 25., "shot_confidence": .9, "importance": .2}]
    result = evaluate_broadcast_recall(clips, reference([]),
                                      export_request=ExportRequest(only_included=False,
                                                                   min_confidence=.8, min_importance=.5))
    assert result["false_positive"] == 1
    assert result["windows"][0]["false_contacts"] == [15.]
