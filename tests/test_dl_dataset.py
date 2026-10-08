from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from snooker_ai.dl.dataset import (
    HEADS, build_targets, evaluate_contacts, evaluate_video, load_feature_video,
    load_manifest, split_manifest, validate_manifest,
)


def source() -> dict:
    return {"id": "clip", "group_id": "match_a", "split": "train", "duration": 20,
            "source_path": "example.mp4", "features_path": "example.npz",
            "reviewed_intervals": [{"start": 2, "end": 8}],
            "contacts": [{"lower": 4.1, "upper": 4.3, "provenance": "native-source-review"}],
            "clip_reviewed_intervals": [{"start": 2, "end": 8, "weight": 0.35}],
            "clips": [{"start": 3, "end": 6, "contact_lower": 4.1, "contact_upper": 4.3,
                       "weight": 0.35, "provenance": "classic-generated-reviewed"}],
            "labels": []}


def test_partial_review_leaves_unknown_frames_and_unknown_behaviors_masked():
    labels = build_targets(np.arange(21), source())
    assert not labels.mask[[0, 1, 9, 20]].any()
    assert not labels.mask[:, 3:].any()
    assert labels.mask[2, 0] == 1
    assert labels.mask[4, 0] == 0  # uncertain contact bag is not a framewise negative
    assert labels.contact_bags[0].tolist() == [4]
    assert labels.mask[3, 1] == pytest.approx(0.35)
    assert labels.targets[3, 1] == 1
    assert labels.targets[2, 1] == 0


def test_one_uncertain_contact_is_a_bag_not_many_positive_event_frames():
    video = source()
    video["contacts"][0].update(lower=4, upper=6)
    labels = build_targets(np.arange(21), video)
    assert len(labels.contact_bags) == 1
    assert labels.contact_bags[0].tolist() == [4, 5, 6]
    assert not labels.mask[4:7, 0].any()
    assert not labels.targets[:, 0].any()


def test_missing_extracted_contact_samples_raise_instead_of_dropping_positive():
    with pytest.raises(ValueError, match="No extracted feature sample"):
        build_targets(np.arange(8, 21), source())


def test_missing_minutes_cannot_create_a_contact_bag_or_an_editorial_endpoint():
    with pytest.raises(ValueError, match='No extracted feature sample'):
        build_targets(np.array([0., 100.]), source())


def test_only_explicit_clean_clip_supervises_handling_and_replay_negatives():
    video = source()
    video["clips"][0].update(confirmed_replay_free=True, confirmed_handling_free=True)
    video["labels"] = [{"head": "handling", "start": 7, "end": 8, "value": 1,
                        "weight": 1, "provenance": "source-review"}]
    labels = build_targets(np.arange(21), video)
    assert labels.mask[3:7, 3].tolist() == pytest.approx([0.35] * 4)
    assert not labels.mask[2, 3:].any()
    assert labels.targets[7, 4] == 1
    assert labels.mask[7, 4] == 1


def test_match_cuts_cannot_leak_across_train_dev_splits():
    first = source()
    second = copy.deepcopy(first)
    second.update(id="another_frame_same_match", split="dev")
    manifest = {"manifest_version": 1, "heads": list(HEADS), "videos": [first, second]}
    with pytest.raises(ValueError, match="leaks"):
        validate_manifest(manifest)
    second["group_id"] = "different_match"
    parts = split_manifest(manifest)
    assert len(parts["train"]["videos"]) == len(parts["dev"]["videos"]) == 1
    assert parts["holdout"]["videos"] == []


def test_contact_must_be_inside_exhaustively_reviewed_scope():
    video = source()
    video["contacts"][0].update(lower=10, upper=11)
    with pytest.raises(ValueError, match="wholly"):
        validate_manifest({"manifest_version": 1, "heads": list(HEADS), "videos": [video]})


def test_feature_archive_validates_cadence_dimension_and_finite_features(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, timestamps=np.arange(5), features=np.ones((5, 3), np.float32),
             feature_spec=json.dumps({"dimension": 3, "extractor": "RGB-motion"}))
    times, features, spec = load_feature_video(path)
    assert times.dtype == np.float64 and features.dtype == np.float32
    assert spec["dimension"] == 3
    np.savez(path, timestamps=np.array([0, 1, 1, 3, 4]), features=features,
             feature_spec=json.dumps({"dimension": 3}))
    with pytest.raises(ValueError, match="strictly"):
        load_feature_video(path)
    np.savez(path, timestamps=np.arange(5), features=features,
             feature_spec=json.dumps({"dimension": 4}))
    with pytest.raises(ValueError, match="dimension"):
        load_feature_video(path)


def test_matching_maximizes_coverage_instead_of_greedily_matching_first_contact():
    report = evaluate_contacts([1.41, 1.0],
        [{"lower": 1, "upper": 1.4}, {"lower": 1.45, "upper": 2}],
        [{"start": 0, "end": 3}], tolerance_seconds=0.06)
    assert report["true_positive"] == 2
    assert report["false_positive"] == report["false_negative"] == 0


def test_duplicate_predicted_contacts_reduce_precision_and_unknown_source_is_ignored():
    report = evaluate_contacts([4.2, 4.25, 15], source()["contacts"],
                               source()["reviewed_intervals"])
    assert report["precision"] == 0.5
    assert report["recall"] == 1
    assert report["unreviewed_predictions_ignored"] == 1


def test_empty_positive_reference_does_not_report_fake_perfect_recall():
    report = evaluate_contacts([4], [], [{"start": 0, "end": 8}])
    assert report["false_positive"] == 1
    assert report["precision"] == 0
    assert report["recall"] is None


def test_boundary_errors_and_referee_clip_contamination_are_separate_from_event_recall():
    video = source()
    video["labels"] = [{"head": "handling", "start": 7, "end": 8, "value": 1,
                        "provenance": "source-handling-review"},
                       {"head": "replay", "start": 9, "end": 11, "value": 1,
                        "provenance": "source-replay-review"}]
    report = evaluate_video([{"start": 3.1, "end": 7.5, "contact": 4.2},
                             {"start": 9.5, "end": 10.5, "contact": 10}], video)
    assert report["recall"] == 1
    assert report["false_handling_clips"] == report["false_replay_clips"] == 1
    assert report["boundary_errors"][0]["end_error_seconds"] == 1.5
    assert "not physical-stop truth" in report["boundary_reference_kind"]


def test_late_event_timing_can_still_keep_the_contact_in_delivered_windows():
    video = source()
    report = evaluate_video([{'contact': 5.5, 'start': 3, 'end': 7}], video, tolerance_seconds=0)
    assert report['recall'] == 0
    assert report['source_contact_coverage']['fully_covered_recall'] == 1


def test_accurate_event_record_does_not_hide_an_excluded_contact_window():
    video = source()
    report = evaluate_video([{'contact': 4.2, 'start': 5, 'end': 7}], video)
    assert report['recall'] == 1
    assert report['source_contact_coverage']['missing_brackets'] == 1


def test_partial_contact_brackets_and_exclusive_clip_end_are_not_full_coverage():
    video = source()
    video['contacts'] = [{'lower': 4, 'upper': 4.5}, {'lower': 8, 'upper': 8}]
    report = evaluate_video([{'contact': 4.2, 'start': 4.2, 'end': 6},
                             {'contact': 8, 'start': 7, 'end': 8}], video)
    coverage = report['source_contact_coverage']
    assert coverage['fully_covered_brackets'] == 0
    assert coverage['partial_reference_indexes'] == [0]
    assert coverage['missing_reference_indexes'] == [1]


def test_repository_manifest_has_only_match_isolated_development_labels():
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "data/evaluation/dl/manifest.json")
    assert manifest["independent_holdout_available"] is False
    assert sum(len(row["contacts"]) for row in manifest["videos"]) == 121
    splits = split_manifest(manifest)
    assert splits["holdout"]["videos"] == []
    assert {video["group_id"] for video in splits["train"]["videos"]}.isdisjoint(
        {video["group_id"] for video in splits["dev"]["videos"]})


def test_handling_only_review_does_not_supervise_no_stroke_or_no_useful_footage():
    manifest = load_manifest(Path(__file__).resolve().parents[1] / 'data/evaluation/dl/manifest.json')
    video = next(row for row in manifest['videos'] if row['id'] == 'zhao_trump_referee_oct07')
    labels = build_targets(np.array([876., 1045., 1073., 1103.]), video)
    assert not labels.mask[:, 0].any()
    assert not labels.mask[:, 1:3].any()
    np.testing.assert_array_equal(labels.targets[:, 4], np.ones(4))
    assert labels.mask[:, 4].all()


def test_export_evaluation_keeps_delivered_replays_visible_in_contamination_metrics(tmp_path):
    from tools.dl_evaluate import prediction_clips
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"source_path": "example.mp4", "shots": [
        {"cue_strike": 4.2, "clip_start": 3, "clip_end": 6, "included": True},
        {"cue_strike": 10, "clip_start": 9.5, "clip_end": 10.5,
         "included": True, "possible_replay": True},
        {"cue_strike": 14, "clip_start": 13, "clip_end": 15, "included": False},
    ]}), encoding="utf-8")
    clips = prediction_clips(path, "example.mp4")
    assert len(clips) == 2
    video = source()
    video["labels"] = [{"head": "replay", "start": 9, "end": 11,
                        "value": 1, "provenance": "source-review"}]
    assert evaluate_video(clips, video)["false_replay_clips"] == 1
    with pytest.raises(ValueError, match="different source"):
        prediction_clips(path, "unrelated.mp4")


@pytest.mark.parametrize("field", ["reviewed_intervals", "clip_reviewed_intervals",
                                   "contacts", "clips", "labels"])
@pytest.mark.parametrize("invalid_weight", [float("nan"), float("inf"), -0.1, 1.1, None])
def test_malformed_annotation_weights_cannot_silently_mask_or_poison_training(field, invalid_weight):
    video = source()
    video["labels"] = [{"head": "replay", "start": 3, "end": 4, "value": 0,
                        "provenance": "source-review"}]
    video[field][0]["weight"] = invalid_weight
    with pytest.raises(ValueError, match="weights must be finite"):
        validate_manifest({"manifest_version": 1, "heads": list(HEADS), "videos": [video]})
    # Public target construction enforces the guard even without a manifest load.
    with pytest.raises(ValueError, match="weights must be finite"):
        build_targets(np.arange(21), video)


def test_reviewed_live_contact_implies_only_bracket_local_replay_negative():
    from tools.dl_prepare_data import live_contact_replay_negatives
    video = source()
    contacts_before = copy.deepcopy(video["contacts"])
    derived = live_contact_replay_negatives(video["contacts"])
    assert video["contacts"] == contacts_before
    assert len(derived) == 1
    assert derived[0]["head"] == "replay" and derived[0]["value"] == 0
    assert derived[0]["provenance"] == "derived_from_reviewed_live_contact:native-source-review"
    assert "handling remain unknown" in derived[0]["inference"]
    video["labels"] = derived
    labels = build_targets(np.array([0, 4, 4.1, 4.2, 4.3, 4.4, 20]), video)
    assert labels.mask[:, 3].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert not labels.mask[:, 4].any()


def test_repository_replay_negatives_preserve_every_live_contact_bracket_and_provenance():
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "data/evaluation/dl/manifest.json")
    for video in manifest["videos"]:
        inferred = [label for label in video["labels"]
                    if label["provenance"].startswith("derived_from_reviewed_live_contact:")]
        assert len(inferred) == len(video["contacts"])
        for label, contact in zip(inferred, video["contacts"]):
            assert (label["start"], label["end"]) == (contact["lower"], contact["upper"])
            assert label["head"] == "replay" and label["value"] == 0
            assert label["provenance"].endswith(contact["provenance"])


def test_editorial_uncertainty_masks_keep_edges_without_inventing_physical_stop_truth():
    video = source()
    video["clips"][0].update(start_lower=2.8, start_upper=3.2,
                             end_lower=5.8, end_upper=6.2, weight=0.7)
    times = np.array([2, 2.8, 3, 3.2, 4.2, 5.8, 6, 6.2, 7])
    labels = build_targets(times, video)
    assert labels.mask[[1, 2, 3, 5, 6, 7], 1].tolist() == [0] * 6
    assert labels.targets[[5, 6, 7], 2].tolist() == [1] * 3
    assert labels.mask[8, 2] == pytest.approx(0.35)
    assert not labels.targets[8, 2]


def test_source_adjudication_reports_original_and_corrected_contact_results():
    video = source()
    video["contacts"][0].update(original_bounds={"lower": 4.1, "upper": 4.3},
                                 lower=5.1, upper=5.3)
    report = evaluate_video([{"contact": 5.2, "start": 3, "end": 6}], video,
                            tolerance_seconds=0)
    assert report["true_positive"] == 1
    assert report["original_contact_metrics"]["true_positive"] == 0
    assert report["original_contact_metrics"]["false_negative"] == 1
    assert report["adjudicated_contact_count"] == 1


def test_frozen_selby_source_review_preserves_original_labels_and_scoped_supervision():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "data/evaluation/dl/manifest.json")
    video = next(row for row in manifest["videos"] if row["id"] == "selby_lisowski_benchmark")
    review = json.loads((root / "data/evaluation/dl/selby_opening_editorial_review.json").read_text())
    assert len(video["clips"]) == len(review["clips"]) == 12
    assert review["human_approved"] is False
    assert review["reviewer_consulted_model_predictions"] is False
    assert review["reviewer_consulted_classic_boundaries"] is False
    assert sum(row["head"] == "handling" and row["value"] == 0 for row in review["labels"]) == 12
    assert sum(row["head"] == "handling" and row["value"] == 1 for row in review["labels"]) == 4
    corrected = [row for row in video["contacts"] if "original_bounds" in row]
    assert len(corrected) == 1
    assert corrected[0]["original_bounds"] == {"lower": 441.0, "upper": 441.08}
    assert (corrected[0]["lower"], corrected[0]["upper"]) == (441.92, 441.96)
    # Classic's frozen reference remains the original source label.
    benchmark = json.loads((root / "benchmarks/selby_lisowski_recall.json").read_text())
    original = [row for window in benchmark["windows"] for row in window["contacts"]
                if row["description"] == "pink_pot" and row["lower"] == 441.0]
    assert len(original) == 1 and original[0]["upper"] == 441.08
    labels = build_targets(np.arange(0, video["duration"], 0.125), video)
    assert np.any(labels.mask[:, 1] > 0) and np.any(labels.mask[:, 2] > 0)
    assert np.any(labels.mask[:, 4] * labels.targets[:, 4] > 0)
    assert np.any(labels.mask[:, 4] * (1 - labels.targets[:, 4]) > 0)
