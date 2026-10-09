"""Contract checks for the separate learned temporal backend."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snooker_ai.dl.temporal import HEAD_NAMES, TemporalModelConfig
from snooker_ai.dl.training import TrainingConfig, event_peaks, split_video_groups


def test_group_split_keeps_related_videos_together() -> None:
    videos = [
        {"id": "a1", "group_id": "match-a"},
        {"id": "a2", "group_id": "match-a"},
        {"id": "b", "group_id": "match-b"},
        {"id": "c", "group_id": "match-c"},
    ]
    first = split_video_groups(videos, seed=7, test_groups=["match-c"])
    second = split_video_groups(videos, seed=7, test_groups=["match-c"])
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"] + first["validation"]).isdisjoint(first["test"])
    assert first["test"] == ["match-c"]


def test_single_match_cannot_masquerade_as_holdout() -> None:
    videos = [{"id": "frame-a", "group_id": "same-match"}, {"id": "frame-b", "group_id": "same-match"}]
    with pytest.raises(ValueError, match="distinct match groups"):
        split_video_groups(videos)
    result = split_video_groups(videos, allow_development_only=True)
    assert result == {"train": ["same-match"], "validation": [], "test": []}


def test_model_configuration_has_full_long_context() -> None:
    config = TemporalModelConfig(input_dim=384)
    assert config.receptive_field == 253
    assert TemporalModelConfig.from_dict({"input_dim": 32, "local_dilations": [1]}).local_dilations == (1,)
    with pytest.raises(ValueError):
        TemporalModelConfig(input_dim=0)


def test_spatial_motion_model_recognizes_a_translated_local_pattern_and_preserves_halo():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from snooker_ai.dl.temporal import TemporalHighlightNet, predict_probabilities
    torch.manual_seed(21)
    model = TemporalHighlightNet(TemporalModelConfig(input_dim=676, hidden_dim=8,
                                                    flow_grid=(8, 14), dropout=0)).eval()
    first, second = np.zeros((80, 676), np.float32), np.zeros((80, 676), np.float32)
    first_flow = first[:, 4:].reshape(80, 6, 8, 14)
    second_flow = second[:, 4:].reshape(80, 6, 8, 14)
    first_flow[20:35, 4:, 3, 4] = 3
    second_flow[20:35, 4:, 3, 8] = 3
    a = predict_probabilities(model, first, chunk_frames=80)
    b = predict_probabilities(model, second, chunk_frames=13)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
    loss = model(torch.from_numpy(first)[None]).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_neural_ensemble_checkpoint_preserves_its_members_and_chunk_predictions(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from snooker_ai.dl.temporal import (TemporalHighlightNet, TemporalHighlightEnsemble,
        checkpoint_payload, load_temporal_checkpoint, predict_probabilities)
    torch.manual_seed(8)
    members = [TemporalHighlightNet(TemporalModelConfig(input_dim=4, hidden_dim=8, dropout=0))
               for _ in range(2)]
    model = TemporalHighlightEnsemble(members, [.25, .75]).eval()
    x = np.random.default_rng(8).normal(size=(60, 4)).astype(np.float32)
    expected = sum(w*predict_probabilities(member.eval(), x) for w, member in zip([.25,.75], members))
    np.testing.assert_allclose(predict_probabilities(model, x, chunk_frames=11), expected, atol=2e-6)
    path = tmp_path/'ensemble.pt'
    torch.save(checkpoint_payload(model, feature_spec={'dimension':4}, dataset_fingerprint='test',
                                  training={}, calibration={}), path)
    restored, _ = load_temporal_checkpoint(path, expected_feature_spec={'dimension':4})
    np.testing.assert_allclose(predict_probabilities(restored, x), expected, atol=2e-6)
    with pytest.raises(ValueError):
        TemporalHighlightEnsemble(members, [1, -1])


def test_event_probability_nms_uses_seconds_and_resolves_plateau_once() -> None:
    timestamps = np.array([0, .1, .2, .3, .8, 1.0])
    probability = np.array([.1, .8, .8, .2, .7, .1])
    assert event_peaks(timestamps, probability, .6, min_gap_seconds=.5) == [1, 4]
    assert event_peaks(timestamps, probability, .9) == []


def test_calibration_event_peaks_share_runtime_uncertainty_suppression() -> None:
    from snooker_ai.dl.selection import predict_event_times, select_highlights

    times = np.arange(0, 8, .125)
    outputs = np.zeros((len(times), len(HEAD_NAMES)), dtype=np.float32)
    # The close peaks share a broad support interval; suppress by uncertainty
    # overlap even though their centre distance exceeds ordinary 1-second NMS.
    outputs[15:37, 0] = .65
    outputs[18, 0], outputs[30, 0] = .95, .9
    outputs[48:52, 0] = .8
    settings = TrainingConfig(max_event_uncertainty_seconds=6).event_candidate_settings
    times_shared = predict_event_times(times, outputs, .6, duration=8, settings=settings)
    candidates, _, _ = select_highlights(times, outputs, 8, settings, {"event": .6})
    assert times_shared == [candidate.timestamp for candidate in candidates]
    indexes = event_peaks(times, outputs[:, 0], .6, max_uncertainty_seconds=6)
    assert times_shared == [float(times[index]) for index in indexes]
    assert times_shared == [float(times[18]), float(times[49])]
    assert times_shared[-1] == float(times[49])  # Exact plateau centre.


def test_halo_inference_matches_full_sequence_at_chunk_boundaries() -> None:
    torch = pytest.importorskip("torch")
    from snooker_ai.dl.temporal import TemporalHighlightNet, predict_probabilities

    torch.manual_seed(1)
    torch.set_num_threads(1)
    model = TemporalHighlightNet(TemporalModelConfig(input_dim=7, hidden_dim=8, dropout=0))
    features = np.random.default_rng(4).normal(size=(347, 7)).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        full = torch.sigmoid(model(torch.from_numpy(features)[None])[0]).numpy()
    chunked = predict_probabilities(model, features, chunk_frames=31)
    np.testing.assert_allclose(chunked, full, atol=2e-6, rtol=2e-6)


def test_contact_interval_teaches_one_event_and_ignores_unlabelled_heads() -> None:
    torch = pytest.importorskip("torch")
    from snooker_ai.dl.training import masked_multitask_loss

    logits = torch.zeros((1, 8, len(HEAD_NAMES)), requires_grad=True)
    targets = torch.zeros_like(logits)
    weights = torch.zeros_like(logits)
    loss, metrics = masked_multitask_loss(logits, targets, weights, [[np.arange(2, 6)]], [[1.0]])
    loss.backward()
    assert metrics["interval_event"] > 0
    assert torch.count_nonzero(logits.grad[:, :, 1:]) == 0
    assert torch.count_nonzero(logits.grad[:, :2, :]) == 0
    # A broad interval is not four positive event labels.
    assert int((logits.grad[0, 2:6, 0] < 0).sum()) == 1


def test_rare_timing_preferences_and_hard_negatives_do_not_disappear_in_easy_frames():
    torch = pytest.importorskip('torch')
    from snooker_ai.dl.training import masked_multitask_loss
    logits = torch.full((1, 1000, len(HEAD_NAMES)), -10., requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = 0  # Weak timing preference.
        logits[0, 1, 0] = 3  # A confident false contact in reviewed footage.
    targets, weights = torch.zeros_like(logits), torch.zeros_like(logits)
    targets[0, 0, 0] = 1
    weights[..., 0] = 1
    weights[0, 0, 0] = .35
    loss, metrics = masked_multitask_loss(logits, targets, weights, [[]], [[]])
    assert metrics['weak_event_timing'] == pytest.approx(.35*np.log(2))
    loss.backward()
    assert logits.grad[0, 0, 0] < -.17
    assert logits.grad[0, 1, 0] > .01
    assert abs(float(logits.grad[0, 2, 0])) < 1e-7


def test_context_resets_between_disjoint_extracted_source_windows() -> None:
    pytest.importorskip("torch")
    from snooker_ai.dl.temporal import TemporalHighlightNet, predict_probabilities

    model = TemporalHighlightNet(TemporalModelConfig(input_dim=4, hidden_dim=8, dropout=0))
    features = np.random.default_rng(3).normal(size=(64, 4)).astype(np.float32)
    timestamps = np.concatenate((np.arange(32) / 8, 100 + np.arange(32) / 8))
    segmented = predict_probabilities(model, features, timestamps=timestamps)
    separate = np.concatenate((predict_probabilities(model, features[:32]),
                               predict_probabilities(model, features[32:])))
    np.testing.assert_allclose(segmented, separate, atol=2e-6)


def test_supervision_masks_and_provenance_weights_affect_loss() -> None:
    torch = pytest.importorskip("torch")
    from snooker_ai.dl.training import masked_multitask_loss

    logits = torch.zeros((1, 3, len(HEAD_NAMES)), requires_grad=True)
    labels = torch.zeros_like(logits)
    weights = torch.zeros_like(logits)
    labels[0, 1, 1] = 1
    weights[0, 0, 1] = 1
    weights[0, 1, 1] = .25
    loss, _ = masked_multitask_loss(logits, labels, weights, [[]], [[]])
    loss.backward()
    assert logits.grad[0, 0, 1] > 0
    assert logits.grad[0, 1, 1] < 0
    assert abs(float(logits.grad[0, 1, 1] / logits.grad[0, 0, 1])) == pytest.approx(.25)
    assert torch.count_nonzero(logits.grad[0, 2]) == 0


def test_all_weak_labels_retain_their_absolute_lower_strength() -> None:
    torch = pytest.importorskip("torch")
    from snooker_ai.dl.training import masked_multitask_loss

    logits = torch.zeros((1, 3, len(HEAD_NAMES)))
    labels = torch.zeros_like(logits)
    weights = torch.zeros_like(logits)
    weights[:, :, 1] = 1
    strong, _ = masked_multitask_loss(logits, labels, weights, [[]], [[]])
    weak, _ = masked_multitask_loss(logits, labels, weights * .35, [[]], [[]])
    assert float(weak / strong) == pytest.approx(.35)


def test_checkpoint_round_trip_rejects_feature_semantic_mismatch(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from snooker_ai.dl.temporal import (
        TemporalHighlightNet,
        checkpoint_payload,
        load_temporal_checkpoint,
        predict_probabilities,
    )

    model = TemporalHighlightNet(TemporalModelConfig(input_dim=4, hidden_dim=8))
    model.set_feature_normalization(np.array([1, 2, 3, 4]), np.ones(4) * 2)
    spec = {"rgb_model": "test-rgb", "flow_model": "test-flow", "sample_fps": 8}
    path = tmp_path / "model.pt"
    torch.save(checkpoint_payload(model, feature_spec=spec, dataset_fingerprint="test",
                                  training={"split_groups": {"train": ["a"], "validation": ["b"]}},
                                  calibration={"thresholds": {"event": .4}}), path)
    restored, metadata = load_temporal_checkpoint(path, expected_feature_spec=spec)
    features = np.random.default_rng(1).normal(size=(100, 4)).astype(np.float32)
    np.testing.assert_allclose(predict_probabilities(model, features), predict_probabilities(restored, features))
    assert metadata["dataset_fingerprint"] == "test"
    with pytest.raises(ValueError, match="specification differs"):
        load_temporal_checkpoint(path, expected_feature_spec={"rgb_model": "different"})
    invalid = checkpoint_payload(model, feature_spec=spec, dataset_fingerprint="test",
                                 training={}, calibration={})
    invalid["model_state"]["feature_std"].zero_()
    torch.save(invalid, path)
    with pytest.raises(ValueError, match="normalization"):
        load_temporal_checkpoint(path)


def test_train_cli_saves_real_weights_and_frozen_test_threshold(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from snooker_ai.dl.temporal import TemporalHighlightNet, load_temporal_checkpoint
    from snooker_ai.dl.training import train_temporal

    timestamps = np.arange(0, 8, .125)
    feature_spec = {"rgb_model": "synthetic-fixture", "sample_fps": 8, "input_dim": 4}
    videos = []
    for group in ("train", "validation", "test"):
        feature_path = tmp_path / f"{group}.npz"
        np.savez(feature_path, timestamps=timestamps,
                 features=np.random.default_rng(len(group)).normal(size=(64, 4)).astype(np.float32),
                 feature_spec=json.dumps(feature_spec))
        videos.append({
            "id": group, "group_id": group, "source_path": f"{group}.mp4",
            "features_path": feature_path.name, "duration": 8,
            "reviewed_intervals": [{"start": 0, "end": 8}],
            "contacts": [{"lower": 3, "upper": 3.25, "description": "fixture contact",
                          "provenance": "manual_source_review"}],
            "clips": [],
            "labels": [{"head": "keep", "start": 2.5, "end": 4.5, "value": 1,
                        "weight": 1, "provenance": "manual_source_review"}],
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "heads": list(HEAD_NAMES), "videos": videos}))
    checkpoint = tmp_path / "model.pt"
    report = train_temporal(
        manifest, checkpoint,
        config=TrainingConfig(epochs=2, patience=1, batch_size=1, window_frames=64,
                              stride_frames=32, hidden_dim=8, dropout=0),
        validation_groups=["validation"], test_groups=["test"],
    )
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".pt.report.json").is_file()
    assert report["independent_test_available"]
    assert report["independent_test"]["thresholds_frozen_before_test"]
    assert report["calibration"]["scope"] == "group_disjoint_validation"
    assert report["deployment_quality_verified"] is False
    assert report["best_epoch"] in (1, 2)
    restored, _ = load_temporal_checkpoint(checkpoint)
    with np.load(tmp_path / "train.npz") as archive:
        np.testing.assert_allclose(restored.feature_mean.numpy(), archive["features"].mean(axis=0), atol=1e-6)
    torch.manual_seed(2026)
    initial = TemporalHighlightNet(TemporalModelConfig(input_dim=4, hidden_dim=8, dropout=0))
    assert not torch.allclose(restored.fusion[-1].weight, initial.fusion[-1].weight)
