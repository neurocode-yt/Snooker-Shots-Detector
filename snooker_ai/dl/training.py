"""Train and validate the neural DL Algo temporal selector on reviewed footage.

Run ``python -m snooker_ai.dl.training MANIFEST --output MODEL.pt`` in the
optional DL environment. Feature extraction is a separate, reusable stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snooker_ai.dl.temporal import (
    HEAD_NAMES,
    TemporalHighlightNet,
    TemporalModelConfig,
    checkpoint_payload,
    contiguous_feature_spans,
    predict_probabilities,
    require_torch,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    patience: int = 8
    batch_size: int = 4
    window_frames: int = 512
    stride_frames: int = 256
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    hidden_dim: int = 96
    dropout: float = 0.15
    validation_fraction: float = 0.25
    seed: int = 2026
    min_event_gap_seconds: float = 1.0
    event_uncertainty_ratio: float = 0.5
    max_event_uncertainty_seconds: float = 2.0
    event_tolerance_seconds: float = 0.25
    target_event_recall: float = 0.98

    def __post_init__(self) -> None:
        continuous = (
            self.learning_rate, self.weight_decay, self.dropout,
            self.validation_fraction, self.min_event_gap_seconds,
            self.event_uncertainty_ratio, self.max_event_uncertainty_seconds,
            self.event_tolerance_seconds, self.target_event_recall,
        )
        if not np.isfinite(continuous).all():
            raise ValueError("Training and event decoding settings must be finite.")
        if min(self.epochs, self.patience, self.batch_size, self.window_frames) < 1:
            raise ValueError("Training counts must be positive.")
        if self.stride_frames < 1 or self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must be positive and no larger than window_frames.")
        if not 0 < self.validation_fraction < 1 or not 0 < self.target_event_recall <= 1:
            raise ValueError("Validation fraction and requested event recall are out of range.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if not 0 <= self.event_uncertainty_ratio <= 1 or self.max_event_uncertainty_seconds < 0:
            raise ValueError("Event uncertainty settings are out of range.")
        if self.min_event_gap_seconds < 0 or self.event_tolerance_seconds < 0:
            raise ValueError("Event NMS spacing and evaluation tolerance must be nonnegative.")

    @property
    def event_candidate_settings(self) -> dict[str, float]:
        return {
            "nms_seconds": self.min_event_gap_seconds,
            "uncertainty_ratio": self.event_uncertainty_ratio,
            "max_uncertainty_seconds": self.max_event_uncertainty_seconds,
        }


@dataclass
class _PreparedVideo:
    entry: dict[str, Any]
    timestamps: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    contact_bags: tuple[np.ndarray, ...]
    contact_weights: np.ndarray


def split_video_groups(
    videos: list[dict[str, Any]],
    *,
    seed: int = 2026,
    validation_fraction: float = 0.25,
    validation_groups: list[str] | None = None,
    test_groups: list[str] | None = None,
    allow_development_only: bool = False,
) -> dict[str, list[str]]:
    """Split whole matches, never windows or related exports from one match."""
    groups = sorted({str(video["group_id"]) for video in videos})
    tests = sorted(set(test_groups or []))
    validations = sorted(set(validation_groups or []))
    if set(tests + validations) - set(groups):
        raise ValueError("A requested validation/test group is absent from the manifest.")
    if set(tests) & set(validations):
        raise ValueError("Validation and test groups must be disjoint.")
    available = [group for group in groups if group not in tests]
    if validation_groups is None:
        if len(available) < 2:
            if not allow_development_only:
                raise ValueError(
                    "At least two distinct match groups are required for validation. "
                    "Use --development-only explicitly for an unvalidated development model."
                )
        else:
            rng = random.Random(seed)
            shuffled = list(available)
            rng.shuffle(shuffled)
            count = min(len(available) - 1, max(1, round(len(available) * validation_fraction)))
            validations = sorted(shuffled[:count])
    elif not validations and not allow_development_only:
        raise ValueError(
            "The frozen manifest has no validation match. "
            "Use --development-only explicitly for an unvalidated development model."
        )
    train = sorted(set(available) - set(validations))
    if not train:
        raise ValueError("The group split leaves no training match.")
    return {"train": train, "validation": validations, "test": tests}


def masked_multitask_loss(
    logits: Any,
    targets: Any,
    weights: Any,
    contact_bags: list[list[Any]],
    contact_weights: list[list[float]],
    *,
    positive_weights: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """Weighted focal losses plus interval-censored multiple-instance events.

    A contact bracket is one uncertain event, not a sequence of positive labels.
    Its maximum logit must be positive; a sparse-count penalty discourages
    widening a single strike across the entire uncertain bracket. Samples in
    these bags must be masked out of framewise negative event supervision.
    """
    torch = require_torch()
    functional = torch.nn.functional
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("Logits, targets and supervision weights must have the same shape.")
    if logits.ndim != 3 or logits.shape[-1] != len(HEAD_NAMES):
        raise ValueError("Expected a [batch, frames, five heads] training tensor.")
    valid = weights > 0
    clean_targets = torch.where(valid, targets, torch.zeros_like(targets))
    if positive_weights is None:
        positive_weights = torch.ones(len(HEAD_NAMES), device=logits.device)
    bce = functional.binary_cross_entropy_with_logits(logits, clean_targets, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = torch.where(clean_targets > 0.5, probability, 1 - probability)
    class_weight = torch.where(clean_targets > 0.5, positive_weights, 1.0)
    focal = bce * (1 - correct_probability).pow(2) * class_weight * weights
    # Preserve absolute provenance strength: an all-weak head must contribute
    # less than a human-labelled head, rather than cancelling its .35 weight.
    denominator = valid.sum(dim=(0, 1)).clamp_min(1)
    per_head = focal.sum(dim=(0, 1)) / denominator
    active_heads = (weights.sum(dim=(0, 1)) > 0).float()
    auxiliary = (per_head[1:] * active_heads[1:]).sum() / active_heads[1:].sum().clamp_min(1)
    # Event-positive bags and event-negative frames keep equal scale even when
    # a validation match has contact labels but no editorial boundary labels.
    frame_loss = per_head[0] * active_heads[0] + auxiliary
    bag_losses, bag_masses = [], []
    for batch_index, bags in enumerate(contact_bags):
        if len(bags) != len(contact_weights[batch_index]):
            raise ValueError("Each contact bag needs a provenance weight.")
        for bag, weight in zip(bags, contact_weights[batch_index]):
            if weight <= 0:
                continue
            indexes = torch.as_tensor(bag, dtype=torch.long, device=logits.device)
            if indexes.numel() == 0:
                raise ValueError("An interval-censored contact has no feature samples.")
            values = logits[batch_index, indexes, 0]
            positive = functional.softplus(-values.max(dim=0).values)
            # One stroke per interval; keep the penalty weaker than contact recall.
            sparse = 0.05 * (torch.sigmoid(values).sum() - 1).abs()
            bag_losses.append((positive + sparse) * weight)
            bag_masses.append(weight)
    event_loss = logits.sum() * 0
    if bag_losses:
        event_loss = torch.stack(bag_losses).sum() / len(bag_masses)
    total = frame_loss + event_loss
    details = {head: float(per_head[index].detach()) for index, head in enumerate(HEAD_NAMES)}
    details["interval_event"] = float(event_loss.detach())
    details["total"] = float(total.detach())
    return total, details


def event_peaks(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    min_gap_seconds: float = 1.0,
    uncertainty_ratio: float = .5,
    max_uncertainty_seconds: float = 2.0,
) -> list[int]:
    """Compatibility index wrapper around the runtime's event candidate decoder."""
    from snooker_ai.dl.selection import predict_event_times

    timestamps = np.asarray(timestamps)
    probabilities = np.asarray(probabilities)
    if timestamps.shape != probabilities.shape or timestamps.ndim != 1:
        raise ValueError("Event timestamps/probabilities must be matching vectors.")
    if not len(timestamps):
        return []
    outputs = np.zeros((len(timestamps), len(HEAD_NAMES)), dtype=np.float64)
    outputs[:, 0] = probabilities
    predicted = predict_event_times(
        timestamps, outputs, threshold,
        settings={"nms_seconds": min_gap_seconds, "uncertainty_ratio": uncertainty_ratio,
                  "max_uncertainty_seconds": max_uncertainty_seconds},
    )
    return [int(np.searchsorted(timestamps, timestamp)) for timestamp in predicted]


def event_metrics(
    videos: list[_PreparedVideo],
    predictions: list[np.ndarray],
    threshold: float,
    config: TrainingConfig,
) -> dict[str, Any]:
    from snooker_ai.dl.dataset import evaluate_contacts
    from snooker_ai.dl.selection import predict_event_times

    true_positive = false_positive = false_negative = ignored = 0
    for video, probabilities in zip(videos, predictions):
        predicted = predict_event_times(
            video.timestamps,
            probabilities,
            threshold,
            duration=float(video.entry["duration"]),
            settings=config.event_candidate_settings,
        )
        metrics = evaluate_contacts(
            predicted,
            video.entry.get("contacts", []),
            video.entry.get("reviewed_intervals", []),
            tolerance_seconds=config.event_tolerance_seconds,
        )
        true_positive += metrics["true_positive"]
        false_positive += metrics["false_positive"]
        false_negative += metrics["false_negative"]
        ignored += metrics["unreviewed_predictions_ignored"]
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f2 = 5 * precision * recall / max(1e-12, 4 * precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "unlabelled_predictions_ignored": ignored,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "threshold": float(threshold),
        "scope": "raw_event_candidates_before_replay_handling_veto_and_clip_boundaries",
    }


def _calibration(
    videos: list[_PreparedVideo], predictions: list[np.ndarray], config: TrainingConfig
) -> dict[str, Any]:
    contact_count = sum(len(video.entry.get("contacts", [])) for video in videos)
    if contact_count:
        event_options = [event_metrics(videos, predictions, float(t), config) for t in np.arange(.05, 1, .05)]
        qualifying = [option for option in event_options if option["recall"] >= config.target_event_recall]
        if qualifying:
            event = max(qualifying, key=lambda value: (value["precision"], value["threshold"]))
        else:
            event = max(event_options, key=lambda value: (value["f2"], value["recall"]))
    else:
        event = event_metrics(videos, predictions, .5, config)
        event["status"] = "no_positive_event_validation"
    thresholds = {"event": event["threshold"]}
    frame_metrics: dict[str, Any] = {}
    for head_index, head in enumerate(HEAD_NAMES[1:], start=1):
        all_labels, all_probabilities, all_weights = [], [], []
        for video, output in zip(videos, predictions):
            selected = video.mask[:, head_index] > 0
            all_labels.extend(video.targets[selected, head_index].tolist())
            all_probabilities.extend(output[selected, head_index].tolist())
            all_weights.extend(video.mask[selected, head_index].tolist())
        labels = np.asarray(all_labels)
        probability = np.asarray(all_probabilities)
        weights = np.asarray(all_weights)
        if len(labels) == 0:
            thresholds[head] = .5
            frame_metrics[head] = {"labelled_samples": 0, "status": "unvalidated"}
            continue
        options = []
        for threshold in np.arange(.05, 1, .05):
            positive = probability >= threshold
            tp = float(weights[positive & (labels > .5)].sum())
            fp = float(weights[positive & (labels <= .5)].sum())
            fn = float(weights[~positive & (labels > .5)].sum())
            options.append((2 * tp / max(1e-12, 2 * tp + fp + fn), float(threshold)))
        f1, threshold = max(options, key=lambda value: (value[0], -abs(value[1] - .5)))
        thresholds[head] = threshold
        ece = 0.0
        for lower in np.arange(0, 1, .1):
            in_bin = (probability >= lower) & (probability < lower + .1 + 1e-9)
            mass = float(weights[in_bin].sum())
            if mass:
                confidence = float(np.average(probability[in_bin], weights=weights[in_bin]))
                accuracy = float(np.average(labels[in_bin], weights=weights[in_bin]))
                ece += mass / float(weights.sum()) * abs(confidence - accuracy)
        frame_metrics[head] = {
            "labelled_samples": len(labels),
            "positive_samples": int((labels > .5).sum()),
            "weighted_f1": f1,
            "brier": float(np.average((probability - labels) ** 2, weights=weights)),
            "expected_calibration_error": ece,
            "threshold": threshold,
            "status": "positive_and_negative_labels" if 0 < (labels > .5).sum() < len(labels)
                      else "positive_only_labels" if (labels > .5).any() else "negative_only_labels",
        }
    return {
        "thresholds": thresholds,
        "temperatures": [1.0] * len(HEAD_NAMES),
        "event": event,
        "raw_event_candidates": event,
        "event_candidate_settings": config.event_candidate_settings,
        "decoded_highlights": {
            "status": "not_evaluated_in_temporal_training",
            "reason": "Replay/handling vetoes and editorial boundaries require a separate decoded-clip benchmark.",
        },
        "frame_heads": frame_metrics,
        "event_tolerance_seconds": config.event_tolerance_seconds,
        "target_event_recall": config.target_event_recall,
        "target_event_recall_achieved": bool(contact_count and event["recall"] >= config.target_event_recall),
    }


def _prepare_videos(manifest_path: Path) -> tuple[list[_PreparedVideo], dict[str, Any], str]:
    from snooker_ai.dl.dataset import build_targets, load_feature_video, load_manifest

    manifest = load_manifest(manifest_path)
    path_root = (manifest_path.parent / manifest.get("paths_relative_to", ".")).resolve()
    entries = manifest["videos"]
    prepared: list[_PreparedVideo] = []
    common_spec: dict[str, Any] | None = None
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    for entry in entries:
        feature_path = Path(entry["features_path"])
        if not feature_path.is_absolute():
            feature_path = path_root / feature_path
        timestamps, features, spec = load_feature_video(feature_path)
        if common_spec is None:
            common_spec = spec
        elif common_spec != spec or features.shape[1] != prepared[0].features.shape[1]:
            raise ValueError("All training videos must use the same learned feature specification.")
        labels = build_targets(timestamps, entry, sample_fps=spec.get('sample_fps'))
        prepared.append(
            _PreparedVideo(
                entry=entry,
                timestamps=timestamps,
                features=features,
                targets=labels.targets,
                mask=labels.mask,
                contact_bags=labels.contact_bags,
                contact_weights=np.asarray(labels.contact_weights, dtype=np.float32),
            )
        )
        with feature_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                fingerprint.update(block)
    if not prepared:
        raise ValueError("The training manifest has no videos.")
    return prepared, common_spec or {}, fingerprint.hexdigest()


def _windows(videos: list[_PreparedVideo], config: TrainingConfig) -> list[tuple[int, int, int]]:
    result = []
    for index, video in enumerate(videos):
        for span_start, span_stop in contiguous_feature_spans(video.timestamps):
            final_start = max(span_start, span_stop - config.window_frames)
            starts = sorted(set(range(span_start, final_start + 1, config.stride_frames)) | {final_start})
            for start in starts:
                stop = min(span_stop, start + config.window_frames)
                if video.mask[start:stop].sum() or any(
                    bag[0] >= start and bag[-1] < stop for bag in video.contact_bags
                ):
                    result.append((index, start, stop))
    if not result:
        raise ValueError("The split contains no supervised temporal windows.")
    return result


def _batch(
    videos: list[_PreparedVideo], windows: list[tuple[int, int, int]], device: str
) -> tuple[Any, Any, Any, list[list[np.ndarray]], list[list[float]]]:
    torch = require_torch()
    length = max(stop - start for _, start, stop in windows)
    inputs = np.zeros((len(windows), length, videos[0].features.shape[1]), dtype=np.float32)
    targets = np.zeros((len(windows), length, len(HEAD_NAMES)), dtype=np.float32)
    masks = np.zeros_like(targets)
    bags, bag_weights = [], []
    for row, (index, start, stop) in enumerate(windows):
        video = videos[index]
        count = stop - start
        inputs[row, :count] = video.features[start:stop]
        # Repeating the last observed embedding avoids introducing a synthetic cut.
        inputs[row, count:] = video.features[stop - 1]
        targets[row, :count] = video.targets[start:stop]
        masks[row, :count] = video.mask[start:stop]
        selected_bags, selected_weights = [], []
        for bag, weight in zip(video.contact_bags, video.contact_weights):
            if bag[0] >= start and bag[-1] < stop:
                selected_bags.append(bag - start)
                selected_weights.append(float(weight))
        bags.append(selected_bags)
        bag_weights.append(selected_weights)
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(masks).to(device),
        bags,
        bag_weights,
    )


def _positive_weights(videos: list[_PreparedVideo], device: str) -> Any:
    torch = require_torch()
    positive = sum((video.mask * (video.targets > .5)).sum(axis=0) for video in videos)
    negative = sum((video.mask * (video.targets <= .5)).sum(axis=0) for video in videos)
    weights = np.sqrt(negative / np.maximum(positive, 1)).clip(1, 20).astype(np.float32)
    weights[0] = 1  # Event positives are supervised once per interval, separately.
    return torch.from_numpy(weights).to(device)


def train_temporal(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: TrainingConfig | None = None,
    device: str = "cpu",
    validation_groups: list[str] | None = None,
    test_groups: list[str] | None = None,
    allow_development_only: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    config = config or TrainingConfig()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    videos, feature_spec, fingerprint = _prepare_videos(Path(manifest_path).resolve())
    # Prepared source manifests freeze splits before any model selection.
    # Ad-hoc feature manifests without explicit splits use the seeded group split.
    if any("split" in video.entry for video in videos):
        mapping = {"train": "train", "dev": "validation", "holdout": "test"}
        declared = {
            split: sorted({str(video.entry["group_id"]) for video in videos
                           if mapping[video.entry.get("split", "train")] == split})
            for split in ("train", "validation", "test")
        }
        if validation_groups is not None and set(validation_groups) != set(declared["validation"]):
            raise ValueError("Requested validation groups contradict the frozen manifest split.")
        if test_groups is not None and set(test_groups) != set(declared["test"]):
            raise ValueError("Requested test groups contradict the frozen manifest split.")
        validation_groups, test_groups = declared["validation"], declared["test"]
    splits = split_video_groups(
        [video.entry for video in videos],
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        validation_groups=validation_groups,
        test_groups=test_groups,
        allow_development_only=allow_development_only,
    )
    by_split = {
        split: [video for video in videos if str(video.entry["group_id"]) in groups]
        for split, groups in splits.items()
    }
    train_videos = by_split["train"]
    validation_videos = by_split["validation"]
    stop_videos = validation_videos or train_videos
    model = TemporalHighlightNet(
        TemporalModelConfig(
            input_dim=train_videos[0].features.shape[1],
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    # The holdout never contributes even unsupervised normalization statistics.
    total = sum(len(video.features) for video in train_videos)
    mean = sum(video.features.sum(axis=0, dtype=np.float64) for video in train_videos) / total
    second = sum((video.features.astype(np.float64) ** 2).sum(axis=0) for video in train_videos)
    std = np.sqrt(np.maximum(second / total - mean ** 2, 0)).clip(.001)
    model.set_feature_normalization(mean, std)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_windows, stop_windows = _windows(train_videos, config), _windows(stop_videos, config)
    positive_weights = _positive_weights(train_videos, device)
    generator = random.Random(config.seed)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        model.train()
        shuffled = list(train_windows)
        generator.shuffle(shuffled)
        lr_scale = min(1, (epoch + 1) / 3) * .5 * (1 + math.cos(math.pi * epoch / config.epochs))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_scale
        training_loss = 0.0
        for start in range(0, len(shuffled), config.batch_size):
            batch_windows = shuffled[start : start + config.batch_size]
            features, targets, masks, bags, bag_weights = _batch(train_videos, batch_windows, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = masked_multitask_loss(
                model(features), targets, masks, bags, bag_weights,
                positive_weights=positive_weights,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Temporal training produced a non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch_windows)
        model.eval()
        selection_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(stop_windows), config.batch_size):
                batch_windows = stop_windows[start : start + config.batch_size]
                features, targets, masks, bags, bag_weights = _batch(stop_videos, batch_windows, device)
                loss, _ = masked_multitask_loss(
                    model(features), targets, masks, bags, bag_weights,
                    positive_weights=positive_weights,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Temporal validation produced a non-finite loss.")
                selection_loss += float(loss) * len(batch_windows)
        selection_loss /= len(stop_windows)
        entry = {
            "epoch": epoch + 1,
            "training_loss": training_loss / len(train_windows),
            "selection_loss": selection_loss,
            "selection_scope": "group_disjoint_validation" if validation_videos else "development_training",
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if selection_loss < best_loss - .0001:
            best_loss, best_epoch = selection_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite temporal training checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    calibration_predictions = [
        predict_probabilities(model, video.features, device=device, timestamps=video.timestamps)
        for video in stop_videos
    ]
    calibration = _calibration(stop_videos, calibration_predictions, config)
    calibration["scope"] = "group_disjoint_validation" if validation_videos else "development_training_only"
    calibration["used_for_model_selection"] = True
    report: dict[str, Any] = {
        "configuration": asdict(config),
        "seed": config.seed,
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "deterministic_algorithms_warn_only": True,
        },
        "split_groups": splits,
        "split_video_ids": {split: [video.entry["id"] for video in subset] for split, subset in by_split.items()},
        "dataset_fingerprint": fingerprint,
        "feature_spec": feature_spec,
        "best_epoch": best_epoch,
        "selection_loss": best_loss,
        "history": history,
        "calibration": calibration,
        "independent_test_available": bool(by_split["test"]),
        "deployment_quality_verified": False,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "supervision": {
            split: {
                "contacts": sum(len(video.contact_bags) for video in subset),
                "heads": {
                    head: {
                        "labelled_samples": sum(int((video.mask[:, index] > 0).sum()) for video in subset),
                        "positive_samples": sum(int(((video.targets[:, index] > .5)
                                                     & (video.mask[:, index] > 0)).sum()) for video in subset),
                        "weighted_samples": sum(float(video.mask[:, index].sum()) for video in subset),
                    }
                    for index, head in enumerate(HEAD_NAMES)
                },
            }
            for split, subset in by_split.items()
        },
    }
    if by_split["test"]:
        test_predictions = [
            predict_probabilities(model, video.features, device=device, timestamps=video.timestamps)
            for video in by_split["test"]
        ]
        report["independent_test"] = event_metrics(
            by_split["test"], test_predictions, calibration["thresholds"]["event"], config
        )
        report["independent_test"]["thresholds_frozen_before_test"] = True
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model,
        feature_spec=feature_spec,
        dataset_fingerprint=fingerprint,
        training={key: value for key, value in report.items() if key not in ("feature_spec", "calibration")},
        calibration=calibration,
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    output_path.with_suffix(output_path.suffix + ".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--event-nms-seconds", type=float, default=1.0)
    parser.add_argument("--event-uncertainty-ratio", type=float, default=.5)
    parser.add_argument("--max-event-uncertainty-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-group", action="append")
    parser.add_argument("--test-group", action="append")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        window_frames=args.window_frames, stride_frames=max(1, args.window_frames // 2),
        hidden_dim=args.hidden_dim, learning_rate=args.learning_rate, seed=args.seed,
        min_event_gap_seconds=args.event_nms_seconds,
        event_uncertainty_ratio=args.event_uncertainty_ratio,
        max_event_uncertainty_seconds=args.max_event_uncertainty_seconds,
    )
    report = train_temporal(
        args.manifest, args.output, config=config, device=args.device,
        validation_groups=args.validation_group, test_groups=args.test_group,
        allow_development_only=args.development_only,
    )
    print(json.dumps({"checkpoint": str(args.output), "calibration": report["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
