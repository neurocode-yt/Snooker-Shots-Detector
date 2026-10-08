"""Reviewed, interval-censored labels for the independent neural editor.

An unannotated frame is unknown, not a negative. Contacts use bags of plausible
feature samples rather than invented exact impact timestamps. Clip preferences
can be lower-weight labels from classic exports that were visually reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

HEADS = ("event", "keep", "end", "replay", "handling")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class TargetArrays:
    targets: np.ndarray
    mask: np.ndarray
    contact_bags: tuple[np.ndarray, ...]
    contact_weights: np.ndarray


def _bounds(row: Mapping[str, Any], low: str = "start", high: str = "end") -> tuple[float, float]:
    left, right = float(row[low]), float(row[high])
    if not np.isfinite([left, right]).all() or left < 0 or right < left:
        raise ValueError(f"Invalid interval {left}..{right}")
    return left, right


def _weight(row: Mapping[str, Any], default: float = 1) -> float:
    try:
        strength = float(row.get("weight", default))
    except (TypeError, ValueError) as exc:
        raise ValueError("Annotation weights must be finite numbers in [0,1]") from exc
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Annotation weights must be finite numbers in [0,1]")
    return strength


def _editorial_bounds(clip: Mapping[str, Any], name: str) -> tuple[float, float] | None:
    lower_key, upper_key = f"{name}_lower", f"{name}_upper"
    if lower_key not in clip and upper_key not in clip:
        return None
    if lower_key not in clip or upper_key not in clip:
        raise ValueError("Editorial uncertainty requires both lower and upper bounds")
    left, right = _bounds(clip, lower_key, upper_key)
    if not left <= float(clip[name]) <= right:
        raise ValueError("Preferred editorial boundary is outside its uncertainty interval")
    return left, right


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported DL annotation manifest version")
    if tuple(manifest.get("heads", [])) != HEADS:
        raise ValueError(f"Neural head order must be {HEADS}")
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for video in manifest.get("videos", []):
        if not video.get("id") or video["id"] in seen or not video.get("group_id"):
            raise ValueError("Every video needs a unique id and underlying match group_id")
        seen.add(video["id"])
        split = video.get("split", "train")
        if split not in ("train", "dev", "holdout"):
            raise ValueError(f"Unknown split: {split}")
        group = video["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Underlying match {group!r} leaks across dataset splits")
        groups[group] = split
        duration = float(video["duration"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration must be positive")
        reviewed = [_bounds(row) for row in video.get("reviewed_intervals", [])]
        for row in video.get("reviewed_intervals", []):
            _weight(row)
        for left, right in reviewed:
            if right > duration + 1e-6:
                raise ValueError("Reviewed interval exceeds video duration")
        for row in video.get("clip_reviewed_intervals", []):
            _, right = _bounds(row)
            _weight(row, 0.35)
            if right > duration + 1e-6:
                raise ValueError("Clip-review interval exceeds video duration")
        for contact in video.get("contacts", []):
            left, right = _bounds(contact, "lower", "upper")
            _weight(contact)
            if not any(a <= left and right <= b for a, b in reviewed):
                raise ValueError("Every contact must lie wholly in an exhaustively reviewed interval")
            if not contact.get("provenance"):
                raise ValueError("Contact annotations need provenance")
        for clip in video.get("clips", []):
            left, right = _bounds(clip)
            _weight(clip, 0.35)
            start_bounds, end_bounds = _editorial_bounds(clip, "start"), _editorial_bounds(clip, "end")
            if end_bounds and end_bounds[1] > duration + 1e-6:
                raise ValueError("Editorial uncertainty exceeds video duration")
            if start_bounds and end_bounds and start_bounds[1] >= end_bounds[0]:
                raise ValueError("Editorial start and end uncertainty overlap")
            if right > duration + 1e-6 or left == right:
                raise ValueError("Clip boundaries exceed duration or are empty")
            if not clip.get("provenance"):
                raise ValueError("Clip preference labels need explicit provenance")
        for label in video.get("labels", []):
            left, right = _bounds(label)
            _weight(label)
            if right > duration + 1e-6 or label["head"] not in HEADS:
                raise ValueError("Invalid head label interval")
            if not 0 <= float(label["value"]) <= 1 or not label.get("provenance"):
                raise ValueError("Label values and provenance are required")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def load_feature_video(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a frozen feature archive without pickle/object deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        features = np.asarray(archive["features"], dtype=np.float32)
        spec = json.loads(str(archive["feature_spec"].item()))
    if timestamps.ndim != 1 or features.ndim != 2 or len(timestamps) != len(features):
        raise ValueError("Feature archive must contain timestamps[T] and features[T,D]")
    if not len(timestamps) or not np.isfinite(timestamps).all() or not np.isfinite(features).all():
        raise ValueError("Feature archive is empty or contains non-finite values")
    if (np.diff(timestamps) <= 0).any() or timestamps[0] < 0:
        raise ValueError("Feature timestamps must be nonnegative and strictly increasing")
    if not isinstance(spec, dict):
        raise ValueError("feature_spec must encode a JSON object")
    if spec.get("dimension", features.shape[1]) != features.shape[1]:
        raise ValueError("Feature dimension disagrees with archive metadata")
    return timestamps, features, spec


def build_targets(
    timestamps: np.ndarray,
    video: Mapping[str, Any],
    *,
    end_radius_seconds: float = 0.5,
    sample_fps: float | None = None,
) -> TargetArrays:
    """Construct frame targets and per-head weights, leaving unknowns masked.

    Contact bags include samples in the uncertainty bracket or the nearest
    half-cadence cells when a short native bracket falls between feature samples.
    Bag samples have no pointwise event BCE label. A contact more than half a
    cadence from every sample is an invalid/incomplete extraction, not a miss
    silently discarded from training.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Target timestamps must be a finite nonempty vector")
    if (np.diff(times) <= 0).any():
        raise ValueError("Target timestamps must increase strictly")
    targets = np.zeros((len(times), len(HEADS)), dtype=np.float32)
    weights = np.zeros_like(targets)
    if sample_fps is not None:
        if not np.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError('Supervision sample FPS must be positive and finite')
        cadence = 1 / sample_fps
    else:
        # Missing minutes between isolated observations are not sampling cells.
        cadence = min(1., float(np.median(np.diff(times)))) if len(times) > 1 else 0.0
    support = max(1e-6, cadence / 2 + 1e-6)
    event_idx, keep_idx, end_idx, replay_idx, handling_idx = range(5)
    for interval in video.get("reviewed_intervals", []):
        left, right = _bounds(interval)
        weights[(times >= left) & (times <= right), event_idx] = _weight(interval)
    bags: list[np.ndarray] = []
    bag_weights: list[float] = []
    for contact in video.get("contacts", []):
        left, right = _bounds(contact, "lower", "upper")
        indexes = np.flatnonzero((times >= left) & (times <= right))
        if not len(indexes):
            distance = np.maximum(left - times, np.maximum(times - right, 0))
            indexes = np.flatnonzero(distance <= support)
        if not len(indexes):
            raise ValueError(f"No extracted feature sample supports contact {left}..{right}")
        weights[indexes, event_idx] = 0
        bags.append(indexes)
        bag_weights.append(_weight(contact))
    # Clip targets are supervised only in their explicitly reviewed coverage.
    for interval in video.get("clip_reviewed_intervals", []):
        left, right = _bounds(interval)
        selected = (times >= left) & (times <= right)
        strength = _weight(interval, 0.35)
        weights[selected, keep_idx] = strength
        weights[selected, end_idx] = strength
    for clip in video.get("clips", []):
        left, right = _bounds(clip)
        selected = (times >= left) & (times <= right)
        strength = _weight(clip, 0.35)
        targets[selected, keep_idx] = 1
        weights[selected, keep_idx] = strength
        # An end means an editorial clip endpoint, never a physical ball stop.
        end_bounds = _editorial_bounds(clip, "end")
        edge = ((times >= end_bounds[0]) & (times <= end_bounds[1])) if end_bounds else (
            np.abs(times - right) <= max(end_radius_seconds, support))
        targets[edge, end_idx] = 1
        weights[edge, end_idx] = strength
        # Source reviewers provide a permissible boundary range, not invented
        # frame-level keep truth on either side of one preferred timestamp.
        for boundary in ("start", "end"):
            bounds = _editorial_bounds(clip, boundary)
            if bounds:
                weights[(times >= bounds[0]) & (times <= bounds[1]), keep_idx] = 0
        if clip.get("confirmed_replay_free"):
            weights[selected, replay_idx] = strength
        if clip.get("confirmed_handling_free"):
            weights[selected, handling_idx] = strength
    # Explicit human/source-review labels win over weak editorial preferences.
    for label in video.get("labels", []):
        left, right = _bounds(label)
        selected = (times >= left) & (times <= right)
        head_idx = HEADS.index(label["head"])
        targets[selected, head_idx] = float(label["value"])
        weights[selected, head_idx] = _weight(label)
    return TargetArrays(targets, weights, tuple(bags), np.asarray(bag_weights, dtype=np.float32))


def split_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Preserve all cuts of a match in one explicit split; never split by frame."""
    validate_manifest(manifest)
    return {
        split: {
            **{key: value for key, value in manifest.items() if key != "videos"},
            "split": split,
            "videos": [video for video in manifest["videos"] if video.get("split", "train") == split],
        }
        for split in ("train", "dev", "holdout")
    }


def evaluate_contacts(
    predicted: Sequence[float],
    contacts: Sequence[Mapping[str, Any]],
    reviewed_intervals: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float = 0.5,
) -> dict[str, Any]:
    """Maximum-cardinality one-to-one matching with uncertainty-bound errors."""
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Contact tolerance must be finite and nonnegative")
    intervals = [_bounds(interval) for interval in reviewed_intervals]
    values = [float(value) for value in predicted]
    if not np.isfinite(values).all():
        raise ValueError("Predicted contact times must be finite")
    eligible = [(index, value) for index, value in enumerate(values)
                if any(left <= value <= right for left, right in intervals)]
    reference = [_bounds(contact, "lower", "upper") for contact in contacts]
    matches: list[dict[str, Any]] = []
    matched_predictions: set[int] = set()
    matched_references: set[int] = set()
    if eligible and reference:
        # More expensive than all possible valid errors together: maximize count
        # before minimizing error. The rectangular assignment remains one-to-one.
        penalty = (len(eligible) + len(reference) + 1) * (tolerance_seconds + 1)
        errors = np.array([[max(left - value, value - right, 0)
                            for left, right in reference] for _, value in eligible])
        costs = np.where(errors <= tolerance_seconds + 1e-9, errors, penalty)
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if errors[row, column] > tolerance_seconds + 1e-9:
                continue
            original_index, value = eligible[row]
            matched_predictions.add(original_index)
            matched_references.add(int(column))
            matches.append({"prediction_index": original_index, "reference_index": int(column),
                            "timestamp": value, "error_outside_bounds": float(errors[row, column])})
    true_positive = len(matches)
    false_positive = len(eligible) - true_positive
    false_negative = len(reference) - true_positive
    return {
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / len(eligible) if eligible else None,
        "recall": true_positive / len(reference) if reference else None,
        "matches": matches,
        "missed_reference_indexes": [index for index in range(len(reference))
                                     if index not in matched_references],
        "false_prediction_indexes": [index for index, _ in eligible
                                     if index not in matched_predictions],
        "unreviewed_predictions_ignored": len(values) - len(eligible),
        "tolerance_seconds": tolerance_seconds,
        "scope": "Exhaustively reviewed source intervals only",
    }


def evaluate_video(
    predicted_clips: Sequence[Mapping[str, Any]], video: Mapping[str, Any],
    *, tolerance_seconds: float = 0.5,
) -> dict[str, Any]:
    """Contact accuracy plus editorial-boundary and labelled contaminant checks."""
    times = [float(clip["contact"]) for clip in predicted_clips]
    result = evaluate_contacts(times, video.get("contacts", []),
                               video.get("reviewed_intervals", []),
                               tolerance_seconds=tolerance_seconds)
    # Timing accuracy alone cannot tell whether the delivered windows show a
    # stroke. A late event estimate may still keep it; conversely an accurate
    # event record can accompany a clip that accidentally cuts the stroke out.
    clip_bounds = [_bounds(clip) for clip in predicted_clips]
    full, partial, missing = [], [], []
    for index, contact in enumerate(video.get('contacts', [])):
        lower, upper = _bounds(contact, 'lower', 'upper')
        if any(left <= lower and right > upper for left, right in clip_bounds):
            full.append(index)
        elif any(left <= upper and right > lower for left, right in clip_bounds):
            partial.append(index)
        else:
            missing.append(index)
    count = len(video.get('contacts', []))
    result['source_contact_coverage'] = {
        'fully_covered_brackets': len(full), 'partial_brackets': len(partial),
        'missing_brackets': len(missing), 'fully_covered_recall': len(full)/count if count else None,
        'fully_covered_reference_indexes': full, 'partial_reference_indexes': partial,
        'missing_reference_indexes': missing,
        'scope': 'Single source clip contains the complete contact uncertainty interval; clip ends are exclusive. '
                 'This checks clipping records, not decoded frames or complete shot outcomes.',
    }
    boundary_errors = []
    for match in result["matches"]:
        reference = video["contacts"][match["reference_index"]]
        preferred = [clip for clip in video.get("clips", [])
                     if abs(float(clip.get("contact_lower", -1)) - reference["lower"]) < 1e-6
                     and abs(float(clip.get("contact_upper", -1)) - reference["upper"]) < 1e-6]
        if len(preferred) != 1:
            continue
        clip = predicted_clips[match["prediction_index"]]
        boundary_errors.append({
            "prediction_index": match["prediction_index"],
            "start_error_seconds": float(clip["start"]) - float(preferred[0]["start"]),
            "end_error_seconds": float(clip["end"]) - float(preferred[0]["end"]),
            "start_error_outside_bounds": max(
                float(preferred[0].get("start_lower", preferred[0]["start"])) - float(clip["start"]),
                float(clip["start"]) - float(preferred[0].get("start_upper", preferred[0]["start"])), 0),
            "end_error_outside_bounds": max(
                float(preferred[0].get("end_lower", preferred[0]["end"])) - float(clip["end"]),
                float(clip["end"]) - float(preferred[0].get("end_upper", preferred[0]["end"])), 0),
            "provenance": preferred[0]["provenance"],
        })
    contaminants: dict[str, list[dict[str, Any]]] = {"replay": [], "handling": []}
    for index, clip in enumerate(predicted_clips):
        left, right = _bounds(clip)
        for label in video.get("labels", []):
            if label["head"] not in contaminants or float(label["value"]) != 1:
                continue
            start, end = _bounds(label)
            overlap = max(0.0, min(right, end) - max(left, start))
            if overlap > 1e-6:
                contaminants[label["head"]].append({"prediction_index": index,
                    "overlap_seconds": overlap, "label_provenance": label["provenance"]})
    result.update({"boundary_errors": boundary_errors,
                   "boundary_reference_kind": "Reviewed editorial preferences; not physical-stop truth",
                   "labelled_contaminants": contaminants,
                   "false_replay_clips": len({row["prediction_index"] for row in contaminants["replay"]}),
                   "false_handling_clips": len({row["prediction_index"] for row in contaminants["handling"]})})
    adjudicated = sum("original_bounds" in contact for contact in video.get("contacts", []))
    if adjudicated:
        original_contacts = [{**contact, **contact.get("original_bounds", {})}
                             for contact in video["contacts"]]
        result["original_contact_metrics"] = evaluate_contacts(
            times, original_contacts, video.get("reviewed_intervals", []),
            tolerance_seconds=tolerance_seconds)
        result["adjudicated_contact_count"] = adjudicated
        result["contact_annotation_note"] = "Primary results use additive source adjudications; original labels also reported"
    return result
