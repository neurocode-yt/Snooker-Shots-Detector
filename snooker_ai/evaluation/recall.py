"""Evaluate every live shot in independently labelled broadcast intervals."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from snooker_ai.config import Config
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import ExportRequest, ShotRecord


def _match_contacts(predicted: list[float], events: list[dict], tolerance: float) -> list[tuple[int, int, float]]:
    """Maximize matched events before minimizing timing error.

    Each prediction has an unmatched dummy column. Its penalty exceeds the
    total possible normalized error of all real matches, so an ambiguous close
    pair cannot steal the sole match available to another prediction.
    """
    if not predicted or not events:
        return []
    penalty = len(predicted)+len(events)+1
    costs = np.full((len(predicted), len(events)+len(predicted)), float(penalty))
    costs[:, :len(events)] = 2*penalty
    errors = {}
    for i, timestamp in enumerate(predicted):
        for j, event in enumerate(events):
            error = max(event["lower"]-timestamp, timestamp-event["upper"], 0.)
            if error <= tolerance:
                costs[i, j] = error/max(1., tolerance)
                errors[i, j] = error
    rows, columns = linear_sum_assignment(costs)
    return [(int(i), int(j), errors[i, j]) for i, j in zip(rows, columns) if (i, j) in errors]


def evaluate_broadcast_recall(
    shots: list[dict], reference: dict, tolerance: float = .5, *,
    delivered: bool = False, export_request: ExportRequest | None = None,
) -> dict:
    """Count missed/extra events only inside completely inspected intervals.

    A hidden contact may have a labelled time interval rather than a fabricated
    exact frame. Matching is one-to-one; a duplicate prediction is a false
    positive. Unlabelled match time supplies no evidence of accuracy.
    Analysis records use the renderer's exact selection policy. Already
    delivered clips must all count, regardless of their diagnostic replay or
    inclusion flags.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Contact tolerance must be finite and nonnegative")
    records = [ShotRecord.model_validate({"shot_id": i+1, **shot}) for i, shot in enumerate(shots)]
    if any(not math.isfinite(shot.cue_strike) for shot in records):
        raise ValueError("Predicted contact timestamps must be finite")
    request = export_request or ExportRequest()
    if not delivered:
        records = Exporter(Config({"device": "cpu"}))._filter_shots(records, request)
    windows = sorted(reference["windows"], key=lambda w: w["start"])
    for i, window in enumerate(windows):
        lo, hi = window["start"], window["end"]
        if not all(math.isfinite(v) for v in (lo, hi)) or lo >= hi:
            raise ValueError("Invalid labelled interval")
        if i and lo < windows[i-1]["end"]:
            raise ValueError("Labelled intervals must not overlap")
        for event in window["contacts"]:
            if not lo <= event["lower"] <= event["upper"] < hi:
                raise ValueError("Contact bounds must lie inside the labelled interval")
    details = []
    for window in windows:
        predicted = sorted(s.cue_strike for s in records
                           if window["start"] <= s.cue_strike < window["end"])
        events = window["contacts"]
        used_pred, used_reference = set(), set()
        matched = []
        for i, j, error in _match_contacts(predicted, events, tolerance):
            used_pred.add(i)
            used_reference.add(j)
            matched.append({"predicted_contact": predicted[i], "reference": events[j],
                            "error_outside_bounds_seconds": error})
        details.append({"name": window["name"], "split": window.get("split", "unspecified"),
                        "start": window["start"], "end": window["end"],
                        "reference_count": len(events), "predicted_count": len(predicted),
                        "matched": matched,
                        "missed": [e for j, e in enumerate(events) if j not in used_reference],
                        "false_contacts": [t for i, t in enumerate(predicted) if i not in used_pred]})
    true_positive = sum(len(w["matched"]) for w in details)
    false_positive = sum(len(w["false_contacts"]) for w in details)
    missed = sum(len(w["missed"]) for w in details)
    precision = true_positive/(true_positive+false_positive) if true_positive+false_positive else None
    recall = true_positive/(true_positive+missed) if true_positive+missed else None
    return {"scope": "Completely labelled intervals only; full-match accuracy is unmeasured.",
            "prediction_scope": "delivered_clips" if delivered else "export_selection",
            "export_selection": None if delivered else {
                key: getattr(request, key) for key in (
                    "include_replays", "only_included", "min_confidence", "min_importance",
                )},
            "contact_tolerance_seconds": tolerance, "matched": true_positive,
            "missed": missed, "false_positive": false_positive,
            "precision": precision, "recall": recall, "windows": details}
