"""Shot detection evaluation against ground-truth annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from snooker_ai.utils.logging import get_logger

logger = get_logger("evaluation")


@dataclass
class EvaluationReport:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    missed_shot_rate: float = 0.0
    false_shot_rate: float = 0.0
    median_strike_error: float = 0.0
    median_end_error: float = 0.0
    mean_strike_error: float = 0.0
    mean_end_error: float = 0.0
    matched: int = 0
    predicted: int = 0
    ground_truth: int = 0
    notes: list[str] = field(default_factory=list)
    per_video: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_gt_strikes(path: Path) -> list[dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        labels = data.get("labels") or data.get("shots") or data.get("events") or []
    else:
        labels = data
    out = []
    for item in labels:
        if "cue_strike" in item:
            out.append(
                {
                    "cue_strike": float(item["cue_strike"]),
                    "ball_motion_end": float(
                        item.get("ball_motion_end") or item.get("end") or item["cue_strike"] + 3
                    ),
                }
            )
        elif "timestamp" in item and item.get("event_type", "cue_strike") == "cue_strike":
            out.append(
                {
                    "cue_strike": float(item["timestamp"]),
                    "ball_motion_end": float(item.get("end") or item["timestamp"] + 3),
                }
            )
    return out


def match_events(
    pred: list[float],
    gt: list[float],
    tolerance: float = 0.75,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy 1-1 matching within tolerance. Returns matches, unmatched_pred, unmatched_gt."""
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = abs(p - g)
            if d <= tolerance:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_g = set(), set()
    matches: list[tuple[int, int, float]] = []
    for d, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, d))
    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    unmatched_g = [j for j in range(len(gt)) if j not in used_g]
    return matches, unmatched_p, unmatched_g


def evaluate_predictions(
    pred_shots: list[dict[str, Any]],
    gt_shots: list[dict[str, float]],
    strike_tol: float = 0.75,
) -> dict[str, Any]:
    pred_t = [float(s["cue_strike"]) for s in pred_shots]
    gt_t = [float(s["cue_strike"]) for s in gt_shots]
    matches, um_p, um_g = match_events(pred_t, gt_t, tolerance=strike_tol)

    tp = len(matches)
    fp = len(um_p)
    fn = len(um_g)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    strike_errs = [m[2] for m in matches]
    end_errs = []
    for i, j, _ in matches:
        pe = float(pred_shots[i].get("ball_motion_end", pred_t[i]))
        ge = float(gt_shots[j].get("ball_motion_end", gt_t[j]))
        end_errs.append(abs(pe - ge))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_shot_rate": fn / len(gt_t) if gt_t else 0.0,
        "false_shot_rate": fp / len(pred_t) if pred_t else 0.0,
        "median_strike_error": float(np.median(strike_errs)) if strike_errs else 0.0,
        "mean_strike_error": float(np.mean(strike_errs)) if strike_errs else 0.0,
        "median_end_error": float(np.median(end_errs)) if end_errs else 0.0,
        "mean_end_error": float(np.mean(end_errs)) if end_errs else 0.0,
        "matched": tp,
        "predicted": len(pred_t),
        "ground_truth": len(gt_t),
    }


def evaluate_dataset(
    dataset_dir: str | Path,
    strike_tol: float = 0.75,
) -> EvaluationReport:
    """
    Dataset layout:
      video_id/
        predictions.json  # {shots: [{cue_strike, ball_motion_end, ...}]}
        ground_truth.json
    """
    dataset_dir = Path(dataset_dir)
    report = EvaluationReport()
    report.notes.append(
        "Targets (not measured claims): recall≥0.98, precision≥0.97, "
        "median strike error≤0.25s, median end error≤0.50s."
    )

    if not dataset_dir.is_dir():
        report.notes.append(f"Dataset directory not found: {dataset_dir}")
        return report

    all_metrics = []
    for sub in sorted(dataset_dir.iterdir()):
        if not sub.is_dir():
            continue
        pred_path = sub / "predictions.json"
        gt_path = sub / "ground_truth.json"
        if not pred_path.exists() or not gt_path.exists():
            # Also accept analysis.json style
            if (sub / "analysis.json").exists() and gt_path.exists():
                pred_data = json.loads((sub / "analysis.json").read_text(encoding="utf-8"))
                pred_shots = pred_data.get("shots", [])
            else:
                continue
        else:
            pred_data = json.loads(pred_path.read_text(encoding="utf-8"))
            pred_shots = pred_data.get("shots", pred_data if isinstance(pred_data, list) else [])

        gt_shots = _load_gt_strikes(gt_path)
        m = evaluate_predictions(pred_shots, gt_shots, strike_tol=strike_tol)
        m["video"] = sub.name
        all_metrics.append(m)
        report.per_video.append(m)

    if not all_metrics:
        report.notes.append("No video folders with predictions+ground_truth found.")
        return report

    def avg(key: str) -> float:
        return float(np.mean([m[key] for m in all_metrics]))

    report.precision = avg("precision")
    report.recall = avg("recall")
    report.f1 = avg("f1")
    report.missed_shot_rate = avg("missed_shot_rate")
    report.false_shot_rate = avg("false_shot_rate")
    report.median_strike_error = avg("median_strike_error")
    report.median_end_error = avg("median_end_error")
    report.mean_strike_error = avg("mean_strike_error")
    report.mean_end_error = avg("mean_end_error")
    report.matched = sum(m["matched"] for m in all_metrics)
    report.predicted = sum(m["predicted"] for m in all_metrics)
    report.ground_truth = sum(m["ground_truth"] for m in all_metrics)
    return report
