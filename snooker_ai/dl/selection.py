"""Turn independent neural event/edit predictions into reviewable highlights.

This module consumes probabilities only. It has no classic detector, motion,
ball tracking, or segmenter fallback. An editorial endpoint is never evidence
of an observed physical ball stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any

import numpy as np

from snooker_ai.types import CameraViewType, ConfidenceLevel, ShotRecord, StrikeCandidate

HEADS = ("event", "keep", "end", "replay", "handling")


@dataclass(frozen=True)
class _Peak:
    index: int
    probability: float
    lower: float
    upper: float
    censored: bool


def _number(settings: dict, name: str, default: float, minimum: float = 0.0) -> float:
    value = float(settings.get(name, default))
    if not isfinite(value) or value < minimum:
        raise ValueError(f"Invalid DL selection setting {name}: {value}")
    return value


def _validated_heads(settings: dict) -> dict[str, bool]:
    validation = settings.get("head_validation")
    if validation is None:
        validation = settings.get("calibration", {}).get("frame_heads", {})
    result = {}
    for head in HEADS[1:]:
        report = validation.get(head, {})
        result[head] = report is True or (
            isinstance(report, dict) and report.get("status") == "positive_and_negative_labels"
        )
    return result


def _cadence(times: np.ndarray, settings: dict, duration: float | None = None) -> float:
    if "cadence_seconds" in settings:
        return _number(settings, "cadence_seconds", .2, 1e-9)
    if "sample_fps" in settings:
        return 1 / _number(settings, "sample_fps", 5.0, 1e-9)
    # Sparse reviewed windows can contain isolated frames separated by minutes.
    # Those gaps do not describe the feature extractor's sampling cadence.
    if len(times) > 1:
        return min(1.0, float(np.median(np.diff(times))))
    return min(.2, duration) if duration is not None and duration > 0 else .2


def _local_maxima(
    values: np.ndarray, threshold: float, *, timestamps: np.ndarray | None = None,
    max_gap: float | None = None,
) -> list[int]:
    """A plateau is one event per contiguous observation block."""
    def continuous(previous: int, current: int) -> bool:
        return timestamps is None or max_gap is None or timestamps[current] - timestamps[previous] <= max_gap

    peaks = []
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and continuous(end, end + 1) and values[end + 1] == values[index]:
            end += 1
        if values[index] >= threshold and (
            index == 0 or not continuous(index - 1, index) or values[index] > values[index - 1]
        ) and (end == len(values) - 1 or not continuous(end, end + 1) or values[end] > values[end + 1]):
            peaks.append((index + end) // 2)
        index = end + 1
    return peaks


def _event_peaks(
    times: np.ndarray, event: np.ndarray, threshold: float, cadence: float,
    duration: float, settings: dict,
) -> list[_Peak]:
    ratio = _number(settings, "uncertainty_ratio", .5)
    if ratio > 1:
        raise ValueError("uncertainty_ratio must be at most one")
    radius = _number(settings, "max_uncertainty_seconds", 2.0) / 2
    gap = _number(settings, "max_observation_gap_seconds", max(.5, cadence * 1.5))
    peaks = []
    segments = np.cumsum(np.r_[False, np.diff(times) > gap])
    for index in _local_maxima(event, threshold, timestamps=times, max_gap=gap):
        support = max(threshold * ratio, float(event[index]) * ratio)
        left = right = index
        while left > 0 and times[index] - times[left - 1] <= radius and (
            times[left] - times[left - 1] <= gap and event[left - 1] >= support
        ):
            left -= 1
        while right + 1 < len(times) and times[right + 1] - times[index] <= radius and (
            times[right + 1] - times[right] <= gap and event[right + 1] >= support
        ):
            right += 1
        censored = bool(
            (left > 0 and times[left] - times[left - 1] <= gap and event[left - 1] >= support)
            or (right + 1 < len(times) and times[right + 1] - times[right] <= gap and event[right + 1] >= support)
        )
        peaks.append(_Peak(
            index, float(event[index]), max(0.0, float(times[left]) - cadence / 2),
            min(duration, float(times[right]) + cadence / 2), censored,
        ))
    spacing = max(cadence / 2, _number(settings, "nms_seconds", 1.0))
    bucket_size = max(spacing * 2, radius * 2 + cadence, 1e-6)
    buckets: dict[tuple[int, int], list[_Peak]] = {}
    accepted = []
    for peak in sorted(peaks, key=lambda value: (-value.probability, times[value.index])):
        bucket = floor(float(times[peak.index]) / bucket_size)
        segment = int(segments[peak.index])
        duplicate = False
        for neighbor in range(bucket - 1, bucket + 2):
            for other in buckets.get((segment, neighbor), []):
                distance = abs(float(times[peak.index] - times[other.index]))
                intersection = max(0.0, min(peak.upper, other.upper) - max(peak.lower, other.lower))
                union = max(peak.upper, other.upper) - min(peak.lower, other.lower)
                if distance <= spacing or (
                    distance <= spacing * 2 and union > 0 and intersection / union >= .5
                ):
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            accepted.append(peak)
            buckets.setdefault((segment, bucket), []).append(peak)
    return sorted(accepted, key=lambda value: times[value.index])


def _confident_spans(
    times: np.ndarray, values: np.ndarray, threshold: float, cadence: float,
    duration: float, minimum_duration: float, max_gap: float,
) -> list[tuple[float, float]]:
    spans = []
    index = 0
    while index < len(times):
        if values[index] < threshold:
            index += 1
            continue
        first = last = index
        while last + 1 < len(times) and values[last + 1] >= threshold and (
            times[last + 1] - times[last] <= max_gap
        ):
            last += 1
        left = max(0.0, float(times[first]) - cadence / 2)
        right = min(duration, float(times[last]) + cadence / 2)
        if last > first and right - left + 1e-9 >= minimum_duration:
            spans.append((left, right))
        index = last + 1
    return spans


def _prediction_inputs(
    timestamps: np.ndarray, probabilities: np.ndarray, duration: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.asarray(timestamps, dtype=np.float64)
    output = np.asarray(probabilities, dtype=np.float64)
    duration = float(duration)
    if times.ndim != 1 or output.shape != (len(times), len(HEADS)):
        raise ValueError("DL predictions must be timestamps[N] and probabilities[N,5]")
    if not isfinite(duration) or duration < 0 or not np.isfinite(times).all() or not np.isfinite(output).all():
        raise ValueError("DL predictions and duration must be finite")
    if (np.diff(times) <= 0).any() or (times < 0).any() or (times > duration + 1e-9).any():
        raise ValueError("DL timestamps must increase strictly within the source duration")
    if (output < 0).any() or (output > 1).any():
        raise ValueError("DL model outputs must be probabilities between zero and one")
    return times, output, duration


def predict_event_times(
    timestamps: np.ndarray, probabilities: np.ndarray, threshold: float,
    settings: dict | None = None, *, duration: float | None = None,
) -> list[float]:
    """Decode events identically for threshold calibration and live selection.

    This returns all decoded event candidates before the validated replay and
    replay head decides clip inclusion. Handling remains advisory because it
    can coexist with a live stroke. Supply the actual source duration
    when available; otherwise the final sample's half-cadence cell is used.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("DL timestamps must be a one-dimensional vector")
    cadence = _cadence(times, settings or {})
    if duration is None:
        duration = float(times[-1] + cadence / 2) if len(times) else 0.0
    times, output, duration = _prediction_inputs(times, probabilities, duration)
    threshold = float(threshold)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("DL event threshold must be a finite probability")
    if not len(times) or duration == 0:
        return []
    cadence = _cadence(times, settings or {}, duration)
    peaks = _event_peaks(times, output[:, 0], threshold, cadence, duration, settings or {})
    return [float(times[peak.index]) for peak in peaks]


def _learned_end(
    times: np.ndarray, probabilities: np.ndarray, strike: float, minimum_end: float,
    available_end: float, thresholds: dict[str, float], validated: dict[str, bool],
    cadence: float, settings: dict,
) -> tuple[float, float, str, bool]:
    horizon = min(available_end, strike + _number(settings, "max_post_seconds", 30.0))
    horizon = max(minimum_end, horizon)
    selected = np.flatnonzero((times >= strike) & (times <= horizon))
    fallback = min(horizon, max(minimum_end, strike + _number(settings, "fallback_post_seconds", 8.0)))
    if not len(selected):
        return fallback, 0.0, "bounded_uncertainty_fallback", True
    local_times = times[selected]
    keep = probabilities[selected, 1]
    end = probabilities[selected, 2]
    hold = _number(settings, "minimum_probability_duration_seconds", .4)
    max_gap = _number(settings, "max_observation_gap_seconds", max(.5, cadence * 1.5))
    keep_end = None
    if validated["keep"]:
        # Require positive keep support after the event, then a sustained fall.
        supported = False
        low_start = None
        last_supported_time = None
        for index, (time, value) in enumerate(zip(local_times, keep)):
            if index and local_times[index] - local_times[index - 1] > max_gap:
                supported = False
                low_start = None
            if value >= thresholds["keep"]:
                supported = True
                last_supported_time = float(time)
                low_start = None
            elif supported and time >= minimum_end:
                if low_start is None or (index and local_times[index] - local_times[index - 1] > max_gap):
                    low_start = index
                if time - local_times[low_start] + cadence + 1e-9 >= hold:
                    # The first low observation already shows unwanted footage.
                    # End at the last supported observation instead of including
                    # that first referee/cutaway frame in an exclusive clip end.
                    keep_end = max(minimum_end, last_supported_time)
                    break
    end_peak = None
    if validated["end"]:
        for index in _local_maxima(end, thresholds["end"], timestamps=local_times, max_gap=max_gap):
            if local_times[index] >= minimum_end:
                end_peak = (float(local_times[index]), float(end[index]))
                break
    unknown = not (validated["keep"] and validated["end"])
    if end_peak is not None and keep_end is not None:
        agreement = _number(settings, "end_agreement_seconds", 2.0)
        disagreement = abs(end_peak[0] - keep_end) > agreement
        # Agreeing heads define an uncertainty range. Choose its earlier edge
        # while preserving the minimum outcome window, so one optimistic head
        # cannot extend an otherwise finished highlight into handling footage.
        boundary = max(end_peak[0], keep_end) if disagreement else min(end_peak[0], keep_end)
        return min(horizon, max(minimum_end, boundary)), end_peak[1], "learned_end_and_keep", unknown or disagreement
    if end_peak is not None:
        return max(minimum_end, end_peak[0]), end_peak[1], "learned_end", True
    if keep_end is not None:
        local_index = int(np.argmin(np.abs(local_times - keep_end)))
        confidence = float(1 - keep[local_index])
        return max(minimum_end, keep_end), confidence, "learned_keep_fall", True
    return fallback, 0.0, "bounded_uncertainty_fallback", True


def select_highlights(
    timestamps: np.ndarray, probabilities: np.ndarray, duration: float,
    settings: dict, thresholds: dict,
) -> tuple[list[StrikeCandidate], list[ShotRecord], dict[str, Any]]:
    """Select clips from event, keep, end, replay, handling probabilities.

    Pass calibration ``frame_heads`` as ``settings['head_validation']``.
    Unvalidated heads remain advisory. Optional time settings are seconds:
    pre_roll (>=2), min_clip_seconds (>=4), min_post_seconds (>=2),
    max_post_seconds (30), fallback_post_seconds (8), nms_seconds (1),
    max_uncertainty_seconds (2), minimum_probability_duration_seconds (.4).
    """
    times, output, duration = _prediction_inputs(timestamps, probabilities, duration)
    limits = {head: float(thresholds.get(head, .5)) for head in HEADS}
    if any(not isfinite(value) or not 0 <= value <= 1 for value in limits.values()):
        raise ValueError("DL thresholds must be finite probabilities")
    validated = _validated_heads(settings)
    diagnostics: dict[str, Any] = {
        "algorithm": "dl_algo", "boundary_kind": "neural_edit_estimate", "stop_confirmed": False,
        "validated_heads": validated, "unvalidated_heads": [head for head, valid in validated.items() if not valid],
        "predictions": len(times), "event_threshold": limits["event"],
    }
    if not len(times) or duration == 0:
        diagnostics.update(candidates=0, included_shots=0, excluded_replays=0, excluded_handling=0, review_required=0)
        return [], [], diagnostics
    cadence = _cadence(times, settings, duration)
    peaks = _event_peaks(times, output[:, 0], limits["event"], cadence, duration, settings)
    hold = _number(settings, "minimum_probability_duration_seconds", .4)
    max_gap = _number(settings, "max_observation_gap_seconds", max(.5, cadence * 1.5))
    spans = {}
    for index, head in ((3, "replay"), (4, "handling")):
        confidence_floor = _number(settings, f"{head}_confidence", .8)
        spans[head] = _confident_spans(
            times, output[:, index], max(limits[head], confidence_floor), cadence, duration, hold, max_gap,
        ) if validated[head] else []
    pre_roll = max(2.0, _number(settings, "pre_roll", 2.0))
    min_clip = max(4.0, _number(settings, "min_clip_seconds", 4.0))
    min_post = max(2.0, _number(settings, "min_post_seconds", 2.0))
    candidates = []
    records: list[dict[str, Any]] = []
    for peak in peaks:
        strike = float(times[peak.index])
        replay_span = next((span for span in spans["replay"] if span[0] <= strike <= span[1]), None)
        handling = any(left <= strike <= right for left, right in spans["handling"])
        # Two independently learned sigmoid heads may both be true. Referee
        # presence cannot disprove an event that passes the cue-event threshold.
        excluded_handling = False
        candidate = StrikeCandidate(
            timestamp=strike, confidence=peak.probability,
            uncertainty_start=peak.lower, uncertainty_end=peak.upper,
            possible_replay=replay_span is not None,
            camera_view=CameraViewType.REPLAY if replay_span else CameraViewType.OTHER,
            evidence={f"neural_{head}_probability": float(output[peak.index, index]) for index, head in enumerate(HEADS)},
        )
        candidates.append(candidate)
        records.append({
            "peak": peak, "strike": strike, "replay_span": replay_span,
            "handling": handling, "excluded_handling": excluded_handling,
            "replay_uncertain": bool(replay_span is None and output[peak.index, 3] >= max(limits["replay"], .8)),
            "handling_uncertain": bool(not handling and output[peak.index, 4] >= max(limits["handling"], .8)),
            "included": replay_span is None and not excluded_handling,
        })
    live = [record for record in records if record["included"]]
    for index, record in enumerate(live):
        strike = record["strike"]
        before = [right for left, right in spans["replay"] if right <= strike]
        after = [left for left, right in spans["replay"] if left > strike]
        available_start = max(before, default=0.0)
        available_end = min(after, default=duration)
        if index + 1 < len(live):
            available_end = min(available_end, max(strike, live[index + 1]["peak"].lower))
        start = max(available_start, strike - pre_roll, 0.0)
        protected_post = min_post + max(0.0, record["peak"].upper - strike)
        minimum_end = min(available_end, max(start + min_clip, strike + protected_post))
        end, confidence, reason, review = _learned_end(
            times, output, strike, minimum_end, available_end, limits, validated, cadence, settings,
        )
        record.update(start=start, end=min(available_end, end), available_start=available_start,
                      available_end=available_end, minimum_end=minimum_end,
                      protected_post=protected_post, end_confidence=confidence, end_reason=reason,
                      review=review, trimmed_for_next_shot=None)
    # Allocate shared source time once. Minimum post-impact visibility has
    # priority when two contacts make both two-second lead-ins impossible.
    for left, right in zip(live, live[1:]):
        if left["end"] > right["start"]:
            partition = min(right["strike"], max(right["start"], left["minimum_end"]))
            left["end"] = min(left["end"], partition)
            left["available_end"] = min(left["available_end"], partition)
            left["trimmed_for_next_shot"] = right["strike"]
            right["start"] = max(right["start"], left["end"])
            right["minimum_end"] = min(right["available_end"], max(
                right["start"] + min_clip, right["strike"] + right["protected_post"],
            ))
            right["end"] = min(right["available_end"], max(right["end"], right["minimum_end"]))
    shots = []
    for record in records:
        peak, strike = record["peak"], record["strike"]
        if not record["included"]:
            available_start, available_end = record["replay_span"] or (0.0, duration)
            start = max(available_start, strike - pre_roll)
            end = min(available_end, max(strike + min_post, start + min_clip))
            record.update(start=start, end=end, available_start=available_start, available_end=available_end,
                          minimum_end=end, protected_post=min_post, end_confidence=0.0,
                          end_reason="excluded_replay" if record["replay_span"] else "handling_conflict",
                          review=True, trimmed_for_next_shot=None)
        start, end = float(record["start"]), float(record["end"])
        if end <= start + 1e-9:
            continue
        # A shortened pre-roll is represented explicitly for strict validation.
        trimmed_pre = max(0.0, start - max(0.0, strike - pre_roll, record["available_start"]))
        review = bool(record["review"] or peak.censored or record["handling"] or trimmed_pre > 0 or
                      record["replay_uncertain"] or record["handling_uncertain"] or
                      end - strike + 1e-9 < record["protected_post"] or end - start + 1e-9 < min_clip)
        observed = times[(times >= start) & (times <= end)]
        extraction_gap = bool(len(observed) > 1 and (np.diff(observed) > max_gap).any())
        review = review or extraction_gap
        evidence = {
            "algorithm": "dl_algo", "boundary_kind": "neural_edit_estimate", "stop_confirmed": False,
            "physical_stop_observed": False, "boundary_reason": record["end_reason"],
            "extraction_gap": extraction_gap,
            "event_uncertainty_start": peak.lower, "event_uncertainty_end": peak.upper,
            "event_support_censored": peak.censored, "validated_heads": validated,
            "end_before_ball_stop_seconds": 0.0, "minimum_clip_seconds": min_clip,
            "minimum_strike_visibility_seconds": record["protected_post"],
            "minimum_clip_end_timestamp": min(end, record["minimum_end"]),
            "usable_source_start_timestamp": record["available_start"],
            "usable_source_end_timestamp": record["available_end"],
            "pre_roll_trimmed_seconds": trimmed_pre,
            "estimated_editorial_end_timestamp": end,
            "handling_event_conflict": record["handling"],
            "replay_uncertain": record["replay_uncertain"],
            "handling_uncertain": record["handling_uncertain"],
            **{f"neural_{head}_probability": float(output[peak.index, index]) for index, head in enumerate(HEADS)},
        }
        if record["trimmed_for_next_shot"] is not None:
            evidence["trimmed_for_next_shot"] = record["trimmed_for_next_shot"]
        # A short incoming lead-in cannot safely hide uncertainty in a fade.
        fade_guard = _number(settings, "transition_seconds", .24) + cadence
        if peak.lower - start < fade_guard:
            evidence["minimum_clip_seconds"] = max(min_clip, end - start)
            evidence["mix_disabled_for_contact_uncertainty"] = True
        confidence = peak.probability
        shots.append(ShotRecord(
            shot_id=len(shots) + 1, preparation_start=start,
            cue_strike=strike, cue_strike_timestamp=strike, ball_motion_start=strike,
            ball_motion_end=end, physical_stop_timestamp=end, last_ball_motion_timestamp=end,
            stop_confirmation_timestamp=end, clip_start=start, clip_start_timestamp=start,
            clip_end=end, clip_end_timestamp=end, shot_confidence=confidence,
            strike_confidence=confidence, start_confidence=confidence,
            end_confidence=record["end_confidence"], stop_confidence=record["end_confidence"],
            included=record["included"], possible_replay=record["replay_span"] is not None,
            manual_review_required=review, evidence=evidence, importance=confidence,
            camera_views=[CameraViewType.REPLAY.value] if record["replay_span"] else [],
            confidence_level=ConfidenceLevel.LOW if review else (
                ConfidenceLevel.HIGH if confidence >= .8 else ConfidenceLevel.MEDIUM
            ),
        ))
    diagnostics.update(
        candidates=len(candidates), included_shots=sum(shot.included for shot in shots),
        excluded_replays=sum(shot.possible_replay for shot in shots),
        excluded_handling=sum(record["excluded_handling"] for record in records),
        review_required=sum(shot.manual_review_required for shot in shots),
        replay_spans=[list(span) for span in spans["replay"]],
        cadence_seconds=cadence,
        boundary_reasons={reason: sum(shot.evidence["boundary_reason"] == reason for shot in shots)
                          for reason in sorted({shot.evidence["boundary_reason"] for shot in shots})},
    )
    return candidates, shots, diagnostics
