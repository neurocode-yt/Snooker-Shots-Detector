"""Apply optional pot selection after the original detector completes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from snooker_ai.config import Config
from snooker_ai.pot_detection.scoreboard import ScoreboardReader, ScoreReading
from snooker_ai.types import AnalysisResult, PotStatus, ShotRecord, ShotSelection
from snooker_ai.utils.persistence import atomic_text
from snooker_ai.utils.video import open_capture, sampled_frames

_POLICY = 4


@dataclass
class StableScore:
    start: float
    end: float
    left: int
    right: int
    active_player: int | None
    confidence: float


def stable_scores(readings: list[ScoreReading], *, max_gap: float = 0.8) -> list[StableScore]:
    groups = []
    for reading in sorted(readings, key=lambda item: item.t):
        if (
            groups
            and reading.t - groups[-1][-1].t <= max_gap
            and (reading.left, reading.right, reading.layout)
            == (groups[-1][-1].left, groups[-1][-1].right, groups[-1][-1].layout)
        ):
            groups[-1].append(reading)
        else:
            groups.append([reading])
    result = []
    for group in groups:
        if len(group) < 2 or group[-1].t - group[0].t < 0.3:
            continue
        known = [item for item in group if item.active_player is not None]
        active = None
        if len(known) >= 2 and group[-1].t - known[-1].t <= max_gap:
            last = known[-1]
            prior = known[-2]
            if last.active_player == prior.active_player and 0.3 <= last.t - prior.t <= max_gap:
                active = last.active_player
        result.append(
            StableScore(
                group[0].t,
                group[-1].t,
                group[0].left,
                group[0].right,
                active,
                min(item.confidence for item in group),
            )
        )
    return result


def classify_shot(
    shot: ShotRecord,
    readings: list[ScoreReading],
    end: float,
    *,
    baseline_seconds: float = 6,
    outcome_deadline: float | None = None,
    max_score_gap: float = 0.8,
):
    strike = shot.cue_strike_timestamp
    before = stable_scores(
        [item for item in readings if strike - baseline_seconds <= item.t < strike - 0.05],
        max_gap=max_score_gap,
    )
    after = stable_scores(
        [item for item in readings if strike + 0.2 <= item.t <= end], max_gap=max_score_gap
    )
    if shot.possible_replay:
        return PotStatus.NO_POT, {"reason": "replay"}
    if not before or not after:
        return PotStatus.UNKNOWN, {"reason": "scoreboard_not_readable_before_and_after"}
    base = before[-1]
    if strike - base.end > baseline_seconds:
        return PotStatus.UNKNOWN, {"reason": "stale_pre_shot_score"}
    evidence = {
        "score_before": [base.left, base.right],
        "active_player": base.active_player,
        "baseline_end": base.end,
    }
    changes = []
    reset = False
    for run in after:
        delta = (run.left - base.left, run.right - base.right)
        if min(delta) < 0:
            if run.left != 0 or run.right != 0:
                return PotStatus.UNKNOWN, {**evidence, "reason": "score_correction_during_outcome"}
            reset = True
            break
        if delta != (0, 0):
            changes.append((run, delta))
    if not changes:
        latest = after[-1]
        evidence.update(score_after=[latest.left, latest.right], confirmation_time=latest.end)
        if reset:
            return PotStatus.UNKNOWN, {**evidence, "reason": "frame_reset_or_score_correction"}
        if outcome_deadline is not None and end + 0.8 < outcome_deadline:
            return PotStatus.UNKNOWN, {
                **evidence,
                "reason": "next_shot_or_video_end_prevented_score_confirmation",
            }
        # A missing scoreboard in the last part of the stroke does not prove
        # that a pot failed to score. Require a stable result near its end.
        required = min(end - 0.8, shot.physical_stop_timestamp + 0.8)
        if latest.end < required:
            return PotStatus.UNKNOWN, {**evidence, "reason": "score_update_window_incomplete"}
        return PotStatus.NO_POT, {**evidence, "reason": "score_unchanged"}
    run, delta = changes[-1]
    evidence.update(
        score_after=[run.left, run.right],
        score_delta=list(delta),
        confirmation_time=run.end,
        confidence=min(base.confidence, run.confidence),
        score_update_time=run.start,
    )
    if any(change[1][0] > 0 and change[1][1] > 0 for change in changes):
        return PotStatus.UNKNOWN, {**evidence, "reason": "both_players_scored"}
    side = 0 if delta[0] > 0 else 1
    points = delta[side]
    if not 1 <= points <= 15:
        return PotStatus.UNKNOWN, {**evidence, "reason": "score_jump_not_a_single_scoring_stroke"}
    if not reset and (after[-1].left, after[-1].right) == (base.left, base.right):
        return PotStatus.UNKNOWN, {**evidence, "reason": "score_corrected_back"}
    if base.active_player is not None and side != base.active_player:
        if 4 <= points <= 7:
            return PotStatus.FOUL, {**evidence, "reason": "opponent_received_points"}
        return PotStatus.UNKNOWN, {**evidence, "reason": "turn_indicator_conflicts_with_score"}
    if base.active_player is None and points >= 4:
        return PotStatus.UNKNOWN, {**evidence, "reason": "foul_or_pot_without_turn_indicator"}
    return PotStatus.POT, {
        **evidence,
        "points": points,
        "scoring_player": side,
        "reason": "striker_score_increased",
    }


class PotSelector:
    def __init__(self, config: Config, job_dir: Path):
        self.config, self.job_dir = config, Path(job_dir)
        self.settings = dict(config.get("pot_selection", {}))
        self.reader = ScoreboardReader(float(self.settings.get("min_digit_confidence", 0.78)))

    def _signature(self, result: AnalysisResult):
        source = Path(result.source_path)
        stat = source.stat()
        return hashlib.sha256(
            json.dumps(
                {
                    "policy": _POLICY,
                    "source": str(source.resolve()),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime_ns,
                    "settings": self.settings,
                    "digit_templates": hashlib.sha256(
                        (Path(__file__).with_name("assets") / "score_digits.npz").read_bytes()
                    ).hexdigest(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def scan(self, result: AnalysisResult, progress=None) -> list[ScoreReading]:
        baseline = float(self.settings.get("baseline_seconds", 6))
        grace = float(self.settings.get("score_update_grace_seconds", 8))
        intervals = []
        shots = sorted(result.shots, key=lambda shot: shot.cue_strike_timestamp)
        for i, shot in enumerate(shots):
            if shot.possible_replay:
                continue
            following = (
                shots[i + 1].cue_strike_timestamp - 0.1
                if i + 1 < len(shots)
                else result.original_duration
            )
            start = max(0.0, shot.cue_strike_timestamp - baseline)
            end = min(
                result.original_duration,
                following,
                max(shot.physical_stop_timestamp, shot.cue_strike_timestamp) + grace,
            )
            if end <= start:
                continue
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        signature = self._signature(result)
        readings = []
        period = 1 / max(0.5, float(self.settings.get("sample_fps", 2)))
        checkpoint_every = max(1, int(self.settings.get("checkpoint_every_samples", 120)))
        cap = None
        try:
            for i, (start, end) in enumerate(intervals):
                cache = self.job_dir / "pot_score_cache" / f"window_{i:04d}.json"
                saved = None
                partial = None
                try:
                    data = json.loads(cache.read_text("utf-8"))
                    if data["signature"] == signature and data["range"] == [start, end]:
                        decoded = [ScoreReading(**item) for item in data["readings"]]
                        if data.get("complete", True):
                            saved = decoded
                        else:
                            partial = (
                                decoded,
                                int(data["next_index"]),
                                float(data["next_sample_time"]),
                            )
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                if saved is None:
                    if cap is None:
                        cap = open_capture(
                            result.source_path,
                            prefer_hwaccel=bool(self.config.get("analysis.hwaccel_decode", True)),
                        )
                        if not cap.isOpened():
                            raise RuntimeError("Cannot open source for pot confirmation")
                    fps = cap.get(cv2.CAP_PROP_FPS) or result.metadata.fps
                    if fps <= 0:
                        raise RuntimeError("Cannot determine source cadence for pot confirmation")
                    if partial is not None:
                        saved, index, next_sample = partial
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                            raise RuntimeError("Cannot seek to saved scoreboard checkpoint")
                        if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - index) > 0.5:
                            raise RuntimeError(
                                "Decoder did not reach the saved scoreboard checkpoint"
                            )
                    else:
                        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        next_sample = start
                    frames = sampled_frames(
                        cap,
                        proxy_fps=fps,
                        to_source=lambda t: t,
                        start_time=start,
                        end_time=end,
                        sample_period=period,
                        start_idx=index,
                        next_sample_time=next_sample,
                    )
                    if saved is None:
                        saved = []
                    sampled = 0
                    try:
                        for frame_idx, t, frame in frames:
                            while next_sample <= t + 1e-9:
                                next_sample += period
                            reading = self.reader.read(frame, t)
                            if reading is not None:
                                saved.append(reading)
                            sampled += 1
                            if sampled == 1 or sampled % checkpoint_every == 0:
                                atomic_text(
                                    cache,
                                    json.dumps(
                                        {
                                            "signature": signature,
                                            "range": [start, end],
                                            "complete": False,
                                            "next_index": frame_idx + 1,
                                            "next_sample_time": next_sample,
                                            "readings": [asdict(item) for item in saved],
                                        }
                                    ),
                                )
                            if progress and sampled % 20 == 0:
                                progress(
                                    (i + (t - start) / max(end - start, 0.001))
                                    / max(len(intervals), 1),
                                    f"Checking pot scores at {t:.1f}s",
                                )
                    finally:
                        frames.close()
                    atomic_text(
                        cache,
                        json.dumps(
                            {
                                "signature": signature,
                                "range": [start, end],
                                "complete": True,
                                "readings": [asdict(item) for item in saved],
                            }
                        ),
                    )
                readings.extend(saved)
                if progress:
                    progress(
                        (i + 1) / max(len(intervals), 1),
                        f"Confirming pots: score window {i + 1}/{len(intervals)}",
                    )
        finally:
            if cap is not None:
                cap.release()
        return sorted(
            {round(item.t, 6): item for item in readings}.values(), key=lambda item: item.t
        )

    def apply(
        self,
        result: AnalysisResult,
        progress=None,
        selection: ShotSelection = ShotSelection.POTS_ONLY,
    ) -> AnalysisResult:
        selection = ShotSelection(selection)
        include_fouls = selection is ShotSelection.POTS_AND_FOULS
        readings = self.scan(result, progress)
        grace = float(self.settings.get("score_update_grace_seconds", 8))
        baseline = float(self.settings.get("baseline_seconds", 6))
        counts = {status.value: 0 for status in PotStatus}
        shots = sorted(result.shots, key=lambda shot: shot.cue_strike_timestamp)
        for i, shot in enumerate(shots):
            following = (
                shots[i + 1].cue_strike_timestamp - 0.1
                if i + 1 < len(shots)
                else result.original_duration
            )
            end = min(
                result.original_duration,
                following,
                max(shot.physical_stop_timestamp, shot.cue_strike_timestamp) + grace,
            )
            status, evidence = classify_shot(
                shot,
                readings,
                end,
                baseline_seconds=baseline,
                outcome_deadline=max(shot.physical_stop_timestamp, shot.cue_strike_timestamp)
                + grace,
                max_score_gap=max(0.8, 1.6 / max(0.5, float(self.settings.get("sample_fps", 2)))),
            )
            if (
                include_fouls
                and status is PotStatus.UNKNOWN
                and evidence.get("reason")
                in {"foul_or_pot_without_turn_indicator", "turn_indicator_conflicts_with_score"}
            ):
                # The increase is confirmed even when its pot/foul attribution
                # is uncertain. This workflow deliberately includes both.
                status = PotStatus.POINTS
                evidence["reason"] = "confirmed_points_change"
            scoring = status in (
                {PotStatus.POT, PotStatus.FOUL, PotStatus.POINTS}
                if include_fouls
                else {PotStatus.POT}
            )
            if scoring and i > 0:
                previous = shots[i - 1]
                if (
                    previous.pot_status is PotStatus.UNKNOWN
                    and shot.cue_strike_timestamp < previous.physical_stop_timestamp + grace
                    and evidence.get("score_update_time", end) < shot.cue_strike_timestamp + 2
                ):
                    status = PotStatus.UNKNOWN
                    evidence["reason"] = "delayed_previous_shot_score_is_ambiguous"
            counts[status.value] += 1
            base_included = shot.evidence.setdefault("pots_only_base_included", shot.included)
            base_review = shot.evidence.setdefault(
                "pots_only_base_review_required", shot.manual_review_required
            )
            shot.pot_status = status
            shot.evidence["pot_confirmation"] = evidence
            if not shot.evidence.get("pots_only_manual_override", False):
                kept_statuses = (
                    {PotStatus.POT, PotStatus.FOUL, PotStatus.POINTS}
                    if include_fouls
                    else {PotStatus.POT}
                )
                shot.included = bool(base_included and status in kept_statuses)
                shot.manual_review_required = bool(base_review or status is PotStatus.UNKNOWN)
        result.selection = selection
        result.pot_selection_summary = {
            **counts,
            "included": sum(shot.included for shot in shots),
            "method": "stable_score_and_striker",
            "policy": _POLICY,
            "selection": selection.value,
        }
        result.edited_duration = sum(shot.duration() for shot in shots if shot.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        atomic_text(
            self.job_dir / "pot_selection.json",
            json.dumps(
                {
                    "summary": result.pot_selection_summary,
                    "shots": [
                        {
                            "shot_id": shot.shot_id,
                            "strike": shot.cue_strike_timestamp,
                            "status": shot.pot_status.value,
                            "included": shot.included,
                            "evidence": shot.evidence["pot_confirmation"],
                        }
                        for shot in shots
                    ],
                },
                indent=2,
            ),
        )
        return result
