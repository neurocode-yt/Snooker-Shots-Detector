"""Build shot records while preserving exact strict-mode boundaries."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from snooker_ai.config import Config
from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.types import (
    CameraViewType,
    ConfidenceLevel,
    EditMode,
    FrameFeatures,
    ShotRecord,
    StrikeCandidate,
)
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.timebase import clamp

logger = get_logger("segmentation")


class SegmentBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.ball_stop = BallStopDetector(config)
        conf = config.section("confidence")
        self.high = float(conf.get("high", 0.70))
        self.medium = float(conf.get("medium", 0.50))
        self.low = float(conf.get("low", 0.35))
        self.fail_safe = float(conf.get("fail_safe_keep_extra_seconds", 1.0))
        self.strict_pre_roll = float(
            config.mode_settings(EditMode.STRICT).get("pre_roll", 2.0)
        )
        self.conflict_confidence_margin = float(
            config.get("strike_fusion.conflict_confidence_margin", 0.05)
        )
        self.audio_support_threshold = float(config.get("audio.onset_delta", 0.15))

    def _level(self, score: float) -> ConfidenceLevel:
        if score >= self.high:
            return ConfidenceLevel.HIGH
        if score >= self.medium:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _strict_start(self, strike_t: float, duration: float) -> float:
        """The immutable strict start rule from the editing contract."""
        return clamp(float(strike_t) - self.strict_pre_roll, 0.0, duration)

    def build(
        self,
        candidates: list[StrikeCandidate],
        features: list[FrameFeatures],
        duration: float,
        mode: EditMode,
    ) -> list[ShotRecord]:
        mode_cfg = self.config.mode_settings(mode)
        pre_roll = float(mode_cfg.get("pre_roll", 2.0))
        post_roll = float(mode_cfg.get("post_roll", 2.0))
        include_prep = bool(mode_cfg.get("include_preparation", False))
        prep_max = float(mode_cfg.get("preparation_max", pre_roll))
        include_reaction = bool(mode_cfg.get("include_reaction", False))
        reaction_max = float(mode_cfg.get("reaction_max", post_roll))
        retain_replays = bool(mode_cfg.get("retain_replays", False))
        strict = mode == EditMode.STRICT

        # Remove sub-frame/nearby duplicates before stop searches.  This never
        # changes the winning candidate's timestamp.
        ordered = self._deduplicate_candidates(candidates)
        shots: list[ShotRecord] = []
        feature_times = [feature.t for feature in features]

        for cand in ordered:
            stop = self.ball_stop.detect_stop(
                cand, features, duration, times=feature_times
            )

            # A practice stroke/feathering candidate with no sustained ball
            # motion is not a shot.  Ambiguous cases that did show movement are
            # retained conservatively and flagged by StopDetection.
            if stop.reason == "no_sustained_ball_motion":
                logger.debug("Rejected strike %.3f: no sustained ball motion", cand.timestamp)
                continue

            physical_stop = stop.physical_stop_timestamp
            uncapped_physical_stop = physical_stop
            confirmation = stop.stop_confirmation_timestamp
            last_motion = stop.last_ball_motion_timestamp
            end_confidence = stop.end_confidence
            stop_confirmed = stop.confirmed
            stop_reason = stop.reason
            stop_review = stop.manual_review_required
            minimum_clip_end = physical_stop

            if strict:
                clip_start = self._strict_start(cand.timestamp, duration)
                prep_start = clip_start
                # The strict contract uses an actual physical stop whenever it
                # is observed early.  If tracking remains unresolved, cap the
                # exported shot at the configured post-strike horizon and mark
                # the boundary for review instead of allowing a long false
                # track to run away.
                max_after = mode_cfg.get("max_seconds_after_strike")
                if max_after is not None:
                    shot_cap = clamp(
                        cand.timestamp + max(0.0, float(max_after)),
                        clip_start,
                        duration,
                    )
                    if physical_stop > shot_cap + 1e-9:
                        physical_stop = shot_cap
                        confirmation = shot_cap
                        last_motion = min(last_motion, shot_cap)
                        end_confidence = min(end_confidence, 0.20)
                        stop_confirmed = False
                        stop_reason = "max_seconds_after_strike_review_cap"
                        stop_review = True
                # Confirmation is look-ahead only.  No pad, confidence tail, or
                # confidence tail is allowed to alter this boundary. A separate
                # practical clip limit prevents one unresolved track from
                # producing a 40–50 second segment.
                max_clip = float(mode_cfg.get("max_clip_seconds", 12.0))
                clip_cap = clamp(clip_start + max_clip, clip_start, duration)
                if physical_stop > clip_cap + 1e-9:
                    physical_stop = clip_cap
                    confirmation = clip_cap
                    last_motion = min(last_motion, clip_cap)
                    end_confidence = min(end_confidence, 0.20)
                    stop_confirmed = False
                    stop_reason = "max_clip_duration_review_cap"
                    stop_review = True
                min_after = max(
                    0.0,
                    float(mode_cfg.get("min_seconds_after_strike", 0.0)),
                )
                minimum_clip_end = clamp(
                    cand.timestamp + min_after,
                    clip_start,
                    duration,
                )
                clip_end = clamp(
                    max(physical_stop, minimum_clip_end),
                    clip_start,
                    min(duration, clip_cap),
                )
            else:
                if include_prep:
                    prep_start = self._preparation_start(
                        cand.timestamp,
                        features,
                        prep_max,
                        duration,
                        times=feature_times,
                    )
                    clip_start = prep_start
                else:
                    clip_start = clamp(cand.timestamp - pre_roll, 0.0, duration)
                    prep_start = clip_start

                reaction = post_roll
                if include_reaction:
                    reaction = max(post_roll, min(reaction_max, post_roll + 1.0))
                clip_end = clamp(physical_stop + reaction, clip_start, duration)

            shot_conf = float(cand.confidence)
            level = self._level(shot_conf)
            review = (
                level != ConfidenceLevel.HIGH
                or stop_review
                or not stop_confirmed
                or float(cand.evidence.get("cue_geometry_confirmed", 1.0)) < 0.5
            )

            if not strict and level == ConfidenceLevel.LOW:
                clip_start = clamp(clip_start - self.fail_safe, 0.0, duration)
                clip_end = clamp(clip_end + self.fail_safe, clip_start, duration)

            possible_replay = bool(cand.possible_replay)
            included = True
            if possible_replay and not retain_replays:
                # Only explicit replay camera classifications auto-exclude.  A
                # weak visual-signature guess remains reviewable and cannot
                # silently delete a live shot.
                if cand.camera_view in {
                    CameraViewType.REPLAY,
                    CameraViewType.SLOW_MOTION_REPLAY,
                } or float(cand.evidence.get("replay_signature_confirmed", 0.0)) >= 0.5:
                    included = False
                review = True

            evidence = dict(cand.evidence)
            evidence.update(
                {
                    # Used only to suppress mid-motion duplicate candidates;
                    # the exported strict boundary may be capped separately.
                    "uncapped_physical_stop_timestamp": uncapped_physical_stop,
                    "minimum_clip_end_timestamp": minimum_clip_end,
                    "last_ball_motion_timestamp": last_motion,
                    "physical_stop_timestamp": physical_stop,
                    "stop_confirmation_timestamp": confirmation,
                    "stop_confirmed": stop_confirmed,
                    "stop_reason": stop_reason,
                }
            )
            views = self._views_between(
                features, clip_start, clip_end, cand, times=feature_times
            )
            shots.append(
                ShotRecord(
                    shot_id=len(shots) + 1,
                    preparation_start=prep_start,
                    cue_strike=cand.timestamp,
                    cue_strike_timestamp=cand.timestamp,
                    ball_motion_start=stop.motion_start,
                    # Legacy field remains the physical all-ball stop.
                    ball_motion_end=physical_stop,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    clip_start_timestamp=clip_start,
                    clip_end_timestamp=clip_end,
                    shot_confidence=shot_conf,
                    start_confidence=stop.start_confidence,
                    end_confidence=end_confidence,
                    last_ball_motion_timestamp=last_motion,
                    physical_stop_timestamp=physical_stop,
                    stop_confirmation_timestamp=confirmation,
                    strike_confidence=shot_conf,
                    stop_confidence=end_confidence,
                    camera_views=views,
                    possible_replay=possible_replay,
                    manual_review_required=review,
                    evidence=evidence,
                    included=included,
                    confidence_level=level,
                )
            )

        shots = self._resolve_overlaps(shots, strict=strict)
        logger.info("Built %d shot segments (mode=%s)", len(shots), mode.value)
        return shots

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[StrikeCandidate],
        distance_s: float = 0.60,
    ) -> list[StrikeCandidate]:
        result: list[StrikeCandidate] = []
        for cand in sorted(candidates, key=lambda c: c.timestamp):
            if not result or cand.timestamp - result[-1].timestamp >= distance_s:
                result.append(cand)
                continue
            if cand.confidence > result[-1].confidence:
                result[-1] = cand
        return result

    @staticmethod
    def _preparation_start(
        strike_t: float,
        features: list[FrameFeatures],
        prep_max: float,
        duration: float,
        times: list[float] | None = None,
    ) -> float:
        start = clamp(strike_t - prep_max, 0.0, duration)
        if times is None:
            times = [feature.t for feature in features]
        lo = bisect_left(times, start)
        hi = bisect_left(times, strike_t)
        for f in reversed(features[lo:hi]):
            if strike_t - f.t > prep_max:
                break
            if f.motion_score > 0.4:
                start = f.t
                break
        return start

    @staticmethod
    def _views_between(
        features: list[FrameFeatures],
        start: float,
        end: float,
        cand: StrikeCandidate,
        times: list[float] | None = None,
    ) -> list[str]:
        values = {cand.camera_view.value}
        if times is None:
            times = [feature.t for feature in features]
        lo = bisect_left(times, start)
        hi = bisect_right(times, end)
        values.update(f.view_type.value for f in features[lo:hi])
        return sorted(values)

    def _resolve_overlaps(
        self,
        shots: list[ShotRecord],
        *,
        strict: bool = False,
        **_legacy_kwargs,
    ) -> list[ShotRecord]:
        """Resolve mutually impossible strikes without cutting shot footage.

        A genuine next cue strike cannot occur before every ball from the prior
        shot has stopped.  Such a candidate is therefore a collision, cushion
        impact, feathering artefact, or uncertain prior boundary.  In strict
        mode, two records also cannot occupy the same source time: the configured
        pre-roll and four-second minimum are part of the shot contract.  When
        two confirmed-looking events conflict, retain the better-supported one
        instead of blindly keeping the first; this removes opening preparation
        movements followed shortly by the real, audio-supported strike.
        """
        if not shots:
            return []
        ordered = sorted(shots, key=lambda s: s.cue_strike)
        resolved: list[ShotRecord] = []

        for shot in ordered:
            keep_current = True
            while resolved:
                prev = resolved[-1]
                prev_stop = float(
                    prev.evidence.get("uncapped_physical_stop_timestamp")
                    or getattr(prev, "physical_stop_timestamp", 0.0)
                    or prev.ball_motion_end
                )
                near_duplicate = abs(shot.cue_strike - prev.cue_strike) < 0.60
                mid_motion = shot.cue_strike <= prev_stop + 1e-6
                source_overlap = strict and shot.clip_start < prev.clip_end - 1e-6

                if not (near_duplicate or mid_motion or source_overlap):
                    break

                if self._prefer_later_conflicting_shot(prev, shot):
                    shot.evidence["replaced_conflicting_strike"] = prev.cue_strike
                    resolved.pop()
                    # Re-check the replacement against the shot before it. This
                    # matters for bursts such as valid -> false peak -> valid.
                    continue

                prev.manual_review_required = prev.manual_review_required or (
                    shot.shot_confidence >= self.high
                )
                evidence_key = (
                    "rejected_mid_motion_strike"
                    if mid_motion
                    else "rejected_overlapping_strike"
                )
                prev.evidence[evidence_key] = shot.cue_strike
                keep_current = False
                break

            if keep_current:
                resolved.append(shot)

        for i, shot in enumerate(resolved, start=1):
            shot.shot_id = i
            if strict:
                expected_start = self._strict_start(shot.cue_strike, float("inf"))
                shot.clip_start = expected_start
                physical = float(
                    getattr(shot, "physical_stop_timestamp", 0.0)
                    or shot.ball_motion_end
                )
                minimum_end = float(
                    shot.evidence.get("minimum_clip_end_timestamp") or physical
                )
                shot.clip_end = max(physical, minimum_end)
                shot.ball_motion_end = physical
                shot.cue_strike_timestamp = shot.cue_strike
                shot.clip_start_timestamp = shot.clip_start
                shot.clip_end_timestamp = shot.clip_end
        return resolved

    def _prefer_later_conflicting_shot(
        self,
        previous: ShotRecord,
        current: ShotRecord,
    ) -> bool:
        """Return whether a later incompatible candidate has stronger support.

        Audio remains supporting evidence only: both records have already passed
        visual strike confirmation.  It is used here only to break a confidence
        tie, which distinguishes a real cue impact from visually similar player
        preparation without allowing sound to create a shot by itself.
        """

        confidence_delta = current.shot_confidence - previous.shot_confidence
        if abs(confidence_delta) > self.conflict_confidence_margin:
            return confidence_delta > 0.0

        previous_audio = float(previous.evidence.get("audio_onset", 0.0) or 0.0)
        current_audio = float(current.evidence.get("audio_onset", 0.0) or 0.0)
        previous_has_audio = previous_audio >= self.audio_support_threshold
        current_has_audio = current_audio >= self.audio_support_threshold
        if current_has_audio != previous_has_audio:
            return current_has_audio

        previous_quiet = float(
            previous.evidence.get("pre_ball_quiet_ratio", 0.0) or 0.0
        )
        current_quiet = float(
            current.evidence.get("pre_ball_quiet_ratio", 0.0) or 0.0
        )
        if abs(current_quiet - previous_quiet) > 1e-6:
            return current_quiet > previous_quiet

        # Stable tie-break: preserve the earlier event. This is safer for an
        # actual shot followed by a cushion/collision peak with equal evidence.
        return False

    def recompute_durations(
        self,
        shots: list[ShotRecord],
        original: float,
    ) -> tuple[float, float]:
        edited = sum(s.duration() for s in shots if s.included)
        removed = max(0.0, original - edited)
        return edited, removed
