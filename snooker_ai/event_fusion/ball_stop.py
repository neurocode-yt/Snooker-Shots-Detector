"""Conservative, hysteretic all-ball stop detection.

The strict editing boundary is *not* the time at which stillness becomes
confirmed.  We look forward for ``confirmation_seconds`` and then return the
first stationary observation after the final valid moving observation.

Ball-normalised track kinematics are preferred whenever they are available.
The aggregate compensated residual remains a fallback for older analyses and
views where individual balls cannot be tracked reliably.  Camera cuts and
unobservable table frames are unknown observations; they can never prove that
the balls stopped.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.timebase import clamp

logger = get_logger("event_fusion.ball_stop")

_TABLE_VIEWS = {
    CameraViewType.MAIN_TABLE,
    CameraViewType.WIDE_ARENA,
    CameraViewType.BALL_CLOSEUP,
    CameraViewType.OTHER,  # scene classification is intentionally conservative
}


@dataclass(frozen=True)
class StopDetection:
    """Complete timing evidence for one shot's end."""

    motion_start: float
    last_ball_motion_timestamp: float
    physical_stop_timestamp: float
    stop_confirmation_timestamp: float
    end_confidence: float
    start_confidence: float
    confirmed: bool
    manual_review_required: bool = False
    reason: str = ""


class BallStopDetector:
    """Find the first physical all-ball stop after a confirmed cue strike.

    The detector uses separate start and stop thresholds.  Once moving, a ball
    remains moving until every trustworthy observation stays below the lower
    stop threshold for the full confirmation period. A practical duration cap is
    used as a review boundary when tracking is unresolved, so one bad track
    cannot stretch a shot to the end of a match.
    """

    def __init__(self, config: Config):
        bcfg = config.section("ball_stop")
        self.confirm_s = float(
            bcfg.get(
                "stationary_confirmation_seconds",
                bcfg.get("confirm_seconds", bcfg.get("settle_seconds", 0.5)),
            )
        )
        self.min_travel_s = float(bcfg.get("min_motion_seconds", 0.20))
        self.baseline_window = float(bcfg.get("baseline_window_seconds", 1.2))
        max_after = bcfg.get("max_seconds_after_strike", 10.0)
        self.max_after_strike = (
            float(max_after) if max_after is not None and float(max_after) > 0 else None
        )

        # Normalised speed is measured in ball diameters / second.
        self.speed_start = float(bcfg.get("motion_start_normalized_speed", 0.45))
        self.speed_stop = float(bcfg.get("motion_stop_normalized_speed", 0.16))
        if self.speed_start <= self.speed_stop:
            raise ValueError("ball_stop motion start threshold must exceed stop threshold")

        # Fallback activity thresholds for analyses without track kinematics.
        self.activity_start = float(bcfg.get("motion_start_activity", 0.18))
        self.activity_stop = float(bcfg.get("motion_stop_activity", 0.10))
        if self.activity_start <= self.activity_stop:
            raise ValueError("ball_stop activity start threshold must exceed stop threshold")

        self.baseline_start_factor = float(bcfg.get("baseline_start_factor", 2.4))
        self.baseline_stop_factor = float(bcfg.get("baseline_stop_factor", 1.65))
        self.stationary_tolerance = float(
            bcfg.get("stationary_tolerance_ball_diameters", 0.08)
        )
        self.unknown_review_s = float(bcfg.get("unknown_review_seconds", 0.35))
        self.min_motion_samples = int(bcfg.get("min_motion_samples", 2))
        self.stop_motion_reconfirm_samples = max(
            1, int(bcfg.get("stop_motion_reconfirm_samples", 2))
        )
        self.residual_stop = float(bcfg.get("ball_residual_stop", 0.10))
        self.residual_start = float(bcfg.get("ball_residual_start", 0.20))
        self.quiet_raw_threshold = float(bcfg.get("quiet_raw_threshold", 0.18))
        self.quiet_residual_max = float(bcfg.get("quiet_residual_max", 0.80))
        self.quiet_speed_max = float(bcfg.get("quiet_speed_max", 10.0))
        self.stale_occlusion_max_count = int(
            bcfg.get("stale_occlusion_max_count", 2)
        )
        self.stale_occlusion_quiet_s = float(
            bcfg.get("stale_occlusion_quiet_seconds", self.confirm_s)
        )

    # ------------------------------------------------------------------ evidence

    @staticmethod
    def _get_float(f: FrameFeatures, name: str, default: float = 0.0) -> float:
        value = getattr(f, name, default)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return float(default)

    def _is_valid_observation(self, f: FrameFeatures) -> bool:
        """Whether this frame is allowed to contribute stationary evidence."""
        if not bool(getattr(f, "observation_valid", True)):
            return False
        if not bool(getattr(f, "table_observable", True)):
            return False
        if f.scene_cut_score >= 0.5:
            return False
        if f.view_type in {
            CameraViewType.REPLAY,
            CameraViewType.SLOW_MOTION_REPLAY,
            CameraViewType.SCOREBOARD,
            CameraViewType.ADVERTISEMENT,
        }:
            return False
        if f.view_type not in _TABLE_VIEWS and f.table_confidence < 0.25:
            return False
        # A failed camera model is unknown, not proof of stillness.
        if f.camera_motion_magnitude > 12.0 and f.table_confidence < 0.45:
            return False
        return True

    def _normalised_speed(self, f: FrameFeatures) -> float:
        speed = self._get_float(f, "max_ball_normalized_speed")
        if speed > 0.0:
            return speed
        diameter = self._get_float(f, "ball_diameter_px")
        px_speed = self._get_float(f, "max_ball_speed")
        if diameter > 0.5 and px_speed > 0.0:
            return px_speed / diameter
        return 0.0

    def _has_track_evidence(self, f: FrameFeatures) -> bool:
        return bool(
            self._get_float(f, "ball_diameter_px") > 0.5
            and (
                int(getattr(f, "ball_count", 0) or 0) > 0
                or self._get_float(f, "cue_ball_track_confidence") > 0.0
            )
        )

    def _activity(self, f: FrameFeatures) -> float:
        """Instantaneous compensated activity in [0, 1].

        Track-local residual and normalised speed take precedence.  Aggregate
        table flow is used only as a compatibility fallback, because player and
        cue motion can otherwise dominate it.
        """
        speed = self._normalised_speed(f)
        local_residual = self._get_float(f, "ball_residual_motion")
        moving_count = int(getattr(f, "moving_ball_count", 0) or 0)

        if self._has_track_evidence(f):
            speed_score = float(np.clip(speed / max(self.speed_start * 2.5, 1e-6), 0, 1))
            residual_score = float(np.clip(local_residual, 0, 1))
            count_score = min(1.0, moving_count / 2.0)
            return float(max(speed_score, residual_score, count_score * 0.8))

        raw = self._get_float(f, "motion_raw", self._get_float(f, "motion_score"))
        if raw > 0.0:
            return float(np.clip(raw, 0, 1))
        mean_n = float(np.clip(f.residual_motion_mean / 2.2, 0, 1))
        max_n = float(np.clip(f.residual_motion_max / 6.0, 0, 1))
        area_n = float(np.clip(f.motion_area_ratio / 0.08, 0, 1))
        return float(np.clip(0.30 * mean_n + 0.50 * max_n + 0.20 * area_n, 0, 1))

    def _baseline(
        self,
        features: list[FrameFeatures],
        strike_t: float,
        times: list[float] | None = None,
    ) -> float:
        if times is None:
            times = [f.t for f in features]
        lo = bisect_left(times, strike_t - self.baseline_window)
        hi = bisect_left(times, strike_t - 0.08)
        values = [
            self._activity(f)
            for f in features[lo:hi]
            if self._is_valid_observation(f)
        ]
        if not values:
            return min(0.05, self.activity_stop * 0.5)
        arr = np.asarray(values, dtype=np.float64)
        # Median of the quieter half rejects feathering/player foreground.
        arr.sort()
        quiet = arr[: max(1, (len(arr) + 1) // 2)]
        return float(np.median(quiet))

    def _quiet_frame_override(self, f: FrameFeatures, speed: float) -> bool:
        """Ignore an isolated false track when global table evidence is quiet."""

        raw = self._get_float(f, "motion_raw", self._get_float(f, "motion_score"))
        residual_max = self._get_float(f, "residual_motion_max")
        return bool(
            raw <= self.quiet_raw_threshold
            and residual_max <= self.quiet_residual_max
            and speed <= self.quiet_speed_max
        )

    def _moving_evidence(
        self,
        f: FrameFeatures,
        *,
        already_moving: bool,
        baseline: float,
    ) -> bool:
        speed = self._normalised_speed(f)
        activity = self._activity(f)
        local_residual = self._get_float(f, "ball_residual_motion")
        moving_count = int(getattr(f, "moving_ball_count", 0) or 0)
        occluded_count = int(getattr(f, "occluded_ball_count", 0) or 0)

        # Once motion has been established, a quiet table/residual frame is
        # stronger than one isolated Hough identity or occlusion count. This
        # prevents player/rail detections from keeping an already-settled shot
        # open indefinitely.
        if (
            already_moving
            and self._quiet_frame_override(f, speed)
            and (occluded_count == 0 or occluded_count >= 8)
        ):
            return False

        if already_moving and occluded_count > 0:
            # A previously tracked moving ball hidden by a player/graphic cannot
            # be declared stationary while it remains unresolved.
            return True
        if moving_count > 0:
            return True

        if self._has_track_evidence(f):
            threshold = self.speed_stop if already_moving else self.speed_start
            if speed >= threshold:
                return True
            residual_threshold = self.residual_stop if already_moving else self.residual_start
            return local_residual >= residual_threshold

        if already_moving:
            threshold = max(self.activity_stop, baseline * self.baseline_stop_factor)
        else:
            threshold = max(self.activity_start, baseline * self.baseline_start_factor)
        return activity >= threshold

    # ------------------------------------------------------------------ main API

    def detect_stop(
        self,
        strike: StrikeCandidate,
        features: list[FrameFeatures],
        duration: float,
        times: list[float] | None = None,
    ) -> StopDetection:
        """Return physical stop and its later confirmation timestamp.

        If stillness cannot be confirmed, the practical duration cap is returned
        and the result is marked for manual review. This prevents a false track
        from creating a match-length clip while preserving the uncertainty.
        """
        duration = max(0.0, float(duration))
        strike_t = clamp(float(strike.timestamp), 0.0, duration)
        cap_t = (
            min(duration, strike_t + self.max_after_strike)
            if self.max_after_strike is not None
            else duration
        )
        if times is None:
            times = [f.t for f in features]
        lo = bisect_left(times, strike_t - 0.25)
        hi = bisect_right(times, cap_t + 1e-6)
        window = features[lo:hi]
        if not window:
            return StopDetection(
                motion_start=strike_t,
                last_ball_motion_timestamp=strike_t,
                physical_stop_timestamp=cap_t,
                stop_confirmation_timestamp=cap_t,
                end_confidence=0.10,
                start_confidence=0.20,
                confirmed=False,
                manual_review_required=True,
                reason="no_motion_observations",
            )

        baseline = self._baseline(features, strike_t, times)
        moving = False
        motion_run = 0
        motion_start = strike_t
        start_conf = 0.35
        last_motion = strike_t
        still_since: float | None = None
        resumed_motion_run = 0
        unknown_since: float | None = None
        unknown_total = 0.0
        quiet_occlusion_since: float | None = None
        stale_occlusion_cleared = False
        saw_ball_tracks = False
        valid_after_strike = 0

        for f in window:
            if f.t < strike_t:
                continue

            valid = self._is_valid_observation(f)
            if not valid:
                if unknown_since is None:
                    unknown_since = f.t
                # Unknown evidence breaks stillness confirmation.  It does not
                # end motion and does not fabricate a potted/stationary ball.
                still_since = None
                resumed_motion_run = 0
                continue

            if unknown_since is not None:
                unknown_total += max(0.0, f.t - unknown_since)
                unknown_since = None

            valid_after_strike += 1
            saw_ball_tracks = saw_ball_tracks or self._has_track_evidence(f)
            is_moving = self._moving_evidence(f, already_moving=moving, baseline=baseline)

            # Player/cue edges can leave one or two pseudo-ball tracks marked as
            # occluded after every real ball has settled.  A short occlusion is
            # still held open (a genuinely rolling ball may be hidden), but a
            # persistent small count is considered stale when the whole table,
            # camera-compensated residual, and measured speed are continuously
            # quiet.  Preserve the first quiet time as the physical stop; the
            # following interval is confirmation look-ahead only.
            occluded_count = int(getattr(f, "occluded_ball_count", 0) or 0)
            quiet_small_occlusion = bool(
                moving
                and 0 < occluded_count <= self.stale_occlusion_max_count
                and int(getattr(f, "moving_ball_count", 0) or 0) == 0
                and self._quiet_frame_override(f, self._normalised_speed(f))
            )
            if quiet_small_occlusion:
                if quiet_occlusion_since is None:
                    quiet_occlusion_since = f.t
                if (
                    f.t - quiet_occlusion_since + 1e-9
                    >= self.stale_occlusion_quiet_s
                ):
                    is_moving = False
                    stale_occlusion_cleared = True
            else:
                quiet_occlusion_since = None

            if not moving:
                if is_moving:
                    motion_run += 1
                    if motion_run == 1:
                        motion_start = max(strike_t, f.t)
                    if motion_run >= self.min_motion_samples:
                        moving = True
                        start_conf = 0.92 if self._has_track_evidence(f) else 0.72
                        last_motion = f.t
                        still_since = None
                else:
                    motion_run = 0
                continue

            if is_moving:
                if still_since is None:
                    last_motion = f.t
                    resumed_motion_run = 0
                    continue

                # Once valid stillness has started, require renewed motion to
                # persist across adjacent observations before reopening the
                # shot.  A referee/player edge or one-frame Hough identity
                # jump otherwise resets confirmation and can make the prior
                # shot absorb an entire respot.  Genuine rolling motion lasts
                # for multiple video frames and still reopens immediately on
                # the configured second sample.
                resumed_motion_run += 1
                if resumed_motion_run >= self.stop_motion_reconfirm_samples:
                    last_motion = f.t
                    still_since = None
                    resumed_motion_run = 0
                continue

            resumed_motion_run = 0
            if f.t < strike_t + self.min_travel_s:
                continue
            if still_since is None:
                # This is the first valid stationary frame after final motion.
                still_since = (
                    quiet_occlusion_since
                    if stale_occlusion_cleared and quiet_occlusion_since is not None
                    else f.t
                )
            if f.t - still_since + 1e-9 < self.confirm_s:
                continue

            physical_stop = clamp(still_since, strike_t, duration)
            confirmation = clamp(f.t, physical_stop, duration)
            confidence = 0.95 if saw_ball_tracks else 0.82
            if stale_occlusion_cleared:
                confidence = min(confidence, 0.68)
            if unknown_total > 0:
                confidence -= min(0.25, unknown_total * 0.08)
            if valid_after_strike < 4:
                confidence = min(confidence, 0.55)
            review = confidence < 0.70 or unknown_total >= self.unknown_review_s
            result = StopDetection(
                motion_start=motion_start,
                last_ball_motion_timestamp=clamp(last_motion, strike_t, physical_stop),
                physical_stop_timestamp=physical_stop,
                stop_confirmation_timestamp=confirmation,
                end_confidence=float(np.clip(confidence, 0, 1)),
                start_confidence=start_conf,
                confirmed=True,
                manual_review_required=review,
                reason=(
                    "confirmed_stale_occlusion_override"
                    if stale_occlusion_cleared
                    else (
                        "confirmed_after_unknown_gap"
                        if unknown_total
                        else "confirmed_stationary"
                    )
                ),
            )
            logger.debug(
                "ball_stop strike=%.3f last=%.3f stop=%.3f confirm=%.3f conf=%.2f",
                strike_t,
                result.last_ball_motion_timestamp,
                result.physical_stop_timestamp,
                result.stop_confirmation_timestamp,
                result.end_confidence,
            )
            return result

        # No confirmed stop. Keep the shot through the practical bound and mark
        # the boundary for review; a false track must not consume the source.
        if cap_t < duration - 1e-6:
            return StopDetection(
                motion_start=motion_start if moving else strike_t,
                last_ball_motion_timestamp=clamp(last_motion, strike_t, cap_t),
                physical_stop_timestamp=cap_t,
                stop_confirmation_timestamp=cap_t,
                end_confidence=0.20,
                start_confidence=start_conf,
                confirmed=False,
                manual_review_required=True,
                reason="max_duration_review_cap",
            )

        reason = "motion_not_confirmed" if moving else "no_sustained_ball_motion"
        return StopDetection(
            motion_start=motion_start if moving else strike_t,
            last_ball_motion_timestamp=clamp(last_motion, strike_t, cap_t),
            physical_stop_timestamp=cap_t,
            stop_confirmation_timestamp=cap_t,
            end_confidence=0.20 if moving else 0.12,
            start_confidence=start_conf,
            confirmed=False,
            manual_review_required=True,
            reason=reason,
        )

    def find_motion_window(
        self,
        strike: StrikeCandidate,
        features: list[FrameFeatures],
        duration: float,
    ) -> tuple[float, float, float, float]:
        """Backward-compatible tuple API.

        The second value is now the physical stop timestamp, never a padded or
        confirmation timestamp.
        """
        result = self.detect_stop(strike, features, duration)
        return (
            result.motion_start,
            result.physical_stop_timestamp,
            result.end_confidence,
            result.start_confidence,
        )
