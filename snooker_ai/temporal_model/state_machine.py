"""Rule-based temporal state machine for the strict shot contract.

The machine is deliberately conservative about declaring a stop: only valid,
observable table frames with no unresolved occlusions advance the stationary
confirmation clock.  Camera cuts and invalid observations preserve an active
shot and reset that clock.
"""

from __future__ import annotations

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType, FrameFeatures, ShotState
from snooker_ai.utils.logging import get_logger

logger = get_logger("temporal")

TABLE_VIEWS = {
    CameraViewType.MAIN_TABLE,
    CameraViewType.WIDE_ARENA,
    CameraViewType.BALL_CLOSEUP,
}

ACTIVE_STATES = {
    ShotState.STRIKE_CANDIDATE,
    ShotState.STRIKE_CONFIRMED,
    ShotState.BALLS_MOVING,
    ShotState.BALLS_SETTLING,
}


class ShotStateMachine:
    """Label samples with the canonical strict-boundary state sequence.

    Canonical sequence::

        WAITING -> CUEING -> STRIKE_CANDIDATE -> STRIKE_CONFIRMED
        -> BALLS_MOVING -> BALLS_SETTLING -> ALL_BALLS_STOPPED

    Ball-diameter-normalized tracker signals are preferred.  The old residual
    ``motion_score`` remains a fallback so Phase-1 feature files and producers
    continue to work.
    """

    def __init__(self, config: Config):
        mcfg = config.section("motion")
        bcfg = config.section("ball_stop")
        scfg = config.section("strike_fusion")
        cut_cfg = config.section("scene_detection")

        self.motion_start = float(mcfg.get("normalized_motion_start_threshold", 0.18))
        self.motion_stop = float(mcfg.get("normalized_motion_stop_threshold", 0.08))
        if self.motion_start <= self.motion_stop:
            raise ValueError(
                "motion.normalized_motion_start_threshold must be greater than "
                "motion.normalized_motion_stop_threshold"
            )

        self.stop_confirmation_s = float(bcfg.get("confirm_seconds", 0.50))
        self.min_strike = float(scfg.get("min_confidence", 0.35))
        self.stationary_s = float(scfg.get("cue_ball_stationary_seconds", 0.30))
        self.stationary_speed = float(
            scfg.get("cue_ball_stationary_normalized_speed", self.motion_stop)
        )
        self.sustained_motion_s = float(scfg.get("cue_ball_sustained_motion_seconds", 0.12))
        self.min_track_confidence = float(scfg.get("cue_ball_min_track_confidence", 0.45))
        self.candidate_timeout_s = float(scfg.get("strike_candidate_timeout_seconds", 0.75))
        self.pre_quiet_motion = float(scfg.get("pre_strike_quiet_max_motion", 0.28))
        self.hard_cut_threshold = float(cut_cfg.get("hard_cut_threshold", 0.42))

        # Legacy residual-motion hysteresis used only when normalized ball
        # observations are unavailable.
        self.legacy_motion_start = float(mcfg.get("state_motion_start_threshold", 0.35))
        self.legacy_motion_stop = float(mcfg.get("state_motion_stop_threshold", 0.20))

    @staticmethod
    def _tableish(feature: FrameFeatures) -> bool:
        return feature.view_type in TABLE_VIEWS or feature.table_confidence >= 0.30

    def _valid_observation(self, feature: FrameFeatures) -> bool:
        replay_or_graphics = feature.view_type in {
            CameraViewType.REPLAY,
            CameraViewType.SLOW_MOTION_REPLAY,
            CameraViewType.SCOREBOARD,
            CameraViewType.ADVERTISEMENT,
        }
        return (
            feature.observation_valid
            and feature.table_observable
            and self._tableish(feature)
            and feature.scene_cut_score < self.hard_cut_threshold
            and not replay_or_graphics
        )

    @staticmethod
    def _has_ball_observation(feature: FrameFeatures) -> bool:
        return (
            feature.ball_diameter_px > 0.0
            or feature.cue_ball_track_confidence > 0.0
            or feature.cue_ball_x is not None
            or feature.cue_ball_y is not None
            or feature.cue_ball_normalized_speed > 0.0
            or feature.max_ball_normalized_speed > 0.0
            or feature.moving_ball_count > 0
            or feature.occluded_ball_count > 0
            or feature.ball_residual_motion > 0.0
        )

    def _moving(self, feature: FrameFeatures, *, already_moving: bool) -> bool:
        """Apply normalized start/stop hysteresis, with a legacy fallback."""

        if self._has_ball_observation(feature):
            threshold = self.motion_stop if already_moving else self.motion_start
            speed = max(
                feature.cue_ball_normalized_speed,
                feature.max_ball_normalized_speed,
                feature.ball_residual_motion,
            )
            return feature.moving_ball_count > 0 or speed >= threshold

        threshold = self.legacy_motion_stop if already_moving else self.legacy_motion_start
        apparent_camera_motion = (
            feature.camera_motion_magnitude > 6.0 and feature.residual_motion_mean < 1.0
        )
        return feature.motion_score >= threshold and not apparent_camera_motion

    def _cue_ball_stationary(self, feature: FrameFeatures) -> bool:
        if self._has_ball_observation(feature):
            track_is_usable = (
                feature.cue_ball_track_confidence >= self.min_track_confidence
                or feature.cue_ball_detected
            )
            return (
                track_is_usable
                and feature.cue_ball_normalized_speed <= self.stationary_speed
                and feature.moving_ball_count == 0
            )
        return feature.motion_score <= self.pre_quiet_motion

    def label(self, features: list[FrameFeatures]) -> list[FrameFeatures]:
        if not features:
            return features

        state = ShotState.WAITING
        quiet_since: float | None = None
        candidate_since: float | None = None
        movement_since: float | None = None
        still_since: float | None = None

        for feature in features:
            valid = self._valid_observation(feature)

            # An invalid/cut/occluded frame can never prove stillness.  Keep an
            # active shot open and require a fresh continuous confirmation run.
            if not valid or feature.occluded_ball_count > 0:
                if state == ShotState.BALLS_SETTLING:
                    still_since = None
                if state not in ACTIVE_STATES:
                    state = ShotState.WAITING
                    quiet_since = None
                    candidate_since = None
                    movement_since = None
                feature.state = state
                continue

            moving_from_rest = self._moving(feature, already_moving=False)
            moving_with_hysteresis = self._moving(feature, already_moving=True)
            cue_stationary = self._cue_ball_stationary(feature)

            if state == ShotState.ALL_BALLS_STOPPED:
                # Keep the terminal state observable for one sample, then arm
                # the next shot from a clean WAITING state.
                state = ShotState.WAITING
                quiet_since = feature.t if cue_stationary else None
                candidate_since = None
                movement_since = None
                still_since = None

            elif state == ShotState.WAITING:
                if cue_stationary:
                    if quiet_since is None:
                        quiet_since = feature.t
                    elif feature.t - quiet_since >= self.stationary_s:
                        state = ShotState.CUEING
                else:
                    quiet_since = None

            elif state == ShotState.CUEING:
                quiet_ready = (
                    quiet_since is not None and feature.t - quiet_since >= self.stationary_s
                )
                if feature.strike_score >= self.min_strike and quiet_ready:
                    state = ShotState.STRIKE_CANDIDATE
                    candidate_since = feature.t
                    movement_since = feature.t if moving_from_rest else None
                elif not cue_stationary and not moving_from_rest:
                    state = ShotState.WAITING
                    quiet_since = None

            elif state == ShotState.STRIKE_CANDIDATE:
                if moving_from_rest:
                    if movement_since is None:
                        movement_since = feature.t
                    elif feature.t - movement_since >= self.sustained_motion_s:
                        state = ShotState.STRIKE_CONFIRMED
                else:
                    movement_since = None
                    if (
                        candidate_since is not None
                        and feature.t - candidate_since > self.candidate_timeout_s
                    ):
                        state = ShotState.CUEING if cue_stationary else ShotState.WAITING
                        candidate_since = None

            elif state == ShotState.STRIKE_CONFIRMED:
                if moving_with_hysteresis or moving_from_rest:
                    state = ShotState.BALLS_MOVING
                elif (
                    candidate_since is not None
                    and feature.t - candidate_since > self.candidate_timeout_s
                ):
                    state = ShotState.WAITING

            elif state == ShotState.BALLS_MOVING:
                if not moving_with_hysteresis:
                    state = ShotState.BALLS_SETTLING
                    still_since = feature.t

            elif state == ShotState.BALLS_SETTLING:
                if moving_with_hysteresis:
                    state = ShotState.BALLS_MOVING
                    still_since = None
                elif still_since is None:
                    still_since = feature.t
                elif feature.t - still_since >= self.stop_confirmation_s:
                    state = ShotState.ALL_BALLS_STOPPED

            feature.state = state

        logger.info("State machine labelled %d frames", len(features))
        return features

    def _smooth(self, features: list[FrameFeatures]) -> list[FrameFeatures]:
        """Compatibility hook retained for callers of the old implementation.

        Contract states are intentionally not majority-smoothed: doing so could
        turn an invalid observation into evidence that all balls stopped.
        """

        return features
