"""Cue-strike detection from a stationary-to-moving cue-ball transition.

Generic table motion and audio are useful proposal signals, but neither may
confirm a strike when cue-ball kinematics are available.  A confirmed event
requires a previously stationary white ball followed immediately by sustained
white-ball motion.  Cue/contact audio is deliberately supporting evidence only.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.logging import get_logger

logger = get_logger("event_fusion.strike")

TABLE_VIEWS = {
    CameraViewType.MAIN_TABLE,
    CameraViewType.WIDE_ARENA,
    CameraViewType.BALL_CLOSEUP,
    CameraViewType.OTHER,
}


class StrikeDetector:
    def __init__(self, config: Config):
        cfg = config.section("strike_fusion")
        self.w_motion = float(cfg.get("residual_motion_onset", 0.24))
        self.w_audio = float(cfg.get("audio_transient", 0.10))
        self.w_view = float(cfg.get("table_view_confidence", 0.10))
        self.w_accel = float(cfg.get("cue_ball_acceleration_weight", 0.34))
        self.w_sustained = float(cfg.get("cue_ball_sustained_weight", 0.22))
        self.min_conf = float(cfg.get("min_confidence", 0.48))
        self.min_dist = float(cfg.get("candidate_peak_min_distance_seconds", 1.2))
        self.refine_r = float(cfg.get("refine_search_radius_seconds", 0.65))
        self.pre_quiet_s = float(cfg.get("pre_strike_quiet_seconds", 0.45))
        self.post_motion_s = float(cfg.get("post_strike_motion_seconds", 0.30))
        self.stationary_speed = float(cfg.get("cue_ball_stationary_normalized_speed", 0.12))
        self.start_speed = float(cfg.get("cue_ball_motion_start_normalized_speed", 0.45))
        self.continue_speed = float(cfg.get("cue_ball_motion_continue_normalized_speed", 0.18))
        self.min_sustained_frames = int(cfg.get("cue_ball_min_sustained_frames", 2))
        self.min_track_conf = float(cfg.get("cue_ball_min_track_confidence", 0.35))
        self.pre_quiet_max_motion = float(cfg.get("pre_strike_quiet_max_motion", 0.28))
        self.contact_pre_quiet_min_ratio = float(
            cfg.get("cue_contact_pre_strike_quiet_min_ratio", 0.70)
        )
        self.pre_quiet_max_ball_speed = float(
            cfg.get("pre_strike_quiet_max_ball_speed", 5.0)
        )
        self.pre_ball_quiet_min_ratio = float(
            cfg.get("pre_strike_ball_quiet_min_ratio", 0.60)
        )
        self.fallback_pre_ball_quiet_min_ratio = float(
            cfg.get("fallback_pre_strike_ball_quiet_min_ratio", 0.50)
        )
        self.cue_contact_noise_override = float(
            cfg.get("cue_contact_noise_override_score", 0.70)
        )
        self.contact_bridge_max_gap_frames = int(
            cfg.get("cue_contact_bridge_max_gap_frames", 1)
        )
        self.contact_bridge_min_peak_speed = float(
            cfg.get("cue_contact_bridge_min_peak_normalized_speed", 2.0)
        )
        self.occlusion_jump_diameters = float(
            cfg.get("occlusion_identity_jump_ball_diameters", 1.50)
        )
        self.local_norm_window = int(cfg.get("local_norm_window_frames", 40))
        self.allow_legacy_fallback = bool(cfg.get("allow_legacy_motion_fallback", True))
        self.sparse_pre_quiet_s = float(
            cfg.get("sparse_candidate_pre_quiet_seconds", 1.5)
        )
        self.sparse_post_s = float(cfg.get("sparse_candidate_post_seconds", 1.0))
        self.sparse_activity_threshold = float(
            cfg.get("sparse_candidate_activity_threshold", 0.35)
        )
        self.sparse_ball_threshold = float(
            cfg.get("sparse_candidate_ball_activity_threshold", 0.45)
        )
        self.sparse_min_active = int(cfg.get("sparse_candidate_min_active_samples", 2))
        self.sparse_gap_s = float(cfg.get("sparse_candidate_gap_seconds", self.min_dist))
        audio_cfg = config.section("audio")
        self.max_audio_weight = float(audio_cfg.get("max_audio_weight", 0.25))

    # ------------------------------------------------------------------ utilities

    @staticmethod
    def _value(f: FrameFeatures, name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(f, name, default) or 0.0)
        except (TypeError, ValueError):
            return default

    def _cue_speed(self, f: FrameFeatures) -> float:
        value = self._value(f, "cue_ball_normalized_speed")
        if value > 0:
            return value
        px = self._value(f, "cue_ball_speed")
        diameter = self._value(f, "ball_diameter_px")
        return px / diameter if px > 0 and diameter > 0.5 else 0.0

    def _cue_accel(self, f: FrameFeatures) -> float:
        return max(0.0, self._value(f, "cue_ball_acceleration"))

    def _track_conf(self, f: FrameFeatures) -> float:
        return self._value(f, "cue_ball_track_confidence")

    @staticmethod
    def _valid(f: FrameFeatures) -> bool:
        if not bool(getattr(f, "observation_valid", True)):
            return False
        if not bool(getattr(f, "table_observable", True)):
            return False
        if f.scene_cut_score >= 0.5:
            return False
        if f.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY):
            return False
        return f.table_confidence >= 0.18 or f.view_type in TABLE_VIEWS

    def _has_cue_kinematics(self, features: list[FrameFeatures]) -> bool:
        reliable = sum(
            1
            for f in features
            if self._track_conf(f) >= self.min_track_conf
            and self._value(f, "ball_diameter_px") > 0.5
        )
        return reliable >= min(3, max(1, len(features) // 20))

    @staticmethod
    def _local_normalize(arr: np.ndarray, window: int) -> np.ndarray:
        if len(arr) == 0:
            return arr
        result = np.zeros_like(arr, dtype=np.float64)
        half = max(1, window // 2)
        for i in range(len(arr)):
            lo, hi = max(0, i - half), min(len(arr), i + half + 1)
            scale = float(np.percentile(arr[lo:hi], 95))
            if scale > 1e-8:
                result[i] = min(1.0, float(arr[i]) / scale)
        return result

    def _transition_metrics(
        self,
        features: list[FrameFeatures],
        idx: int,
        times: list[float] | None = None,
    ) -> dict[str, float]:
        f = features[idx]
        t = f.t
        # These windows used to be built by scanning the complete feature list
        # for every frame.  A full match therefore did O(n^2) timestamp checks
        # during both scoring and candidate detection.  Features are chronological,
        # so binary-searching the two short windows preserves the exact members
        # while making the pass effectively linear.
        if times is None:
            times = [x.t for x in features]
        pre_lo = bisect_left(times, t - self.pre_quiet_s)
        pre_hi = bisect_left(times, t - 0.03)
        post_lo = bisect_left(times, t)
        post_hi = bisect_right(times, t + self.post_motion_s)
        pre = [
            x
            for x in features[pre_lo:pre_hi]
            if self._valid(x)
            and self._track_conf(x) >= self.min_track_conf * 0.7
        ]
        post = [
            x
            for x in features[post_lo:post_hi]
            if self._valid(x)
            and self._track_conf(x) >= self.min_track_conf * 0.7
        ]

        pre_speeds = [self._cue_speed(x) for x in pre]
        post_speeds = [self._cue_speed(x) for x in post]
        pre_raw = [
            self._value(x, "motion_raw", self._value(x, "motion_score"))
            for x in pre
        ]
        pre_quiet_ratio = (
            sum(v <= self.pre_quiet_max_motion for v in pre_raw) / len(pre_raw)
            if pre_raw
            else 0.0
        )
        pre_raw_median = float(np.median(pre_raw)) if pre_raw else 1.0
        pre_ball_speeds = [
            self._value(x, "max_ball_normalized_speed") for x in pre
        ]
        pre_ball_quiet_ratio = (
            sum(v <= self.pre_quiet_max_ball_speed for v in pre_ball_speeds)
            / len(pre_ball_speeds)
            if pre_ball_speeds
            else 0.0
        )
        pre_ball_speed_median = (
            float(np.median(pre_ball_speeds)) if pre_ball_speeds else 1.0
        )
        stationary_ratio = (
            sum(s <= self.stationary_speed for s in pre_speeds) / len(pre_speeds)
            if pre_speeds
            else 0.0
        )
        sustained_count = sum(s >= self.continue_speed for s in post_speeds)
        sustained_run = 0
        for speed in post_speeds:
            if speed >= self.continue_speed:
                sustained_run += 1
            else:
                break
        # At acute broadcast angles the cue/player can cover the white ball for
        # one sampled frame exactly at impact.  Keep a second run count that may
        # bridge that short tracker hole; confirmation below only permits it
        # when strong cue-at-ball contact and a fast subsequent launch agree.
        bridged_sustained_count = 0
        gap_count = 0
        for speed in post_speeds:
            if speed >= self.continue_speed:
                bridged_sustained_count += 1
                continue
            gap_count += 1
            if gap_count > self.contact_bridge_max_gap_frames:
                break
        sustained_ratio = (
            min(1.0, sustained_count / max(1, self.min_sustained_frames))
            if post_speeds
            else 0.0
        )
        current_speed = self._cue_speed(f)
        previous_speed = self._cue_speed(features[idx - 1]) if idx > 0 else 0.0
        accel = max(
            self._cue_accel(f),
            (current_speed - previous_speed)
            / max(1e-3, f.t - features[idx - 1].t)
            if idx > 0
            else 0.0,
        )
        crossing = float(
            current_speed >= self.start_speed
            and previous_speed < self.start_speed
        )
        track_conf = min(
            1.0,
            max([self._track_conf(x) for x in post] or [self._track_conf(f)]),
        )
        tip_scores = [self._value(x, "cue_contact_score") for x in post]
        tip_visible = sum(1 for x in post if bool(getattr(x, "cue_tip_visible", False)))
        cue_contact = max(tip_scores or [0.0])
        cue_approach = max(
            [self._value(x, "cue_approach_speed") for x in post] or [0.0]
        )
        # A real cue-ball launch has a coherent displacement over consecutive
        # observations.  One-frame Hough identity jumps caused by a walking
        # player often have a large apparent speed but immediately reverse or
        # disappear; they fail this direction/displacement check.
        points = [
            (self._value(x, "cue_ball_x", 0.0), self._value(x, "cue_ball_y", 0.0))
            for x in post
            if getattr(x, "cue_ball_x", None) is not None
            and getattr(x, "cue_ball_y", None) is not None
        ]
        direction_consistency = 0.0
        displacement = 0.0
        if len(points) >= 2:
            vectors = np.diff(np.asarray(points, dtype=np.float64), axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            total = float(np.sum(lengths))
            displacement = float(np.linalg.norm(np.asarray(points[-1]) - np.asarray(points[0])))
            if total > 1e-6:
                direction_consistency = float(np.clip(displacement / total, 0, 1))
        diameter = max(self._value(f, "ball_diameter_px"), 1.0)
        return {
            "stationary_ratio": float(stationary_ratio),
            "sustained_ratio": float(sustained_ratio),
            "sustained_count": float(sustained_count),
            "sustained_run": float(sustained_run),
            "bridged_sustained_count": float(bridged_sustained_count),
            "post_peak_cue_speed": float(max(post_speeds or [0.0])),
            "cue_speed": float(current_speed),
            "previous_cue_speed": float(previous_speed),
            "cue_acceleration": float(max(0.0, accel)),
            "speed_crossing": crossing,
            "track_confidence": float(track_conf),
            "cue_displacement_diameters": float(displacement / diameter),
            "cue_direction_consistency": float(direction_consistency),
            "pre_sample_count": float(len(pre)),
            "pre_motion_quiet_ratio": float(pre_quiet_ratio),
            "pre_motion_raw_median": pre_raw_median,
            "pre_ball_quiet_ratio": float(pre_ball_quiet_ratio),
            "pre_ball_speed_median": pre_ball_speed_median,
            "cue_tip_visible_count": float(tip_visible),
            "cue_contact_score": float(cue_contact),
            "cue_approach_speed": float(cue_approach),
            "cue_geometry_confirmed": float(
                cue_contact >= 0.20 or cue_approach >= 0.15
            ),
        }

    def _transition_confirmed(self, metrics: dict[str, float]) -> bool:
        object_tracks_quiet = bool(
            metrics["pre_ball_quiet_ratio"] >= self.pre_ball_quiet_min_ratio
            and metrics["pre_ball_speed_median"] <= self.pre_quiet_max_ball_speed
        )
        # A high-quality cue-at-ball contact plus a stationary white ball is
        # stronger evidence than heuristic object tracks.  This branch recovers
        # real launches when player/cue edges briefly create false ball centres.
        contact_overrides_object_noise = bool(
            metrics["cue_geometry_confirmed"] >= 0.5
            and metrics["cue_contact_score"] >= self.cue_contact_noise_override
        )
        # A player starting the cue action can contaminate one coarse residual
        # sample immediately before impact.  Strong cue-at-ball geometry plus a
        # verified white-ball launch may tolerate that isolated foreground
        # sample; ball handling/refereeing without cue contact cannot use this
        # exception.  The quiet median gate below still has to pass.
        pre_motion_quiet = bool(
            metrics["pre_motion_quiet_ratio"] >= 0.80
            or (
                contact_overrides_object_noise
                and metrics["pre_motion_quiet_ratio"]
                >= self.contact_pre_quiet_min_ratio
            )
        )
        uninterrupted_launch = bool(
            metrics["sustained_run"] >= self.min_sustained_frames
        )
        contact_bridged_launch = bool(
            contact_overrides_object_noise
            and metrics["bridged_sustained_count"] >= self.min_sustained_frames
            and metrics["post_peak_cue_speed"] >= self.contact_bridge_min_peak_speed
        )
        return bool(
            metrics["stationary_ratio"] >= 0.70
            and metrics["pre_sample_count"] >= 3
            and pre_motion_quiet
            and metrics["pre_motion_raw_median"] <= self.pre_quiet_max_motion
            # Object-ball Hough tracks can produce an isolated speed spike while
            # the cloth and the real cue ball are visibly still.  Do not let one
            # such identity jump suppress an otherwise complete cue-contact +
            # white-ball-launch sequence.
            and (object_tracks_quiet or contact_overrides_object_noise)
            and metrics["speed_crossing"] > 0
            and (uninterrupted_launch or contact_bridged_launch)
            and metrics["cue_displacement_diameters"] >= 0.25
            and metrics["cue_direction_consistency"] >= 0.45
            and metrics["track_confidence"] >= self.min_track_conf
        )

    def _sparse_dense_transition_confirmed(self, metrics: dict[str, float]) -> bool:
        """Relaxed native-rate confirmation for a 2fps proposal.

        A sparse sample can enter the cue launch after the first frame of the
        transition, making the strict stationary-ratio/crossing gates fail even
        though the dense trajectory is coherent.  Require a quiet raw-motion
        median, a strong sustained launch, displacement, and direction so this
        fallback cannot turn a generic residual spike into a shot.
        """
        return bool(
            metrics.get("pre_sample_count", 0.0) >= 3
            and metrics.get("pre_motion_raw_median", 1.0)
            <= self.pre_quiet_max_motion
            and metrics.get("post_peak_cue_speed", 0.0)
            >= max(3.0, self.start_speed * 2.5)
            and metrics.get("sustained_run", 0.0) >= self.min_sustained_frames
            and metrics.get("cue_displacement_diameters", 0.0) >= 0.75
            and metrics.get("cue_direction_consistency", 0.0) >= 0.35
            and metrics.get("track_confidence", 0.0) >= self.min_track_conf * 0.80
        )

    def _ball_onset_metrics(
        self,
        features: list[FrameFeatures],
        idx: int,
        times: list[float] | None = None,
    ) -> dict[str, float]:
        """Conservative visual fallback when the cue ball is briefly hidden.

        This path is deliberately stricter than the legacy table-motion fallback:
        a quiet pre-window must be followed by a sustained, ball-scale onset.  It
        can lower confidence and request review, but audio or a single residual
        spike can never create the event alone.
        """

        f = features[idx]
        t = f.t
        if times is None:
            times = [x.t for x in features]
        pre_lo = bisect_left(times, t - self.pre_quiet_s)
        pre_hi = bisect_left(times, t - 0.03)
        post_lo = bisect_left(times, t)
        post_hi = bisect_right(times, t + max(self.post_motion_s, 0.20))
        pre = [
            x
            for x in features[pre_lo:pre_hi]
            if self._valid(x)
        ]
        post = [
            x
            for x in features[post_lo:post_hi]
            if self._valid(x)
        ]
        raw = [self._value(x, "motion_raw", self._value(x, "motion_score")) for x in pre]
        pre_quiet = (
            sum(v <= self.pre_quiet_max_motion for v in raw) / len(raw)
            if raw
            else 0.0
        )
        pre_ball_speeds = [self._value(x, "max_ball_normalized_speed") for x in pre]
        pre_ball_quiet = (
            sum(v <= self.pre_quiet_max_ball_speed for v in pre_ball_speeds)
            / len(pre_ball_speeds)
            if pre_ball_speeds
            else 0.0
        )
        post_raw = [
            self._value(x, "motion_raw", self._value(x, "motion_score")) for x in post
        ]
        post_norm = [self._value(x, "max_ball_normalized_speed") for x in post]
        post_local = [self._value(x, "ball_residual_motion") for x in post]
        qualifying = [
            r >= 0.22 and (n >= 1.0 or local >= 0.35)
            for r, n, local in zip(post_raw, post_norm, post_local)
        ]
        run = 0
        for ok in qualifying:
            if ok:
                run += 1
            else:
                break

        # This fallback exists only for an impact-time cue-ball occlusion or
        # tracker identity break.  If a reliable white-ball observation remains
        # fixed while a colour is picked up or respotted, that is explicitly not
        # a cue strike.  Compare post-onset observations with the last reliable
        # pre-onset white-ball position to distinguish the two cases.
        reliable_pre = [
            x
            for x in pre
            if bool(getattr(x, "cue_ball_detected", False))
            and self._track_conf(x) >= self.min_track_conf
            and getattr(x, "cue_ball_x", None) is not None
            and getattr(x, "cue_ball_y", None) is not None
        ]
        pre_cue_stationary_ratio = (
            sum(self._cue_speed(x) <= self.stationary_speed for x in reliable_pre)
            / len(reliable_pre)
            if reliable_pre
            else 0.0
        )
        post_after_onset = [x for x in post if x.t > t + 1e-6]
        missing_after_onset = any(
            not bool(getattr(x, "cue_ball_detected", False))
            or self._track_conf(x) < self.min_track_conf * 0.70
            for x in post_after_onset
        )
        cue_identity_jump = False
        if reliable_pre:
            anchor = reliable_pre[-1]
            ax = self._value(anchor, "cue_ball_x")
            ay = self._value(anchor, "cue_ball_y")
            diameter = max(self._value(anchor, "ball_diameter_px"), 1.0)
            for x in post_after_onset:
                if (
                    bool(getattr(x, "cue_ball_detected", False))
                    and getattr(x, "cue_ball_x", None) is not None
                    and getattr(x, "cue_ball_y", None) is not None
                ):
                    displacement = float(
                        np.hypot(
                            self._value(x, "cue_ball_x") - ax,
                            self._value(x, "cue_ball_y") - ay,
                        )
                        / diameter
                    )
                    if displacement >= self.occlusion_jump_diameters:
                        cue_identity_jump = True
                        break
        return {
            "pre_motion_quiet_ratio": float(pre_quiet),
            "pre_motion_raw_median": float(np.median(raw)) if raw else 1.0,
            "pre_ball_quiet_ratio": float(pre_ball_quiet),
            "pre_ball_speed_median": (
                float(np.median(pre_ball_speeds)) if pre_ball_speeds else 1.0
            ),
            "ball_onset_run": float(run),
            "ball_onset_raw": float(max(post_raw or [0.0])),
            "ball_onset_normalized_speed": float(max(post_norm or [0.0])),
            "ball_onset_local_residual": float(max(post_local or [0.0])),
            "pre_cue_sample_count": float(len(reliable_pre)),
            "pre_cue_stationary_ratio": float(pre_cue_stationary_ratio),
            "cue_missing_after_onset": float(missing_after_onset),
            "cue_identity_jump": float(cue_identity_jump),
            "cue_observation_disrupted": float(
                missing_after_onset or cue_identity_jump
            ),
        }

    def _ball_onset_confirmed(self, metrics: dict[str, float]) -> bool:
        return bool(
            metrics["pre_motion_quiet_ratio"] >= 0.80
            and metrics["pre_motion_raw_median"] <= self.pre_quiet_max_motion
            and metrics["pre_ball_quiet_ratio"]
            >= self.fallback_pre_ball_quiet_min_ratio
            and metrics["pre_ball_speed_median"] <= self.pre_quiet_max_ball_speed
            and metrics["pre_cue_sample_count"] >= 2
            and metrics["pre_cue_stationary_ratio"] >= 0.70
            and metrics["cue_observation_disrupted"] >= 0.5
            and metrics["ball_onset_run"] >= 2
            and metrics["ball_onset_raw"] >= 0.22
            and metrics["ball_onset_normalized_speed"] >= 1.0
        )

    # ------------------------------------------------------------------ scoring

    def score_frames(self, features: list[FrameFeatures]) -> list[FrameFeatures]:
        if not features:
            return features

        raw = np.asarray(
            [self._value(f, "motion_raw", f.motion_score) for f in features],
            dtype=np.float64,
        )
        onset = np.zeros_like(raw)
        onset[1:] = np.clip(raw[1:] - raw[:-1], 0, None)
        onset = self._local_normalize(onset, self.local_norm_window)
        cue_available = self._has_cue_kinematics(features)
        times = [f.t for f in features]

        for i, f in enumerate(features):
            view = float(np.clip(f.table_confidence + 0.15, 0, 1))
            audio = float(np.clip(max(f.audio_onset, f.audio_highband), 0, 1))
            generic = self.w_motion * float(onset[i]) + self.w_view * view

            if cue_available:
                metrics = self._transition_metrics(features, i, times)
                accel_score = float(np.clip(metrics["cue_acceleration"] / 4.0, 0, 1))
                visual = (
                    self.w_accel * accel_score
                    + self.w_sustained * metrics["sustained_ratio"]
                    + 0.10 * metrics["stationary_ratio"]
                    + 0.08 * metrics["track_confidence"]
                    + 0.06 * metrics["cue_contact_score"]
                )
                score = generic * 0.45 + visual
                if not self._transition_confirmed(metrics):
                    # Proposal evidence can remain visible in diagnostics, but
                    # cannot cross the confirmation threshold by itself.
                    score = min(score * 0.22, self.min_conf * 0.75)
            else:
                # Compatibility path for old saved analyses and synthetic unit
                # fixtures that contain no cue-ball tracks.
                area = float(np.clip(f.motion_area_ratio / 0.02, 0, 1))
                score = generic + 0.18 * area

            # Audio never contributes enough to create a candidate by itself.
            score += min(self.w_audio, self.max_audio_weight, 0.12) * audio
            if not self._valid(f):
                score *= 0.10
            if f.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY):
                score *= 0.05
            f.strike_score = float(np.clip(score, 0, 1))
        return features

    # ------------------------------------------------------------------ candidates

    def detect_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        if not features:
            return []
        cue_available = self._has_cue_kinematics(features)
        times = [f.t for f in features]
        proposals: list[tuple[int, dict[str, float]]] = []

        if cue_available:
            for i in range(1, len(features)):
                metrics = self._transition_metrics(features, i, times)
                if self._transition_confirmed(metrics):
                    proposals.append((i, metrics))
                else:
                    # If the white ball disappears at impact, retain a separate
                    # low-confidence visual onset candidate for manual review.
                    # It is NMS'd against any cue-confirmed candidate below.
                    onset = self._ball_onset_metrics(features, i, times)
                    if self._ball_onset_confirmed(onset):
                        proposals.append(
                            (
                                i,
                                {
                                    **onset,
                                    "cue_ball_motion_confirmed": 0.0,
                                    "occlusion_inferred": 1.0,
                                    "cue_geometry_confirmed": 0.0,
                                },
                            )
                        )
        elif self.allow_legacy_fallback:
            scores = np.asarray([f.strike_score for f in features], dtype=np.float64)
            for i in range(1, len(scores) - 1):
                if scores[i] < self.min_conf:
                    continue
                # Prefer the first threshold crossing (the onset) over the later
                # residual peak.  This keeps legacy/no-track fixtures usable and
                # is closer to cue contact than a post-impact maximum.
                crossing = scores[i - 1] < self.min_conf <= scores[i]
                local_peak = scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]
                if (crossing or local_peak) and self._legacy_pre_quiet_ok(
                    features, i, times
                ):
                    proposals.append((i, {}))

        # Temporal non-maximum suppression without changing the selected time.
        kept: list[tuple[int, dict[str, float]]] = []
        for item in proposals:
            i, _ = item
            if not kept or features[i].t - features[kept[-1][0]].t >= self.min_dist:
                kept.append(item)
            elif features[i].strike_score > features[kept[-1][0]].strike_score:
                kept[-1] = item

        candidates: list[StrikeCandidate] = []
        for idx, metrics in kept:
            f = features[idx]
            score = float(f.strike_score)
            if cue_available:
                # Visual confirmation supplies a confidence floor; audio can
                # improve confidence but never supplies confirmation.
                if metrics.get("occlusion_inferred", 0.0) >= 0.5:
                    score = max(
                        score,
                        0.52
                        + 0.10 * metrics["pre_motion_quiet_ratio"]
                        + 0.08 * min(1.0, metrics["ball_onset_run"] / 3.0)
                        + 0.08 * min(1.0, metrics["ball_onset_normalized_speed"] / 4.0),
                    )
                else:
                    fallback_floor = 0.72 if metrics.get("cue_geometry_confirmed", 0.0) < 0.5 else 0.82
                    score = max(
                        score,
                        fallback_floor
                        + 0.10 * metrics["stationary_ratio"]
                        + 0.10 * metrics["sustained_ratio"]
                        + 0.08 * metrics["track_confidence"]
                        + 0.06 * metrics["cue_contact_score"],
                    )
            evidence = {
                "strike_score": score,
                "motion_score": f.motion_score,
                "motion_raw": self._value(f, "motion_raw", f.motion_score),
                "audio_onset": f.audio_onset,
                "table_confidence": f.table_confidence,
                "cue_ball_motion_confirmed": 1.0 if cue_available else 0.0,
                **metrics,
            }
            candidates.append(
                StrikeCandidate(
                    timestamp=f.t,
                    confidence=float(np.clip(score, 0, 1)),
                    evidence=evidence,
                    uncertainty_start=max(0.0, f.t - self.refine_r),
                    uncertainty_end=f.t + self.refine_r,
                    camera_view=f.view_type,
                    possible_replay=False,
                )
            )
        logger.info("Found %d cue-strike candidates", len(candidates))
        return candidates

    def detect_sparse_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Propose strike windows from a deliberately sparse (usually 2 fps) pass.

        The normal detector intentionally requires a frame-level cue-ball
        transition and therefore cannot confirm a strike when the sparse pass
        skips the impact.  This method only proposes broad windows using the
        change from a quiet table to sustained ball activity.  Every proposal is
        re-decoded at native/refine fps and must pass the regular detector before
        it is exported as a high-confidence shot.
        """
        if not features:
            return []
        times = [f.t for f in features]

        def activity(f: FrameFeatures) -> float:
            raw = self._value(f, "motion_raw", self._value(f, "motion_score"))
            residual = self._value(f, "ball_residual_motion")
            cue_speed = self._value(f, "cue_ball_normalized_speed")
            moving = min(1.0, self._value(f, "moving_ball_count") / 2.0)
            return float(
                np.clip(
                    max(
                        raw,
                        residual,
                        self._value(f, "motion_score"),
                        min(1.0, cue_speed / max(self.start_speed, 1e-6)),
                        moving,
                    ),
                    0.0,
                    1.0,
                )
            )

        values = [activity(f) for f in features]
        proposals: list[StrikeCandidate] = []
        for i, f in enumerate(features):
            if not self._valid(f):
                continue
            pre_lo = bisect_left(times, f.t - self.sparse_pre_quiet_s)
            pre_hi = bisect_left(times, f.t - 0.01)
            post_hi = bisect_right(times, f.t + self.sparse_post_s)
            pre = values[pre_lo:pre_hi]
            post = values[i:post_hi]
            if len(pre) < 1 or len(post) < self.sparse_min_active:
                continue
            # Max-ball-speed is intentionally down-weighted above: sparse
            # Hough tracks often report a large identity jump while the table
            # is still.  A median quiet gate tolerates that isolated jitter.
            quiet = float(np.median(pre)) <= 0.55
            active = [v for v in post if v >= self.sparse_activity_threshold]
            ball_active = sum(
                self._value(x, "max_ball_normalized_speed") >= 1.5
                or self._value(x, "moving_ball_count") >= 1
                or self._value(x, "ball_residual_motion") >= self.sparse_activity_threshold
                or self._value(x, "cue_ball_normalized_speed") >= self.start_speed * 0.75
                for x in features[i:post_hi]
            )
            peak = max(post or [0.0])
            previous_activity = values[i - 1] if i > 0 else 0.0
            rising = values[i] - previous_activity >= 0.20
            cue_rising = (
                self._value(f, "cue_ball_normalized_speed")
                >= self.start_speed * 0.75
                and self._value(features[i - 1], "cue_ball_normalized_speed")
                < self.start_speed * 0.50
                if i > 0
                else False
            )
            # A sparse cadence can land in the middle of a noisy rolling-ball
            # interval.  In that case a clear cue-speed/residual onset is still
            # a useful proposal even though the long quiet median is imperfect.
            if (
                (not quiet and not rising and not cue_rising)
                or len(active) < self.sparse_min_active
                or ball_active < self.sparse_min_active
            ):
                continue
            score = float(
                np.clip(
                    0.40
                    + 0.20 * min(1.0, peak)
                    + 0.15 * min(1.0, ball_active / 3.0)
                    + 0.10 * float(np.clip(f.table_confidence, 0.0, 1.0))
                    + 0.08 * float(np.clip(max(f.audio_onset, f.audio_highband), 0.0, 1.0)),
                    0.0,
                    0.78,
                )
            )
            # At 2fps the proposal can be one sample before the actual launch;
            # leave enough uncertainty for the dense pass to snap forward.
            radius = max(self.refine_r, 1.5)
            proposals.append(
                StrikeCandidate(
                    timestamp=f.t,
                    confidence=score,
                    evidence={
                        "sparse_proposal": 1.0,
                        "sparse_activity_peak": float(peak),
                        "sparse_active_samples": float(len(active)),
                        "sparse_ball_active_samples": float(ball_active),
                        "sparse_pre_quiet_median": float(np.median(pre)),
                    },
                    uncertainty_start=max(0.0, f.t - radius),
                    uncertainty_end=f.t + radius,
                    camera_view=f.view_type,
                    possible_replay=False,
                )
            )

        # Temporal NMS keeps the earliest proposal in a burst.  Dense refinement
        # will snap it to the exact frame-level transition.
        kept: list[StrikeCandidate] = []
        for candidate in proposals:
            if not kept or candidate.timestamp - kept[-1].timestamp >= self.sparse_gap_s:
                kept.append(candidate)
            elif candidate.confidence > kept[-1].confidence:
                kept[-1] = candidate
        logger.info("Found %d sparse strike proposals", len(kept))
        return kept

    def _legacy_pre_quiet_ok(
        self,
        features: list[FrameFeatures],
        idx: int,
        times: list[float] | None = None,
    ) -> bool:
        t = features[idx].t
        if times is None:
            times = [f.t for f in features]
        lo = bisect_left(times, t - self.pre_quiet_s)
        hi = bisect_left(times, t - 0.05)
        pre = [
            self._value(f, "motion_raw", f.motion_score)
            for f in features[lo:hi]
            if self._valid(f)
        ]
        if not pre:
            return True
        return float(np.median(pre)) <= self.pre_quiet_max_motion

    def refine_boundaries(
        self,
        candidates: list[StrikeCandidate],
        dense_features: list[FrameFeatures],
    ) -> list[StrikeCandidate]:
        """Snap candidates to the first dense confirmed cue-ball transition."""
        if not candidates or not dense_features:
            return candidates
        cue_available = self._has_cue_kinematics(dense_features)
        if not cue_available:
            return candidates
        times = [f.t for f in dense_features]

        refined: list[StrikeCandidate] = []
        for cand in candidates:
            lo = bisect_left(times, cand.uncertainty_start)
            hi = bisect_right(times, cand.uncertainty_end)
            indices = range(lo, hi)
            match: tuple[int, dict[str, float]] | None = None
            for i in indices:
                if i <= 0:
                    continue
                metrics = self._transition_metrics(dense_features, i, times)
                strict_match = self._transition_confirmed(metrics)
                sparse_match = self._sparse_dense_transition_confirmed(metrics)
                if strict_match or sparse_match:
                    if sparse_match and not strict_match:
                        metrics = {**metrics, "sparse_dense_transition": 1.0}
                    match = (i, metrics)
                    break
            if match is None:
                cand.confidence *= 0.75
                cand.evidence["dense_transition_confirmed"] = 0.0
                refined.append(cand)
                continue

            i, metrics = match
            f = dense_features[i]
            cand.timestamp = f.t
            cand.confidence = max(cand.confidence, float(f.strike_score), 0.75)
            cand.uncertainty_start = dense_features[max(0, i - 1)].t
            cand.uncertainty_end = f.t
            cand.camera_view = f.view_type
            cand.evidence = {
                **cand.evidence,
                **metrics,
                "dense_transition_confirmed": 1.0,
                "refined_strike": f.strike_score,
            }
            refined.append(cand)
        return refined
