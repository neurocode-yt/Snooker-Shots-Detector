"""Cue-strike detection from a stationary-to-moving cue-ball transition.

A confirmed event requires a previously stationary white ball followed by
sustained white-ball motion, or explicit visual impact-occlusion evidence.
Commentary, applause, and cue audio do not affect selection or confidence.
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
        self.hard_cut_threshold = float(config.get("scene_detection.hard_cut_threshold", .42))
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

    # ------------------------------------------------------------------ utilities

    @staticmethod
    def _stabilized_features(features: list[FrameFeatures]) -> list[FrameFeatures]:
        """Compare cue positions in one camera reference while retaining raw pixels.

        The tracker already compensates its history for pan/zoom. Feature rows
        previously stored each white centre in a different image coordinate
        system, so spatial rest checks contradicted that measured stillness.
        Working copies keep source/UI coordinates untouched.
        """
        result = []
        cumulative = np.eye(3, dtype=np.float64)
        previous = None
        for i in range(len(features)):
            f = features[i]
            if f.cue_ball_image_diameter_px is not None:
                result.append(f)
                previous = f
                continue
            matrix = np.asarray(f.camera_frame_transform, dtype=np.float64)
            measured = matrix.size == 6 and np.isfinite(matrix).all() and f.observation_fps >= 10
            contiguous = bool(previous is not None and previous.camera_scene_id == f.camera_scene_id
                              and previous.observation_fps >= 10
                              and abs(f.t-previous.t-f.camera_frame_dt) <= .005
                              and f.camera_frame_dt > 0)
            if measured and contiguous and f.observation_valid:
                step = np.eye(3, dtype=np.float64)
                step[:2] = matrix.reshape(2, 3)
                cumulative = step@cumulative
            else:
                cumulative = np.eye(3, dtype=np.float64)
            if measured:
                changes = {"cue_ball_image_diameter_px": f.ball_diameter_px}
                if previous is not None and previous.camera_scene_id == f.camera_scene_id and not contiguous:
                    # A transform belongs to its measured frame pair. Missing
                    # decoded rows cannot be treated as repeated camera steps.
                    changes["observation_valid"] = False
                if f.cue_ball_x is not None and f.cue_ball_y is not None:
                    try:
                        point = np.linalg.solve(cumulative, np.array([f.cue_ball_x, f.cue_ball_y, 1.]))
                        scale = np.sqrt(abs(np.linalg.det(cumulative[:2, :2])))
                        if np.isfinite(point).all() and scale > 1e-6:
                            changes.update(cue_ball_x=float(point[0]),cue_ball_y=float(point[1]),
                                           ball_diameter_px=float(f.ball_diameter_px/scale))
                    except np.linalg.LinAlgError:
                        changes["observation_valid"] = False
                result.append(f.model_copy(update=changes))
            else:
                result.append(f)
            previous = f
        return result

    @staticmethod
    def _image_diameter(f: FrameFeatures) -> float:
        return float(f.cue_ball_image_diameter_px or f.ball_diameter_px)

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
        if not f.match_context_valid or f.broadcast_replay or f.table_handling:
            return False
        if not bool(getattr(f, "observation_valid", True)):
            return False
        if not bool(getattr(f, "table_observable", True)):
            return False
        if f.scene_cut_score >= 0.5:
            return False
        if f.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY):
            return False
        return f.table_confidence >= 0.18 or f.view_type in TABLE_VIEWS

    @staticmethod
    def _same_view(first: FrameFeatures, second: FrameFeatures) -> bool:
        return first.camera_scene_id == second.camera_scene_id

    def _has_cue_kinematics(self, features: list[FrameFeatures]) -> bool:
        reliable = sum(
            1
            for f in features
            if self._track_conf(f) >= self.min_track_conf
            and self._value(f, "ball_diameter_px") > 0.5
        )
        return reliable >= min(3, max(1, len(features) // 20))

    def _stationary_ratio(self, frames: list[FrameFeatures]) -> float:
        if not frames:
            return 0.0
        ratio = sum((f.cue_ball_stable_normalized_speed
                     if f.cue_ball_stable_normalized_speed is not None
                     else self._cue_speed(f)) <= self.stationary_speed
                    for f in frames) / len(frames)
        points = [f for f in frames if f.cue_ball_x is not None and f.cue_ball_y is not None]
        if len(points) >= 3 and points[-1].t - points[0].t >= 0.15:
            xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in points])
            diameter = max(1.0, float(np.median([f.ball_diameter_px for f in points])))
            # Alternating subpixel circle centres can look fast at 30 fps.
            # A tight spatial cluster is stillness; sustained drift is not.
            distances = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
            clustered = float(np.mean(distances <= 0.12 * diameter))
            if clustered >= 0.80:
                ratio = max(ratio, clustered)
        return ratio

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
        # Keep the observed launch before a cut/invalid camera estimate, but
        # never join it to a new velocity sequence on the other side.
        post_end = next((j for j in range(post_lo, post_hi)
                         if not self._valid(features[j]) or not self._same_view(features[j], f)), post_hi)
        if post_end < post_hi and f.observation_fps >= 10:
            invalid = [j for j in range(post_lo, post_hi) if not self._valid(features[j])]
            established = [x for x in features[post_lo:post_end]
                           if x.cue_ball_detected and x.cue_ball_stable_normalized_speed is not None
                           and x.cue_ball_stable_normalized_speed >= self.start_speed]
            # One short registration failure after an already measured roll
            # need not erase that roll. It cannot establish motion which only
            # appears on the far side, and it can never bridge a camera cut.
            if (len(invalid) == 1 and len(established) >= 3 and post_lo < post_end < post_hi-1
                    and all(self._same_view(x, f) and x.scene_cut_score < self.hard_cut_threshold
                            for x in features[post_lo:post_hi])
                    and features[post_end+1].t-features[post_end-1].t <= 2/f.observation_fps+.005):
                post_end = post_hi
        pre = [
            x
            for x in features[pre_lo:pre_hi]
            if self._valid(x)
            and self._same_view(x, f)
            and self._track_conf(x) >= self.min_track_conf * 0.7
        ]
        post = [
            x
            for x in features[post_lo:post_end]
            if self._valid(x)
            and self._same_view(x, f)
            and self._track_conf(x) >= self.min_track_conf * 0.7
        ]

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
        stationary_ratio = self._stationary_ratio(pre)
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
        previous_speed = self._cue_speed(features[idx - 1]) if idx > 0 and self._same_view(features[idx - 1], f) else current_speed
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
        pre_address = max([self._value(x, "cue_contact_score") for x in pre
                           if x.cue_tip_visible and x.cue_tip_distance_to_ball <= 3*max(self._image_diameter(x), 1)] or [0.0])
        # A real cue-ball launch has a coherent displacement over consecutive
        # observations.  One-frame Hough identity jumps caused by a walking
        # player often have a large apparent speed but immediately reverse or
        # disappear; they fail this direction/displacement check.
        point_frames = [x for x in post if x.cue_ball_x is not None and x.cue_ball_y is not None]
        points = [(x.cue_ball_x, x.cue_ball_y) for x in point_frames]
        direction_consistency = 0.0
        displacement = 0.0
        # One-frame identity swaps to a white logo must not destroy an
        # otherwise continuous launch. Remove only an isolated out-and-back
        # jump; repeated swaps, a missing trajectory, or a real reversal stay.
        if len(points) >= 4:
            diameter = max(self._value(f, "ball_diameter_px"), 1.0)
            xy = np.asarray(points, dtype=np.float64)
            spikes = [i for i in range(1, len(xy) - 1)
                      if point_frames[i + 1].t - point_frames[i - 1].t <= 0.10 + 1e-6
                      and np.linalg.norm(xy[i] - xy[i - 1]) > 4 * diameter
                      and np.linalg.norm(xy[i] - xy[i + 1]) > 4 * diameter
                      and np.linalg.norm(xy[i + 1] - xy[i - 1]) < 2 * diameter]
            if len(spikes) == 1:
                points = [point for i, point in enumerate(points) if i != spikes[0]]
        if len(points) >= 2:
            vectors = np.diff(np.asarray(points, dtype=np.float64), axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            total = float(np.sum(lengths))
            displacement = float(np.linalg.norm(np.asarray(points[-1]) - np.asarray(points[0])))
            if total > 1e-6:
                direction_consistency = float(np.clip(displacement / total, 0, 1))
        diameter = max(self._value(f, "ball_diameter_px"), 1.0)
        anchor_direction = 1.0
        anchor_excursion = 0.0
        largest_step = 0.0
        other_step_travel = 0.0
        reliable_pre = [x for x in pre if x.cue_ball_detected and self._track_conf(x) >= .55
                        and x.cue_ball_x is not None and x.cue_ball_y is not None]
        if reliable_pre and points:
            anchor = np.array([reliable_pre[-1].cue_ball_x, reliable_pre[-1].cue_ball_y])
            anchored = np.vstack((anchor, np.asarray(points)))
            step_lengths = np.linalg.norm(np.diff(anchored, axis=0), axis=1)
            path = float(np.sum(step_lengths))
            anchor_direction = float(np.linalg.norm(anchored[-1]-anchor)) / max(path, 1e-6)
            anchor_excursion = float(np.max(np.linalg.norm(anchored[1:]-anchor, axis=1))) / diameter
            largest_step = float(np.max(step_lengths)) / diameter
            other_step_travel = path / diameter - largest_step
        stable_post = [x.cue_ball_stable_normalized_speed for x in post
                       if x.cue_ball_stable_normalized_speed is not None]
        # The first collision can bend the white's path inside the post window.
        # Verify the initial three reliable points separately, before asking a
        # longer trajectory to remain straight through that physical collision.
        initial = point_frames[:3]
        initial_direction = initial_displacement = 0.0
        if (len(initial) == 3 and initial[0].t-t <= .10
                and initial[-1].t-initial[0].t <= .20
                and all(self._cue_speed(x) >= self.continue_speed for x in initial)):
            xy = np.asarray([(x.cue_ball_x, x.cue_ball_y) for x in initial])
            net = float(np.linalg.norm(xy[-1]-xy[0]))
            path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            initial_direction = net / max(path, 1e-6)
            initial_displacement = net / diameter
        metrics = {
            "observation_fps": self._value(f, "observation_fps"),
            "cue_shape_measured": float(f.cue_ball_quality),
            "post_observation_contiguous": float(post_end == post_hi),
            "ball_diameter_px": diameter,
            "observed_ball_diameter_px": self._image_diameter(f),
            "stationary_ratio": float(stationary_ratio),
            "sustained_ratio": float(sustained_ratio),
            "sustained_count": float(sustained_count),
            "sustained_run": float(sustained_run),
            "bridged_sustained_count": float(bridged_sustained_count),
            "post_peak_cue_speed": float(max(post_speeds or [0.0])),
            "cue_speed": float(current_speed),
            "cue_track_visible": float(
                f.cue_ball_x is not None and f.cue_ball_y is not None
                and self._track_conf(f) >= self.min_track_conf * 0.7
            ),
            "previous_cue_speed": float(previous_speed),
            "cue_acceleration": float(max(0.0, accel)),
            "speed_crossing": crossing,
            "track_confidence": float(track_conf),
            "cue_displacement_diameters": float(displacement / diameter),
            "cue_direction_consistency": float(direction_consistency),
            "anchor_direction_consistency": anchor_direction,
            "anchor_excursion_diameters": anchor_excursion,
            "largest_cue_step_diameters": largest_step,
            "other_cue_step_travel_diameters": other_step_travel,
            "independent_object_motion_count": float(sum(
                x.cue_ball_detected and x.moving_ball_count > 0
                and x.cue_ball_stable_normalized_speed is not None
                and x.max_ball_normalized_speed >= max(1.5, x.cue_ball_stable_normalized_speed + 1.)
                for x in post)),
            "stable_cue_observed": float(bool(stable_post)),
            "stable_cue_motion_count": float(sum(v >= self.continue_speed for v in stable_post)),
            "stable_cue_peak_speed": float(max(stable_post or [0])),
            "initial_launch_direction": initial_direction,
            "initial_launch_displacement": initial_displacement,
            "pre_sample_count": float(len(pre)),
            "pre_motion_quiet_ratio": float(pre_quiet_ratio),
            "pre_motion_raw_median": pre_raw_median,
            "pre_ball_quiet_ratio": float(pre_ball_quiet_ratio),
            "pre_ball_speed_median": pre_ball_speed_median,
            "cue_tip_visible_count": float(tip_visible),
            "cue_contact_score": float(cue_contact),
            "cue_approach_speed": float(cue_approach),
            "pre_cue_address_score": float(pre_address),
            "pre_cue_spatial_quiet_ratio": float(
                np.mean(np.linalg.norm(
                    np.asarray([(x.cue_ball_x, x.cue_ball_y) for x in reliable_pre])
                    - np.median([(x.cue_ball_x, x.cue_ball_y) for x in reliable_pre], axis=0),
                    axis=1) <= .15*diameter)
                if len(reliable_pre) >= 3 else 0.),
            "pre_measured_cue_quiet_ratio": float(sum(
                x.cue_ball_detected and x.cue_ball_quality and x.cue_ball_observations >= 3
                and self._track_conf(x) >= .75
                and x.cue_ball_stable_normalized_speed is not None
                and x.cue_ball_stable_normalized_speed <= self.stationary_speed
                for x in pre) / max(len(pre), 1)),
            "cue_geometry_confirmed": float(
                cue_contact >= 0.20 or cue_approach >= 0.15
            ),
        }
        metrics["gradual_launch_confirmed"] = float(
            self._gradual_launch_confirmed(features, idx, times, metrics))
        metrics["addressed_departure_confirmed"] = float(
            self._addressed_departure_confirmed(features, idx, times, metrics))
        metrics["soft_addressed_departure_confirmed"] = float(
            self._soft_addressed_departure_confirmed(features, idx, times, metrics))
        metrics["measured_slow_departure_confirmed"] = float(
            self._measured_slow_departure_confirmed(features, idx, times, metrics))
        metrics["addressed_collision_launch_confirmed"] = float(
            self._addressed_collision_launch_confirmed(metrics))
        metrics["camera_drift_launch_confirmed"] = float(
            self._camera_drift_launch_confirmed(features, idx, times, metrics))
        metrics["quiet_anchor_contradiction"] = float(
            (self._transition_confirmed(metrics) or self._sparse_dense_transition_confirmed(metrics))
            and self._quiet_anchor_contradiction(features, idx, times, metrics))
        return metrics

    def _quiet_anchor_contradiction(self, features: list[FrameFeatures], idx: int,
                                    times: list[float], metrics: dict[str, float]) -> bool:
        """Reject apparent travel supplied only by unreliable identity swaps.

        Rest/cue edges can replace a still white for a moment. Reacquiring that
        same white produces a velocity spike which decays while its centre
        stays still. Every reliable post centre must corroborate the original
        quiet position; a measured departure or immediate object-ball collision
        prevents this rejection.
        """
        if (metrics.get("observation_fps", 0) < 10
                or metrics.get("independent_object_motion_count", 0) >= 3
                or metrics.get("cue_displacement_diameters", 0) < .25
                or metrics.get("stable_cue_peak_speed", 0) < 2):
            return False
        current = features[idx]
        pre = [f for f in features[bisect_left(times, current.t-.75):idx]
               if self._valid(f) and self._same_view(f, current) and f.cue_ball_detected
               and self._track_conf(f) >= .65
               and f.cue_ball_x is not None and f.cue_ball_y is not None
               and f.cue_ball_stable_normalized_speed is not None
               and f.cue_ball_stable_normalized_speed <= self.stationary_speed]
        if len(pre) < 3 or pre[-1].t-pre[0].t < .1:
            return False
        diameter = max(1., float(np.median([f.ball_diameter_px for f in pre])))
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in pre])
        anchor = np.median(xy, axis=0)
        if float(np.mean(np.linalg.norm(xy-anchor, axis=1) <= .12*diameter)) < .75:
            return False
        post = features[idx:bisect_right(times, current.t+.6)]
        if len(post) < 6 or post[-1].t-post[0].t < .4:
            return False
        if any(not self._valid(f) or not self._same_view(f, current) or f.observation_fps < 10
               or f.scene_cut_score >= self.hard_cut_threshold for f in post):
            return False
        if any(b.t-a.t > 2/min(a.observation_fps, b.observation_fps)+.005
               for a, b in zip(post, post[1:])):
            return False
        points = [f for f in post if f.cue_ball_detected
                  and f.cue_ball_x is not None and f.cue_ball_y is not None]
        if not points or max(np.linalg.norm(np.array([f.cue_ball_x, f.cue_ball_y])-anchor)
                             for f in points) < .3*diameter:
            return False
        reliable = [f for f in points if self._track_conf(f) >= .65]
        if len(reliable) < 3 or reliable[-1].t-reliable[0].t < .12:
            return False
        distances = np.linalg.norm(
            np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in reliable])-anchor, axis=1)
        return bool(max(distances) <= .12*diameter)

    def _addressed_departure_confirmed(self, features, idx, times, metrics) -> bool:
        """Measured stillness and cue contact can outvote foreground flow."""
        if (metrics.get("observation_fps", 0) < 10
                or metrics.get("post_observation_contiguous", 0) < .5
                or metrics.get("stationary_ratio", 0) < .80
                or metrics.get("pre_sample_count", 0) < 3
                or metrics.get("pre_ball_quiet_ratio", 0) < .90
                or metrics.get("pre_cue_address_score", 0) < .40
                or metrics.get("cue_contact_score", 0) < .50
                or metrics.get("track_confidence", 0) < .75
                or metrics.get("stable_cue_peak_speed", 0) < 2.):
            return False
        current = features[idx]
        if (current.cue_ball_stable_normalized_speed or 0.) < .75:
            return False
        pre = features[bisect_left(times, current.t-.75):idx]
        if any(not self._valid(f) or not self._same_view(f, current) for f in pre):
            return False
        quiet = [f for f in pre if f.cue_ball_detected and f.cue_ball_quality and self._track_conf(f) >= .65
                 and f.cue_ball_x is not None and f.cue_ball_y is not None]
        # A bridge can obscure the final short pre-window. The longer window
        # must still contain six measured quiet whites, with strong contact
        # and a recent quiet anchor when visibility falls below three quarters.
        if (len(quiet) < 6 or len(quiet)<.50*len(pre) or quiet[-1].t-quiet[0].t < .20
                or current.t-quiet[-1].t>.40
                or (len(quiet)<.75*len(pre) and metrics.get("cue_contact_score",0)<.85)):
            return False
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in quiet])
        diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet])))
        anchor = np.median(xy, axis=0)
        if (current.cue_ball_x is None or current.cue_ball_y is None
                or np.linalg.norm(np.asarray([current.cue_ball_x, current.cue_ball_y])-anchor)
                < .12*diameter):
            return False
        if np.max(np.linalg.norm(xy-anchor, axis=1)) > .25*diameter:
            return False
        post = features[idx:bisect_right(times, current.t+.55)]
        if (len(post) < 10 or post[-1].t-current.t < .40
                or any(not self._valid(f) or not self._same_view(f, current)
                       or f.observation_fps < 10 for f in pre+post)
                or any(b.t-a.t > 2/min(a.observation_fps, b.observation_fps)+.005
                       for a, b in zip(pre+post, (pre+post)[1:]))):
            return False
        moving = [f for f in post if f.cue_ball_detected and f.cue_ball_quality
                  and self._track_conf(f) >= .75
                  and f.cue_ball_x is not None and f.cue_ball_y is not None]
        if len(moving) < 8 or len(moving) < .65*len(post):
            return False
        diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet+moving])))
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in moving])
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        net = float(np.linalg.norm(xy[-1]-xy[0]))
        direction_required=.85 if metrics.get("cue_contact_score",0)>=.85 else .90
        return bool(net/diameter >= .75 and net/max(float(np.sum(steps)), 1e-6) >= direction_required
                    and max(steps)/diameter <= 3.
                    and np.linalg.norm(xy[0]-anchor)/diameter <= 1.5
                    and sum((f.cue_ball_stable_normalized_speed or 0.) >= self.continue_speed
                            for f in moving) >= .80*len(moving))

    def _gradual_launch_confirmed(self, features: list[FrameFeatures], idx: int,
                                  times: list[float], metrics: dict[str, float]) -> bool:
        """Give a gentle addressed stroke time to establish a measured roll."""
        if (metrics["observation_fps"] < 10 or metrics["ball_diameter_px"] < 24
                or metrics["stationary_ratio"] < .90 or metrics["pre_sample_count"] < 6
                or metrics["pre_ball_quiet_ratio"] < .90 or metrics["pre_motion_raw_median"] > .50
                or metrics["track_confidence"] < .80 or not metrics["speed_crossing"]
                or metrics["cue_speed"] < self.start_speed):
            return False
        current = features[idx]
        before = features[bisect_left(times, current.t-.50):idx]
        after = features[idx:bisect_right(times, current.t+.80)]
        rows = before+after
        if (len(before) < 6 or len(after) < 8 or after[-1].t-current.t < .65
                or any(not self._valid(f) or not self._same_view(f, current)
                       or f.observation_fps < 10 or not f.cue_ball_detected
                       or f.cue_ball_x is None or f.cue_ball_y is None
                       or self._track_conf(f) < .75 for f in rows)
                or any(not 0 < b.t-a.t <= 2/min(a.observation_fps, b.observation_fps)+.005
                       for a, b in zip(rows, rows[1:]))):
            return False
        diameter = max(1., float(np.median([f.ball_diameter_px for f in rows])))
        anchor = np.median([(f.cue_ball_x, f.cue_ball_y) for f in before], axis=0)
        quiet_xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in before])
        if max(np.linalg.norm(quiet_xy-anchor, axis=1))/diameter > .15:
            return False
        contact = max((f.cue_contact_score for f in after
                       if f.t <= current.t+.20 and f.cue_tip_visible), default=0.)
        if contact < .75:
            return False
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in after])
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)/diameter
        net = float(np.linalg.norm(xy[-1]-xy[0]))/diameter
        direction = net/max(float(np.sum(steps)), 1e-6)
        stable = [f.cue_ball_stable_normalized_speed for f in after]
        return bool(net >= .75 and direction >= .95 and max(steps) <= .25
                    and np.linalg.norm(xy[0]-anchor)/diameter <= .25
                    and sum(v is not None and v >= self.continue_speed for v in stable)/len(stable) >= .80)

    def _soft_addressed_departure_confirmed(self, features, idx, times, metrics) -> bool:
        """Resolve slow addressed travel despite circle-centre pixel jitter."""
        current = features[idx]
        if (not current.cue_ball_quality or current.observation_fps < 10
                or metrics.get("stationary_ratio", 0) < .90
                or metrics.get("pre_sample_count", 0) < 8
                or metrics.get("pre_ball_quiet_ratio", 0) < .50
                or metrics.get("pre_ball_speed_median", 1) > self.pre_quiet_max_ball_speed
                or max(metrics.get("pre_cue_address_score", 0),
                       metrics.get("cue_contact_score", 0)) < .75
                or metrics.get("pre_motion_raw_median", 1) > .50
                or metrics.get("speed_crossing", 0) < .5
                or metrics.get("stable_cue_peak_speed", 0) < 1.0):
            return False
        before = features[bisect_left(times, current.t-.50):idx]
        after = features[idx:bisect_right(times, current.t+.80)]
        rows = before+after
        if (np.median([f.camera_motion_magnitude for f in rows]) > 3.
                and metrics.get("post_peak_cue_speed", 0) < 3.):
            return False
        if (len(before) < 10 or len(after) < 16 or after[-1].t-current.t < .70
                or any(not self._valid(f) or not self._same_view(f, current)
                       or f.observation_fps < 10 for f in rows)
                or any(not 0 < b.t-a.t <= 2/min(a.observation_fps, b.observation_fps)+.005
                       for a, b in zip(rows, rows[1:]))):
            return False
        def positioned(f):
            return (f.cue_ball_detected and self._track_conf(f) >= .75
                    and f.cue_ball_x is not None and f.cue_ball_y is not None)
        quiet = [f for f in before if positioned(f)]
        moving = [f for f in after if positioned(f)]
        if (len(quiet) < .80*len(before) or len(moving) < .70*len(after)
                or sum(f.cue_ball_quality for f in quiet) < .50*len(quiet)
                or sum(f.cue_ball_quality for f in moving) < .75*len(moving)):
            return False
        diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet+moving])))
        quiet_xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in quiet])
        anchor = np.median(quiet_xy, axis=0)
        if np.max(np.linalg.norm(quiet_xy-anchor, axis=1))/diameter > .25:
            return False
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in moving])
        net = float(np.linalg.norm(xy[-1]-xy[0]))/diameter
        if (net < .75 or np.linalg.norm(xy[0]-anchor)/diameter > .40
                or np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))/diameter > .35):
            return False
        centred = xy-np.mean(xy, axis=0)
        _, singular, axes = np.linalg.svd(centred, full_matrices=False)
        if singular[0]**2/max(float(np.sum(singular**2)), 1e-6) < .95:
            return False
        axis = axes[0] * (1 if np.dot(xy[-1]-xy[0], axes[0]) >= 0 else -1)
        progress = centred@axis
        perpendicular = centred-progress[:, None]*axis
        return bool(np.max(np.linalg.norm(perpendicular, axis=1))/diameter <= .15
                    and np.min(np.diff(progress))/diameter >= -.12
                    and sum((f.cue_ball_stable_normalized_speed or 0.) >= self.continue_speed
                            for f in moving) >= .80*len(moving))

    def _camera_drift_launch_confirmed(self, features, idx, times, metrics) -> bool:
        """Separate a fast addressed launch from smooth zoom registration drift.

        Two small coloured balls may not register the cloth plane accurately.
        A bounded, smooth pre-contact drift then contradicts tracker stillness.
        Require measured cue address, a continuous reliable white and an abrupt
        multi-frame departure far faster than that fitted drift. Camera motion
        or the presence of a player alone supplies no launch evidence.
        """
        current=features[idx]
        if (current.observation_fps < 10 or not current.cue_ball_quality
                or metrics.get("track_confidence",0) < .80
                or metrics.get("post_observation_contiguous",0) < .5
                or metrics.get("cue_contact_score",0) < .40
                or metrics.get("stable_cue_peak_speed",0) < 6.
                or metrics.get("stable_cue_motion_count",0) < 5
                or metrics.get("initial_launch_displacement",0) < .50
                or metrics.get("initial_launch_direction",0) < .90
                or metrics.get("cue_displacement_diameters",0) < 1.50
                or metrics.get("cue_direction_consistency",0) < .80
                or metrics.get("largest_cue_step_diameters",0) > 1.50):
            return False
        before=features[bisect_left(times,current.t-.65):idx]
        if (len(before)<12 or before[-1].t-before[0].t<.45
                or np.median([f.camera_motion_magnitude for f in before])<3.
                or any(not self._valid(f) or not self._same_view(f,current)
                       or f.observation_fps<10 or not f.cue_ball_detected
                       or not f.cue_ball_quality or self._track_conf(f)<.75
                       or f.cue_ball_x is None or f.cue_ball_y is None for f in before)
                or any(not 0<b.t-a.t<=2/min(a.observation_fps,b.observation_fps)+.005
                       for a,b in zip(before+[current],(before+[current])[1:]))
                or not (max((f.cue_contact_score for f in before if f.cue_tip_visible),default=0)>=.75
                        or (max((f.cue_contact_score for f in before if f.cue_tip_visible),default=0)>=.40
                            and metrics.get("cue_contact_score",0)>=.75))
                or np.median([f.motion_raw for f in before])>.50
                or np.median([f.max_ball_normalized_speed for f in before])>2.5):
            return False
        diameter=max(1.,float(np.median([f.ball_diameter_px for f in before])))
        xy=np.asarray([(f.cue_ball_x,f.cue_ball_y) for f in before])
        clock=np.asarray([f.t-current.t for f in before])
        design=np.column_stack((clock,np.ones(len(clock))))
        fit=np.linalg.lstsq(design,xy,rcond=None)[0]
        drift=float(np.linalg.norm(fit[0]))/diameter
        residual=np.linalg.norm(xy-design@fit,axis=1)/diameter
        return bool(drift<=2.5 and max(residual)<=.10
                    and self._cue_speed(current)>=max(3.,3*drift))

    def _measured_slow_departure_confirmed(self, features, idx, times, metrics) -> bool:
        """A measured quiet white can launch below fast-roll thresholds."""
        current = features[idx]
        if (not current.cue_ball_quality or current.observation_fps < 10
                or metrics.get("stationary_ratio", 0) < .90
                or metrics.get("pre_sample_count", 0) < 8
                or metrics.get("pre_ball_quiet_ratio", 0) < .90
                or metrics.get("pre_motion_raw_median", 1) > .50
                or max(metrics.get("pre_cue_address_score", 0), metrics.get("cue_contact_score", 0)) < .20
                or metrics.get("track_confidence", 0) < .75
                or metrics.get("stable_cue_motion_count", 0) < 4
                or metrics.get("stable_cue_peak_speed", 0) < 1.):
            return False
        before = features[bisect_left(times, current.t-.50):idx]
        after = []
        for frame in features[idx:bisect_right(times, current.t+.60)]:
            if not self._valid(frame) or not self._same_view(frame, current):
                break
            after.append(frame)
        if (len(before) < 8 or len(after) < 8 or after[-1].t-current.t < .28
                or any(not self._valid(f) or not self._same_view(f, current)
                       or f.observation_fps < 10 for f in before+after)
                or any(b.t-a.t > 2/min(a.observation_fps, b.observation_fps)+.005
                       for a, b in zip(before+after, (before+after)[1:]))):
            return False
        if (np.median([f.camera_motion_magnitude for f in before+after]) > 3.
                and metrics.get("post_peak_cue_speed", 0) < 3.):
            # A slow detector-centre drift while the camera zooms cannot
            # establish contact. Strong addressed/accelerating launch paths
            # supply separate proof when a real shot occurs during that zoom.
            return False
        quiet = [f for f in before if f.cue_ball_quality and f.cue_ball_detected
                 and self._track_conf(f) >= .75 and f.cue_ball_x is not None and f.cue_ball_y is not None]
        moving = [f for f in after if f.cue_ball_quality and f.cue_ball_detected
                  and self._track_conf(f) >= .75 and f.cue_ball_x is not None and f.cue_ball_y is not None]
        if len(quiet) < .75*len(before) or len(moving) < .65*len(after):
            return False
        diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet+moving])))
        qxy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in quiet])
        anchor = np.median(qxy, axis=0)
        if np.mean(np.linalg.norm(qxy-anchor, axis=1)/diameter <= .15) < .80:
            return False
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in moving])
        jumps = np.linalg.norm(np.diff(xy, axis=0), axis=1)/diameter
        discontinuity = next((i+1 for i,v in enumerate(jumps) if v > .50), len(xy))
        if discontinuity < len(xy):
            moving, xy = moving[:discontinuity], xy[:discontinuity]
            if len(moving) < 8 or moving[-1].t-moving[0].t < .28:
                return False
        net = float(np.linalg.norm(xy[-1]-xy[0]))
        path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        return bool(net/diameter >= .50 and net/max(path, 1e-6) >= .90
                    and .10 <= np.linalg.norm(xy[0]-anchor)/diameter <= .50
                    and np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))/diameter <= .50
                    and sum((f.cue_ball_stable_normalized_speed or 0.) >= self.continue_speed
                            for f in moving) >= .80*len(moving))

    @staticmethod
    def _single_step_identity_jump(metrics: dict[str, float]) -> bool:
        # A white-to-red/bridge identity swap can look like a straight launch:
        # the tracker reports a large jump, then its velocity filter decays
        # while the replacement object is still. Require actual travel beyond
        # that single step before accepting its apparent sustained speed.
        largest = metrics.get("largest_cue_step_diameters", 0.0)
        remaining = metrics.get("other_cue_step_travel_diameters", 0.0)
        # Sparse sampling can contain a complete genuine collision in one
        # step. Even at native cadence, independently moving object balls and
        # contact geometry explain a white that stops abruptly after impact.
        collision = (metrics.get("independent_object_motion_count", 0) >= 3
                     and metrics.get("cue_contact_score", 0) >= .65)
        return bool(metrics.get("observation_fps", 0) >= 10 and not collision
                    and largest >= .75 and remaining < max(.50, .10 * largest))

    @staticmethod
    def _identity_return(metrics: dict[str, float]) -> bool:
        # A mistaken bridge identity can travel away from a stationary white
        # and return to its exact original centre. A trajectory beginning at
        # that wrong identity alone can look straight. Include the real quiet
        # anchor; strong contact or coherent initial launch can justify a true
        # early collision and return instead.
        return bool(metrics.get("anchor_excursion_diameters", 0) >= 1.0
                    and metrics.get("anchor_direction_consistency", 1) < .20
                    and metrics.get("pre_cue_address_score", 0) < .65
                    and not (metrics.get("initial_launch_displacement", 0) >= .50
                             and metrics.get("initial_launch_direction", 0) >= .80
                             and metrics.get("cue_contact_score", 0) >= .40))

    @staticmethod
    def _anchored_launch_confirmed(metrics: dict[str, float]) -> bool:
        """Independent white-ball evidence can outvote cue/player cloth flow."""
        straight_launch = (metrics.get("cue_direction_consistency", 0) >= .90
                           and metrics.get("anchor_direction_consistency", 0) >= .90)
        # A nearby red can deflect the white within the first 0.3 seconds. Its
        # initial straight leg, actual cue contact and continued stabilized
        # travel establish the launch even though the complete path bends.
        early_collision = (metrics.get("cue_contact_score", 0) >= .70
                           and metrics.get("initial_launch_direction", 0) >= .95
                           and metrics.get("initial_launch_displacement", 0) >= .50
                           and metrics.get("cue_direction_consistency", 0) >= .55
                           and metrics.get("anchor_direction_consistency", 0) >= .65)
        return bool(metrics.get("stationary_ratio", 0) >= .80
                    and (metrics.get("pre_ball_quiet_ratio", 0) >= .80
                         or (metrics.get("pre_ball_quiet_ratio", 0) >= .50
                             and metrics.get("cue_geometry_confirmed", 0) >= .5
                             and metrics.get("cue_contact_score", 0) >= .70))
                    and metrics.get("pre_sample_count", 0) >= 4
                    and metrics.get("stable_cue_motion_count", 0) >= 3
                    and metrics.get("stable_cue_peak_speed", 0) >= 4
                    and metrics.get("track_confidence", 0) >= .70
                    and metrics.get("cue_displacement_diameters", 0) >= 1.0
                    and (straight_launch or early_collision)
                    and metrics.get("anchor_excursion_diameters", 0) >= 1.5
                    and metrics.get("initial_launch_displacement", 0) >= .50
                    and metrics.get("initial_launch_direction", 0) >= .90)

    @staticmethod
    def _stabilized_launch_confirmed(metrics: dict[str, float]) -> bool:
        return bool(metrics.get("stationary_ratio", 0) >= .80
                    and metrics.get("pre_ball_quiet_ratio", 0) >= .80
                    and metrics.get("stable_cue_motion_count", 0) >= 3
                    and ((metrics.get("stable_cue_peak_speed", 0) >= 4
                          and metrics.get("cue_displacement_diameters", 0) >= .75)
                         or (metrics.get("pre_cue_address_score", 0) >= .70
                             and metrics.get("stable_cue_peak_speed", 0) >= 1.5
                             and metrics.get("cue_displacement_diameters", 0) >= .50))
                    and metrics.get("cue_direction_consistency", 0) >= .80
                    and metrics.get("track_confidence", 0) >= .70
                    and metrics.get("pre_cue_address_score", 0) >= .20)

    def _transition_confirmed(self, metrics: dict[str, float]) -> bool:
        if metrics.get("quiet_anchor_contradiction", 0) >= .5 or self._single_step_identity_jump(metrics):
            return False
        if max(metrics.get("gradual_launch_confirmed", 0),
               metrics.get("addressed_departure_confirmed", 0),
               metrics.get("soft_addressed_departure_confirmed", 0),
               metrics.get("measured_slow_departure_confirmed", 0),
               metrics.get("addressed_collision_launch_confirmed", 0),
               metrics.get("camera_drift_launch_confirmed", 0)) >= .5:
            return True
        if self._brief_addressed_launch_confirmed(metrics):
            return True
        if self._short_launch_confirmed(metrics) or self._addressed_slow_launch_confirmed(metrics):
            return True
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
        stabilized_launch = self._stabilized_launch_confirmed(metrics)
        anchored_launch = self._anchored_launch_confirmed(metrics)
        coherent_ball_launch = stabilized_launch or bool(
            metrics["stationary_ratio"] >= 0.80
            and metrics["pre_ball_quiet_ratio"] >= 0.80
            and metrics["post_peak_cue_speed"] >= 4.0
            and ((metrics["cue_displacement_diameters"] >= 1.25
                 and metrics["cue_direction_consistency"] >= 0.80)
                 or (metrics.get("initial_launch_displacement", 0) >= 1.25
                     and metrics.get("initial_launch_direction", 0) >= .80
                     and metrics.get("pre_cue_address_score", 0) >= .70
                     and metrics["cue_contact_score"] >= .40))
            and metrics["track_confidence"] >= 0.70
        )
        # A player starting the cue action can contaminate one coarse residual
        # sample immediately before impact.  Strong cue-at-ball geometry plus a
        # verified white-ball launch may tolerate that isolated foreground
        # sample; ball handling/refereeing without cue contact cannot use this
        # exception.  The quiet median gate below still has to pass.
        pre_motion_quiet = bool(
            anchored_launch or metrics["pre_motion_quiet_ratio"] >= 0.80
            or (
                contact_overrides_object_noise
                and metrics["pre_motion_quiet_ratio"]
                >= self.contact_pre_quiet_min_ratio
            )
            or (coherent_ball_launch and metrics["pre_motion_quiet_ratio"] >= 0.35
                and metrics["pre_motion_raw_median"] <= 0.50)
            or (coherent_ball_launch and metrics.get("pre_cue_address_score", 0) >= 0.70
                and metrics["pre_motion_raw_median"] <= 0.50)
            or (stabilized_launch and metrics["pre_motion_raw_median"] <= .50)
            or (coherent_ball_launch and metrics.get("cue_shape_measured", 0) >= .5
                and metrics.get("pre_sample_count", 0) >= 8
                and metrics.get("pre_measured_cue_quiet_ratio", 0) >= .80
                and metrics.get("pre_ball_quiet_ratio", 0) >= .90
                and metrics["cue_contact_score"] >= .40
                and metrics["pre_motion_raw_median"] <= .50)
        )
        uninterrupted_launch = bool(
            metrics["sustained_run"] >= self.min_sustained_frames
        )
        contact_bridged_launch = bool(
            contact_overrides_object_noise
            and metrics["bridged_sustained_count"] >= self.min_sustained_frames
            and metrics["post_peak_cue_speed"] >= self.contact_bridge_min_peak_speed
        )
        speed_crossed = bool(
            metrics["speed_crossing"] > 0
            or (
                metrics["previous_cue_speed"] <= self.stationary_speed * 2.5
                and metrics["cue_speed"] >= self.start_speed
            )
            or (
                contact_overrides_object_noise
                and metrics["post_peak_cue_speed"] >= self.start_speed
                and metrics.get("cue_track_visible", 1.0) < 0.5
            )
        )
        return bool(
            not self._identity_return(metrics)
            and metrics["stationary_ratio"] >= 0.65
            and metrics["pre_sample_count"] >= 2
            and pre_motion_quiet
            and (anchored_launch or metrics["pre_motion_raw_median"] <= self.pre_quiet_max_motion
                 or (coherent_ball_launch and metrics["pre_motion_raw_median"] <= 0.50))
            # Object-ball Hough tracks can produce an isolated speed spike while
            # the cloth and the real cue ball are visibly still.  Do not let one
            # such identity jump suppress an otherwise complete cue-contact +
            # white-ball-launch sequence.
            and (object_tracks_quiet or contact_overrides_object_noise)
            and speed_crossed
            and (uninterrupted_launch or contact_bridged_launch)
            and metrics["cue_displacement_diameters"] >= 0.50
            and (metrics["cue_direction_consistency"] >= 0.40
                 or (metrics.get("initial_launch_displacement", 0) >= 1.25
                     and metrics.get("initial_launch_direction", 0) >= .95
                     and metrics.get("pre_cue_address_score", 0) >= .70
                     and metrics["cue_contact_score"] >= .40))
            and metrics["track_confidence"] >= self.min_track_conf * 0.90
            and (metrics.get("stable_cue_observed", 0) < .5
                 or (metrics.get("stable_cue_motion_count", 0) >= 2
                     and metrics.get("stable_cue_peak_speed", 0) >= .75*self.start_speed))
        )

    @staticmethod
    def _brief_addressed_launch_confirmed(metrics: dict[str, float]) -> bool:
        """A new angle can show a short measured address before contact."""
        return bool(metrics.get("cue_shape_measured", 0) >= .5
                    and metrics.get("observation_fps", 0) >= 10
                    and metrics.get("post_observation_contiguous", 0) >= .5
                    and 3 <= metrics.get("pre_sample_count", 0) <= 6
                    and metrics.get("stationary_ratio", 0) >= .90
                    and metrics.get("pre_ball_quiet_ratio", 0) >= .90
                    and metrics.get("pre_motion_raw_median", 1) <= .50
                    and metrics.get("pre_cue_address_score", 0) >= .85
                    and metrics.get("cue_contact_score", 0) >= .50
                    and metrics.get("track_confidence", 0) >= .80
                    and metrics.get("stable_cue_motion_count", 0) >= 6
                    and metrics.get("stable_cue_peak_speed", 0) >= .60
                    and metrics.get("initial_launch_displacement", 0) >= .35
                    and metrics.get("initial_launch_direction", 0) >= .95
                    and metrics.get("cue_displacement_diameters", 0) >= .75
                    and metrics.get("cue_direction_consistency", 0) >= .95
                    and metrics.get("anchor_direction_consistency", 0) >= .90
                    and metrics.get("anchor_excursion_diameters", 0) >= 1.)

    @staticmethod
    def _addressed_slow_launch_confirmed(metrics: dict[str, float]) -> bool:
        """Strong cue address and straight ball travel survive foreground flow."""
        return bool(metrics.get("stationary_ratio", 0) >= .90
                    and metrics.get("pre_sample_count", 0) >= 6
                    and metrics.get("pre_ball_quiet_ratio", 0) >= .90
                    # The cue can become clearly visible only at impact. Its
                    # measured contact is as useful as the preceding address,
                    # with the same quiet anchor and stabilized launch checks.
                    and max(metrics.get("pre_cue_address_score", 0),
                            metrics.get("cue_contact_score", 0)) >= .85
                    and metrics.get("pre_motion_raw_median", 1) <= .80
                    and metrics.get("track_confidence", 0) >= .80
                    and metrics.get("speed_crossing", 0) > 0
                    and metrics.get("cue_speed", 0) >= 1.25
                    and metrics.get("sustained_run", 0) >= 3
                    and metrics.get("stable_cue_motion_count", 0) >= 3
                    and metrics.get("stable_cue_peak_speed", 0) >= 1.50
                    and metrics.get("cue_displacement_diameters", 0) >= .40
                    and metrics.get("cue_direction_consistency", 0) >= .95
                    and metrics.get("anchor_excursion_diameters", 0) >= .50
                    and metrics.get("anchor_direction_consistency", 0) >= .90)

    @staticmethod
    def _addressed_collision_launch_confirmed(metrics: dict[str, float]) -> bool:
        """A measured address can launch into a collision within the post window.

        Foreground flow and the collision's bend must not erase the independent
        quiet white, cue geometry and sustained departure. Spatial rest and
        several modest steps keep a moving hand or identity jump out.
        """
        return bool(
            metrics.get("observation_fps", 0) >= 10
            and metrics.get("cue_shape_measured", 0) >= .5
            and metrics.get("post_observation_contiguous", 0) >= .5
            and metrics.get("pre_sample_count", 0) >= 8
            and metrics.get("stationary_ratio", 0) >= .90
            and metrics.get("pre_cue_spatial_quiet_ratio", 0) >= .90
            and metrics.get("pre_ball_quiet_ratio", 0) >= .90
            and metrics.get("pre_motion_raw_median", 1) <= .50
            and max(metrics.get("pre_cue_address_score", 0), metrics.get("cue_contact_score", 0)) >= .85
            and metrics.get("track_confidence", 0) >= .80
            and metrics.get("cue_speed", 0) >= 1.
            and metrics.get("post_peak_cue_speed", 0) >= 3.
            and metrics.get("stable_cue_motion_count", 0) >= 3
            and metrics.get("stable_cue_peak_speed", 0) >= 1.5
            and (metrics.get("cue_displacement_diameters", 0) >= .75
                 or (metrics.get("cue_displacement_diameters", 0) >= .50
                     and metrics.get("initial_launch_displacement", 0) >= .25
                     and metrics.get("initial_launch_direction", 0) >= .90))
            and metrics.get("cue_direction_consistency", 0) >= .65
            and metrics.get("anchor_direction_consistency", 0) >= .65
            and metrics.get("anchor_excursion_diameters", 0) >= .80
            and metrics.get("largest_cue_step_diameters", 0) <= .65)

    @staticmethod
    def _short_launch_confirmed(metrics: dict[str, float]) -> bool:
        """Resolve a short collision at high spatial resolution.

        In an extreme close-up the white can strike a touching colour and stop
        or rebound before travelling half its diameter. Require a measured
        quiet anchor and a coherent initial leg spanning many image pixels;
        neither small-ball jitter nor aggregate camera/player motion qualifies.
        """
        strong_measured_contact = bool(
            metrics.get("observed_ball_diameter_px", metrics.get("ball_diameter_px", 0)) >= 80
            and metrics.get("cue_shape_measured", 0) >= .5
            and metrics.get("pre_cue_spatial_quiet_ratio", 0) >= .90
            and max(metrics.get("pre_cue_address_score", 0), metrics.get("cue_contact_score", 0)) >= .85)
        initial_direction = (.80 if strong_measured_contact
                             and metrics.get("cue_contact_score", 0) >= .85 else .95)
        return bool(metrics.get("observed_ball_diameter_px", metrics.get("ball_diameter_px", 0)) >= 60
                    and metrics.get("stationary_ratio", 0) >= .85
                    and metrics.get("pre_sample_count", 0) >= 6
                    and metrics.get("pre_ball_quiet_ratio", 0) >= .90
                    and metrics.get("pre_motion_raw_median", 1) <= .35
                    and metrics.get("track_confidence", 0) >= .75
                    and metrics.get("cue_speed", 0) >= .50
                    and metrics.get("post_peak_cue_speed", 0) >= 1.50
                    and metrics.get("stable_cue_motion_count", 0) >= 4
                    and metrics.get("stable_cue_peak_speed", 0) >= 1.25
                    and metrics.get("initial_launch_displacement", 0) >= (.12 if strong_measured_contact else .20)
                    and metrics.get("initial_launch_direction", 0) >= initial_direction
                    and metrics.get("anchor_excursion_diameters", 0) >= .35
                    and metrics.get("anchor_direction_consistency", 0) >= .65)

    def _sparse_dense_transition_confirmed(self, metrics: dict[str, float]) -> bool:
        """Relaxed native-rate confirmation for a 2fps proposal.

        A sparse sample can enter the cue launch after the first frame of the
        transition, making the strict stationary-ratio/crossing gates fail even
        though the dense trajectory is coherent.  Require a quiet raw-motion
        median, a strong sustained launch, displacement, and direction so this
        fallback cannot turn a generic residual spike into a shot.
        """
        return bool(
            metrics.get("quiet_anchor_contradiction", 0) < .5
            # A cushion rebound followed by brief occlusion is continuing
            # play. Zero stabilized speed at the reversal is not a new rest.
            and (metrics.get("observation_fps", 0) < 10
                 or metrics.get("pre_cue_spatial_quiet_ratio", 1) >= .65)
            and not self._identity_return(metrics)
            and not self._single_step_identity_jump(metrics)
            and metrics.get("pre_sample_count", 0.0) >= 3
            and metrics.get("stationary_ratio", 0.0) >= 0.50
            and metrics.get("pre_ball_quiet_ratio", 0.0) >= self.fallback_pre_ball_quiet_min_ratio
            and (
                metrics.get("cue_speed", 0.0) >= self.start_speed
                or metrics.get("cue_track_visible", 1.0) < 0.5
            )
            and metrics.get("pre_motion_raw_median", 1.0)
            <= self.pre_quiet_max_motion
            and metrics.get("post_peak_cue_speed", 0.0)
            >= max(3.0, self.start_speed * 2.5)
            and metrics.get("sustained_run", 0.0) >= self.min_sustained_frames
            and metrics.get("cue_displacement_diameters", 0.0) >= 0.75
            and metrics.get("cue_direction_consistency", 0.0) >= 0.35
            and metrics.get("track_confidence", 0.0) >= self.min_track_conf * 0.80
            and (metrics.get("stable_cue_observed", 0) < .5
                 or metrics.get("stable_cue_motion_count", 0) >= 2)
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
        onset_observed = all(self._valid(x) and self._same_view(x, f)
                             for x in features[pre_lo:post_hi])
        pre = [
            x
            for x in features[pre_lo:pre_hi]
            if self._valid(x)
            and self._same_view(x, f)
        ]
        post = [
            x
            for x in features[post_lo:post_hi]
            if self._valid(x)
            and self._same_view(x, f)
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
        pre_cue_stationary_ratio = self._stationary_ratio(reliable_pre)
        post_after_onset = [x for x in post if x.t > t + 1e-6]
        missing_after_onset = any(
            not bool(getattr(x, "cue_ball_detected", False))
            or self._track_conf(x) < self.min_track_conf * 0.70
            for x in post_after_onset
        )
        cue_identity_jump = False
        cue_stationary_after_onset = False
        if reliable_pre:
            anchor = reliable_pre[-1]
            ax = self._value(anchor, "cue_ball_x")
            ay = self._value(anchor, "cue_ball_y")
            diameter = max(self._value(anchor, "ball_diameter_px"), 1.0)
            reliable_post = [x for x in post
                             if x.cue_ball_detected and self._track_conf(x) >= self.min_track_conf
                             and x.cue_ball_x is not None and x.cue_ball_y is not None
                             and .70 <= self._value(x, "ball_diameter_px") / diameter <= 1.40]
            # Missing interleaved observations do not establish an occlusion
            # when the same stationary white is repeatedly seen through the
            # alleged onset. Match its scale because a partial occluder may
            # temporarily inflate the fitted circle. Seeing the same white
            # through half the onset window contradicts that early onset;
            # a later genuine departure remains eligible at its own time.
            cue_stationary_after_onset = bool(
                len(reliable_post) >= 2
                and reliable_post[-1].t - reliable_post[0].t >= .10
                and reliable_post[-1].t >= t + .50 * max(self.post_motion_s, .20) - 1e-6
                and all(np.hypot(x.cue_ball_x-ax, x.cue_ball_y-ay) / diameter <= .25
                        for x in reliable_post)
                and self._stationary_ratio(reliable_post) >= .80)
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
            "onset_observation_contiguous": float(onset_observed),
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
            "onset_camera_motion_median": float(np.median([x.camera_motion_magnitude for x in post])) if post else 0.,
            "onset_cue_address_score": max((x.cue_contact_score for x in pre+post if x.cue_tip_visible),default=0.),
            "pre_cue_sample_count": float(len(reliable_pre)),
            "pre_cue_stationary_ratio": float(pre_cue_stationary_ratio),
            "cue_missing_after_onset": float(missing_after_onset),
            "cue_identity_jump": float(cue_identity_jump),
            "cue_stationary_after_onset": float(cue_stationary_after_onset),
            "cue_observation_disrupted": float(
                missing_after_onset or cue_identity_jump
            ),
        }

    def _ball_onset_confirmed(self, metrics: dict[str, float]) -> bool:
        return bool(
            metrics.get("onset_observation_contiguous", 1.) >= .5
            and metrics["pre_motion_quiet_ratio"] >= 0.80
            and metrics["pre_motion_raw_median"] <= self.pre_quiet_max_motion
            and metrics["pre_ball_quiet_ratio"]
            >= self.fallback_pre_ball_quiet_min_ratio
            and metrics["pre_ball_speed_median"] <= self.pre_quiet_max_ball_speed
            and metrics["pre_cue_sample_count"] >= 2
            and metrics["pre_cue_stationary_ratio"] >= 0.70
            and metrics["cue_observation_disrupted"] >= 0.5
            and metrics.get("cue_stationary_after_onset", 0.0) < .5
            and metrics["ball_onset_run"] >= 2
            and metrics["ball_onset_raw"] >= 0.22
            and metrics["ball_onset_normalized_speed"] >= 1.0
            and (metrics.get("onset_camera_motion_median",0)<=3.
                 or (metrics.get("onset_cue_address_score",0)>=.65
                     and metrics["ball_onset_normalized_speed"]>=6.))
        )

    # ------------------------------------------------------------------ scoring

    def score_frames(self, features: list[FrameFeatures]) -> list[FrameFeatures]:
        if not features:
            return features
        source_features = features
        features = self._stabilized_features(features)

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

            if not self._valid(f):
                score *= 0.10
            if f.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY):
                score *= 0.05
            f.strike_score = float(np.clip(score, 0, 1))
        for i in range(len(features)):
            source_features[i].strike_score = features[i].strike_score
        return source_features

    # ------------------------------------------------------------------ candidates

    def detect_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        if not features:
            return []
        features = self._stabilized_features(features)
        cue_available = self._has_cue_kinematics(features)
        times = [f.t for f in features]
        proposals: list[tuple[int, dict[str, float]]] = []

        if cue_available:
            for i in range(1, len(features)):
                if not self._valid(features[i]):
                    continue
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
                # Visual confirmation supplies the confidence floor.
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
                "table_confidence": f.table_confidence,
                "cue_ball_motion_confirmed": 1.0 if cue_available else 0.0,
                **metrics,
            }
            contact_t, contact_start = self._occluded_contact_time(
                features, idx, times, max(metrics.get("cue_contact_score", 0.0),
                                         metrics.get("pre_cue_address_score", 0.0)),
                stabilized_launch=(self._stabilized_launch_confirmed(metrics)
                                   or self._anchored_launch_confirmed(metrics)))
            if contact_t < f.t:
                evidence["impact_occlusion_contact"] = 1.0
            candidates.append(
                StrikeCandidate(
                    timestamp=contact_t,
                    confidence=float(np.clip(score, 0, 1)),
                    evidence=evidence,
                    uncertainty_start=contact_start if contact_t < f.t else max(0.0, f.t-self.refine_r),
                    uncertainty_end=f.t if contact_t < f.t else f.t+self.refine_r,
                    camera_view=f.view_type,
                    possible_replay=False,
                )
            )
        logger.info("Found %d cue-strike candidates", len(candidates))
        for inferred in (self._camera_contact_candidates(features)
                         + self._address_memory_candidates(features)
                         + self._cut_launch_candidates(features)
                         + self._reacquired_roll_candidates(features)
                         + self._micro_contact_candidates(features)
                         + self._occluded_launch_candidates(features)
                         + self._interrupted_departure_candidates(features)):
            if not any(abs(c.timestamp-inferred.timestamp) < self.min_dist for c in candidates):
                candidates.append(inferred)
        candidates.sort(key=lambda c: c.timestamp)
        return candidates

    def _interrupted_departure_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Join a visible departure to its roll behind a nearby object ball.

        A low camera can lose the white behind a red immediately after cue
        contact, longer than the ordinary post window. Its measured departure,
        nearby reappearance and continued travel together establish the shot.
        """
        times = [f.t for f in features]
        recovered: list[StrikeCandidate] = []

        def positioned(f: FrameFeatures) -> bool:
            return (f.cue_ball_detected and f.cue_ball_x is not None
                    and f.cue_ball_y is not None and self._track_conf(f) >= .65)

        def continuously_observed(rows: list[FrameFeatures]) -> bool:
            return (all(f.observation_fps >= 10 for f in rows)
                    and all(0 < b.t-a.t <= 2/min(a.observation_fps, b.observation_fps)+.005
                            for a, b in zip(rows, rows[1:])))

        for i, current in enumerate(features):
            if (current.observation_fps < 10 or not self._valid(current)
                    or not positioned(current) or self._cue_speed(current) < self.start_speed
                    or self._image_diameter(current) < 24
                    or (recovered and current.t-recovered[-1].timestamp < self.min_dist)):
                continue
            pre = features[bisect_left(times, current.t-.50):bisect_left(times, current.t-.03)]
            if (len(pre) < 6 or any(not self._valid(f) or not self._same_view(f, current) for f in pre)
                    or sum(positioned(f) for f in pre)/len(pre) < .80
                    or self._stationary_ratio(pre) < .90
                    or np.median([f.motion_raw for f in pre]) > .50
                    or sum(f.max_ball_normalized_speed <= self.pre_quiet_max_ball_speed for f in pre)/len(pre) < .90):
                continue
            quiet = [f for f in pre if positioned(f)]
            if len(quiet) < 6 or quiet[-1].t-quiet[0].t < .20:
                continue
            anchor = quiet[-1]
            diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet])))
            anchor_xy = np.asarray([anchor.cue_ball_x, anchor.cue_ball_y])
            if max(np.hypot(f.cue_ball_x-anchor_xy[0], f.cue_ball_y-anchor_xy[1]) for f in quiet) > .20*diameter:
                continue
            measured_address = bool(current.cue_ball_quality
                                    and sum(f.cue_ball_quality for f in quiet) >= .80*len(quiet)
                                    and max((f.cue_contact_score for f in quiet
                                             if f.cue_tip_visible), default=0.) >= .20)
            post = []
            for f in features[i:bisect_right(times, current.t+(1.6 if measured_address else 1.))]:
                if not self._valid(f) or not self._same_view(f, current):
                    break
                post.append(f)
            # An observed hidden ball is different from missing source images.
            # Require native-rate coverage on both sides and through the gap.
            if not continuously_observed(pre+post):
                continue
            gap_index = next((j for j, f in enumerate(post) if not positioned(f)), None)
            if gap_index is None or gap_index < 2 or post[gap_index].t-current.t > (.25 if measured_address else .15):
                continue
            initial = post[:gap_index]
            if (any(f.cue_ball_stable_normalized_speed is None or f.cue_ball_stable_normalized_speed < .30 for f in initial)
                    or np.hypot(current.cue_ball_x-anchor_xy[0], current.cue_ball_y-anchor_xy[1]) < max(2., .05*diameter)):
                continue
            return_index = next((j for j in range(gap_index+1, len(post)) if positioned(post[j])), None)
            if return_index is None:
                continue
            reappearance = post[return_index]
            if (not .20 <= reappearance.t-post[gap_index].t <= .65
                    or reappearance.t-current.t > .75):
                continue
            tail = post[return_index:]
            visible = [f for f in tail if positioned(f)]
            if (len(visible) < 4 or len(visible)/len(tail) < .75
                    or visible[-1].t-visible[0].t < .16):
                continue
            stable = [f.cue_ball_stable_normalized_speed for f in visible
                      if f.cue_ball_stable_normalized_speed is not None]
            if sum(v >= self.continue_speed for v in stable) < 3 or max(stable, default=0) < 2.:
                continue
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in visible])
            net = float(np.linalg.norm(xy[-1]-xy[0]))
            path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            excursion = float(np.linalg.norm(xy[-1]-anchor_xy))/diameter
            coherent = net/max(path, 1e-6) >= .80
            if measured_address and not coherent and len(visible) >= 12:
                centred = xy-np.mean(xy, axis=0)
                _, singular, axes = np.linalg.svd(centred, full_matrices=False)
                axis = axes[0] * (1 if np.dot(xy[-1]-xy[0], axes[0]) >= 0 else -1)
                progress = centred@axis
                perpendicular = centred-progress[:, None]*axis
                coherent = bool(
                    all(f.cue_ball_quality for f in visible)
                    and singular[0]**2/max(float(np.sum(singular**2)), 1e-6) >= .95
                    and np.max(np.linalg.norm(perpendicular, axis=1))/diameter <= .15
                    and np.min(np.diff(progress))/diameter >= -.12
                    and np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))/diameter <= .35)
            if (net/diameter < .65 or excursion < .85 or not coherent
                    or np.linalg.norm(xy[0]-anchor_xy) > .75*diameter):
                continue
            recovered.append(StrikeCandidate(
                timestamp=current.t, confidence=.78, uncertainty_start=anchor.t,
                uncertainty_end=current.t, camera_view=current.view_type,
                evidence={"occlusion_inferred": 1., "ball_onset_run": float(len(visible)),
                          "interrupted_cue_departure": 1., "cue_ball_motion_confirmed": 1.,
                          "cue_geometry_confirmed": 0., "cue_displacement_diameters": net/diameter,
                          "cue_direction_consistency": net/max(path, 1e-6),
                          "anchor_excursion_diameters": excursion,
                          "occlusion_duration_seconds": reappearance.t-post[gap_index].t},
            ))
        return recovered

    def _occluded_launch_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Reconnect a quiet addressed white to a coherent roll after occlusion.

        The normal half-second transition window can contain no white at all
        when the bridge/cue covers it before contact. Keep the last measured
        quiet anchor for a bounded interval. A continuous native-rate roll,
        rather than a moving hand or one displaced detection, must follow it.
        Its first visible movement is an upper bound on the hidden contact.
        """
        times = [f.t for f in features]
        recovered: list[StrikeCandidate] = []

        def positioned(f: FrameFeatures, confidence: float = .40) -> bool:
            return (f.cue_ball_detected and f.cue_ball_x is not None
                    and f.cue_ball_y is not None and self._track_conf(f) >= confidence)

        for i, current in enumerate(features):
            if (current.observation_fps < 10 or not self._valid(current)
                    or not positioned(current) or self._cue_speed(current) < self.start_speed
                    or (recovered and current.t-recovered[-1].timestamp < self.min_dist)):
                continue
            history = features[bisect_left(times, current.t-1.5):i]
            quiet = [f for f in history if self._valid(f) and self._same_view(f, current)
                     and positioned(f, .65)
                     and (f.cue_ball_observations == 0 or (f.cue_ball_observations >= 3 and f.cue_ball_quality))
                     and (f.cue_ball_stable_normalized_speed
                          if f.cue_ball_stable_normalized_speed is not None
                          else self._cue_speed(f)) <= self.stationary_speed]
            if not quiet:
                continue
            anchor = quiet[-1]
            if not .15 <= current.t-anchor.t <= 1.2:
                continue
            cluster = [f for f in history if anchor.t-.5 <= f.t <= anchor.t
                       and self._valid(f) and self._same_view(f, current) and positioned(f, .65)]
            if len(cluster) >= 4 and any(f.cue_ball_observations > 0 for f in cluster):
                reference = np.median([(f.cue_ball_x, f.cue_ball_y) for f in cluster], axis=0)
                reference_diameter = max(1., float(np.median([f.ball_diameter_px for f in cluster])))
                trusted = [f for f in quiet if f.t <= anchor.t
                           and np.linalg.norm(np.asarray([f.cue_ball_x, f.cue_ball_y])-reference)
                           <= .12*reference_diameter]
                if not trusted:
                    continue
                anchor = trusted[-1]
                cluster = [f for f in history if anchor.t-.5 <= f.t <= anchor.t
                           and self._valid(f) and self._same_view(f, current) and positioned(f, .65)]
            measured = (current.cue_ball_quality
                        and sum(f.cue_ball_quality for f in cluster) >= .50*len(cluster))
            pre_address = max((f.cue_contact_score for f in cluster if f.cue_tip_visible), default=0)
            if (len(cluster) < 4 or cluster[-1].t-cluster[0].t < .20
                    or self._stationary_ratio(cluster) < .80
                    or pre_address < (.35 if measured else .65)):
                continue
            hidden = [f for f in history if f.t > anchor.t]
            if (not hidden or any(not self._valid(f) or not self._same_view(f, current) for f in hidden)
                    or sum(not positioned(f) for f in hidden)/len(hidden) < .60):
                continue
            post = features[i:bisect_right(times, current.t+(.65 if measured else .30))]
            if (len(post) < 4 or any(not self._valid(f) or not self._same_view(f, current) for f in post)
                    or any(b.t-a.t > 2/current.observation_fps+.005 for a, b in zip(post, post[1:]))):
                continue
            visible = [f for f in post if positioned(f)]
            if len(visible) < 4 or len(visible)/len(post) < (.65 if measured else .75):
                continue
            if pre_address < .65 and max((f.cue_contact_score for f in visible if f.cue_tip_visible), default=0) < .75:
                continue
            stable = [f.cue_ball_stable_normalized_speed for f in visible
                      if f.cue_ball_stable_normalized_speed is not None]
            if sum(speed >= self.continue_speed for speed in stable) < 3 or max(stable, default=0) < (.75 if measured else 1.5):
                continue
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in visible])
            diameter = max(1., float(np.median([f.ball_diameter_px for f in cluster+visible])))
            displacement = float(np.linalg.norm(xy[-1]-xy[0]))/diameter
            path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            direction = float(np.linalg.norm(xy[-1]-xy[0]))/max(path, 1e-6)
            coherent = direction >= .90
            if measured and len(xy) >= 8 and not coherent:
                centred = xy-np.mean(xy, axis=0)
                _, singular, axes = np.linalg.svd(centred, full_matrices=False)
                axis = axes[0] * (1 if np.dot(xy[-1]-xy[0], axes[0]) >= 0 else -1)
                progress = centred@axis
                perpendicular = centred-progress[:, None]*axis
                coherent = bool(singular[0]**2/max(float(np.sum(singular**2)), 1e-6) >= .90
                                and np.max(np.linalg.norm(perpendicular, axis=1))/diameter <= .15
                                and np.min(np.diff(progress))/diameter >= -.12)
            anchor_xy = np.asarray([anchor.cue_ball_x, anchor.cue_ball_y])
            excursion = float(np.linalg.norm(xy[-1]-anchor_xy))/diameter
            if displacement < (.50 if measured else .65) or excursion < .80 or not coherent:
                continue
            # Reacquisition must start near the measured ball and then depart.
            # A remote white logo or glove cannot replace the missing identity.
            remote_roll = bool(measured and current.table_full_view and anchor.table_full_view
                               and current.t-anchor.t <= .70
                               and max(f.cue_contact_score for f in cluster) >= .80
                               and all(f.cue_ball_quality and f.table_full_view for f in visible)
                               and .70 <= current.ball_diameter_px/max(anchor.ball_diameter_px, 1.) <= 1.40
                               and coherent)
            if np.linalg.norm(xy[0]-anchor_xy) > .75*diameter and not remote_roll:
                continue
            recovered.append(StrikeCandidate(
                timestamp=current.t, confidence=.78,
                uncertainty_start=anchor.t, uncertainty_end=current.t,
                camera_view=current.view_type,
                evidence={"occlusion_inferred": 1., "ball_onset_run": float(len(visible)),
                          "occluded_anchor_launch": 1., "contact_time_upper_bound": 1.,
                          "cue_ball_motion_confirmed": 1., "cue_geometry_confirmed": 1.,
                          "anchor_gap_seconds": current.t-anchor.t,
                          "cue_displacement_diameters": displacement,
                          "cue_direction_consistency": direction},
            ))
        return recovered

    def _occluded_contact_time(self, features: list[FrameFeatures], index: int,
                               times: list[float], confirmed_contact: float = 0.0,
                               stabilized_launch: bool = False) -> tuple[float, float]:
        """Bound short impact occlusion by the final still and first rolling white."""
        current = features[index]
        lo = bisect_left(times, current.t-(.35 if stabilized_launch else .20))
        recent = features[lo:index+1]
        if not recent or any(not self._same_view(f, current) or not self._valid(f) for f in recent):
            return current.t, current.t
        if max(current.cue_contact_score, confirmed_contact,
               max(f.cue_contact_score for f in recent)) < (.20 if stabilized_launch else .40):
            return current.t, current.t
        for anchor_index in range(len(recent)-2, -1, -1):
            anchor = recent[anchor_index]
            anchor_speed = (anchor.cue_ball_stable_normalized_speed
                            if stabilized_launch and anchor.cue_ball_stable_normalized_speed is not None
                            else self._cue_speed(anchor))
            if (anchor.cue_ball_detected and anchor_speed <= self.stationary_speed
                and self._track_conf(anchor) >= 0.65):
                onset = recent[anchor_index+1]
                if not onset.cue_ball_detected or self._cue_speed(onset) >= self.start_speed:
                    return onset.t, anchor.t
                break
        return current.t, current.t

    def _micro_contact_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """A touching-ball pot can launch the colour with minimal white travel."""
        if not any(f.object_ball_launch_count for f in features):
            return []
        times = [f.t for f in features]
        recovered = []
        for i, current in enumerate(features):
            if (current.observation_fps < 10 or not self._valid(current)
                    or not current.cue_ball_quality or not current.cue_ball_detected
                    or self._image_diameter(current) < 60
                    or (recovered and current.t-recovered[-1].timestamp < self.min_dist)):
                continue
            pre = features[bisect_left(times, current.t-.50):i]
            post = features[i:bisect_right(times, current.t+1.20)]
            if (len(pre) < 8 or len(post) < 12
                    or any(not self._valid(f) or not self._same_view(f,current)
                           or f.observation_fps < 10 for f in pre+post)
                    or any(b.t-a.t > 2/min(a.observation_fps,b.observation_fps)+.005
                           for a,b in zip(pre+post,(pre+post)[1:]))):
                continue
            quiet = [f for f in pre if f.cue_ball_quality and f.cue_ball_detected
                     and self._track_conf(f) >= .75 and f.cue_ball_x is not None and f.cue_ball_y is not None]
            if len(quiet) < .75*len(pre) or self._stationary_ratio(quiet) < .90:
                continue
            diameter = max(1.,float(np.median([f.ball_diameter_px for f in quiet])))
            xy = np.asarray([(f.cue_ball_x,f.cue_ball_y) for f in quiet])
            anchor = np.median(xy,axis=0)
            if np.mean(np.linalg.norm(xy-anchor,axis=1)/diameter <= .12) < .90:
                continue
            if max((f.cue_contact_score for f in quiet+post if f.cue_tip_visible),default=0) < .85:
                continue
            launches = [f for f in post if f.object_ball_launch_count > 0]
            if len(launches) < 3 or launches[-1].t-launches[0].t < .08:
                continue
            departures = [f for f in post if f.cue_ball_detected and f.cue_ball_quality
                          and self._track_conf(f) >= .75
                          and f.cue_ball_x is not None and f.cue_ball_y is not None
                          and np.linalg.norm(np.asarray([f.cue_ball_x,f.cue_ball_y])-anchor)
                          >= max(3.,.05*diameter)]
            if not departures or departures[0].t > launches[0].t:
                continue
            onset = departures[0]
            recovered.append(StrikeCandidate(
                timestamp=onset.t,confidence=.75,camera_view=onset.view_type,
                uncertainty_start=quiet[-1].t,uncertainty_end=launches[0].t,
                evidence={"micro_contact_object_launch":1.,"occlusion_inferred":1.,
                          "dense_transition_confirmed":1.,"cue_geometry_confirmed":1.,
                          "ball_onset_run":float(len(launches)),"contact_time_upper_bound":1.},
            ))
        return recovered

    def _reacquired_roll_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Keep a shot already rolling out of an observed, quiet occlusion.

        A hidden white supplies no stationary measurement. Require newly
        measured ball-quality observations and a sustained, coherent roll;
        partial views additionally need visible cue-contact geometry. Record
        an interval for the hidden contact instead of assigning exact impact.
        """
        times = [f.t for f in features]
        recovered = []
        for i, current in enumerate(features):
            if (i == 0 or not current.cue_ball_quality or not self._valid(current)
                    or not current.cue_ball_detected
                    or (features[i-1].cue_ball_detected and features[i-1].cue_ball_quality
                        and self._track_conf(features[i-1]) >= .65)
                    or current.observation_fps < 10 or self._track_conf(current) < .65
                    or (recovered and current.t-recovered[-1].timestamp < self.min_dist)):
                continue
            pre = features[bisect_left(times, current.t-.70):i]
            post = []
            # A soft safety can emerge from a ball/bridge occlusion and take
            # half a second to establish its short, coherent roll. Preserve
            # the same displacement and physical-motion proof over that span.
            for f in features[i:bisect_right(times, current.t+.55)]:
                if not self._valid(f) or not self._same_view(f, current):
                    break
                post.append(f)
            if (len(pre) < 12 or pre[-1].t-pre[0].t < .50 or len(post) < 8
                    or post[-1].t-post[0].t < .35
                    or any(not self._valid(f) or not self._same_view(f, current)
                           or f.observation_fps < 10 for f in pre+post)
                    or any(b.t-a.t > 2/min(a.observation_fps, b.observation_fps)+.005
                           for a, b in zip(pre+post, (pre+post)[1:]))):
                continue
            if (sum(f.cue_ball_detected and f.cue_ball_quality and self._track_conf(f) >= .75
                    for f in pre) > .20*len(pre)
                    or np.median([f.max_ball_normalized_speed for f in pre]) > 1.
                    or sum(f.max_ball_normalized_speed <= self.pre_quiet_max_ball_speed
                           for f in pre) < .75*len(pre)):
                continue
            visible = [f for f in post if f.cue_ball_quality and f.cue_ball_detected
                       and self._track_conf(f) >= .75
                       and f.cue_ball_x is not None and f.cue_ball_y is not None]
            if len(visible) < 8 or len(visible)/len(post) < .75:
                continue
            diameter = max(1., float(np.median([f.ball_diameter_px for f in visible])))
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in visible])
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            net = float(np.linalg.norm(xy[-1]-xy[0]))
            direction = net/max(float(np.sum(steps)), 1e-6)
            initial_leg = False
            if (direction < .90 and len(visible) >= 8
                    and visible[7].t-current.t >= .35):
                initial_xy = xy[:8]
                initial_steps = np.linalg.norm(np.diff(initial_xy, axis=0), axis=1)
                initial_net = float(np.linalg.norm(initial_xy[-1]-initial_xy[0]))
                initial_direction = initial_net/max(float(np.sum(initial_steps)), 1e-6)
                if initial_net/diameter >= .65 and initial_direction >= .95:
                    # The white can collide or settle before the longer window
                    # ends. Its already measured initial roll remains proof.
                    visible, xy, steps = visible[:8], initial_xy, initial_steps
                    net, direction = initial_net, initial_direction
                    initial_leg = True
            stable = [f.cue_ball_stable_normalized_speed or 0. for f in visible[2:]]
            addressed_partial_roll = bool(
                not current.table_full_view and self._image_diameter(current) >= 12 and net/diameter >= 2.5
                and direction >= .95 and max(stable, default=0) >= 10.
                and current.cue_tip_visible
                and 0 < current.cue_tip_distance_to_ball <= 2*self._image_diameter(current)
                and all(f.camera_motion_magnitude < 3. for f in visible))
            if (net/diameter < .65 or direction < .90 or max(steps)/diameter > 1.5
                    or sum(v >= 1.5 for v in stable) < (.40 if addressed_partial_roll else .70)*len(stable)):
                continue
            contact = max((f.cue_contact_score for f in visible if f.cue_tip_visible), default=0)
            if np.median([f.motion_raw for f in pre]) > .50 and contact < .85:
                continue
            if contact < .45 and not (addressed_partial_roll or
                                      current.table_full_view and net/diameter >= 1.75
                                      and max(stable, default=0) >= 6.):
                continue
            moving = next((f for f in visible[1:] if (f.cue_ball_stable_normalized_speed or 0.) >= 1.5), None)
            if moving is None or moving.t-current.t > .15:
                continue
            recovered.append(StrikeCandidate(
                timestamp=current.t, confidence=.68, camera_view=current.view_type,
                uncertainty_start=pre[0].t, uncertainty_end=moving.t,
                evidence={"occlusion_inferred": 1., "reacquired_ball_roll": 1.,
                          "reacquired_initial_leg": float(initial_leg),
                          "contact_time_upper_bound": 1., "ball_onset_run": float(len(visible)),
                          "dense_transition_confirmed": 1., "cue_ball_motion_confirmed": 1.,
                          "cue_geometry_confirmed": float(contact >= .45),
                          "cue_displacement_diameters": net/diameter,
                          "cue_direction_consistency": direction},
            ))
        return recovered

    def _cut_launch_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Link a measured cue address directly to a roll at an angle change.

        The white need not disappear in the old view before the director cuts.
        Establish stillness and cue geometry there, then independently measure
        an immediate launch and cue contact in the destination. Coordinates
        never cross the camera boundary; the hidden impact stays uncertain.
        """
        times = [f.t for f in features]
        recovered = []

        def positioned(f):
            return (f.cue_ball_detected and self._track_conf(f) >= .65
                    and f.cue_ball_x is not None and f.cue_ball_y is not None)

        def continuous(rows):
            return (all(f.observation_fps >= 10 for f in rows)
                    and all(0 < b.t-a.t <= 2/min(a.observation_fps, b.observation_fps)+.005
                            for a, b in zip(rows, rows[1:])))

        for i, cut in enumerate(features):
            if (i == 0 or cut.scene_cut_score < self.hard_cut_threshold
                    or cut.camera_scene_id == features[i-1].camera_scene_id
                    or cut.observation_fps < 10 or not cut.table_observable
                    or not cut.match_context_valid or cut.table_handling or cut.broadcast_replay
                    or cut.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY)):
                continue
            before = features[bisect_left(times, cut.t-.55):i]
            if (len(before) < 6 or not continuous(before+[cut])
                    or any(not self._valid(f) or not self._same_view(f, before[-1]) for f in before)):
                continue
            quiet = [f for f in before if positioned(f)]
            if (len(quiet) < 4 or len(quiet)/len(before) < .70
                    or quiet[-1].t-quiet[0].t < .15
                    or cut.t-quiet[-1].t > .12 or self._stationary_ratio(quiet) < .90):
                continue
            diameter = max(1., float(np.median([f.ball_diameter_px for f in quiet])))
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in quiet])
            if np.max(np.linalg.norm(xy-np.median(xy, axis=0), axis=1)) > .20*diameter:
                continue
            address = [f for f in quiet if f.cue_tip_visible
                       and f.cue_tip_distance_to_ball <= 2*max(1., f.ball_diameter_px)
                       and f.cue_contact_score >= .50]
            if (not address or max(f.cue_contact_score for f in address) < .80
                    or np.median([f.motion_raw for f in before]) > .80):
                continue
            post = features[i:bisect_right(times, cut.t+.45)]
            if (len(post) < 7 or not continuous(before+post)
                    or any(not self._same_view(f, cut) for f in post)
                    or any(not self._valid(f) for f in post[1:])):
                continue
            visible = [f for f in post if positioned(f)]
            if (len(visible) < 6 or len(visible)/len(post) < .80
                    or visible[0].t-cut.t > .08
                    or visible[-1].t-visible[0].t < .25):
                continue
            early = [f for f in visible if f.t <= cut.t+.15]
            if (max((f.cue_contact_score for f in early if f.cue_tip_visible), default=0) < .30
                    or sum((f.cue_ball_stable_normalized_speed or 0.) >= 1.5 for f in early) < 2):
                continue
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in visible])
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            diameter = max(1., float(np.median([f.ball_diameter_px for f in visible])))
            net = float(np.linalg.norm(xy[-1]-xy[0]))
            direction = net/max(float(np.sum(steps)), 1e-6)
            if net/diameter < .80 or direction < .90 or max(steps)/diameter > 3.:
                continue
            if sum((f.cue_ball_stable_normalized_speed or 0.) >= self.start_speed
                   for f in visible[2:]) < .8*len(visible[2:]):
                continue
            recovered.append(StrikeCandidate(
                timestamp=cut.t, confidence=.78, camera_view=cut.view_type,
                uncertainty_start=quiet[-1].t, uncertainty_end=early[-1].t,
                evidence={"camera_contact_inferred": 1., "cut_address_launch": 1.,
                          "contact_time_upper_bound": 1., "occlusion_inferred": 1.,
                          "dense_transition_confirmed": 1., "cue_geometry_confirmed": 1.,
                          "ball_onset_run": float(len(visible)),
                          "cross_view_launch_displacement": net/diameter,
                          "cue_direction_consistency": direction},
            ))
        return recovered

    def _address_memory_candidates(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Retain a quiet cue address when the next angle hides the white.

        The destination must remain observed and quiet until its first visible
        white immediately rolls away from the cue. Coordinates are compared
        only within each view. The reacquisition time is an upper bound on the
        hidden contact, with uncertainty back to the last visible quiet white.
        """
        times = [f.t for f in features]
        recovered: list[StrikeCandidate] = []
        for index, cut in enumerate(features):
            if index == 0 or cut.scene_cut_score < self.hard_cut_threshold:
                continue
            if cut.camera_scene_id == features[index-1].camera_scene_id:
                continue
            if (cut.observation_fps < 10 or not cut.match_context_valid
                    or cut.broadcast_replay or cut.table_handling
                    or cut.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY)):
                continue
            before = features[bisect_left(times, cut.t-.6):index]
            if len(before) < 6 or before[-1].t-before[0].t < .4:
                continue
            if not all(self._valid(f) and self._same_view(f, before[-1])
                       and f.observation_fps >= 10 and f.cue_ball_detected
                       and self._track_conf(f) >= .65
                       and f.cue_ball_x is not None and f.cue_ball_y is not None for f in before):
                continue
            if self._stationary_ratio(before) < .90:
                continue
            if float(np.median([f.motion_raw for f in before])) > self.pre_quiet_max_motion:
                continue
            if sum(f.max_ball_normalized_speed <= self.pre_quiet_max_ball_speed
                   for f in before) < .9*len(before):
                continue
            if sum(f.cue_tip_visible and f.cue_tip_distance_to_ball <= 2*max(f.ball_diameter_px, 1)
                   and f.cue_contact_score >= .35 for f in before) < 2:
                continue

            hidden: list[FrameFeatures] = []
            onset_index = None
            for j in range(index+1, bisect_right(times, cut.t+4.)):
                frame = features[j]
                if (not self._valid(frame) or not self._same_view(frame, cut)
                        or frame.scene_cut_score >= self.hard_cut_threshold):
                    break
                if frame.cue_ball_detected:
                    onset_index = j
                    break
                hidden.append(frame)
            if onset_index is None or not hidden or features[onset_index].t-cut.t < .5:
                continue
            onset = features[onset_index]
            quiet = [f for f in hidden if f.t >= onset.t-.6]
            if len(quiet) < 6 or quiet[-1].t-quiet[0].t < .4:
                continue
            if float(np.median([f.motion_raw for f in quiet])) > self.pre_quiet_max_motion:
                continue
            if sum(f.max_ball_normalized_speed <= 1. for f in quiet) < .8*len(quiet):
                continue
            post = features[onset_index:bisect_right(times, onset.t+.55)]
            if len(post) < 6 or post[-1].t-post[0].t < .4:
                continue
            if not all(self._valid(f) and self._same_view(f, onset)
                       and f.scene_cut_score < self.hard_cut_threshold
                       and f.cue_ball_detected and self._track_conf(f) >= .65
                       and f.cue_ball_x is not None and f.cue_ball_y is not None for f in post):
                continue
            observed = before + [cut] + hidden + post
            if any(f.observation_fps < 10 for f in observed):
                continue
            if any(b.t-a.t > 2/min(a.observation_fps, b.observation_fps)+.005
                   for a, b in zip(observed, observed[1:])):
                continue
            # Reacquisition must already be moving, not a new stationary
            # address which happens to produce a stroke later in this window.
            if any((f.cue_ball_stable_normalized_speed or 0.) < 2. for f in post[1:3]):
                continue
            diameter = max(1., float(np.median([f.ball_diameter_px for f in post])))
            xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in post])
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            distance = float(np.linalg.norm(xy[-1]-xy[0]))
            displacement = distance/diameter
            direction = distance/max(float(np.sum(steps)), 1e-6)
            if displacement < 1. or direction < .85 or max(steps)/diameter > 1.:
                continue
            if sum((f.cue_ball_stable_normalized_speed or 0.) >= 2.
                   for f in post[1:]) < .8*(len(post)-1):
                continue
            early = [f for f in post if f.t <= onset.t+.15]
            if not any(f.cue_tip_visible and f.cue_contact_score >= .65
                       and f.cue_tip_distance_to_ball <= max(f.ball_diameter_px, 1) for f in early):
                continue
            recovered.append(StrikeCandidate(
                timestamp=onset.t, confidence=.72, camera_view=onset.view_type,
                uncertainty_start=before[-1].t, uncertainty_end=onset.t,
                evidence={"camera_contact_inferred": 1., "camera_address_memory": 1.,
                          "occlusion_inferred": 1., "cue_geometry_confirmed": 1.,
                          "dense_transition_confirmed": 1., "ball_onset_run": float(len(post)),
                          "address_memory_seconds": onset.t-before[-1].t,
                          "reacquired_roll_displacement": displacement,
                          "reacquired_roll_direction": direction},
            ))
        return recovered

    def _camera_contact_candidates(self, features: list[FrameFeatures], sparse: bool = False) -> list[StrikeCandidate]:
        """Associate an impact hidden in a low view with rolling in the next view.

        Never subtract coordinates between cameras. Require cue-address evidence,
        a stationary white before its disappearance, a new local motion burst,
        and a coherent moving white reacquired after a cut. This is inferred
        contact with an explicit uncertainty interval, not an exact visible tap.
        """
        times = [f.t for f in features]
        result = []
        for index, cut in enumerate(features):
            if index == 0 or cut.scene_cut_score < 0.5:
                continue
            old_scene = features[index-1].camera_scene_id
            prior = features[bisect_left(times, cut.t-2.5):index]
            visible = [f for f in prior if self._valid(f) and f.camera_scene_id == old_scene
                       and f.cue_ball_detected and self._track_conf(f) >= 0.65]
            if not visible:
                continue
            anchor = visible[-1]
            if not 0.05 <= cut.t-anchor.t <= (1.8 if sparse else 1.2):
                continue
            quiet = [f for f in visible if anchor.t-0.55 <= f.t <= anchor.t]
            if len(quiet) < 2 or self._stationary_ratio(quiet) < 0.80:
                continue
            if sum(f.max_ball_normalized_speed <= self.pre_quiet_max_ball_speed for f in quiet)/len(quiet) < 0.80:
                continue
            if not any(f.cue_tip_visible and f.cue_tip_distance_to_ball <= 3.0*max(f.ball_diameter_px, 1)
                       for f in quiet):
                continue
            hidden = [f for f in prior if f.t > anchor.t and f.camera_scene_id == old_scene
                      and self._valid(f) and not f.cue_ball_detected]
            if len(hidden) < (1 if sparse else 2):
                continue
            baseline = float(np.median([f.motion_raw for f in quiet]))
            bursts = [f for f in hidden if f.motion_raw >= max(0.60, baseline+0.30)
                      and f.max_ball_normalized_speed >= 1.0]
            if not bursts and not sparse:
                continue
            post = [f for f in features[index:bisect_right(times, cut.t+(1.5 if sparse else 0.8))]
                    if self._valid(f) and f.camera_scene_id == cut.camera_scene_id
                    and f.cue_ball_detected and self._track_conf(f) >= 0.65
                    and f.cue_ball_x is not None and f.cue_ball_y is not None]
            if len(post) < 3:
                continue
            # A hidden impact must already be rolling when it is reacquired.
            # Quiet reacquisition followed by a later launch is another shot.
            initial = post[:3]
            initial_xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in initial])
            sparse_steps = np.linalg.norm(np.diff(initial_xy, axis=0), axis=1)
            if not sparse and sum(f.cue_ball_normalized_speed >= self.start_speed for f in initial[1:]) < 2:
                continue
            if sparse and max(sparse_steps) < .5*max(1, initial[-1].ball_diameter_px):
                continue
            xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in post])
            displacement = float(np.linalg.norm(xy[-1]-xy[0]))
            path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            diameter = max(1.0, float(np.median([f.ball_diameter_px for f in post])))
            if displacement < diameter or displacement/max(path, 1e-6) < 0.65:
                continue
            if not sparse and max(f.cue_ball_normalized_speed for f in post) < self.start_speed:
                continue
            event = bursts[0] if bursts else hidden[0]
            evidence = {"camera_contact_inferred": 1.0, "occlusion_inferred": 1.0,
                        "ball_onset_run": float(len(bursts)), "cue_geometry_confirmed": 1.0,
                        "cross_view_launch_displacement": displacement/diameter,
                        "dense_transition_confirmed": 0.0 if sparse else 1.0,
                        "sparse_proposal": float(sparse)}
            result.append(StrikeCandidate(timestamp=event.t, confidence=0.65, evidence=evidence,
                                          uncertainty_start=anchor.t, uncertainty_end=post[0].t,
                                          camera_view=event.view_type))
        return result

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
            pre = [values[j] for j in range(pre_lo, pre_hi)
                   if self._valid(features[j]) and self._same_view(features[j], f)]
            post = [values[j] for j in range(i, post_hi)
                    if self._valid(features[j]) and self._same_view(features[j], f)]
            if len(pre) < 1 or len(post) < self.sparse_min_active:
                continue
            # Max-ball-speed is intentionally down-weighted above: sparse
            # Hough tracks often report a large identity jump while the table
            # is still.  A median quiet gate tolerates that isolated jitter.
            pre_median = float(np.median(pre))
            previous_activity = values[i - 1] if i > 0 else 0.0
            rising = values[i] - previous_activity >= 0.20
            cue_speed_curr = self._value(f, "cue_ball_normalized_speed")
            cue_speed_prev = self._value(features[i - 1], "cue_ball_normalized_speed") if i > 0 else 0.0
            cue_rising = (
                cue_speed_curr >= self.start_speed * 0.75
                and cue_speed_prev < self.start_speed * 0.50
            )
            ball_residual_curr = self._value(f, "ball_residual_motion")
            ball_residual_prev = self._value(features[i - 1], "ball_residual_motion") if i > 0 else 0.0
            ball_rising = ball_residual_curr >= 0.25 and ball_residual_curr - ball_residual_prev >= 0.12

            active = [v for v in post if v >= self.sparse_activity_threshold]
            ball_active = sum(
                self._value(x, "max_ball_normalized_speed") >= 1.5
                or self._value(x, "moving_ball_count") >= 1
                or self._value(x, "ball_residual_motion") >= self.sparse_activity_threshold
                or self._value(x, "cue_ball_normalized_speed") >= self.start_speed * 0.75
                for x in features[i:post_hi]
            )
            peak = max(post or [0.0])

            onset_from_quiet = bool(
                pre_median <= self.sparse_activity_threshold
                and values[i] >= self.sparse_activity_threshold
                and values[i] - pre_median >= 0.15
            )
            has_launch_onset = cue_rising or ball_rising or (rising and cue_speed_curr >= self.start_speed * 0.50)
            quiet = pre_median <= (0.45 if has_launch_onset else self.pre_quiet_max_motion)

            cue_pre = [x for x in features[pre_lo:pre_hi]
                       if self._valid(x) and self._same_view(x, f) and x.cue_ball_x is not None
                       and x.cue_ball_y is not None and self._track_conf(x) >= self.min_track_conf]
            cue_quiet = (len(cue_pre) >= 2 and
                         sum(self._cue_speed(x) <= self.stationary_speed for x in cue_pre)
                         / len(cue_pre) >= 0.7)
            # At 2 fps a fast white ball can exceed the tracker's association
            # distance and report zero speed despite clear centre displacement.
            # Consecutive visible positions still warrant native verification.
            spatial_steps = []
            for j in range(i, post_hi):
                left = features[j - 1] if j else None
                right = features[j]
                spatial_steps.append(bool(
                    left is not None and 0 < right.t - left.t <= 0.75
                    and self._valid(left) and self._valid(right)
                    and self._same_view(left, right)
                    and all(x.cue_ball_x is not None and x.cue_ball_y is not None
                            and self._track_conf(x) >= self.min_track_conf for x in (left, right))
                    and np.hypot(right.cue_ball_x - left.cue_ball_x,
                                 right.cue_ball_y - left.cue_ball_y)
                    >= 0.5 * max(1.0, right.ball_diameter_px)
                ))
            cue_launch_proposal = (cue_quiet and (cue_rising or spatial_steps[0])
                                   and max(ball_active, sum(spatial_steps)) >= self.sparse_min_active)
            # Foreground movement must not hide a stationary-white launch.
            # This only opens a native verification window, never confirms it.
            if not cue_launch_proposal and (
                not quiet
                or not (rising or cue_rising or ball_rising or onset_from_quiet)
                or len(active) < self.sparse_min_active
                or ball_active < self.sparse_min_active
            ):
                continue
            score = float(
                np.clip(
                    0.40
                    + 0.20 * min(1.0, peak)
                    + 0.15 * min(1.0, ball_active / 3.0)
                    + 0.10 * float(np.clip(f.table_confidence, 0.0, 1.0)),
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
                        "sparse_pre_quiet_median": pre_median,
                    },
                    uncertainty_start=max(0.0, f.t - radius),
                    uncertainty_end=f.t + radius,
                    camera_view=f.view_type,
                    possible_replay=False,
                )
            )

        # Temporal NMS keeps the earliest proposal in a burst.  Dense refinement
        # will snap it to the exact frame-level transition.
        proposals.extend(self._cue_address_proposals(features, times))
        proposals.sort(key=lambda c: c.timestamp)
        kept: list[StrikeCandidate] = []
        for candidate in proposals:
            if not kept or candidate.timestamp - kept[-1].timestamp >= self.sparse_gap_s:
                kept.append(candidate)
            elif candidate.confidence > kept[-1].confidence:
                kept[-1].confidence = candidate.confidence
                kept[-1].uncertainty_start = min(kept[-1].uncertainty_start, candidate.uncertainty_start)
                kept[-1].uncertainty_end = max(kept[-1].uncertainty_end, candidate.uncertainty_end)
        logger.info("Found %d sparse strike proposals", len(kept))
        for inferred in self._camera_contact_candidates(features, sparse=True):
            if not any(abs(c.timestamp-inferred.timestamp) < self.sparse_gap_s for c in kept):
                kept.append(inferred)
        kept.sort(key=lambda c: c.timestamp)
        return kept

    def _cue_address_proposals(self, features: list[FrameFeatures], times: list[float]) -> list[StrikeCandidate]:
        """Recover sparse close-up launches when table flow is unobservable.

        These are deliberately unconfirmed proposals: reliable localized white
        and cue-address observations can trigger native decoding even when a
        sparse optical-flow model failed. Foreign/handling evidence still vetoes.
        """
        def usable(f):
            return (f.match_context_valid and not f.table_handling and not f.broadcast_replay
                    and f.table_observable and f.table_confidence >= .5
                    and f.view_type not in {CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY,
                                           CameraViewType.ADVERTISEMENT, CameraViewType.SCOREBOARD})
        proposed = []
        for i, f in enumerate(features):
            if not usable(f):
                continue
            pre = [p for p in features[bisect_left(times, f.t-1.6):i]
                   if usable(p) and self._same_view(p, f) and p.cue_ball_detected
                   and p.cue_ball_x is not None and p.cue_ball_y is not None and self._track_conf(p) >= .65]
            if not pre or not any(p.cue_tip_visible and p.cue_tip_distance_to_ball <= 3*max(p.ball_diameter_px,1)
                                  for p in pre):
                continue
            diameter = max(1, float(np.median([p.ball_diameter_px for p in pre])))
            xy = np.array([(p.cue_ball_x,p.cue_ball_y) for p in pre])
            quiet = (len(pre) >= 2 and np.max(np.linalg.norm(xy-xy[-1],axis=1)) <= .35*diameter)
            quiet = quiet or (len(pre) == 1 and self._cue_speed(pre[0]) <= self.stationary_speed)
            # Sparse flow can fail during a slow zoom while a clearly addressed
            # white remains in view. This is still only permission to decode
            # the native window, never evidence that an impact happened.
            quiet = quiet or (np.median([self._cue_speed(p) for p in pre]) <= 1.25
                              and any(p.cue_tip_visible and p.cue_tip_distance_to_ball <= 1.25*max(p.ball_diameter_px,1)
                                      for p in pre))
            if not quiet:
                continue
            departed = (f.cue_ball_x is not None and f.cue_ball_y is not None
                        and np.hypot(f.cue_ball_x-pre[-1].cue_ball_x,f.cue_ball_y-pre[-1].cue_ball_y) >= .5*diameter)
            if f.cue_ball_detected and not departed and self._cue_speed(f) < 1.5:
                continue
            proposed.append(StrikeCandidate(timestamp=f.t, confidence=.6,
                evidence={"sparse_proposal":1.,"cue_address_proposal":1.},
                uncertainty_start=max(0,pre[-1].t-.5), uncertainty_end=f.t+1.5,
                camera_view=f.view_type))
        # A director often cuts from quiet overhead aiming to a bridge-level
        # view where the white is temporarily hidden. A coordinate-based onset
        # cannot propose that shot. Decode the short addressing interval, while
        # leaving actual strike confirmation entirely to native observations.
        for i, cut in enumerate(features):
            if i == 0 or cut.scene_cut_score < .5 or not usable(cut):
                continue
            old_scene = features[i-1].camera_scene_id
            pre = [p for p in features[bisect_left(times, cut.t-1.6):i]
                   if usable(p) and p.camera_scene_id == old_scene
                   and p.cue_ball_detected and self._track_conf(p) >= .65]
            if len(pre) < 2 or self._stationary_ratio(pre) < .8:
                continue
            if sum(p.max_ball_normalized_speed <= self.pre_quiet_max_ball_speed for p in pre)/len(pre) < .8:
                continue
            if not any(p.cue_tip_visible and p.cue_tip_distance_to_ball <= 3*max(p.ball_diameter_px, 1)
                       for p in pre):
                continue
            proposed.append(StrikeCandidate(
                timestamp=cut.t+1, confidence=.55,
                evidence={"sparse_proposal": 1., "camera_address_proposal": 1.},
                uncertainty_start=max(0, pre[-1].t-.5), uncertainty_end=cut.t+3,
                camera_view=cut.view_type,
            ))
        return proposed

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
            and self._same_view(f, features[idx])
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
        dense_features = self._stabilized_features(dense_features)
        if any(f.view_classified for f in dense_features):
            for candidate in candidates:
                for key in ("dense_transition_confirmed", "sparse_dense_transition",
                            "occlusion_inferred", "camera_contact_inferred", "camera_address_memory",
                            "ball_onset_run", "cue_ball_motion_confirmed"):
                    candidate.evidence[key] = 0.0
        cue_available = self._has_cue_kinematics(dense_features)
        if not cue_available:
            return candidates
        times = [f.t for f in dense_features]
        camera_contacts = (self._camera_contact_candidates(dense_features)
                           + self._address_memory_candidates(dense_features)
                           + self._cut_launch_candidates(dense_features))
        native_occlusions: list[StrikeCandidate] | None = None

        refined: list[StrikeCandidate] = []
        for cand in candidates:
            # This pass must stand on its own evidence. Otherwise a rejected
            # sparse match can survive through the downstream acceptance OR.
            cand.evidence["sparse_dense_transition"] = 0.0
            for key in ("dense_transition_confirmed", "occlusion_inferred",
                        "camera_contact_inferred", "camera_address_memory", "ball_onset_run",
                        "cue_ball_motion_confirmed"):
                cand.evidence[key] = 0.0
            lo = bisect_left(times, cand.uncertainty_start)
            hi = bisect_right(times, cand.uncertainty_end)
            indices = range(lo, hi)
            match: tuple[int, dict[str, float]] | None = None
            for i in indices:
                if i <= 0:
                    continue
                if not self._valid(dense_features[i]):
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
                inferred = next((c for c in camera_contacts
                                 if cand.uncertainty_start <= c.timestamp <= cand.uncertainty_end), None)
                if inferred is None:
                    if native_occlusions is None:
                        native_occlusions = [c for c in self.detect_candidates(dense_features)
                                             if c.evidence.get("occlusion_inferred", 0) >= .5
                                             and c.evidence.get("ball_onset_run", 0) >= 2]
                    inferred = next((c for c in native_occlusions
                                     if cand.uncertainty_start <= c.timestamp <= cand.uncertainty_end), None)
                if inferred is not None:
                    cand.timestamp = inferred.timestamp
                    cand.confidence = inferred.confidence
                    cand.camera_view = inferred.camera_view
                    cand.evidence.update(inferred.evidence)
                    cand.evidence["native_occlusion_confirmed"] = 1.0
                    cand.uncertainty_start = inferred.uncertainty_start
                    cand.uncertainty_end = inferred.uncertainty_end
                    refined.append(cand)
                    continue
                cand.confidence *= 0.75
                cand.evidence["dense_transition_confirmed"] = 0.0
                refined.append(cand)
                continue

            i, metrics = match
            f = dense_features[i]
            contact_t, contact_start = self._occluded_contact_time(
                dense_features, i, times, max(metrics.get("cue_contact_score", 0.0),
                                             metrics.get("pre_cue_address_score", 0.0)),
                stabilized_launch=(self._stabilized_launch_confirmed(metrics)
                                   or self._anchored_launch_confirmed(metrics)))
            cand.timestamp = contact_t
            cand.confidence = max(cand.confidence, float(f.strike_score), 0.75)
            cand.uncertainty_start = contact_start if contact_t < f.t else dense_features[max(0, i - 1)].t
            cand.uncertainty_end = f.t
            cand.camera_view = f.view_type
            cand.evidence = {
                **cand.evidence,
                **metrics,
                "dense_transition_confirmed": 1.0,
                "refined_strike": f.strike_score,
                "impact_occlusion_contact": float(contact_t < f.t),
            }
            refined.append(cand)
        return refined
