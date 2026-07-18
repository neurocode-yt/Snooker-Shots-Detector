"""Prediction-based, label-aware tracking for snooker-ball observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from snooker_ai.object_detection.detector import Detection


@dataclass
class Track:
    track_id: int
    label: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y
    last_t: float = 0.0  # last observed timestamp (backward-compatible meaning)
    hits: int = 0
    active: bool = True
    velocity: tuple[float, float] = (0.0, 0.0)  # smoothed px/s
    acceleration: tuple[float, float] = (0.0, 0.0)  # smoothed px/s^2
    confidence: float = 0.0
    visible: bool = True
    occluded: bool = False
    missed_frames: int = 0
    diameter: float = 0.0
    cue_color_confidence: float = 0.0
    shape_confidence: float = 0.0
    cloth_surround_confidence: float = 0.0
    predicted_position: Optional[tuple[float, float]] = None
    last_update_t: float = 0.0

    @property
    def radius(self) -> float:
        return self.diameter * 0.5

    @property
    def vx(self) -> float:
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        return float(self.velocity[1])

    @property
    def ax(self) -> float:
        return float(self.acceleration[0])

    @property
    def ay(self) -> float:
        return float(self.acceleration[1])

    def speed(self) -> float:
        """Observed speed for the current frame.

        A missed observation returns zero rather than replaying the speed between
        two stale positions.  The retained prediction is available internally for
        occlusion reasoning through :meth:`BallTracker.occluded_moving_count`.
        """

        if not self.active or not self.visible:
            return 0.0
        return float(np.hypot(*self.velocity))

    def predicted_speed(self) -> float:
        if not self.active:
            return 0.0
        return float(np.hypot(*self.velocity))

    def stable_speed(
        self,
        window_seconds: float = 0.24,
        diameter_px: float | None = None,
    ) -> float:
        """Estimate motion over a short history, suppressing detector jitter.

        Hough/component centres commonly wander by a pixel or two while a ball is
        resting.  The instantaneous derivative therefore reports implausibly high
        speeds at video cadence.  A least-squares line over the recent positions,
        together with a small displacement/coherence deadband, keeps that jitter
        from proving either a strike or a continued roll while preserving a real
        directed launch.
        """

        if not self.active or not self.visible or len(self.positions) < 2:
            return 0.0
        end_t = float(self.positions[-1][0])
        start_t = end_t - max(float(window_seconds), 0.05)
        recent = [p for p in self.positions if p[0] >= start_t - 1e-9]
        if len(recent) < 2:
            return float(np.hypot(*self.velocity))
        times = np.asarray([p[0] for p in recent], dtype=np.float64)
        span = float(times[-1] - times[0])
        if span < 1e-6:
            return 0.0
        xy = np.asarray([[p[1], p[2]] for p in recent], dtype=np.float64)
        centered = times - float(np.mean(times))
        denom = float(np.dot(centered, centered))
        if denom <= 1e-9:
            return 0.0
        slope = np.sum(centered[:, None] * (xy - np.mean(xy, axis=0)), axis=0) / denom
        net = float(np.linalg.norm(xy[-1] - xy[0]))
        path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        diameter = float(diameter_px or self.diameter or 0.0)
        # The deadband is deliberately small in image space; it only removes
        # sub-ball-scale wandering and does not erase a genuine slow roll.
        if diameter > 0.0 and net < 0.08 * diameter:
            return 0.0
        coherence = net / max(path, 1e-6)
        if path > 0.0 and coherence < 0.35:
            return 0.0
        return float(np.hypot(*slope))

    def predict(self, t: float) -> tuple[float, float]:
        if not self.positions:
            return 0.0, 0.0
        _, x, y = self.positions[-1]
        dt = max(0.0, float(t) - float(self.last_t))
        # Constant-velocity prediction; acceleration is deliberately not projected
        # because collision estimates can be noisy and cause identity swaps.
        return x + self.velocity[0] * dt, y + self.velocity[1] * dt


class BallTracker:
    def __init__(
        self,
        max_distance: float = 40.0,
        max_missed: float = 1.0,
        velocity_alpha: float = 0.55,
        acceleration_alpha: float = 0.40,
    ):
        self.max_distance = float(max_distance)
        self.max_missed = float(max_missed)
        self.velocity_alpha = float(np.clip(velocity_alpha, 0.0, 1.0))
        self.acceleration_alpha = float(np.clip(acceleration_alpha, 0.0, 1.0))
        self._next_id = 1
        self.tracks: list[Track] = []
        self._diameter_history: list[float] = []

    def update(self, t: float, detections: list[Detection]) -> list[Track]:
        t = float(t)
        detections = list(detections)
        active_indices = [i for i, tr in enumerate(self.tracks) if tr.active]

        if not active_indices:
            for detection in detections:
                self._spawn(t, detection)
            return [tr for tr in self.tracks if tr.active]

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if detections:
            costs = np.full((len(active_indices), len(detections)), 1e6, dtype=np.float64)
            gates = np.zeros_like(costs)
            diameter_estimate = self.estimated_ball_diameter()
            for row, track_index in enumerate(active_indices):
                track = self.tracks[track_index]
                px, py = track.predict(t)
                dt = max(0.0, t - track.last_t)
                track_scale = track.diameter or diameter_estimate
                for column, detection in enumerate(detections):
                    detection_scale = detection.diameter_px
                    scale = max(track_scale, detection_scale, diameter_estimate, 1.0)
                    # Permit rapid balls while keeping the gate scale-aware.  The
                    # predicted point carries most of the displacement already.
                    gate = max(self.max_distance, 3.5 * scale) + min(
                        self.max_distance * 1.5, track.predicted_speed() * dt * 0.35
                    )
                    distance = float(np.hypot(detection.cx - px, detection.cy - py))
                    label_penalty = self._label_penalty(track, detection, gate)
                    size_penalty = 0.0
                    if track_scale > 0.0 and detection_scale > 0.0:
                        size_penalty = min(
                            gate * 0.45,
                            abs(np.log((detection_scale + 1e-6) / (track_scale + 1e-6)))
                            * gate
                            * 0.28,
                        )
                    costs[row, column] = distance + label_penalty + size_penalty
                    gates[row, column] = gate

            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if costs[row, column] > gates[row, column]:
                    continue
                track_index = active_indices[row]
                self._apply_detection(self.tracks[track_index], t, detections[column])
                matched_tracks.add(track_index)
                matched_detections.add(column)

        for column, detection in enumerate(detections):
            if column not in matched_detections:
                self._spawn(t, detection)

        for track_index in active_indices:
            if track_index in matched_tracks:
                continue
            self._mark_missed(self.tracks[track_index], t)

        return [tr for tr in self.tracks if tr.active]

    @staticmethod
    def _label_penalty(track: Track, detection: Detection, gate: float) -> float:
        if track.label == detection.label:
            return 0.0
        # Cue-ball identity is valuable, but a single low-colour frame must not
        # terminate its track.  Strong cue observations are expensive to assign to
        # an object-ball track; weak observations may bridge lighting changes.
        if track.label == "cue_ball" and detection.label != "cue_ball":
            return gate * (0.30 if detection.color_confidence < 0.55 else 0.60)
        if track.label != "cue_ball" and detection.label == "cue_ball":
            return gate * (0.62 if detection.color_confidence >= 0.65 else 0.38)
        return gate * 0.20

    def _apply_detection(self, track: Track, t: float, detection: Detection) -> None:
        previous_velocity = track.velocity
        if track.positions:
            previous_t, previous_x, previous_y = track.positions[-1]
            dt = t - previous_t
        else:
            previous_x = detection.cx
            previous_y = detection.cy
            dt = 0.0

        if dt > 1e-6:
            observed_velocity = (
                (float(detection.cx) - previous_x) / dt,
                (float(detection.cy) - previous_y) / dt,
            )
            # Do not over-smooth the first valid motion observation.
            alpha = self.velocity_alpha if track.hits >= 2 else 1.0
            new_velocity = (
                alpha * observed_velocity[0] + (1.0 - alpha) * previous_velocity[0],
                alpha * observed_velocity[1] + (1.0 - alpha) * previous_velocity[1],
            )
            observed_acceleration = (
                (new_velocity[0] - previous_velocity[0]) / dt,
                (new_velocity[1] - previous_velocity[1]) / dt,
            )
            accel_alpha = self.acceleration_alpha if track.hits >= 2 else 1.0
            track.acceleration = (
                accel_alpha * observed_acceleration[0]
                + (1.0 - accel_alpha) * track.acceleration[0],
                accel_alpha * observed_acceleration[1]
                + (1.0 - accel_alpha) * track.acceleration[1],
            )
            track.velocity = new_velocity
        else:
            track.velocity = (0.0, 0.0)
            track.acceleration = (0.0, 0.0)

        track.positions.append((t, float(detection.cx), float(detection.cy)))
        if len(track.positions) > 64:
            del track.positions[:-64]
        track.last_t = t
        track.last_update_t = t
        track.hits += 1
        track.visible = True
        track.occluded = False
        track.missed_frames = 0
        track.active = True
        track.predicted_position = (float(detection.cx), float(detection.cy))
        track.confidence = float(
            np.clip(0.65 * track.confidence + 0.35 * detection.confidence, 0.0, 1.0)
        )
        diameter = detection.diameter_px
        if diameter > 0.0:
            track.diameter = (
                diameter if track.diameter <= 0.0 else 0.80 * track.diameter + 0.20 * diameter
            )
            self._remember_diameter(diameter)
        if detection.label == "cue_ball":
            track.cue_color_confidence = max(
                detection.color_confidence,
                0.82 * track.cue_color_confidence + 0.18 * detection.color_confidence,
            )
            # Promote only with meaningful white-colour support; never demote on one
            # missed/poorly lit observation.
            if detection.color_confidence >= 0.45 or track.label == "cue_ball":
                track.label = "cue_ball"
        track.shape_confidence = (
            float(detection.shape_confidence)
            if track.shape_confidence <= 0.0
            else 0.75 * track.shape_confidence + 0.25 * float(detection.shape_confidence)
        )
        track.cloth_surround_confidence = (
            float(detection.cloth_surround_confidence)
            if track.cloth_surround_confidence <= 0.0
            else 0.70 * track.cloth_surround_confidence
            + 0.30 * float(detection.cloth_surround_confidence)
        )

    def _mark_missed(self, track: Track, t: float) -> None:
        track.visible = False
        track.occluded = True
        track.missed_frames += 1
        track.last_update_t = t
        track.predicted_position = track.predict(t)
        track.confidence = float(np.clip(track.confidence * 0.92, 0.0, 1.0))
        if t - track.last_t > self.max_missed:
            track.active = False
            track.occluded = False
            track.velocity = (0.0, 0.0)
            track.acceleration = (0.0, 0.0)

    def _spawn(self, t: float, detection: Detection) -> None:
        diameter = detection.diameter_px
        track = Track(
            track_id=self._next_id,
            label=detection.label,
            positions=[(t, float(detection.cx), float(detection.cy))],
            last_t=t,
            hits=1,
            active=True,
            velocity=(0.0, 0.0),
            acceleration=(0.0, 0.0),
            confidence=float(detection.confidence),
            visible=True,
            occluded=False,
            missed_frames=0,
            diameter=diameter,
            cue_color_confidence=(
                float(detection.color_confidence) if detection.label == "cue_ball" else 0.0
            ),
            shape_confidence=float(detection.shape_confidence),
            cloth_surround_confidence=float(detection.cloth_surround_confidence),
            predicted_position=(float(detection.cx), float(detection.cy)),
            last_update_t=t,
        )
        self._next_id += 1
        self.tracks.append(track)
        self._remember_diameter(diameter)

    def _remember_diameter(self, diameter: float) -> None:
        if diameter <= 0.0 or not np.isfinite(diameter):
            return
        self._diameter_history.append(float(diameter))
        if len(self._diameter_history) > 256:
            del self._diameter_history[:-256]

    def estimated_ball_diameter(self) -> float:
        visible = [
            tr.diameter
            for tr in self.tracks
            if tr.active and tr.visible and tr.diameter > 0.0
        ]
        values = visible if visible else self._diameter_history
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        # Suppress gross false-circle scales before taking the final median.
        kept = arr[(arr >= 0.55 * median) & (arr <= 1.8 * median)]
        return float(np.median(kept)) if kept.size else median

    def max_speed(self) -> float:
        speeds = [tr.speed() for tr in self.tracks if tr.active and tr.visible]
        return max(speeds) if speeds else 0.0

    def mean_speed(self) -> float:
        speeds = [tr.speed() for tr in self.tracks if tr.active and tr.visible]
        return float(np.mean(speeds)) if speeds else 0.0

    def max_normalized_speed(self, ball_diameter_px: Optional[float] = None) -> float:
        diameter = float(ball_diameter_px or self.estimated_ball_diameter())
        if diameter <= 1e-6:
            return 0.0
        speeds = [
            track.stable_speed(diameter_px=diameter)
            for track in self.tracks
            if track.active
            and track.visible
            and track.hits >= 2
            and self._is_ball_quality_track(track)
        ]
        return float(max(speeds) / diameter) if speeds else 0.0

    def stable_track_speed(self, track: Track, ball_diameter_px: float = 0.0) -> float:
        """Return one track's jitter-resistant speed in ball diameters/second."""

        if not self._is_ball_quality_track(track):
            return 0.0
        diameter = float(ball_diameter_px or track.diameter or self.estimated_ball_diameter())
        if diameter <= 1e-6:
            return 0.0
        return float(track.stable_speed(diameter_px=diameter) / diameter)

    @staticmethod
    def _is_ball_quality_track(track: Track) -> bool:
        """Reject ball-sized highlights embedded in players, rails or graphics."""

        shape_ok = track.shape_confidence <= 0.0 or track.shape_confidence >= 0.48
        surround_ok = (
            track.cloth_surround_confidence <= 0.0
            or track.cloth_surround_confidence >= 0.45
        )
        return bool(shape_ok and surround_ok)

    def occluded_moving_count(
        self,
        min_normalized_speed: float = 0.10,
        ball_diameter_px: Optional[float] = None,
    ) -> int:
        fallback_diameter = float(ball_diameter_px or self.estimated_ball_diameter())
        count = 0
        for track in self.tracks:
            if not track.active or not track.occluded:
                continue
            if not self._is_ball_quality_track(track):
                continue
            diameter = track.diameter or fallback_diameter
            if diameter <= 1e-6:
                continue
            # Once a track is hidden, its last measured velocity is the only
            # conservative evidence available; do not pretend the ball stopped
            # merely because no new centre was observed.
            if track.predicted_speed() / diameter >= min_normalized_speed:
                count += 1
        return count

    def cue_ball_track(self) -> Optional[Track]:
        candidates = [tr for tr in self.tracks if tr.active and tr.label == "cue_ball"]
        if not candidates:
            return None
        # Prefer a visible, repeatedly observed, strongly white track.  This avoids
        # returning the first bright false circle forever.
        return max(
            candidates,
            key=lambda tr: (
                1 if tr.visible else 0,
                tr.cue_color_confidence,
                tr.confidence,
                tr.hits,
                tr.last_t,
            ),
        )
