"""Measured red-edge flow when touching balls defeat colour components."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.types import CameraViewType, FrameFeatures


@dataclass
class _FlowTrack:
    center: np.ndarray
    diameter: float
    points: np.ndarray
    origin: np.ndarray
    references: np.ndarray
    reference_origin: np.ndarray
    groups: np.ndarray
    rest: list = field(default_factory=list)
    moving: list = field(default_factory=list)


class ClusterRedFlow:
    """Require a measured red rim, stable witnesses and coherent native flow.

    Forward/backward optical flow follows the original red pixels without
    assigning a changing connected colour component to a new ball. Native
    camera transforms are checked against points on two separate stationary
    spheres before any apparent red displacement can confirm a contact.
    """

    def __init__(self, config: Config):
        self.detector = ObjectDetector(config)
        self.previous: FrameFeatures | None = None
        self.gray: np.ndarray | None = None
        self.matrix = np.eye(3)
        self.tracks: list[_FlowTrack] = []
        self.contacts: list[tuple[float, float]] = []

    @staticmethod
    def _world(points, matrix):
        return np.linalg.solve(matrix, np.c_[points.reshape(-1, 2), np.ones(len(points))].T).T[
            :, :2
        ]

    @staticmethod
    def _follow(previous, gray, points):
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(15, 15), maxLevel=2
        )
        if moved is None:
            return points[:0], np.zeros(len(points), dtype=bool)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, moved, None, winSize=(15, 15), maxLevel=2
        )
        if back is None:
            return points[:0], np.zeros(len(points), dtype=bool)
        valid = (
            (status.ravel() > 0)
            & (back_status.ravel() > 0)
            & np.isfinite(moved.reshape(-1, 2)).all(axis=1)
            & (np.linalg.norm(back - points, axis=2).ravel() <= 0.75)
        )
        return moved[valid].reshape(-1, 1, 2), valid

    @staticmethod
    def _corners(gray, red, circle):
        x, y, radius = circle
        yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
        mask = (red & ((xx - x) ** 2 + (yy - y) ** 2 <= (0.95 * radius) ** 2)).astype(
            np.uint8
        ) * 255
        return cv2.goodFeaturesToTrack(gray, 20, 0.01, 2, mask=mask, blockSize=3)

    def _circles(self, gray, hsv, cloth, red, diameter, bounds=None):
        x0, y0, x1, y1 = bounds or (0, 0, gray.shape[1], gray.shape[0])
        circles = cv2.HoughCircles(
            gray[y0:y1, x0:x1],
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=0.5 * diameter,
            param1=80,
            param2=18,
            minRadius=max(6, round(0.30 * diameter)),
            maxRadius=round(0.8 * diameter),
        )
        result = []
        for x, y, radius in circles[0] if circles is not None else []:
            x, y, radius = float(x + x0), float(y + y0), float(radius)
            yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
            inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.6 * radius) ** 2
            if not inside.any() or np.mean(red[inside]) < 0.45:
                continue
            white, deviation, surround = self.detector._colour_scores(hsv, cloth, x, y, radius)
            if white <= 0.2 and deviation >= 0.8 and surround >= 0.25:
                result.append((x, y, radius, surround))
        return result

    def _seed(self, gray, hsv, cloth, red, context):
        if (
            not context.cue_ball_detected
            or not context.cue_ball_quality
            or context.cue_ball_x is None
            or context.cue_ball_y is None
            or context.cue_ball_observations < 3
            or context.cue_ball_track_confidence < 0.75
            or context.cue_ball_stable_normalized_speed is None
            or context.cue_ball_stable_normalized_speed > 0.3
            or len(self.tracks) >= 6
        ):
            return
        diameter = context.cue_ball_image_diameter_px or context.ball_diameter_px
        if not 20 <= diameter <= 80:
            return
        x, y = context.cue_ball_x, context.cue_ball_y
        bounds = (
            max(0, round(x - 2.5 * diameter)),
            max(0, round(y - 2.5 * diameter)),
            min(gray.shape[1], round(x + 2.5 * diameter)),
            min(gray.shape[0], round(y + 2.5 * diameter)),
        )
        nearby = self._circles(gray, hsv, cloth, red, diameter, bounds)
        nearby = [
            c for c in nearby if 0.4 * diameter <= np.hypot(c[0] - x, c[1] - y) <= 2 * diameter
        ]
        if not nearby:
            return
        references = []
        groups = []
        for circle in self._circles(gray, hsv, cloth, red, diameter):
            if circle[3] < 0.6 or np.hypot(circle[0] - x, circle[1] - y) < 3 * diameter:
                continue
            corners = self._corners(gray, red, circle[:3])
            if corners is not None and len(corners) >= 3:
                groups.extend([len(references)] * len(corners))
                references.append(corners)
        if len(references) < 2:
            return
        references = np.concatenate(references)
        if len(references) < 6:
            return
        world_references = self._world(references, self.matrix)
        scale = np.sqrt(abs(np.linalg.det(self.matrix[:2, :2])))
        for circle in nearby:
            center = self._world(np.array(circle[:2]).reshape(1, 1, 2), self.matrix)[0]
            if any(np.linalg.norm(center - t.center) < 0.65 * t.diameter for t in self.tracks):
                continue
            points = self._corners(gray, red, circle[:3])
            if points is None or len(points) < 5:
                continue
            origin = self._world(points, self.matrix)
            self.tracks.append(
                _FlowTrack(
                    center,
                    2 * circle[2] / scale,
                    points,
                    origin,
                    references.copy(),
                    world_references.copy(),
                    np.asarray(groups),
                )
            )

    def observe(self, frame, context: FrameFeatures):
        previous = self.previous
        self.previous = context
        step = np.asarray(context.camera_frame_transform, dtype=float)
        valid = (
            context.observation_fps >= 20
            and context.observation_valid
            and context.match_context_valid
            and not context.broadcast_replay
            and not context.table_handling
            and not context.rack_idle
            and context.view_type == CameraViewType.BALL_CLOSEUP
            and context.scene_cut_score < 0.42
            and context.camera_motion_magnitude <= 1.5
            and step.size == 6
            and np.isfinite(step).all()
        )
        contiguous = (
            valid
            and previous is not None
            and previous.camera_scene_id == context.camera_scene_id
            and context.camera_frame_dt > 0
            and abs(context.t - previous.t - context.camera_frame_dt) <= 0.005
            and context.t - previous.t <= 1.5 / context.observation_fps
        )
        if not contiguous:
            self.tracks = []
            self.matrix = np.eye(3)
            self.gray = None
            self.contacts = []
        if not valid:
            return None
        transform = np.eye(3)
        transform[:2] = step.reshape(2, 3)
        if (
            not 0.98 <= np.linalg.det(transform[:2, :2]) <= 1.02
            or np.linalg.cond(transform[:2, :2]) > 1.05
        ):
            self.tracks = []
            self.gray = None
            self.matrix = np.eye(3)
            self.contacts = []
            return None
        if contiguous:
            self.matrix = transform @ self.matrix
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.contacts.append((context.t, context.cue_contact_score))
        self.contacts = [(t, score) for t, score in self.contacts if context.t - t <= 1.0]
        kept = []
        for track in self.tracks:
            points, good = self._follow(self.gray, gray, track.points)
            refs, ref_good = self._follow(self.gray, gray, track.references)
            if len(points) < 4 or len(refs) < 6:
                continue
            origin = track.origin[good]
            ref_origin = track.reference_origin[ref_good]
            groups = track.groups[ref_good]
            witnesses = sum(np.count_nonzero(groups == g) >= 3 for g in np.unique(groups))
            if witnesses < 2:
                continue
            ref_motion = self._world(refs, self.matrix) - ref_origin
            residual = float(np.percentile(np.linalg.norm(ref_motion, axis=1), 90) / track.diameter)
            if residual > 0.1:
                continue
            motion = self._world(points, self.matrix) - origin
            displacement = np.median(motion, axis=0)
            spread = float(
                np.percentile(np.linalg.norm(motion - displacement, axis=1), 75) / track.diameter
            )
            if spread > 0.1:
                continue
            sample = (context.t, *displacement)
            rest = np.asarray(track.rest)
            armed = (
                len(rest) >= 7
                and rest[-1, 0] - rest[0, 0] >= 0.19
                and np.max(np.linalg.norm(rest[:, 1:3] - np.median(rest[:, 1:3], axis=0), axis=1))
                <= 0.1 * track.diameter
            )
            anchor = np.median(rest[:, 1:3], axis=0) if len(rest) else displacement
            away = np.linalg.norm(displacement - anchor) / track.diameter
            if armed and (track.moving or away >= 0.10):
                track.moving.append(sample)
                moving = np.asarray(track.moving)
                xy = np.vstack((anchor, moving[:, 1:3]))
                steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                net = np.linalg.norm(xy[-1] - xy[0])
                elapsed = context.t - moving[0, 0]
                if elapsed > 0.6:
                    continue
                contact = max(
                    (
                        score
                        for t, score in self.contacts
                        if moving[0, 0] - 0.45 <= t <= moving[0, 0] + 0.1
                    ),
                    default=0.0,
                )
                if (
                    len(moving) >= 4
                    and elapsed >= 0.095
                    and net / track.diameter >= 0.65
                    and net / max(np.sum(steps), 1e-6) >= 0.90
                    and np.max(steps) / track.diameter <= 0.5
                    and contact >= 0.6
                ):
                    return {
                        "upper": float(moving[0, 0]),
                        "confirmed": context.t,
                        "rest_frames": len(rest),
                        "moving_frames": len(moving),
                        "feature_count": len(points),
                        "reference_points": len(refs),
                        "reference_spheres": witnesses,
                        "travel_diameters": float(net / track.diameter),
                        "direction_consistency": float(net / max(np.sum(steps), 1e-6)),
                        "registration_residual_diameters": residual,
                        "feature_spread_diameters": spread,
                        "contact_score": contact,
                    }
            else:
                track.rest.append(sample)
                track.rest = track.rest[-16:]
            track.points, track.origin = points, origin
            track.references, track.reference_origin, track.groups = refs, ref_origin, groups
            kept.append(track)
        self.tracks = kept
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red = ((hue < 15) | (hue >= 140)) & (saturation >= 110) & (value >= 55)
        cloth = cv2.inRange(hsv, self.detector.cloth_lower, self.detector.cloth_upper)
        self._seed(gray, hsv, cloth, red, context)
        self.gray = gray
        return None
