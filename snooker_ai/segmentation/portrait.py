"""End-only evidence that a camera is showing an upright player portrait."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import FrameFeatures, ShotRecord
from snooker_ai.utils.video import open_capture


@dataclass(frozen=True)
class PortraitCutaway:
    strike_timestamp: float
    start_timestamp: float
    confirmation_timestamp: float


@lru_cache(maxsize=1)
def _face_cascade():
    # OpenCV ships a classical boosted Haar cascade; this is not a neural model.
    data = getattr(cv2, "data", None)
    if data is None:
        return None
    path = Path(data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        return None
    model = cv2.CascadeClassifier(str(path))
    return None if model.empty() else model


def portrait_frame(frame: np.ndarray, config: Config) -> bool:
    image = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(config.get("table_detection.hsv_lower", [35, 40, 40]), np.uint8),
        np.array(config.get("table_detection.hsv_upper", [95, 255, 255]), np.uint8),
    )
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or not 0.04 <= np.mean(green > 0) <= 0.25:
        return False
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # A ball-focused low camera has cloth much higher in the image. Require a
    # shallow foreground strip, then a face well above it in an upright pose.
    if y < 0.70 * 360 or height > 0.30 * 360 or width < 0.65 * 640:
        return False
    cascade = _face_cascade()
    if cascade is None:
        return False
    faces = cascade.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(35, 35)
    )
    return any(
        fw * fh >= 0.007 * 640 * 360 and fy + fh <= y - 0.30 * 360 for fx, fy, fw, fh in faces
    )


class PortraitCutawayDetector:
    def __init__(self, config: Config):
        self.config = config
        self.cut_threshold = float(config.get("scene_detection.hard_cut_threshold", 0.42))
        self.motion_threshold = float(config.get("ball_stop.motion_start_normalized_speed", 0.45))

    def detect(
        self, source: str | Path, shots: list[ShotRecord], features: list[FrameFeatures]
    ) -> list[PortraitCutaway]:
        eligible = [
            s
            for s in shots
            if s.included
            and not s.possible_replay
            and not s.user_modified
            and s.manual_review_required
            and s.clip_end - s.cue_strike > 3.0
        ]
        if not eligible or not features or _face_cascade() is None:
            return []
        ordered = sorted(features, key=lambda f: f.t)
        times = [f.t for f in ordered]
        capture = open_capture(source)
        results = []
        try:
            if not capture.isOpened():
                return []
            for shot in eligible:
                start = max(shot.cue_strike + 2.0, shot.clip_end - 12.0)
                capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                next_sample = start
                previous_pts = None
                run_start = None
                run_scene = None
                previous_sample = None
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    if not np.isfinite(t) or previous_pts is not None and t <= previous_pts:
                        break
                    previous_pts = t
                    if t > shot.clip_end + 0.4:
                        break
                    if t < next_sample - 1e-6:
                        continue
                    next_sample = t + 0.2
                    at = bisect_left(times, t)
                    context = min(
                        ordered[max(0, at - 1) : at + 1], key=lambda f: abs(f.t - t), default=None
                    )
                    context_gap = times[at] - times[at - 1] if 0 < at < len(times) else 0.0
                    valid = (
                        context is not None
                        and abs(context.t - t) <= 0.26
                        and context_gap <= 0.52
                        and context.match_context_valid
                        and not context.broadcast_replay
                        and not context.table_full_view
                        and context.scene_cut_score < self.cut_threshold
                        and context.camera_motion_magnitude <= 3
                        and context.moving_ball_count == 0
                        and context.max_ball_normalized_speed < self.motion_threshold
                    )
                    matched = bool(valid and portrait_frame(frame, self.config))
                    if (
                        not matched
                        or previous_sample is not None
                        and t - previous_sample > 0.26
                        or run_scene is not None
                        and context.camera_scene_id != run_scene
                    ):
                        run_start = None
                        run_scene = None
                    if matched:
                        if run_start is None:
                            run_start, run_scene = t, context.camera_scene_id
                        if t - run_start >= 0.60 - 1e-6:
                            native_start = self._native_start(
                                capture, run_start, shot.cue_strike + 2.0
                            )
                            results.append(PortraitCutaway(shot.cue_strike, native_start, t))
                            break
                    previous_sample = t
        finally:
            capture.release()
        return results

    def _native_start(self, capture, sampled_start: float, floor: float) -> float:
        """Backtrace the sampled portrait run to its first actual source frame."""
        capture.set(cv2.CAP_PROP_POS_MSEC, max(floor, sampled_start - 0.24) * 1000)
        earliest = None
        previous = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not np.isfinite(t) or (previous is not None and not 0 < t - previous <= 0.10):
                return sampled_start
            previous = t
            if t > sampled_start + 1e-6:
                break
            if portrait_frame(frame, self.config):
                if earliest is None:
                    earliest = t
            else:
                earliest = None
        return sampled_start if earliest is None else earliest
