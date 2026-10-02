"""Visual camera-view classification with explicit cloth visibility cues."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType


class ViewClassifier:
    """
    Classifies broadcast frames without a trained CNN (Phase 1).

    Cloth extent distinguishes a complete table from a zoomed or clipped table.
    Graphics scores are diagnostic only: a colourful banner cannot prove replay.
    """

    def __init__(self, config: Config):
        cfg = config.section("camera_view")
        table_cfg = config.section("table_detection")
        self.main_table_ratio = float(cfg.get("table_green_ratio_main", 0.18))
        self.partial_table_ratio = float(cfg.get("table_green_ratio_partial", 0.06))
        # Threshold only — must not shadow the skin_ratio() method below.
        self.closeup_skin_threshold = float(cfg.get("closeup_face_skin_ratio", 0.12))
        self.graphics_edge = float(cfg.get("graphics_edge_density", 0.15))
        self.replay_score_thr = float(cfg.get("replay_graphics_score", 0.55))
        hsv = table_cfg.get("hsv_lower", [35, 40, 40])
        hsv_u = table_cfg.get("hsv_upper", [95, 255, 255])
        self.hsv_lower = np.array(hsv, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_u, dtype=np.uint8)

    def green_ratio(self, frame_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def edge_density(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.count_nonzero(edges)) / float(edges.size)

    def skin_ratio(self, frame_bgr: np.ndarray) -> float:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def replay_graphic_score(self, frame_bgr: np.ndarray) -> float:
        """
        Heuristic for 'REPLAY' / bug overlays: high edge density in corners
        + lower centre green than main table, or strong chroma in top band.
        """
        h, w = frame_bgr.shape[:2]
        top = frame_bgr[: max(1, h // 8), :]
        top_edges = self.edge_density(top)
        # Saturated non-green colours in banner region
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        high_sat = float(np.mean(sat > 120))
        return float(np.clip(0.6 * top_edges * 5 + 0.4 * high_sat, 0.0, 1.0))

    def classify(self, frame_bgr: np.ndarray) -> tuple[CameraViewType, float, dict[str, Any]]:
        # Classification should not run several full-resolution conversions for
        # every native frame; all these cues are spatially coarse.
        h, w = frame_bgr.shape[:2]
        frame = frame_bgr
        if w > 640:
            frame = cv2.resize(frame_bgr, (640, max(1, round(h * 640 / w))))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        g = float(np.count_nonzero(green)) / float(green.size)
        e = self.edge_density(frame)
        s = self.skin_ratio(frame)
        r = self.replay_graphic_score(frame)
        geometry = self._cloth_geometry(green)
        extra: dict[str, Any] = {
            "green_ratio": g,
            "edge_density": e,
            "skin_ratio": s,
            "replay_graphic_score": r,
            "is_replay_candidate": False,
            "broadcast_graphics_candidate": bool(r >= self.replay_score_thr),
            **geometry,
        }

        if g >= self.main_table_ratio:
            if geometry["full_table_candidate"]:
                return CameraViewType.MAIN_TABLE, g, extra
            return CameraViewType.BALL_CLOSEUP, g, extra

        if g >= self.partial_table_ratio:
            # A wide strip of cloth in the foreground is a useful local ball
            # view, even when most of the frame contains a player or crowd.
            if geometry["cloth_width_ratio"] >= 0.55:
                return CameraViewType.BALL_CLOSEUP, g, extra
            if s > self.closeup_skin_threshold:
                return CameraViewType.PLAYER_CLOSEUP, g, extra
            return CameraViewType.WIDE_ARENA, g, extra

        if e >= self.graphics_edge and g < self.partial_table_ratio:
            return CameraViewType.SCOREBOARD, g, extra

        if s >= self.closeup_skin_threshold * 1.5:
            return CameraViewType.PLAYER_CLOSEUP, g, extra

        if g < 0.02 and e < 0.06:
            return CameraViewType.AUDIENCE, g, extra

        return CameraViewType.OTHER, g, extra

    @staticmethod
    def _cloth_geometry(mask: np.ndarray) -> dict[str, Any]:
        """Report extent only; perspective calibration belongs to the tracker."""
        h, w = mask.shape
        work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "cloth_width_ratio": 0.0, "cloth_height_ratio": 0.0,
                "cloth_clipped": False, "full_table_candidate": False,
            }
        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)
        clipped = bool(x <= w * 0.01 or x + bw >= w * 0.99 or y <= h * 0.01 or y + bh >= h * 0.99)
        full = bool(
            not clipped and bw / w >= 0.30 and bh / h >= 0.22
            and 0.65 <= bw / max(1, bh) <= 3.2
        )
        return {
            "cloth_width_ratio": float(bw / w),
            "cloth_height_ratio": float(bh / h),
            "cloth_clipped": clipped,
            "full_table_candidate": full,
        }
