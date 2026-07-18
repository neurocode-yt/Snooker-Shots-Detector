"""Heuristic camera-view classification for Phase 1."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType


class ViewClassifier:
    """
    Classifies broadcast frames without a trained CNN (Phase 1).

    Uses green cloth ratio, edge density (graphics), skin-tone ratio (close-ups),
    and simple spatial layout cues. Phase 2+ should replace with a learned classifier.
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
        g = self.green_ratio(frame_bgr)
        e = self.edge_density(frame_bgr)
        s = self.skin_ratio(frame_bgr)
        r = self.replay_graphic_score(frame_bgr)
        extra: dict[str, Any] = {
            "green_ratio": g,
            "edge_density": e,
            "skin_ratio": s,
            "replay_graphic_score": r,
            "is_replay_candidate": r >= self.replay_score_thr,
        }

        if r >= self.replay_score_thr and g >= self.partial_table_ratio:
            return CameraViewType.REPLAY, g, extra

        if g >= self.main_table_ratio:
            # Distinguish full table vs zoomed ball close-up: high green + very zoomed feel
            if g > 0.55 and e < 0.08:
                return CameraViewType.BALL_CLOSEUP, g, extra
            return CameraViewType.MAIN_TABLE, g, extra

        if g >= self.partial_table_ratio:
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
