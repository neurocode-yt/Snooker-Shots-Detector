"""HSV + contour based table playing-surface localisation (Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.utils.acceleration import acceleration_enabled
from snooker_ai.utils.logging import get_logger

logger = get_logger("table_detection")


@dataclass
class TableObservation:
    confidence: float
    mask: Optional[np.ndarray]  # uint8 0/255 same size as frame
    contour: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None  # (4,2) if quadrilateral found
    homography: Optional[np.ndarray] = None  # 3x3 to normalised table
    area_ratio: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None  # x,y,w,h


# Normalised top-down table rectangle (snooker approx aspect 2:1)
_CANONICAL_CORNERS = np.array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float32
)


class TableLocalizer:
    def __init__(self, config: Config):
        cfg = config.section("table_detection")
        self.hsv_lower = np.array(cfg.get("hsv_lower", [35, 40, 40]), dtype=np.uint8)
        self.hsv_upper = np.array(cfg.get("hsv_upper", [95, 255, 255]), dtype=np.uint8)
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.05))
        self.morph_k = int(cfg.get("morph_kernel", 7))
        self.eps = float(cfg.get("approx_epsilon", 0.02))
        self.min_conf = float(cfg.get("min_confidence", 0.25))
        self.smooth_n = int(cfg.get("temporal_smooth_frames", 5))
        fb = cfg.get("fallback_centre_crop", [0.1, 0.15, 0.9, 0.85])
        self.fallback = [float(x) for x in fb]
        self._mask_history: list[np.ndarray] = []
        self.use_opencl = acceleration_enabled(config)

    def reset(self) -> None:
        """Forget view-local mask history after a camera cut."""
        self._mask_history.clear()

    def _fallback_mask(self, h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = self.fallback
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 255
        return mask

    def detect(self, frame_bgr: np.ndarray) -> TableObservation:
        h, w = frame_bgr.shape[:2]
        frame_input = cv2.UMat(frame_bgr) if self.use_opencl else frame_bgr
        hsv = cv2.cvtColor(frame_input, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_k, self.morph_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        if isinstance(mask, cv2.UMat):
            mask = mask.get()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            fb = self._fallback_mask(h, w)
            return TableObservation(
                confidence=0.1,
                mask=fb,
                area_ratio=float(np.count_nonzero(fb)) / float(h * w),
            )

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        area_ratio = area / float(h * w)

        clean = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean, [contour], -1, 255, thickness=-1)

        # Temporal smoothing
        self._mask_history.append(clean)
        if len(self._mask_history) > self.smooth_n:
            self._mask_history.pop(0)
        if len(self._mask_history) > 1:
            stack = np.stack(self._mask_history, axis=0).astype(np.float32)
            clean = (np.mean(stack, axis=0) > 127).astype(np.uint8) * 255

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.eps * peri, True)
        corners = None
        homography = None
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            # Order corners consistently (tl, tr, br, bl)
            corners = _order_quad(corners)
            try:
                homography = cv2.getPerspectiveTransform(corners, _CANONICAL_CORNERS * 100)
            except cv2.error:
                homography = None

        x, y, bw, bh = cv2.boundingRect(contour)
        # Confidence: area + rectangularity
        rect_area = float(bw * bh) if bw * bh else 1.0
        fill = area / rect_area
        conf = float(np.clip(area_ratio / 0.35, 0.0, 1.0) * 0.7 + fill * 0.3)
        if area_ratio < self.min_area_ratio:
            conf *= 0.5
            if conf < self.min_conf:
                fb = self._fallback_mask(h, w)
                return TableObservation(
                    confidence=max(conf, 0.15),
                    mask=fb,
                    area_ratio=float(np.count_nonzero(fb)) / float(h * w),
                    bbox=(x, y, bw, bh),
                )

        return TableObservation(
            confidence=conf,
            mask=clean,
            contour=contour,
            corners=corners,
            homography=homography,
            area_ratio=area_ratio,
            bbox=(x, y, bw, bh),
        )


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
