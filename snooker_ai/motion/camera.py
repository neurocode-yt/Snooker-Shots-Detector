"""Estimate global camera motion between frames via optical flow + affine model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.utils.acceleration import acceleration_enabled


@dataclass
class CameraMotion:
    magnitude: float  # mean translation-like motion in px
    scale: float  # approx zoom factor (1.0 = none)
    inliers: int
    transform: Optional[np.ndarray]  # 2x3 affine
    is_cut_like: bool


class CameraMotionEstimator:
    def __init__(self, config: Config):
        cfg = config.section("camera_motion")
        self.max_corners = int(cfg.get("max_corners", 200))
        self.quality = float(cfg.get("quality_level", 0.01))
        self.min_distance = int(cfg.get("min_distance", 8))
        win = int(cfg.get("lk_win_size", 21))
        self.lk_win = (win, win)
        self.ransac = float(cfg.get("ransac_threshold", 3.0))
        self.min_inliers = int(cfg.get("min_inliers", 12))
        self.cut_mag = float(cfg.get("motion_magnitude_cut", 8.0))
        self.estimation_scale = float(
            np.clip(cfg.get("estimation_scale", 0.5), 0.25, 1.0)
        )
        self.use_opencl = acceleration_enabled(config)

    def estimate(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> CameraMotion:
        work_prev = prev_gray
        work_gray = gray
        work_mask = mask
        if self.estimation_scale < 0.999:
            scale = self.estimation_scale
            work_prev = cv2.resize(
                prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            work_gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            if mask is not None:
                work_mask = cv2.resize(
                    mask,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )

        # Prefer static features: invert table mask so we track cushions/background
        feature_mask = None
        if work_mask is not None:
            feature_mask = cv2.bitwise_not(work_mask)
            # Keep a border of table edges too
            feature_mask = cv2.dilate(feature_mask, np.ones((5, 5), np.uint8))

        prev_input = cv2.UMat(work_prev) if self.use_opencl else work_prev
        next_input = cv2.UMat(work_gray) if self.use_opencl else work_gray
        mask_input = cv2.UMat(feature_mask) if self.use_opencl and feature_mask is not None else feature_mask
        pts = cv2.goodFeaturesToTrack(
            prev_input,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            mask=mask_input,
        )
        if pts is None:
            # Without enough static features we cannot distinguish a genuine
            # table transition from a pan/zoom.  Mark the observation unknown;
            # treating it as a clean zero-motion frame would prematurely close a
            # strict shot.
            return CameraMotion(0.0, 1.0, 0, None, True)

        pts_input = pts
        pts_np = pts.get() if isinstance(pts, cv2.UMat) else pts
        if len(pts_np) < 8:
            return CameraMotion(0.0, 1.0, 0, None, True)

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_input, next_input, pts_input, None, winSize=self.lk_win, maxLevel=3
        )
        if nxt is None:
            return CameraMotion(0.0, 1.0, 0, None, True)

        if isinstance(nxt, cv2.UMat):
            nxt = nxt.get()
        if isinstance(status, cv2.UMat):
            status = status.get()

        good_prev = pts_np[status.flatten() == 1]
        good_next = nxt[status.flatten() == 1]
        if len(good_prev) < 6:
            return CameraMotion(0.0, 1.0, 0, None, True)

        transform, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac,
        )
        if transform is None:
            disp = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)
            mag = float(
                np.median(np.linalg.norm(disp, axis=1)) / self.estimation_scale
            )
            return CameraMotion(mag, 1.0, 0, None, mag >= self.cut_mag)

        n_in = int(inliers.sum()) if inliers is not None else 0
        # Affine: [a -b tx; b a ty] for partial
        a, b = transform[0, 0], transform[0, 1]
        # Convert the translation back to full proxy pixels.  The affine scale
        # and rotation terms are resolution-independent.
        transform = transform.copy()
        transform[0, 2] /= self.estimation_scale
        transform[1, 2] /= self.estimation_scale
        tx, ty = transform[0, 2], transform[1, 2]
        scale = float(np.hypot(a, b))
        mag = float(np.hypot(tx, ty))
        is_cut = mag >= self.cut_mag or n_in < self.min_inliers
        return CameraMotion(mag, scale if scale > 0 else 1.0, n_in, transform, is_cut)

    def warp_prev(self, prev_gray: np.ndarray, transform: np.ndarray) -> np.ndarray:
        h, w = prev_gray.shape[:2]
        return cv2.warpAffine(
            prev_gray,
            transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
