"""Residual motion inside the table after camera compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.motion.camera import CameraMotionEstimator
from snooker_ai.utils.acceleration import acceleration_enabled, disable_acceleration


@dataclass
class MotionSample:
    residual_mean: float
    residual_max: float
    motion_area_ratio: float
    camera_magnitude: float
    is_camera_unstable: bool
    motion_score: float  # smoothed 0..1 for strike onset
    motion_raw: float  # unsmoothed 0..1 for ball-stop decisions
    # Residual flow restricted to ball-sized regions, normalised to ball
    # diameters/second.  Aggregate table flow is retained for compatibility but
    # must not be used as the primary all-ball stop signal.
    ball_residual_motion: float = 0.0
    observation_valid: bool = True


class ResidualMotionAnalyzer:
    def __init__(self, config: Config):
        self.cam = CameraMotionEstimator(config)
        mcfg = config.section("motion")
        self.flow_levels = int(mcfg.get("flow_levels", 3))
        self.flow_winsize = int(mcfg.get("flow_winsize", 15))
        self.mean_thr = float(mcfg.get("residual_mean_threshold", 0.55))
        self.max_thr = float(mcfg.get("residual_max_threshold", 1.8))
        self.area_thr = float(mcfg.get("motion_area_ratio_threshold", 0.004))
        self.ema_alpha = float(mcfg.get("ema_alpha", 0.60))
        self.flow_scale = float(np.clip(mcfg.get("flow_scale", 0.5), 0.25, 1.0))
        self._ema_score = 0.0
        self.use_opencl = acceleration_enabled(config)

    def analyze(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        table_mask: Optional[np.ndarray],
        ball_regions: Optional[list[tuple[float, float, float]]] = None,
        frame_dt: float = 1.0,
    ) -> MotionSample:
        cam = self.cam.estimate(prev_gray, gray, mask=table_mask)
        aligned = prev_gray
        if cam.transform is not None and not cam.is_cut_like:
            aligned = self.cam.warp_prev(prev_gray, cam.transform)

        # Flow on a half-resolution table is ~5x cheaper while preserving
        # ball-sized motion after scale normalisation.  Camera estimation stays
        # at the source proxy resolution so the affine compensation remains
        # stable.
        scale = self.flow_scale
        if scale < 0.999:
            flow_prev = cv2.resize(aligned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            flow_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            flow_mask = (
                cv2.resize(table_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                if table_mask is not None
                else None
            )
        else:
            flow_prev, flow_gray, flow_mask = aligned, gray, table_mask

        def calculate_flow(first, second):
            return cv2.calcOpticalFlowFarneback(
                first,
                second,
                None,
                pyr_scale=0.5,
                levels=self.flow_levels,
                winsize=self.flow_winsize,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        use_opencl = self.use_opencl and cv2.ocl.useOpenCL()
        try:
            flow_inputs = (
                cv2.UMat(flow_prev),
                cv2.UMat(flow_gray),
            ) if use_opencl else (flow_prev, flow_gray)
            flow = calculate_flow(*flow_inputs)
        except cv2.error as exc:
            if not use_opencl:
                raise
            disable_acceleration(str(exc))
            self.use_opencl = False
            flow = calculate_flow(flow_prev, flow_gray)
        if isinstance(flow, cv2.UMat):
            flow = flow.get()
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        if scale < 0.999:
            mag /= scale

        if flow_mask is not None and flow_mask.shape[:2] == mag.shape[:2]:
            region = mag[flow_mask > 0]
            mask_area = float(np.count_nonzero(flow_mask))
        else:
            region = mag.reshape(-1)
            mask_area = float(mag.size)

        if region.size == 0:
            return MotionSample(
                0.0,
                0.0,
                0.0,
                cam.magnitude,
                cam.is_cut_like,
                0.0,
                0.0,
                0.0,
                not cam.is_cut_like,
            )

        residual_mean = float(np.mean(region))
        residual_max = float(np.percentile(region, 95))

        # Adaptive pixel threshold: above noise floor on this frame.
        # Fixed threshold alone over-triggers on cloth grain / compression.
        med = float(np.median(region))
        mad = float(np.median(np.abs(region - med))) + 1e-6
        adaptive = med + 3.5 * 1.4826 * mad
        pix_thr = max(self.mean_thr, adaptive)
        moving = float(np.count_nonzero(region > pix_thr))
        motion_area_ratio = moving / max(mask_area, 1.0)

        if cam.is_cut_like:
            raw_score = 0.0
        else:
            # Emphasize p95 residual (real ball paths) over sparse noise area
            mean_n = float(np.clip(residual_mean / (self.mean_thr * 4.0), 0, 1))
            max_n = float(np.clip(residual_max / (self.max_thr * 3.0), 0, 1))
            area_n = float(np.clip(motion_area_ratio / 0.06, 0, 1))
            raw_score = float(0.30 * mean_n + 0.50 * max_n + 0.20 * area_n)
            # Strong camera translation residual often leaks — dampen
            if cam.magnitude > 2.5:
                raw_score *= float(np.clip(1.0 - (cam.magnitude - 2.5) / 10.0, 0.35, 1.0))

        # Measure only small regions around detected balls.  A player, cue, or
        # scoreboard can dominate whole-table flow but cannot create a
        # ball-sized residual here.  ``ball_regions`` are (cx, cy, diameter_px)
        # in the current frame.
        ball_residual = 0.0
        if ball_regions and not cam.is_cut_like:
            dt = max(float(frame_dt), 1e-3)
            # Sub-pixel codec shimmer around every stationary ball is common at
            # 25/30 fps.  Estimate that frame's table-wide optical-flow floor and
            # remove it before converting a local patch to ball diameters/second;
            # otherwise a quiet table reports ``1.0`` on nearly every frame.
            table_noise_floor = max(0.12, float(np.percentile(region, 95)) * 0.75)
            for cx, cy, diameter in ball_regions:
                radius = max(2, int(round(float(diameter) * 0.75 * scale)))
                ix, iy = int(round(float(cx) * scale)), int(round(float(cy) * scale))
                y0, y1 = max(0, iy - radius), min(mag.shape[0], iy + radius + 1)
                x0, x1 = max(0, ix - radius), min(mag.shape[1], ix + radius + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                patch = mag[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                # P90 suppresses isolated codec pixels while retaining a moving
                # ball's edge; subtract the common noise floor, then divide by
                # measured diameter and elapsed time.  A moving blur is several
                # pixels even after this floor, while a resting ball is not.
                local = max(0.0, float(np.percentile(patch, 90)) - table_noise_floor)
                normalized = local / max(float(diameter), 1.0) / dt
                ball_residual = max(ball_residual, float(np.clip(normalized / 0.80, 0, 1)))

        self._ema_score = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self._ema_score

        return MotionSample(
            residual_mean=residual_mean,
            residual_max=residual_max,
            motion_area_ratio=motion_area_ratio,
            camera_magnitude=cam.magnitude,
            is_camera_unstable=cam.is_cut_like,
            motion_score=float(self._ema_score),
            motion_raw=float(raw_score),
            ball_residual_motion=float(ball_residual),
            observation_valid=not cam.is_cut_like,
        )

    def reset(self) -> None:
        self._ema_score = 0.0
