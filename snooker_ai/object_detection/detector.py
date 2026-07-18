"""Lightweight snooker-ball observations inside the localized table.

The production detector hook remains optional, but the CPU fallback is more than a
plain Hough-circle pass: it combines cloth-deviation components with circular
proposals, estimates ball scale from the visible table, and assigns a separate
cue-ball colour confidence.  The resulting observations are intentionally small
and dependency-free so they can feed the tracker on every analysis frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.utils.logging import get_logger

logger = get_logger("object_detection")


@dataclass
class Detection:
    """A single image-space object observation.

    The first five fields preserve the original constructor/API.  ``radius`` and
    ``diameter`` default from the bounding box, so older callers automatically
    provide useful scale information to the tracker.
    """

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    cx: float
    cy: float
    radius: float = 0.0
    diameter: float = 0.0
    color_confidence: float = 0.0
    shape_confidence: float = 0.0
    cloth_surround_confidence: float = 0.0

    def __post_init__(self) -> None:
        _, _, w, h = self.bbox
        if self.radius <= 0.0 and self.diameter > 0.0:
            self.radius = float(self.diameter) * 0.5
        if self.diameter <= 0.0 and self.radius > 0.0:
            self.diameter = float(self.radius) * 2.0
        if self.radius <= 0.0 and self.diameter <= 0.0:
            size = float(max(0, min(w, h)))
            self.diameter = size
            self.radius = size * 0.5
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.color_confidence = float(np.clip(self.color_confidence, 0.0, 1.0))
        self.shape_confidence = float(np.clip(self.shape_confidence, 0.0, 1.0))
        self.cloth_surround_confidence = float(
            np.clip(self.cloth_surround_confidence, 0.0, 1.0)
        )

    @property
    def diameter_px(self) -> float:
        return float(self.diameter if self.diameter > 0.0 else self.radius * 2.0)


@dataclass
class _Proposal:
    cx: float
    cy: float
    radius: float
    shape_confidence: float
    source_count: int = 1


class ObjectDetector:
    def __init__(self, config: Config):
        self.cfg = config.section("object_detection")
        self.enabled = bool(self.cfg.get("enabled", False))
        table_cfg = config.section("table_detection")
        self.cloth_lower = np.array(
            table_cfg.get("hsv_lower", [35, 40, 40]), dtype=np.uint8
        )
        self.cloth_upper = np.array(
            table_cfg.get("hsv_upper", [95, 255, 255]), dtype=np.uint8
        )
        self.model = None
        self._diameter_ema = 0.0
        if self.enabled:
            self._try_load_model()

    def _try_load_model(self) -> None:
        path = self.cfg.get("model_path")
        if not path:
            logger.warning("object_detection.enabled but no model_path; using CPU observations")
            self.enabled = False
            return
        try:
            if not Path(path).is_file():
                logger.warning("Model not found: %s - falling back to CPU observations", path)
                self.enabled = False
                return
            # Keep the hook explicit.  A path alone is not treated as a loaded model.
            logger.warning(
                "No model runtime is configured for %s; using CPU observations", path
            )
            self.enabled = False
        except Exception as exc:  # pragma: no cover - defensive filesystem failure
            logger.warning("Failed to inspect detector model: %s", exc)
            self.enabled = False

    def detect(
        self,
        frame_bgr: np.ndarray,
        table_mask: Optional[np.ndarray] = None,
        *,
        use_hough: bool = True,
    ) -> list[Detection]:
        if self.model is not None:
            return self._detect_model(frame_bgr, table_mask)
        return self._detect_blobs(frame_bgr, table_mask, use_hough=use_hough)

    def estimated_ball_diameter(self) -> float:
        """Return the temporally smoothed image-space ball diameter in pixels."""

        return float(self._diameter_ema)

    def _detect_model(
        self, frame_bgr: np.ndarray, table_mask: Optional[np.ndarray]
    ) -> list[Detection]:
        # A future learned backend must return the same scale-aware observations.
        return self._detect_blobs(frame_bgr, table_mask)

    @staticmethod
    def _table_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return x0, y0, x1, y1

    def _diameter_prior(self, table_w: int, table_h: int) -> float:
        # A snooker ball is about 1.47% of the table's playing length.  Perspective
        # changes apparent scale, so this is only a proposal prior and is updated
        # from accepted circular components below.
        geometric = max(table_w, table_h) * 0.0147
        upper = max(5.0, min(table_w, table_h) * 0.09)
        prior = float(np.clip(geometric, 4.0, upper))
        if self._diameter_ema > 0.0:
            prior = 0.7 * self._diameter_ema + 0.3 * prior
        return prior

    def _detect_blobs(
        self,
        frame_bgr: np.ndarray,
        table_mask: Optional[np.ndarray],
        *,
        use_hough: bool = True,
    ) -> list[Detection]:
        """Find ball-scale cloth deviations and circular candidates.

        Large non-cloth regions (hands, cue, cushions and overlays) are rejected by
        scale/circularity.  Hough proposals recover individual balls in tight packs;
        component proposals make isolated and softly focused balls less dependent on
        edge contrast.
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        if table_mask is None or table_mask.shape[:2] != (h, w):
            mask = np.full((h, w), 255, dtype=np.uint8)
        else:
            mask = (table_mask > 0).astype(np.uint8) * 255

        x0, y0, x1, y1 = self._table_bbox(mask)
        if x1 <= x0 or y1 <= y0:
            return []
        roi = frame_bgr[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        table_h, table_w = roi.shape[:2]
        diameter_prior = self._diameter_prior(table_w, table_h)
        radius_prior = diameter_prior * 0.5

        # Ignore the uncertain table boundary/cushion transition.
        edge_margin = max(1, int(round(diameter_prior * 0.25)))
        kernel_edge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_margin + 1, 2 * edge_margin + 1)
        )
        inner_mask = cv2.erode(roi_mask, kernel_edge)
        if np.count_nonzero(inner_mask) < 0.25 * np.count_nonzero(roi_mask):
            inner_mask = roi_mask

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cloth = cv2.inRange(hsv, self.cloth_lower, self.cloth_upper)
        # Broadcast grading often gives the white ball a pale green cast.  Such
        # pixels can still fall inside the broad cloth HSV range, so explicitly
        # remove bright, low-saturation pixels before forming cloth deviation.
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 145], dtype=np.uint8),
            np.array([179, 110, 255], dtype=np.uint8),
        )
        white = cv2.bitwise_and(white, inner_mask)
        cloth = cv2.bitwise_and(cloth, cv2.bitwise_not(white))
        deviation = cv2.bitwise_and(cv2.bitwise_not(cloth), inner_mask)
        # A light opening removes codec speckle without joining a cluster of reds.
        deviation = cv2.morphologyEx(
            deviation,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        proposals: list[_Proposal] = []

        # The cue ball is often motion-blurred at the exact impact frame and can
        # lose the crisp circular edge required by HoughCircles.  A dedicated
        # bright/neutral component pass keeps that launch observable.  The
        # component still has to be ball-sized and is later checked for a green
        # cloth annulus, so white shirts, cushions and broadcast graphics do not
        # become cue-ball observations.
        white_contours, _ = cv2.findContours(
            white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        white_min_area = max(5.0, np.pi * radius_prior * radius_prior * 0.20)
        white_max_area = np.pi * radius_prior * radius_prior * 5.0
        for contour in white_contours:
            area = float(cv2.contourArea(contour))
            if not (white_min_area <= area <= white_max_area):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not (0.30 * diameter_prior <= radius <= 1.35 * diameter_prior):
                continue
            circularity = float(
                np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0, 1)
            )
            fill = float(np.clip(area / (np.pi * radius * radius + 1e-6), 0, 1))
            shape = float(np.clip(0.50 + 0.30 * circularity + 0.20 * fill, 0, 1))
            proposals.append(_Proposal(float(cx), float(cy), float(radius), shape))

        contours, _ = cv2.findContours(
            deviation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        target_area = np.pi * radius_prior * radius_prior
        min_area = max(4.0, target_area * 0.16)
        max_area = target_area * 3.8
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0, 1))
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not (0.24 * diameter_prior <= radius <= 1.10 * diameter_prior):
                continue
            fill = float(np.clip(area / (np.pi * radius * radius + 1e-6), 0, 1))
            shape = float(np.clip(0.65 * circularity + 0.35 * fill, 0, 1))
            if shape < 0.28:
                continue
            proposals.append(_Proposal(float(cx), float(cy), float(radius), shape))

        # Circular edge proposals split touching components and recover pale balls.
        min_r = max(2, int(round(diameter_prior * 0.28)))
        max_r = max(min_r + 1, int(round(diameter_prior * 0.78)))
        if use_hough:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(4.0, diameter_prior * 0.65),
                param1=90,
                param2=14,
                minRadius=min_r,
                maxRadius=max_r,
            )
        else:
            circles = None
        if circles is not None:
            for cx, cy, radius in circles[0]:
                ix, iy = int(round(float(cx))), int(round(float(cy)))
                if not (0 <= ix < table_w and 0 <= iy < table_h):
                    continue
                if inner_mask[iy, ix] == 0:
                    continue
                local_r = max(2, int(round(float(radius) * 0.8)))
                yy0, yy1 = max(0, iy - local_r), min(table_h, iy + local_r + 1)
                xx0, xx1 = max(0, ix - local_r), min(table_w, ix + local_r + 1)
                patch = deviation[yy0:yy1, xx0:xx1]
                if patch.size == 0 or np.count_nonzero(patch) / patch.size < 0.08:
                    continue
                proposals.append(
                    _Proposal(float(cx), float(cy), float(radius), 0.58)
                )

        proposals = self._merge_proposals(proposals, diameter_prior)
        detections: list[Detection] = []
        plausible_diameters: list[float] = []
        for proposal in proposals:
            color_conf, deviation_conf, surround_conf = self._colour_scores(
                hsv, cloth, proposal.cx, proposal.cy, proposal.radius
            )
            if deviation_conf < 0.08:
                continue
            # The white ball inherits a green cast under some broadcast colour
            # grades.  Permit that lower neutral-colour score only when a strong
            # majority of the surrounding annulus is genuine cloth; bright rail
            # and scoreboard details fail the surround gate.
            cue_ball = color_conf >= 0.50 and surround_conf >= 0.65
            label = "cue_ball" if cue_ball else "object_ball"
            shape = proposal.shape_confidence
            observation_conf = float(
                np.clip(0.24 + 0.42 * shape + 0.24 * deviation_conf, 0.0, 0.92)
            )
            if cue_ball:
                observation_conf = max(
                    observation_conf, float(np.clip(0.42 + 0.48 * color_conf, 0, 0.96))
                )
            radius = float(proposal.radius)
            diameter = radius * 2.0
            gx, gy = float(x0 + proposal.cx), float(y0 + proposal.cy)
            bx = int(round(gx - radius))
            by = int(round(gy - radius))
            size = max(1, int(round(diameter)))
            detections.append(
                Detection(
                    label=label,
                    confidence=observation_conf,
                    bbox=(bx, by, size, size),
                    cx=gx,
                    cy=gy,
                    radius=radius,
                    diameter=diameter,
                    color_confidence=color_conf if cue_ball else deviation_conf,
                    shape_confidence=shape,
                    cloth_surround_confidence=surround_conf,
                )
            )
            if shape >= 0.45 and 0.40 * diameter_prior <= diameter <= 2.2 * diameter_prior:
                plausible_diameters.append(diameter)

        if plausible_diameters:
            measured = float(np.median(plausible_diameters))
            if self._diameter_ema <= 0.0:
                self._diameter_ema = measured
            else:
                # Slow scale adaptation prevents one false circle changing all gates.
                ratio = measured / max(self._diameter_ema, 1e-6)
                if 0.55 <= ratio <= 1.8:
                    self._diameter_ema = 0.85 * self._diameter_ema + 0.15 * measured
        elif self._diameter_ema <= 0.0:
            self._diameter_ema = diameter_prior

        # There is exactly one cue ball.  Keep only the strongest white blob
        # whose annulus is surrounded by cloth; rail/pocket highlights become
        # ordinary low-priority objects and cannot spawn competing cue tracks.
        cue_candidates = [d for d in detections if d.label == "cue_ball"]
        if len(cue_candidates) > 1:
            best_cue = max(
                cue_candidates,
                key=lambda d: (
                    d.color_confidence * d.cloth_surround_confidence,
                    d.confidence,
                    d.shape_confidence,
                ),
            )
            for detection in cue_candidates:
                if detection is best_cue:
                    continue
                detection.label = "object_ball"
                detection.confidence *= 0.65

        # Higher confidence first makes downstream tie-breaking deterministic.
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    @staticmethod
    def _merge_proposals(
        proposals: list[_Proposal], diameter_prior: float
    ) -> list[_Proposal]:
        merged: list[_Proposal] = []
        for proposal in sorted(proposals, key=lambda p: p.shape_confidence, reverse=True):
            match: Optional[_Proposal] = None
            for current in merged:
                distance = float(np.hypot(proposal.cx - current.cx, proposal.cy - current.cy))
                if distance <= max(2.0, 0.42 * diameter_prior):
                    match = current
                    break
            if match is None:
                merged.append(proposal)
                continue
            total = match.source_count + proposal.source_count
            match.cx = (match.cx * match.source_count + proposal.cx * proposal.source_count) / total
            match.cy = (match.cy * match.source_count + proposal.cy * proposal.source_count) / total
            match.radius = (
                match.radius * match.source_count + proposal.radius * proposal.source_count
            ) / total
            match.shape_confidence = min(
                1.0, max(match.shape_confidence, proposal.shape_confidence) + 0.08
            )
            match.source_count = total
        return merged

    @staticmethod
    def _colour_scores(
        roi_hsv: np.ndarray,
        cloth_mask: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> tuple[float, float, float]:
        h, w = roi_hsv.shape[:2]
        r = max(2, int(round(radius * 0.72)))
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0, 0.0
        patch = roi_hsv[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(r * r)
        pixels = patch[disc]
        if pixels.size == 0:
            return 0.0, 0.0, 0.0
        # Reuse the full ROI HSV conversion already needed for cloth masking.
        # Converting every individual proposal again was pure duplicate CPU
        # work and becomes expensive over tens of thousands of match frames.
        saturation = float(np.median(pixels[:, 1]))
        value = float(np.median(pixels[:, 2]))
        # White cue ball: bright and neutral.  Both conditions are required, which
        # avoids classifying bright yellow/pink balls as the cue ball.
        brightness = float(np.clip((value - 135.0) / 100.0, 0, 1))
        neutrality = float(np.clip((120.0 - saturation) / 105.0, 0, 1))
        white_score = float(np.sqrt(brightness * neutrality))
        cloth_patch = cloth_mask[y0:y1, x0:x1]
        deviation = float(np.count_nonzero((cloth_patch == 0) & disc) / max(1, np.count_nonzero(disc)))
        # Annulus outside the candidate should be green cloth.  This strongly
        # suppresses bright pocket jaws, rail bolts and scoreboard glyphs.
        outer_r = max(r + 1, int(round(radius * 1.9)))
        ox0, ox1 = max(0, ix - outer_r), min(w, ix + outer_r + 1)
        oy0, oy1 = max(0, iy - outer_r), min(h, iy + outer_r + 1)
        oyy, oxx = np.ogrid[oy0:oy1, ox0:ox1]
        outer = (oxx - cx) ** 2 + (oyy - cy) ** 2 <= float(outer_r * outer_r)
        inner = (oxx - cx) ** 2 + (oyy - cy) ** 2 <= float(max(r, 1) * max(r, 1))
        annulus = outer & ~inner
        cloth_outer = cloth_mask[oy0:oy1, ox0:ox1]
        surround = (
            float(np.count_nonzero((cloth_outer > 0) & annulus) / np.count_nonzero(annulus))
            if np.count_nonzero(annulus)
            else 0.0
        )
        return white_score, float(np.clip(deviation, 0, 1)), float(np.clip(surround, 0, 1))
