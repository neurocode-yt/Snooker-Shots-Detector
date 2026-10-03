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
    cue_sphere_supported: bool = False
    cue_sphere_red_occlusion: bool = False
    cue_sphere_black_occlusion: bool = False

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
    cue_sphere_supported: bool = False
    cue_sphere_red_occlusion: bool = False
    cue_sphere_black_occlusion: bool = False


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
        partial_view: bool = False,
        table_bounds: tuple[int, int, int, int] | None = None,
    ) -> list[Detection]:
        if self.model is not None:
            return self._detect_model(frame_bgr, table_mask)
        return self._detect_blobs(frame_bgr, table_mask, use_hough=use_hough, partial_view=partial_view,
                                  table_bounds=table_bounds)

    def estimated_ball_diameter(self) -> float:
        """Return the temporally smoothed image-space ball diameter in pixels."""

        return float(self._diameter_ema)

    def reset(self) -> None:
        """Discard view-local scale when seeking or switching cameras."""
        self._diameter_ema = 0.0

    def _detect_model(
        self, frame_bgr: np.ndarray, table_mask: Optional[np.ndarray]
    ) -> list[Detection]:
        # A future learned backend must return the same scale-aware observations.
        return self._detect_blobs(frame_bgr, table_mask)

    @staticmethod
    def _table_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        points = cv2.findNonZero(mask)
        if points is None:
            return 0, 0, mask.shape[1], mask.shape[0]
        x, y, w, h = cv2.boundingRect(points)
        return x, y, x + w, y + h

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
        partial_view: bool = False,
        table_bounds: tuple[int, int, int, int] | None = None,
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
        if partial_view:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # In low views a bridge and the white ball can indent the
                # cloth contour together. Keep that playing-surface interior
                # available; subsequent foreground/shape tests reject hands.
                cv2.drawContours(mask, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, -1)

        x0, y0, x1, y1 = table_bounds if table_bounds is not None else self._table_bbox(mask)
        upper_ball_margin = max(4, int(round(min(h, w) * .06))) if partial_view else 0
        if partial_view:
            # A low camera sees the upper half of a ball above the visible
            # cloth boundary. Keep those source pixels in the ROI; only the
            # independently checked ivory-sphere proposals may use them.
            y0 = max(0, y0 - upper_ball_margin)
        if x1 <= x0 or y1 <= y0:
            return []
        roi = frame_bgr[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        table_h, table_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if partial_view:
            upper_mask = cv2.dilate(
                roi_mask, np.ones((upper_ball_margin + 1, 1), np.uint8),
                anchor=(0, 0),
            )
            warm_spheres = self._warm_cue_spheres(hsv, upper_mask)
        else:
            warm_spheres = []
        diameter_prior = self._diameter_prior(table_w, table_h)
        if partial_view:
            # A close-up can show a 50px white ball on only a fraction of the
            # playing surface. Table-length scale is meaningless there. Seed
            # from an enclosed, circular white component with a cloth annulus.
            observed = max(
                self._closeup_diameter(roi, roi_mask, hsv=hsv),
                max((p.radius * 2 for p in warm_spheres), default=0.0),
            )
            if observed > diameter_prior and (
                self._diameter_ema <= 0 or observed > 1.6*self._diameter_ema
            ):
                diameter_prior = observed
                self._diameter_ema = observed
        radius_prior = diameter_prior * 0.5

        # Ignore the uncertain table boundary/cushion transition.
        edge_margin = max(1, int(round(diameter_prior * 0.25)))
        kernel_edge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_margin + 1, 2 * edge_margin + 1)
        )
        inner_mask = cv2.erode(roi_mask, kernel_edge)
        if np.count_nonzero(inner_mask) < 0.25 * np.count_nonzero(roi_mask):
            inner_mask = roi_mask

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
        # Keep connected foreground before opening can split a glove, fingers
        # or cue into plausible ball-sized fragments. Include pixels outside
        # the table mask so the glove stays connected to its arm. Red clusters
        # remain eligible; broad foreground regions and long thin cues do not.
        foreground_components = cv2.morphologyEx(
            cv2.bitwise_not(cloth), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_components)
        foreground_ids = []
        ball_area = np.pi * radius_prior * radius_prior
        for component in range(1, count):
            _, _, cw, ch, area = stats[component]
            if area < ball_area * 2.5:
                continue
            component_pixels = hsv[labels == component]
            red = ((component_pixels[:, 0] < 15) | (component_pixels[:, 0] > 165)) & (component_pixels[:, 1] > 110)
            elongated = max(cw, ch) > 3 * diameter_prior and max(cw, ch) > 3 * min(cw, ch)
            if elongated or (area > ball_area * 6 and float(np.mean(red)) < 0.65):
                foreground_ids.append(component)
        foreground = np.isin(labels, foreground_ids) if foreground_ids else None
        if foreground is not None:
            margin = max(1, int(round(diameter_prior * 0.35)))
            foreground = cv2.dilate(
                foreground.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)),
            ) > 0
        # A light opening removes codec speckle without joining a cluster of reds.
        deviation = cv2.morphologyEx(
            deviation,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        proposals: list[_Proposal] = list(warm_spheres)

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
            in_foreground = False
            if foreground is not None:
                px = int(np.clip(round(proposal.cx), 0, table_w - 1))
                py = int(np.clip(round(proposal.cy), 0, table_h - 1))
                in_foreground = bool(foreground[py, px])
            color_conf, deviation_conf, surround_conf = self._colour_scores(
                hsv, cloth, proposal.cx, proposal.cy, proposal.radius
            )
            if in_foreground and not proposal.cue_sphere_supported:
                continue
            if deviation_conf < 0.08:
                continue
            # The white ball inherits a green cast under some broadcast colour
            # grades.  Permit that lower neutral-colour score only when a strong
            # majority of the surrounding annulus is genuine cloth; bright rail
            # and scoreboard details fail the surround gate.
            ref_diameter = self._diameter_ema if self._diameter_ema > 0.0 else diameter_prior
            radius = float(proposal.radius)
            diameter = radius * 2.0
            size_ok = True
            if ref_diameter > 0.0:
                size_ratio = diameter / ref_diameter
                size_ok = 0.40 <= size_ratio <= 2.2
            shape = proposal.shape_confidence
            shape_ok = shape >= 0.38 or self._diameter_ema <= 0.0

            cue_ball = size_ok and shape_ok and (
                color_conf >= 0.50 and surround_conf >= 0.65
                or proposal.cue_sphere_supported and surround_conf >= 0.30
                or proposal.cue_sphere_red_occlusion and surround_conf >= 0.10
                or proposal.cue_sphere_black_occlusion and surround_conf >= 0.20
            )
            label = "cue_ball" if cue_ball else "object_ball"
            observation_conf = float(
                np.clip(0.24 + 0.42 * shape + 0.24 * deviation_conf, 0.0, 0.92)
            )
            if cue_ball:
                observation_conf = max(
                    observation_conf, float(np.clip(0.42 + 0.48 * color_conf, 0, 0.96))
                )
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
                    cue_sphere_supported=proposal.cue_sphere_supported and cue_ball,
                    cue_sphere_red_occlusion=proposal.cue_sphere_red_occlusion and cue_ball,
                    cue_sphere_black_occlusion=proposal.cue_sphere_black_occlusion and cue_ball,
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
                    1 if d.cue_sphere_supported else 0,
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

    def _closeup_diameter(
        self, roi: np.ndarray, mask: np.ndarray, *, hsv: np.ndarray | None = None
    ) -> float:
        if hsv is None:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 150), (179, 110, 255)) & mask
        cloth = cv2.inRange(hsv, self.cloth_lower, self.cloth_upper)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = (0.0, 0.0)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 30 or perimeter <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not 4 <= radius <= min(roi.shape[:2]) * 0.16:
                continue
            circularity = 4 * np.pi * area / perimeter**2
            fill = area / max(1, np.pi * radius**2)
            _, _, surround = self._colour_scores(hsv, cloth, cx, cy, radius)
            if circularity >= 0.55 and fill >= 0.40 and surround >= 0.70:
                # A tiny specular highlight on a green/yellow ball can be more
                # circular than the shaded white sphere. Prefer the large
                # coherent neutral component in a close-up.
                score = circularity * fill * surround * radius
                if score > best[0]:
                    best = (score, radius * 2)
        return best[1]

    def _warm_cue_spheres(self, hsv: np.ndarray, mask: np.ndarray) -> list[_Proposal]:
        """Separate shaded ivory spheres from a pink bridge in low views.

        A neutral mask alone joins the ball to the player's hand and cue. Warm
        ivory pixels preserve the ball's curved outline without including most
        skin. The compact outline, neutral highlight area and cloth support are
        independent requirements; hue or brightness alone never identifies it.
        """
        warm = cv2.inRange(hsv, (15, 0, 125), (44, 140, 255))
        # The sphere's ivory mask can include achromatic highlights. Pink skin
        # under the arena lighting has a near-neutral magenta cast; including
        # that cast joins the white ball to the bridge instead of its outline.
        neutral = cv2.inRange(hsv, (0, 0, 150), (139, 25, 255))
        candidate_mask = cv2.bitwise_and(cv2.bitwise_or(warm, neutral), mask)
        original_mask = candidate_mask
        candidate_mask = cv2.morphologyEx(
            candidate_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        cloth = cv2.inRange(hsv, self.cloth_lower, self.cloth_upper)
        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detached_contours = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area < 30 or perimeter <= 0:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            (_, _), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / perimeter**2
            fill = area / max(1, np.pi * radius**2)
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            hull_perimeter = float(cv2.arcLength(hull, True))
            hull_circularity = 4*np.pi*hull_area/max(1, hull_perimeter**2)
            rotated_w, rotated_h = cv2.minAreaRect(contour)[1]
            rotated_aspect = max(rotated_w, rotated_h) / max(1, min(rotated_w, rotated_h))
            appendage = (
                rotated_aspect >= 1.25
                or (max(bw, bh) >= 1.15 * min(bw, bh) and .70 <= hull_circularity < .85
                    and area/max(1, hull_area) >= .70)
            )
            # Retry only an elongated outline with a thin cue attached. Opening
            # a round fragmented blob would invent supporting sphere geometry.
            if not appendage or not (
                circularity < .65 or fill < .48 or max(bw, bh) > 1.80 * min(bw, bh)
            ):
                continue
            size = min(7, max(3, int(round(radius * .55)) | 1))
            x0, x1 = max(0, x-size), min(hsv.shape[1], x+bw+size)
            y0, y1 = max(0, y-size), min(hsv.shape[0], y+bh+size)
            # Re-open the original measured pixels rather than filling its
            # outline or applying successive openings to its shaded edge.
            patch = original_mask[y0:y1, x0:x1]
            offset = np.array([[[x0, y0]]], np.int32)
            opened = cv2.morphologyEx(
                patch, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            )
            separated, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detached_contours.extend(c + offset for c in separated)
        contours = list(contours) + detached_contours
        proposals = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area < 30 or perimeter <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            # The cropped playing surface can be a shallow strip in a low
            # camera while a ball occupies much of that strip's height.
            if not 4 <= radius <= min(min(hsv.shape[:2]) * .48, max(hsv.shape[:2]) * .18):
                continue
            _, _, bw, bh = cv2.boundingRect(contour)
            circularity = 4 * np.pi * area / perimeter**2
            fill = area / max(1, np.pi * radius**2)
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            hull_perimeter = float(cv2.arcLength(hull, True))
            hull_circularity = 4 * np.pi * hull_area / max(1, hull_perimeter**2)
            solidity = area / max(1, hull_area)
            # Compression makes the shaded lower edge jagged. Accept that
            # outline only when its convex envelope remains round and filled;
            # fingers and loose fragments fail these independent shape gates.
            compact_shaded_outline = (
                circularity >= 0.55 and hull_circularity >= 0.85 and solidity >= 0.85
            )
            # The visible warm hemisphere may be half as tall as it is wide;
            # its enclosing circle and independent fill gate retain ball scale.
            if (circularity < 0.65 and not compact_shaded_outline) or fill < 0.48 or max(bw, bh) > 2.0 * min(bw, bh):
                continue
            r = max(2, int(round(radius * 0.72)))
            x0, x1 = max(0, round(cx)-r), min(hsv.shape[1], round(cx)+r+1)
            y0, y1 = max(0, round(cy)-r), min(hsv.shape[0], round(cy)+r+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            pixels = hsv[y0:y1, x0:x1][(xx-cx)**2+(yy-cy)**2 <= r*r]
            if not len(pixels):
                continue
            hue, sat, value = np.median(pixels, axis=0)
            neutral_cap = float(np.mean((pixels[:, 1] < 110) & (pixels[:, 2] >= 210)))
            # Pink skin, neutral gloves and saturated yellow/brown balls fail
            # different gates even when one fragment happens to look circular.
            if not (15 <= hue <= 44 and 35 <= sat <= 125 and value >= 150):
                continue
            if neutral_cap < 0.25:
                continue
            _, _, surround = self._colour_scores(hsv, cloth, cx, cy, radius)
            red_occlusion = .10 <= surround < .30 and self._red_neighbors_occlude_sphere(hsv, cx, cy, radius)
            black_occlusion = .20 <= surround < .45 and self._black_neighbor_occludes_sphere(hsv, cx, cy, radius)
            if surround < 0.30 and not (red_occlusion or black_occlusion):
                continue
            proposals.append(_Proposal(
                float(cx), float(cy), float(radius),
                float(0.65 + 0.25 * circularity + 0.10 * fill),
                cue_sphere_supported=True,
                cue_sphere_red_occlusion=red_occlusion,
                cue_sphere_black_occlusion=black_occlusion,
            ))
        return proposals

    @staticmethod
    def _black_neighbor_occludes_sphere(hsv: np.ndarray, cx: float, cy: float, radius: float) -> bool:
        """Verify a round black ball directly in front of the ivory sphere."""
        reach = int(np.ceil(radius * 3.0))
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, ix-reach), min(hsv.shape[1], ix+reach+1)
        y0, y1 = max(0, iy-reach), min(hsv.shape[0], iy+reach+1)
        dark = cv2.inRange(hsv[y0:y1, x0:x1], (0, 0, 0), (179, 255, 80))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            (bx, by), br = cv2.minEnclosingCircle(contour)
            _, _, bw, bh = cv2.boundingRect(contour)
            if not .65*radius <= br <= 1.5*radius or max(bw, bh) > 1.5*min(bw, bh):
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            if (4*np.pi*area/perimeter**2 < .60 or area/max(1, np.pi*br**2) < .60
                    or area/max(1, hull_area) < .80):
                continue
            dx, dy = bx+x0-cx, by+y0-cy
            if dy >= .6*radius and abs(dx) <= radius and 1.1*radius <= np.hypot(dx, dy) <= 2.6*radius:
                return True
        return False

    @staticmethod
    def _red_neighbors_occlude_sphere(hsv: np.ndarray, cx: float, cy: float, radius: float) -> bool:
        """Require two distinct round red balls beside the ivory sphere.

        A bridge and two foreground reds can hide almost all of its cloth
        annulus in a low camera. Red pixels alone cannot relax that cloth gate.
        """
        reach = int(np.ceil(radius * 3.2))
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, ix-reach), min(hsv.shape[1], ix+reach+1)
        y0, y1 = max(0, iy-reach), min(hsv.shape[0], iy+reach+1)
        patch = hsv[y0:y1, x0:x1]
        red = cv2.inRange(patch, (0, 140, 60), (12, 255, 255))
        red |= cv2.inRange(patch, (165, 140, 60), (179, 255, 255))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        directions = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            (rx, ry), rr = cv2.minEnclosingCircle(contour)
            _, _, bw, bh = cv2.boundingRect(contour)
            if not .55 * radius <= rr <= 1.50 * radius or max(bw, bh) > 1.8 * min(bw, bh):
                continue
            if 4*np.pi*area/perimeter**2 < .50 or area/max(1, np.pi*rr**2) < .45:
                continue
            delta = np.array([rx+x0-cx, ry+y0-cy])
            distance = float(np.linalg.norm(delta))
            if not 1.25 * radius <= distance <= 2.6 * radius:
                continue
            direction = delta / max(distance, 1e-6)
            if any(float(np.dot(direction, other)) <= .5 for other in directions):
                return True
            directions.append(direction)
        return False

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
            # A Hough edge can be displaced into the shaded lower hemisphere.
            # Preserve the independently fitted ivory component when combining
            # it with weaker ordinary circular proposals.
            if proposal.cue_sphere_supported and not match.cue_sphere_supported:
                match.cx, match.cy, match.radius = proposal.cx, proposal.cy, proposal.radius
            elif not match.cue_sphere_supported:
                match.cx = (match.cx * match.source_count + proposal.cx * proposal.source_count) / total
                match.cy = (match.cy * match.source_count + proposal.cy * proposal.source_count) / total
                match.radius = (
                    match.radius * match.source_count + proposal.radius * proposal.source_count
                ) / total
            match.cue_sphere_supported |= proposal.cue_sphere_supported
            match.cue_sphere_red_occlusion |= proposal.cue_sphere_red_occlusion
            match.cue_sphere_black_occlusion |= proposal.cue_sphere_black_occlusion
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
