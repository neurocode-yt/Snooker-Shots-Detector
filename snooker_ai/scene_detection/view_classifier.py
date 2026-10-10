"""Visual camera-view classification with explicit cloth visibility cues."""

from __future__ import annotations

from typing import Any
from functools import lru_cache
from pathlib import Path

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
            "replay_stinger_signature": self.replay_stinger_signature(frame),
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
    def replay_stinger_signature(frame: np.ndarray) -> list[float]:
        """Recognize measured broadcast graphics, never colour alone.

        These fingerprints associate opening and closing wipes; one graphic
        does not establish that the following play is a replay.
        """
        small = cv2.resize(frame, (320, 180))
        sphere = ViewClassifier._red_sphere_stinger(small)
        if sphere:
            return sphere
        blue = ViewClassifier._blue_title_stinger(small)
        if blue:
            return blue
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        if np.mean(hsv[:, :, 2] < 115) < .55:
            return []
        yellow = cv2.inRange(hsv, (15, 90, 140), (42, 255, 255)) > 0
        pink = cv2.inRange(hsv, (135, 85, 120), (175, 255, 255)) > 0
        if min(np.count_nonzero(yellow), np.count_nonzero(pink)) < 300:
            return []
        yy, xx = np.ogrid[:180, :320]
        dx, dy = xx-160, yy-90
        radius = np.hypot(dx, dy)
        angle = ((np.arctan2(dy, dx)+np.pi)*24/(2*np.pi)).astype(int) % 24
        rings = []
        for mask in (pink, yellow):
            candidates = []
            for r in range(36, 111, 2):
                annulus = np.abs(radius-r) <= 2.5
                coloured = annulus & mask
                if np.count_nonzero(coloured) / max(1, np.count_nonzero(annulus)) < .35:
                    continue
                if len(np.unique(angle[coloured])) < 20:
                    continue
                candidates.append(r)
            rings.append(candidates)
        if not any(3 <= yellow_r-pink_r <= 18 for pink_r in rings[0] for yellow_r in rings[1]):
            return []
        title = cv2.cvtColor(small[58:128, 106:214], cv2.COLOR_BGR2GRAY)
        return (cv2.resize(title, (16, 12)).astype(np.float32).ravel()/255).tolist()

    @staticmethod
    @lru_cache(maxsize=1)
    def _blue_title_template() -> np.ndarray | None:
        return cv2.imread(str(Path(__file__).parent/'assets/championship_title.png'), cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _blue_title_stinger(small: np.ndarray) -> list[float]:
        """Recognize a source-verified title wipe, rather than blue banners.

        A fullscreen cyan field must contain the measured Championship League
        lettering. The binary title template ignores video grading; paired
        opening and closing graphics still establish replay ownership elsewhere.
        A different vector length keeps this family separate from neon rings.
        """
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cyan = cv2.inRange(hsv, (85, 75, 145), (105, 220, 255))
        if np.mean(cyan > 0) < .60:
            return []
        title = (((hsv[60:126,64:256,1] < 80)
                  & (hsv[60:126,64:256,2] > 185))*255).astype(np.uint8)
        template = ViewClassifier._blue_title_template()
        if template is None or template.shape != title.shape or np.std(title) < 10:
            return []
        if float(cv2.matchTemplate(title, template, cv2.TM_CCOEFF_NORMED)[0,0]) < .90:
            return []
        return [2.]+(cv2.resize(title, (16,12)).astype(np.float32).ravel()/255).tolist()

    @staticmethod
    @lru_cache(maxsize=1)
    def _red_sphere_templates() -> tuple[np.ndarray, ...]:
        bank = cv2.imread(str(Path(__file__).parent/'assets/red_replay_sphere.png'), cv2.IMREAD_GRAYSCALE)
        if bank is None or bank.shape[1] != 96 or bank.shape[0] % 96:
            return ()
        return tuple(cv2.GaussianBlur(part, (3,3), 0)
                     for part in np.split(bank, bank.shape[0]//96))

    @staticmethod
    def _red_sphere_stinger(small: np.ndarray) -> list[float]:
        """Recognize the rendered sphere/reflection pattern of a replay wipe.

        The disc must occupy a substantial part of the frame and match a
        source-derived rendering template. A photographed red ball, red shirt
        or lens flare alone supplies no marker. Animation phases share one
        canonical fingerprint only after this measured template match.
        """
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv,(140,55,45),(179,255,255))
               | cv2.inRange(hsv,(0,55,45),(12,255,255)))
        if np.mean(red > 0) < .08:
            return []
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        templates = ViewClassifier._red_sphere_templates()
        if not templates:
            return []
        for contour in contours:
            x,y,width,height = cv2.boundingRect(contour)
            if (cv2.contourArea(contour) < .08*red.size or x < 2 or x+width >= 318
                    or not .70 <= width/max(1,height) <= 1.35
                    or cv2.contourArea(cv2.convexHull(contour))/max(1,width*height) < .65):
                continue
            patch = cv2.cvtColor(small[y:y+height,x:x+width], cv2.COLOR_BGR2GRAY)
            patch = cv2.GaussianBlur(cv2.resize(patch,(96,96)),(3,3),0)
            if np.std(patch) < 10:
                continue
            score = max(float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
                        for template in templates)
            if score >= .91:
                return [3.,0.]+(cv2.resize(templates[0],(16,12)).astype(np.float32).ravel()/255).tolist()
        return []

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
