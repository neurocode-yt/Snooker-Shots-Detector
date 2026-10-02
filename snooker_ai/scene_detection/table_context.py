"""Per-view geometry and visible hand/ball interactions for broadcast footage.

These observations are deliberately distinct: visible cloth permits local ball
tracking, while an enclosed playing surface is needed to prove all-ball stillness.
The hand detector describes visible interaction, not the person's identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from snooker_ai.object_detection.detector import Detection
from snooker_ai.table_detection.localizer import TableObservation, _order_quad


@dataclass(frozen=True)
class ViewGeometry:
    full_table: bool = False
    homography: np.ndarray | None = None


def view_geometry(table: TableObservation, shape: tuple[int, ...]) -> ViewGeometry:
    """Calibrate only when the cloth has four enclosed, plausible outer corners.

    A hull tolerates a player's bridge obscuring part of the cloth. A crop edge
    cannot become a table cushion; partial/close views have no global calibration.
    """
    h, w = shape[:2]
    if table.contour is None or table.confidence < 0.25 or table.area_ratio < 0.09:
        return ViewGeometry()
    hull = cv2.convexHull(table.contour)
    x, y, bw, bh = cv2.boundingRect(hull)
    margin = max(2, int(min(w, h) * 0.006))
    if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
        return ViewGeometry()
    perimeter = cv2.arcLength(hull, True)
    quad = hull
    for epsilon in (0.005, 0.01, 0.015, 0.025):
        quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(quad) == 4:
            break
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return ViewGeometry()
    corners = _order_quad(quad.reshape(4, 2).astype(np.float32))
    # RDP chooses a raster vertex near a corner, which can displace a long
    # shallow edge noticeably. Fit its actual cloth-border points and intersect
    # adjacent lines instead of calibrating from that displaced pixel.
    boundary = table.contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        first, second = corners[i], corners[(i+1) % 4]
        delta = second-first
        length2 = float(np.dot(delta, delta))
        if length2 <= 1:
            return ViewGeometry()
        relative = boundary-first
        along = relative @ delta / length2
        distance = np.abs(relative[:, 0]*delta[1]-relative[:, 1]*delta[0]) / np.sqrt(length2)
        selected = boundary[(along > 0.05) & (along < 0.95) & (distance <= 3)]
        if len(selected) < 4:
            lines.append((delta/np.sqrt(length2), first))
        else:
            vx, vy, px, py = cv2.fitLine(selected, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            lines.append((np.array([vx, vy]), np.array([px, py])))
    refined = []
    for i in range(4):
        direction_a, point_a = lines[(i-1) % 4]
        direction_b, point_b = lines[i]
        matrix = np.column_stack((direction_a, -direction_b))
        if abs(np.linalg.det(matrix)) < 0.05:
            return ViewGeometry()
        factor = np.linalg.solve(matrix, point_b-point_a)[0]
        refined.append(point_a+factor*direction_a)
    corners = np.asarray(refined, np.float32)
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
    if min(edges) < min(w, h) * 0.12 or max(edges) / min(edges) > 6:
        return ViewGeometry()
    area = cv2.contourArea(corners)
    if not 0.09 <= area / (h * w) <= 0.80:
        return ViewGeometry()
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return ViewGeometry(True, cv2.getPerspectiveTransform(corners, dst))


def ball_layout(frame: np.ndarray, detections: list[Detection], geometry: ViewGeometry) -> list[float]:
    """Fixed white/yellow/green/brown/blue/pink/black positions; missing=-1.

    Layout repetition is supporting replay evidence, never proof on its own.
    Reds are deliberately excluded because their interchangeable identities
    would make a fixed-order vector misleading.
    """
    if not geometry.full_table or geometry.homography is None:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = [-1.0] * 14
    quality = [0.0] * 7
    for d in detections:
        if d.shape_confidence < 0.48 or d.cloth_surround_confidence < 0.45:
            continue
        r = max(1, int(d.radius * 0.55))
        x, y = int(round(d.cx)), int(round(d.cy))
        patch = hsv[max(0, y-r):y+r+1, max(0, x-r):x+r+1]
        if not patch.size:
            continue
        hue, sat, value = np.median(patch.reshape(-1, 3), axis=0)
        if d.label == "cue_ball":
            colour = 0
        elif value < 65:
            colour = 6
        elif 17 <= hue <= 35 and sat > 100 and value > 110:
            colour = 1
        elif 40 <= hue <= 90 and sat > 100:
            colour = 2
        elif 5 <= hue < 23 and value < 150 and sat > 90:
            colour = 3
        elif 90 <= hue <= 135 and sat > 90:
            colour = 4
        elif hue >= 140 and sat < 170 and value > 100:
            colour = 5
        else:
            continue
        if d.confidence <= quality[colour]:
            continue
        point = cv2.perspectiveTransform(np.array([[[d.cx, d.cy]]], np.float32), geometry.homography)[0, 0]
        if np.all((point >= 0) & (point <= 1)):
            result[colour*2:colour*2+2] = point.astype(float).tolist()
            quality[colour] = d.confidence
    return result


class TableInteractionDetector:
    """Detect sustained contact between a broad hand/glove and a coloured ball.

    A bridge beside the white ball alone is never handling evidence. The gate
    requires contact with an object ball (including its last visible position),
    no visible cue-address geometry, and sustained observations in one view.
    """

    def __init__(self) -> None:
        self.since: float | None = None
        self.last_t: float | None = None
        self.recent_objects: list[tuple[float, Detection]] = []
        self.last_cue_address: float | None = None

    def reset(self) -> None:
        self.since = self.last_t = None
        self.recent_objects = []
        self.last_cue_address = None

    def observe(self, frame: np.ndarray, table: TableObservation, detections: list[Detection],
                t: float, cue_visible: bool = False) -> tuple[bool, float]:
        if table.mask is None or table.confidence < 0.25:
            self.reset()
            return False, 0.0
        scale = min(1.0, 480 / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        mask = cv2.resize(table.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        glove = cv2.inRange(hsv, (0, 0, 150), (179, 90, 255))
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        # Red/pink balls can satisfy YCrCb skin thresholds. Saturated round
        # components must not join into a fictitious hand over a red pack.
        skin[(hsv[:, :, 1] > 160) | (hsv[:, :, 2] < 70)] = 0
        hands = cv2.morphologyEx(glove | skin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        diameter = float(np.median([d.diameter_px for d in detections])) if detections else 8 / scale
        diameter = max(3.0, diameter * scale)
        for d in detections:
            if d.shape_confidence >= 0.48:
                cv2.circle(hands, (int(d.cx*scale), int(d.cy*scale)),
                           max(1, int(d.radius*scale*0.85)), 0, -1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(hands)
        accepted = []
        for i in range(1, count):
            _, _, bw, bh, area = stats[i]
            if area < 2.5 * np.pi * (diameter / 2) ** 2 or max(bw, bh) < 2.8 * diameter:
                continue
            on_cloth = np.count_nonzero((labels == i) & (mask > 0))
            neutral_fraction = np.count_nonzero((labels == i) & (glove > 0)) / max(area, 1)
            # Skin-colour contact alone includes a player's bridge. A neutral
            # glove covering a coloured ball is stronger handling evidence;
            # bare hands remain unknown unless another detector proves pickup.
            if neutral_fraction >= 0.65 and max(bw, bh) <= 16*diameter and on_cloth >= max(0.35*area, 0.8*diameter**2):
                accepted.append(i)
        hand_mask = np.isin(labels, accepted).astype(np.uint8)
        radius = max(2, int(diameter * 0.8))
        near = cv2.dilate(hand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
        objects = [d for d in detections if d.label != "cue_ball" and d.shape_confidence >= 0.48]
        self.recent_objects = [(seen, d) for seen, d in self.recent_objects if t-seen <= 0.6]
        for d in objects:
            self.recent_objects = [(seen, previous) for seen, previous in self.recent_objects
                                   if np.hypot(previous.cx-d.cx, previous.cy-d.cy) > d.diameter_px]
            self.recent_objects.append((t, d))
        interacting = False
        for _, d in self.recent_objects:
            x, y = int(d.cx * scale), int(d.cy * scale)
            if 0 <= x < near.shape[1] and 0 <= y < near.shape[0] and near[y, x]:
                interacting = True
                break
        continuous = self.last_t is not None and 0 < t - self.last_t <= 0.76
        if cue_visible:
            self.last_cue_address = t
        # Cue geometry can disappear for a few frames at contact while the
        # player's pale bridge remains next to a colour. Keep that established
        # cue-address context briefly through the launch rather than calling
        # the same hand referee handling as soon as the white blurs.
        recent_address = self.last_cue_address is not None and t-self.last_cue_address <= 1.0
        if interacting and not recent_address:
            if not continuous or self.since is None:
                self.since = t
        else:
            self.since = None
        handling = self.since is not None and t - self.since >= 0.15
        self.last_t = t
        return handling, (0.85 if handling else 0.35 if interacting else 0.0)
