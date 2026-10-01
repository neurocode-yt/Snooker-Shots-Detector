"""Cheap, conservative waiting-table gate for whole-match proposal scans."""

from __future__ import annotations

import cv2
import numpy as np


class RackIdleGate:
    """Skip costly tracking only while a compact red rack and cloth stay still.

    Any cloth movement, obscured table, missing rack, or sampling gap wakes the
    detector immediately. A wake also requests a retrospective contact scan so
    the first shot after a long wait is not lost. Native refinement never uses
    this gate. This identifies waiting footage, not an official frame score.
    """

    def __init__(self, settle_seconds: float = 2.0):
        self.settle_seconds = settle_seconds
        self.previous: np.ndarray | None = None
        self.previous_t: float | None = None
        self.quiet_since: float | None = None
        self.idle = False
        self.just_released = False
        self.table_ratio = 0.0
        self.racked = False
        self.red_area_ratio = 0.0

    def observe(self, frame: np.ndarray, t: float) -> bool:
        small = cv2.resize(frame, (480, 270))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 45, 35), (95, 255, 255))
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(green.shape, np.uint8)
        if contours:
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        table_area = cv2.countNonZero(mask)
        self.table_ratio = table_area / mask.size
        red = cv2.bitwise_and(
            cv2.inRange(hsv, (0, 100, 55), (12, 255, 255))
            | cv2.inRange(hsv, (170, 100, 55), (179, 255, 255)), mask,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(red)
        self.red_area_ratio = cv2.countNonZero(red) / max(table_area, 1)
        racked = False
        if count > 1 and self.table_ratio >= 0.12:
            component = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
            _, _, w, h, area = component
            # A single red ball or an incomplete/scattered pack is insufficient.
            racked = bool(
                0.0035 <= area / table_area <= 0.018
                and area / max(cv2.countNonZero(red), 1) >= 0.94
                and 0.9 <= w / max(h, 1) <= 2.2
                and 0.35 <= area / max(w * h, 1) <= 0.72
            )
        self.racked = racked
        continuous = self.previous_t is not None and 0 < t - self.previous_t <= 0.76
        quiet = False
        if racked and continuous and self.previous is not None:
            difference = np.max(cv2.absdiff(small, self.previous), axis=2)
            changed = np.count_nonzero((difference > 35) & (mask > 0))
            quiet = changed <= max(4, table_area * 0.0005)
        was_idle = self.idle
        if quiet:
            if self.quiet_since is None:
                self.quiet_since = t
            self.idle = t - self.quiet_since >= self.settle_seconds
        else:
            self.quiet_since = None
            self.idle = False
        self.just_released = was_idle and not self.idle
        self.previous, self.previous_t = small, t
        return self.idle
