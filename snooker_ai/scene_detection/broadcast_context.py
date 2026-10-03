"""Conservative, inexpensive scoreboard appearance continuity.

This does not read player names.  It compares the stable name regions of a
recognisable two-player score overlay, excluding the changing score digits.
Missing overlays and unfamiliar layouts provide no identity evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType


@dataclass(frozen=True)
class BroadcastContextObservation:
    timestamp: float
    scoreboard_visible: bool = False
    identity_known: bool = False
    identity_similarity: float = 0.0
    foreign_match: bool = False
    foreign_interval_start: float | None = None
    target_ready: bool = False
    target_interval_start: float | None = None


class BroadcastContextGuard:
    """Learn a repeated live scoreboard and require sustained foreign evidence.

    A guard belongs to one chronological analysis stream.  Native refinement
    windows should inherit already measured coarse context; constructing a new
    guard inside a foreign-match window would learn that window as its target.
    """

    def __init__(self, config: Config):
        cfg = config.section("broadcast_context")
        self.learning_samples = max(3, int(cfg.get("learning_samples", 8)))
        self.learning_seconds = float(cfg.get("learning_seconds", 3.0))
        self.same_threshold = float(cfg.get("same_identity_similarity", 0.78))
        self.foreign_threshold = float(cfg.get("foreign_identity_similarity", 0.52))
        self.foreign_samples = max(2, int(cfg.get("foreign_samples", 3)))
        self.foreign_seconds = float(cfg.get("foreign_seconds", 1.0))
        self.max_gap = float(cfg.get("identity_max_gap_seconds", 2.0))
        self.return_max_seconds = float(cfg.get("target_return_max_seconds", 15.0))
        self.return_non_table_seconds = float(cfg.get("target_return_non_table_seconds", 1.0))
        self._learning: list[np.ndarray] = []
        self._learning_start = 0.0
        self._template: np.ndarray | None = None
        self._last_foreign = 0.0
        self._foreign_start: float | None = None
        self._foreign_count = 0
        self._last_time: float | None = None
        self._foreign_active = False
        self._non_table_start: float | None = None
        self._return_table_start: float | None = None

    @property
    def target_ready(self) -> bool:
        return self._template is not None

    def export_target(self) -> list[float]:
        """Return only the stable learned target for a resumable coarse cache."""
        return self._template.tolist() if self._template is not None else []

    def restore_target(self, values: list[float]) -> bool:
        """Restore a valid name-region fingerprint without trusting corrupt caches."""
        try:
            template = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return False
        if template.shape != (2 * 96 * 16,):
            return False
        if not np.all(np.isfinite(template)) or np.any(template < 0):
            return False
        norm = float(np.linalg.norm(template))
        if not np.isfinite(norm) or norm <= 0:
            return False
        self._template = template / norm
        self.reset_observations(preserve_target=True)
        return True

    def reset_observations(self, preserve_target: bool = True) -> None:
        """Begin another chronological window without relearning foreign footage."""
        if not preserve_target:
            self._template = None
        self._learning = []
        self._last_time = None
        self._foreign_start = None
        self._foreign_count = 0
        self._last_foreign = 0.0
        self._foreign_active = False
        self._non_table_start = None
        self._return_table_start = None

    def _observe_return_episode(
        self, timestamp: float, view_type: CameraViewType | None,
    ) -> None:
        """Remember a new table episode, without assigning it a match yet.

        A foreign-table insert can end on a player or arena view, followed by
        the target's break-off close-up with no scoreboard.  The following
        target scoreboard can identify this bounded table episode afterwards.
        A bare camera cut or elapsed time supplies no identity evidence.
        """
        if self._last_time is not None and timestamp - self._last_time > self.max_gap:
            self._non_table_start = None
            self._return_table_start = None
        if not self._foreign_active or view_type is None:
            return
        if view_type in (CameraViewType.MAIN_TABLE, CameraViewType.BALL_CLOSEUP):
            if self._non_table_start is not None:
                if timestamp - self._non_table_start >= self.return_non_table_seconds:
                    self._return_table_start = timestamp
                self._non_table_start = None
            if (self._return_table_start is not None
                    and timestamp - self._return_table_start > self.return_max_seconds):
                self._return_table_start = None
        else:
            self._return_table_start = None
            if self._non_table_start is None:
                self._non_table_start = timestamp

    @staticmethod
    def scoreboard_signature(frame_bgr: np.ndarray) -> np.ndarray | None:
        """Locate paired bright score boxes, then fingerprint outer name areas.

        Supported layouts have two compact neutral/yellow score boxes near the
        lower centre.  Requiring that geometry avoids learning white gloves,
        advertisements or arena signage as a scoreboard.  Other layouts are
        deliberately unknown rather than an automatic rejection.
        """
        h, w = frame_bgr.shape[:2]
        if min(h, w) < 40:
            return None
        scale = min(1.0, 640.0 / w)
        frame = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)))
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        neutral = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 180)
        yellow = (
            (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 180)
        )
        mask = ((neutral | yellow) * 255).astype(np.uint8)
        mask[: int(h * 0.72)] = 0
        mask[:, : int(w * 0.37)] = 0
        mask[:, int(w * 0.63):] = 0
        # Remove horizontal decoration lines without merging separate boxes.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if not (0.018 * w <= bw <= 0.085 * w and 0.012 * h <= bh <= 0.065 * h):
                continue
            if not (0.8 <= bw / bh <= 4.5):
                continue
            if cv2.contourArea(contour) / (bw * bh) < 0.48:
                continue
            boxes.append((x, y, bw, bh))
        pairs = []
        for left in boxes:
            lx, ly, lw, lh = left
            if not 0.39 <= (lx + lw / 2) / w <= 0.48:
                continue
            for right in boxes:
                rx, ry, rw, rh = right
                if not 0.52 <= (rx + rw / 2) / w <= 0.61:
                    continue
                if abs(ly - ry) > max(lh, rh) * 0.25 or abs(lh - rh) > max(lh, rh) * 0.35:
                    continue
                pairs.append((left, right))
        if not pairs:
            return None
        left, right = max(
            pairs, key=lambda pair: pair[0][2] * pair[0][3] + pair[1][2] * pair[1][3]
        )
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        unit = (rx + rw / 2) - (lx + lw / 2)
        y0, y1 = min(ly, ry), max(ly + lh, ry + rh)
        regions = [
            (round(lx + lw / 2 - 2.65 * unit), round(lx - 0.12 * unit)),
            (round(rx + rw + 0.12 * unit), round(rx + rw / 2 + 2.65 * unit)),
        ]
        parts = []
        for x0, x1 in regions:
            x0, x1 = max(0, x0), min(w, x1)
            patch = neutral[y0:y1, x0:x1].astype(np.uint8)
            if patch.size == 0 or np.mean(patch) < 0.018:
                return None
            # Flags or unrelated white blocks cannot dominate the fingerprint.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
            for label in range(1, count):
                _, _, bw, bh, area = stats[label]
                if area > patch.shape[0] * patch.shape[1] * 0.12 or (
                    bw > bh * 2 and bh > patch.shape[0] * 0.75
                ):
                    patch[labels == label] = 0
            part = cv2.resize(patch.astype(np.float32), (96, 16), interpolation=cv2.INTER_AREA)
            part = cv2.GaussianBlur(part, (3, 3), 0.5).ravel()
            norm = float(np.linalg.norm(part))
            if norm < 0.05:
                return None
            parts.append(part / norm)
        return np.concatenate(parts) / np.sqrt(2.0)

    def observe(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        view_type: CameraViewType | None = None,
    ) -> BroadcastContextObservation:
        timestamp = float(timestamp)
        if self._last_time is not None and timestamp < self._last_time:
            raise ValueError("Broadcast context observations must be chronological")
        self._observe_return_episode(timestamp, view_type)
        self._last_time = timestamp
        if view_type is not None and view_type != CameraViewType.MAIN_TABLE:
            return BroadcastContextObservation(timestamp, target_ready=self._template is not None)
        signature = self.scoreboard_signature(frame_bgr)
        if signature is None:
            return BroadcastContextObservation(timestamp, target_ready=self._template is not None)
        if self._template is None:
            if not self._learning:
                self._learning_start = timestamp
            elif float(np.dot(signature, self._learning[-1])) < self.same_threshold:
                self._learning = []
                self._learning_start = timestamp
            self._learning.append(signature)
            if (
                len(self._learning) >= self.learning_samples
                and timestamp - self._learning_start >= self.learning_seconds
            ):
                template = np.mean(self._learning, axis=0)
                self._template = template / np.linalg.norm(template)
                self._learning = []
            return BroadcastContextObservation(
                timestamp, scoreboard_visible=True, target_ready=self._template is not None
            )
        similarity = float(np.clip(np.dot(signature, self._template), 0.0, 1.0))
        strong_foreign = similarity < self.foreign_threshold
        target_interval_start = None
        if strong_foreign:
            # Contradictory scoreboard evidence invalidates the entire proposed
            # return, even before enough samples confirm another foreign span.
            self._return_table_start = None
            if self._foreign_start is None or timestamp - self._last_foreign > self.max_gap:
                self._foreign_start = timestamp
                self._foreign_count = 0
            self._foreign_count += 1
            self._last_foreign = timestamp
        else:
            self._foreign_start = None
            self._foreign_count = 0
            if similarity >= self.same_threshold:
                if self._foreign_active:
                    target_interval_start = self._return_table_start
                self._foreign_active = False
                self._return_table_start = None
                self._non_table_start = None
        confirmed = bool(
            strong_foreign and self._foreign_start is not None
            and self._foreign_count >= self.foreign_samples
            and timestamp - self._foreign_start >= self.foreign_seconds
        )
        if confirmed:
            self._foreign_active = True
        return BroadcastContextObservation(
            timestamp, scoreboard_visible=True,
            identity_known=similarity >= self.same_threshold or strong_foreign,
            identity_similarity=similarity, foreign_match=confirmed,
            foreign_interval_start=self._foreign_start if confirmed else None,
            target_ready=True,
            target_interval_start=target_interval_start,
        )
