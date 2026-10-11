"""Source evidence for short countdown montages following a broadcast title card."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.scene_detection.broadcast_context import BroadcastContextGuard
from snooker_ai.types import FrameFeatures, StrikeCandidate
from snooker_ai.utils.video import open_capture


@dataclass(frozen=True)
class TimerShape:
    box: tuple[int, int, int, int]
    prefix: np.ndarray
    seconds: np.ndarray


def _zero(mask: np.ndarray) -> bool:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    holes = [cv2.boundingRect(c) for c, h in zip(contours, hierarchy[0]) if h[3] >= 0]
    if len(holes) != 1:
        return False
    x, y, w, h = holes[0]
    return 0.35 <= (y + h / 2) / mask.shape[0] <= 0.65 and w * h >= 0.12 * mask.size


def timer_shape(frame: np.ndarray) -> TimerShape | None:
    """Require four regular digits and two colon dots in the upper-left HUD.

    Only 00/01-minute prefixes qualify. Longer live frame clocks supply no
    promotion evidence. This shape check does not read arbitrary screen text.
    """
    small = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(small[:44, :160], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 185)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0]
    digits = sorted(b for b in boxes if 11 <= b[3] <= 26 and 0.25 <= b[2] / b[3] <= 0.85)
    dots = [b for b in boxes if 2 <= b[2] <= 5 and 2 <= b[3] <= 5]
    for first in dots:
        for second in dots:
            if abs(first[0] - second[0]) > 1 or not 6 <= second[1] - first[1] <= 16:
                continue
            left = [b for b in digits if b[0] + b[2] < first[0]]
            right = [b for b in digits if b[0] > first[0] + first[2]]
            if len(left) < 2 or len(right) < 2:
                continue
            group = left[-2:] + right[:2]
            heights = [b[3] for b in group]
            tops = [b[1] for b in group]
            if max(heights) - min(heights) > 2 or max(tops) - min(tops) > 2:
                continue
            height = float(np.median(heights))
            gaps = [b[0] - (a[0] + a[2]) for a, b in zip(group, group[1:])]
            if any(not 1 <= gap <= 0.8 * height for gap in gaps):
                continue
            if not group[0][1] < first[1] < second[1] < group[0][1] + group[0][3]:
                continue
            glyphs = [mask[y : y + h, x : x + w] for x, y, w, h in group]
            if not _zero(glyphs[0]) or not (group[1][2] / group[1][3] <= 0.5 or _zero(glyphs[1])):
                continue
            normalized = [
                cv2.resize(g, (16, 24), interpolation=cv2.INTER_NEAREST).astype(bool)
                for g in glyphs
            ]
            x0, y0 = group[0][:2]
            end = group[-1]
            return TimerShape(
                (x0, y0, end[0] + end[2] - x0, max(tops) + max(heights) - y0),
                np.concatenate(normalized[:2], axis=1),
                np.concatenate(normalized[2:], axis=1),
            )
    return None


def title_card(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 225)
    ink = (hsv[:, :, 1] > 65) & (hsv[:, :, 2] < 250)
    if np.mean(white) < 0.85 or not 0.003 <= np.mean(ink) <= 0.12:
        return False
    return np.sum(ink[90:270, 160:480]) / max(np.sum(ink), 1) >= 0.70


def _has_live_scorebar(frame: np.ndarray) -> bool:
    """Recognize the lower ITV match strip even when score-box parsing fails."""
    small = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    band = hsv[324:356, 128:512]
    dark = band[:, :, 2] < 95
    if np.mean(dark) < 0.30:
        return False
    for left, right in ((128, 250), (250, 390), (390, 512)):
        region = hsv[324:356, left:right]
        if np.mean(region[:, :, 2] < 95) < 0.18:
            return False
        white_text = (region[:, :, 1] < 100) & (region[:, :, 2] > 165)
        yellow_text = (
            (region[:, :, 0] >= 20)
            & (region[:, :, 0] <= 40)
            & (region[:, :, 1] > 85)
            & (region[:, :, 2] > 165)
        )
        if np.mean(white_text | yellow_text) < 0.02:
            return False
    return True


def _montage(rows: list[tuple[float, np.ndarray]]) -> bool:
    # Any recognisable live scoreboard vetoes this context rejection. The
    # legacy parser misses the compact ITV strip used in this source, so also
    # recognize that lower band with a conservative dark-bar/text signature.
    if any(
        BroadcastContextGuard.scoreboard_signature(frame) is not None or _has_live_scorebar(frame)
        for _, frame in rows
    ):
        return False
    timers = [(t, timer_shape(frame)) for t, frame in rows]
    timers = [(t, shape) for t, shape in timers if shape is not None]
    if len(timers) < 3 or timers[-1][0] - timers[0][0] < 1.0:
        return False
    anchor = timers[0][1]
    stable = [
        (t, shape)
        for t, shape in timers
        if max(abs(a - b) for a, b in zip(shape.box, anchor.box)) <= 2
        and np.mean(shape.prefix == anchor.prefix) >= 0.9
    ]
    return (
        len(stable) >= 3
        and stable[-1][0] - stable[0][0] >= 1.0
        and any(np.mean(shape.seconds == stable[0][1].seconds) < 0.90 for _, shape in stable[1:])
    )


class PromotionDetector:
    def detect(
        self, source: str | Path, candidates: list[StrikeCandidate], features: list[FrameFeatures]
    ) -> set[float]:
        ordered = sorted(features, key=lambda f: f.t)
        times = [f.t for f in ordered]
        eligible = []
        for candidate in candidates:
            if candidate.possible_replay:
                continue
            near = ordered[
                bisect_left(times, candidate.timestamp - 2.5) : bisect_right(
                    times, candidate.timestamp + 2.5
                )
            ]
            # Ordinary table continuity, including a live frame-clock HUD,
            # supplies no montage evidence. At least three actual scene
            # changes in this bounded interval must be measured first.
            if (
                len(near) < 4
                or any(f.table_full_view for f in near)
                or sum(a.camera_scene_id != b.camera_scene_id for a, b in zip(near, near[1:])) < 3
            ):
                continue
            eligible.append(candidate)
        if not eligible:
            return set()
        capture = open_capture(source)
        rejected = set()
        try:
            if not capture.isOpened():
                return set()
            for candidate in eligible:

                def observe(start, end, step):
                    result = []
                    previous = None
                    for requested in np.arange(max(0.0, start), end + 1e-6, step):
                        capture.set(cv2.CAP_PROP_POS_MSEC, float(requested) * 1000)
                        ok, frame = capture.read()
                        if not ok:
                            break
                        t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                        if not np.isfinite(t) or previous is not None and t <= previous:
                            break
                        previous = t
                        result.append((t, frame))
                    return result

                rows = observe(candidate.timestamp - 2.5, candidate.timestamp + 2.5, 0.5)
                if not _montage(rows):
                    continue
                lead = observe(candidate.timestamp - 12.0, candidate.timestamp - 3.0, 0.5)
                cards = [t for t, frame in lead if title_card(frame)]
                if any(0.45 <= b - a <= 0.55 for a, b in zip(cards, cards[1:])):
                    rejected.add(candidate.timestamp)
        finally:
            capture.release()
        return rejected
