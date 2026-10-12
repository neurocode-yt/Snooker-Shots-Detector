"""Classical digit-template OCR for supported broadcast score overlays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ScoreReading:
    t: float
    left: int
    right: int
    active_player: int | None
    confidence: float
    layout: str


def _glyph(mask: np.ndarray) -> tuple[np.ndarray, float, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if h < 4 or w < 1:
        return None
    cropped = mask[y : y + h, x : x + w]
    _, hierarchy = cv2.findContours(cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(item[3] >= 0 for item in hierarchy[0]) if hierarchy is not None else 0
    resized = cv2.resize(cropped.astype(np.float32) / 255, (24, 36), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (3, 3), 0.6), w / h, holes


@lru_cache(maxsize=1)
def _digit_templates():
    asset = Path(__file__).with_name("assets") / "score_digits.npz"
    if asset.is_file():
        with np.load(asset, allow_pickle=False) as data:
            return [
                (int(digit), (image.copy(), float(aspect), int(holes)))
                for digit, image, aspect, holes in zip(
                    data["digits"], data["images"], data["aspects"], data["holes"]
                )
            ]
    templates = []
    for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN):
        for scale in (0.45, 0.65, 0.85, 1.0, 1.3):
            for thickness in (1, 2):
                for digit in range(10):
                    mask = np.zeros((64, 48), np.uint8)
                    cv2.putText(mask, str(digit), (5, 46), font, scale, 255, thickness, cv2.LINE_AA)
                    templates.append((digit, _glyph((mask > 100).astype(np.uint8) * 255)))
    # Fallback for development checkouts without the packaged glyph samples.
    try:
        from PIL import Image, ImageDraw, ImageFont

        fonts = Path("C:/Windows/Fonts")
        for name in (
            "arial.ttf",
            "arialbd.ttf",
            "segoeui.ttf",
            "segoeuib.ttf",
            "calibri.ttf",
            "calibrib.ttf",
        ):
            if not (fonts / name).is_file():
                continue
            for size in (18, 24, 32):
                font = ImageFont.truetype(str(fonts / name), size)
                for digit in range(10):
                    canvas = Image.new("L", (64, 64))
                    ImageDraw.Draw(canvas).text((5, 5), str(digit), font=font, fill=255)
                    templates.append(
                        (digit, _glyph((np.asarray(canvas) > 100).astype(np.uint8) * 255))
                    )
    except ImportError:
        pass
    return templates


@lru_cache(maxsize=1)
def _template_bank():
    templates = [(digit, value) for digit, value in _digit_templates() if value is not None]
    return (
        np.asarray([digit for digit, _ in templates]),
        np.stack([value[0] for _, value in templates]),
        np.asarray([value[1] for _, value in templates]),
        np.asarray([value[2] for _, value in templates]),
    )


@lru_cache(maxsize=2048)
def _match_digit(shape: tuple, pixels: bytes):
    glyph = _glyph(np.frombuffer(pixels, dtype=np.uint8).reshape(shape))
    if glyph is None:
        return None
    image, aspect, holes = glyph
    digits, targets, aspects, template_holes = _template_bank()
    sums = targets.sum(axis=(1, 2)) + image.sum()
    values = (
        2 * np.minimum(targets, image).sum(axis=(1, 2)) / np.maximum(sums, 0.001)
        - 0.12 * np.abs(aspects - aspect)
        - 0.12 * np.abs(template_holes - holes)
    )
    scores = np.zeros(10)
    np.maximum.at(scores, digits, values)
    order = np.argsort(scores)
    best = int(order[-1])
    return best, float(scores[best]), float(scores[best] - scores[order[-2]])


def read_digits(mask: np.ndarray, *, min_confidence: float = 0.78) -> tuple[int, float] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if (
            h < max(5, 0.18 * mask.shape[0])
            or area < 4
            or w > 1.1 * h
            or y + h < 0.62 * mask.shape[0]
        ):
            continue
        components.append((x, label, (x, y, w, h)))
    components.sort()
    if not 1 <= len(components) <= 3:
        return None
    digits, confidences = [], []
    for _, label, (x, y, w, h) in components:
        pixels = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
        match = _match_digit(pixels.shape, pixels.tobytes())
        if match is None:
            return None
        best, confidence, margin = match
        if confidence < min_confidence or margin < 0.025:
            return None
        digits.append(str(best))
        confidences.append(confidence)
    value = int("".join(digits))
    return value, min(confidences)


class ScoreboardReader:
    def __init__(self, min_confidence: float = 0.78):
        self.min_confidence = min_confidence

    @staticmethod
    def _pair(frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = (
            ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 170))
            | (
                (hsv[:, :, 0] >= 18)
                & (hsv[:, :, 0] <= 40)
                & (hsv[:, :, 1] > 100)
                & (hsv[:, :, 2] > 170)
            )
        ).astype(np.uint8) * 255
        bright[: int(h * 0.72)] = 0
        bright[:, : int(w * 0.37)] = 0
        bright[:, int(w * 0.63) :] = 0
        horizontal = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN, np.ones((1, max(3, round(w * 0.12))), np.uint8)
        )
        bright = cv2.subtract(bright, horizontal)
        # Hands and shirts above the overlay can touch a score box and make
        # its connected component too tall. Anchor both boxes to the common
        # bright score row before extracting their contours.
        left_rows = np.count_nonzero(bright[:, round(w * 0.39) : round(w * 0.48)], axis=1)
        right_rows = np.count_nonzero(bright[:, round(w * 0.52) : round(w * 0.61)], axis=1)
        common = ((left_rows >= w * 0.025) & (right_rows >= w * 0.025)).astype(np.uint8)
        common = cv2.morphologyEx(
            common[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)
        ).ravel()
        changes = np.diff(np.r_[0, common, 0].astype(np.int8))
        bands = [
            (start, end)
            for start, end in zip(np.where(changes == 1)[0], np.where(changes == -1)[0])
            if 0.012 * h <= end - start <= 0.065 * h
        ]
        # Captions below the scores (for example Average Frame Time) can
        # contain bright text in both columns. Choose a band only after its
        # filled rectangular boxes have been verified.
        candidates = list(reversed(bands)) if bands else [(0, h)]
        for y0, y1 in candidates:
            candidate = bright.copy()
            candidate[:y0] = 0
            candidate[y1:] = 0
            candidate[:, round(w * 0.48) : round(w * 0.52)] = 0
            pair = ScoreboardReader._find_box_pair(candidate, h, w)
            if pair is not None:
                return pair, hsv, candidate
        return None

    @staticmethod
    def _find_box_pair(bright: np.ndarray, h: int, w: int):
        solid = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if (
                0.018 * w <= bw <= 0.085 * w
                and 0.012 * h <= bh <= 0.065 * h
                and 0.8 <= bw / bh <= 4.5
                and cv2.contourArea(contour) / (bw * bh) >= 0.48
            ):
                boxes.append((x, y, bw, bh))
        pairs = [
            (left, right)
            for left in boxes
            for right in boxes
            if 0.39 <= (left[0] + left[2] / 2) / w <= 0.48
            and 0.52 <= (right[0] + right[2] / 2) / w <= 0.61
            and abs(left[1] - right[1]) <= max(left[3], right[3]) * 0.25
            and abs(left[3] - right[3]) <= max(left[3], right[3]) * 0.35
        ]
        if not pairs:
            return None
        return max(pairs, key=lambda pair: sum(box[2] * box[3] for box in pair))

    def read(self, frame: np.ndarray, t: float) -> ScoreReading | None:
        found = self._pair(frame)
        if found is not None:
            pair, hsv, bright = found
            readings = []
            for x, y, w, h in pair:
                crop = frame[y : y + h, x : x + w]
                cloth = bright[y : y + h, x : x + w]
                # Fill text holes so its black pixels can be isolated from the
                # black background outside slanted score-box edges.
                contours, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                interior = np.zeros_like(cloth)
                cv2.drawContours(interior, contours, -1, 255, -1)
                interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                digits = ((gray < 115) & (interior > 0)).astype(np.uint8) * 255
                reading = read_digits(digits, min_confidence=self.min_confidence)
                if reading is None:
                    return None
                readings.append(reading)
            indicators = []
            for side, (x, y, w, h) in enumerate(pair):
                indicator_span = min(round(h * 1.2), round(frame.shape[1] * 0.023))
                x0, x1 = (
                    (max(0, x - indicator_span), x - 2)
                    if side == 0
                    else (x + w + 2, min(frame.shape[1], x + w + indicator_span))
                )
                area = hsv[y : y + h, x0:x1]
                colored = (
                    (area[:, :, 1] > 130)
                    & (area[:, :, 2] > 120)
                    & (
                        ((area[:, :, 0] >= 95) & (area[:, :, 0] <= 135))
                        | ((area[:, :, 0] >= 18) & (area[:, :, 0] <= 40))
                    )
                )
                indicators.append(float(np.mean(colored)) if colored.size else 0.0)
            active = (
                int(np.argmax(indicators))
                if max(indicators) > 0.025 and abs(indicators[0] - indicators[1]) > 0.025
                else None
            )
            return ScoreReading(
                t,
                readings[0][0],
                readings[1][0],
                active,
                min(item[1] for item in readings),
                "paired_boxes",
            )
        return self._read_outer_scores(frame, t)

    def _read_outer_scores(self, frame: np.ndarray, t: float) -> ScoreReading | None:
        """Older Eurosport layout: frame count in yellow, points outside it."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = (
            (hsv[:, :, 0] >= 18)
            & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 100)
            & (hsv[:, :, 2] > 170)
        ).astype(np.uint8) * 255
        yellow[: int(h * 0.72)] = 0
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if not (
                0.055 * w <= bw <= 0.10 * w
                and 0.016 * h <= bh <= 0.065 * h
                and abs((x + bw / 2) / w - 0.5) < 0.035
                and 2 <= bw / bh <= 4
            ):
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            readings = []
            for x0, x1 in (
                (max(0, x - round(bh * 1.6)), x),
                (x + bw, min(w, x + bw + round(bh * 1.6))),
            ):
                crop = gray[y : y + bh, x0:x1]
                reading = read_digits(
                    (crop > 180).astype(np.uint8) * 255, min_confidence=self.min_confidence
                )
                if reading is None:
                    break
                readings.append(reading)
            if len(readings) != 2:
                continue
            # The white turn marker is at the outer edge of each name panel.
            indicators = []
            for center in (x - bh, x + bw + bh * 12.1):
                x0, x1 = max(0, round(center - bh * 0.5)), min(w, round(center + bh * 0.5))
                crop = gray[y : y + bh, x0:x1]
                contours, _ = cv2.findContours(
                    (crop > 200).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                circular = False
                for item in contours:
                    _, _, cw, ch = cv2.boundingRect(item)
                    circular |= (
                        0.7 <= cw / max(ch, 1) <= 1.3
                        and 0.35 * bh <= ch <= 0.75 * bh
                        and cv2.contourArea(item) / max(cw * ch, 1) > 0.55
                    )
                indicators.append(circular)
            active = indicators.index(True) if sum(indicators) == 1 else None
            return ScoreReading(
                t,
                readings[0][0],
                readings[1][0],
                active,
                min(item[1] for item in readings),
                "outer_scores",
            )
        return None
