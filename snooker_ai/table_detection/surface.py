"""Separate a bright playing surface from darker green arena advertising."""

import cv2
import numpy as np


def playing_surface_mask(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Use a brightness split only for an enclosed table below a dark backdrop.

    Ordinary cloth and clipped local views retain the supplied mask. A strong
    brightness difference alone cannot establish playing-surface geometry.
    """
    pixels = hsv[:, :, 2][mask > 0]
    if len(pixels) < 100:
        return mask
    threshold, bright_pixels = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_count = cv2.countNonZero(bright_pixels)
    dark_count = len(pixels) - bright_count
    fraction = dark_count / len(pixels)
    if not 0.18 <= fraction <= 0.80:
        return mask
    # Otsu has already classified this compact array. Pixel sums are exact
    # integers, so both mean brightnesses match the full-image masked means.
    bright_sum = cv2.norm(pixels, cv2.NORM_L1, mask=bright_pixels)
    dark_sum = cv2.sumElems(pixels)[0] - bright_sum
    if bright_sum / bright_count - dark_sum / dark_count < 65:
        return mask
    bright = cv2.inRange(hsv, (0, 0, int(threshold) + 1), (255, 255, 255))
    work = cv2.bitwise_and(bright, bright, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=2)
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if not 0.09 <= area / (h * w) <= 0.70:
            continue
        hull = cv2.convexHull(contour)
        x, y, bw, bh = cv2.boundingRect(hull)
        if (
            x <= w * 0.01
            or x + bw >= w * 0.99
            or y <= h * 0.01
            or y + bh >= h * 0.99
            or bw < w * 0.30
            or bh < h * 0.22
            or not 0.65 <= bw / bh <= 3.2
        ):
            continue
        # Only the number above the candidate matters. Avoid allocating a
        # pair of full-image coordinate arrays for the dark backdrop.
        rows = slice(0, max(0, y - 5))
        if np.count_nonzero((mask[rows] > 0) & (bright[rows] == 0)) / dark_count < 0.80:
            continue
        perimeter = cv2.arcLength(hull, True)
        quad = hull
        for epsilon in (0.015, 0.025, 0.030, 0.040):
            quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(quad) == 4:
                break
        if len(quad) != 4 or area / max(1, cv2.contourArea(hull)) < 0.70:
            continue
        # Keep measured cloth pixels, including holes for balls and hands;
        # downstream localisation owns filling and temporal mask smoothing.
        region = np.zeros_like(mask)
        cv2.drawContours(region, [hull], -1, 255, -1)
        return cv2.bitwise_and(work, region)
    return mask
