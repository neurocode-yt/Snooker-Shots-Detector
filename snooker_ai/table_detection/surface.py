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
    threshold, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (mask > 0) & (hsv[:, :, 2] <= threshold)
    bright = (mask > 0) & (hsv[:, :, 2] > threshold)
    fraction = np.mean(pixels <= threshold)
    if not 0.18 <= fraction <= 0.80:
        return mask
    if float(np.mean(hsv[:, :, 2][bright]) - np.mean(hsv[:, :, 2][dark])) < 65:
        return mask
    work = bright.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=2)
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    dark_y = np.nonzero(dark)[0]
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
            or np.mean(dark_y < y - 5) < 0.80
        ):
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
