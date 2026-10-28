"""Bound a scored stroke when the bridge hides the ordinary white-ball track."""

from bisect import bisect_left

import cv2
import numpy as np

from snooker_ai.types import CameraViewType, StrikeCandidate
from snooker_ai.utils.video import open_capture
from snooker_ai.utils.work_control import check_pause


def _pale(frame, circle):
    x, y, radius = circle
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.65 * radius) ** 2
    pixels = frame[inside]
    return float(np.mean((pixels[:, 1] < 140) & (pixels[:, 2] > 150))) if len(pixels) else 0.0


def _rest_circle(images, diameter, position):
    """Require a measured circular rim and seven quiet native appearances."""
    tracks = []
    for t, hsv in images:
        check_pause()
        gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        kept = []
        for circle, samples, previous in tracks:
            x, y, radius = circle
            yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
            inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.65 * radius) ** 2
            patch = gray[inside]
            quiet = (
                len(patch) == len(previous)
                and np.mean(np.abs(patch.astype(float) - previous)) <= 6
                and np.mean(np.abs(patch.astype(float) - previous) > 20) <= 0.1
                and _pale(hsv, circle) >= 0.5
            )
            if quiet and t - samples[-1] <= 0.065:
                kept.append((circle, (samples + [t])[-16:], patch))
            elif len(samples) >= 7 and t - samples[-1] <= 0.32:
                kept.append((circle, samples, previous))
        x0, y0 = np.maximum(0, np.floor(position - 3 * diameter)).astype(int)
        x1, y1 = np.minimum(
            [gray.shape[1], gray.shape[0]], np.ceil(position + 3 * diameter)
        ).astype(int)
        if x1 <= x0 or y1 <= y0:
            tracks = kept
            continue
        circles = cv2.HoughCircles(
            gray[y0:y1, x0:x1],
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=0.3 * diameter,
            param1=80,
            param2=18,
            minRadius=max(6, round(0.3 * diameter)),
            maxRadius=round(0.7 * diameter),
        )
        for x, y, radius in circles[0] if circles is not None else []:
            circle = (float(x + x0), float(y + y0), float(radius))
            if _pale(hsv, circle) < 0.5 or any(
                np.linalg.norm(np.asarray(circle[:2]) - np.asarray(tr[0][:2])) < 0.15 * diameter
                for tr in kept
            ):
                continue
            yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
            inside = (xx - circle[0]) ** 2 + (yy - circle[1]) ** 2 <= (0.65 * circle[2]) ** 2
            kept.append((circle, [t], gray[inside]))
        tracks = kept
    choices = []
    for circle, samples, appearance in tracks:
        if len(samples) < 7 or samples[-1] - samples[0] < 0.24 - 1e-6:
            continue
        distance = np.linalg.norm(np.asarray(circle[:2]) - position) / diameter
        if 0.5 <= distance <= 3:
            choices.append((distance, samples[-1], circle, len(samples), appearance))
    return sorted(choices, key=lambda item: item[0])


def scoring_white_contact(source, proposal, features, config):
    """Scores request a check; rest, disappearance and measured travel prove it.

    Contact remains bounded by the last stationary source image and the first
    rolling observation. This does not substitute a score time for cue contact.
    """
    if (
        proposal.evidence.get("score_assisted_proposal", 0) < 0.5
        or not 1 <= proposal.evidence.get("score_delta", 0) <= 7
    ):
        return None
    rows = sorted(
        (f for f in features if proposal.uncertainty_start <= f.t <= proposal.uncertainty_end),
        key=lambda f: f.t,
    )
    times = [f.t for f in rows]

    def valid(f):
        return (
            f.observation_fps >= 20
            and f.observation_valid
            and f.match_context_valid
            and not f.broadcast_replay
            and not f.table_handling
            and not f.rack_idle
            and f.view_type == CameraViewType.BALL_CLOSEUP
            and f.scene_cut_score < 0.42
            and f.camera_motion_magnitude <= 1.5
        )

    def measured(f):
        return (
            f.cue_ball_detected
            and f.cue_ball_quality
            and f.cue_ball_track_confidence >= 0.7
            and f.cue_ball_x is not None
            and f.cue_ball_y is not None
            and np.isfinite([f.cue_ball_x, f.cue_ball_y, f.ball_diameter_px]).all()
        )

    cap = None
    try:
        for i, onset in enumerate(rows):
            if not measured(onset) or i == 0 or measured(rows[i - 1]):
                continue
            before = rows[bisect_left(times, onset.t - 0.8) : i]
            after = rows[i : bisect_left(times, onset.t + 0.6)]
            native = before + after
            moving = [f for f in after if measured(f)]
            if (
                len(before) < 12
                or len(moving) < 6
                or moving[-1].t - onset.t < 0.2
                or any(not valid(f) or f.camera_scene_id != onset.camera_scene_id for f in native)
                or any(
                    not 0 < b.t - a.t <= 1.6 / min(a.observation_fps, b.observation_fps)
                    for a, b in zip(native, native[1:])
                )
                or max((f.cue_contact_score for f in native if f.cue_tip_visible), default=0) < 0.6
            ):
                continue
            diameter = float(np.median([f.ball_diameter_px for f in moving]))
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in moving])
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            travel = float(np.linalg.norm(xy[-1] - xy[0])) / max(diameter, 1)
            direction = travel * diameter / max(float(sum(steps)), 1e-6)
            if (
                not 20 <= diameter <= 80
                or travel < 2
                or direction < 0.94
                or max(steps) / diameter > 2
            ):
                continue
            if cap is None:
                cap = open_capture(
                    source, prefer_hwaccel=bool(config.get("analysis.hwaccel_decode", True))
                )
                if not cap.isOpened():
                    return None
            cap.set(cv2.CAP_PROP_POS_MSEC, before[0].t * 1000)
            images, rolling = [], None
            while True:
                check_pause()
                ok, image = cap.read()
                if not ok:
                    break
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not np.isfinite(t) or t > onset.t + 0.02:
                    break
                if image.shape[1] > 960:
                    image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                if abs(t - onset.t) <= 0.015:
                    rolling = hsv
                elif before[0].t - 0.015 <= t < onset.t - 0.015:
                    images.append((t, hsv))
            if rolling is None or len(images) < 12:
                continue
            gray = cv2.cvtColor(cv2.cvtColor(rolling, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            for _, last_rest, circle, count, appearance in _rest_circle(images, diameter, xy[0]):
                pale_after = _pale(rolling, circle)
                x, y, radius = circle
                yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
                inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.65 * radius) ** 2
                difference = np.abs(gray[inside].astype(float) - appearance)
                changed = float(np.mean(difference > 20))
                if (
                    not 0.04 <= onset.t - last_rest <= 0.32 + 1e-6
                    or np.mean(difference) < 12
                    or changed < 0.35
                ):
                    continue
                return StrikeCandidate(
                    timestamp=onset.t,
                    confidence=0.78,
                    uncertainty_start=last_rest,
                    uncertainty_end=onset.t,
                    camera_view=onset.view_type,
                    evidence={
                        **proposal.evidence,
                        "native_scoring_white_reacquisition": 1.0,
                        "score_supported_object_launch": 1.0,
                        "contact_time_upper_bound": 1.0,
                        "white_rest_samples": float(count),
                        "white_rolling_samples": float(len(moving)),
                        "white_travel_diameters": travel,
                        "white_direction_consistency": direction,
                        "white_pale_after": pale_after,
                        "white_rest_changed_fraction": changed,
                        "white_last_rest": last_rest,
                        "object_launch_confirmation_time": moving[-1].t,
                    },
                )
    finally:
        if cap is not None:
            cap.release()
    return None
