"""Recover live contacts hidden by a brief change of camera."""

from bisect import bisect_left, bisect_right
from contextlib import ExitStack

import cv2
import numpy as np

from snooker_ai.types import CameraViewType, StrikeCandidate
from snooker_ai.utils.video import open_capture
from snooker_ai.utils.work_control import check_pause

_MAIN = CameraViewType.MAIN_TABLE
_CLOSEUPS = {CameraViewType.BALL_CLOSEUP, CameraViewType.PLAYER_CLOSEUP}


def _live(frame):
    return (
        frame.match_context_valid
        and not frame.broadcast_replay
        and not frame.table_handling
        and not frame.rack_idle
    )


def _measured(frame):
    return (
        frame.observation_valid
        and frame.observation_fps >= 20
        and _live(frame)
        and frame.table_full_view
        and frame.view_type is _MAIN
        and frame.camera_motion_magnitude <= 1.5
        and frame.cue_ball_quality
        and frame.cue_ball_track_confidence >= 0.75
        and frame.cue_ball_x is not None
        and frame.cue_ball_y is not None
    )


def _track(frames):
    if not (frames and all(_measured(frame) for frame in frames)):
        return None
    if any(
        a.camera_scene_id != b.camera_scene_id or not 0 < b.t - a.t <= 0.069
        for a, b in zip(frames, frames[1:])
    ):
        return None
    return np.array([(frame.cue_ball_x, frame.cue_ball_y) for frame in frames])


def _layout_matches(before, after):
    # Compare the fixed overhead view and require three independently measured
    # coloured anchors to survive the insert. The white is allowed to move.
    if len(before.ball_layout_signature) != 14 or len(after.ball_layout_signature) != 14:
        return False
    prior = np.array(before.ball_layout_signature[2:]).reshape(-1, 2)
    current = np.array(after.ball_layout_signature[2:]).reshape(-1, 2)
    visible = (prior >= 0).all(axis=1) & (current >= 0).all(axis=1)
    return np.count_nonzero(np.linalg.norm(prior[visible] - current[visible], axis=1) <= 0.025) >= 3


def recover_cut_contacts(features, shots, source=None):
    """Join a measured rest to a measured roll across a short live close-up."""
    with ExitStack() as stack:
        capture = None

        def check_departure(lo, hi):
            nonlocal capture
            if source is None:
                return None
            if capture is None:
                capture = open_capture(source)
                stack.callback(capture.release)
            fps = capture.get(cv2.CAP_PROP_FPS)
            return _departure(capture, lo, hi, fps) if capture.isOpened() and fps >= 20 else None

        return _recover(features, shots, check_departure)


def _recover(features, shots, check_departure):
    seen = sorted(shot.cue_strike for shot in shots if not shot.possible_replay)
    times = [frame.t for frame in features]
    last_main = None
    recovered = []
    for i, frame in enumerate(features):
        if frame.view_type is not _MAIN:
            continue
        previous_main = last_main
        last_main = i
        if not i or previous_main is None or previous_main == i - 1:
            continue
        old = features[previous_main]
        if (
            frame.camera_scene_id == old.camera_scene_id
            or not 0.30 <= frame.t - old.t <= 6
            or bisect_left(seen, frame.t + 1.4) > bisect_right(seen, old.t - 0.35)
        ):
            continue
        insert = features[previous_main + 1 : i]
        if (
            not all(_live(sample) and sample.view_type in _CLOSEUPS for sample in insert)
            or insert[-1].t - insert[0].t < 0.24
        ):
            continue
        origin = None
        for end in range(previous_main + 1, max(5, previous_main - 5), -1):
            rest = _track(features[end - 6 : end])
            if rest is not None and np.max(np.ptp(rest, axis=0)) <= 0.14 * old.ball_diameter_px:
                origin = np.median(rest, axis=0)
                break
        if origin is None:
            continue

        departure_checked = False
        has_address = sum(f.cue_tip_visible for f in insert[-12:]) >= 3
        before = features[bisect_left(times, old.t - 0.40) : previous_main + 1]
        for start in range(i, bisect_right(times, frame.t + 0.16)):
            post = features[start : start + 6]
            path = _track(post)
            if path is None or len(path) != 6 or post[0].camera_scene_id != frame.camera_scene_id:
                continue
            start_frame = post[0]
            diameter = start_frame.cue_ball_image_diameter_px or start_frame.ball_diameter_px
            if diameter <= 0:
                continue
            vector = path[-1] - path[0]
            travel = float(np.linalg.norm(vector))
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            if (
                travel < 0.65 * diameter
                or min(steps) < 0.07 * diameter
                or travel < 0.94 * sum(steps)
            ):
                continue
            direction = vector / travel
            offset = path[0] - origin
            along = float(offset @ direction)
            distance = float(np.linalg.norm(offset - along * direction))
            elapsed = max(0.0, along) / travel * (post[-1].t - start_frame.t)
            after = features[start : bisect_right(times, start_frame.t + 0.30)]
            same_layout = any(
                _layout_matches(a, b)
                for a in before
                if _live(a) and a.camera_scene_id == old.camera_scene_id
                for b in after
                if _live(b) and b.camera_scene_id == frame.camera_scene_id
            )
            projected = (
                travel >= 1.2 * diameter
                and min(steps) >= 0.13 * diameter
                and travel >= 0.97 * sum(steps)
                and max(steps) <= 3 * min(steps)
                and distance <= 0.42 * diameter
                and along >= -0.24 * diameter
                and elapsed <= 0.45
                and (same_layout or has_address)
            )
            departure = None
            if not projected:
                # After an object-ball collision the returned trajectory
                # points elsewhere. Require visible departure on the cloth in
                # the close-up and verify the return has the same table layout.
                if departure_checked or not same_layout or np.linalg.norm(offset) < 0.8 * diameter:
                    continue
                departure_checked = True
                departure = check_departure(max(insert[0].t + 0.32, frame.t - 1.35), insert[-1].t)
                if departure is None:
                    continue
            upper = departure if departure is not None else start_frame.t
            timestamp = departure if departure is not None else max(insert[0].t, upper - elapsed)
            contact = StrikeCandidate(
                timestamp=timestamp,
                confidence=0.81,
                camera_view=frame.view_type,
                uncertainty_start=max(old.t, timestamp - 0.16),
                uncertainty_end=upper,
                evidence={
                    "camera_return_launch": 1.0,
                    "camera_return_rest_samples": 6.0,
                    "camera_return_post_samples": 6.0,
                    "camera_return_travel_diameters": travel / diameter,
                    "camera_return_projection_error_diameters": distance / diameter,
                    "camera_return_visible_departure": float(departure is not None),
                    "contact_time_upper_bound": 1.0,
                    "cue_ball_motion_confirmed": 1.0,
                    "refined_ball_motion_start": upper,
                },
            )
            recovered.append(contact)
            seen.insert(bisect_right(seen, timestamp), timestamp)
            break
    return recovered


def cloth_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (
        cv2.inRange(hsv, (35, 65, 32), (92, 255, 255)),
        cv2.inRange(hsv, (0, 0, 132), (179, 125, 255)),
    )


def departed_ball(history, current_cloth):
    """Find a stationary, ball-sized pale patch that the stroke cleared to cloth."""
    first, middle, last = history
    cloth, pale = last
    if cv2.countNonZero(cv2.bitwise_xor(cloth, current_cloth)) > 0.055 * cloth.size:
        return None
    stable = cv2.bitwise_and(first[1], middle[1])
    stable &= pale
    contours, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (5 <= w <= 88 and 5 <= h <= 56 and 0.35 <= w / h <= 5):
            continue
        size = min(w, h)
        if x < size or y < size or x + w + size >= cloth.shape[1] or y + h + size >= cloth.shape[0]:
            continue
        region = np.s_[y - size : y + h + size, x - size : x + w + size]
        if cv2.countNonZero(cloth[region]) < 0.37 * cloth[region].size:
            continue
        # A cue touching the ball can extend the outline. Locate its disk core.
        core = cv2.copyMakeBorder(stable[y : y + h, x : x + w], 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        dist = cv2.distanceTransform(core, cv2.DIST_L2, 3)
        _, radius, _, (u, v) = cv2.minMaxLoc(dist)
        if not 2.4 <= radius <= 24 or 2 * radius < 0.57 * size:
            continue
        cx, cy = x + u - 1, y + v - 1
        rad = max(2, round(0.60 * radius))
        center = np.s_[cy - rad : cy + rad + 1, cx - rad : cx + rad + 1]
        cleared = cv2.countNonZero(current_cloth[center]) / current_cloth[center].size
        if cleared >= 0.56:
            return cx, cy, round(2 * radius, 1)
    return None


def _departure(capture, lower, upper, fps):
    # Decode only the brief insert preceding a measured roll. The first frame
    # used for each pale mask predates the departure by about a quarter-second.
    start = max(0.0, lower - 0.36)
    if not capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000):
        return None
    recent = []
    half = max(1, round(0.12 * fps))
    budget = int((upper - start + 0.12) * fps) + 1
    for _ in range(budget):
        check_pause()
        ok, image = capture.read()
        if not ok:
            break
        stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not np.isfinite(stamp) or stamp > upper + 1e-6:
            break
        if stamp < start - 0.06:
            continue
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        recent.append(cloth_masks(image))
        if stamp < lower or len(recent) <= 2 * half:
            continue
        history = [recent[-1 - 2 * half], recent[-1 - half], recent[-2]]
        if departed_ball(history, recent[-1][0]) is not None:
            return stamp
    return None
