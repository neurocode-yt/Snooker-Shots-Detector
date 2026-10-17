"""Native pixel evidence for a scored ball travelling straight at the camera.

Its centre can barely move while its image grows. This is a bounded fallback
for an observed score gap, not an alternative source of ordinary cue strikes.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.video import open_capture
from snooker_ai.utils.work_control import check_pause


@dataclass
class _Blob:
    x: float
    y: float
    area: float
    radius: float
    family: int
    sphere: bool


@dataclass
class _DepthTrack:
    blob: _Blob
    rest: list = field(default_factory=list)
    growth: list = field(default_factory=list)


class DepthLaunchPixels:
    """Follow an initially measured sphere through short codec/blur distortion."""

    def __init__(self, config: Config):
        self.detector = ObjectDetector(config)
        self.tracks: list[_DepthTrack] = []
        self.last: FrameFeatures | None = None

    def _blobs(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, value = cv2.split(hsv)
        spheres = self.detector._closeup_colour_spheres(
            hsv, np.full(hsv.shape[:2], 255, dtype=np.uint8))
        families = [((hue < 15) | (hue >= 140)) & (sat >= 110) & (value >= 55),
                    (hue >= 16) & (hue <= 34) & (sat >= 110) & (value >= 55),
                    (hue >= 95) & (hue <= 135) & (sat >= 110) & (value >= 55)]
        blobs = []
        for family, pixels in enumerate(families):
            contours, _ = cv2.findContours(pixels.astype(np.uint8)*255,
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                moments = cv2.moments(contour)
                if area < 100 or moments['m00'] <= 0:
                    continue
                x, y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
                _, _, width, height = cv2.boundingRect(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)
                if (not .65 <= width/max(height, 1) <= 1.65 or perimeter <= 0
                        or area/max(hull_area, 1) < .85
                        or 4*np.pi*hull_area/perimeter**2 < .85):
                    continue
                (_, _), radius = cv2.minEnclosingCircle(contour)
                sphere = any(np.hypot(p.cx-x, p.cy-y) <= .25*radius
                             and .7 <= p.radius/max(radius, 1) <= 1.3 for p in spheres)
                blobs.append(_Blob(x, y, area, radius, family, sphere))
        return blobs

    def observe(self, frame, context: FrameFeatures):
        previous = self.last
        self.last = context
        valid = (context.observation_fps >= 20 and context.observation_valid
                 and context.match_context_valid and not context.broadcast_replay
                 and not context.table_handling and not context.rack_idle
                 and context.view_type == CameraViewType.BALL_CLOSEUP
                 and context.scene_cut_score < .42 and context.camera_motion_magnitude <= 1.5)
        contiguous = (valid and previous is not None
                      and previous.camera_scene_id == context.camera_scene_id
                      and 0 < context.t-previous.t <= 1.5/context.observation_fps)
        if not valid or not contiguous:
            self.tracks = []
        if not valid:
            return None
        blobs = self._blobs(frame)
        kept, used = [], set()
        for track in self.tracks:
            options = [(i, blob) for i, blob in enumerate(blobs)
                       if i not in used and blob.family == track.blob.family
                       and np.hypot(blob.x-track.blob.x, blob.y-track.blob.y) <= track.blob.radius
                       and .7 <= blob.area/max(track.blob.area, 1) <= 1.8]
            if not options:
                continue
            index, blob = min(options, key=lambda item: np.hypot(
                item[1].x-track.blob.x, item[1].y-track.blob.y))
            used.add(index)
            sample = (context.t, blob.x, blob.y, blob.area, blob.radius)
            if len(track.rest) >= 7 and track.rest[-1][0]-track.rest[0][0] >= .19:
                baseline = np.median(np.asarray(track.rest), axis=0)
                diameter = 2*baseline[4]
                rest_xy = np.asarray(track.rest)[:, 1:3]
                quiet = (np.max(np.linalg.norm(rest_xy-baseline[1:3], axis=1)) <= .1*diameter
                         and max(p[3] for p in track.rest)/min(p[3] for p in track.rest) <= 1.12)
                centered = np.hypot(blob.x-baseline[1], blob.y-baseline[2]) <= .75*diameter
                if quiet and centered and blob.area >= 1.15*baseline[3]:
                    if (track.growth and (blob.area < .98*track.growth[-1][3]
                                         or context.t-track.growth[-1][0] > 1.5/context.observation_fps
                                         or context.t-track.growth[0][0] > .4)):
                        track.growth = []
                    track.growth.append(sample)
                    if (len(track.growth) >= 4 and context.t-track.growth[0][0] >= .095
                            and blob.area >= 1.6*baseline[3]
                            and blob.area >= 1.25*track.growth[0][3]):
                        return {'upper': track.growth[0][0], 'confirmed': context.t,
                                'area_ratio': float(blob.area/baseline[3]),
                                'rest_frames': len(track.rest), 'growth_frames': len(track.growth)}
                else:
                    track.growth = []
                    track.rest.append(sample)
                    track.rest = track.rest[-16:]
            elif blob.sphere:
                track.rest.append(sample)
            else:
                continue
            track.blob = blob
            kept.append(track)
        for index, blob in enumerate(blobs):
            if index not in used and blob.sphere:
                kept.append(_DepthTrack(blob, [(context.t, blob.x, blob.y, blob.area, blob.radius)]))
        self.tracks = kept
        return None


def scoring_depth_contact(source: str | Path, proposal: StrikeCandidate,
                          features: list[FrameFeatures], config: Config):
    if (proposal.evidence.get('score_assisted_proposal', 0) < .5
            or not 1 <= proposal.evidence.get('score_delta', 0) <= 7):
        return None
    rows = sorted((f for f in features if proposal.uncertainty_start <= f.t
                   <= proposal.uncertainty_end), key=lambda f: f.t)
    if len(rows) < 12 or not any(f.view_type == CameraViewType.BALL_CLOSEUP
                               and f.observation_fps >= 20 for f in rows):
        return None
    times = [f.t for f in rows]
    cap = open_capture(source)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, times[0]*1000)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or rows[0].observation_fps
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = max(f.observation_fps for f in rows)
        pixels = DepthLaunchPixels(config)
        while True:
            check_pause()
            ok, frame = cap.read()
            if not ok:
                return None
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not np.isfinite(t):
                return None
            if t > times[-1]+.01:
                return None
            at = bisect_left(times, t)
            context = min(rows[max(0, at-1):at+1], key=lambda f: abs(f.t-t), default=None)
            if (context is None or abs(context.t-t) > .51/max(source_fps, context.observation_fps, 1)
                    or pixels.last is not None and context.t == pixels.last.t):
                continue
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, round(frame.shape[0]*960/frame.shape[1])))
            launch = pixels.observe(frame, context)
            if launch is not None:
                upper = launch['upper']
                return StrikeCandidate(
                    timestamp=max(proposal.uncertainty_start, upper-.15), confidence=.78,
                    uncertainty_start=max(proposal.uncertainty_start, upper-.35), uncertainty_end=upper,
                    camera_view=context.view_type,
                    evidence={**proposal.evidence, 'score_supported_object_launch': 1.,
                              'native_scoring_depth_launch': 1., 'contact_time_upper_bound': 1.,
                              'object_launch_confirmation_time': launch['confirmed'],
                              'depth_launch_motion_start': upper,
                              'depth_launch_scene_id': float(context.camera_scene_id),
                              'depth_launch_area_ratio': launch['area_ratio'],
                              'depth_launch_rest_frames': float(launch['rest_frames']),
                              'depth_launch_growth_frames': float(launch['growth_frames'])})
    finally:
        cap.release()
