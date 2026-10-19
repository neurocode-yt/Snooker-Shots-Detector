"""Source-pixel proof of a shaded red departing beside a stationary white."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.types import CameraViewType, FrameFeatures


@dataclass
class _RedRegion:
    x: float
    y: float
    area: float
    radius: float


@dataclass
class _RedTrack:
    region: _RedRegion
    diameter: float
    rest: list = field(default_factory=list)
    moving: list = field(default_factory=list)


class RedLaunchPixels:
    """Require a measured circular rim, quiet address and continuous red travel.

    Colour can describe only the lit upper cap of a dark red. Its full grey
    image edge establishes sphere scale once; subsequent colour-cap motion
    must belong to that same nearby region at native cadence.
    """

    def __init__(self, config: Config):
        self.detector = ObjectDetector(config)
        self.tracks: list[_RedTrack] = []
        self.last: FrameFeatures | None = None

    @staticmethod
    def _regions(hsv):
        hue, sat, value = cv2.split(hsv)
        mask = (((hue < 15) | (hue >= 140)) & (sat >= 110) & (value >= 55)).astype(np.uint8)*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            moments = cv2.moments(contour)
            if area < 100 or moments['m00'] <= 0:
                continue
            _, _, width, height = cv2.boundingRect(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            if (not .55 <= width/max(height, 1) <= 2.0 or perimeter <= 0
                    or area/max(hull_area, 1) < .75 or 4*np.pi*hull_area/perimeter**2 < .75):
                continue
            (_, _), radius = cv2.minEnclosingCircle(contour)
            if 10 <= radius <= 50:
                regions.append(_RedRegion(moments['m10']/moments['m00'],
                                          moments['m01']/moments['m00'], area, radius))
        return regions

    def _seed_diameter(self, gray, hsv, cloth, region, context):
        if (not context.cue_ball_detected or not context.cue_ball_quality
                or context.cue_ball_observations < 3 or context.cue_ball_track_confidence < .75
                or context.cue_ball_x is None or context.cue_ball_y is None
                or context.cue_ball_stable_normalized_speed is None
                or context.cue_ball_stable_normalized_speed > .3):
            return None
        white_diameter = context.cue_ball_image_diameter_px or context.ball_diameter_px
        if not 20 <= white_diameter <= 80:
            return None
        distance = np.hypot(region.x-context.cue_ball_x, region.y-context.cue_ball_y)
        if not .4*white_diameter <= distance <= 2*white_diameter:
            return None
        radius = region.radius
        x0, x1 = max(0, round(region.x-2*radius)), min(gray.shape[1], round(region.x+2*radius))
        y0, y1 = max(0, round(region.y-2*radius)), min(gray.shape[0], round(region.y+2*radius))
        circles = cv2.HoughCircles(
            gray[y0:y1, x0:x1], cv2.HOUGH_GRADIENT, dp=1, minDist=radius*.8,
            param1=80, param2=18, minRadius=round(radius*.85), maxRadius=round(radius*1.2),
        )
        choices = []
        for x, y, r in circles[0] if circles is not None else []:
            x, y, r = float(x+x0), float(y+y0), float(r)
            if (np.hypot(x-region.x, y-region.y) > .6*radius
                    or not .65 <= 2*r/white_diameter <= 1.5
                    or not .30 <= region.area/(np.pi*r*r) <= 1.1):
                continue
            _, deviation, surround = self.detector._colour_scores(hsv, cloth, x, y, r)
            if deviation >= .8 and surround >= .25:
                choices.append((np.hypot(x-region.x, y-region.y), 2*r))
        return min(choices)[1] if choices else None

    def observe(self, frame, context: FrameFeatures):
        previous = self.last
        self.last = context
        valid = (context.observation_fps >= 20 and context.observation_valid
                 and context.match_context_valid and not context.broadcast_replay
                 and not context.table_handling and not context.rack_idle
                 and context.view_type == CameraViewType.BALL_CLOSEUP
                 and context.scene_cut_score < .42 and context.camera_motion_magnitude <= .5)
        contiguous = (valid and previous is not None and previous.camera_scene_id == context.camera_scene_id
                      and 0 < context.t-previous.t <= 1.5/context.observation_fps)
        if not valid or not contiguous:
            self.tracks = []
        if not valid:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cloth = cv2.inRange(hsv, self.detector.cloth_lower, self.detector.cloth_upper)
        regions = self._regions(hsv)
        used, kept = set(), []
        for track in self.tracks:
            options = [(i, region) for i, region in enumerate(regions)
                       if i not in used and np.hypot(region.x-track.region.x,
                                                    region.y-track.region.y) <= .5*track.diameter
                       and .65 <= region.area/track.region.area <= 1.5]
            if not options:
                continue
            i, region = min(options, key=lambda p: np.hypot(p[1].x-track.region.x, p[1].y-track.region.y))
            used.add(i)
            sample = (context.t, region.x, region.y, region.area)
            rest = np.asarray(track.rest)
            anchor = np.median(rest[:, 1:3], axis=0)
            armed = (len(rest) >= 7 and rest[-1, 0]-rest[0, 0] >= .19
                     and np.max(np.linalg.norm(rest[:, 1:3]-anchor, axis=1)) <= .1*track.diameter
                     and np.max(rest[:, 3])/np.min(rest[:, 3]) <= 1.25)
            away = np.hypot(region.x-anchor[0], region.y-anchor[1])/track.diameter
            # Freeze the quiet anchor at the same boundary used to prove rest.
            # Folding the first slow departure into rest would then invalidate
            # that rest and lose the whole otherwise continuous launch.
            if armed and (track.moving or away >= .10):
                track.moving.append(sample)
                moving = np.asarray(track.moving)
                xy = np.vstack((anchor, moving[:, 1:3]))
                steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                net = np.linalg.norm(xy[-1]-xy[0])
                elapsed = context.t-track.moving[0][0]
                if (elapsed > .6 or np.max(moving[:, 3])/np.median(rest[:, 3]) > 1.6
                        or np.min(moving[:, 3])/np.median(rest[:, 3]) < .60):
                    continue
                if (len(moving) >= 4 and elapsed >= .095 and net/track.diameter >= .65
                        and net/max(np.sum(steps), 1e-6) >= .90
                        and np.max(steps)/track.diameter <= .5):
                    return {'upper': track.moving[0][0], 'confirmed': context.t,
                            'rest_frames': len(rest), 'moving_frames': len(moving),
                            'travel_diameters': float(net/track.diameter),
                            'direction_consistency': float(net/max(np.sum(steps), 1e-6))}
            else:
                track.rest.append(sample)
                track.rest = track.rest[-16:]
            track.region = region
            kept.append(track)
        for i, region in enumerate(regions):
            if i in used:
                continue
            diameter = self._seed_diameter(gray, hsv, cloth, region, context)
            if diameter is not None:
                kept.append(_RedTrack(region, diameter, [(context.t, region.x, region.y, region.area)]))
        self.tracks = kept
        return None
