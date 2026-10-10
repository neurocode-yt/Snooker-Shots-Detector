"""Native geometry and foreground checks for automatic live-shot endings.

Foreground pixels supply obstruction and table-coverage evidence. Existing ball
kinematics and the ordinary stop detector supply stillness; raw caches stay intact.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.event_fusion.ball_stop import BallStopDetector, StopDetection
from snooker_ai.scene_detection.table_context import ViewGeometry, view_geometry
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.types import CameraViewType, FrameFeatures, ShotRecord, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.utils.video import open_capture


@dataclass(frozen=True)
class VisualHandEntry:
    strike_timestamp: float
    entry_timestamp: float
    confirmation_timestamp: float
    source_entry_pts: float
    source_confirmation_pts: float
    confidence: float = .85


@dataclass(frozen=True)
class HandComponent:
    box: tuple[int, int, int, int]
    diameter: float
    canonical: tuple[float, float]
    broad: bool
    entry: bool
    prior_clear_times: tuple[float, ...] = ()
    kind: str = "glove"

    @property
    def centre(self) -> np.ndarray:
        x, y, width, height = self.box
        return np.array([x + width / 2, y + height / 2])


@dataclass
class TailObservation:
    # Native observations use decoded presentation time, including VFR video.
    # Proxy observations use TimeMapper.to_source(frame_index / proxy_fps).
    t: float
    context: FrameFeatures | None
    full_table: bool = False
    reset: bool = False
    components: list[HandComponent] = field(default_factory=list)


class _TailPixels:
    """View-local geometry and evidence that an entry location was clear."""

    def __init__(self, config: Config):
        self.config = config
        data = config.as_dict()
        data.setdefault("table_detection", {})["temporal_smooth_frames"] = 1
        self.localizer = TableLocalizer(Config(data))
        self.precise_localizer = TableLocalizer(Config(data))
        self.cut_threshold = float(config.get("scene_detection.hard_cut_threshold", .42))
        self.last_t: float | None = None
        self.scene_id: int | None = None
        self.thumbnail: np.ndarray | None = None
        self.geometry: ViewGeometry | None = None
        self.geometry_t = float("-inf")
        self.green_history: list[tuple[float, np.ndarray]] = []
        self.surface_history: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.localizer.reset()
        self.precise_localizer.reset()
        self.geometry = None
        self.geometry_t = float("-inf")
        self.green_history.clear()
        self.surface_history.clear()

    def observe(self, frame: np.ndarray, t: float,
                context: FrameFeatures | None) -> TailObservation:
        small = cv2.resize(frame, (640, 360))
        thumb = cv2.resize(small, (64, 36))
        change = (float(np.mean(cv2.absdiff(thumb, self.thumbnail)))
                  if self.thumbnail is not None else 0.)
        broken = (
            context is None
            or (self.last_t is not None and not 0 < t-self.last_t <= .26)
            or (context is not None and (
                context.scene_cut_score >= self.cut_threshold
                or self.scene_id is not None and context.camera_scene_id != self.scene_id
                or context.camera_motion_magnitude > 3
                or not context.match_context_valid or context.broadcast_replay
            ))
            or change > 24
        )
        if broken:
            self.reset()
        self.thumbnail = thumb
        self.last_t = t
        self.scene_id = context.camera_scene_id if context is not None else None
        result = TailObservation(t, context, reset=broken)
        # A reset observation cannot supply clearance for a later glove.
        if broken:
            return result

        table = self.localizer.detect(small)
        geometry = view_geometry(table, small.shape)
        if not geometry.full_table and frame.shape[1] >= 960:
            # Rounding a shallow cushion to 640px can turn a measured four-
            # corner hull into five vertices. Retry calibration at the same
            # spatial resolution used by native shot analysis; all component
            # histories and boundaries stay in the original 640px coordinates.
            precise = cv2.resize(frame, (960, 540))
            precise_table = self.precise_localizer.detect(precise)
            measured = view_geometry(precise_table, precise.shape)
            if measured.full_table and measured.homography is not None:
                scale = np.diag([1.5, 1.5, 1.]).astype(np.float64)
                geometry = ViewGeometry(True, measured.homography @ scale)
        # Corner fitting can fail while a player/referee covers a small piece
        # of an otherwise unchanged main view. Revalidate the last measured
        # playing-surface polygon against current cloth pixels instead of
        # expiring its calibration solely because that corner is obscured.
        if (not geometry.full_table and self.geometry is not None
                and context is not None and context.view_type == CameraViewType.MAIN_TABLE
                and context.table_confidence >= .60):
            polygon = self._polygon(self.geometry, small.shape[:2]) > 0
            observed = table.mask > 0
            intersection = np.count_nonzero(polygon & observed)
            union = np.count_nonzero(polygon | observed)
            if (intersection/max(1, np.count_nonzero(polygon)) >= .85
                    and intersection/max(1, union) >= .80):
                self.geometry_t = t
        if geometry.full_table:
            polygon = self._polygon(geometry, small.shape[:2])
            coverage = np.count_nonzero((polygon > 0) & (table.mask > 0)) / max(
                1, np.count_nonzero(polygon))
            if coverage >= .75:
                self.geometry, self.geometry_t = geometry, t
        if self.geometry is None or t-self.geometry_t > 1:
            return result
        result.full_table = True
        cloth = self._polygon(self.geometry, small.shape[:2])
        # A fitted corner can clip a real patch beside the cushion. Foreground
        # clearance must also include cloth actually observed in this stable
        # view before the hand, rather than depend only on that fitted outline.
        self.surface_history = [(seen, mask) for seen, mask in self.surface_history
                                if 0 < t-seen <= .5]
        for _, measured_surface in self.surface_history:
            cloth = cv2.bitwise_or(cloth, measured_surface)
        cloth = cv2.bitwise_or(cloth, table.mask)
        self.surface_history.append((t, table.mask.copy()))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 80, 40), (95, 255, 255))
        self.green_history = [(seen, mask) for seen, mask in self.green_history
                              if 0 < t-seen <= .5]
        result.components = self._components(small, hsv, green, cloth)
        result.components.extend(self._foreground_components(green, cloth))
        self.green_history.append((t, green))
        return result

    @staticmethod
    def _polygon(geometry: ViewGeometry, shape: tuple[int, int]) -> np.ndarray:
        corners = cv2.perspectiveTransform(
            np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], np.float32),
            np.linalg.inv(geometry.homography))[0]
        polygon = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(polygon, np.round(corners).astype(np.int32), 255)
        return polygon

    def _components(self, image: np.ndarray, hsv: np.ndarray, green: np.ndarray,
                    cloth: np.ndarray) -> list[HandComponent]:
        neutral = cv2.inRange(hsv, (0, 0, 155), (179, 78, 255))
        blue, green_channel, red = [c.astype(np.int16) for c in cv2.split(image)]
        neutral[(red-blue > 22) & (red-green_channel > 8)] = 0
        neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centres = cv2.connectedComponentsWithStats(neutral)
        inverse = np.linalg.inv(self.geometry.homography)
        accepted = []
        for index in range(1, count):
            x, y, width, height, area = stats[index].tolist()
            point = cv2.perspectiveTransform(
                np.array([[centres[index]]], np.float32), self.geometry.homography)[0, 0]
            if not np.all((point >= -.06) & (point <= 1.06)):
                continue
            across = cv2.perspectiveTransform(
                np.array([[[0, point[1]], [1, point[1]]]], np.float32), inverse)[0]
            diameter = max(3., float(np.linalg.norm(across[1]-across[0])) * .0295)
            if area < .7*diameter**2 or max(width, height) < 1.5*diameter:
                continue
            component = labels == index
            on_cloth = np.count_nonzero(component & (cloth > 0))
            thickness = float(cv2.distanceTransform(
                component.astype(np.uint8), cv2.DIST_L2, 3).max()*2)
            radius = max(2, round(diameter*.6))
            ring = (cv2.dilate(component.astype(np.uint8), cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0) & ~component
            green_ring = np.count_nonzero(ring & (green > 0)) / max(1, np.count_nonzero(ring))
            broad = (area >= 1.7*diameter**2 and max(width, height) >= 2*diameter
                     and min(width, height) >= 1.1*diameter and thickness >= .7*diameter
                     and max(width, height) <= 10*diameter)
            entry = on_cloth >= max(.18*area, .55*diameter**2) and green_ring >= .30
            roi = cloth[y:y+height, x:x+width] > 0
            roi_size = np.count_nonzero(roi)
            clear = tuple(seen for seen, prior_green in self.green_history
                          if roi_size >= .25*width*height and np.count_nonzero(
                              (prior_green[y:y+height, x:x+width] > 0) & roi) >= .65*roi_size)
            accepted.append(HandComponent(
                (x, y, width, height), diameter, tuple(float(v) for v in point),
                bool(broad), bool(entry), clear))
        return accepted

    def _foreground_components(self, green: np.ndarray,
                               cloth: np.ndarray) -> list[HandComponent]:
        """Measure a new broad obstruction crossing previously clear cloth.

        A referee's dark body reaches the table before their white glove.
        Keep ball-sized regions and thin cue shafts out of this proposal.
        The temporal verifier still requires entry from the rail, prior clear
        cloth, continuous footage and no player addressing the white.
        """
        foreground = ((cloth > 0) & (green == 0)).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                     np.ones((5, 5), np.uint8))
        count, labels, stats, centres = cv2.connectedComponentsWithStats(foreground)
        rail = (cloth > 0) & (cv2.erode(cloth, np.ones((3, 3), np.uint8)) == 0)
        inverse = np.linalg.inv(self.geometry.homography)
        accepted = []
        for index in range(1, count):
            x, y, width, height, area = stats[index].tolist()
            point = cv2.perspectiveTransform(
                np.array([[centres[index]]], np.float32), self.geometry.homography)[0, 0]
            across = cv2.perspectiveTransform(
                np.array([[[0, point[1]], [1, point[1]]]], np.float32), inverse)[0]
            diameter = max(3., float(np.linalg.norm(across[1]-across[0])) * .0295)
            if (area < .7*diameter**2 or max(width, height) < 1.5*diameter
                    or min(width, height) < .7*diameter
                    or np.count_nonzero((labels == index) & rail) < .6*diameter):
                continue
            broad = (area >= 3*diameter**2 and max(width, height) >= 3*diameter
                     and min(width, height) >= 1.2*diameter)
            roi = cloth[y:y+height, x:x+width] > 0
            roi_size = np.count_nonzero(roi)
            clear = tuple(seen for seen, prior_green in self.green_history
                          if np.count_nonzero(
                              (prior_green[y:y+height, x:x+width] > 0) & roi) >= .80*roi_size)
            accepted.append(HandComponent(
                (x, y, width, height), diameter, tuple(float(v) for v in point),
                bool(broad), True, clear, "foreground"))
        return accepted


def _matching_component(row: TailObservation, previous: HandComponent,
                        dt: float, *, broad: bool) -> HandComponent | None:
    options = [component for component in row.components
               if component.kind == previous.kind
               and component.entry and (component.broad or not broad)
               and np.linalg.norm(component.centre-previous.centre)
               <= max((2 if broad else 1.6)*component.diameter, dt*500)]
    return min(options, key=lambda item: np.linalg.norm(item.centre-previous.centre)) if options else None


def _footprint_overlap(first: HandComponent, second: HandComponent) -> float:
    x, y, width, height = first.box
    other_x, other_y, other_width, other_height = second.box
    intersection = (max(0, min(x+width, other_x+other_width)-max(x, other_x))
                    * max(0, min(y+height, other_y+other_height)-max(y, other_y)))
    return intersection / max(1, width*height+other_width*other_height-intersection)


def _confirmed_entries(rows: list[TailObservation], strike: float,
                       cut_threshold: float) -> list[VisualHandEntry]:
    entries = []
    for index, row in enumerate(rows):
        for component in row.components:
            if not component.broad or not component.entry:
                continue
            last_index, last = index, component
            for following in range(index+1, len(rows)):
                dt = rows[following].t-rows[last_index].t
                if dt > .25:
                    break
                match = _matching_component(rows[following], last, dt, broad=True)
                if match is not None:
                    last_index, last = following, match
                if rows[last_index].t-row.t >= .16:
                    break
            if rows[last_index].t-row.t < .16:
                continue
            confirmation = rows[last_index].t
            first, initial = index, component
            for preceding in range(index-1, -1, -1):
                dt = rows[first].t-rows[preceding].t
                if dt > .13:
                    break
                match = _matching_component(rows[preceding], initial, dt, broad=False)
                if match is not None:
                    # An edge-touching fragment may overlap the cushion and
                    # have no measured cloth clearance. Keep the earliest
                    # verified footprint instead of erasing the following,
                    # continuously observed entry with that uncertain fragment.
                    earlier_clear = [seen for seen in match.prior_clear_times
                                     if rows[preceding].t-.5 <= seen < rows[preceding].t]
                    if len(earlier_clear) < 2 or max(earlier_clear)-min(earlier_clear) < .079:
                        break
                    first, initial = preceding, match
            onset = rows[first].t
            # A broad arm can have its centre deep inside the table while its
            # measured outline still crosses the rail. Foreground proposals
            # already require that rail intersection; detached gloves must
            # independently start near the measured edge.
            if (onset < strike+2 or initial.kind != "foreground"
                    and min(*initial.canonical, *(1-v for v in initial.canonical)) > .08):
                continue
            clear = [seen for seen in initial.prior_clear_times if onset-.5 <= seen < onset]
            if len(clear) < 2 or max(clear)-min(clear) < .079:
                continue
            earlier = [r for r in rows if onset-.6 <= r.t < onset-.13]
            if any(c.entry and np.linalg.norm(c.centre-initial.centre) <= 2*c.diameter
                   and _footprint_overlap(c, initial) >= .50
                   and c.kind == initial.kind
                   for r in earlier for c in r.components):
                continue
            prior = [r for r in rows if onset-.65 <= r.t <= confirmation]
            if (not prior or any(b.t-a.t > .12 for a, b in zip(prior, prior[1:]))
                    or any(r.reset or r.context is None for r in prior)):
                continue
            contexts = [r.context for r in prior]
            if (len({f.camera_scene_id for f in contexts}) != 1
                    or any(f.scene_cut_score >= cut_threshold or not f.match_context_valid
                           or f.broadcast_replay for f in contexts)):
                continue
            # Cue geometry from the player leaving earlier in the clip must
            # not protect the referee's subsequent entry. A bridge present at
            # the entry itself still vetoes the boundary, as does its earlier
            # component identity above.
            if any(r.context.cue_tip_visible or r.context.cue_contact_score >= .2
                   for r in prior if r.t >= onset-.12):
                continue
            # Geometry can warm up in an otherwise continuously observed view.
            # Require at least .4s of full-table coverage before entry; earlier
            # warmup rows still supply the cut/context vetoes above.
            first_full = next((i for i, r in enumerate(prior) if r.full_table), len(prior))
            full = prior[first_full:]
            if not full or full[0].t > onset-.4 or any(not r.full_table for r in full):
                continue
            if not entries or onset-entries[-1].entry_timestamp > .3:
                entries.append(VisualHandEntry(strike, onset, confirmation, onset, confirmation))
    return entries


class VisualTailDetector:
    """Scan clip endings cheaply, then verify source-frame foreground entry."""

    def __init__(self, config: Config):
        self.config = config
        self.cut_threshold = float(config.get("scene_detection.hard_cut_threshold", .42))

    @staticmethod
    def eligible(shot: ShotRecord) -> bool:
        return bool(shot.included and not shot.possible_replay and not shot.user_modified)

    @staticmethod
    def _context(features: list[FrameFeatures], times: list[float], t: float) -> FrameFeatures | None:
        index = bisect_left(times, t)
        nearby = features[max(0, index-1):index+1]
        if not nearby:
            return None
        nearest = min(nearby, key=lambda f: abs(f.t-t))
        return nearest if abs(nearest.t-t) <= .26 else None

    def _observe_window(self, capture: cv2.VideoCapture, start: float, end: float,
                        mapper: TimeMapper, features: list[FrameFeatures], times: list[float],
                        *, native: bool) -> list[TailObservation]:
        pixels = _TailPixels(self.config)
        rows = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            return rows
        if native:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0, start)*1000)
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(mapper.to_proxy(start)*fps)))
        next_sample = start
        previous_pts = None
        while True:
            # Native source images are kept at their original cadence. Never
            # substitute index/average_fps when presentation timestamps fail.
            ok, frame = capture.read()
            if not ok:
                break
            if native:
                t = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
                if not np.isfinite(t) or (previous_pts is not None and t <= previous_pts):
                    return []
                previous_pts = t
            else:
                frame_index = max(0., capture.get(cv2.CAP_PROP_POS_FRAMES)-1)
                t = mapper.to_source(frame_index/fps)
            if t > end+1e-6:
                break
            if t < start-1e-6 or (not native and t < next_sample-1e-6):
                continue
            if not native:
                while next_sample <= t+1e-6:
                    next_sample += .2
            context = self._context(features, times, t)
            rows.append(pixels.observe(frame, t, context))
        return rows

    def revalidated_stops(
        self, source: str | Path, time_mapper: TimeMapper, shots: list[ShotRecord],
        features: list[FrameFeatures],
    ) -> dict[float, StopDetection]:
        """Recheck uncertain ends against native, measured table coverage.

        A rounded cushion can defeat the raw four-corner fit after movement.
        Revalidated geometry permits existing ball kinematics to be evaluated
        by the ordinary stop detector. It supplies no motion or stillness by
        itself, and neither cached features nor contact candidates are changed.
        """
        if not bool(self.config.get("visual_tail.enabled", True)) or not features:
            return {}
        ordered = sorted(features, key=lambda f: f.t)
        times = [f.t for f in ordered]
        eligible = [shot for shot in shots if self.eligible(shot)
                    and (not shot.evidence.get("stop_confirmed") or shot.manual_review_required)]
        if not eligible:
            return {}
        capture = open_capture(source)
        stops = {}
        detector = BallStopDetector(self.config)
        try:
            if not capture.isOpened():
                return {}
            for shot in eligible:
                start = max(shot.cue_strike, shot.clip_end-20)
                end = min(time_mapper.source_duration, shot.clip_end+.6)
                context_window = ordered[bisect_left(times, start):bisect_right(times, end)]
                if not any(not f.table_full_view and f.observation_fps >= 10
                           and f.view_type == CameraViewType.MAIN_TABLE
                           and f.table_confidence >= .80 for f in context_window):
                    continue
                rows = self._observe_window(capture, start, end, time_mapper,
                                            ordered, times, native=True)
                measured = []
                recovered = False
                for row in rows:
                    context = row.context
                    # A nearby sparse or differently timed frame cannot prove
                    # native stillness. Missing evidence stays explicitly unknown.
                    exact = (context is not None and context.observation_fps >= 10
                             and abs(context.t-row.t) <= .001)
                    if not exact:
                        measured.append(FrameFeatures(t=row.t, observation_valid=False,
                            table_observable=False, table_full_view=False))
                        continue
                    full = bool(row.full_table and not row.reset
                                and context.view_type == CameraViewType.MAIN_TABLE
                                and context.table_confidence >= .80)
                    recovered |= full and not context.table_full_view
                    measured.append(context.model_copy(update={'t': row.t, 'table_full_view': full}))
                if not recovered:
                    continue
                candidate = StrikeCandidate(timestamp=shot.cue_strike, confidence=shot.strike_confidence)
                stop = detector.detect_stop(candidate, measured, time_mapper.source_duration)
                # Only shorten an unresolved edit using a confirmed ordinary
                # stop. Unknown gaps retain the detector's timing review flag.
                if (stop.confirmed and stop.physical_stop_timestamp < shot.clip_end
                        and stop.physical_stop_timestamp >= shot.cue_strike+.2):
                    stops[shot.cue_strike] = replace(stop, manual_review_required=True)
        finally:
            capture.release()
        return stops

    def detect(self, source: str | Path, proxy: str | Path, time_mapper: TimeMapper,
               shots: list[ShotRecord], features: list[FrameFeatures]) -> list[VisualHandEntry]:
        if not bool(self.config.get("visual_tail.enabled", True)) or not features:
            return []
        eligible = [shot for shot in shots if self.eligible(shot)]
        if not eligible:
            return []
        ordered = sorted(features, key=lambda item: item.t)
        times = [feature.t for feature in ordered]
        proxy_capture = open_capture(proxy)
        native_capture = None
        entries = []
        try:
            if not proxy_capture.isOpened():
                return []
            for shot in eligible:
                start = max(shot.cue_strike+1, shot.clip_end-20)
                end = min(time_mapper.source_duration, shot.clip_end+.4)
                if end <= start:
                    continue
                rows = self._observe_window(proxy_capture, start, end, time_mapper,
                                            ordered, times, native=False)
                hits = [row for row in rows if row.t >= shot.cue_strike+2
                        and any(c.broad and c.entry for c in row.components)]
                triggers = [a.t for a, b in zip(hits, hits[1:]) if 0 < b.t-a.t <= .25]
                if not triggers:
                    continue
                if native_capture is None:
                    native_capture = open_capture(source)
                if not native_capture.isOpened():
                    break
                last_window_end = float("-inf")
                for trigger in triggers:
                    if trigger < last_window_end:
                        continue
                    # Measure the clear table before a person hides a corner.
                    # The extra lead-in supplies geometry and clearance; the
                    # native entry, continuity and cue vetoes still decide.
                    window_start = max(shot.cue_strike, trigger-3.4)
                    last_window_end = min(end, trigger+.6)
                    native = self._observe_window(native_capture, window_start, last_window_end,
                                                  time_mapper, ordered, times, native=True)
                    confirmed = [entry for entry in _confirmed_entries(
                        native, shot.cue_strike, self.cut_threshold)
                        if entry.entry_timestamp < shot.clip_end]
                    if confirmed:
                        entries.append(confirmed[0])
                        break
        finally:
            proxy_capture.release()
            if native_capture is not None:
                native_capture.release()
        return entries
