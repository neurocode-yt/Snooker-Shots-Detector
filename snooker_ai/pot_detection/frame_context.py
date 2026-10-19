"""Keep the opening and final live contact of visually identified snooker frames."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.event_fusion.rack_idle import RackIdleGate
from snooker_ai.types import (
    AnalysisResult,
    CameraViewType,
    FrameFeatures,
    StrikeCandidate,
    TimelineEvent,
)
from snooker_ai.utils.persistence import atomic_text
from snooker_ai.utils.video import open_capture, sampled_frames
from snooker_ai.utils.work_control import check_pause


@dataclass(frozen=True)
class FrameRange:
    start: float
    opening_end: float
    end: float


def frame_ranges(readings, duration, max_score_gap=0.8):
    # Imported at call time: the score selector also uses these frame helpers.
    from snooker_ai.pot_detection.selector import stable_scores

    runs = stable_scores(readings, max_gap=max_score_gap)
    episodes = []
    zero = None
    previous = None
    for run in runs:
        if (run.left, run.right) == (0, 0):
            if zero is None:
                zero = run
            continue
        if zero is not None:
            # A transient zero followed by the old score is an overlay glitch
            # or correction. It cannot split a frame into extra highlights.
            returns_to_old = previous is not None and (
                run.left >= previous.left and run.right >= previous.right
            )
            if not returns_to_old:
                episodes.append((0.0 if previous is None else zero.start, run.start))
            zero = None
        previous = run
    if zero is not None:
        episodes.append((0.0 if previous is None else zero.start, duration))
    return [
        FrameRange(start, opening_end, episodes[i + 1][0] if i + 1 < len(episodes) else duration)
        for i, (start, opening_end) in enumerate(episodes)
    ]


def rack_cut_contact(features: list[FrameFeatures], rack_start: float):
    """Recover a contact immediately before a cut to the intact opening rack.

    One measured departure before the cut must be followed by independently
    measured sustained white-ball travel in the new full-table view. Quiet
    address and visible cue geometry are required before contact. The camera
    cut itself, scoreboard, and rack alone can never create a shot.
    """
    frames = sorted(
        (f for f in features if rack_start - 3 <= f.t <= rack_start + 2), key=lambda f: f.t
    )
    for i in range(6, len(frames) - 4):
        before, launch = frames[i - 6 : i], frames[i]
        after = next(
            (
                frames[j : j + 4]
                for j in range(i + 1, min(i + 5, len(frames) - 3))
                if all(f.cue_ball_quality for f in frames[j : j + 4])
            ),
            [],
        )
        if len(after) != 4 or after[0].t - launch.t > 0.20:
            continue
        intervening = [f for f in frames if launch.t < f.t < after[0].t]
        if any(
            not f.match_context_valid or f.broadcast_replay or f.table_handling for f in intervening
        ):
            continue
        native = before + [launch] + after
        if any(
            not f.observation_valid
            or not f.match_context_valid
            or f.broadcast_replay
            or f.table_handling
            or f.observation_fps < 20
            for f in native
        ):
            continue
        fps = min(f.observation_fps for f in native)
        if (
            any(
                not 0 < b.t - a.t <= 1.6 / fps
                for part in [before + [launch], after]
                for a, b in zip(part, part[1:])
            )
            or before[-1].t - before[0].t < 0.15
        ):
            continue
        if not all(f.camera_scene_id == before[0].camera_scene_id for f in before + [launch]):
            continue
        if (
            launch.camera_scene_id == after[0].camera_scene_id
            or not all(f.camera_scene_id == after[0].camera_scene_id for f in after)
            or not all(
                f.view_type is CameraViewType.MAIN_TABLE
                and f.table_full_view
                and f.camera_motion_magnitude <= 1.5
                for f in after
            )
        ):
            continue
        if any(
            not f.cue_ball_quality
            or f.cue_ball_track_confidence < 0.75
            or f.cue_ball_x is None
            or f.cue_ball_y is None
            for f in native
        ):
            continue
        diameter = float(before[-1].cue_ball_image_diameter_px or before[-1].ball_diameter_px)
        if diameter <= 0:
            continue
        rest = np.array([(f.cue_ball_x, f.cue_ball_y) for f in before])
        center = np.median(rest, axis=0)
        if (
            np.max(np.linalg.norm(rest - center, axis=1)) > 0.12 * diameter
            or max(f.cue_contact_score for f in before) < 0.6
            or sum(f.cue_tip_visible for f in before) < 3
        ):
            continue
        departure = (
            np.linalg.norm(np.array([launch.cue_ball_x, launch.cue_ball_y]) - center) / diameter
        )
        if not 0.45 <= departure <= 3:
            continue
        points = np.array([(f.cue_ball_x, f.cue_ball_y) for f in after])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        d = float(after[0].cue_ball_image_diameter_px or after[0].ball_diameter_px)
        travel = np.linalg.norm(points[-1] - points[0])
        coherence = travel / max(float(sum(steps)), 1e-6)
        if d <= 0 or np.min(steps) < 0.3 * d or travel < 1.5 * d or coherence < 0.98:
            continue
        return StrikeCandidate(
            timestamp=launch.t,
            confidence=0.85,
            uncertainty_start=before[-1].t,
            uncertainty_end=launch.t,
            camera_view=launch.view_type,
            evidence={
                "frame_opening_cut_contact": 1.0,
                "opening_rest_samples": 6.0,
                "opening_post_samples": 4.0,
                "opening_departure_diameters": float(departure),
                "opening_post_travel_diameters": float(travel / d),
                "opening_post_direction_consistency": float(coherence),
                "opening_rack_timestamp": rack_start,
                "cue_geometry_confirmed": 1.0,
                "cue_ball_motion_confirmed": 1.0,
                "dense_transition_confirmed": 1.0,
                "refined_ball_motion_start": launch.t,
            },
        )
    return None


class FrameContextDetector:
    def __init__(self, config, job_dir):
        self.config = config
        self.job_dir = Path(job_dir)

    def scan_racks(self, result, ranges, progress=None):
        """Scan only zero-score opening intervals, with a separate reusable cache."""
        path = Path(result.source_path)
        stat = path.stat()
        signature = hashlib.sha256(
            json.dumps(
                [
                    1,
                    str(path.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    [[r.start, r.opening_end] for r in ranges],
                ]
            ).encode()
        ).hexdigest()
        cache = self.job_dir / "frame_context_cache.json"
        racks = []
        try:
            data = json.loads(cache.read_text("utf-8"))
            if data["signature"] == signature:
                racks = data["racks"]
                if not isinstance(racks, list) or len(racks) > len(ranges):
                    racks = []
                if len(racks) == len(ranges):
                    return racks
        except (OSError, ValueError, KeyError, TypeError):
            pass
        cap = open_capture(
            path, prefer_hwaccel=bool(self.config.get("analysis.hwaccel_decode", True))
        )
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open source to check frame openings")
            fps = cap.get(cv2.CAP_PROP_FPS) or result.metadata.fps
            for index, interval in enumerate(ranges):
                if index < len(racks):
                    continue
                check_pause()
                gate = RackIdleGate()
                cap.set(cv2.CAP_PROP_POS_MSEC, interval.start * 1000)
                stream = sampled_frames(
                    cap,
                    proxy_fps=fps,
                    to_source=lambda t: t,
                    start_time=interval.start,
                    end_time=interval.opening_end,
                    sample_period=0.5,
                    start_idx=int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                )
                found = []
                try:
                    for _, t, image in stream:
                        check_pause()
                        gate.observe(image, t)
                        if gate.racked:
                            found.append(t)
                        if progress and not int(t * 2) % 40:
                            progress(
                                index / max(len(ranges), 1), f"Checking frame {index + 1} opening"
                            )
                finally:
                    stream.close()
                groups = []
                for t in found:
                    if groups and t - groups[-1][-1] <= 0.8:
                        groups[-1].append(t)
                    else:
                        groups.append([t])
                racks.append([[g[0], g[-1]] for g in groups if len(g) >= 2])
                atomic_text(cache, json.dumps({"signature": signature, "racks": racks}))
        finally:
            cap.release()
        return racks

    def recover_opening(self, result, rack):
        # Reuse the normal native observations, stop detector and clip builder.
        # Import locally so score-only jobs do not initialize another analyzer.
        from snooker_ai.pipeline.analyzer import Analyzer
        from snooker_ai.utils.timebase import TimeMapper

        owner = Analyzer(self.config, self.job_dir / "frame_opening")
        owner._native_source = Path(result.source_path)
        source_signature = owner._analysis_signature(Path(result.source_path))
        start, end = max(0.0, rack[0] - 3), min(result.original_duration, rack[1] + 3)
        features = [f for f in result.features if start <= f.t <= end]
        if not features or sum(f.observation_fps >= 20 for f in features) < 30:
            hint = StrikeCandidate(
                timestamp=rack[0],
                confidence=0.5,
                uncertainty_start=start,
                uncertainty_end=end,
                evidence={"rack_restart": 1.0},
            )
            _, features = owner._refine_candidate_windows(
                Path(result.proxy_path or result.source_path),
                None,
                TimeMapper(result.original_duration, source_fps=result.metadata.fps),
                result.original_duration,
                [hint],
                [],
                resume=True,
                signature=f"frame-opening-1:{source_signature}:{rack[0]:.6f}",
            )
            owner._annotate_scenes(features, result.scenes)
        candidate = rack_cut_contact(features, rack[0])
        if candidate is None:
            return None
        # Extend only this recovered shot through its ordinary stopping window.
        mapper = TimeMapper(result.original_duration, source_fps=result.metadata.fps)
        _, observed = owner._refine_candidate_windows(
            Path(result.proxy_path or result.source_path),
            None,
            mapper,
            result.original_duration,
            [candidate],
            [],
            resume=True,
            signature=f"frame-opening-stop-1:{source_signature}:{candidate.timestamp:.6f}",
            existing_dense=result.features,
        )
        features = owner._merge_feature_layers(features, observed)
        # Keep the independently proven contact; normal refinement may be unable
        # to join the two camera coordinates that caused the original omission.
        shot = owner.segmenter.build([candidate], features, result.original_duration, result.mode)
        if not shot:
            return None
        result.strike_candidates.append(candidate)
        return shot[0]

    def identify(self, result: AnalysisResult, readings, progress=None):
        for shot in result.shots:
            shot.evidence.pop("frame_context", None)
        ranges = frame_ranges(
            readings,
            result.original_duration,
            max_score_gap=max(
                0.8, 1.6 / max(0.5, float(self.config.get("pot_selection.sample_fps", 2)))
            ),
        )
        ranges = [
            r
            for r in ranges
            if any(r.start <= s.cue_strike < r.end and not s.possible_replay for s in result.shots)
        ]
        if not ranges:
            return []
        original_count = len(result.shots)
        old_strikes = {s.shot_id: s.cue_strike for s in result.shots}
        racks = self.scan_racks(result, ranges, progress)
        identified = []
        for interval, episodes in zip(ranges, racks):
            check_pause()
            live = sorted(
                (
                    s
                    for s in result.shots
                    if not s.possible_replay
                    and interval.start <= s.cue_strike < interval.end
                    and s.evidence.get("cue_geometry_confirmed") != 0
                ),
                key=lambda s: s.cue_strike,
            )
            opening = None
            for rack in episodes:
                # A tight intact rack is supporting evidence only. Match it to
                # an independently detected contact, not to setup/handling clips.
                near = [s for s in live if rack[0] - 1 <= s.cue_strike <= rack[1] + 1]
                if near:
                    opening = min(near, key=lambda s: abs(s.cue_strike - rack[0]))
                    break
                opening = self.recover_opening(result, rack)
                if opening is not None:
                    result.shots.append(opening)
                    break
            if opening is None:
                continue
            live = [s for s in live if s.cue_strike >= opening.cue_strike]
            if opening not in live:
                live.insert(0, opening)
            identified.append(
                {
                    "frame": len(identified) + 1,
                    "start": interval.start,
                    "end": interval.end,
                    "first_strike": opening.cue_strike,
                    "last_strike": live[-1].cue_strike,
                }
            )
        # New contacts can change shot IDs, so assign bookends only after sorting.
        if len(result.shots) != original_count:
            result.shots.sort(key=lambda s: s.cue_strike)
            ids = {}
            for index, shot in enumerate(result.shots, 1):
                shot.shot_id = index
                ids[shot.cue_strike] = index
            for shot in result.shots:
                if shot.linked_live_shot_id in old_strikes:
                    shot.linked_live_shot_id = ids[old_strikes[shot.linked_live_shot_id]]
            for event in result.events:
                old_id = event.metadata.get("shot_id")
                if old_id in old_strikes:
                    event.metadata["shot_id"] = ids[old_strikes[old_id]]
            for shot in result.shots:
                if shot.cue_strike not in old_strikes.values():
                    result.events.extend(
                        [
                            TimelineEvent(
                                event_type="cue_strike",
                                timestamp=shot.cue_strike,
                                confidence=shot.strike_confidence,
                                metadata={"shot_id": shot.shot_id},
                            ),
                            TimelineEvent(
                                event_type="ball_stop",
                                timestamp=shot.physical_stop_timestamp,
                                confidence=shot.stop_confidence,
                                metadata={"shot_id": shot.shot_id},
                            ),
                        ]
                    )
            result.events.sort(key=lambda event: event.timestamp)
        for frame in identified:
            for shot in result.shots:
                if frame["first_strike"] <= shot.cue_strike <= frame["last_strike"]:
                    shot.evidence["frame_context"] = {
                        "frame": frame["frame"],
                        "first": shot.cue_strike == frame["first_strike"],
                        "last": shot.cue_strike == frame["last_strike"],
                    }
                    if shot.cue_strike == frame["first_strike"]:
                        frame["first_shot_id"] = shot.shot_id
                    if shot.cue_strike == frame["last_strike"]:
                        frame["last_shot_id"] = shot.shot_id
        return identified
