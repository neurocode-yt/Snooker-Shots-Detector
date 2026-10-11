"""Recheck source table coverage for a measured roll reacquired after occlusion."""

from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.segmentation.visual_tail import _TailPixels
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.video import open_capture


def raw_visibility(features: list[FrameFeatures]) -> list[FrameFeatures]:
    # A later result policy must remeasure derived geometry instead of trusting
    # a previous policy's conclusion as if it came from the raw feature cache.
    return [
        f.model_copy(
            update={"table_full_view": False, "native_contact_coverage_revalidated": False}
        )
        if f.native_contact_coverage_revalidated
        else f
        for f in features
    ]


class ContactVisibilityDetector:
    def __init__(self, config: Config):
        self.config = config

    def revalidate(
        self,
        source: str | Path,
        features: list[FrameFeatures],
        candidates: list[StrikeCandidate],
    ) -> list[FrameFeatures]:
        ordered = sorted(raw_visibility(features), key=lambda f: f.t)
        times = [f.t for f in ordered]
        candidate_times = sorted(candidate.timestamp for candidate in candidates)
        eligible = []
        for i, current in enumerate(ordered):
            proposal_index = bisect_right(candidate_times, current.t - 2)
            proposal_at_contact = (
                proposal_index < len(candidate_times)
                and candidate_times[proposal_index] < current.t + 2
            )
            if (
                i == 0
                or current.table_full_view
                or current.view_type != CameraViewType.MAIN_TABLE
                or not current.observation_valid
                or not current.match_context_valid
                or current.broadcast_replay
                or current.table_handling
                or current.observation_fps < 10
                or not current.cue_ball_quality
                or not current.cue_ball_detected
                or current.cue_ball_track_confidence < 0.65
                or current.camera_motion_magnitude > 3
                or current.cue_ball_normalized_speed < 3
                or not proposal_at_contact
                or (
                    ordered[i - 1].cue_ball_detected
                    and ordered[i - 1].cue_ball_quality
                    and ordered[i - 1].cue_ball_track_confidence >= 0.65
                )
            ):
                continue
            pre = ordered[bisect_left(times, current.t - 1.25) : i]
            anchors = [
                f
                for f in pre
                if f.cue_ball_quality
                and f.cue_ball_detected
                and f.cue_ball_track_confidence >= 0.65
                and f.cue_ball_x is not None
                and f.cue_ball_y is not None
                and f.camera_scene_id == current.camera_scene_id
                and (f.cue_ball_stable_normalized_speed or 0.0) <= 0.35
            ]
            if len(anchors) < 3 or anchors[-1].t - anchors[0].t < 0.079:
                continue
            xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in anchors])
            diameter = max(1.0, float(np.median([f.ball_diameter_px for f in anchors])))
            if np.max(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)) / diameter > 0.15:
                continue
            eligible.append((i, current))
        if not eligible:
            return ordered
        capture = open_capture(source)
        try:
            if not capture.isOpened():
                return ordered
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            for i, current in eligible:
                pixels = _TailPixels(self.config)
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, current.t - 0.12) * 1000)
                measured = []
                previous = None
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    if not np.isfinite(t) or previous is not None and not 0 < t - previous <= 0.10:
                        measured = []
                        break
                    previous = t
                    if t > current.t + 0.09:
                        break
                    at = bisect_left(times, t)
                    context = min(
                        ordered[max(0, at - 1) : at + 1], key=lambda f: abs(f.t - t), default=None
                    )
                    if context is None or abs(context.t - t) > 0.51 / fps:
                        measured = []
                        break
                    observation = pixels.observe(frame, t, context)
                    if t >= current.t - 0.51 / fps:
                        measured.append(observation)
                if (
                    len(measured) >= 3
                    and measured[-1].t - measured[0].t >= 0.079
                    and all(row.full_table for row in measured)
                ):
                    ordered[i] = current.model_copy(
                        update={
                            "table_full_view": True,
                            "native_contact_coverage_revalidated": True,
                        }
                    )
        finally:
            capture.release()
        return ordered
