"""Hard-cut / fade / dissolve detection via histogram differences."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import CameraViewType, SceneSegment
from snooker_ai.utils.logging import get_logger

logger = get_logger("scene_detection")


@dataclass
class SceneObservation:
    """Compact scene sample retained by the streaming detector.

    A full 71-minute match produces thousands of scene samples.  Keeping the
    original 960x540 BGR arrays consumed more than 12 GiB; these scalar records
    contain everything needed to build the same scene timeline.
    """

    timestamp: float
    cut_score: float
    view_type: CameraViewType
    table_ratio: float
    is_replay_candidate: bool = False
    cut_confirmed: bool = False


class SceneObservationStream:
    """Incrementally compute scene evidence while retaining only two histograms."""

    def __init__(self, detector: "SceneDetector"):
        self.detector = detector
        self.observations: list[SceneObservation] = []
        self._previous_hist: np.ndarray | None = None
        self._two_back_hist: np.ndarray | None = None

    def observe(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        histogram: np.ndarray | None = None,
    ) -> SceneObservation:
        # Feature extraction already computes this histogram for online cut
        # handling; accepting it avoids a duplicate full-frame HSV conversion.
        if histogram is None:
            histogram = self.detector.histogram(frame_bgr)
        adjacent = (
            self.detector.cut_score_simple(self._previous_hist, histogram)
            if self._previous_hist is not None
            else 0.0
        )

        # Reproduce the original one-sample look-ahead fade confirmation.  Once
        # sample i+1 arrives, observation i can be updated without retaining its
        # frame or histogram.
        if (
            len(self.observations) >= 1
            and self._two_back_hist is not None
            and self.observations[-1].cut_score < self.detector.hard_cut
            and self.observations[-1].cut_score >= self.detector.fade_cut
        ):
            lookahead = self.detector.cut_score_simple(self._two_back_hist, histogram)
            confirmed = lookahead >= self.detector.fade_cut * 1.2
            self.observations[-1].cut_score = lookahead * 0.8 if confirmed else 0.0
            self.observations[-1].cut_confirmed = confirmed

        view, table_ratio, extra = self.detector.classifier.classify(frame_bgr)
        observation = SceneObservation(
            timestamp=float(timestamp),
            cut_score=float(adjacent),
            view_type=view,
            table_ratio=float(table_ratio),
            is_replay_candidate=bool(extra.get("is_replay_candidate", False)),
            cut_confirmed=bool(adjacent >= self.detector.hard_cut),
        )
        self.observations.append(observation)
        self._two_back_hist = self._previous_hist
        self._previous_hist = histogram
        return observation

    def finish(self, duration: float) -> list[SceneSegment]:
        if (
            self.observations
            and self.detector.fade_cut
            <= self.observations[-1].cut_score
            < self.detector.hard_cut
        ):
            # The legacy detector did not accept a fade on the final sample
            # because no look-ahead frame exists to confirm it.
            self.observations[-1].cut_score = 0.0
        return self.detector.detect_from_observations(self.observations, duration)


class SceneDetector:
    def __init__(self, config: Config):
        self.cfg = config.section("scene_detection")
        self.view_cfg = config.section("camera_view")
        self.hard_cut = float(self.cfg.get("hard_cut_threshold", 0.42))
        self.fade_cut = float(self.cfg.get("fade_threshold", 0.18))
        self.min_scene = float(self.cfg.get("min_scene_seconds", 0.4))
        self.hist_bins = int(self.cfg.get("histogram_bins", 32))
        self.classifier = ViewClassifier(config)

    def start_stream(self) -> SceneObservationStream:
        return SceneObservationStream(self)

    def histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.hist_bins, self.hist_bins, self.hist_bins],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def cut_score(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        # Correlation: 1 = identical, -1 = opposite → convert to distance-like [0,1]
        corr = float(cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL))
        return float(np.clip(1.0 - (corr + 1.0) / 2.0 * 2.0 + 0.5 * (1 - corr), 0.0, 1.0))

    def cut_score_simple(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        corr = float(
            cv2.compareHist(
                hist_a.astype(np.float32).reshape(-1, 1),
                hist_b.astype(np.float32).reshape(-1, 1),
                cv2.HISTCMP_CORREL,
            )
        )
        return float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    def detect_from_frames(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        duration: float,
    ) -> list[SceneSegment]:
        if not frames:
            return []
        stream = self.start_stream()
        for frame, timestamp in zip(frames, timestamps):
            stream.observe(frame, timestamp)
        return stream.finish(duration)

    def detect_from_observations(
        self,
        observations: list[SceneObservation],
        duration: float,
    ) -> list[SceneSegment]:
        """Build segments from compact samples produced during feature extraction."""

        if not observations:
            return []

        cut_times = [
            (item.timestamp, item.cut_score)
            for item in observations
            if item.cut_confirmed and item.cut_score >= self.fade_cut
        ]
        cut_confidence = {timestamp: score for timestamp, score in cut_times}
        observation_times = [item.timestamp for item in observations]

        # Build segments
        boundaries = [0.0] + [t for t, _ in cut_times] + [duration]
        # Deduplicate close boundaries
        cleaned: list[float] = [boundaries[0]]
        for b in boundaries[1:]:
            if b - cleaned[-1] >= self.min_scene * 0.5:
                cleaned.append(b)
        if cleaned[-1] < duration:
            cleaned.append(duration)

        segments: list[SceneSegment] = []
        for i in range(len(cleaned) - 1):
            start, end = cleaned[i], cleaned[i + 1]
            if end - start < self.min_scene and i > 0:
                # Merge short segment into previous
                if segments:
                    segments[-1].end = end
                continue
            # Representative frame near middle
            mid_t = (start + end) / 2.0
            insertion = bisect_left(observation_times, mid_t)
            if insertion <= 0:
                idx = 0
            elif insertion >= len(observation_times):
                idx = len(observation_times) - 1
            else:
                before = insertion - 1
                idx = (
                    before
                    if mid_t - observation_times[before]
                    <= observation_times[insertion] - mid_t
                    else insertion
                )
            representative = observations[idx]
            cut_conf = cut_confidence.get(start, 0.0)
            segments.append(
                SceneSegment(
                    start=start,
                    end=end,
                    view_type=representative.view_type,
                    cut_confidence=cut_conf,
                    table_ratio=representative.table_ratio,
                    is_replay_candidate=representative.is_replay_candidate,
                )
            )

        logger.info("Detected %d camera segments", len(segments))
        return segments


def detect_scenes(
    frames: list[np.ndarray],
    timestamps: list[float],
    duration: float,
    config: Config,
) -> list[SceneSegment]:
    return SceneDetector(config).detect_from_frames(frames, timestamps, duration)


def view_at_time(scenes: list[SceneSegment], t: float) -> CameraViewType:
    for s in scenes:
        if s.start <= t < s.end:
            return s.view_type
    if scenes and t >= scenes[-1].start:
        return scenes[-1].view_type
    return CameraViewType.OTHER
