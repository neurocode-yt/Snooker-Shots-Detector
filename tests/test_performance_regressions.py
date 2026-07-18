"""Regression tests for full-match performance safeguards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snooker_ai.event_fusion.ball_stop import StopDetection
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.scene_detection.detector import SceneDetector
from snooker_ai.types import (
    CameraViewType,
    FrameFeatures,
    SceneSegment,
    StrikeCandidate,
)
from snooker_ai.utils.timebase import TimeMapper


class _CountingFeatures(list[FrameFeatures]):
    """Count full-list iteration while allowing normal indexed window slices."""

    yielded = 0

    def __iter__(self):
        for item in super().__iter__():
            self.yielded += 1
            yield item


def test_strike_scoring_does_not_rescan_full_match_for_every_frame(config):
    count = 600
    features = _CountingFeatures(
        FrameFeatures(
            t=i / 10,
            table_confidence=0.9,
            table_observable=True,
            observation_valid=True,
            ball_diameter_px=10.0,
            ball_count=8,
            cue_ball_detected=True,
            cue_ball_x=100.0,
            cue_ball_y=80.0,
            cue_ball_track_confidence=0.9,
        )
        for i in range(count)
    )

    detector = StrikeDetector(config)
    detector.score_frames(features)
    detector.detect_candidates(features)

    # A handful of setup/result passes are fine. The former implementation
    # yielded roughly 4 * count^2 items through per-frame window scans.
    assert features.yielded < count * 20


def test_scene_stream_retains_compact_observations_only(config, synthetic_green_frame):
    stream = SceneDetector(config).start_stream()
    for index in range(8):
        stream.observe(synthetic_green_frame, index * 0.5)

    assert len(stream.observations) == 8
    for observation in stream.observations:
        assert not any(
            isinstance(value, np.ndarray) for value in vars(observation).values()
        )
    # Only the two histograms required for fade look-ahead remain resident.
    arrays = [
        value for value in vars(stream).values() if isinstance(value, np.ndarray)
    ]
    assert len(arrays) <= 2


def test_analysis_checkpoints_round_trip_and_reject_stale_signatures(
    config, tmp_path: Path
):
    analyzer = Analyzer(config, tmp_path / "job")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fingerprint")
    signature = analyzer._analysis_signature(source)
    features = [FrameFeatures(t=1.0, table_confidence=0.9)]
    scenes = [
        SceneSegment(
            start=0.0,
            end=2.0,
            view_type=CameraViewType.MAIN_TABLE,
        )
    ]
    candidates = [StrikeCandidate(timestamp=1.0, confidence=0.9)]

    analyzer._save_coarse_cache(signature, features, scenes, candidates)
    restored = analyzer._load_coarse_cache(signature)
    assert restored is not None
    restored_features, restored_scenes, restored_candidates = restored
    assert restored_features[0].t == 1.0
    assert restored_scenes[0].view_type == CameraViewType.MAIN_TABLE
    assert restored_candidates[0].timestamp == 1.0
    assert analyzer._load_coarse_cache("stale") is None

    analyzer._save_dense_window(signature, 0, 0.0, 2.0, features)
    dense = analyzer._load_dense_window(signature, 0, 0.0, 2.0)
    assert dense is not None and dense[0].t == 1.0
    assert analyzer._load_dense_window(signature, 0, 0.0, 2.1) is None


def test_unresolved_final_candidate_refinement_is_bounded(
    config, tmp_path: Path, monkeypatch
):
    analyzer = Analyzer(config, tmp_path / "job")
    candidate = StrikeCandidate(timestamp=100.0, confidence=0.9)
    coarse = [FrameFeatures(t=99.0), FrameFeatures(t=100.0)]
    calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        analyzer.segmenter.ball_stop,
        "detect_stop",
        lambda *_args, **_kwargs: StopDetection(
            motion_start=100.0,
            last_ball_motion_timestamp=110.0,
            physical_stop_timestamp=110.0,
            stop_confirmation_timestamp=110.0,
            end_confidence=0.1,
            start_confidence=0.5,
            confirmed=False,
            manual_review_required=True,
            reason="max_duration_review_cap",
        ),
    )

    def fake_extract(*_args, start_time=0.0, end_time=None, **_kwargs):
        calls.append((start_time, float(end_time)))
        return [], [], []

    monkeypatch.setattr(analyzer, "_extract_features", fake_extract)
    analyzer._refine_candidate_windows(
        tmp_path / "unused.mp4",
        None,
        TimeMapper(source_duration=3600.0, proxy_duration=3600.0),
        3600.0,
        [candidate],
        coarse,
        resume=False,
    )

    assert calls
    assert max(end for _, end in calls) <= 110.8 + 1e-9


def test_confirmed_shot_refines_strike_and_stop_edges_not_entire_roll(
    config, tmp_path: Path, monkeypatch
):
    analyzer = Analyzer(config, tmp_path / "job")
    candidate = StrikeCandidate(timestamp=100.0, confidence=0.9)
    coarse = [FrameFeatures(t=99.0), FrameFeatures(t=100.0)]
    calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        analyzer.segmenter.ball_stop,
        "detect_stop",
        lambda *_args, **_kwargs: StopDetection(
            motion_start=100.0,
            last_ball_motion_timestamp=104.9,
            physical_stop_timestamp=105.0,
            stop_confirmation_timestamp=105.5,
            end_confidence=0.9,
            start_confidence=0.9,
            confirmed=True,
            reason="confirmed_all_balls_stationary",
        ),
    )

    def fake_extract(*_args, start_time=0.0, end_time=None, **_kwargs):
        calls.append((start_time, float(end_time)))
        return [], [], []

    monkeypatch.setattr(analyzer, "_extract_features", fake_extract)
    analyzer._refine_candidate_windows(
        tmp_path / "unused.mp4",
        None,
        TimeMapper(source_duration=300.0, proxy_duration=300.0),
        300.0,
        [candidate],
        coarse,
        resume=False,
    )

    # Native refinement now decodes one continuous interval from pre-strike
    # context through the physical stop and reacquisition tail.
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(98.0)
    assert calls[0][1] == pytest.approx(107.8)
