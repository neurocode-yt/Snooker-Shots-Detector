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
    candidate = StrikeCandidate(timestamp=100.0, confidence=0.9, evidence={"dense_transition_confirmed": 1.0})
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
    assert max(end for _, end in calls) <= 160.0 + 1e-9


@pytest.mark.parametrize("native_stop", [105.0, 107.0])
def test_confirmed_shot_refines_strike_and_stop_edges_not_entire_roll(
    config, tmp_path: Path, monkeypatch, native_stop
):
    analyzer = Analyzer(config, tmp_path / "job")
    candidate = StrikeCandidate(timestamp=100.0, confidence=0.9, evidence={"dense_transition_confirmed": 1.0})
    coarse = [FrameFeatures(t=99.0), FrameFeatures(t=100.0)]
    calls: list[tuple[float, float, float]] = []
    decoded_ends = []

    def detect_stop(_candidate, features, *_args, **_kwargs):
        # Native tracking can resolve a later stop than the coarse travel pass.
        end = native_stop if features and features[0].t == 102 else 105.0
        return StopDetection(
            motion_start=100.0,
            last_ball_motion_timestamp=end - 0.1,
            physical_stop_timestamp=end,
            stop_confirmation_timestamp=end + 0.5,
            end_confidence=0.9,
            start_confidence=0.9,
            confirmed=bool(features and features[-1].t >= end + 0.5),
            reason="confirmed_all_balls_stationary",
        )

    monkeypatch.setattr(analyzer.segmenter.ball_stop, "detect_stop", detect_stop)

    def fake_extract(*_args, start_time=0.0, end_time=None, sample_fps=0, stop_when=None, **_kwargs):
        calls.append((start_time, float(end_time), sample_fps))
        features = []
        for i in range(int((float(end_time) - start_time) * sample_fps) + 1):
            features.append(FrameFeatures(t=start_time + i / sample_fps))
            if stop_when and stop_when(features):
                break
        decoded_ends.append(features[-1].t)
        return features, [], []

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

    # Long travel runs at 10fps with early exit; only the two short boundary
    # intervals run at native cadence.
    assert calls[0] == (98.0, 102.0, 30.0)
    assert calls[1] == (98.0, 160.0, 10.0)
    assert calls[2] == (102.0, 160.0, 30.0)
    assert decoded_ends[2] == pytest.approx(native_stop + 0.7)
    assert candidate.evidence["refined_stop_timestamp"] == native_stop


def test_audio_seed_uses_bounded_native_rate_verification_window(
    config, tmp_path: Path, monkeypatch
):
    analyzer = Analyzer(config, tmp_path / "job")
    seed = StrikeCandidate(
        timestamp=100.0,
        confidence=0.6,
        evidence={"audio_seed": 1.0, "audio_peak_score": 0.8},
    )
    calls: list[tuple[float, float, float]] = []

    def fake_extract(
        *_args, sample_fps=None, start_time=0.0, end_time=None, **_kwargs
    ):
        calls.append((start_time, float(end_time), float(sample_fps)))
        return [], [], []

    monkeypatch.setattr(analyzer, "_extract_features", fake_extract)
    analyzer._refine_candidate_windows(
        tmp_path / "unused.mp4",
        None,
        TimeMapper(source_duration=3600.0, proxy_duration=3600.0),
        3600.0,
        [seed],
        [FrameFeatures(t=99.0), FrameFeatures(t=100.0)],
        resume=False,
    )

    assert calls == [(98.5, 106.0, 10.0)]


def test_rejected_contact_does_not_open_a_long_tracking_window(config, tmp_path, monkeypatch):
    analyzer = Analyzer(config, tmp_path / "job")
    calls = []

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return [FrameFeatures(t=98 + i / 30) for i in range(121)], [], []

    monkeypatch.setattr(analyzer, "_extract_features", extract)
    analyzer._refine_candidate_windows(
        tmp_path / "video.mp4", None, TimeMapper(source_duration=3600), 3600,
        [StrikeCandidate(timestamp=100, confidence=0.8)], [], resume=False,
    )
    assert len(calls) == 1
    assert calls[0]["sample_fps"] == 30
    assert calls[0]["end_time"] == 102


def test_rejected_proposal_cannot_overwrite_a_verified_stop(config, tmp_path, monkeypatch):
    analyzer = Analyzer(config, tmp_path / "job")
    monkeypatch.setattr(analyzer.strike_det, "score_frames", lambda features: features)
    monkeypatch.setattr(analyzer.strike_det, "detect_candidates", lambda features: [])

    def confirm(candidates, features):
        for candidate in candidates:
            candidate.evidence["dense_transition_confirmed"] = float(candidate.timestamp == 100)
        return candidates

    monkeypatch.setattr(analyzer.strike_det, "refine_boundaries", confirm)
    monkeypatch.setattr(
        analyzer.segmenter.ball_stop, "detect_stop",
        lambda *args, **kwargs: StopDetection(
            motion_start=100, last_ball_motion_timestamp=104.9,
            physical_stop_timestamp=105, stop_confirmation_timestamp=105.5,
            end_confidence=0.95, start_confidence=0.9, confirmed=True,
        ),
    )

    def extract(*args, start_time=0, end_time=None, sample_fps=30, stop_when=None, **kwargs):
        features = []
        for i in range(int((end_time - start_time) * sample_fps) + 1):
            contaminated = start_time == 102 and end_time == 106
            features.append(FrameFeatures(t=start_time + i / sample_fps, max_ball_normalized_speed=99 if contaminated else 0))
            if stop_when and stop_when(features):
                break
        return features, [], []

    monkeypatch.setattr(analyzer, "_extract_features", extract)
    _, features = analyzer._refine_candidate_windows(
        tmp_path / "video.mp4", None, TimeMapper(source_duration=300), 300,
        [StrikeCandidate(timestamp=100, confidence=0.9), StrikeCandidate(timestamp=104, confidence=0.8)],
        [], resume=False,
    )
    at_stop = [feature for feature in features if 105 <= feature.t <= 105.5]
    assert at_stop
    assert all(feature.max_ball_normalized_speed == 0 for feature in at_stop)


@pytest.mark.parametrize("missing_interval", [False, True])
def test_audio_recovery_reuses_only_complete_native_rate_observations(
    config, tmp_path, monkeypatch, missing_interval
):
    analyzer = Analyzer(config, tmp_path / "job")
    candidate = StrikeCandidate(timestamp=100, confidence=0.8, evidence={"audio_seed": 1.0, "dense_transition_confirmed": 1.0})
    existing = [
        FrameFeatures(t=98 + i / 30)
        for i in range(301)
        if not missing_interval or not 102 < 98 + i / 30 < 103
    ]
    monkeypatch.setattr(
        analyzer.segmenter.ball_stop, "detect_stop",
        lambda *args, **kwargs: StopDetection(
            motion_start=100, last_ball_motion_timestamp=104.9,
            physical_stop_timestamp=105, stop_confirmation_timestamp=105.5,
            end_confidence=0.9, start_confidence=0.9, confirmed=True,
        ),
    )
    calls = []

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return [], [], []

    monkeypatch.setattr(analyzer, "_extract_features", extract)
    _, dense = analyzer._refine_candidate_windows(
        tmp_path / "video.mp4", None,
        TimeMapper(source_duration=300, proxy_duration=300), 300,
        [candidate], existing, resume=False, force_native_audio=True, existing_dense=existing,
    )
    if missing_interval:
        assert len(calls) == 1
        assert calls[0]["sample_fps"] == 10
    else:
        # Reuse contact/travel observations, then verify the stop in a dedicated
        # native window whose tracker was warmed before the tentative boundary.
        assert len(calls) == 1
        assert calls[0]["sample_fps"] == 30
        assert calls[0]["start_time"] == 102
        assert calls[0]["end_time"] == 160
        assert dense and dense[0].t == pytest.approx(98)
