"""Focused regressions for the non-negotiable strict clip boundaries."""

from __future__ import annotations

import pytest

from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import CameraViewType, EditMode, FrameFeatures, StrikeCandidate


def _tracked_frame(t: float, *, moving: bool = False, **overrides) -> FrameFeatures:
    values = {
        "t": t,
        "view_type": CameraViewType.MAIN_TABLE,
        "table_confidence": 0.95,
        "table_observable": True,
        "observation_valid": True,
        "ball_diameter_px": 12.0,
        "ball_count": 8,
        "cue_ball_detected": True,
        "cue_ball_track_confidence": 0.95,
        "max_ball_normalized_speed": 0.8 if moving else 0.0,
        "moving_ball_count": 1 if moving else 0,
        "occluded_ball_count": 0,
        "ball_residual_motion": 0.4 if moving else 0.0,
        "motion_raw": 0.7 if moving else 0.02,
        "motion_score": 0.7 if moving else 0.02,
    }
    values.update(overrides)
    return FrameFeatures(**values)


def _sequence(
    *,
    strike_t: float = 1.0,
    moving_through: float = 1.4,
    end_t: float = 3.0,
) -> list[FrameFeatures]:
    return [
        _tracked_frame(round(i / 10, 10), moving=strike_t <= i / 10 <= moving_through)
        for i in range(int(end_t * 10) + 1)
    ]


def test_short_shot_keeps_strike_visible_when_stop_minus_two_precedes_it(config):
    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence()

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.0)
    assert stop.confirmed is True
    assert stop.last_ball_motion_timestamp == pytest.approx(1.4)
    assert stop.physical_stop_timestamp == pytest.approx(1.5)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.0)

    shots = SegmentBuilder(config).build([strike], features, 3.0, EditMode.STRICT)
    assert len(shots) == 1
    shot = shots[0]
    assert shot.physical_stop_timestamp == pytest.approx(1.5)
    assert shot.clip_end == pytest.approx(1.1)
    assert shot.clip_end_timestamp == pytest.approx(1.1)
    assert shot.stop_confirmation_timestamp == pytest.approx(2.0)


def test_low_global_flow_cannot_end_a_visibly_rolling_ball(config):
    # Small balls occupy too little cloth area to raise aggregate flow, and
    # duplicate broadcast frames produce intermittent zero residual flow.
    features = [
        _tracked_frame(
            i / 10, moving=1 <= i / 10 < 4,
            motion_raw=0.01, residual_motion_max=0.01,
            ball_residual_motion=0.4 if 1 <= i / 10 < 4 and i % 2 else 0.0,
        )
        for i in range(56)
    ]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=5.5
    )
    assert stop.confirmed
    assert stop.physical_stop_timestamp == pytest.approx(4.0)


def test_intermittent_slow_roll_cannot_accumulate_false_stillness(config):
    features = [
        _tracked_frame(
            i / 10, moving=10 <= i <= 14 or (15 <= i < 40 and i % 2 == 1),
            ball_kinematics_valid=True,
        )
        for i in range(56)
    ]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=5.5,
    )
    assert stop.confirmed
    assert stop.physical_stop_timestamp == pytest.approx(4.0)
    assert stop.stop_confirmation_timestamp == pytest.approx(4.5)


def test_measured_stationary_tracks_override_codec_shimmer(config):
    features = [
        _tracked_frame(
            i / 10, moving=1 <= i / 10 < 4,
            ball_kinematics_valid=True, max_ball_speed=100,
            ball_residual_motion=1.0, motion_raw=0.6,
        )
        for i in range(56)
    ]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=5.5,
    )
    assert stop.confirmed
    assert stop.physical_stop_timestamp == pytest.approx(4.0)


def test_verified_native_stop_survives_later_proposal_track_noise(config):
    strike = StrikeCandidate(timestamp=2, confidence=0.95, evidence={
        "refined_ball_motion_start": 2.03, "refined_last_motion_timestamp": 6.97,
        "refined_stop_timestamp": 7, "refined_stop_confirmation_timestamp": 7.5,
        "refined_stop_confidence": 0.95,
    })
    # Another seek/tracker may add noisy movement to the diagnostic timeline.
    features = [_tracked_frame(i / 10, moving=i >= 20) for i in range(101)]
    shots = SegmentBuilder(config).build([strike], features, duration=10)
    assert len(shots) == 1
    assert shots[0].clip_end == 5
    assert shots[0].physical_stop_timestamp == 7
    assert shots[0].evidence["stop_reason"] == "confirmed_native_stop"


def test_incomplete_native_stillness_cannot_override_live_motion(config):
    strike = StrikeCandidate(timestamp=2, confidence=0.95, evidence={
        "refined_stop_timestamp": 7, "refined_stop_confirmation_timestamp": 7.1,
        "refined_stop_confidence": 0.95,
    })
    features = [_tracked_frame(i / 10, moving=i >= 20) for i in range(101)]
    shots = SegmentBuilder(config).build([strike], features, duration=10)
    assert shots[0].clip_end == 10
    assert shots[0].evidence["stop_confirmed"] is False


@pytest.mark.parametrize("moving_through,expected_end", [(1.4, 1.1), (4.9, 3.0)])
def test_stop_minus_two_survives_serialization_and_export_validation(config, moving_through, expected_end):
    from snooker_ai.types import ShotRecord

    shots = SegmentBuilder(config).build(
        [StrikeCandidate(timestamp=1, confidence=0.95)],
        _sequence(moving_through=moving_through, end_t=7), duration=7,
    )
    restored = ShotRecord.model_validate_json(shots[0].model_dump_json())
    assert restored.clip_end == pytest.approx(expected_end)
    assert restored.physical_stop_timestamp == pytest.approx(moving_through + 0.1)
    Exporter(config)._validate_strict_boundaries([restored], source_duration=7, source_fps=30)


def test_unconfirmed_source_end_is_not_shortened(config):
    shots = SegmentBuilder(config).build(
        [StrikeCandidate(timestamp=1, confidence=0.95)],
        _sequence(moving_through=7, end_t=7), duration=7,
    )
    assert shots[0].clip_end == 7
    assert shots[0].evidence["end_before_ball_stop_seconds"] == 0
    Exporter(config)._validate_strict_boundaries(shots, source_duration=7, source_fps=30)


@pytest.mark.parametrize("shot_duration,trim", [
    (1.0, 2.0), (4.999, 2.0), (5.0, 2.0), (6.0, 2.0),
    (6.999, 2.0), (7.0, 3.0), (7.001, 3.0), (12.0, 3.0),
])
def test_duration_based_trim_survives_build_serialization_and_export(config, shot_duration, trim):
    from snooker_ai.types import ShotRecord

    strike_t = 3.2
    physical_stop = strike_t + shot_duration
    candidate = StrikeCandidate(timestamp=strike_t, confidence=0.95, evidence={
        "refined_stop_timestamp": physical_stop,
        "refined_stop_confirmation_timestamp": physical_stop + 0.5,
        "refined_stop_confidence": 0.95,
    })
    shot = SegmentBuilder(config).build([candidate], [], duration=30)[0]
    restored = ShotRecord.model_validate_json(shot.model_dump_json())
    assert restored.evidence["end_before_ball_stop_seconds"] == trim
    assert restored.evidence["shot_duration_for_end_trim_seconds"] == pytest.approx(shot_duration)
    assert restored.clip_start == pytest.approx(strike_t - 2)
    assert restored.clip_end == pytest.approx(max(strike_t + 0.1, physical_stop - trim))
    assert restored.physical_stop_timestamp == pytest.approx(physical_stop)
    assert restored.stop_confirmation_timestamp == pytest.approx(physical_stop + 0.5)
    Exporter(config)._validate_strict_boundaries([restored], source_duration=30, source_fps=30)


def test_camera_cut_during_confirmation_cannot_prove_a_stop(config):
    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence()
    by_time = {frame.t: frame for frame in features}
    # The first tentative still interval is 1.5--1.7.  A cut makes the next
    # observation unrelated, so confirmation must restart in the new view.
    by_time[1.8].scene_cut_score = 1.0

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.0)
    assert stop.confirmed is True
    assert stop.physical_stop_timestamp == pytest.approx(1.9)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.4)
    assert stop.physical_stop_timestamp != pytest.approx(1.8)


def test_missing_frames_cannot_confirm_stillness(config):
    features = [f for f in _sequence(end_t=4) if f.t <= 1.6 or f.t >= 3]
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=4,
    )
    assert stop.confirmed
    assert stop.physical_stop_timestamp == pytest.approx(3)
    assert stop.stop_confirmation_timestamp == pytest.approx(3.5)
    assert stop.manual_review_required


def test_cut_breaks_stale_occlusion_confirmation(config):
    features = _sequence()
    for frame in features:
        if frame.t >= 1.5:
            frame.occluded_ball_count = 1
        if frame.t == 1.8:
            frame.scene_cut_score = 1
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=3,
    )
    assert stop.confirmed
    assert stop.physical_stop_timestamp == pytest.approx(1.9)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.4)


def test_cut_breaks_consecutive_motion_samples(config):
    features = _sequence(moving_through=0)
    for i, frame in enumerate(features):
        if frame.t in (1, 1.2):
            features[i] = _tracked_frame(frame.t, moving=True)
        elif frame.t == 1.1:
            frame.scene_cut_score = 1
    stop = BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), features, duration=3,
    )
    assert not stop.confirmed
    assert stop.reason == "no_sustained_ball_motion"


def test_stationary_false_candidate_is_rejected_before_duration_cap(config):
    config._data["ball_stop"]["max_seconds_after_strike"] = 7
    features = _sequence(moving_through=0, end_t=12)
    candidate = StrikeCandidate(timestamp=1, confidence=0.95)
    assert SegmentBuilder(config).build([candidate], features, duration=12) == []


def test_occluded_ball_keeps_shot_open_until_it_reappears_stationary(config):
    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence()
    for frame in features:
        if 1.5 <= frame.t <= 1.8:
            frame.occluded_ball_count = 1
            frame.ball_count = 7

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.0)
    assert stop.confirmed is True
    assert stop.last_ball_motion_timestamp == pytest.approx(1.8)
    assert stop.physical_stop_timestamp == pytest.approx(1.9)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.4)


def test_persistent_small_occlusion_clears_after_table_is_globally_quiet(config):
    """Stale player-edge tracks must not stretch a settled shot to the cap."""

    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence(end_t=3.0)
    for frame in features:
        if frame.t >= 1.5:
            frame.occluded_ball_count = 2
            frame.ball_count = 7

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.0)
    assert stop.confirmed is True
    assert stop.physical_stop_timestamp == pytest.approx(1.5)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.0)
    assert stop.manual_review_required is True
    assert stop.reason == "confirmed_stale_occlusion_override"


def test_isolated_tracking_spike_does_not_extend_shot_into_referee_respot(config):
    """A one-frame player/referee edge must not reopen settled ball motion."""

    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence(moving_through=1.4, end_t=3.5)
    spike = next(frame for frame in features if frame.t == 1.8)
    spike.max_ball_normalized_speed = 0.9
    spike.moving_ball_count = 1
    spike.ball_residual_motion = 0.4
    spike.motion_raw = 0.7
    spike.motion_score = 0.7

    # The referee enters only after the original stop should be confirmed.
    for frame in features:
        if frame.t >= 2.4:
            frame.residual_motion_mean = 1.5
            frame.residual_motion_max = 8.0
            frame.motion_area_ratio = 0.12

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.5)
    assert stop.confirmed is True
    assert stop.physical_stop_timestamp == pytest.approx(1.5)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.0)
    assert stop.physical_stop_timestamp < 2.4


def test_sustained_renewed_ball_motion_reopens_stop_confirmation(config):
    """Two adjacent motion observations must still preserve a genuine roll."""

    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = _sequence(moving_through=1.4, end_t=3.5)
    by_time = {frame.t: frame for frame in features}
    for t in (1.8, 1.9):
        by_time[t] = _tracked_frame(t, moving=True)
    features = sorted(by_time.values(), key=lambda frame: frame.t)

    stop = BallStopDetector(config).detect_stop(strike, features, duration=3.5)
    assert stop.confirmed is True
    assert stop.last_ball_motion_timestamp == pytest.approx(1.9)
    assert stop.physical_stop_timestamp == pytest.approx(2.0)
    assert stop.stop_confirmation_timestamp == pytest.approx(2.5)


def test_unresolved_long_roll_is_capped_and_requires_review(config):
    config._data["ball_stop"]["max_seconds_after_strike"] = 7.0
    config._data["modes"]["strict"]["max_seconds_after_strike"] = 7.0
    config._data["modes"]["strict"]["max_clip_seconds"] = 9.0
    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = [
        _tracked_frame(round(i / 10, 10), moving=i / 10 >= 1.0)
        for i in range(201)
    ]

    stop = BallStopDetector(config).detect_stop(strike, features, duration=20.0)
    assert stop.confirmed is False
    assert stop.physical_stop_timestamp == pytest.approx(8.0)
    assert stop.stop_confirmation_timestamp == pytest.approx(8.0)
    assert stop.manual_review_required is True

    shots = SegmentBuilder(config).build([strike], features, 20.0, EditMode.STRICT)
    assert len(shots) == 1
    # The unresolved roll runs to the seven-second review horizon instead of
    # being truncated at strike+4.
    assert shots[0].clip_end == pytest.approx(8.0)
    assert shots[0].evidence["stop_reason"] == "max_duration_review_cap"
    assert shots[0].manual_review_required is True


@pytest.mark.parametrize(
    ("strike_t", "expected_start"),
    [(5.25, 3.25), (0.65, 0.0)],
)
def test_strict_start_is_exactly_two_seconds_before_strike_or_source_zero(
    config, strike_t: float, expected_start: float
):
    strike = StrikeCandidate(timestamp=strike_t, confidence=0.95)
    # Use 20 Hz so the non-round strike timestamp itself is represented.
    features = [
        _tracked_frame(
            round(i / 20, 10),
            moving=strike_t <= i / 20 <= strike_t + 0.5,
        )
        for i in range(int((strike_t + 2.0) * 20) + 1)
    ]

    shots = SegmentBuilder(config).build(
        [strike], features, strike_t + 2.0, EditMode.STRICT
    )
    assert len(shots) == 1
    shot = shots[0]
    assert shot.cue_strike == pytest.approx(strike_t)
    assert shot.clip_start == pytest.approx(expected_start, abs=1e-12)
    assert shot.clip_start_timestamp == pytest.approx(expected_start, abs=1e-12)
    if strike_t >= 2.0:
        assert shot.cue_strike - shot.clip_start == pytest.approx(2.0, abs=1e-12)
