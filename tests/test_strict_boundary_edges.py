"""Focused regressions for the non-negotiable strict clip boundaries."""

from __future__ import annotations

import pytest

from snooker_ai.event_fusion.ball_stop import BallStopDetector
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


def test_stationary_confirmation_is_metadata_inside_minimum_hold(config):
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
    assert shot.clip_end == pytest.approx(3.0)  # source ends before strike+4
    assert shot.clip_end_timestamp == pytest.approx(3.0)
    assert shot.clip_end > shot.stop_confirmation_timestamp


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
    strike = StrikeCandidate(timestamp=1.0, confidence=0.95)
    features = [
        _tracked_frame(round(i / 10, 10), moving=i / 10 >= 1.0)
        for i in range(201)
    ]

    stop = BallStopDetector(config).detect_stop(strike, features, duration=20.0)
    assert stop.confirmed is False
    assert stop.physical_stop_timestamp == pytest.approx(11.0)
    assert stop.stop_confirmation_timestamp == pytest.approx(11.0)
    assert stop.manual_review_required is True

    shots = SegmentBuilder(config).build([strike], features, 20.0, EditMode.STRICT)
    assert len(shots) == 1
    assert shots[0].clip_end == pytest.approx(5.0)
    assert shots[0].evidence["stop_reason"] == "max_seconds_after_strike_review_cap"
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
