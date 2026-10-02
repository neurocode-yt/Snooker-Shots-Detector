"""Ball-stop evidence across broadcast views, handling and cutaways."""

from __future__ import annotations

import pytest

from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


def _frame(t: float, *, moving: bool = False, **overrides) -> FrameFeatures:
    values = {
        "t": t,
        "view_type": CameraViewType.MAIN_TABLE,
        "table_confidence": 0.95,
        "table_observable": True,
        "observation_valid": True,
        "table_full_view": True,
        "camera_scene_id": 1,
        "ball_diameter_px": 12.0,
        "ball_count": 8,
        "cue_ball_track_confidence": 0.95,
        "ball_kinematics_valid": True,
        "max_ball_normalized_speed": 0.8 if moving else 0.0,
        "moving_ball_count": 1 if moving else 0,
        "ball_residual_motion": 0.4 if moving else 0.0,
        "motion_raw": 0.7 if moving else 0.02,
        "motion_score": 0.7 if moving else 0.02,
    }
    values.update(overrides)
    return FrameFeatures(**values)


def _sequence(end: float = 5.0, moving_until: float = 1.4):
    return [
        _frame(round(i / 10, 10), moving=1 <= i / 10 <= moving_until)
        for i in range(int(end * 10) + 1)
    ]


def _detect(config, frames, duration=5.0, **kwargs):
    return BallStopDetector(config).detect_stop(
        StrikeCandidate(timestamp=1, confidence=0.95), frames, duration, **kwargs
    )


def test_visible_closeup_motion_can_establish_a_shot(config):
    frames = _sequence(moving_until=2.4)
    for frame in frames:
        if 1 <= frame.t < 2.5:
            frame.table_full_view = False
            frame.view_type = CameraViewType.BALL_CLOSEUP
    result = _detect(config, frames)
    assert result.confirmed
    assert result.motion_start == pytest.approx(1)
    assert result.last_ball_motion_timestamp == pytest.approx(2.4)
    assert result.physical_stop_timestamp == pytest.approx(2.5)
    assert result.reason == "confirmed_stationary_after_unseen_interval_upper_bound"


def test_stationary_closeup_cannot_prove_all_balls_stopped(config):
    config._data["ball_stop"]["max_seconds_after_strike"] = 7
    frames = _sequence(end=10)
    for frame in frames:
        if frame.t >= 1.5:
            frame.table_full_view = False
            frame.view_type = CameraViewType.BALL_CLOSEUP
    result = _detect(config, frames, duration=10)
    assert not result.confirmed
    assert result.last_ball_motion_timestamp == pytest.approx(1.4)
    assert result.reason == "unobserved_all_ball_stop_review_cap"


def test_reacquired_quiet_table_supplies_upper_bound_not_unseen_stop(config):
    frames = _sequence()
    for frame in frames:
        if 1.5 <= frame.t < 3:
            frame.table_full_view = False
    result = _detect(config, frames)
    assert result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(3)
    assert result.stop_confirmation_timestamp == pytest.approx(3.5)
    assert result.last_ball_motion_timestamp == pytest.approx(1.4)
    assert result.manual_review_required
    assert result.end_confidence <= 0.68
    assert result.reason == "confirmed_stationary_after_unseen_interval_upper_bound"


def test_scene_identity_change_breaks_stillness_even_without_cut_score(config):
    frames = _sequence()
    for frame in frames:
        if frame.t >= 1.8:
            frame.camera_scene_id = 2
    result = _detect(config, frames)
    assert result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(1.9)
    assert result.stop_confirmation_timestamp == pytest.approx(2.4)


def test_camera_change_keeps_one_ongoing_motion_interval(config):
    frames = _sequence(moving_until=3)
    for frame in frames:
        if frame.t >= 2:
            frame.camera_scene_id = 2
        if 2 <= frame.t <= 2.4:
            frame.table_full_view = False
    result = _detect(config, frames)
    assert result.confirmed
    assert result.motion_start == pytest.approx(1)
    assert result.last_ball_motion_timestamp == pytest.approx(3)
    assert result.physical_stop_timestamp == pytest.approx(3.1)
    assert result.reason == "confirmed_after_unknown_gap"


def test_verified_resumed_roll_restores_observed_stop_timing(config):
    frames = _sequence()
    for i, frame in enumerate(frames):
        if 1.5 <= frame.t < 2:
            frame.table_full_view = False
        elif frame.t in (2.2, 2.3):
            frames[i] = _frame(frame.t, moving=True)
    result = _detect(config, frames)
    assert result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(2.4)
    assert result.last_ball_motion_timestamp == pytest.approx(2.3)
    assert result.reason == "confirmed_after_unknown_gap"


@pytest.mark.parametrize("invalid_flag", ["match_context_valid", "broadcast_replay"])
def test_other_match_or_replay_cannot_supply_strike_motion(config, invalid_flag):
    frames = _sequence(moving_until=3)
    for frame in frames:
        if frame.t >= 1:
            setattr(frame, invalid_flag, invalid_flag == "broadcast_replay")
    result = _detect(config, frames)
    assert not result.confirmed
    assert result.reason == "no_sustained_ball_motion"


def test_handled_ball_motion_is_an_unconfirmed_edit_boundary(config):
    frames = _sequence(moving_until=3)
    for frame in frames:
        if frame.t >= 1.8:
            frame.table_handling = True
    result = _detect(config, frames)
    assert not result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(1.8)
    assert result.last_ball_motion_timestamp == pytest.approx(1.7)
    assert result.stop_confirmation_timestamp == pytest.approx(2)
    assert result.reason == "unconfirmed_ball_handling_boundary"


def test_one_handling_observation_does_not_end_a_rolling_shot(config):
    frames = _sequence(moving_until=3)
    next(frame for frame in frames if frame.t == 1.8).table_handling = True
    result = _detect(config, frames)
    assert result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(3.1)
    assert result.last_ball_motion_timestamp == pytest.approx(3)


def test_unknown_stop_cannot_borrow_next_strikes_movement(config):
    frames = _sequence(end=8)
    for i, frame in enumerate(frames):
        if 1.5 <= frame.t < 4:
            frame.table_full_view = False
        elif frame.t >= 4:
            frames[i] = _frame(frame.t, moving=frame.t <= 5)
    result = _detect(config, frames, duration=8, next_strike_timestamp=4)
    assert not result.confirmed
    assert result.last_ball_motion_timestamp == pytest.approx(1.4)
    assert result.physical_stop_timestamp < 4
    assert result.reason == "unobserved_stop_before_next_strike"


def test_quiet_partial_view_cannot_clear_occluded_moving_ball(config):
    frames = _sequence()
    for frame in frames:
        if frame.t >= 1.5:
            frame.table_full_view = False
            frame.occluded_ball_count = 1
    result = _detect(config, frames)
    assert not result.confirmed
    assert result.last_ball_motion_timestamp == pytest.approx(1.4)
    assert "unobserved_all_ball_stop" in result.reason


def test_handling_from_another_match_cannot_bound_current_shot(config):
    frames = _sequence()
    for frame in frames:
        if 1.5 <= frame.t < 3:
            frame.table_handling = True
            frame.match_context_valid = False
    result = _detect(config, frames)
    assert result.confirmed
    assert result.physical_stop_timestamp == pytest.approx(3)
    assert result.reason == "confirmed_stationary_after_unseen_interval_upper_bound"


@pytest.mark.parametrize("centres,expected_occluded", [
    ([100, 100.9, 101.5, 100.05], 0),
    ([100, 102, 104, 106], 1),
])
def test_lost_track_retains_same_coherent_speed_as_visible_track(centres, expected_occluded):
    from snooker_ai.tracking.tracker import BallTracker, Track

    track = Track(
        track_id=1, label="object_ball", diameter=10, hits=4,
        positions=[(i / 10, x, 100) for i, x in enumerate(centres)],
        last_t=0.3, velocity=(20, 0), shape_confidence=0.9,
        cloth_surround_confidence=0.9,
    )
    tracker = BallTracker()
    tracker.tracks = [track]
    visible_speed = tracker.stable_track_speed(track)
    tracker._mark_missed(track, 0.4)

    assert track.last_visible_stable_speed / track.diameter == pytest.approx(visible_speed)
    assert tracker.occluded_moving_count(min_normalized_speed=0.1) == expected_occluded


def test_crowded_weak_circle_identity_cannot_prove_moving_red_but_isolated_red_can():
    from snooker_ai.tracking.tracker import BallTracker, Track
    track = Track(track_id=1, label="object_ball", diameter=14, hits=10,
                  positions=[(0, 480, 377), (.2, 495, 373)], last_t=.2,
                  shape_confidence=.58, cloth_surround_confidence=.46)
    tracker = BallTracker()
    tracker.tracks = [track]
    assert tracker.max_normalized_speed() == 0
    track.cloth_surround_confidence = .9
    assert tracker.max_normalized_speed() > 1


def test_ambiguous_red_cluster_uses_pixels_without_borrowing_identity_motion():
    import cv2
    import numpy as np
    from snooker_ai.tracking.tracker import BallTracker, Track
    tracker = BallTracker()
    tracker.tracks = [Track(track_id=1, label="object_ball", diameter=12,
                           positions=[(0, 50, 50), (.2, 58, 50)], hits=2,
                           shape_confidence=.58, cloth_surround_confidence=.46)]
    before = np.full((100, 100), 100, np.uint8)
    cv2.circle(before, (50, 50), 6, 30, -1)
    cv2.circle(before, (62, 50), 6, 30, -1)
    assert not tracker.ambiguous_region_motion(before, before.copy())
    after = before.copy()
    cv2.circle(after, (62, 50), 6, 100, -1)
    cv2.circle(after, (63, 50), 6, 30, -1)
    assert tracker.ambiguous_region_motion(before, after)


def test_visible_cluster_pixel_motion_overrules_stationary_individual_tracks(config):
    frames = _sequence(moving_until=1.4)
    for frame in frames:
        if 1 <= frame.t < 2:
            frame.ambiguous_ball_motion = True
    result = _detect(config, frames)
    assert result.physical_stop_timestamp >= 2


@pytest.mark.parametrize("visible", [True, False])
def test_slow_roll_uses_pixel_evidence_even_during_a_brief_track_hole(visible):
    import cv2
    import numpy as np
    from snooker_ai.tracking.tracker import BallTracker, Track
    tracker = BallTracker()
    track = Track(track_id=1, label="object_ball", diameter=12, hits=6,
                  positions=[(0, 50, 50), (.3, 51.5, 50)], last_t=.3,
                  last_update_t=.4, shape_confidence=.9, cloth_surround_confidence=.9,
                  visible=visible, last_visible_stable_speed=5)
    tracker.tracks = [track]
    before = np.full((100,100), 100, np.uint8)
    cv2.circle(before, (51,50), 6, 30, -1)
    assert not tracker.ambiguous_region_motion(before, before.copy())
    after = np.full_like(before, 100)
    cv2.circle(after, (52,50), 6, 30, -1)
    assert tracker.ambiguous_region_motion(before, after)
    track.last_update_t = 1
    if not visible:
        assert not tracker.ambiguous_region_motion(before, after)


def test_actual_ball_pixel_motion_interrupts_stillness_even_if_circle_is_missing(config):
    frames = _sequence(moving_until=1.4)
    next(f for f in frames if f.t == 1.8).ambiguous_ball_motion = True
    result = _detect(config, frames)
    assert result.physical_stop_timestamp >= 1.9
