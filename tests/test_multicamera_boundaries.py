"""Editing contracts when multi-camera visibility bounds a shot."""

from __future__ import annotations

import pytest

from snooker_ai.rendering.exporter import Exporter
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


def _features(end=12.0, *, handling_from=None, moving_intervals=((1, 1.4),)):
    frames = []
    for i in range(int(end * 10) + 1):
        t = round(i / 10, 10)
        moving = any(start <= t <= stop for start, stop in moving_intervals)
        frames.append(FrameFeatures(
            t=t, view_type=CameraViewType.MAIN_TABLE, table_confidence=0.95,
            ball_diameter_px=12, ball_count=8, ball_kinematics_valid=True,
            max_ball_normalized_speed=0.8 if moving else 0,
            moving_ball_count=int(moving), motion_raw=0.7 if moving else 0.02,
            table_handling=handling_from is not None and t >= handling_from,
        ))
    return frames


def _supported(timestamp, confidence=0.95, **evidence):
    return StrikeCandidate(timestamp=timestamp, confidence=confidence, evidence={
        "dense_transition_confirmed": 1.0, **evidence,
    })


def test_native_stop_beyond_next_supported_strike_cannot_delete_that_strike(config):
    frames = _features(moving_intervals=((1, 1.4), (5, 6)))
    for frame in frames:
        if 1.5 <= frame.t < 5:
            frame.table_full_view = False
    first = _supported(1, refined_ball_motion_start=1,
                       refined_last_motion_timestamp=6,
                       refined_stop_timestamp=6.1,
                       refined_stop_confirmation_timestamp=6.6,
                       refined_stop_confidence=0.95)
    second = _supported(5, confidence=0.9)
    shots = SegmentBuilder(config).build([first, second], frames, duration=12)

    assert [shot.cue_strike for shot in shots] == [1, 5]
    assert shots[0].evidence["stop_confirmed"] is False
    assert shots[0].physical_stop_timestamp < 5
    assert "rejected_mid_motion_strike" not in shots[0].evidence
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


def test_early_handling_limits_available_footage_without_export_error(config):
    frames = _features(handling_from=1.8)
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    assert len(shots) == 1
    shot = shots[0]
    assert shot.clip_end == pytest.approx(1.8)
    assert shot.evidence["usable_source_end_timestamp"] == pytest.approx(1.8)
    assert shot.evidence["stop_reason"] == "unconfirmed_ball_handling_boundary"
    assert shot.evidence["stop_confirmed"] is False
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


def test_overlap_repair_does_not_extend_next_clip_into_handling(config):
    # The first clip occupies the beginning of the next shot's pre-roll, so
    # repairing it creates a new four-second minimum end for that second shot.
    # A later handling boundary remains an absolute availability limit.
    frames = _features(end=12, moving_intervals=((3, 4), (6, 6.5)), handling_from=6.8)
    first = _supported(3, refined_ball_motion_start=3,
                       refined_last_motion_timestamp=5.4,
                       refined_stop_timestamp=5.5,
                       refined_stop_confirmation_timestamp=6,
                       refined_stop_confidence=0.95)
    second = _supported(6, confidence=0.9)
    builder = SegmentBuilder(config)
    shots = builder.build([first], frames, duration=12)
    shots.extend(builder.build([second], frames, duration=12))
    builder._trim_adjacent_windows(shots, source_duration=12)
    assert shots[1].clip_start == pytest.approx(5)
    assert shots[1].evidence["pre_roll_trimmed_seconds"] == pytest.approx(1)
    assert shots[1].clip_end <= 6.8
    assert shots[1].evidence["usable_source_end_timestamp"] == pytest.approx(6.8)
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


@pytest.mark.parametrize("boundary", ["handling", "foreign"])
def test_minimum_padding_after_confirmed_stop_cannot_include_unusable_action(config, boundary):
    frames = _features(handling_from=2.2 if boundary == "handling" else None)
    if boundary == "foreign":
        for frame in frames:
            frame.match_context_valid = frame.t < 2.2
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    assert len(shots) == 1
    assert shots[0].evidence["stop_confirmed"] is True
    assert shots[0].physical_stop_timestamp == pytest.approx(1.5)
    assert shots[0].clip_end == pytest.approx(2.2)
    assert shots[0].evidence["usable_source_end_reason"] == (
        "ball_handling_clip_boundary" if boundary == "handling" else "foreign_match_clip_boundary")
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


@pytest.mark.parametrize("boundary", ["handling", "foreign"])
def test_preparation_after_unusable_action_starts_at_first_usable_frame(config, boundary):
    frames = _features(moving_intervals=((5, 6),))
    for frame in frames:
        if 2 <= frame.t <= 4:
            if boundary == "handling":
                frame.table_handling = True
            else:
                frame.match_context_valid = False
    shots = SegmentBuilder(config).build([_supported(5)], frames, duration=12)
    assert len(shots) == 1
    shot = shots[0]
    assert shot.clip_start == pytest.approx(4.1)
    assert shot.preparation_start == pytest.approx(4.1)
    assert shot.evidence["usable_source_start_timestamp"] == pytest.approx(4.1)
    assert shot.evidence["usable_source_start_reason"] == (
        "ball_handling_clip_boundary" if boundary == "handling" else "foreign_match_clip_boundary")
    assert not any((f.table_handling or not f.match_context_valid)
                   for f in frames if shot.clip_start <= f.t <= shot.clip_end)
    transition_padding = shot.evidence["minimum_clip_transition_padding_seconds"]
    assert shot.duration() - transition_padding >= 4 - 1e-6
    assert shot.clip_end - shot.cue_strike >= 2
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)

    # Export must not silently accept an old pre-roll that reintroduces the
    # known referee/cutaway footage after the segment's aliases are updated.
    shot.clip_start = shot.clip_start_timestamp = 3
    with pytest.raises(ValueError, match="strict start boundary"):
        Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


def test_isolated_handling_flag_does_not_remove_preparation(config):
    frames = _features(moving_intervals=((5, 6),))
    next(frame for frame in frames if frame.t == 4).table_handling = True
    shots = SegmentBuilder(config).build([_supported(5)], frames, duration=12)
    assert shots[0].clip_start == pytest.approx(3)
    assert shots[0].evidence["usable_source_start_timestamp"] == 0


def _non_table_window(frames, start, end, fps=10):
    for frame in frames:
        frame.view_classified = True
        frame.observation_fps = fps
        if start <= frame.t <= end:
            frame.table_observable = False
            frame.observation_valid = False
            frame.table_full_view = False
            frame.view_type = CameraViewType.PLAYER_CLOSEUP
    return frames


def test_two_second_reaction_returns_to_moving_table_without_edit_cap(config):
    frames = _non_table_window(_features(moving_intervals=((1, 8),)), 2, 3.9)
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    shot = shots[0]
    assert shot.evidence["usable_source_end_reason"] == ""
    assert shot.clip_end > 3.9
    assert shot.physical_stop_timestamp == pytest.approx(8.1)
    assert shot.evidence["stop_confirmed"] is True
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


@pytest.mark.parametrize("fps", [2, 10])
def test_long_measured_non_table_cutaway_limits_edit_without_inventing_stop(config, fps):
    frames = _non_table_window(_features(moving_intervals=((1, 9),)), 4.5, 8)
    if fps == 2:
        frames = [frame for frame in frames
                  if not 4.5 <= frame.t <= 8 or round(frame.t * 10) % 5 == 0]
        for frame in frames:
            if 4.5 <= frame.t <= 8:
                frame.observation_fps = fps
        # Cheap coarse observations between native contact/stop windows can
        # prove repeated non-table footage without proving ball stillness.
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    shot = shots[0]
    assert shot.clip_end == pytest.approx(4.5)
    assert shot.evidence["usable_source_end_timestamp"] == pytest.approx(4.5)
    assert shot.evidence["usable_source_end_reason"] == "non_table_cutaway_clip_boundary"
    assert shot.physical_stop_timestamp > shot.clip_end
    assert shot.duration() - shot.evidence["minimum_clip_transition_padding_seconds"] >= 4
    assert shot.clip_end - shot.cue_strike >= 2
    Exporter(config)._validate_strict_boundaries(shots, source_duration=12, source_fps=25)


def test_unobserved_sampling_hole_cannot_confirm_non_table_cutaway(config):
    frames = _non_table_window(_features(moving_intervals=((1, 9),)), 4.5, 8)
    frames = [frame for frame in frames if not 5.1 <= frame.t <= 7.5]
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    assert shots[0].evidence["usable_source_end_reason"] == ""
    assert shots[0].clip_end > 4.5


def test_long_partial_table_view_is_not_a_non_table_cutaway(config):
    frames = _features(moving_intervals=((1, 9),))
    for frame in frames:
        frame.view_classified = True
        frame.observation_fps = 10
        if 4.5 <= frame.t <= 8:
            frame.table_full_view = False
            frame.view_type = CameraViewType.BALL_CLOSEUP
    shots = SegmentBuilder(config).build([_supported(1)], frames, duration=12)
    assert shots[0].evidence["usable_source_end_reason"] == ""
    assert shots[0].physical_stop_timestamp == pytest.approx(9.1)
    assert shots[0].clip_end > 4.5
