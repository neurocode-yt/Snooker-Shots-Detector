"""Contract tests for strict shot state, schema, and configuration foundations."""

from __future__ import annotations

import pytest

from snooker_ai.config import load_config
from snooker_ai.temporal_model.state_machine import ShotStateMachine
from snooker_ai.types import CameraViewType, FrameFeatures, ShotRecord, ShotState


def _frame(t: float, **overrides) -> FrameFeatures:
    values = {
        "t": t,
        "view_type": CameraViewType.MAIN_TABLE,
        "table_confidence": 0.9,
        "table_observable": True,
        "observation_valid": True,
        "ball_diameter_px": 12.0,
        "cue_ball_detected": True,
        "cue_ball_x": 100.0,
        "cue_ball_y": 80.0,
        "cue_ball_track_confidence": 0.95,
        "cue_ball_normalized_speed": 0.0,
        "max_ball_normalized_speed": 0.0,
        "moving_ball_count": 0,
        "occluded_ball_count": 0,
        "ball_residual_motion": 0.0,
        "motion_score": 0.0,
        "strike_score": 0.0,
    }
    values.update(overrides)
    return FrameFeatures(**values)


def _through_active_motion() -> list[FrameFeatures]:
    frames = [_frame(i / 10) for i in range(5)]
    frames.append(_frame(0.5, strike_score=0.9))
    for t in (0.6, 0.7, 0.8, 0.9):
        frames.append(
            _frame(
                t,
                cue_ball_normalized_speed=0.65,
                max_ball_normalized_speed=0.65,
                cue_ball_acceleration=4.0,
                moving_ball_count=2,
                ball_residual_motion=0.4,
            )
        )
    return frames


def test_required_states_exist_without_removing_legacy_states():
    required = {
        "WAITING",
        "CUEING",
        "STRIKE_CANDIDATE",
        "STRIKE_CONFIRMED",
        "BALLS_MOVING",
        "BALLS_SETTLING",
        "ALL_BALLS_STOPPED",
    }
    assert required <= ShotState.__members__.keys()
    assert ShotState.BETWEEN_SHOTS.value == "BETWEEN_SHOTS"
    assert ShotState.BALLS_STOPPED.value == "BALLS_STOPPED"


def test_contract_feature_fields_round_trip():
    feature = _frame(
        2.5,
        cue_ball_speed=24.0,
        cue_ball_normalized_speed=2.0,
        cue_ball_acceleration=10.0,
        max_ball_normalized_speed=2.4,
        moving_ball_count=3,
        occluded_ball_count=1,
        ball_residual_motion=0.7,
    )
    restored = FrameFeatures.model_validate(feature.model_dump())
    assert restored.table_observable is True
    assert restored.observation_valid is True
    assert restored.ball_diameter_px == 12.0
    assert (restored.cue_ball_x, restored.cue_ball_y) == (100.0, 80.0)
    assert restored.cue_ball_normalized_speed == 2.0
    assert restored.max_ball_normalized_speed == 2.4
    assert restored.moving_ball_count == 3
    assert restored.occluded_ball_count == 1
    assert restored.ball_residual_motion == 0.7


def test_shot_record_mirrors_legacy_and_contract_fields():
    legacy = ShotRecord(
        shot_id=1,
        ball_motion_end=8.4,
        shot_confidence=0.91,
        end_confidence=0.87,
    )
    assert legacy.last_ball_motion_timestamp == 8.4
    assert legacy.physical_stop_timestamp == 8.4
    assert legacy.stop_confirmation_timestamp == 8.4
    assert legacy.strike_confidence == 0.91
    assert legacy.stop_confidence == 0.87

    contract = ShotRecord(
        shot_id=2,
        last_ball_motion_timestamp=12.20,
        physical_stop_timestamp=12.24,
        stop_confirmation_timestamp=12.74,
        strike_confidence=0.96,
        stop_confidence=0.94,
    )
    assert contract.ball_motion_end == 12.24
    assert contract.shot_confidence == 0.96
    assert contract.end_confidence == 0.94


def test_strict_config_expresses_exact_boundary_contract():
    config = load_config()
    strict = config.mode_settings("strict")
    assert strict["pre_roll"] == 2.0
    assert strict["post_roll"] == 0.0
    assert strict["min_seconds_after_strike"] == 4.0
    assert strict["max_seconds_after_strike"] == 4.0
    assert strict["max_clip_seconds"] == 12.0
    assert config.get("ball_stop.confirm_seconds") == 0.50
    assert config.get("ball_stop.settle_seconds") == 0.50
    assert config.get("ball_stop.end_pad_seconds") == 0.0
    assert (
        config.get("motion.normalized_motion_start_threshold")
        > config.get("motion.normalized_motion_stop_threshold")
    )
    assert config.get("strike_fusion.cue_ball_stationary_seconds") > 0.0
    assert config.get("strike_fusion.cue_ball_sustained_motion_seconds") > 0.0


def test_state_machine_emits_required_sequence():
    frames = _through_active_motion()
    frames.extend(_frame(t) for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6))
    labelled = ShotStateMachine(load_config()).label(frames)
    states = [frame.state for frame in labelled]

    required = [
        ShotState.WAITING,
        ShotState.CUEING,
        ShotState.STRIKE_CANDIDATE,
        ShotState.STRIKE_CONFIRMED,
        ShotState.BALLS_MOVING,
        ShotState.BALLS_SETTLING,
        ShotState.ALL_BALLS_STOPPED,
    ]
    positions = [states.index(state) for state in required]
    assert positions == sorted(positions)


def test_normalized_motion_uses_start_stop_hysteresis():
    frames = _through_active_motion()
    # 0.12 is below the 0.18 start threshold but above the 0.08 stop threshold.
    frames.append(_frame(1.0, max_ball_normalized_speed=0.12))
    frames.append(_frame(1.1, max_ball_normalized_speed=0.05))
    labelled = ShotStateMachine(load_config()).label(frames)
    assert labelled[-2].state == ShotState.BALLS_MOVING
    assert labelled[-1].state == ShotState.BALLS_SETTLING


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"observation_valid": False},
        {"table_observable": False},
        {"scene_cut_score": 1.0},
        {"occluded_ball_count": 1},
    ],
)
def test_invalid_cut_or_occluded_frames_never_count_as_stopped(invalid_fields):
    frames = _through_active_motion()
    frames.extend((_frame(1.0), _frame(1.1)))
    frames.append(_frame(1.2, **invalid_fields))
    frames.extend(_frame(t) for t in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9))

    labelled = ShotStateMachine(load_config()).label(frames)
    by_time = {frame.t: frame.state for frame in labelled}
    assert by_time[1.2] != ShotState.ALL_BALLS_STOPPED
    assert by_time[1.7] != ShotState.ALL_BALLS_STOPPED
    stopped_at = next(frame.t for frame in labelled if frame.state == ShotState.ALL_BALLS_STOPPED)
    assert stopped_at >= 1.8
