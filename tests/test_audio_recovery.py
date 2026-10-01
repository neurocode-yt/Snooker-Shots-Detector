"""Audio peaks must be corroborated by observable ball motion."""

import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


def _window(**overrides):
    return [
        FrameFeatures(
            t=i / 10, table_confidence=0.9, ball_diameter_px=10, ball_count=4,
            motion_raw=0.7 if i >= 10 else 0.0,
            motion_score=0.7 if i >= 10 else 0.0,
            ball_residual_motion=0.5 if i >= 10 else 0.0,
            max_ball_normalized_speed=2 if i >= 10 else 0.0,
            **overrides,
        )
        for i in range(6, 25)
    ]


def test_audio_recovery_accepts_ball_onset():
    assert Analyzer._audio_seed_visual_support(StrikeCandidate(timestamp=1, confidence=0.6), _window())


@pytest.mark.parametrize("overrides", [
    {"table_observable": False}, {"observation_valid": False},
    {"scene_cut_score": 1.0}, {"view_type": CameraViewType.REPLAY},
    {"view_type": CameraViewType.SCOREBOARD}, {"view_type": CameraViewType.ADVERTISEMENT},
])
def test_audio_recovery_rejects_unknown_and_nonplay_views(overrides):
    assert not Analyzer._audio_seed_visual_support(
        StrikeCandidate(timestamp=1, confidence=0.6), _window(**overrides)
    )


def test_audio_recovery_rejects_player_or_graphics_motion_without_ball_onset():
    features = _window()
    for feature in features:
        feature.ball_residual_motion = 0
        feature.max_ball_normalized_speed = 0
    assert not Analyzer._audio_seed_visual_support(StrikeCandidate(timestamp=1, confidence=0.6), features)


def test_audio_recovery_rejects_collision_during_existing_ball_motion():
    features = _window()
    for feature in features:
        feature.ball_residual_motion = 0.8
        feature.max_ball_normalized_speed = 3.0
    assert not Analyzer._audio_seed_visual_support(StrikeCandidate(timestamp=1, confidence=0.6), features)


def test_audio_recovery_requires_observable_prestrike_context():
    features = [feature for feature in _window() if feature.t >= 1]
    assert not Analyzer._audio_seed_visual_support(StrikeCandidate(timestamp=1, confidence=0.6), features)
