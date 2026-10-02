"""Replay regressions for ordinary broadcast cuts and verified duplicates."""

from __future__ import annotations

import math

import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


def _candidate(timestamp: float, **kwargs) -> StrikeCandidate:
    return StrikeCandidate(
        timestamp=timestamp, confidence=0.9, camera_view=CameraViewType.MAIN_TABLE, **kwargs,
    )


def _shot_features(
    strike: float, *, scene: int = 0, layout_offset: float = 0.0,
    static: bool = False, layouts: bool = True, partial: bool = False,
    handling: bool = False, cut: bool = False,
) -> list[FrameFeatures]:
    result = []
    for sample in range(-4, 13):
        dt = sample / 4.0
        motion = max(0.0, 0.8 * math.exp(-max(0.0, dt) * 0.7)) if dt >= 0 else 0.0
        travel = 0.0 if static else max(0.0, dt) * 0.13
        layout = [
            0.12 + travel + layout_offset, 0.3 + 0.02 * math.sin(travel * 4),
            0.16, 0.3, 0.16, 0.7, 0.16, 0.5, 0.5, 0.5, 0.72, 0.5, 0.91, 0.5,
        ]
        result.append(FrameFeatures(
            t=strike + dt, table_confidence=0.95, table_observable=True,
            observation_valid=True, table_full_view=not partial,
            table_handling=handling, camera_scene_id=scene,
            view_type=CameraViewType.MAIN_TABLE, motion_score=motion,
            scene_cut_score=0.9 if cut and sample == -4 else 0.0,
            ball_layout_signature=layout if layouts else [],
            # Identical scoreboard/name appearance never proves a replay.
            appearance_signature=[0.2, 0.7, 0.1, 0.4],
        ))
    return result


def test_normal_camera_changes_and_similar_motion_do_not_remove_live_shots(config):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10, layouts=False)
    features += _shot_features(30, scene=2, layouts=False, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not first.possible_replay
    assert not second.possible_replay


def test_different_layout_with_identical_motion_and_scoreboard_is_live(config):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10) + _shot_features(30, scene=3, layout_offset=0.15, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay
    assert "replay_match_to" not in second.evidence


def test_repeated_preparation_and_ball_trajectory_after_transition_is_replay(config):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10) + _shot_features(30, scene=4, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not first.possible_replay
    assert second.possible_replay
    assert second.evidence["replay_match_to"] == 10.0
    assert second.evidence["replay_signature_confirmed"] == 1.0
    assert second.evidence["replay_layout_confirmed"] == 1.0


def test_repeated_still_colour_spots_do_not_prove_a_duplicate(config):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10, static=True) + _shot_features(30, static=True, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay


@pytest.mark.parametrize("partial,handling", [(True, False), (False, True)])
def test_partial_views_and_referee_handling_cannot_supply_a_replay_layout(
    config, partial, handling,
):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10)
    features += _shot_features(30, partial=partial, handling=handling, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay


def test_lone_nearby_replay_view_does_not_mark_a_live_strike(config):
    candidate = _candidate(10)
    features = _shot_features(10)
    features[2].view_type = CameraViewType.REPLAY  # a preceding transition frame

    ReplayDetector(config).mark_candidates([candidate], features)

    assert not candidate.possible_replay


def test_explicit_broadcast_replay_marker_supports_an_alternate_angle(config):
    candidate = _candidate(10)
    features = _shot_features(10, partial=True, layouts=False)
    features[4].broadcast_replay = True

    ReplayDetector(config).mark_candidates([candidate], features)

    assert candidate.possible_replay
    assert candidate.evidence["replay_broadcast_marker"] == 1.0


def test_trusted_explicit_candidate_view_remains_a_replay(config):
    candidate = StrikeCandidate(
        timestamp=10, confidence=0.9, camera_view=CameraViewType.SLOW_MOTION_REPLAY,
    )

    ReplayDetector(config).mark_candidates([candidate], _shot_features(10, partial=True))

    assert candidate.possible_replay


def test_old_motion_only_replay_flags_are_recomputed(config):
    first = _candidate(10)
    second = _candidate(30, possible_replay=True, evidence={
        "replay_match_to": 10.0, "replay_signature_confirmed": 1.0,
    })
    features = _shot_features(10, layouts=False)
    features += _shot_features(30, cut=True, layouts=False)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay
    assert not second.evidence


def test_duplicate_association_is_temporal_without_reordering_caller_list(config):
    first, second = _candidate(10), _candidate(30)
    candidates = [second, first]
    features = _shot_features(30, cut=True) + _shot_features(10)

    returned = ReplayDetector(config).mark_candidates(candidates, features)

    assert returned is candidates
    assert returned[0] is second
    assert second.possible_replay
    assert not first.possible_replay


def test_old_matches_outside_replay_lookback_do_not_link(config):
    first, second = _candidate(10), _candidate(150)
    features = _shot_features(10) + _shot_features(150, cut=True)

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay


def test_another_match_cannot_supply_duplicate_ball_layouts(config):
    first, second = _candidate(10), _candidate(30)
    features = _shot_features(10) + _shot_features(30, cut=True)
    for feature in features:
        if feature.t >= 29:
            feature.match_context_valid = False

    ReplayDetector(config).mark_candidates([first, second], features)

    assert not second.possible_replay
