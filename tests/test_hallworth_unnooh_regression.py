"""Regressions from native observations in Hallworth–Un-Nooh."""

import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pot_detection.camera_return import recover_cut_contacts
from snooker_ai.pot_detection.frame_context import FrameContextDetector, rack_cut_contact
from snooker_ai.pot_detection.scoreboard import ScoreboardReader
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, ShotRecord
from tests.test_pot_selection import fixture_result, readings

FIXTURES = Path(__file__).parent / "fixtures/classic_hard/hallworth-unnooh"


def observations(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return [FrameFeatures.model_validate(row) for row in data]


def test_contact_survives_cut_in_middle_of_break_off(config, tmp_path):
    features = observations("cut_break")
    coarse = [
        FrameFeatures(t=t, rack_observation_valid=True, red_rack_intact=True)
        for t in (1148.031, 1148.551)
    ]
    support = readings((0, 0), start=1148.5, end=1150.5)
    analyzer = Analyzer(config, tmp_path)
    contacts = analyzer._recover_frame_openings([], features, coarse, 1160, readings=support)
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.timestamp == pytest.approx(1147.76)
    assert SegmentBuilder._candidate_supported(contact)
    assert contact.evidence["opening_departure_diameters"] < 0.45

    # The very small pre-cut displacement needs all eight post-cut observations.
    assert rack_cut_contact([f for f in features if f.t < 1148.08], 1148.031) is None


def test_returns_to_moving_white_on_the_missed_red():
    features = observations("cut_red")
    found = recover_cut_contacts(features, [])
    assert len(found) == 1
    assert found[0].timestamp == pytest.approx(5335.03, abs=0.08)
    assert SegmentBuilder._candidate_supported(found[0])
    assert not recover_cut_contacts(features, [ShotRecord(shot_id=1, cue_strike=5335.03)])


def test_break_off_with_short_native_track_precedes_later_safety(config, tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    result.original_duration = 2100
    result.shots = [
        ShotRecord(
            shot_id=1,
            cue_strike=2050.88,
            evidence={
                "cue_geometry_confirmed": 1.0,
                "cue_ball_motion_confirmed": 0.0,
                "dense_transition_confirmed": 1.0,
                "observation_fps": 25.0,
                "pre_sample_count": 12.0,
                "sustained_count": 4.0,
                "pre_ball_quiet_ratio": 1.0,
                "cue_displacement_diameters": 2.96,
                "cue_direction_consistency": 0.999,
                "initial_launch_displacement": 2.0,
                "pre_cue_address_score": 0.849,
            },
        ),
        ShotRecord(shot_id=2, cue_strike=2070.36),
    ]
    later = result.shots[0].evidence.copy()
    later.update(sustained_count=8.0, pre_cue_address_score=0.0)
    result.shots[1].evidence = later
    detector = FrameContextDetector(config, tmp_path)
    monkeypatch.setattr(detector, "scan_racks", lambda *_: [[]])
    scores = readings((0, 0), start=2055, end=2071) + readings((1, 0), start=2072, end=2100)
    assert detector.identify(result, scores)[0]["first_shot_id"] == 1
    result.shots[0].evidence["pre_cue_address_score"] = 0.0
    assert detector.identify(result, scores)[0]["first_shot_id"] == 2


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("table_handling", True),
        ("broadcast_replay", True),
        ("match_context_valid", False),
    ],
)
def test_camera_return_requires_live_play(attribute, value):
    features = observations("cut_red")
    for feature in features:
        if 5333 <= feature.t <= 5334:
            setattr(feature, attribute, value)
    assert not recover_cut_contacts(features, [])


@pytest.mark.parametrize("stamp,expected", [(45, (0, 0, 1)), (498, (17, 9, 0))])
def test_real_red_turn_arrow_and_score(stamp, expected):
    result = ScoreboardReader().read(cv2.imread(str(FIXTURES / f"score-{stamp}.png")), stamp)
    assert result is not None
    assert (result.left, result.right, result.active_player) == expected
