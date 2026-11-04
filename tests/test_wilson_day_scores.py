"""Real dim outer-score glyphs, panel edges and turn dots from Wilson–Day."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from dataclasses import replace

from snooker_ai.pot_detection.scoreboard import ScoreboardReader
from snooker_ai.pot_detection.selector import classify_shot
from snooker_ai.types import PotStatus, ShotRecord


FIXTURES = Path(__file__).parent / "fixtures" / "wilson_day_scores"


@pytest.mark.parametrize("case", json.loads((FIXTURES / "cases.json").read_text()))
def test_dim_source_scores_and_name_panel_turn_markers(case):
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[550:610, 230:1040] = cv2.imread(str(FIXTURES / case["file"]))
    reading = ScoreboardReader().read(frame, case["t"])
    assert reading is not None
    assert (reading.left, reading.right) == (case["left"], case["right"])
    assert reading.active_player == case["active"]
    assert reading.confidence >= 0.78
    assert reading.layout == "outer_scores"


def test_conflicting_threshold_pairs_cannot_confirm_a_white_score(monkeypatch):
    matches = iter([None, (12, .9), (12, .9), (18, .9), (18, .9), None, None])
    monkeypatch.setattr("snooker_ai.pot_detection.scoreboard.read_digits",
                        lambda *args, **kwargs: next(matches))
    assert ScoreboardReader()._light_digits(np.zeros((23, 36), np.uint8)) is None


def test_one_successful_threshold_cannot_confirm_a_white_score(monkeypatch):
    matches = iter([None, (12, .9), None, None, None, None, None])
    monkeypatch.setattr("snooker_ai.pot_detection.scoreboard.read_digits",
                        lambda *args, **kwargs: next(matches))
    assert ScoreboardReader()._light_digits(np.zeros((23, 36), np.uint8)) is None


def test_real_black_pot_is_selected_when_the_score_changes_from_22_to_29():
    readings = []
    for filename, times in [("5620.png", [9, 9.5]), ("5627.png", [12, 12.5])]:
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[550:610, 230:1040] = cv2.imread(str(FIXTURES / filename))
        reading = ScoreboardReader().read(frame, times[0])
        assert reading is not None
        readings.extend(replace(reading, t=t) for t in times)
    shot = ShotRecord(shot_id=1, cue_strike=10, clip_start=8, clip_end=17,
                      ball_motion_end=15)
    status, evidence = classify_shot(shot, readings, 17)
    assert status is PotStatus.POT
    assert evidence["score_delta"] == [0, 7]
