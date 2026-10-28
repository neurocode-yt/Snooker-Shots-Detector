"""Measured WST graphics must bracket replay, including a late foul score."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import FrameFeatures, StrikeCandidate

ROOT = Path(__file__).parent / "fixtures/wilson_thepchaiya_native"


def graphic(name):
    return cv2.imread(str(ROOT / f"wst_{name}.png"))


def test_source_wst_open_and_close_require_glyph_shape_and_fullscreen_field():
    for name in ["opening", "closing"]:
        assert len(ViewClassifier.replay_stinger_signature(graphic(name))) == 196
    missing = graphic("opening")
    missing[40:140, 110:210] = (240, 240, 240)
    assert not ViewClassifier.replay_stinger_signature(missing)
    unrelated = graphic("opening")
    cv2.fillPoly(unrelated, [np.array([[160, 58], [124, 122], [195, 122]])], (15, 15, 15))
    assert not ViewClassifier.replay_stinger_signature(unrelated)
    small = np.zeros((180, 320, 3), np.uint8)
    small[:90, :160] = cv2.resize(graphic("opening"), (160, 90))
    assert not ViewClassifier.replay_stinger_signature(small)


@pytest.mark.parametrize(
    "closing,prior,excluded", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_paired_wipes_own_the_duplicate_and_preserve_live_foul(config, closing, prior, excluded):
    signature = ViewClassifier.replay_stinger_signature(graphic("opening"))
    rows = [
        FrameFeatures(
            t=t, appearance_signature=signature if t == 4014.5 or closing and t == 4020 else []
        )
        for t in [4004.6, 4014.5, 4015.48, 4020, 4045]
    ]
    live = StrikeCandidate(timestamp=4004.6, confidence=0.9)
    replay = StrikeCandidate(timestamp=4015.48, confidence=0.9)
    following = StrikeCandidate(timestamp=4045, confidence=0.9)
    candidates = ([live] if prior else []) + [replay, following]
    ReplayDetector(config).mark_candidates(candidates, rows)
    assert replay.possible_replay == excluded
    assert not live.possible_replay and not following.possible_replay


def test_score_recheck_assigns_late_foul_to_live_stroke_across_saved_replay(
    config, tmp_path, monkeypatch
):
    from snooker_ai.pot_detection.selector import PotSelector
    from snooker_ai.types import PotStatus, ShotSelection
    from tests.test_pot_selection import fixture_result, readings, shot

    result = fixture_result(tmp_path)
    result.original_duration = result.metadata.duration = 4100
    result.shots = [shot(4004.6, 4008.6), shot(4015.48, 4019.56), shot(4045, 4050)]
    for i, item in enumerate(result.shots, 1):
        item.shot_id = i
        item.strike_confidence = 0.9
    signature = ViewClassifier.replay_stinger_signature(graphic("opening"))
    markers = [{"t": t, "appearance_signature": signature} for t in [4014.5, 4019.5]]
    monkeypatch.setattr(
        "snooker_ai.pot_detection.selector.cached_source_markers", lambda *args: markers
    )
    samples = readings((28, 63), active=1, start=3995, end=4009)
    samples += readings((35, 63), active=0, start=4022.5, end=4044.5)
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *args: samples)
    selector.apply(result, selection=ShotSelection.POTS_AND_FOULS)
    assert result.shots[0].pot_status is PotStatus.FOUL and result.shots[0].included
    assert result.shots[1].possible_replay and not result.shots[1].included
    assert result.shots[1].pot_status is PotStatus.NO_POT
