"""A measured native six restores the blue's previously unreadable baseline."""

import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.pot_detection import scoreboard
from snooker_ai.pot_detection.selector import PotSelector, classify_shot
from snooker_ai.types import AnalysisResult, PotStatus, ShotRecord, VideoMetadata


FOLDER = Path(__file__).parent / 'fixtures/pot_scoreboards/blue_26_31'
CASE = json.loads((FOLDER / 'source.json').read_text())


def source_readings():
    reader = scoreboard.ScoreboardReader()
    return [reader.read(cv2.imread(str(FOLDER / (f['name'] + '.png'))), f['source_time'])
            for f in CASE['heldout_frames']]


@pytest.mark.parametrize('sample', CASE['heldout_frames'], ids=lambda f: f['name'])
@pytest.mark.parametrize('width', [1280, 1920])
def test_heldout_native_points_and_turn_indicator(sample, width):
    frame = cv2.imread(str(FOLDER / (sample['name'] + '.png')))
    frame = cv2.resize(frame, (width, round(width * 720 / 1280)))
    result = scoreboard.ScoreboardReader().read(frame, sample['source_time'])
    assert result is not None
    assert (result.left, result.right) == tuple(sample['score'])
    assert result.active_player == sample['active_player']
    assert result.confidence >= .78


def test_actual_blue_is_selected_from_stable_26_to_31_source_scores():
    readings = source_readings()
    assert all(r is not None for r in readings)
    shot = ShotRecord(shot_id=150, cue_strike=3996., clip_start=3994., clip_end=3998.48,
                      physical_stop_timestamp=3999.12, included=True)
    status, evidence = classify_shot(shot, readings, 4007.12, baseline_seconds=12)
    assert status is PotStatus.POT
    assert evidence['points'] == 5
    assert evidence['score_before'] == [26, 1]
    assert evidence['score_after'] == [31, 1]


def test_without_native_six_the_actual_blue_baseline_is_unreadable(monkeypatch):
    bank = scoreboard._template_bank()
    assert bank[0][-1] == 6
    with monkeypatch.context() as patch:
        patch.setattr(scoreboard, '_template_bank', lambda: tuple(a[:-1] for a in bank))
        scoreboard._match_digit.cache_clear()
        try:
            readings = source_readings()
            assert all(r is None for r in readings[:3])
            assert all(r is not None for r in readings[3:])
        finally:
            scoreboard._match_digit.cache_clear()


def test_template_changes_invalidate_old_empty_score_cache(config, tmp_path, monkeypatch):
    # Use an isolated asset path to exercise the production cache identity.
    import snooker_ai.pot_detection.selector as module

    asset = tmp_path / 'assets/score_digits.npz'
    asset.parent.mkdir()
    asset.write_bytes(b'old-template-bank')
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'selector.py'))
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'unchanged-source')
    result = AnalysisResult(job_id='blue', source_path=str(source),
                            metadata=VideoMetadata(path=str(source), duration=4100, fps=25))
    selector = PotSelector(config, tmp_path / 'job')
    old = selector._signature(result)
    asset.write_bytes(b'new-template-bank-with-native-six')
    assert selector._signature(result) != old
