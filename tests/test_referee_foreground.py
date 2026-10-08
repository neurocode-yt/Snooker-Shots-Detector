"""Original-source referee entries cap footage before respotting begins."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.segmentation.visual_tail import HandComponent, TailObservation, _confirmed_entries
from snooker_ai.types import FrameFeatures, ShotRecord, StrikeCandidate

CASES = json.loads((Path(__file__).parent/'fixtures/referee_foreground_windows.json').read_text())['cases']


def rows_for(name):
    return [TailObservation(
        t=r['t'], full_table=r['full_table'], reset=r['reset'],
        context=FrameFeatures.model_validate(r['context']) if r['context'] else None,
        components=[HandComponent(**c) for c in r['components']],
    ) for r in CASES[name]['rows']]


@pytest.mark.parametrize('name', CASES)
def test_original_source_foreground_crossing_survives_confident_stop(config, tmp_path, name):
    case = CASES[name]
    assert case['expected_entry'] is not None
    rows = rows_for(name)
    entries = _confirmed_entries(rows, case['strike'], .42)
    assert entries[0].entry_timestamp == pytest.approx(case['expected_entry'])
    candidate = StrikeCandidate(timestamp=case['strike'], confidence=.9)
    shot = ShotRecord(shot_id=1, cue_strike=case['strike'],
                      clip_start=case['strike']-2, clip_end=case['expected_entry']+4,
                      end_confidence=.95, evidence={'stop_confirmed': True})
    analyzer = Analyzer(config, tmp_path)
    assert analyzer._record_visual_tail_boundaries([candidate], [shot], entries, 1229.8, 25.)
    assert candidate.evidence['visual_hand_entry_clip_cap_timestamp'] <= case['expected_entry']-.04+1e-6


@pytest.mark.parametrize('invalid', ['unclear_history', 'camera_cut', 'replay', 'cue_address', 'partial_view'])
def test_foreground_cap_requires_clear_rail_entry_and_live_continuity(invalid):
    name = 'black_respot_second'
    case = CASES[name]
    rows = rows_for(name)
    for row in rows:
        if invalid == 'unclear_history':
            row.components = [replace(c, prior_clear_times=()) for c in row.components]
        elif invalid == 'replay':
            row.context.broadcast_replay = True
        elif invalid == 'cue_address':
            row.context.cue_tip_visible = True
        elif invalid == 'partial_view':
            row.full_table = False
    if invalid == 'camera_cut':
        for row in rows:
            if row.t >= case['expected_entry']-.3:
                row.context.scene_cut_score = .9
    assert _confirmed_entries(rows, case['strike'], .42) == []
