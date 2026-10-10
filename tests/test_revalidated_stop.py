"""Native geometry recovery must never invent or prematurely trim a stop."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from snooker_ai.segmentation.visual_tail import TailObservation, VisualTailDetector
from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.types import FrameFeatures, ShotRecord, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper


def measure_rows():
    rows = []
    for index in range(150):
        t = 3.+index*.04
        moving = t < 6
        context = FrameFeatures(t=t, observation_fps=25., view_type='main_table',
            table_full_view=False, table_confidence=.95, ball_count=10, ball_diameter_px=10.,
            ball_kinematics_valid=True, moving_ball_count=int(moving),
            max_ball_normalized_speed=2. if moving else .02,
            motion_raw=.3 if moving else .02, residual_motion_max=1. if moving else .1)
        rows.append(TailObservation(t=t, context=context, full_table=True))
    return rows


def setup_measurement(config, monkeypatch, rows):
    capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None)
    monkeypatch.setattr('snooker_ai.segmentation.visual_tail.open_capture', lambda *a: capture)
    detector = VisualTailDetector(config)
    monkeypatch.setattr(detector, '_observe_window', lambda *a, **k: rows)
    shot = ShotRecord(shot_id=1, cue_strike=3., clip_start=1., clip_end=9.,
                      strike_confidence=.9, manual_review_required=True)
    return detector, shot


def test_native_revalidated_coverage_uses_motion_evidence_and_keeps_raw_features(config, monkeypatch):
    rows = measure_rows()
    original = copy.deepcopy(rows)
    detector, shot = setup_measurement(config, monkeypatch, rows)
    result = detector.revalidated_stops('source', TimeMapper(12.), [shot], [r.context for r in rows])
    assert result[3.].confirmed
    assert result[3.].physical_stop_timestamp == pytest.approx(6.)
    assert result[3.].manual_review_required
    assert rows == original


@pytest.mark.parametrize('defect', ['moving_ball', 'occluded_ball', 'sparse', 'offset_pts',
                                   'closeup', 'unmeasured_geometry', 'reset', 'replay'])
def test_native_coverage_cannot_override_motion_or_missing_source_proof(config, monkeypatch, defect):
    rows = measure_rows()
    for row in rows:
        if defect == 'moving_ball':
            row.context.moving_ball_count = 1
            row.context.max_ball_normalized_speed = 2.
        elif defect == 'occluded_ball':
            row.context.occluded_ball_count = 1
            row.context.ambiguous_ball_motion = True
        elif defect == 'sparse':
            row.context.observation_fps = 2.
        elif defect == 'offset_pts':
            row.t += .01
        elif defect == 'closeup':
            row.context.view_type = 'ball_closeup'
        elif defect == 'unmeasured_geometry':
            row.full_table = False
        elif defect == 'reset':
            row.reset = True
        else:
            row.context.broadcast_replay = True
    detector, shot = setup_measurement(config, monkeypatch, rows)
    assert detector.revalidated_stops('source', TimeMapper(12.), [shot], [r.context for r in rows]) == {}


def test_user_edit_is_preserved_during_stop_revalidation(config, monkeypatch):
    rows = measure_rows()
    detector, shot = setup_measurement(config, monkeypatch, rows)
    shot.user_modified = True
    assert detector.revalidated_stops('source', TimeMapper(12.), [shot], [r.context for r in rows]) == {}


def test_source_break_stillness_survives_rounded_cushions_without_changing_raw_tracks(config, monkeypatch):
    data=json.loads((Path(__file__).parent/'fixtures/classic_hard/revalidated-break-stop.json').read_text())
    rows=[TailObservation(t=r['t'],full_table=r['full_table'],reset=r['reset'],
                          context=FrameFeatures.model_validate(r['context']) if r['context'] else None)
          for r in data['rows']]
    features=[r.context for r in rows if r.context]
    original=copy.deepcopy(features)
    raw=BallStopDetector(config).detect_stop(StrikeCandidate(timestamp=1.8,confidence=.9),features,22.4)
    assert not raw.confirmed
    detector,shot=setup_measurement(config,monkeypatch,rows)
    shot.cue_strike=1.8
    shot.clip_start=0.
    shot.clip_end=22.4
    measured=detector.revalidated_stops('source',TimeMapper(160.04),[shot],features)
    assert measured[1.8].physical_stop_timestamp==pytest.approx(data['expected_revalidated_stop'])
    assert measured[1.8].manual_review_required
    assert features==original
