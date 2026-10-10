"""A rounded foreground wrist must not hide the subsequent real white roll."""

import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.types import FrameFeatures


FIXTURES=Path(__file__).parent/'fixtures/classic_library'


@pytest.mark.parametrize('timestamp,expected_white',[(98.5,True),(99.6,False)])
def test_main_camera_forearm_is_rejected_and_nearby_real_ivory_ball_is_kept(config,timestamp,expected_white):
    frame=cv2.imread(str(FIXTURES/f'forearm-{timestamp:.1f}.png'))
    table=TableLocalizer(config).detect(frame)
    x,y,w,h=table.bbox
    found=ObjectDetector(config).detect(frame,table.mask,partial_view=True,
        allow_closeup_scale=False,main_table_view=True,table_bounds=(x,y,x+w,y+h))
    cue=[d for d in found if d.label=='cue_ball']
    assert bool(cue)==expected_white
    if cue:
        assert cue[0].cx==pytest.approx(633.1,abs=3)
        assert cue[0].cy==pytest.approx(218.6,abs=3)


def test_hidden_red_returns_as_one_real_shot_without_preparation_false_contact(config):
    case=json.loads((FIXTURES/'forearm-recovery.json').read_text())
    features=[FrameFeatures.model_validate(f) for f in case['features']]
    detector=StrikeDetector(config)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    assert len(candidates)==1
    lower,upper=case['source_contact_bounds']
    assert lower-.25 <= candidates[0].timestamp <= upper+.25
    assert candidates[0].evidence['reacquired_ball_roll']==1
    assert candidates[0].uncertainty_start <= upper
