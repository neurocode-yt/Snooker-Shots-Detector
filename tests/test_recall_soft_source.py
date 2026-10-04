"""A source-verified soft stroke needs enough time to establish its roll."""
import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures, StrikeCandidate


def observations():
    fixture=json.loads((Path(__file__).parent/'fixtures/recall_soft_launch.json').read_text())
    return [FrameFeatures.model_validate(row) for row in fixture['features']]


def test_soft_source_contact_survives_detection_and_native_refinement(config):
    rows=observations()
    detector=StrikeDetector(config)
    detector.score_frames(rows)
    candidates=detector.detect_candidates(rows)
    assert len(candidates)==1
    assert 5017.36 <= candidates[0].timestamp <= 5017.46
    proposal=StrikeCandidate(timestamp=5017.5,confidence=.5,
                             uncertainty_start=5017.,uncertainty_end=5018.)
    refined=detector.refine_boundaries([proposal],rows)
    assert len(refined)==1
    assert 5017.36 <= refined[0].timestamp <= 5017.46
    assert refined[0].evidence['dense_transition_confirmed']==1
    assert refined[0].evidence.get('occlusion_inferred',0)==0


@pytest.mark.parametrize('invalid',['no_cue_contact','no_roll','missing_images','camera_cut','handling','identity_jump'])
def test_longer_confirmation_never_turns_unrelated_evidence_into_a_soft_shot(config,invalid):
    rows=observations()
    if invalid=='missing_images':
        rows=[f for f in rows if not 5017.65<f.t<5017.95]
    for f in rows:
        if invalid=='no_cue_contact':
            f.cue_contact_score=0
            f.cue_tip_visible=False
        elif invalid=='no_roll' and f.t>5017.5:
            f.cue_ball_x=530.5
            f.cue_ball_y=267.5
            f.cue_ball_stable_normalized_speed=0
        elif invalid=='camera_cut' and f.t>5017.6:
            f.camera_scene_id+=1
        elif invalid=='handling':
            f.table_handling=True
        elif invalid=='identity_jump' and f.t>5017.6:
            f.cue_ball_x+=100
    detector=StrikeDetector(config)
    detector.score_frames(rows)
    candidates=detector.detect_candidates(rows)
    assert not [c for c in candidates if 5017.3<=c.timestamp<=5017.5]
