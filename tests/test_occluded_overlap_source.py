import json
from pathlib import Path

import pytest

from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate


def source_case():
    case=json.loads((Path(__file__).parent/'fixtures/classic_library/occluded-red-overlap.json').read_text())
    return case,[StrikeCandidate.model_validate(c) for c in case['candidates']],\
        [FrameFeatures.model_validate(f) for f in case['features']]


def test_native_returned_red_survives_overlapping_unresolved_blue_stop(config):
    case,candidates,features=source_case()
    assert candidates[1].confidence < .70
    shots=SegmentBuilder(config).build(candidates,features,case['duration'])
    assert [round(s.cue_strike,2) for s in shots]==[89.84,102.36]
    assert shots[0].clip_end <= shots[1].clip_start
    assert shots[1].manual_review_required


@pytest.mark.parametrize('evidence_key',['native_occlusion_confirmed','reacquired_ball_roll','ball_onset_run'])
def test_weak_preparation_proposal_cannot_use_native_return_overlap_exception(config,evidence_key):
    case,candidates,features=source_case()
    candidates[1].evidence[evidence_key]=0
    shots=SegmentBuilder(config).build(candidates,features,case['duration'])
    assert [round(s.cue_strike,2) for s in shots]==[89.84]
