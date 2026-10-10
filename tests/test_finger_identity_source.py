import json
from pathlib import Path

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures


def test_strong_cue_geometry_on_finger_cannot_confirm_out_and_back_identity_swap(config):
    case=json.loads((Path(__file__).parent/'fixtures/classic_library/finger-identity-return.json').read_text())
    features=[FrameFeatures.model_validate(f) for f in case['features']]
    detector=StrikeDetector(config)
    index=min(range(len(features)),key=lambda i:abs(features[i].t-case['negative_contact']))
    metrics=detector._transition_metrics(detector._stabilized_features(features),index)
    assert metrics['cue_contact_score']>.80
    assert metrics['pre_cue_address_score']>.65
    assert not detector._transition_confirmed(metrics)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    assert len(candidates)==1
    lower,upper=case['source_contact_bounds']
    assert lower-.25 <= candidates[0].timestamp <= upper+.25
    detector.refine_boundaries(candidates,features)
    assert candidates[0].evidence['dense_transition_confirmed']==1
