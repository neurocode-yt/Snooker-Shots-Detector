"""Live reds and a black pot must survive rebound/occlusion in the latest upload."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate

CASES = json.loads((Path(__file__).parent/'fixtures/recall_referee_followup.json').read_text())['cases']


@pytest.mark.parametrize('name', CASES)
def test_reported_contact_survives_normal_native_confirmation(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case['features']]
    lower, upper = case['source_contact_bounds']
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    assert lower-.04 <= candidates[0].timestamp <= upper+.04
    proposal = StrikeCandidate(timestamp=upper, confidence=.5,
                               uncertainty_start=lower-.3, uncertainty_end=upper+.5,
                               evidence={'sparse_proposal': 1.})
    refined = detector.refine_boundaries([proposal], features)
    assert lower-.04 <= refined[0].timestamp <= upper+.04
    shots = SegmentBuilder(config).build(refined, features, 1229.8)
    assert len(shots) == 1 and shots[0].included
    assert shots[0].clip_start <= lower and shots[0].clip_end >= upper+2.


@pytest.mark.parametrize('name', CASES)
@pytest.mark.parametrize('invalid', ['replay', 'handling', 'foreign'])
def test_recovered_contacts_require_live_play(config, name, invalid):
    features = [FrameFeatures.model_validate(f) for f in CASES[name]['features']]
    for f in features:
        if invalid == 'replay':
            f.broadcast_replay = True
        elif invalid == 'handling':
            f.table_handling = True
        else:
            f.match_context_valid = False
    assert StrikeDetector(config).detect_candidates(features) == []


@pytest.mark.parametrize('invalid', ['missing_images', 'stationary_destination', 'remote_identity', 'no_quality'])
def test_interrupted_red_needs_same_physical_white_and_continuous_images(config, invalid):
    features = [FrameFeatures.model_validate(f) for f in CASES['red_behind_object']['features']]
    if invalid == 'missing_images':
        features = [f for f in features if not 1084.38 < f.t < 1084.52]
    for f in features:
        if f.t < 1084.56:
            continue
        if invalid == 'stationary_destination':
            f.cue_ball_x, f.cue_ball_y = 517., 347.
        elif invalid == 'remote_identity' and f.cue_ball_x is not None:
            f.cue_ball_x += 500.
        elif invalid == 'no_quality':
            f.cue_ball_quality = False
    assert StrikeDetector(config)._interrupted_departure_candidates(features) == []


def test_rolling_white_reappearance_cannot_replace_black_contact(config):
    features = [FrameFeatures.model_validate(f) for f in CASES['black_then_cushion_rebound']['features']]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    proposal = StrikeCandidate(timestamp=1101.56, confidence=.8,
                               uncertainty_start=1101.4, uncertainty_end=1101.8,
                               evidence={'sparse_proposal': 1.})
    refined = detector.refine_boundaries([proposal], features)
    assert refined[0].evidence['dense_transition_confirmed'] == 0.
    assert refined[0].evidence.get('native_occlusion_confirmed', 0) == 0.
