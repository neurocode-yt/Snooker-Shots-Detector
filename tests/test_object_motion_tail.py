"""Late visible pots survive early settling trims and transition fades."""
import json
from pathlib import Path

import pytest

from snooker_ai.rendering.mix import plan_mix
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate

DATA=json.loads((Path(__file__).parent/'fixtures/object_motion_tail.json').read_text())


def rows():
    return [FrameFeatures.model_validate(f) for f in DATA['features']]


def build(config,features):
    candidate=StrikeCandidate.model_validate(DATA['candidate'])
    return SegmentBuilder(config).build([candidate],features,1229.771)[0]


def test_source_red_reaches_the_pocket_before_the_outgoing_fade(config):
    shot=build(config,rows())
    following=shot.model_copy(deep=True,update={
        'cue_strike':350.,'cue_strike_timestamp':350.,'clip_start':348.,'clip_end':354.})
    mix=plan_mix([shot,following],25.,.24)
    assert shot.included
    assert shot.clip_end-mix.overlaps[0]>=DATA['pot_bounds'][1]+.04
    assert shot.evidence['last_independent_object_motion_timestamp']==340.52
    assert shot.physical_stop_timestamp==341.04


@pytest.mark.parametrize('corruption',['cue_only','single_spike','replay','foreign',
                                      'handling','unmeasured_white','gaps'])
def test_unreliable_object_motion_cannot_extend_the_source_clip(config,corruption):
    features=rows()
    for f in features:
        if f.t<338.:
            continue
        if corruption in ('cue_only','single_spike'):
            f.max_ball_normalized_speed=f.cue_ball_stable_normalized_speed or 0.
            if corruption=='single_spike' and f.t==340.44:
                f.max_ball_normalized_speed=20.
        elif corruption=='replay':
            f.broadcast_replay=True
        elif corruption=='foreign':
            f.match_context_valid=False
        elif corruption=='handling':
            f.table_handling=True
        elif corruption=='unmeasured_white':
            f.cue_ball_quality=False
    if corruption=='gaps':
        features=[f for i,f in enumerate(features) if f.t<338. or i%7==0]
    builder=SegmentBuilder(config)
    assert builder._last_independent_object_motion(
        features,337.68,341.04,[f.t for f in features])==0.


def test_paired_replay_boundary_starts_at_the_visible_fade(config):
    features=[FrameFeatures.model_validate(f) for f in DATA['wipe_features']]
    replay=next(span for span in SegmentBuilder(config)._unusable_spans(features)
                if span[2]=='replay_clip_boundary')
    assert replay[0]==1121.6


def test_legacy_zero_cadence_row_breaks_object_proof_without_crashing(config):
    measured=FrameFeatures(
        t=1.,observation_fps=25.,cue_ball_detected=True,cue_ball_quality=True,
        cue_ball_track_confidence=.9,cue_ball_stable_normalized_speed=0.,
        moving_ball_count=1,max_ball_normalized_speed=3.,ball_kinematics_valid=True)
    legacy=FrameFeatures(t=1.04)
    assert SegmentBuilder(config)._last_independent_object_motion(
        [measured,legacy],.5,2.,[1.,1.04])==0.


@pytest.mark.parametrize('case',['single_cut','independent_marker','sparse'])
def test_replay_fade_bound_cannot_borrow_an_unproved_lead(config,case):
    features=[FrameFeatures.model_validate(f) for f in DATA['wipe_features']]
    opening=next(f.t for f in features if f.broadcast_replay)
    for f in features:
        if case=='single_cut' and f.t<opening:
            f.scene_cut_score=.8 if f.t==opening-.04 else 0.
        elif case=='independent_marker':
            f.replay_stinger_original_marker=True
        elif case=='sparse':
            f.observation_fps=2.
    replay=next(span for span in SegmentBuilder(config)._unusable_spans(features)
                if span[2]=='replay_clip_boundary')
    assert replay[0]==opening
