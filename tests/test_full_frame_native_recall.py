"""Complete-frame omissions and preparation controls from the latest upload."""

import json
from pathlib import Path

import numpy as np
import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate

CASES=json.loads((Path(__file__).parent/'fixtures/full_frame_native_windows.json').read_text())['cases']


def rows(name):
    return [FrameFeatures.model_validate(f) for f in CASES[name]['features']]


@pytest.mark.parametrize('name',CASES)
def test_full_frame_source_contacts_and_preparation_controls(config,name):
    features=rows(name)
    detector=StrikeDetector(config)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    bounds=CASES[name]['source_contact_bounds']
    if bounds is None:
        assert candidates==[]
        return
    lower,upper=bounds
    assert len(candidates)==1
    assert lower-.08<=candidates[0].timestamp<=upper+.20
    proposal=StrikeCandidate(timestamp=upper,confidence=.5,
                             uncertainty_start=lower-.30,uncertainty_end=upper+.40,
                             evidence={'sparse_proposal':1.})
    refined=detector.refine_boundaries([proposal],features)
    assert lower-.08<=refined[0].timestamp<=upper+.20
    shots=SegmentBuilder(config).build(refined,features,1229.771)
    assert len(shots)==1 and shots[0].included
    assert shots[0].clip_start<=lower and shots[0].clip_end>=upper


POSITIVE=[name for name,case in CASES.items() if case['source_contact_bounds'] is not None]


@pytest.mark.parametrize('name',POSITIVE)
@pytest.mark.parametrize('invalid',['replay','foreign','handling'])
def test_contacts_do_not_use_replay_foreign_or_handling_frames(config,name,invalid):
    features=rows(name)
    for f in features:
        if invalid=='replay':
            f.broadcast_replay=True
        elif invalid=='foreign':
            f.match_context_valid=False
        else:
            f.table_handling=True
    assert StrikeDetector(config).detect_candidates(features)==[]


@pytest.mark.parametrize('name',['short_touching_red','short_occluded_red','red_with_collision',
                                 'soft_safety_after_replay','yellow_during_zoom',
                                 'brown_during_zoom','red_after_bridge_occlusion'])
def test_contact_geometry_without_physical_white_departure_is_not_a_shot(config,name):
    features=rows(name)
    working=StrikeDetector._stabilized_features(features)
    anchor=next((f.cue_ball_x,f.cue_ball_y) for f in working
                if f.cue_ball_quality and f.cue_ball_x is not None)
    for f in working:
        if f.cue_ball_x is not None:
            f.cue_ball_x,f.cue_ball_y=anchor
        f.cue_ball_normalized_speed=0.
        f.cue_ball_speed=0.
        f.cue_ball_stable_normalized_speed=0.
        f.cue_ball_acceleration=0.
        f.moving_ball_count=0
        f.max_ball_normalized_speed=0.
    assert StrikeDetector(config).detect_candidates(working)==[]


def test_soft_reacquisition_still_requires_measurable_travel(config):
    working=StrikeDetector._stabilized_features(rows('soft_safety_after_replay'))
    visible=[f for f in working if f.cue_ball_quality and f.cue_ball_x is not None]
    anchor=np.array([visible[0].cue_ball_x,visible[0].cue_ball_y])
    for f in visible:
        point=anchor+.4*(np.array([f.cue_ball_x,f.cue_ball_y])-anchor)
        f.cue_ball_x,f.cue_ball_y=point.tolist()
    assert StrikeDetector(config).detect_candidates(working)==[]


def test_zoom_during_yellow_preparation_does_not_create_an_earlier_contact(config):
    features=rows('yellow_during_zoom')
    detector=StrikeDetector(config)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    assert len(candidates)==1 and 579.20<=candidates[0].timestamp<=579.28
    assert candidates[0].evidence['camera_drift_launch_confirmed']==1.


def test_yellow_zoom_without_the_later_stroke_is_not_a_contact(config):
    features=[f for f in rows('yellow_during_zoom') if f.t<=579.24]
    detector=StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features)==[]


@pytest.mark.parametrize('name,edge',[('brown_during_zoom',317.92),
                                     ('red_after_bridge_occlusion',675.72)])
def test_retained_contact_preparation_alone_is_not_a_shot(config,name,edge):
    features=[f for f in rows(name) if f.t<=edge]
    detector=StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features)==[]


def camera_sequence(moving=False):
    transform=np.eye(3)
    step=np.array([[1.01,0.,-.5],[0.,1.01,-.4],[0.,0.,1.]])
    features=[]
    for i in range(40):
        transform=step@transform
        contact=moving and i>=20
        world=np.array([100.+max(0,i-19)*1.8 if moving else 100.,100.,1.])
        point=transform@world
        features.append(FrameFeatures(
            t=i*.04,observation_fps=25.,camera_scene_id=1,
            camera_frame_dt=.04,camera_frame_transform=step[:2].reshape(-1).tolist(),
            table_confidence=.9,ball_count=8,ball_kinematics_valid=True,
            cue_ball_detected=True,cue_ball_quality=True,cue_ball_observations=i+5,
            cue_ball_x=point[0],cue_ball_y=point[1],
            ball_diameter_px=10.*np.sqrt(np.linalg.det(transform[:2,:2])),
            cue_ball_track_confidence=.9,cue_ball_normalized_speed=4.5 if contact else 0.,
            cue_ball_stable_normalized_speed=4.5 if contact else 0.,
            cue_ball_acceleration=50. if i==20 and moving else 0.,
            max_ball_normalized_speed=4.5 if contact else 0.,
            motion_raw=.8 if contact else .1,cue_tip_visible=True,cue_contact_score=.95))
    return features


def test_camera_reference_preserves_raw_image_coordinates_and_static_white(config):
    features=camera_sequence()
    original=[f.model_dump() for f in features]
    working=StrikeDetector._stabilized_features(features)
    points=np.array([(f.cue_ball_x,f.cue_ball_y) for f in working])
    assert np.max(np.linalg.norm(points-points[0],axis=1))<1e-6
    assert [f.model_dump() for f in features]==original
    assert StrikeDetector(config).detect_candidates(features)==[]


def test_true_white_launch_survives_camera_zoom(config):
    features=camera_sequence(moving=True)
    detector=StrikeDetector(config)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    assert len(candidates)==1
    assert .76<=candidates[0].timestamp<=.84


def test_missing_frame_cannot_reuse_a_camera_transform_for_a_different_pair():
    features=camera_sequence()
    del features[10]
    working=StrikeDetector._stabilized_features(features)
    assert not working[10].observation_valid
    assert features[10].observation_valid
