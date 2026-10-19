"""Actual native red-pot pixels, including the short white-to-red collision."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pot_detection.depth_launch import scoring_depth_contact
from snooker_ai.pot_detection.red_launch import RedLaunchPixels
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate


FOLDER = Path(__file__).parent/'fixtures/classic_hard/shaded-red'
CASE = json.loads((FOLDER/'source.json').read_text())


def samples():
    with np.load(FOLDER/'frames.npz') as saved:
        frames = saved['frames'].copy()
    return list(zip(frames, [FrameFeatures.model_validate(f) for f in CASE['features']]))


def observe(config, rows):
    pixels = RedLaunchPixels(config)
    for frame, context in rows:
        launch = pixels.observe(frame, context)
        if launch is not None:
            return launch
    return None


def proposal():
    return StrikeCandidate(timestamp=611.51, confidence=.5,
        uncertainty_start=603.04, uncertainty_end=614.31,
        evidence={'score_assisted_proposal': 1., 'score_delta': 1., 'score_update_time': 614.51})


def test_native_shaded_red_rests_then_departs_while_white_has_a_short_collision(config):
    launch = observe(config, samples())
    assert launch is not None
    assert CASE['expected_red_motion_upper_range'][0] <= launch['upper'] <= CASE['expected_red_motion_upper_range'][1]
    assert launch['rest_frames'] >= 7
    assert launch['moving_frames'] >= 4
    assert launch['travel_diameters'] >= .65
    assert launch['direction_consistency'] >= .9


@pytest.mark.parametrize('corruption', [
    'replay', 'handling', 'foreign', 'sparse', 'zero_cadence', 'camera', 'scene_cut',
    'scene_change', 'gap', 'still_pixels', 'single_jump', 'no_rest', 'no_white',
    'distant_white', 'poor_white', 'moving_white', 'invalid_observation', 'other_view',
])
def test_red_motion_cannot_bypass_rest_context_and_pixel_proof(config, corruption):
    rows = samples()
    for i, (frame, context) in enumerate(rows):
        if corruption == 'replay':
            context.broadcast_replay = True
        elif corruption == 'handling':
            context.table_handling = True
        elif corruption == 'foreign':
            context.match_context_valid = False
        elif corruption == 'sparse':
            context.observation_fps = 10
        elif corruption == 'zero_cadence':
            context.observation_fps = 0
        elif corruption == 'camera':
            context.camera_motion_magnitude = 1.
        elif corruption == 'scene_cut' and context.t >= 610.92:
            context.scene_cut_score = .8
        elif corruption == 'scene_change' and context.t >= 610.92:
            context.camera_scene_id += 1
        elif corruption == 'still_pixels':
            rows[i] = (rows[0][0], context)
        elif corruption == 'single_jump' and context.t >= 610.92:
            rows[i] = (rows[-1][0], context)
        elif corruption == 'no_rest' and context.t < 610.92:
            rows[i] = (np.zeros_like(frame), context)
        elif corruption == 'no_white':
            context.cue_ball_detected = False
        elif corruption == 'distant_white':
            context.cue_ball_x = -1000
        elif corruption == 'poor_white':
            context.cue_ball_quality = False
        elif corruption == 'moving_white':
            context.cue_ball_stable_normalized_speed = 2.
        elif corruption == 'invalid_observation':
            context.observation_valid = False
        elif corruption == 'other_view':
            context.view_type = 'main_table'
    if corruption == 'gap':
        rows = [(f, c) for f, c in rows if not 610.88 < c.t < 611.0]
    assert observe(config, rows) is None


def test_red_colour_without_a_measured_full_rim_cannot_seed_a_sphere(config, monkeypatch):
    monkeypatch.setattr(cv2, 'HoughCircles', lambda *a, **k: None)
    assert observe(config, samples()) is None


class SourceCapture:
    def __init__(self):
        self.rows = samples()
        self.index = -1
        self.released = False
    def isOpened(self): return True
    def set(self, *_): return True
    def read(self):
        self.index += 1
        return (True, self.rows[self.index][0]) if self.index < len(self.rows) else (False, None)
    def get(self, prop):
        return 25.000245617261392 if prop == cv2.CAP_PROP_FPS else self.rows[self.index][1].t*1000
    def release(self): self.released = True


def test_actual_red_survives_normal_score_recovery_confirmation_and_segmentation(config, tmp_path, monkeypatch):
    capture = SourceCapture()
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: capture)
    owner = Analyzer(config, tmp_path/'red')
    rows = [c for _, c in samples()]
    contacts = owner._recover_scoring_depth('source', [proposal()], [], rows)
    assert len(contacts) == 1 and capture.released
    contact = contacts[0]
    assert contact.evidence['native_scoring_red_launch'] == 1
    lo, hi = CASE['observed_cue_contact_bounds']
    assert lo <= contact.timestamp <= hi
    native = [f.model_copy(deep=True) for f in rows]
    for f in native:
        if f.cue_ball_x is not None:
            f.cue_ball_x += 500
        if f.cue_ball_y is not None:
            f.cue_ball_y += 220
    native += [FrameFeatures.model_validate(f) for f in CASE['segmentation_tail']]
    # Ordinary white-ball refinement still cannot prove this launch. Source
    # proof is retained independently and normal stop/trim logic builds it.
    owner.strike_det.refine_boundaries(contacts, native)
    assert contacts[0].evidence['dense_transition_confirmed'] == 0
    shots = owner.segmenter.build(contacts, native, 6759.658667)
    assert len(shots) == 1
    assert shots[0].clip_start < lo < hi < shots[0].clip_end
    assert 4. < shots[0].duration() < 7.


@pytest.mark.parametrize('missing', [
    'native_scoring_red_launch', 'score_assisted_proposal', 'score_supported_object_launch',
    'red_launch_rest_frames', 'red_launch_moving_frames', 'red_launch_travel_diameters',
    'red_launch_direction_consistency', 'object_launch_confirmation_time', 'score_delta',
    'red_launch_motion_start', 'contact_time_upper_bound',
])
def test_red_scoring_recovery_requires_the_complete_measured_proof(config, monkeypatch, missing):
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: SourceCapture())
    contact = scoring_depth_contact('source', proposal(), [c for _, c in samples()], config)
    assert contact is not None
    contact.evidence.pop(missing)
    assert not SegmentBuilder._candidate_supported(contact)


@pytest.mark.parametrize('points', [2., 4., 7.])
def test_red_fallback_does_not_assign_other_ball_values(config, monkeypatch, points):
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: SourceCapture())
    hint = proposal()
    hint.evidence['score_delta'] = points
    assert scoring_depth_contact('source', hint, [c for _, c in samples()], config) is None
