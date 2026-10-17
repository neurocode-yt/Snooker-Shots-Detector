"""The real head-on pink needs measured growth, native continuity and live context."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pot_detection.depth_launch import DepthLaunchPixels, scoring_depth_contact
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate


FOLDER = Path(__file__).parent/'fixtures/classic_hard/depth-pink'
CASE = json.loads((FOLDER/'source.json').read_text())


def samples():
    with np.load(FOLDER/'frames.npz') as saved:
        frames = saved['frames'].copy()
    return list(zip(frames, [FrameFeatures.model_validate(f) for f in CASE['features']]))


def proposal():
    return StrikeCandidate(timestamp=7667., confidence=.5,
        uncertainty_start=7658.61, uncertainty_end=7670.41,
        evidence={'score_assisted_proposal': 1., 'score_delta': 6., 'score_update_time': 7670.61})


def observe(config, rows):
    pixels = DepthLaunchPixels(config)
    for frame, context in rows:
        launch = pixels.observe(frame, context)
        if launch is not None:
            return launch
    return None


def test_native_source_pink_grows_towards_camera_without_a_cue_ball_track(config):
    rows = samples()
    assert not any(context.cue_ball_detected for _, context in rows)
    assert not any(context.scoring_object_launch_count for _, context in rows)
    launch = observe(config, rows)
    assert launch is not None
    lo, hi = CASE['expected_upper_range']
    assert lo <= launch['upper'] <= hi
    assert launch['area_ratio'] >= 1.6


@pytest.mark.parametrize('corruption', [
    'replay', 'handling', 'foreign', 'sparse', 'zero_cadence', 'cut', 'camera', 'scene_change',
    'gap', 'still_pixels', 'single_scale_jump', 'no_measured_rest',
])
def test_source_growth_cannot_bypass_context_or_pixel_proof(config, corruption):
    rows = samples()
    for i, (frame, context) in enumerate(rows):
        if corruption == 'replay': context.broadcast_replay = True
        elif corruption == 'handling': context.table_handling = True
        elif corruption == 'foreign': context.match_context_valid = False
        elif corruption == 'sparse': context.observation_fps = 10
        elif corruption == 'zero_cadence': context.observation_fps = 0
        elif corruption == 'cut' and context.t >= 7668.7: context.scene_cut_score = .8
        elif corruption == 'camera': context.camera_motion_magnitude = 3
        elif corruption == 'scene_change' and context.t >= 7668.7: context.camera_scene_id += 1
        elif corruption == 'still_pixels': rows[i] = (rows[0][0], context)
        elif corruption == 'single_scale_jump' and context.t >= 7668.7:
            rows[i] = (rows[-1][0], context)
        elif corruption == 'no_measured_rest' and context.t < 7668.7:
            rows[i] = (np.zeros_like(frame), context)
    if corruption == 'gap':
        rows = [(f, c) for f, c in rows if not 7668.6 < c.t < 7668.77]
    assert observe(config, rows) is None


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
        return 30. if prop == cv2.CAP_PROP_FPS else self.rows[self.index][1].t*1000
    def release(self): self.released = True


def test_native_pixel_fallback_survives_normal_score_recovery_and_segmentation(config, tmp_path, monkeypatch):
    capture = SourceCapture()
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: capture)
    owner = Analyzer(config, tmp_path/'depth')
    rows = [context for _, context in samples()]
    hints = [proposal()]
    contacts = owner._recover_scoring_depth(tmp_path/'source', hints, [], rows)
    assert len(contacts) == 1
    contact = contacts[0]
    assert capture.released
    assert contact.evidence['native_scoring_depth_launch'] == 1
    assert contact.evidence['contact_time_upper_bound'] == 1
    assert contact.timestamp == pytest.approx(7668.583333333334)
    assert SegmentBuilder._candidate_supported(contact)
    # Standard native refinement may find no white transition; it cannot
    # erase the independent source-pixel proof added by scoring recovery.
    owner.strike_det.refine_boundaries(contacts, rows)
    shots = owner.segmenter.build(contacts, rows, 7784.456446)
    assert len(shots) == 1 and shots[0].clip_start < 7668.7 < shots[0].clip_end


@pytest.mark.parametrize('hint_changes', [
    {}, {'score_assisted_proposal': 1.},
    {'score_assisted_proposal': 1., 'score_delta': 8.},
])
def test_unconfirmed_score_cannot_open_the_native_source(config, monkeypatch, hint_changes):
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture',
                        lambda _: pytest.fail('No valid score hint'))
    hint = proposal().model_copy(update={'evidence': hint_changes})
    assert scoring_depth_contact('unused', hint, [c for _, c in samples()], config) is None


def test_already_confirmed_scoring_contact_skips_additional_pixel_decode(config, tmp_path, monkeypatch):
    monkeypatch.setattr('snooker_ai.pipeline.analyzer.scoring_depth_contact',
                        lambda *a: pytest.fail('A confirmed contact needs no second decode'))
    contact = proposal()
    contact.evidence['dense_transition_confirmed'] = 1.
    owner = Analyzer(config, tmp_path/'confirmed')
    assert owner._recover_scoring_depth('unused', [proposal()], [contact], []) == [contact]


@pytest.mark.parametrize('missing', [
    'native_scoring_depth_launch', 'score_assisted_proposal', 'score_supported_object_launch',
    'depth_launch_rest_frames', 'depth_launch_growth_frames', 'depth_launch_area_ratio',
])
def test_size_motion_requires_the_complete_verified_scoring_proof(config, monkeypatch, missing):
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: SourceCapture())
    rows = [context for _, context in samples()]
    contact = scoring_depth_contact('source', proposal(), rows, config)
    contact.evidence[missing] = 0.
    assert SegmentBuilder(config).build([contact], rows, 7784.456446) == []


def test_merged_scene_ids_preserve_motion_but_an_actual_scene_break_revokes_it(config, monkeypatch):
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: SourceCapture())
    rows = [context for _, context in samples()]
    contact = scoring_depth_contact('source', proposal(), rows, config)
    for row in rows:
        row.camera_scene_id = 42
    assert len(SegmentBuilder(config).build([contact], rows, 7784.456446)) == 1
    for row in rows:
        if row.t >= 7668.8:
            row.camera_scene_id = 43
    assert SegmentBuilder(config).build([contact], rows, 7784.456446) == []


def test_pause_during_pixel_verification_releases_native_capture(config, monkeypatch):
    capture = SourceCapture()
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.open_capture', lambda _: capture)
    def pause():
        raise RuntimeError('pause requested')
    monkeypatch.setattr('snooker_ai.pot_detection.depth_launch.check_pause', pause)
    with pytest.raises(RuntimeError, match='pause requested'):
        scoring_depth_contact('source', proposal(), [c for _, c in samples()], config)
    assert capture.released
