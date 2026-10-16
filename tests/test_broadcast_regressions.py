"""Native Murphy/Day broadcast regressions; score fixtures exclude training glyphs."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.pot_detection.scoreboard import ScoreboardReader
from snooker_ai.scene_detection.table_context import view_geometry
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.utils.video import IncompleteVideoError, sampled_frames
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures
from snooker_ai.types import StrikeCandidate
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.utils.timebase import TimeMapper
import json

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.mark.parametrize('timestamp,expected,active', [
    (450, (1, 4), 0), (750, (16, 4), 0), (780, (16, 4), 1), (1530, (0, 7), 1),
])
@pytest.mark.parametrize('scale', [1., .75, 1.5])
def test_central_panel_points_ignore_frame_counts(timestamp, expected, active, scale):
    frame = cv2.imread(str(FIXTURES / 'pot_scoreboards' / f'murphy_panel_{timestamp}.png'))
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    reading = ScoreboardReader().read(frame, timestamp)
    assert reading is not None
    assert (reading.left, reading.right) == expected
    assert reading.active_player == active
    assert reading.layout == 'central_panel'


@pytest.mark.parametrize('size', [(960, 540), (640, 360)])
def test_neon_advertising_cannot_turn_enclosed_table_into_closeup(config, size):
    image = cv2.imread(str(FIXTURES / 'classic_hard' / 'murphy_neon_table.jpg'))
    frame = cv2.resize(image, size)
    table = TableLocalizer(config).detect(frame)
    assert table.bbox[1] > size[1]*.15
    assert view_geometry(table, frame.shape).full_table


def test_real_camera_crop_stays_partial_after_neon_mask_fix(config):
    image = cv2.imread(str(FIXTURES / 'classic_hard' / 'murphy_neon_table.jpg'))
    frame = image[180:600, 320:960]
    table = TableLocalizer(config).detect(frame)
    assert not view_geometry(table, frame.shape).full_table


class TimestampCapture:
    def __init__(self, pts):
        self.pts, self.index = pts, 0

    def grab(self):
        if self.index == len(self.pts):
            return False
        self.index += 1
        return True

    def get(self, prop):
        assert prop == cv2.CAP_PROP_POS_MSEC
        return self.pts[self.index-1]*1000

    def retrieve(self):
        return True, np.full((2, 2, 3), self.index, np.uint8)


def test_native_sampling_uses_presentation_time_even_with_average_fps_drift():
    pts = [7200., 7200.033333, 7200.066667, 7200.1]
    frames = list(sampled_frames(TimestampCapture(pts), proxy_fps=30.0000856,
        to_source=lambda t:t, start_time=7200., end_time=7200.1,
        sample_period=1/30.0000856, start_idx=216000, presentation_timestamps=True))
    assert [t for _, t, _ in frames] == pts
    assert [i for i, _, _ in frames] == list(range(216000, 216004))


def test_invalid_native_pts_cannot_silently_fall_back_to_average_fps():
    frames = sampled_frames(TimestampCapture([5., 5.]), proxy_fps=30,
        to_source=lambda t:t, start_time=5, end_time=6,
        sample_period=1/30, presentation_timestamps=True)
    with pytest.raises(IncompleteVideoError, match='presentation timestamps'):
        list(frames)


def test_green_shadow_cannot_move_the_white_ball_track_to_its_upper_highlight(config):
    image=cv2.imread(str(FIXTURES/'classic_hard/murphy_shaded_cue.jpg'))
    frame=cv2.resize(image,(960,540))
    table=TableLocalizer(config).detect(frame)
    found=ObjectDetector(config).detect(frame,table.mask,partial_view=True)
    cue=[d for d in found if d.label=='cue_ball']
    assert len(cue)==1 and cue[0].cue_sphere_supported
    assert (cue[0].cx,cue[0].cy)==pytest.approx((517.5,345.5),abs=3)
    assert cue[0].radius==pytest.approx(25.5,abs=3)


def test_main_table_tinted_white_retains_its_measured_sphere_support(config):
    image=cv2.imread(str(FIXTURES/'classic_hard/murphy_neon_table.jpg'))
    frame=cv2.resize(image,(960,540))
    table=TableLocalizer(config).detect(frame)
    found=ObjectDetector(config).detect(frame,table.mask,main_table_view=True,
        partial_view=False,allow_closeup_scale=False)
    cue=[d for d in found if d.label=='cue_ball']
    assert len(cue)==1 and cue[0].cue_sphere_supported
    assert (cue[0].cx,cue[0].cy)==pytest.approx((286,225),abs=4)


def test_sharp_colour_supplies_scale_for_out_of_focus_white(config):
    image=cv2.imread(str(FIXTURES/'classic_hard/murphy_defocused_white.jpg'))
    frame=cv2.resize(image,(960,540));table=TableLocalizer(config).detect(frame)
    detector=ObjectDetector(config)
    found=detector.detect(frame,table.mask,partial_view=True)
    reference=[d for d in found if d.object_sphere_supported]
    assert any((d.cx-466.5)**2+(d.cy-229)**2<25 for d in reference)
    cue=[d for d in found if d.label=='cue_ball']
    assert len(cue)==1 and cue[0].cue_sphere_supported
    assert (cue[0].cx,cue[0].cy)==pytest.approx((497,347),abs=5)


def test_defocus_recovery_cannot_replace_removed_white_with_bridge_or_pink(config):
    image=cv2.imread(str(FIXTURES/'classic_hard/murphy_defocused_white.jpg'))
    frame=cv2.resize(image,(960,540))
    cv2.circle(frame,(497,347),98,(20,150,20),-1)
    table=TableLocalizer(config).detect(frame)
    found=ObjectDetector(config).detect(frame,table.mask,partial_view=True)
    assert not any(d.label=='cue_ball' for d in found)


def test_measured_blurred_body_wins_over_its_brighter_partial_cap():
    from snooker_ai.object_detection.detector import _Proposal
    cap = _Proposal(482, 323, 52, .99, cue_sphere_supported=True)
    body = _Proposal(497, 347, 94, .85, cue_sphere_supported=True,
                     cue_body_measured=True)
    for proposals in ([cap, body], [body, cap]):
        merged = ObjectDetector._merge_proposals(proposals, 150)
        assert len(merged) == 1
        assert (merged[0].cx, merged[0].cy, merged[0].radius) == (497, 347, 94)


def test_missing_red_contact_is_recovered_from_native_cue_motion(config):
    data=json.loads((FIXTURES/'classic_hard/murphy_closeup_contact.json').read_text())
    features=[FrameFeatures.model_validate(f) for f in data['features']]
    detector=StrikeDetector(config)
    candidates=detector.detect_candidates(detector.score_frames(features))
    assert any(data['source_contact_bracket'][0]-.001 <= c.timestamp <= data['source_contact_bracket'][1]+.001
               for c in candidates)


def test_native_stop_tail_retains_contact_history_and_exits_before_safety_horizon(config,tmp_path,monkeypatch):
    owner=Analyzer(config,tmp_path/'native-history')
    candidate=StrikeCandidate(timestamp=100.,confidence=.9,evidence={'dense_transition_confirmed':1.})
    monkeypatch.setattr(owner.strike_det,'refine_boundaries',lambda candidates,*a:candidates)
    decoded=[]
    def extract(*args,start_time,end_time,sample_fps,stop_when=None,**kwargs):
        rows=[]
        for i in range(round((end_time-start_time)*sample_fps)+1):
            t=start_time+i/sample_fps
            moving=100<=t<=(102 if sample_fps==30 else 109.9)
            rows.append(FrameFeatures(t=t,observation_fps=sample_fps,view_type='main_table',
                table_confidence=.95,table_full_view=True,ball_count=10,ball_diameter_px=10,
                ball_kinematics_valid=True,moving_ball_count=int(moving),
                max_ball_normalized_speed=2. if moving else .02,
                motion_raw=.3 if moving else .02,residual_motion_max=1. if moving else .1))
            if stop_when and stop_when(rows):break
        decoded.append((sample_fps,rows[-1].t))
        return rows,[],[]
    monkeypatch.setattr(owner,'_extract_features',extract)
    owner._refine_candidate_windows(Path('unused.mp4'),None,TimeMapper(300.,source_fps=30.),
        300.,[candidate],[],resume=False)
    assert candidate.evidence['refined_stop_timestamp']==pytest.approx(107.)
    assert candidate.evidence['refined_stop_upper_bound']==1.
    assert decoded[-1][1]==pytest.approx(107.7)
