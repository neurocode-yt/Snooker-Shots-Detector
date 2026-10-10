"""Native source regressions from original-rule library calibration."""

import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.types import FrameFeatures


FIXTURES = Path(__file__).parent/'fixtures/classic_library'
CASES = json.loads((FIXTURES/'contacts.json').read_text())['cases']


def source_features(name):
    return [FrameFeatures.model_validate(row) for row in CASES[name]['features']]


def test_referee_occlusion_keeps_main_camera_scale_and_next_white(config):
    detector = ObjectDetector(config)
    localizer = TableLocalizer(config)
    for timestamp, partial in [(620.96, False), (621.24, True), (621.72, False), (630.08, False)]:
        frame = cv2.imread(str(FIXTURES/f'frame-{timestamp:.2f}.png'))
        table = localizer.detect(frame)
        x, y, width, height = table.bbox
        found = detector.detect(frame, table.mask, partial_view=partial,
                                table_bounds=(x, y, x+width, y+height), allow_closeup_scale=False)
        assert 10 < detector.estimated_ball_diameter() < 14
        assert any(d.label == 'cue_ball' and abs(d.cx-366) < 4 for d in found)


def test_dissolve_cannot_supply_a_cue_ball_launch(config):
    features = source_features('dissolve')
    detector = StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []


@pytest.mark.parametrize('case_name', ['pack_occlusion', 'proposal_window_pack'])
def test_pack_impact_is_dated_before_the_white_returns_after_collision(config, case_name):
    features = source_features(case_name)
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    lower, upper = CASES[case_name]['source_contact_bounds']
    assert lower <= candidates[0].timestamp <= upper
    assert candidates[0].evidence['impact_occlusion_contact'] == 1


def test_pack_contact_survives_pipeline_native_reconfirmation(config):
    features = source_features('proposal_window_pack')
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    detector.refine_boundaries(candidates, features)
    assert len(candidates) == 1
    assert 51.44 <= candidates[0].timestamp <= 51.52
    assert candidates[0].evidence['native_occlusion_confirmed'] == 1


@pytest.mark.parametrize('fault', ['no_object_departure', 'stationary_return', 'camera_cut',
                                  'referee_handling', 'missing_native_frame'])
def test_long_occlusion_does_not_backdate_without_continuous_physical_evidence(config, fault):
    features = source_features('pack_occlusion')
    detector = StrikeDetector(config)
    index = min(range(len(features)), key=lambda i: abs(features[i].t-651.87564))
    if fault == 'no_object_departure':
        for f in features:
            f.moving_ball_count = 0
    elif fault == 'stationary_return':
        for f in features:
            if f.t > 652:
                f.cue_ball_stable_normalized_speed = 0
    elif fault == 'camera_cut':
        features[index].scene_cut_score = .5
    elif fault == 'referee_handling':
        features[index].table_handling = True
    else:
        del features[index-2:index]
        index -= 2
    assert detector._pack_occlusion_contact_time(features, index, [f.t for f in features], .87) is None
