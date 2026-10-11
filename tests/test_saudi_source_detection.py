"""Source regressions for green arena lighting, advertising and paired wipes."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.config import load_config
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.table_context import view_geometry
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.table_detection.surface import playing_surface_mask
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate

FIXTURES = Path(__file__).parent / "fixtures/classic_hard/saudi"


def source_image(name):
    frame = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    assert frame is not None
    return frame


def observe(frame):
    config = load_config(overrides={"device": "cpu"})
    table = TableLocalizer(config).detect(frame)
    view = ViewClassifier(config).classify(frame)[0]
    detections = ObjectDetector(config).detect(
        frame,
        table.mask,
        partial_view=not view_geometry(table, frame.shape).full_table,
        table_bounds=table.bbox,
        main_table_view=view == CameraViewType.MAIN_TABLE,
    )
    return table, view, [d for d in detections if d.label == "cue_ball"]


@pytest.mark.parametrize(
    "name,x,y,diameter",
    [
        ("main-blue", 347, 272, 10),
        ("close-red", 501, 149, 60),
        ("close-black", 410, 164, 85),
        ("close-touching-red", 323, 363, 107),
    ],
)
def test_green_lit_white_keeps_its_source_position_and_scale(name, x, y, diameter):
    _, _, cues = observe(source_image(name))
    assert len(cues) == 1
    # A shaded close-up can fit the visible pale hemisphere rather than its
    # dark lower rim. Bound that error in ball units, while excluding the
    # separate bridge/rail observations measured in these same frames.
    position_tolerance = max(3, 0.20 * diameter)
    assert cues[0].cx == pytest.approx(x, abs=position_tolerance)
    assert cues[0].cy == pytest.approx(y, abs=position_tolerance)
    assert 0.75 * diameter <= cues[0].diameter_px <= 1.35 * diameter
    assert cues[0].cue_sphere_supported


def test_dark_green_advertising_is_excluded_from_the_live_table():
    table, view, cues = observe(source_image("main-blue"))
    assert view == CameraViewType.MAIN_TABLE
    assert table.bbox[0] > 200 and table.bbox[1] > 150
    assert table.mask[90, 700] == 0
    assert table.mask[350, 450] > 0
    assert all(cue.cy > 200 for cue in cues)


def test_referee_head_during_blue_replacement_is_not_a_cue_ball():
    table, view, cues = observe(source_image("referee-blue"))
    assert view == CameraViewType.MAIN_TABLE
    assert table.mask[124, 567] == 0
    assert all(cue.cy > 200 for cue in cues)


def test_pair_of_gold_wipes_excludes_the_replay_and_preserves_next_live_red(config):
    opening = ViewClassifier.replay_stinger_signature(source_image("gold-opening"))
    closing = ViewClassifier.replay_stinger_signature(source_image("gold-closing"))
    assert len(opening) == len(closing) == 195
    assert not ViewClassifier.replay_stinger_signature(source_image("main-blue"))
    assert not ViewClassifier.replay_stinger_signature(source_image("close-black"))
    features = [FrameFeatures(t=t) for t in (5.36, 15.0, 16.2, 28.0, 30.60)]
    features[1].appearance_signature = opening
    features[3].appearance_signature = closing
    candidates = [StrikeCandidate(timestamp=t, confidence=0.9) for t in (5.36, 16.2, 30.60)]
    result = ReplayDetector(config).mark_candidates(candidates, features)
    assert [candidate.possible_replay for candidate in result] == [False, True, False]


def test_uniform_dark_cloth_and_local_view_keep_the_measured_mask():
    hsv = np.zeros((180, 320, 3), np.uint8)
    hsv[:, :, :] = (60, 200, 80)
    mask = np.full(hsv.shape[:2], 255, np.uint8)
    assert np.array_equal(playing_surface_mask(hsv, mask), mask)
    frame = source_image("close-red")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    assert np.array_equal(playing_surface_mask(hsv, mask), mask)
