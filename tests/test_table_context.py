"""Coverage and hand interactions must remain distinct across camera angles."""

import cv2
import numpy as np
import pytest

from snooker_ai.object_detection.detector import Detection, ObjectDetector
from snooker_ai.scene_detection.table_context import (
    TableInteractionDetector, ball_layout, view_geometry,
)
from snooker_ai.table_detection.localizer import TableLocalizer, TableObservation


def polygon_table(points, shape=(360, 640), confidence=0.9):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return TableObservation(confidence, mask, contour, area_ratio=cv2.contourArea(contour) / mask.size)


def ball(label, x, y, radius=6):
    return Detection(label, 0.9, (x-radius, y-radius, radius*2, radius*2), x, y,
                     radius=radius, shape_confidence=0.95, cloth_surround_confidence=0.9)


def test_enclosed_perspective_table_has_consistent_normalized_corners():
    points = np.array([[140, 70], [500, 80], [570, 300], [70, 300]], np.float32)
    table = polygon_table(points)
    geometry = view_geometry(table, table.mask.shape)
    assert geometry.full_table and geometry.homography is not None
    mapped = cv2.perspectiveTransform(points.reshape(1, 4, 2), geometry.homography)[0]
    assert mapped == pytest.approx(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), abs=0.01)


@pytest.mark.parametrize("points", [
    [[0, 70], [500, 80], [570, 300], [0, 300]],
    [[140, 0], [500, 0], [570, 300], [70, 300]],
    [[140, 70], [500, 80], [639, 300], [70, 300]],
])
def test_camera_crop_cannot_be_calibrated_as_a_complete_table(points):
    table = polygon_table(points)
    geometry = view_geometry(table, table.mask.shape)
    assert not geometry.full_table
    assert geometry.homography is None


def test_fallback_rectangle_does_not_prove_complete_table():
    table = polygon_table([[80, 60], [560, 60], [560, 300], [80, 300]], confidence=0.1)
    assert not view_geometry(table, table.mask.shape).full_table


def test_large_white_ball_is_detected_at_closeup_scale(config):
    frame = np.full((240, 400, 3), (40, 140, 40), dtype=np.uint8)
    cv2.circle(frame, (170, 140), 24, (240, 240, 240), -1)
    detector = ObjectDetector(config)
    detections = detector.detect(frame, np.full(frame.shape[:2], 255, np.uint8),
                                 use_hough=False, partial_view=True)
    white = [d for d in detections if d.label == "cue_ball"]
    assert len(white) == 1
    assert white[0].cx == pytest.approx(170, abs=2)
    assert white[0].diameter_px == pytest.approx(48, abs=5)
    assert detector.estimated_ball_diameter() >= 40


def test_camera_change_discards_overhead_scale_before_closeup(config, synthetic_green_frame):
    detector = ObjectDetector(config)
    mask = TableLocalizer(config).detect(synthetic_green_frame).mask
    detector.detect(synthetic_green_frame, mask, use_hough=False)
    assert 0 < detector.estimated_ball_diameter() < 20
    detector.reset()
    frame = np.full((240, 400, 3), (40, 140, 40), dtype=np.uint8)
    cv2.circle(frame, (170, 140), 24, (240, 240, 240), -1)
    detector.detect(frame, np.full(frame.shape[:2], 255, np.uint8),
                    use_hough=False, partial_view=True)
    assert detector.estimated_ball_diameter() >= 40


def test_coloured_ball_layout_agrees_across_two_perspective_views():
    canonical = np.array([[0,0], [1,0], [1,1], [0,1]], np.float32)
    views = [np.array(points, np.float32) for points in (
        [[120,65], [515,65], [565,305], [65,305]],
        [[100,70], [535,95], [585,310], [55,285]],
    )]
    physical = np.array([[[.2,.4], [.5,.5], [.7,.8]]], np.float32)
    layouts = []
    for corners in views:
        table = polygon_table(corners)
        frame = np.zeros((360,640,3), np.uint8)
        frame[table.mask > 0] = (40,140,40)
        inverse = cv2.getPerspectiveTransform(canonical, corners)
        centres = cv2.perspectiveTransform(physical, inverse)[0]
        detections = []
        for (x,y), label, colour in zip(centres, ("cue_ball", "ball", "ball"),
                                      ((240,240,240), (220,70,20), (20,20,20))):
            cv2.circle(frame, (round(float(x)), round(float(y))), 6, colour, -1)
            detections.append(ball(label, float(x), float(y)))
        layout = ball_layout(frame, detections, view_geometry(table, frame.shape))
        assert len(layout) == 14
        assert layout[:2] == pytest.approx([.2,.4], abs=.015)
        assert layout[8:10] == pytest.approx([.5,.5], abs=.015)
        assert layout[12:14] == pytest.approx([.7,.8], abs=.015)
        assert layout[2:8] == [-1.0] * 6
        layouts.append(layout)
    assert layouts[0] == pytest.approx(layouts[1], abs=.015)


def test_bridge_hand_near_white_ball_is_not_referee_handling():
    table = polygon_table([[30, 30], [450, 30], [450, 250], [30, 250]], (270, 480))
    frame = np.full((270, 480, 3), (40, 140, 40), dtype=np.uint8)
    cv2.ellipse(frame, (180, 180), (30, 10), 0, 0, 360, (100, 150, 200), -1)
    detections = [ball("cue_ball", 215, 180), ball("ball", 350, 100)]
    detector = TableInteractionDetector()
    results = [detector.observe(frame, table, detections, t, cue_visible=True)[0]
               for t in (0, 0.1, 0.2, 0.3)]
    assert not any(results)


def test_white_glove_covering_last_visible_colour_is_sustained_handling():
    table = polygon_table([[30, 30], [450, 30], [450, 250], [30, 250]], (270, 480))
    clear = np.full((270, 480, 3), (40, 140, 40), dtype=np.uint8)
    cv2.circle(clear, (250, 160), 6, (220, 70, 30), -1)
    detector = TableInteractionDetector()
    assert not detector.observe(clear, table, [ball("ball", 250, 160)], 0)[0]
    glove = clear.copy()
    cv2.ellipse(glove, (250, 160), (35, 14), 0, 0, 360, (240, 240, 240), -1)
    results = [detector.observe(glove, table, [], t, cue_visible=False)[0]
               for t in (0.05, 0.15, 0.25, 0.35)]
    assert results[2] and results[3]


def test_small_colour_ball_alone_cannot_be_a_hand_component():
    table = polygon_table([[30, 30], [450, 30], [450, 250], [30, 250]], (270, 480))
    frame = np.full((270, 480, 3), (40, 140, 40), dtype=np.uint8)
    cv2.circle(frame, (250, 160), 6, (120, 130, 230), -1)
    detector = TableInteractionDetector()
    assert not any(detector.observe(frame, table, [ball("ball", 250, 160)], t)[0]
                   for t in (0, 0.1, 0.2, 0.3))


def test_established_bridge_does_not_become_handling_when_contact_blurs_white():
    table = polygon_table([[30, 30], [450, 30], [450, 250], [30, 250]], (270, 480))
    frame = np.full((270, 480, 3), (40, 140, 40), dtype=np.uint8)
    cv2.ellipse(frame, (250, 160), (35, 14), 0, 0, 360, (240, 240, 240), -1)
    detector = TableInteractionDetector()
    colour = ball("ball", 265, 160)
    assert not detector.observe(frame, table, [colour], 0, cue_visible=True)[0]
    assert not any(detector.observe(frame, table, [colour], t, cue_visible=False)[0]
                   for t in (.1, .2, .3, .5, .9))
    # A later sustained glove action still gets its own handling evidence.
    detector.observe(frame, table, [colour], 1.1, cue_visible=False)
    assert detector.observe(frame, table, [colour], 1.3, cue_visible=False)[0]
