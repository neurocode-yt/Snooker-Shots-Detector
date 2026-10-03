"""A table mask must not amputate a ball projecting above the cloth edge."""

import cv2
import numpy as np
import pytest

from snooker_ai.object_detection.detector import ObjectDetector


def colour(hue, saturation, value):
    return tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR
    )[0, 0])


def edge_ball(center):
    frame = np.full((240, 400, 3), (20, 20, 20), np.uint8)
    frame[130:] = colour(60, 180, 180)
    cv2.circle(frame, center, 20, colour(30, 85, 220), -1)
    cv2.circle(frame, (center[0]-5, center[1]-7), 7, colour(25, 20, 255), -1)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[130:] = 255
    return frame, mask


def test_visible_ball_above_cloth_boundary_keeps_its_full_shape(config):
    frame, mask = edge_ball((180, 140))
    found = ObjectDetector(config).detect(frame, mask, use_hough=False, partial_view=True,
                                          table_bounds=(0, 130, 400, 240))
    cue = [d for d in found if d.label == "cue_ball"]
    assert len(cue) == 1
    assert cue[0].cue_sphere_supported
    assert cue[0].cx == pytest.approx(180, abs=3)
    assert cue[0].cy == pytest.approx(140, abs=3)
    assert cue[0].diameter_px == pytest.approx(40, abs=4)


def test_ivory_object_above_playing_surface_cannot_use_upper_margin(config):
    frame, mask = edge_ball((180, 96))
    found = ObjectDetector(config).detect(frame, mask, use_hough=False, partial_view=True,
                                          table_bounds=(0, 130, 400, 240))
    assert not any(d.label == "cue_ball" for d in found)


@pytest.mark.parametrize("shape,occludes", [("ball", True), ("cuff", False),
                                           ("rail", False), ("remote_ball", False)])
def test_dark_neighbor_must_be_a_nearby_round_ball(shape, occludes):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    if shape == "ball":
        cv2.circle(frame, (180, 165), 20, (20, 20, 20), -1)
    elif shape == "remote_ball":
        cv2.circle(frame, (245, 165), 20, (20, 20, 20), -1)
    elif shape == "cuff":
        cv2.ellipse(frame, (180, 165), (35, 10), 0, 0, 360, (20, 20, 20), -1)
    else:
        cv2.rectangle(frame, (100, 150), (250, 170), (20, 20, 20), -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    assert ObjectDetector._black_neighbor_occludes_sphere(hsv, 180, 130, 20) == occludes
