"""Low-camera cue-ball recovery must retain independent foreground guards."""

import cv2
import numpy as np
import pytest

from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.tracking.tracker import BallTracker


def colour(hue, sat, value):
    return tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[[hue, sat, value]]]), cv2.COLOR_HSV2BGR
    )[0, 0])


def test_shaded_ivory_ball_beside_bridge_and_black_seeds_closeup_scale(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    # The bridge and black ball occupy much of the white's outer annulus.
    cv2.ellipse(frame, (206, 108), (39, 13), 0, 0, 360, colour(3, 65, 230), -1)
    cv2.circle(frame, (180, 165), 20, (20, 20, 20), -1)
    cv2.circle(frame, (180, 130), 20, colour(30, 85, 220), -1)
    yy, xx = np.ogrid[:240, :400]
    frame[((xx-180)**2 + (yy-130)**2 <= 20**2) & (yy >= 137)] = colour(42, 155, 115)
    cv2.ellipse(frame, (180, 123), (13, 8), 0, 180, 360, colour(25, 20, 255), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    cue = [d for d in found if d.label == "cue_ball"]
    assert len(cue) == 1
    assert cue[0].cue_sphere_supported
    assert cue[0].cx == pytest.approx(180, abs=3)
    assert cue[0].cy == pytest.approx(130, abs=7)
    assert cue[0].diameter_px == pytest.approx(40, abs=5)
    assert cue[0].cloth_surround_confidence < 0.65


def test_partial_view_does_not_accept_white_glove_attached_to_forearm(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.rectangle(frame, (160, 0), (184, 125), (20, 20, 20), -1)
    cv2.circle(frame, (180, 140), 20, (245, 245, 245), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    assert not any(d.label == "cue_ball" for d in found)


@pytest.mark.parametrize("hue,sat,value", [(3, 65, 230), (30, 220, 245), (165, 90, 225)])
def test_skin_yellow_and_pink_cannot_use_ivory_sphere_recovery(config, hue, sat, value):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.circle(frame, (180, 140), 20, colour(hue, sat, value), -1)
    cv2.circle(frame, (174, 132), 4, (255, 255, 255), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    assert not any(d.cue_sphere_supported for d in found)


def test_ivory_outline_keeps_its_center_when_hough_edge_is_displaced(config, monkeypatch):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.circle(frame, (180, 130), 20, colour(30, 85, 220), -1)
    cv2.circle(frame, (175, 123), 8, colour(25, 20, 255), -1)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: np.array([
        [[181, 138, 23]],
    ], dtype=np.float32))
    cue = [d for d in ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8), partial_view=True,
    ) if d.label == "cue_ball"]
    assert len(cue) == 1
    assert cue[0].cue_sphere_supported
    assert cue[0].cy == pytest.approx(130, abs=1)


@pytest.mark.parametrize("notch_depth,expected_support", [(4, True), (6, False)])
def test_shaded_outline_accepts_compact_codec_jaggedness_but_rejects_deep_gaps(
    config, notch_depth, expected_support,
):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    radii = np.where(np.arange(20) % 2, 20-notch_depth, 20)
    points = np.stack((180+radii*np.cos(angles), 130+radii*np.sin(angles)), axis=1)
    cv2.fillPoly(frame, [points.round().astype(np.int32)], colour(30, 85, 220))
    cv2.circle(frame, (176, 125), 7, colour(25, 20, 255), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    supported = [d for d in found if d.cue_sphere_supported]
    assert bool(supported) == expected_support
    if supported:
        assert supported[0].cx == pytest.approx(180, abs=1)
        assert supported[0].cy == pytest.approx(130, abs=1)


def test_two_round_red_neighbors_explain_low_cloth_support(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.ellipse(frame, (180, 105), (55, 22), 0, 0, 360, colour(173, 22, 250), -1)
    cv2.circle(frame, (180, 130), 20, colour(30, 85, 220), -1)
    cv2.circle(frame, (175, 123), 8, colour(25, 20, 255), -1)
    for x in (150, 210):
        cv2.circle(frame, (x, 149), 20, colour(2, 230, 210), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    cue = [d for d in found if d.label == "cue_ball"]
    assert len(cue) == 1
    assert cue[0].cue_sphere_supported and cue[0].cue_sphere_red_occlusion
    assert cue[0].cx == pytest.approx(180, abs=2)
    assert cue[0].cloth_surround_confidence < .30
    tracker = BallTracker()
    tracker.update(0, cue)
    track = tracker.cue_ball_track()
    assert track is not None and tracker.is_ball_quality_track(track)


@pytest.mark.parametrize("red_count", [0, 1])
def test_one_red_or_red_pixels_cannot_claim_two_ball_occlusion(config, red_count):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.ellipse(frame, (180, 105), (55, 22), 0, 0, 360, colour(173, 22, 250), -1)
    cv2.circle(frame, (180, 130), 20, colour(30, 85, 220), -1)
    cv2.circle(frame, (175, 123), 8, colour(25, 20, 255), -1)
    for i, x in enumerate((150, 210)):
        if i < red_count:
            cv2.circle(frame, (x, 149), 20, colour(2, 230, 210), -1)
        else:
            cv2.rectangle(frame, (x-19, 145), (x+19, 152), colour(2, 230, 210), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    assert not any(d.cue_sphere_red_occlusion for d in found)


def test_narrow_warm_cue_detaches_from_a_round_ivory_ball(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.line(frame, (152, 105), (180, 136), colour(30, 65, 240), 3)
    cv2.circle(frame, (180, 140), 10, colour(30, 85, 220), -1)
    cv2.circle(frame, (177, 136), 5, colour(25, 20, 255), -1)
    cue = [d for d in ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    ) if d.label == "cue_ball"]
    assert len(cue) == 1 and cue[0].cue_sphere_supported
    assert cue[0].cx == pytest.approx(180, abs=2)
    assert cue[0].cy == pytest.approx(140, abs=2)
    assert cue[0].diameter_px == pytest.approx(20, abs=3)


def test_visible_upper_hemisphere_retains_sphere_scale(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.circle(frame, (180, 140), 12, colour(30, 85, 235), -1)
    yy, xx = np.ogrid[:240, :400]
    frame[((xx-180)**2 + (yy-140)**2 <= 12**2) & (yy >= 142)] = colour(42, 170, 100)
    cv2.circle(frame, (177, 134), 5, colour(25, 20, 255), -1)
    cue = [d for d in ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    ) if d.label == "cue_ball"]
    assert len(cue) == 1 and cue[0].cue_sphere_supported
    assert cue[0].diameter_px == pytest.approx(24, abs=3)


@pytest.mark.parametrize("occluder_hue", [2, 170])
def test_ivory_crescent_behind_round_colour_retains_identity_and_scale(config, occluder_hue):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.ellipse(frame, (180, 118), (48, 22), 0, 0, 360, colour(173, 22, 250), -1)
    cv2.circle(frame, (180, 140), 20, colour(30, 85, 220), -1)
    cv2.circle(frame, (175, 132), 8, colour(25, 20, 255), -1)
    cv2.circle(frame, (184, 158), 22, colour(occluder_hue, 180, 210), -1)
    cue = [d for d in ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    ) if d.label == "cue_ball"]
    assert len(cue) == 1 and cue[0].cue_sphere_colour_occlusion
    assert cue[0].cx == pytest.approx(180, abs=3)
    assert cue[0].cy == pytest.approx(140, abs=3)
    assert cue[0].diameter_px == pytest.approx(40, abs=4)
    tracker = BallTracker()
    tracker.update(0, cue)
    assert tracker.is_ball_quality_track(tracker.cue_ball_track())


@pytest.mark.parametrize("substitute", ["skin", "glove", "yellow", "rectangle"])
def test_foreground_and_colour_fragments_cannot_claim_crescent_occlusion(config, substitute):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.ellipse(frame, (180, 118), (48, 22), 0, 0, 360, colour(173, 22, 250), -1)
    sphere_colour = {
        "skin": colour(3, 65, 230), "glove": (245, 245, 245),
        "yellow": colour(30, 220, 245), "rectangle": colour(30, 85, 220),
    }[substitute]
    cv2.circle(frame, (180, 140), 20, sphere_colour, -1)
    cv2.circle(frame, (175, 132), 8, (255, 255, 255), -1)
    if substitute == "rectangle":
        cv2.rectangle(frame, (161, 139), (207, 157), colour(2, 180, 210), -1)
    else:
        cv2.circle(frame, (184, 158), 22, colour(2, 180, 210), -1)
    found = ObjectDetector(config).detect(
        frame, np.full(frame.shape[:2], 255, np.uint8),
        use_hough=False, partial_view=True,
    )
    assert not any(d.cue_sphere_colour_occlusion for d in found)


def test_partial_cloth_contour_does_not_crop_its_boundary_cue_ball(config):
    frame = np.full((240, 400, 3), colour(60, 180, 180), np.uint8)
    cv2.circle(frame, (196, 140), 12, colour(30, 85, 220), -1)
    cv2.circle(frame, (192, 135), 5, colour(25, 20, 255), -1)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:220, :200] = 255
    cue = [d for d in ObjectDetector(config).detect(
        frame, mask, table_bounds=(0, 100, 200, 220),
        use_hough=False, partial_view=True,
    ) if d.label == "cue_ball"]
    assert len(cue) == 1 and cue[0].cue_sphere_supported
    assert cue[0].cx == pytest.approx(196, abs=1)
    assert cue[0].diameter_px == pytest.approx(24, abs=2)
