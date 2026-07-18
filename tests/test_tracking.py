"""Focused tests for scale-aware ball observations and prediction tracking."""

from __future__ import annotations

import pytest
import cv2
import numpy as np

from snooker_ai.object_detection.detector import Detection, ObjectDetector
from snooker_ai.tracking.tracker import BallTracker


def _d(
    x: float,
    y: float,
    *,
    label: str = "object_ball",
    confidence: float = 0.8,
    diameter: float = 10.0,
    color: float = 0.2,
) -> Detection:
    r = diameter * 0.5
    return Detection(
        label=label,
        confidence=confidence,
        bbox=(int(x - r), int(y - r), int(diameter), int(diameter)),
        cx=x,
        cy=y,
        radius=r,
        diameter=diameter,
        color_confidence=color,
        shape_confidence=0.9,
    )


def test_prediction_and_label_awareness_preserve_identity() -> None:
    tracker = BallTracker(max_distance=25.0)
    first = tracker.update(
        0.0,
        [
            _d(0.0, 0.0, label="cue_ball", color=0.95),
            _d(20.0, 0.0),
        ],
    )
    cue_id = next(track.track_id for track in first if track.label == "cue_ball")
    object_id = next(track.track_id for track in first if track.label == "object_ball")

    # Establish opposing velocities, then present detections in reverse order near
    # the crossing.  Nearest-neighbour matching would be prone to swapping them.
    tracker.update(
        0.1,
        [
            _d(4.0, 0.0, label="cue_ball", color=0.95),
            _d(16.0, 0.0),
        ],
    )
    tracker.update(
        0.2,
        [
            _d(12.0, 0.0),
            _d(8.0, 0.0, label="cue_ball", color=0.95),
        ],
    )

    cue = next(track for track in tracker.tracks if track.track_id == cue_id)
    obj = next(track for track in tracker.tracks if track.track_id == object_id)
    assert cue.label == "cue_ball"
    assert cue.positions[-1][1] == pytest.approx(8.0)
    assert obj.positions[-1][1] == pytest.approx(12.0)


def test_missed_observation_does_not_report_stale_speed() -> None:
    tracker = BallTracker(max_distance=30.0, max_missed=1.0)
    tracker.update(0.0, [_d(10.0, 10.0)])
    tracker.update(0.1, [_d(20.0, 10.0)])
    assert tracker.max_speed() > 0.0

    active = tracker.update(0.2, [])
    assert len(active) == 1
    assert active[0].visible is False
    assert active[0].occluded is True
    assert active[0].missed_frames == 1
    assert active[0].speed() == 0.0
    assert tracker.max_speed() == 0.0
    assert tracker.occluded_moving_count() == 1

    tracker.update(1.2, [])
    assert active[0].active is False
    assert tracker.occluded_moving_count() == 0


def test_speed_normalization_uses_ball_diameter() -> None:
    tracker = BallTracker(max_distance=50.0)
    tracker.update(0.0, [_d(0.0, 0.0, diameter=10.0)])
    tracker.update(0.5, [_d(10.0, 0.0, diameter=10.0)])
    assert tracker.max_speed() == pytest.approx(20.0)
    assert tracker.estimated_ball_diameter() == pytest.approx(10.0)
    assert tracker.max_normalized_speed() == pytest.approx(2.0)
    assert tracker.max_normalized_speed(ball_diameter_px=20.0) == pytest.approx(1.0)


def test_cue_ball_selection_prefers_visible_high_colour_confidence() -> None:
    tracker = BallTracker(max_distance=20.0)
    tracks = tracker.update(
        0.0,
        [
            _d(10.0, 10.0, label="cue_ball", confidence=0.9, color=0.40),
            _d(40.0, 10.0, label="cue_ball", confidence=0.75, color=0.95),
        ],
    )
    expected = max(tracks, key=lambda track: track.cue_color_confidence)
    assert tracker.cue_ball_track() is expected

    # The strongest white track is missed, so a visible candidate should win.
    tracker.update(
        0.1,
        [_d(11.0, 10.0, label="cue_ball", confidence=0.9, color=0.40)],
    )
    selected = tracker.cue_ball_track()
    assert selected is not None
    assert selected.visible is True
    assert selected.positions[-1][1] == pytest.approx(11.0)


def test_detection_derives_scale_for_legacy_constructor() -> None:
    detection = Detection("object_ball", 0.6, (1, 2, 12, 10), 7.0, 7.0)
    assert detection.radius == pytest.approx(5.0)
    assert detection.diameter_px == pytest.approx(10.0)


def test_cpu_detector_finds_scale_and_white_cue_ball(config) -> None:
    frame = np.full((240, 400, 3), (35, 105, 35), dtype=np.uint8)
    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
    cv2.circle(frame, (150, 120), 6, (245, 245, 245), thickness=-1)
    cv2.circle(frame, (250, 120), 6, (20, 20, 210), thickness=-1)

    detector = ObjectDetector(config)
    detections = detector.detect(frame, mask)
    cue = [detection for detection in detections if detection.label == "cue_ball"]
    objects = [detection for detection in detections if detection.label == "object_ball"]

    assert cue and objects
    assert cue[0].color_confidence > 0.8
    assert cue[0].diameter_px == pytest.approx(12.0, abs=2.0)
    assert detector.estimated_ball_diameter() == pytest.approx(12.0, abs=3.0)
