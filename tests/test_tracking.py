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


def test_camera_pan_does_not_turn_stationary_balls_into_motion():
    tracker = BallTracker()
    tracker.update(0.0, [_d(20, 20), _d(100, 20)])
    transform = np.array([[1, 0, 5], [0, 1, 0]], dtype=np.float64)
    tracks = tracker.update(0.1, [_d(25, 20), _d(108, 20)], camera_transform=transform)
    assert tracker.stable_track_speed(tracks[0], 10) == pytest.approx(0)
    assert tracker.stable_track_speed(tracks[1], 10) == pytest.approx(3)


def test_expired_tracks_do_not_accumulate_during_a_long_match():
    tracker = BallTracker(max_missed=0.1)
    for i in range(200):
        tracker.update(i, [_d(i * 100, 20)])
    # The current track and a newly expired predecessor are the only residents.
    assert len(tracker.tracks) <= 2


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


def test_glove_connected_to_forearm_is_not_a_ball(config):
    frame = np.full((400, 600, 3), 25, dtype=np.uint8)
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(frame, (100, 60), (500, 340), (35, 105, 35), -1)
    cv2.rectangle(mask, (100, 60), (500, 340), 255, -1)
    # The glove tip lies inside the table, the connected arm crosses its edge.
    cv2.rectangle(frame, (105, 20), (123, 133), (20, 20, 20), -1)
    cv2.circle(frame, (125, 138), 10, (245, 245, 245), -1)
    cv2.circle(frame, (170, 138), 5, (245, 245, 245), -1)
    cv2.circle(frame, (300, 230), 5, (20, 20, 210), -1)
    detector = ObjectDetector(config)
    for _ in range(3):
        found = detector.detect(frame, mask)
        assert not any(np.hypot(d.cx - 125, d.cy - 138) < 14 for d in found)
        assert any(d.label == "cue_ball" and abs(d.cx - 170) < 3 for d in found)
        assert any(abs(d.cx - 300) < 3 and abs(d.cy - 230) < 3 for d in found)


def test_foreground_filter_preserves_touching_red_balls(config, monkeypatch):
    frame = np.full((400, 600, 3), (35, 105, 35), dtype=np.uint8)
    mask = np.full((400, 600), 255, dtype=np.uint8)
    for row in range(4):
        for col in range(row + 1):
            cv2.circle(frame, (300 + 10 * col - 5 * row, 200 + 9 * row), 6, (20, 20, 210), -1)
    # Feed known circular proposals so this checks foreground classification,
    # independently of Hough's response to perfectly flat synthetic balls.
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: np.array([
        [[300, 200, 6], [295, 209, 6], [305, 209, 6]],
    ], dtype=np.float32))
    found = ObjectDetector(config).detect(frame, mask)
    assert any(275 < d.cx < 325 and 190 < d.cy < 240 for d in found)


def test_stationary_ball_covered_by_hand_does_not_inherit_jitter_velocity():
    tracker = BallTracker()
    for i in range(12):
        # Alternating subpixel detector jitter gives a high instantaneous
        # derivative at 30 fps but no coherent rolling trajectory.
        tracker.update(i / 30, [_d(100 + (i % 2) * 0.4, 100)])
    assert tracker.max_normalized_speed() == 0
    assert tracker.tracks[0].predicted_speed() / 10 > 0.1
    tracker.update(12 / 30, [])
    assert tracker.occluded_moving_count() == 0


def test_rolling_ball_covered_by_hand_remains_unresolved():
    tracker = BallTracker()
    for i in range(12):
        tracker.update(i / 30, [_d(100 + i * 0.5, 100)])
    tracker.update(12 / 30, [])
    assert tracker.occluded_moving_count(min_normalized_speed=0.6) == 1
