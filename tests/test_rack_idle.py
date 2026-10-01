import cv2
import numpy as np

from snooker_ai.event_fusion.rack_idle import RackIdleGate
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import FrameFeatures, StrikeCandidate


def racked_frame(cue_x=190):
    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    frame[20:250, 100:380] = (30, 130, 30)
    cv2.fillConvexPoly(frame, np.array([[240, 170], [254, 188], [226, 188]]), (15, 15, 190))
    cv2.circle(frame, (cue_x, 80), 4, (245, 245, 245), -1)
    return frame


def test_rack_wait_wakes_on_cue_launch_and_keeps_restart_evidence():
    gate = RackIdleGate()
    frame = racked_frame()
    for t in np.arange(0, 12, .5):
        gate.observe(frame, float(t))
    assert gate.idle
    assert not gate.observe(racked_frame(cue_x=220), 12)
    assert gate.just_released


def test_rack_wait_does_not_cross_missing_frames_or_hidden_table():
    gate = RackIdleGate()
    for i in range(10):
        gate.observe(racked_frame(), i / 2)
    assert gate.idle
    assert not gate.observe(racked_frame(), 10)
    assert not gate.observe(np.zeros((270, 480, 3), np.uint8), 10.5)


def test_scattered_reds_are_not_a_waiting_rack():
    frame = racked_frame()
    cv2.circle(frame, (150, 210), 4, (15, 15, 190), -1)
    gate = RackIdleGate()
    for i in range(30):
        assert not gate.observe(frame, i / 2)


def test_wait_intervals_preserve_restart_and_gaps():
    frames = [FrameFeatures(t=t, rack_idle=idle) for t, idle in
              [(0, True), (.5, True), (1, False), (1.5, True), (2, True), (8, True)]]
    assert Analyzer._rack_wait_intervals(frames) == [(0, .5), (1.5, 2), (8, 8)]


def test_hand_placed_white_before_break_off_is_rejected():
    frames = [FrameFeatures(t=i / 2, red_rack_intact=True) for i in range(10)]
    candidate = StrikeCandidate(timestamp=1, confidence=.9)
    assert not Analyzer._rack_candidate_supported(candidate, frames)
    frames[4].red_rack_intact = False
    assert Analyzer._rack_candidate_supported(candidate, frames)


def test_visible_cue_contact_is_kept_even_if_break_off_misses_rack():
    frames = [FrameFeatures(t=i / 2, red_rack_intact=True) for i in range(10)]
    candidate = StrikeCandidate(timestamp=1, confidence=.9, evidence={"cue_geometry_confirmed": 1})
    assert Analyzer._rack_candidate_supported(candidate, frames)
    assert Analyzer._rack_candidate_supported(StrikeCandidate(timestamp=1, confidence=.9), [])


def test_returning_reds_then_racking_marks_preparation_not_last_live_shot():
    frames = [FrameFeatures(t=i / 2, rack_observation_valid=True,
                            red_area_ratio=0.0001 if i < 40 else .006,
                            red_rack_intact=i >= 100) for i in range(110)]
    intervals = Analyzer._preparation_intervals(frames)
    assert intervals == [(15.5, 50)]
    assert not any(start <= 10 <= end for start, end in intervals)
    frames[90].scene_cut_score = 1
    assert Analyzer._preparation_intervals(frames) == []
