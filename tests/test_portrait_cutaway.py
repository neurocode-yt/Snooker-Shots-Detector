"""Player portraits limit editing footage without declaring a physical stop."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.segmentation.portrait import (
    PortraitCutaway,
    PortraitCutawayDetector,
    portrait_frame,
)
from snooker_ai.types import FrameFeatures, ShotRecord, StrikeCandidate


FIXTURES = Path(__file__).parent / "fixtures/classic_hard"


@pytest.mark.parametrize(
    "timestamp,expected", [(1.0, False), (55.92, False), (154.0, True), (155.0, True)]
)
def test_source_portraits_and_actual_ball_closeups(config, timestamp, expected):
    assert (
        portrait_frame(cv2.imread(str(FIXTURES / f"portrait-{timestamp:.2f}.jpg")), config)
        is expected
    )


def test_missing_optional_cascade_is_a_safe_noop(config, monkeypatch):
    monkeypatch.setattr("snooker_ai.segmentation.portrait._face_cascade", lambda: None)
    frame = cv2.imread(str(FIXTURES / "portrait-154.00.jpg"))
    assert not portrait_frame(frame, config)
    assert (
        PortraitCutawayDetector(config).detect(
            "unused", [ShotRecord(shot_id=1, cue_strike=1, clip_end=10)], [FrameFeatures(t=5)]
        )
        == []
    )


class Capture:
    def __init__(self):
        self.times = [round(5 + i * 0.2, 4) for i in range(6)]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *args):
        pass

    def read(self):
        if self.index >= len(self.times):
            return False, None
        self.current = self.times[self.index]
        self.index += 1
        return True, np.zeros((360, 640, 3), np.uint8)

    def get(self, *args):
        return self.current * 1000

    def release(self):
        self.released = True


@pytest.mark.parametrize("veto", [None, "motion", "replay", "cut", "full_table", "gap", "scene"])
def test_portrait_confirmation_requires_continuous_quiet_valid_context(config, monkeypatch, veto):
    capture = Capture()
    monkeypatch.setattr("snooker_ai.segmentation.portrait.open_capture", lambda _: capture)
    monkeypatch.setattr("snooker_ai.segmentation.portrait._face_cascade", lambda: object())
    monkeypatch.setattr("snooker_ai.segmentation.portrait.portrait_frame", lambda *_: True)
    features = [FrameFeatures(t=t, table_full_view=False, camera_scene_id=1) for t in capture.times]
    if veto == "gap":
        del features[2:4]
    elif veto == "scene":
        for f in features[2:]:
            f.camera_scene_id = 2
    else:
        for f in features:
            if veto == "motion":
                f.moving_ball_count = 1
            elif veto == "replay":
                f.broadcast_replay = True
            elif veto == "cut":
                f.scene_cut_score = 0.8
            elif veto == "full_table":
                f.table_full_view = True
    shot = ShotRecord(shot_id=1, cue_strike=1, clip_end=10, manual_review_required=True)
    found = PortraitCutawayDetector(config).detect("unused", [shot], features)
    assert capture.released
    assert bool(found) is (veto in (None, "scene"))
    if found:
        assert found[0] == (
            PortraitCutaway(1, 5.4, 6.0) if veto == "scene" else PortraitCutaway(1, 5, 5.6)
        )


@pytest.mark.parametrize(
    "override", [{"included": False}, {"possible_replay": True}, {"user_modified": True}]
)
def test_end_pass_preserves_excluded_replay_and_user_edits(config, monkeypatch, override):
    monkeypatch.setattr(
        "snooker_ai.segmentation.portrait.open_capture",
        lambda _: pytest.fail("ineligible source read"),
    )
    shot = ShotRecord(shot_id=1, cue_strike=1, clip_end=10, manual_review_required=True).model_copy(
        update=override
    )
    assert PortraitCutawayDetector(config).detect("unused", [shot], [FrameFeatures(t=5)]) == []


def test_portrait_cap_preserves_physical_stop_and_export_contract(config, tmp_path):
    candidate = StrikeCandidate(
        timestamp=2,
        confidence=0.95,
        evidence={
            "dense_transition_confirmed": 1.0,
            "refined_stop_timestamp": 12.0,
            "refined_stop_confirmation_timestamp": 12.6,
            "refined_stop_confidence": 0.95,
            "refined_ball_motion_start": 2.0,
            "refined_stop_review_required": 1.0,
        },
    )
    features = [
        FrameFeatures(
            t=i * 0.1,
            ball_count=8,
            ball_kinematics_valid=True,
            moving_ball_count=int(2 <= i * 0.1 <= 3),
            max_ball_normalized_speed=1 if 2 <= i * 0.1 <= 3 else 0,
        )
        for i in range(151)
    ]
    builder = SegmentBuilder(config)
    shots = builder.build([candidate], features, 15)
    assert Analyzer(config, tmp_path)._record_portrait_boundaries(
        [candidate], shots, [PortraitCutaway(2, 7, 7.6)], 15, 25
    )
    edited = builder.build([candidate], features, 15)
    assert edited[0].cue_strike == 2
    assert edited[0].physical_stop_timestamp == 12
    assert edited[0].clip_end == pytest.approx(6.96)
    assert edited[0].manual_review_required
    assert edited[0].evidence["usable_source_end_reason"] == "portrait_cutaway_clip_boundary"
    Exporter(config)._validate_strict_boundaries(edited, source_duration=15, source_fps=25)


@pytest.mark.parametrize("broken_pts", [False, True])
def test_native_backtrace_uses_observed_first_portrait_and_rejects_bad_timestamps(
    config, monkeypatch, broken_pts
):
    capture = Capture()
    capture.times = [4.80, 4.84, 4.88, 4.92, 4.96, 5.00, 5.04]
    if broken_pts:
        capture.times[4] = 4.92
    monkeypatch.setattr(
        "snooker_ai.segmentation.portrait.portrait_frame", lambda *_: capture.current >= 4.92
    )
    start = PortraitCutawayDetector(config)._native_start(capture, 5.04, 4.0)
    assert start == pytest.approx(5.04 if broken_pts else 4.92)
