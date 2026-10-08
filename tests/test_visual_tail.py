"""Glove entry is an end-only obstruction, never a ball-stop measurement."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.config import load_config
from snooker_ai.segmentation.visual_tail import (
    HandComponent, TailObservation, VisualTailDetector, _TailPixels, _confirmed_entries,
)
from snooker_ai.types import FrameFeatures, ShotRecord
from snooker_ai.utils.timebase import TimeMapper


CASES = json.loads((Path(__file__).parent / "fixtures/visual_tail_windows.json").read_text())["cases"]


def source_rows(name="glove_after_red_stop"):
    case = CASES[name]
    rows = [TailObservation(
        t=row["t"], full_table=row["full_table"],
        context=FrameFeatures.model_validate(row["context"]),
        components=[HandComponent(**component) for component in row["components"]],
    ) for row in case["rows"]]
    return case, rows


@pytest.mark.parametrize("name", CASES)
def test_native_source_gloves_and_pale_player_bridges(name):
    case, rows = source_rows(name)
    entries = _confirmed_entries(rows, case["strike"], .42)
    if case["expected_entry"] is None:
        assert entries == []
    else:
        assert len(entries) == 1
        assert entries[0].entry_timestamp == pytest.approx(case["expected_entry"], abs=.001)
        assert entries[0].source_entry_pts == entries[0].entry_timestamp
        assert entries[0].confirmation_timestamp >= entries[0].entry_timestamp+.16
        assert entries[0].confidence == .85


@pytest.mark.parametrize("flag,value", [
    ("cue_tip_visible", True), ("cue_contact_score", .21),
    ("match_context_valid", False), ("broadcast_replay", True),
])
def test_context_vetoes_do_not_become_glove_caps(flag, value):
    case, rows = source_rows()
    for row in rows:
        setattr(row.context, flag, value)
    assert _confirmed_entries(rows, case["strike"], .42) == []


@pytest.mark.parametrize("break_kind", ["missing_context", "missing_image", "camera_cut", "scene_change"])
def test_confirmation_requires_continuous_source_observations(break_kind):
    case, rows = source_rows()
    at = next(i for i, row in enumerate(rows) if row.t >= case["expected_entry"]-.2)
    if break_kind == "missing_context":
        rows[at].context = None
    elif break_kind == "missing_image":
        del rows[at:at+4]
    elif break_kind == "camera_cut":
        rows[at].context.scene_cut_score = .456
    else:
        rows[at].context.camera_scene_id += 1
    assert _confirmed_entries(rows, case["strike"], .42) == []


def test_cut_guard_uses_the_configured_threshold():
    case, rows = source_rows()
    rows[10].context.scene_cut_score = .456
    assert _confirmed_entries(rows, case["strike"], .42) == []
    assert len(_confirmed_entries(rows, case["strike"], .5)) == 1


@pytest.mark.parametrize("clear", [(), (1618.30,), (1618.30, 1618.34), (1., 2.)])
def test_entry_requires_recent_multiple_visible_cloth_observations(clear):
    case, rows = source_rows()
    for row in rows:
        row.components = [replace(component, prior_clear_times=clear) for component in row.components]
    assert _confirmed_entries(rows, case["strike"], .42) == []


def test_masked_bridge_cannot_reappear_as_a_new_hand_entry():
    case, rows = source_rows()
    onset = case["expected_entry"]
    entry_row = next(row for row in rows if abs(row.t-onset) < .001)
    component = next(c for c in entry_row.components if c.entry)
    # The old bridge is visible before a .2s dropout, longer than the native
    # component backtrace allows. Its identity still vetoes this new proposal.
    earlier = next(row for row in rows if abs(row.t-(onset-.28)) < .001)
    earlier.components.append(component)
    assert _confirmed_entries(rows, case["strike"], .42) == []


def test_glove_entry_requires_post_contact_budget():
    case, rows = source_rows()
    assert _confirmed_entries(rows, case["expected_entry"]-1.9, .42) == []


def table_image():
    frame = np.full((360, 640, 3), 20, np.uint8)
    cv2.fillConvexPoly(frame, np.array([[180, 80], [460, 80], [540, 310], [100, 310]]), (40, 140, 40))
    return frame


def pixels_ready(config):
    pixels = _TailPixels(config)
    for i in range(6):
        row = pixels.observe(table_image(), i*.04, FrameFeatures(t=i*.04, camera_scene_id=1))
        assert row.full_table
    return pixels


def test_source_pixels_distinguish_broad_glove_from_balls_cue_skin_and_collar(config):
    def observe(kind):
        pixels = pixels_ready(config)
        frame = table_image()
        if kind == "glove":
            cv2.rectangle(frame, (139, 190), (166, 208), (230, 230, 230), -1)
        elif kind == "ball":
            cv2.circle(frame, (300, 180), 5, (230, 230, 230), -1)
        elif kind == "cue":
            cv2.rectangle(frame, (180, 190), (360, 193), (230, 230, 230), -1)
        elif kind == "skin":
            cv2.rectangle(frame, (139, 190), (166, 208), (150, 180, 240), -1)
        else:
            cv2.rectangle(frame, (300, 140), (390, 240), (20, 20, 20), -1)
            cv2.rectangle(frame, (330, 170), (358, 188), (230, 230, 230), -1)
        return pixels.observe(frame, .24, FrameFeatures(t=.24, camera_scene_id=1))
    glove = [c for c in observe("glove").components if c.broad and c.entry]
    assert glove and len(glove[0].prior_clear_times) >= 2
    for negative in ("ball", "cue", "skin", "collar"):
        assert not any(c.broad and c.entry and c.kind == "glove"
                       for c in observe(negative).components)


@pytest.mark.parametrize("reset_kind", ["missing", "cut", "scene", "pan", "gap", "pixels"])
def test_geometry_and_clearance_reset_together(config, reset_kind):
    pixels = pixels_ready(config)
    frame = table_image()
    feature = FrameFeatures(t=.24, camera_scene_id=1)
    t = .24
    if reset_kind == "missing":
        feature = None
    elif reset_kind == "cut":
        feature.scene_cut_score = .456
    elif reset_kind == "scene":
        feature.camera_scene_id = 2
    elif reset_kind == "pan":
        feature.camera_motion_magnitude = 4
    elif reset_kind == "gap":
        t = 1.
    else:
        frame[:] = 200
    row = pixels.observe(frame, t, feature)
    assert row.reset and not row.full_table
    assert pixels.geometry is None and pixels.green_history == []


def test_pixel_cut_threshold_is_configurable():
    pixels = pixels_ready(load_config(overrides={"scene_detection": {"hard_cut_threshold": .5}}))
    row = pixels.observe(table_image(), .24, FrameFeatures(t=.24, camera_scene_id=1, scene_cut_score=.456))
    assert not row.reset and row.full_table


@pytest.mark.parametrize("kind", ["body_at_rail", "middle_of_table", "packed_balls"])
def test_broad_foreground_requires_connection_to_the_rail(config, kind):
    pixels = pixels_ready(config)
    frame = table_image()
    if kind == "body_at_rail":
        cv2.rectangle(frame, (120, 150), (185, 245), (20, 20, 20), -1)
    elif kind == "middle_of_table":
        cv2.rectangle(frame, (300, 150), (365, 245), (20, 20, 20), -1)
    else:
        for y in (180, 190):
            for x in (295, 305, 315):
                cv2.circle(frame, (x, y), 5, (20, 20, 140), -1)
    row = pixels.observe(frame, .24, FrameFeatures(t=.24, camera_scene_id=1))
    components = [c for c in row.components if c.kind == "foreground"]
    if kind == "body_at_rail":
        assert any(c.broad and c.entry and len(c.prior_clear_times) >= 2 for c in components)
    else:
        assert components == []


def test_feature_context_requires_a_fresh_nearby_observation():
    features = [FrameFeatures(t=1), FrameFeatures(t=3)]
    assert VisualTailDetector._context(features, [1, 3], 1.2) is features[0]
    assert VisualTailDetector._context(features, [1, 3], 2) is None
    assert VisualTailDetector._context([], [], 1) is None


@pytest.mark.parametrize("override", [
    {"included": False}, {"possible_replay": True}, {"user_modified": True},
])
def test_only_automatic_live_ends_are_eligible(override):
    shot = ShotRecord(shot_id=1, cue_strike=1, clip_end=5, end_confidence=.2)
    assert VisualTailDetector.eligible(shot)
    assert not VisualTailDetector.eligible(shot.model_copy(update=override))


def test_confident_stop_still_checks_for_visible_referee_footage():
    shot = ShotRecord(shot_id=1, cue_strike=1, clip_end=5, end_confidence=.95,
                      evidence={"stop_confirmed": True})
    assert VisualTailDetector.eligible(shot)


class FakeCapture:
    def __init__(self, pts):
        self.pts = pts
        self.index = 0

    def set(self, prop, value):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 25.
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts[self.index-1]*1000
        return self.index

    def read(self):
        if self.index == len(self.pts):
            return False, None
        self.index += 1
        return True, table_image()


def test_native_window_uses_actual_presentation_times_with_varying_cadence(config, monkeypatch):
    pts = [3., 3.04, 3.12, 3.16]
    features = [FrameFeatures(t=t) for t in pts]
    detector = VisualTailDetector(config)
    original = copy.deepcopy(features)
    rows = detector._observe_window(FakeCapture(pts), 3, 3.2, TimeMapper(10),
                                    features, pts, native=True)
    assert [row.t for row in rows] == pts
    assert features == original
    assert detector._observe_window(FakeCapture([3., 3.]), 3, 3.2, TimeMapper(10),
                                    features, pts, native=True) == []
