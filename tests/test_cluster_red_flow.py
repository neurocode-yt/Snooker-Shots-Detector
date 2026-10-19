"""Native mechanical-rest red pot with touching balls and camera zoom."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.config import load_config
from snooker_ai.pot_detection.cluster_flow import ClusterRedFlow
from snooker_ai.pot_detection.depth_launch import scoring_depth_contact
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate


FOLDER = Path(__file__).parent / "fixtures/classic_hard/rest-red-flow"
CASE = json.loads((FOLDER / "source.json").read_text())


def samples():
    with np.load(FOLDER / "frames.npz", allow_pickle=False) as data:
        frames = data["frames"].copy()
    return list(zip(frames, [FrameFeatures.model_validate(f) for f in CASE["features"]]))


def observe(config, rows):
    verifier = ClusterRedFlow(config)
    for frame, context in rows:
        launch = verifier.observe(frame, context)
        if launch is not None:
            return launch
    return None


def proposal():
    return StrikeCandidate(
        timestamp=1963.01,
        confidence=0.5,
        uncertainty_start=1954.01,
        uncertainty_end=1965.81,
        evidence={
            "score_assisted_proposal": 1.0,
            "score_delta": 1.0,
            "score_update_time": CASE["score_update_time"],
        },
    )


class SourceCapture:
    def __init__(self):
        self.rows, self.index, self.released = samples(), -1, False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        self.index += 1
        return (True, self.rows[self.index][0]) if self.index < len(self.rows) else (False, None)

    def get(self, prop):
        return CASE["fps"] if prop == cv2.CAP_PROP_FPS else self.rows[self.index][1].t * 1000

    def release(self):
        self.released = True


@pytest.fixture(scope="module")
def confirmed_contact():
    import snooker_ai.pot_detection.depth_launch as module

    capture = SourceCapture()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "open_capture", lambda _: capture)
        contact = scoring_depth_contact(
            "source", proposal(), [f for _, f in samples()], load_config()
        )
    assert contact is not None and capture.released
    assert contact.evidence.get("native_scoring_red_flow") == 1
    return contact


def test_real_red_flow_has_stationary_witnesses_and_continuous_departure(config):
    launch = observe(config, samples())
    assert launch is not None
    assert (
        CASE["expected_motion_upper_range"][0]
        <= launch["upper"]
        <= CASE["expected_motion_upper_range"][1]
    )
    assert launch["rest_frames"] >= 7 and launch["moving_frames"] >= 4
    assert launch["reference_spheres"] >= 2 and launch["reference_points"] >= 6
    assert launch["feature_count"] >= 4 and launch["travel_diameters"] >= 0.65
    assert launch["registration_residual_diameters"] <= 0.1


def test_native_wrapper_retains_the_red_through_normal_stop_and_trim(config, confirmed_contact):
    fs = [
        FrameFeatures.model_validate(f)
        for f in CASE["original_features"] + CASE["segmentation_tail"]
    ]
    shots = SegmentBuilder(config).build([confirmed_contact.model_copy(deep=True)], fs, 6759.658667)
    assert len(shots) == 1 and shots[0].included
    assert shots[0].clip_start < 1961.24 < 1961.36 < shots[0].clip_end
    assert 4 <= shots[0].duration() <= 6


@pytest.mark.parametrize(
    "corruption",
    [
        "replay",
        "handling",
        "foreign",
        "rack",
        "invalid",
        "sparse",
        "zero_cadence",
        "other_view",
        "missing_transform",
        "singular_transform",
        "bad_transform",
        "large_camera",
        "no_white",
        "poor_white",
        "moving_white",
        "no_contact",
        "scene_change",
        "cut",
        "gap",
        "still_pixels",
        "single_jump",
        "uncorrected_camera",
    ],
)
def test_motion_cannot_bypass_source_context_registration_and_rest(config, corruption):
    rows = samples()
    for i, (frame, f) in enumerate(rows):
        if corruption == "replay":
            f.broadcast_replay = True
        elif corruption == "handling":
            f.table_handling = True
        elif corruption == "foreign":
            f.match_context_valid = False
        elif corruption == "rack":
            f.rack_idle = True
        elif corruption == "invalid":
            f.observation_valid = False
        elif corruption == "sparse":
            f.observation_fps = 10
        elif corruption == "zero_cadence":
            f.observation_fps = 0
        elif corruption == "other_view":
            f.view_type = "main_table"
        elif corruption == "missing_transform":
            f.camera_frame_transform = []
        elif corruption == "singular_transform":
            f.camera_frame_transform = [0.0] * 6
        elif corruption == "bad_transform":
            f.camera_frame_transform = [1.1, 0, 0, 0, 1.1, 0]
        elif corruption == "large_camera":
            f.camera_motion_magnitude = 2
        elif corruption == "no_white":
            f.cue_ball_detected = False
        elif corruption == "poor_white":
            f.cue_ball_quality = False
        elif corruption == "moving_white":
            f.cue_ball_stable_normalized_speed = 2
        elif corruption == "no_contact":
            f.cue_contact_score = 0
        elif corruption == "scene_change" and f.t >= 1961.2:
            f.camera_scene_id += 1
        elif corruption == "cut" and f.t >= 1961.2:
            f.scene_cut_score = 0.9
        elif corruption == "still_pixels":
            rows[i] = (rows[0][0], f)
        elif corruption == "single_jump":
            rows[i] = (rows[0][0] if f.t < 1961.28 else rows[-1][0], f)
        elif corruption == "uncorrected_camera":
            f.camera_frame_transform = [1, 0, 0, 0, 1, 0]
    if corruption == "gap":
        rows = [(p, f) for p, f in rows if not 1961.12 < f.t < 1961.4]
    assert observe(config, rows) is None


def test_pure_camera_motion_with_the_real_zoom_transforms_is_not_a_stroke(config):
    rows = samples()
    cumulative = np.eye(3)
    base = rows[0][0]
    for i, (_, context) in enumerate(rows):
        if i:
            step = np.eye(3)
            step[:2] = np.asarray(context.camera_frame_transform).reshape(2, 3)
            cumulative = step @ cumulative
        frame = cv2.warpAffine(
            base, cumulative[:2], (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT
        )
        rows[i] = frame, context
    assert observe(config, rows) is None


def test_red_cannot_be_confirmed_without_two_independent_reference_spheres(config, monkeypatch):
    original = ClusterRedFlow._circles
    monkeypatch.setattr(
        ClusterRedFlow,
        "_circles",
        lambda self, gray, hsv, cloth, red, diameter, bounds=None: (
            [] if bounds is None else original(self, gray, hsv, cloth, red, diameter, bounds)
        ),
    )
    assert observe(config, samples()) is None


@pytest.mark.parametrize(
    "missing",
    [
        "native_scoring_red_flow",
        "score_assisted_proposal",
        "score_supported_object_launch",
        "contact_time_upper_bound",
        "score_delta",
        "red_flow_rest_frames",
        "red_flow_moving_frames",
        "red_flow_feature_count",
        "red_flow_reference_points",
        "red_flow_reference_spheres",
        "red_flow_travel_diameters",
        "red_flow_direction_consistency",
        "red_flow_registration_residual_diameters",
        "red_flow_feature_spread_diameters",
        "red_flow_contact_score",
        "object_launch_confirmation_time",
        "red_flow_motion_start",
    ],
)
def test_scored_red_requires_complete_measured_flow_proof(confirmed_contact, missing):
    contact = confirmed_contact.model_copy(deep=True)
    contact.evidence.pop(missing)
    assert not SegmentBuilder._candidate_supported(contact)
