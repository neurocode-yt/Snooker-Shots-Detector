"""Source regressions: the opening dissolve and the first red behind the bridge."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pot_detection.frame_context import FrameContextDetector, FrameRange, frame_ranges
from snooker_ai.pot_detection.scoreboard import ScoreboardReader, ScoreReading
from snooker_ai.pot_detection.selector import PotSelector
from snooker_ai.pot_detection.white_reacquisition import scoring_white_contact
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.table_detection.localizer import TableLocalizer
from snooker_ai.types import FrameFeatures, StrikeCandidate
from tests.test_pot_selection import fixture_result, readings, shot

ROOT = Path(__file__).parent / "fixtures/wilson_thepchaiya_native"
WHITE = json.loads((ROOT / "white.json").read_text())
OPENING = json.loads((ROOT / "opening.json").read_text())


def test_original_overlay_thin_seven_preserves_the_red_between_sixteen_and_twenty_two():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[610:680, 440:840] = cv2.imread(str(ROOT / "score_28_17.png"))
    found = ScoreboardReader().read(frame, 3694)
    assert found is not None and (found.left, found.right) == (28, 17)
    assert found.confidence >= 0.78


@pytest.mark.parametrize("fallback", [((17, 0.9), (18, 0.9)), ((17, 0.9), None)])
def test_thin_digit_fallback_cannot_accept_disagreement_or_weak_readings(monkeypatch, fallback):
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[610:680, 440:840] = cv2.imread(str(ROOT / "score_28_17.png"))
    answers = iter([(28, 0.99), None, *fallback])
    monkeypatch.setattr(
        "snooker_ai.pot_detection.scoreboard.read_digits", lambda *a, **kw: next(answers)
    )
    assert ScoreboardReader().read(frame, 3694) is None


def hint():
    return StrikeCandidate(
        timestamp=135.52,
        confidence=0.5,
        uncertainty_start=126.52,
        uncertainty_end=138.32,
        evidence={"score_assisted_proposal": 1.0, "score_delta": 1.0, "score_update_time": 138.52},
    )


class Capture:
    def __init__(self, defect=None):
        self.index = 0
        self.frames = []
        x, y = WHITE["origin"]
        width, height = WHITE["size"]
        for item in WHITE["frames"]:
            image = np.zeros((height, width, 3), np.uint8)
            patch = cv2.imread(str(ROOT / item["file"]))
            assert patch is not None
            image[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
            self.frames.append((item["t"], image))
        if defect == "no_rest_pixels":
            for _, image in self.frames:
                image[:] = (40, 130, 40)
        if defect == "no_departure_pixels":
            self.frames[-1] = (self.frames[-1][0], self.frames[-5][1].copy())

    def isOpened(self):
        return True

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_MSEC
        self.index = next(
            (i for i, (t, _) in enumerate(self.frames) if t >= value / 1000 - 0.01),
            len(self.frames),
        )

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        self.index += 1
        return True, self.frames[self.index - 1][1].copy()

    def get(self, key):
        assert key == cv2.CAP_PROP_POS_MSEC
        return self.frames[self.index - 1][0] * 1000

    def release(self):
        pass


def contact(config, monkeypatch, defect=None):
    rows = [FrameFeatures.model_validate(f) for f in WHITE["features"]]
    proposal = hint()
    for f in rows:
        if defect == "sparse":
            f.observation_fps = 2
        elif defect == "replay":
            f.broadcast_replay = True
        elif defect == "handling":
            f.table_handling = True
        elif defect == "foreign":
            f.match_context_valid = False
        elif defect == "invalid":
            f.observation_valid = False
        elif defect == "cut":
            f.scene_cut_score = 0.7
        elif defect == "camera":
            f.camera_motion_magnitude = 6
        elif defect == "no_shape":
            f.cue_ball_quality = False
        elif defect == "no_cue":
            f.cue_tip_visible = False
            f.cue_contact_score = 0
        elif defect == "no_travel" and f.cue_ball_x is not None:
            f.cue_ball_x = 550
            f.cue_ball_y = 350
        elif defect == "continuous_track":
            f.cue_ball_detected = f.cue_ball_quality = True
            f.cue_ball_track_confidence = 0.9
            if f.cue_ball_x is None:
                f.cue_ball_x, f.cue_ball_y = 550, 350
    if defect == "no_score":
        proposal.evidence.clear()
    if defect == "large_score":
        proposal.evidence["score_delta"] = 12
    if defect == "gap":
        rows = [f for f in rows if not 135.3 < f.t < 135.5]
    monkeypatch.setattr(
        "snooker_ai.pot_detection.white_reacquisition.open_capture",
        lambda *a, **kw: Capture(defect),
    )
    return scoring_white_contact("fixture", proposal, rows, config)


def test_first_red_has_independent_native_rest_pixels_and_measured_departure(config, monkeypatch):
    found = contact(config, monkeypatch)
    assert found is not None
    assert found.uncertainty_start == pytest.approx(135.08)
    assert found.timestamp == found.uncertainty_end == pytest.approx(135.24)
    assert found.evidence["white_rest_samples"] >= 7
    assert found.evidence["white_rest_changed_fraction"] >= 0.35
    assert found.evidence["white_travel_diameters"] >= 2
    assert SegmentBuilder._candidate_supported(found)


@pytest.mark.parametrize(
    "defect",
    [
        "sparse",
        "replay",
        "handling",
        "foreign",
        "invalid",
        "cut",
        "camera",
        "no_shape",
        "no_cue",
        "no_travel",
        "no_score",
        "large_score",
        "gap",
        "no_rest_pixels",
        "no_departure_pixels",
        "continuous_track",
    ],
)
def test_score_and_reappearance_cannot_replace_rest_motion_and_live_context(
    config, monkeypatch, defect
):
    assert contact(config, monkeypatch, defect) is None


@pytest.mark.parametrize(
    "missing",
    [
        "native_scoring_white_reacquisition",
        "score_assisted_proposal",
        "score_supported_object_launch",
        "contact_time_upper_bound",
        "white_rest_samples",
        "white_rolling_samples",
        "white_travel_diameters",
        "white_direction_consistency",
        "white_rest_changed_fraction",
        "object_launch_confirmation_time",
    ],
)
def test_segmenter_rejects_incomplete_white_reacquisition_proof(config, monkeypatch, missing):
    found = contact(config, monkeypatch)
    assert found is not None
    found.evidence[missing] = 0.0
    assert not SegmentBuilder._candidate_supported(found)


def test_low_angle_dissolve_resets_ball_scale_even_below_half_table_confidence(config):
    views = [
        TableLocalizer(config).detect(cv2.imread(str(ROOT / f"view_{t}.png")))
        for t in ["59.72", "60.28"]
    ]
    assert min(v.confidence for v in views) < 0.5
    assert Analyzer._table_view_changed(*views)
    assert not Analyzer._table_view_changed(views[1], views[1])
    views[1].confidence = 0.1
    assert not Analyzer._table_view_changed(*views)


def test_default_strike_detector_finds_the_opening_break_after_dissolve_reset(config):
    detector = StrikeDetector(config)
    rows = detector.score_frames([FrameFeatures.model_validate(f) for f in OPENING["features"]])
    found = detector.detect_candidates(rows)
    assert [f.timestamp for f in found] == pytest.approx([60.6])
    assert found[0].evidence["cue_ball_motion_confirmed"] == 1
    assert found[0].evidence["cue_geometry_confirmed"] == 1
    assert SegmentBuilder._candidate_supported(found[0])


def test_delayed_zero_overlay_does_not_put_the_next_break_in_the_previous_frame():
    scores = readings((85, 32), start=750, end=800) + readings((0, 0), start=976, end=996)
    scores += readings((0, 1), start=997, end=1010)
    assert frame_ranges(scores, 1100) == [FrameRange(951, 997, 1100)]


def test_source_zero_observations_and_native_opening_survive_filtered_score_windows(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    result.original_duration = result.metadata.duration = 180
    detector = StrikeDetector(config)
    candidate = detector.detect_candidates(
        detector.score_frames([FrameFeatures.model_validate(f) for f in OPENING["features"]])
    )[0]
    opening = shot(60.6, 68)
    opening.evidence.update(candidate.evidence)
    result.shots = [opening, shot(135.24, 140), shot(160, 165)]
    result.features = [
        FrameFeatures(
            t=t, scoreboard_reading=ScoreReading(t, 0, 0, 1, 0.99, "paired_boxes").__dict__
        )
        for t in [70, 70.5]
    ]
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *a: readings((0, 1), start=138, end=175))
    monkeypatch.setattr(FrameContextDetector, "scan_racks", lambda *a: [[]])
    selector.apply(result)
    assert result.shots[0].included and result.shots[-1].included
    assert result.shots[0].evidence["frame_context"]["first"]
    assert result.shots[-1].evidence["frame_context"]["last"]
