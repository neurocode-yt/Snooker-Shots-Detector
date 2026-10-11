"""The real nearby-red contact survives noise in other-ball tracker speeds."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, ShotRecord

CASE = json.loads(
    (Path(__file__).parent / "fixtures/classic_hard/saudi/closeup-departure-native.json").read_text(
        encoding="utf-8"
    )
)


def detected(config, rows):
    detector = StrikeDetector(config)
    return detector.detect_candidates(detector.score_frames(rows))


def test_native_nearby_red_launch_outvotes_unreliable_other_ball_speeds(config):
    rows = [FrameFeatures.model_validate(row) for row in CASE["features"]]
    detector = StrikeDetector(config)
    stabilized = detector._stabilized_features(rows)
    index = next(i for i, frame in enumerate(stabilized) if frame.t == 101.68)
    metrics = detector._transition_metrics(stabilized, index)
    assert metrics["pre_ball_quiet_ratio"] == 0
    assert metrics["pre_cue_spatial_quiet_ratio"] == 1
    # The cue approached in the preceding image; by the first moving image
    # the measured approach speed has fallen. Neither image is fabricated.
    assert metrics["cue_approach_speed"] < 2
    assert metrics["pre_cue_approach_speed"] > 2
    candidates = detected(config, rows)
    assert len(candidates) == 1
    lower, upper = CASE["source_contact_bounds"]
    source_contact = candidates[0].timestamp + CASE["source_offset"]
    assert lower - 0.25 <= source_contact <= upper + 0.25
    assert candidates[0].evidence["closeup_addressed_departure_confirmed"] == 1


def test_source_referee_reset_stays_rejected(config):
    rows = [FrameFeatures.model_validate(row) for row in CASE["referee_features"]]
    assert detected(config, rows) == []


def source_red_record(config):
    rows = [FrameFeatures.model_validate(row) for row in CASE["features"]]
    candidates = detected(config, rows)
    candidates = StrikeDetector(config).refine_boundaries(candidates, rows)
    records = SegmentBuilder(config).build(candidates, rows, 120.0)
    assert len(records) == 1
    return records[0]


def test_source_black_and_following_red_survive_overlap_repair(config):
    black = ShotRecord.model_validate(CASE["previous_black_shot"])
    red = source_red_record(config)
    assert red.clip_start < black.clip_end
    builder = SegmentBuilder(config)
    assert builder._independently_supported(red)
    repaired = builder._resolve_overlaps([black, red], source_duration=120.0)
    assert [shot.cue_strike for shot in repaired] == [black.cue_strike, red.cue_strike]


@pytest.mark.parametrize("missing", ["dense_transition_confirmed", "cue_shape_measured"])
def test_unverified_closeup_flag_cannot_protect_an_overlap(config, missing):
    red = source_red_record(config)
    red.evidence[missing] = 0.0
    assert not SegmentBuilder(config)._independently_supported(red)


@pytest.mark.parametrize("missing", ["cue", "native_observation", "live_context"])
def test_source_departure_requires_independent_contact_evidence(config, missing):
    rows = [FrameFeatures.model_validate(row) for row in CASE["features"]]
    if missing == "cue":
        changes = dict(cue_tip_visible=False, cue_contact_score=0.0, cue_approach_speed=0.0)
    elif missing == "native_observation":
        changes = dict(observation_valid=False)
    else:
        changes = dict(broadcast_replay=True)
    rows = [row.model_copy(update=changes) for row in rows]
    assert detected(config, rows) == []
