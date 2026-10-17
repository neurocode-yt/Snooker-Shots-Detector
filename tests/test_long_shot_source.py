"""Native source proof for long clips containing preparation or two strokes."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures


DATA = json.loads((Path(__file__).parent / "fixtures/classic_hard/long-shot-recheck.json").read_text())


def rows(case=0):
    return [FrameFeatures.model_validate(f) for f in DATA["cases"][case]["features"]]


@pytest.mark.parametrize("case", [0, 1])
def test_native_source_separates_blue_and_keeps_actual_strike_after_preparation(config, case):
    candidates = StrikeDetector(config).detect_candidates(rows(case))
    assert [c.timestamp for c in candidates] == pytest.approx(DATA["cases"][case]["expected_contacts"])


def test_early_cutaway_and_recovered_blue_are_separate_clips(config):
    features = rows()
    detector = StrikeDetector(config)
    shots = SegmentBuilder(config).build(detector.detect_candidates(features), features, 7784.456446)
    assert len(shots) == 2
    first, second = shots
    assert first.clip_end == pytest.approx(1291.7333333333333)
    assert first.evidence["usable_source_end_reason"] == "non_table_cutaway_clip_boundary"
    assert first.physical_stop_timestamp > first.clip_end  # no invented unseen stop
    assert second.clip_start < 1297.2666666666667 < second.clip_end
    assert second.physical_stop_timestamp == pytest.approx(1301.6)
    assert second.clip_end >= 1299.74  # retain the pot, then apply the configured end trim
    assert first.duration() < 7 and second.duration() < 7


def test_recovered_blue_is_independently_revalidated_in_normal_native_pipeline(config):
    features = rows()
    detector = StrikeDetector(config)
    candidates = detector.refine_boundaries(detector.detect_candidates(features), features)
    blue = next(c for c in candidates if abs(c.timestamp-1297.2666666666667) < .001)
    assert blue.evidence["native_occlusion_confirmed"] == 1
    assert blue.evidence["occlusion_inferred"] == 1
    assert blue.evidence["ball_onset_run"] >= 2
    assert SegmentBuilder(config)._candidate_supported(blue)
    # A saved confirmation cannot survive loss of its native address evidence.
    for f in features:
        f.cue_contact_score = 0
    detector.refine_boundaries([blue], features)
    assert blue.evidence["occlusion_inferred"] == 0
    assert blue.evidence["dense_transition_confirmed"] == 0


@pytest.mark.parametrize("corruption", [
    "address", "pre_motion", "replay", "foreign", "handling", "gap",
    "remote_return", "stationary_return", "first_departure", "second_departure",
])
def test_bridge_interruption_requires_both_measured_departure_and_continued_roll(config, corruption):
    features = rows()
    features = [f for f in features if f.t >= 1294.5]
    for f in features:
        if corruption == "address":
            f.cue_contact_score = 0
        elif corruption == "pre_motion" and 1296.7 <= f.t < 1297.25:
            f.cue_ball_normalized_speed = f.cue_ball_stable_normalized_speed = 4
            if f.cue_ball_x is not None:
                f.cue_ball_x += 300*(f.t-1296.7)
        elif corruption == "replay":
            f.broadcast_replay = True
        elif corruption == "foreign":
            f.match_context_valid = False
        elif corruption == "handling":
            f.table_handling = True
        elif corruption in {"remote_return", "stationary_return"} and f.t >= 1297.43:
            if corruption == "remote_return" and f.cue_ball_y is not None:
                f.cue_ball_y += 300
            elif corruption == "stationary_return":
                f.cue_ball_x, f.cue_ball_y = 396., 146.
                f.cue_ball_stable_normalized_speed = 0
        elif (corruption == "first_departure" and abs(f.t-1297.2666666666667) < .001
              or corruption == "second_departure" and abs(f.t-1297.3) < .001):
            f.cue_ball_x, f.cue_ball_y = 196.5, 137.
            f.cue_ball_stable_normalized_speed = 0
    if corruption == "gap":
        features = [f for f in features if not 1297.34 < f.t < 1297.50]
    assert StrikeDetector(config)._brief_identity_interruption_candidates(features) == []


@pytest.mark.parametrize("corruption", [None, "moving_return", "sparse", "missing_source_index", "gap"])
def test_short_native_cutaway_needs_quiet_return_not_just_empty_images(config, corruption):
    features = rows()
    if corruption == "moving_return":
        for f in features:
            if 1294.13 <= f.t <= 1295.14:
                f.moving_ball_count = 1
                f.max_ball_normalized_speed = 3
    elif corruption == "sparse":
        for f in features:
            f.observation_fps = 2
    elif corruption == "missing_source_index":
        for f in features:
            f.source_frame_index = None
    elif corruption == "gap":
        features = [f for f in features if not 1292 <= f.t <= 1293.8]
    spans = SegmentBuilder(config)._non_table_cutaway_spans(features)
    assert bool(spans) is (corruption is None)
