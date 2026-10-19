"""Frame context is selected independently of whether a stroke scored points."""

import json
from pathlib import Path

import pytest
import numpy as np

from snooker_ai.jobs.store import JobStore
from snooker_ai.pot_detection.frame_context import (
    FrameContextDetector,
    FrameRange,
    frame_ranges,
    rack_cut_contact,
)
from snooker_ai.pot_detection.selector import PotSelector
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, PotStatus, ShotSelection, TimelineEvent
from tests.test_pot_selection import fixture_result, readings, shot


def samples():
    return (
        readings((0, 0), start=0, end=19.5)
        + readings((1, 0), start=20, end=54.5)
        + readings((0, 0), start=55, end=79.5, active=1)
        + readings((0, 1), start=80, end=120, active=1)
    )


def result_with_frames(tmp_path):
    result = fixture_result(tmp_path)
    result.original_duration = result.metadata.duration = 120
    result.shots = [
        shot(t, t + 3).model_copy(update={"shot_id": i})
        for i, t in enumerate([5, 18, 30, 45, 65, 78, 90, 110], 1)
    ]
    return result


def detector_racks(monkeypatch):
    monkeypatch.setattr(
        FrameContextDetector, "scan_racks", lambda *args: [[[4.5, 5.5]], [[64.5, 65.5]]]
    )


def test_score_resets_identify_two_frames_without_splitting_repeated_zero_runs():
    assert frame_ranges(samples(), 120) == [FrameRange(0, 20, 55), FrameRange(55, 80, 120)]
    split = readings((0, 0), start=0, end=5) + readings((0, 0), start=15, end=19.5)
    assert frame_ranges(split + readings((1, 0), start=20), 40) == [FrameRange(0, 20, 40)]


def test_temporary_zero_overlay_returning_to_old_score_is_not_a_frame():
    data = (
        readings((40, 20), start=0, end=10)
        + readings((0, 0), start=11, end=13)
        + readings((40, 20), start=14, end=22)
    )
    assert frame_ranges(data, 30) == []
    assert frame_ranges([], 30) == []
    assert frame_ranges(readings((20, 10)), 30) == []


@pytest.mark.parametrize("selection", [ShotSelection.POTS_ONLY, ShotSelection.POTS_AND_FOULS])
def test_keeps_each_opening_and_last_shot_with_scoring_shots_and_exact_cuts(
    config, tmp_path, monkeypatch, selection
):
    result = result_with_frames(tmp_path)
    original = [(s.clip_start, s.clip_end) for s in result.shots]
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *args: samples())
    detector_racks(monkeypatch)
    selector.apply(result, selection=selection)
    assert [s.shot_id for s in result.shots if s.included] == [1, 2, 4, 5, 6, 8]
    assert result.shots[0].pot_status is PotStatus.NO_POT  # context never becomes a pot
    assert not result.shots[2].included
    assert not result.shots[6].included
    assert [(s.clip_start, s.clip_end) for s in result.shots] == original
    assert result.pot_selection_summary["frame_context_shots"] == 4
    assert result.pot_selection_summary["non_scoring_context_shots"] == 4
    assert len(result.pot_selection_summary["frames"]) == 2
    selector.apply(result, selection=selection)
    assert [s.shot_id for s in result.shots if s.included] == [1, 2, 4, 5, 6, 8]


def test_replays_base_exclusions_and_explicit_manual_exclusions_stay_excluded(
    config, tmp_path, monkeypatch
):
    result = result_with_frames(tmp_path)
    result.shots[0].included = False
    result.shots[0].evidence["pots_only_manual_override"] = True
    result.shots[3].possible_replay = True
    result.shots[6].included = False
    result.shots[7].evidence["cue_geometry_confirmed"] = 0.0
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *args: samples())
    detector_racks(monkeypatch)
    monkeypatch.setattr(FrameContextDetector, "recover_opening", lambda *args: None)
    selector.apply(result)
    assert not result.shots[0].included
    assert not result.shots[3].included
    assert not result.shots[6].included
    assert not result.shots[7].included
    assert result.shots[6].evidence["frame_context"]["last"]


def test_setup_contact_before_rack_is_not_the_opening(config, tmp_path, monkeypatch):
    result = result_with_frames(tmp_path)
    result.shots.insert(0, shot(1, 3).model_copy(update={"shot_id": 99}))
    detector_racks(monkeypatch)
    context = FrameContextDetector(config, tmp_path).identify(result, samples())
    assert context[0]["first_shot_id"] == 1
    assert "frame_context" not in result.shots[0].evidence


def test_missing_opening_is_inserted_once_and_event_ids_follow_new_shot_numbers(
    config, tmp_path, monkeypatch
):
    result = result_with_frames(tmp_path)
    result.shots = result.shots[1:]
    result.events = [
        TimelineEvent(
            event_type="cue_strike", timestamp=s.cue_strike, metadata={"shot_id": s.shot_id}
        )
        for s in result.shots
    ]
    detector_racks(monkeypatch)
    monkeypatch.setattr(FrameContextDetector, "recover_opening", lambda *args: shot(5, 8))
    detector = FrameContextDetector(config, tmp_path)
    frames = detector.identify(result, samples())
    assert frames[0]["first_shot_id"] == 1
    assert len(result.shots) == 8
    assert len([e for e in result.events if e.event_type == "cue_strike"]) == 8
    for event in result.events:
        s = next(s for s in result.shots if s.shot_id == event.metadata["shot_id"])
        assert event.timestamp == (
            s.cue_strike if event.event_type == "cue_strike" else s.physical_stop_timestamp
        )
    detector.identify(result, samples())
    assert len(result.shots) == 8


def test_unidentified_frame_and_stale_context_cannot_keep_an_unknown_shot(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    result.shots[0].evidence["frame_context"] = {"frame": 1, "first": True}
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *args: [])
    selector.apply(result)
    assert not result.shots[0].included
    assert "frame_context" not in result.shots[0].evidence


def native_samples():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/classic_hard/frame-opening-cut/source.json").read_text()
    )
    return [FrameFeatures.model_validate(f) for f in fixture["features"]]


def test_native_opening_contact_survives_the_broadcast_camera_cut():
    contact = rack_cut_contact(native_samples(), 1513.02)
    assert contact is not None
    assert contact.timestamp == pytest.approx(1512.76)
    assert contact.uncertainty_start == pytest.approx(1512.72)
    assert SegmentBuilder._candidate_supported(contact)


def test_default_detector_recovers_the_same_non_scoring_opening_without_adding_duplicates(
    config, tmp_path
):
    from snooker_ai.pipeline.analyzer import Analyzer
    from dataclasses import asdict
    from snooker_ai.pot_detection.scoreboard import ScoreReading

    owner = Analyzer(config, tmp_path)
    reference = [
        FrameFeatures(
            t=1500, scoreboard_reading=asdict(ScoreReading(1500, 0, 0, 0, 0.99, "paired_boxes"))
        ),
        FrameFeatures(
            t=1500.5, scoreboard_reading=asdict(ScoreReading(1500.5, 0, 0, 0, 0.99, "paired_boxes"))
        ),
    ]
    reference.extend(
        FrameFeatures(t=t, rack_observation_valid=True, red_rack_intact=True)
        for t in [1513.02, 1513.52]
    )
    found = owner._recover_frame_openings([], native_samples(), reference, 1530)
    assert len(found) == 1
    assert found[0].timestamp == pytest.approx(1512.76)
    assert len(owner._recover_frame_openings(found, native_samples(), reference, 1530)) == 1
    reference[-1].broadcast_replay = True
    assert not owner._recover_frame_openings([], native_samples(), reference, 1530)


@pytest.mark.parametrize(
    "corruption",
    [
        "replay",
        "foreign",
        "handling",
        "invalid",
        "sparse",
        "no_cue",
        "no_rest",
        "one_moving_image",
        "no_cut",
        "partial_table",
        "gap",
        "unmeasured_ball",
        "no_departure",
    ],
)
def test_rack_and_camera_cut_cannot_manufacture_a_contact(corruption):
    frames = native_samples()
    for f in frames:
        if corruption == "replay":
            f.broadcast_replay = True
        elif corruption == "foreign":
            f.match_context_valid = False
        elif corruption == "handling":
            f.table_handling = True
        elif corruption == "invalid":
            f.observation_valid = False
        elif corruption == "sparse":
            f.observation_fps = 2
        elif corruption == "no_cue":
            f.cue_contact_score = 0
        elif corruption == "no_rest" and f.t < 1512.76:
            f.cue_ball_x += (f.t % 0.1) * 100
        elif corruption == "no_cut":
            f.camera_scene_id = 0
        elif corruption == "partial_table":
            f.table_full_view = False
        elif corruption == "unmeasured_ball":
            f.cue_ball_quality = False
        elif corruption == "no_departure" and f.t == 1512.76:
            f.cue_ball_x = 312.5
        elif corruption == "one_moving_image" and f.t > 1512.88:
            f.cue_ball_y = 164.8
    if corruption == "gap":
        frames = [f for f in frames if not 1512.64 < f.t < 1512.8]
    assert rack_cut_contact(frames, 1513.02) is None


@pytest.mark.parametrize(
    "missing",
    [
        "frame_opening_cut_contact",
        "opening_rest_samples",
        "opening_post_samples",
        "opening_departure_diameters",
        "opening_post_travel_diameters",
        "opening_post_direction_consistency",
        "opening_rack_timestamp",
    ],
)
def test_segment_builder_requires_the_complete_opening_proof(missing):
    contact = rack_cut_contact(native_samples(), 1513.02)
    contact.evidence.pop(missing)
    assert not SegmentBuilder._candidate_supported(contact)


def test_frame_cache_is_carried_to_a_saved_selection(config, tmp_path):
    result = result_with_frames(tmp_path)
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    store = JobStore(config)
    parent = store.create(result.source_path)
    result.job_id = parent
    store.save_analysis(result)
    payload = json.dumps({"signature": "source-and-intervals", "racks": [[[4.5, 5.5]]]})
    (store.job_dir(parent) / "frame_context_cache.json").write_text(payload)
    child = store.create(result.source_path, selection=ShotSelection.POTS_AND_FOULS)
    store.seed_selection_analysis(child, parent)
    assert (store.job_dir(child) / "frame_context_cache.json").read_text() == payload


def test_rack_scan_resumes_completed_frames_and_never_decodes_a_complete_cache(
    config, tmp_path, monkeypatch
):
    from snooker_ai.pot_detection import frame_context

    result = result_with_frames(tmp_path)
    detector = FrameContextDetector(config, tmp_path)
    ranges = [FrameRange(0, 20, 55), FrameRange(55, 80, 120)]
    calls = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, key):
            return 25 if key == frame_context.cv2.CAP_PROP_FPS else 0

        def set(self, *args):
            return True

        def release(self):
            pass

    class Gate:
        racked = True

        def observe(self, *args):
            pass

    fail = True

    def stream(cap, **kwargs):
        calls.append(kwargs["start_time"])
        if fail and kwargs["start_time"] == 55:
            raise RuntimeError("paused")
        yield 0, kwargs["start_time"] + 4.5, np.zeros((8, 8, 3), np.uint8)
        yield 1, kwargs["start_time"] + 5, np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr(frame_context, "open_capture", lambda *a, **kw: Capture())
    monkeypatch.setattr(frame_context, "sampled_frames", stream)
    monkeypatch.setattr(frame_context, "RackIdleGate", Gate)
    with pytest.raises(RuntimeError, match="paused"):
        detector.scan_racks(result, ranges)
    cached = json.loads((tmp_path / "frame_context_cache.json").read_text())
    assert len(cached["racks"]) == 1
    fail = False
    assert detector.scan_racks(result, ranges) == [[[4.5, 5]], [[59.5, 60]]]
    assert calls == [0, 55, 55]
    monkeypatch.setattr(
        frame_context, "open_capture", lambda *a, **kw: pytest.fail("decoded complete cache")
    )
    assert len(detector.scan_racks(result, ranges)) == 2


def test_one_available_shot_can_be_both_frame_bookends_without_duplication(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    result.shots = [shot(5, 8)]
    monkeypatch.setattr(FrameContextDetector, "scan_racks", lambda *a: [[[4.5, 5.5]]])
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *a: readings((0, 0), start=0, end=30))
    selector.apply(result)
    assert result.shots[0].included
    assert result.shots[0].evidence["frame_context"] == {"frame": 1, "first": True, "last": True}
    assert result.pot_selection_summary["frame_context_shots"] == 1
