"""Successful-pot selection must not interpret fouls or frame counts as pots."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.pot_detection.scoreboard import ScoreboardReader, ScoreReading
from snooker_ai.pot_detection.selector import PotSelector, classify_shot, stable_scores
from snooker_ai.types import (
    AnalysisResult,
    PotStatus,
    ShotRecord,
    ShotSelection,
    VideoMetadata,
    ShotUpdate,
)


def readings(score, active=0, start=8, end=22):
    return [
        ScoreReading(t / 2, *score, active, 0.95, "paired_boxes")
        for t in range(round(start * 2), round(end * 2) + 1)
    ]


def shot(t=10, end=15):
    return ShotRecord(
        shot_id=1,
        cue_strike=t,
        physical_stop_timestamp=end,
        clip_start=t - 2,
        clip_end=end,
        included=True,
    )


@pytest.mark.parametrize("points", range(1, 8))
def test_striker_score_increase_confirms_all_ball_values(points):
    samples = readings((20, 12), end=9.5) + readings((20 + points, 12), start=11)
    status, evidence = classify_shot(shot(), samples, 22)
    assert status is PotStatus.POT
    assert evidence["points"] == points


@pytest.mark.parametrize("points", range(4, 8))
def test_foul_points_awarded_to_opponent_are_excluded(points):
    samples = readings((20, 12), end=9.5) + readings((20, 12 + points), start=11, active=1)
    status, evidence = classify_shot(shot(), samples, 22)
    assert status is PotStatus.FOUL
    assert evidence["reason"] == "opponent_received_points"


def test_delayed_score_after_motion_end_is_still_a_pot():
    samples = readings((20, 12), end=18.5) + readings((27, 12), start=19)
    assert classify_shot(shot(), samples, 22)[0] is PotStatus.POT


def test_stable_unchanged_score_is_a_miss():
    assert classify_shot(shot(), readings((20, 12)), 22)[0] is PotStatus.NO_POT


def test_score_sampling_at_one_fps_retains_stable_readings():
    samples = [ScoreReading(t, 20, 12, 0, 0.95, "paired_boxes") for t in [7, 8, 9]]
    samples += [ScoreReading(t, 27, 12, 0, 0.95, "paired_boxes") for t in range(11, 23)]
    assert classify_shot(shot(), samples, 22, max_score_gap=1.6)[0] is PotStatus.POT


def test_one_point_for_other_side_is_ambiguous_instead_of_being_labeled_a_foul():
    samples = readings((20, 12), end=9.5) + readings((20, 13), start=11)
    status, evidence = classify_shot(shot(), samples, 22)
    assert status is PotStatus.UNKNOWN
    assert evidence["reason"] == "turn_indicator_conflicts_with_score"


def test_absent_score_after_stroke_is_unknown_not_a_miss():
    status, _ = classify_shot(shot(), readings((20, 12), end=12), 22)
    assert status is PotStatus.UNKNOWN


def test_quick_next_shot_does_not_confirm_miss_before_score_delay_window_finishes():
    status, evidence = classify_shot(shot(), readings((20, 12), end=16), 16, outcome_deadline=23)
    assert status is PotStatus.UNKNOWN
    assert evidence["reason"] == "next_shot_or_video_end_prevented_score_confirmation"


def test_no_turn_indicator_cannot_confuse_a_black_with_a_seven_point_foul():
    samples = readings((20, 12), active=None, end=9.5) + readings((27, 12), active=None, start=11)
    assert classify_shot(shot(), samples, 22)[0] is PotStatus.UNKNOWN


def test_recent_turn_indicator_wins_after_a_safety_without_score_change():
    samples = readings((20, 12), active=0, start=4, end=7) + readings(
        (20, 12), active=1, start=7.5, end=9.5
    )
    assert stable_scores(samples)[0].active_player == 1
    samples += readings((20, 19), active=1, start=11)
    assert classify_shot(shot(), samples, 22)[0] is PotStatus.POT


def test_one_frame_digit_glitch_does_not_keep_a_miss():
    samples = readings((20, 12))
    samples[10] = ScoreReading(13, 27, 12, 0, 0.95, "paired_boxes")
    assert classify_shot(shot(), samples, 22)[0] is PotStatus.NO_POT


def test_next_shot_score_cannot_be_assigned_to_previous_miss():
    samples = readings((20, 12), end=21.5) + readings((27, 12), start=22)
    assert classify_shot(shot(), samples, 21.9)[0] is PotStatus.NO_POT
    assert (
        classify_shot(shot(22, 25), samples, 28)[0] is PotStatus.UNKNOWN
    )  # update is at contact, not evidence of this stroke


@pytest.mark.parametrize(
    "after,reason",
    [
        ((19, 12), "score_correction_during_outcome"),
        ((24, 16), "both_players_scored"),
        ((40, 12), "score_jump_not_a_single_scoring_stroke"),
    ],
)
def test_reset_correction_and_merged_score_changes_are_ambiguous(after, reason):
    samples = readings((20, 12), end=9.5) + readings(after, start=11)
    status, evidence = classify_shot(shot(), samples, 22)
    assert status is PotStatus.UNKNOWN
    assert evidence["reason"] == reason


def test_confirmed_final_black_is_kept_before_next_frame_scores_reset():
    samples = (
        readings((70, 30), end=9.5)
        + readings((77, 30), start=11, end=18)
        + readings((0, 0), start=18.5)
    )
    assert classify_shot(shot(), samples, 22)[0] is PotStatus.POT


def test_replay_is_excluded_even_when_scores_change():
    candidate = shot().model_copy(update={"possible_replay": True})
    samples = readings((20, 12), end=9.5) + readings((27, 12), start=11)
    assert classify_shot(candidate, samples, 22)[0] is PotStatus.NO_POT


@pytest.mark.parametrize(
    "name,score,active",
    [
        ("lilley", (0, 0), 0),
        ("holt", (8, 0), 0),
        ("holt_8_9", (8, 9), 0),
        ("zhao", (0, 0), 1),
        ("allen", (0, 0), 1),
        ("wst_hand_2_0", (2, 0), 1),
        ("wst_54_30", (54, 30), 1),
        ("wst_59_22", (59, 22), 0),
        ("wst_60_50", (60, 50), 0),
    ],
)
def test_native_broadcast_overlay_layouts(name, score, active):
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "pot_scoreboards" / f"{name}.png"))
    result = ScoreboardReader().read(frame, 50)
    assert result is not None
    assert (result.left, result.right) == score
    assert result.active_player == active
    if name == "allen":
        assert result.layout == "outer_scores"  # centre yellow numbers are frame counts


def score_frame(left, right):
    frame = np.zeros((720, 1280, 3), np.uint8)
    for x, value in [(540, left), (686, right)]:
        cv2.rectangle(frame, (x, 610), (x + 60, 640), (255, 255, 255), -1)
        text = str(value)
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        cv2.putText(
            frame,
            text,
            (x + (60 - width) // 2, 633),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 38, 64, 99, 111, 147])
def test_multi_digit_points_are_read_without_using_frame_counts(value):
    result = ScoreboardReader().read(score_frame(value, 37), 0)
    assert result is not None
    assert (result.left, result.right) == (value, 37)


def test_no_overlay_does_not_invent_zero_scores():
    assert ScoreboardReader().read(np.zeros((720, 1280, 3), np.uint8), 0) is None


def fixture_result(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return AnalysisResult(
        job_id="fixture",
        source_path=str(source),
        original_duration=30,
        metadata=VideoMetadata(path=str(source), duration=30, fps=25),
        shots=[shot()],
    )


def test_unknown_is_excluded_and_manual_override_survives_repeat_selection(
    config, tmp_path, monkeypatch
):
    selector = PotSelector(config, tmp_path)
    result = fixture_result(tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *a, **k: [])
    selector.apply(result)
    assert result.selection is ShotSelection.POTS_ONLY
    assert result.shots[0].included is False
    assert result.shots[0].manual_review_required is True
    result.shots[0].evidence["pots_only_manual_override"] = True
    result.shots[0].included = True
    selector.apply(result)
    assert result.shots[0].included is True


def test_pot_filter_keeps_confirmed_pots_only_and_preserves_clip_boundaries(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    selector = PotSelector(config, tmp_path)
    samples = readings((20, 12), end=9.5) + readings((27, 12), start=11)
    monkeypatch.setattr(selector, "scan", lambda *a, **k: samples)
    original = (result.shots[0].clip_start, result.shots[0].clip_end)
    selector.apply(result)
    assert result.shots[0].included
    assert (result.shots[0].clip_start, result.shots[0].clip_end) == original
    assert result.pot_selection_summary["pot"] == 1


def test_delayed_unresolved_previous_score_cannot_make_next_miss_a_pot(
    config, tmp_path, monkeypatch
):
    result = fixture_result(tmp_path)
    previous = shot(4, 8)
    current = shot(10, 15).model_copy(update={"shot_id": 2})
    result.shots = [previous, current]
    samples = readings((20, 12), start=8, end=9.5) + readings((27, 12), start=10.5)
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *a, **k: samples)
    selector.apply(result)
    assert previous.pot_status is PotStatus.UNKNOWN
    assert current.pot_status is PotStatus.UNKNOWN
    assert not current.included
    assert (
        current.evidence["pot_confirmation"]["reason"] == "delayed_previous_shot_score_is_ambiguous"
    )


def test_confirmed_previous_pot_does_not_block_fast_next_red(config, tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    previous = shot(4, 8)
    current = shot(10, 15).model_copy(update={"shot_id": 2})
    result.shots = [previous, current]
    samples = (
        readings((20, 12), start=2, end=3.5)
        + readings((27, 12), start=5, end=9.5)
        + readings((28, 12), start=10.5)
    )
    selector = PotSelector(config, tmp_path)
    monkeypatch.setattr(selector, "scan", lambda *a, **k: samples)
    selector.apply(result)
    assert [item.pot_status for item in result.shots] == [PotStatus.POT, PotStatus.POT]


def api_setup(config, tmp_path):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    result = fixture_result(tmp_path)
    store = JobStore(config)
    job = store.create(result.source_path)
    result.job_id = job
    store.save_analysis(result)
    return store, job, result


def test_changing_to_pots_creates_separate_job_and_keeps_original_files(
    config, tmp_path, monkeypatch
):
    store, old_id, result = api_setup(config, tmp_path)
    snapshot = {
        p.relative_to(store.job_dir(old_id)): p.read_bytes()
        for p in store.job_dir(old_id).rglob("*")
        if p.is_file()
    }
    calls = []
    monkeypatch.setattr("apps.api.main._run_analysis", lambda *args: calls.append(args))
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json={"job_id": old_id, "selection": "pots_only"})
    new_id = response.json()["job_id"]
    assert new_id != old_id
    assert store.get_meta(new_id)["selection"] == "pots_only"
    assert store.get_meta(new_id)["derived_from_job_id"] == old_id
    assert snapshot == {
        p.relative_to(store.job_dir(old_id)): p.read_bytes()
        for p in store.job_dir(old_id).rglob("*")
        if p.is_file()
    }
    assert calls[0][0] == new_id


def test_full_detector_result_is_reused_for_separate_pot_variant(config, tmp_path, monkeypatch):
    store, old_id, result = api_setup(config, tmp_path)
    old_bytes = (store.job_dir(old_id) / "analysis.json").read_bytes()

    def analyze(self, source, job_id, **kwargs):
        cached = JobStore(config).load_analysis(job_id)
        assert cached.job_id == job_id
        return cached

    monkeypatch.setattr("apps.api.main.Analyzer.analyze", analyze)
    monkeypatch.setattr(
        PotSelector,
        "scan",
        lambda *a, **k: readings((20, 12), end=9.5) + readings((27, 12), start=11),
    )
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/api/jobs", json={"job_id": old_id, "selection": "pots_only", "auto_export": False}
        )
        job = response.json()["job_id"]
        progress = client.get(f"/api/jobs/{job}/progress").json()
    assert progress["status"] == "ready_for_review"
    assert store.load_analysis(job).shots[0].pot_status is PotStatus.POT
    assert (store.job_dir(old_id) / "analysis.json").read_bytes() == old_bytes


def test_inconclusive_pot_job_opens_review_instead_of_creating_empty_video(
    config, tmp_path, monkeypatch
):
    store, old_id, result = api_setup(config, tmp_path)
    monkeypatch.setattr(
        "apps.api.main.Analyzer.analyze",
        lambda self, source, job_id, **kwargs: store.load_analysis(job_id),
    )
    monkeypatch.setattr(PotSelector, "scan", lambda *a, **k: [])
    monkeypatch.setattr(
        "apps.api.main.Exporter.export", lambda *a, **k: pytest.fail("unknown pots cannot export")
    )
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json={"job_id": old_id, "selection": "pots_only"})
        progress = client.get(f"/api/jobs/{response.json()['job_id']}/progress").json()
    assert progress["status"] == "ready_for_review"
    assert progress["download_url"] is None
    assert progress["pot_selection_summary"]["unknown"] == 1


def test_review_include_edit_records_explicit_pot_override(config, tmp_path):
    store, job, result = api_setup(config, tmp_path)
    result.selection = ShotSelection.POTS_ONLY
    result.shots[0].included = False
    store.save_analysis(result)
    store.update_shot(job, 1, ShotUpdate(included=True))
    assert store.load_analysis(job).shots[0].evidence["pots_only_manual_override"]


def test_manually_added_pot_keeps_review_selection_on_continue(config, tmp_path, monkeypatch):
    store, job, result = api_setup(config, tmp_path)
    result.selection = ShotSelection.POTS_ONLY
    store.save_analysis(result)
    added = store.add_shot(job, shot(23, 28))
    assert added.evidence["pots_only_manual_override"]
    result = store.load_analysis(job)
    selector = PotSelector(config, store.job_dir(job))
    monkeypatch.setattr(selector, "scan", lambda *a, **k: [])
    selector.apply(result)
    assert next(item for item in result.shots if item.cue_strike == 23).included


def test_score_scan_resumes_its_own_partial_checkpoint(config, tmp_path, synthetic_video):
    config._data["pot_selection"] = {"checkpoint_every_samples": 4}
    config._data["analysis"]["hwaccel_decode"] = False
    result = AnalysisResult(
        job_id="fixture",
        source_path=str(synthetic_video),
        original_duration=7,
        metadata=VideoMetadata(path=str(synthetic_video), duration=7, fps=15),
        shots=[shot(1, 4)],
    )

    def reading(frame, t):
        return ScoreReading(t, int(t >= 3), 0, 0, 0.99, "paired_boxes")

    whole = PotSelector(config, tmp_path / "whole")
    whole.reader.read = reading
    expected = whole.scan(result)
    interrupted = PotSelector(config, tmp_path / "resumed")

    def fail(frame, t):
        if t >= 3:
            raise RuntimeError("interrupted score scan")
        return reading(frame, t)

    interrupted.reader.read = fail
    with pytest.raises(RuntimeError, match="interrupted score scan"):
        interrupted.scan(result)
    cached = json.loads(
        (tmp_path / "resumed" / "pot_score_cache" / "window_0000.json").read_text("utf-8")
    )
    assert cached["complete"] is False and len(cached["readings"]) == 4
    restored_times = []
    resumed = PotSelector(config, tmp_path / "resumed")

    def observed(frame, t):
        restored_times.append(t)
        return reading(frame, t)

    resumed.reader.read = observed
    assert resumed.scan(result) == expected
    assert restored_times[0] > cached["readings"][-1]["t"]
    # A completed cache needs no decoding or OCR on Continue.
    resumed.reader.read = lambda *args: pytest.fail("completed score window must be reused")
    assert resumed.scan(result) == expected


def test_capabilities_advertise_selection_separately_from_detection_algorithm(config, tmp_path):
    api_setup(config, tmp_path)
    with TestClient(create_app(config)) as client:
        response = client.get("/api/shot-selections")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["selections"]] == ["all", "pots_only"]


def test_default_workflow_preserves_shots_and_never_runs_score_selection(
    config, tmp_path, monkeypatch
):
    store, job, result = api_setup(config, tmp_path)
    before = [item.model_dump() for item in result.shots]
    monkeypatch.setattr(
        "apps.api.main.Analyzer.analyze",
        lambda self, source, job_id, **kwargs: store.load_analysis(job_id),
    )
    monkeypatch.setattr(
        PotSelector,
        "scan",
        lambda *args, **kwargs: pytest.fail("default mode cannot run score selection"),
    )
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json={"job_id": job, "auto_export": False})
    assert response.status_code == 200
    assert response.json()["job_id"] == job
    assert store.get_meta(job)["selection"] == "all"
    assert [item.model_dump() for item in store.load_analysis(job).shots] == before
