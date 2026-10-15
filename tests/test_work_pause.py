"""Pause is durable control flow; resume preserves detections and export intent."""

import concurrent.futures
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.queue import MatchQueue
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.shorts import ShortsManager, ShortsSettings
from snooker_ai.types import AnalysisResult, EditMode, ShotRecord, ShotUpdate, VideoMetadata
from snooker_ai.utils.ffmpeg import run_command
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.utils.work_control import WorkControl, WorkPaused, check_pause, current_control


def until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail("Work did not reach the expected state")


@pytest.fixture
def work_case(config, tmp_path):
    config._data["device"] = "cpu"
    config._data["jobs"] = {"max_workers": 1}
    config._data["paths"].update(
        jobs_dir=str(tmp_path / "jobs"), uploads_dir=str(tmp_path / "uploads")
    )
    source = tmp_path / "Judd Trump vs Mark Allen.mp4"
    source.write_bytes(b"test source")
    store = JobStore(config)
    job = store.create(source)
    result = AnalysisResult(
        job_id=job,
        source_path=str(source),
        original_duration=10,
        metadata=VideoMetadata(path=str(source), duration=10, width=640, height=360, fps=25),
        shots=[
            ShotRecord(shot_id=1, clip_start=0, clip_end=3, cue_strike=1),
            ShotRecord(shot_id=2, clip_start=4, clip_end=7, cue_strike=5),
        ],
    )
    store.save_analysis(result)
    store.update_progress(job, 1, "ready_for_review", shots_detected=2)
    return config, store, result


def test_pause_terminates_only_owned_command_and_propagates_control(tmp_path, monkeypatch):
    import snooker_ai.utils.ffmpeg as ffmpeg

    original = ffmpeg.subprocess.Popen
    started = threading.Event()
    processes = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", launch)
    control = WorkControl(tmp_path)

    def run():
        token = current_control.set(control)
        try:
            with pytest.raises(WorkPaused):
                run_command([sys.executable, "-c", "import time; time.sleep(60)"])
        finally:
            current_control.reset(token)

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        future = pool.submit(run)
        assert started.wait(5)
        control.request()
        future.result(timeout=5)
    assert processes[0].poll() is not None
    assert control.requested()
    control.clear()
    assert not control.requested()


def test_paused_waiting_match_survives_restart_and_keeps_original_request(work_case):
    config, store, result = work_case
    job = result.job_id
    store.update_progress(job, 0, "pending")
    old = MatchQueue(config, lambda *args: pytest.fail("not started"))
    assert old.submit(job, result.source_path, EditMode.STRICT, False, True, False)
    old.pause(job)
    assert store.get_meta(job)["status"] == "paused"
    assert (store.job_dir(job) / "queued.json").is_file()
    assert not JobLease.is_running(store.job_dir(job), ".queue.lock")
    old.close()
    calls = []
    new = MatchQueue(config, lambda *args: calls.append(args))
    new.start()
    assert new.wait_idle() and calls == []
    assert new.submit(job, result.source_path, EditMode.STRICT, False, True, False)
    assert new.wait_idle()
    assert calls[0][3:5] == (False, True) and calls[0][-1] is False
    assert not WorkControl(store.job_dir(job)).requested()
    new.close()


def test_active_match_feed_pause_and_resume(work_case, monkeypatch):
    config, store, result = work_case
    meta = store.get_meta(result.job_id)
    meta["auto_export"] = False
    store._write_meta(result.job_id, meta)
    entered = threading.Event()
    calls = []

    class FakeAnalyzer:
        def analyze(self, source, job, *, progress, **kwargs):
            calls.append(kwargs)
            progress(0.4, "analyzing", "Finding shots at frame 100")
            if len(calls) == 1:
                entered.set()
                while True:
                    progress(0.4, "analyzing", "Finding shots at frame 100")
                    time.sleep(0.025)
            return result

    monkeypatch.setattr("apps.api.main.create_analyzer", lambda *args: FakeAnalyzer())
    with TestClient(create_app(config)) as client:
        queue = client.app.state.match_queue
        assert queue.submit(
            result.job_id, result.source_path, EditMode.STRICT, True, auto_export=False
        )
        assert entered.wait(5)
        item = client.get("/api/work").json()["items"][0]
        assert item["kind"] == "match" and item["can_pause"]
        assert (
            client.post(
                f"/api/jobs/{result.job_id}/pause", headers={"Origin": "https://wrong.example"}
            ).status_code
            == 403
        )
        assert client.post(item["pause_url"]).status_code == 200
        until(
            lambda: (
                store.get_meta(result.job_id)["status"] == "paused"
                and not JobLease.is_running(store.job_dir(result.job_id), ".queue.lock")
            )
        )
        paused = client.get("/api/work").json()["items"][0]
        assert paused["status"] == "paused" and paused["can_resume"] and not paused["can_pause"]
        assert store.get_meta(result.job_id)["progress"] == 0.4
        assert client.post(paused["resume_url"]).status_code == 200
        assert queue.wait_idle()
        assert store.get_meta(result.job_id)["status"] == "ready_for_review"
        assert calls[1]["resume"] is True
        assert client.get("/api/work").json()["items"] == []


def test_pause_saves_exact_coarse_state_and_resume_matches_uninterrupted_scan(
    config, tmp_path, synthetic_video, monkeypatch
):
    config._data["device"] = "cpu"
    config._data["analysis"].update(sample_fps=2, checkpoint_every_samples=4, hwaccel_decode=False)
    mapper = TimeMapper(7, 7, 15)
    kwargs = dict(
        collect_scene_observations=True,
        checkpoint_signature="pause-fixture",
        checkpoint_stage="coarse_features",
    )
    expected, scenes, _ = Analyzer(config, tmp_path / "whole")._extract_features(
        synthetic_video, None, mapper, 7, **kwargs
    )
    analyzer = Analyzer(config, tmp_path / "resumed")
    control = WorkControl(analyzer.job_dir)
    original_write = analyzer._write_checkpoint

    def request_after_checkpoint(data):
        original_write(data)
        if data["source_time"] >= 1.5:
            control.request()

    monkeypatch.setattr(analyzer, "_write_checkpoint", request_after_checkpoint)
    original_check = control.check

    def check_every_sample():
        control.last_check = 0
        original_check()

    monkeypatch.setattr(control, "check", check_every_sample)
    token = current_control.set(control)
    try:
        with pytest.raises(WorkPaused):
            analyzer._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    finally:
        current_control.reset(token)
    control.clear()
    actual, actual_scenes, _ = Analyzer(config, analyzer.job_dir)._extract_features(
        synthetic_video, None, mapper, 7, **kwargs
    )
    assert [item.model_dump() for item in actual] == [item.model_dump() for item in expected]
    assert actual_scenes == scenes


def test_shorts_pause_retains_clip_and_resumes_original_snapshot(work_case, monkeypatch):
    config, store, result = work_case
    entered = threading.Event()
    calls = []

    def render(result, shot, settings, output, config):
        calls.append(shot.shot_id)
        if shot.shot_id == 2 and calls.count(2) == 1:
            entered.set()
            while True:
                check_pause()
                time.sleep(0.025)
        output.write_bytes(f"clip {shot.shot_id}".encode())

    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", render)
    with TestClient(create_app(config)) as client:
        base = f"/api/jobs/{result.job_id}/shorts"
        batch = client.post(base, json={"player_names": "Judd Trump"}).json()
        assert entered.wait(5)
        item = client.get("/api/work").json()["items"][0]
        assert item["kind"] == "shorts" and item["completed"] == 1 and item["total"] == 2
        assert client.post(item["pause_url"]).status_code == 200
        manager = client.app.state.shorts_manager
        until(
            lambda: (
                manager.read(result.job_id, batch["batch_id"])["status"] == "paused"
                and not JobLease.is_running(manager.root(result.job_id), ".shorts.lock")
            )
        )
        store.update_shot(result.job_id, 2, ShotUpdate(included=False))
        paused = client.get("/api/work").json()["items"][0]
        assert paused["can_resume"] and paused["status"] == "paused"
        setup = client.get(base + "?batch_id=" + batch["batch_id"]).json()
        assert [s["shot_id"] for s in setup["shots"]] == [1, 2]
        assert client.post(paused["resume_url"]).status_code == 202
        until(lambda: manager.read(result.job_id, batch["batch_id"])["status"] == "completed")
        assert calls == [1, 2, 2]
        assert client.get("/api/work").json()["items"] == []


def test_queued_shorts_pause_releases_lease_without_waiting_for_another_batch(
    work_case, monkeypatch
):
    config, store, result = work_case
    entered, release = threading.Event(), threading.Event()

    def render(result, shot, settings, output, config):
        entered.set()
        assert release.wait(6)
        output.write_bytes(b"clip")

    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", render)
    manager = ShortsManager(config, store)
    try:
        manager.submit(result.job_id, ShortsSettings(player_names="One"))
        assert entered.wait(5)
        other = store.create(result.source_path)
        copy = result.model_copy(update={"job_id": other})
        store.save_analysis(copy)
        store.update_progress(other, 1, "ready_for_review")
        queued = manager.submit(other, ShortsSettings(player_names="Two"))
        assert manager.pause(other, queued["batch_id"])["status"] == "paused"
        assert not JobLease.is_running(manager.root(other), ".shorts.lock")
    finally:
        release.set()
        manager.close()


def test_manual_export_resume_preserves_exact_export_request(work_case, monkeypatch):
    config, store, result = work_case
    entered = threading.Event()
    requests = []

    class FakeExporter:
        def __init__(self, *args):
            pass

        def export(self, result, directory, request):
            requests.append(request.model_dump())
            if len(requests) == 1:
                entered.set()
                while True:
                    check_pause()
                    time.sleep(0.025)
            return SimpleNamespace(
                joined_path=None,
                clip_paths=[],
                csv_path=None,
                edl_path=None,
                metadata_path=None,
                training_labels_path=None,
            )

    monkeypatch.setattr("apps.api.main.Exporter", FakeExporter)
    with TestClient(create_app(config)) as client, concurrent.futures.ThreadPoolExecutor(1) as pool:
        base = f"/api/jobs/{result.job_id}"
        body = {
            "export_joined": False,
            "export_clips": True,
            "min_confidence": 0.7,
            "output_name": "custom.mp4",
        }
        future = pool.submit(client.post, base + "/export", json=body)
        assert entered.wait(5)
        assert client.post(base + "/pause").status_code == 200
        assert future.result(5).json()["status"] == "paused"
        assert store.get_meta(result.job_id)["resume_export"]["min_confidence"] == 0.7
        assert client.post(base + "/continue").status_code == 200
        assert client.app.state.match_queue.wait_idle()
        assert requests[0] == requests[1]
        assert store.get_meta(result.job_id)["status"] == "completed"
        assert "resume_export" not in store.get_meta(result.job_id)


def test_work_poll_cannot_overwrite_a_just_finished_match(work_case, monkeypatch):
    config, store, result = work_case
    store.update_progress(result.job_id, 1, "completed", "Finished")
    stale = {**store.get_meta(result.job_id), "status": "analyzing", "progress": 0.5}
    monkeypatch.setattr(JobStore, "list_jobs", lambda self: [stale])
    with TestClient(create_app(config)) as client:
        assert client.get("/api/work").json()["items"] == []
        assert store.get_meta(result.job_id)["status"] == "completed"


def test_queue_retries_brief_windows_request_file_sharing_error(work_case, monkeypatch):
    config, store, result = work_case
    path = store.job_dir(result.job_id) / "queued.json"
    path_class = type(path)
    original = path_class.unlink
    attempts = []

    def sharing_then_release(target, *args, **kwargs):
        if target == path:
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError("Another server is reading the waiting request")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(path_class, "unlink", sharing_then_release)
    queue = MatchQueue(config, lambda job, *args: store.update_progress(job, 1, "completed"))
    try:
        assert queue.submit(result.job_id, result.source_path, EditMode.STRICT, True)
        queue.start()
        assert queue.wait_idle()
        assert len(attempts) == 2
        assert store.get_meta(result.job_id)["status"] == "completed"
    finally:
        queue.close()
