"""Interrupted decoding/processes must preserve work and support Continue."""

import json
import subprocess
import sys
import os
import signal

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.pipeline.checkpoints import load_snapshot, save_snapshot
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.utils.video import IncompleteVideoError, sampled_frames


class Capture:
    def __init__(self, frames, fail_at=None):
        self.frames, self.fail_at, self.index = frames, fail_at, 0

    def grab(self):
        if self.index >= self.frames or self.index == self.fail_at:
            return False
        self.index += 1
        return True

    def read(self):
        ok = self.grab()
        return ok, np.zeros((2, 2, 3), np.uint8) if ok else None


@pytest.mark.parametrize("period", [1., 2.])
def test_decoder_failure_does_not_look_like_end_of_video(period):
    frames = sampled_frames(Capture(10, fail_at=4), proxy_fps=1,
                            to_source=lambda t: t, start_time=0, end_time=10,
                            sample_period=period)
    with pytest.raises(IncompleteVideoError, match="4.00s of 10.00s"):
        list(frames)


def test_normal_eof_and_intentional_early_close_are_supported():
    args = dict(proxy_fps=1, to_source=lambda t: t, start_time=0, end_time=10, sample_period=1)
    assert len(list(sampled_frames(Capture(10), **args))) == 10
    iterator = sampled_frames(Capture(100), **args)
    next(iterator)
    iterator.close()


@pytest.mark.parametrize("native", [False, True])
def test_scan_restores_detector_state_without_duplicate_or_missing_samples(
    config, tmp_path, synthetic_video, monkeypatch, native,
):
    config._data["device"] = "cpu"
    config._data["analysis"]["sample_fps"] = 2
    config._data["analysis"]["checkpoint_every_samples"] = 4
    config._data["analysis"]["hwaccel_decode"] = False
    mapper = TimeMapper(source_duration=7, proxy_duration=7, source_fps=15)
    kwargs = dict(collect_scene_observations=not native, checkpoint_stage="native" if native else "coarse_features",
                  sample_fps=15 if native else 2, start_time=2 if native else 0,
                  checkpoint_signature="matching-source-and-settings")
    whole = Analyzer(config, tmp_path / "whole")
    whole._native_source = synthetic_video if native else None
    expected, expected_scenes, _ = whole._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    interrupted = Analyzer(config, tmp_path / "resumed")
    interrupted._native_source = synthetic_video if native else None
    original_write = interrupted._write_checkpoint

    def stop_after_saved_checkpoint(data):
        original_write(data)
        if data["source_time"] >= 3:
            raise RuntimeError("simulated power loss")

    monkeypatch.setattr(interrupted, "_write_checkpoint", stop_after_saved_checkpoint)
    with pytest.raises(RuntimeError, match="power loss"):
        interrupted._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    saved = load_snapshot(interrupted.job_dir / ("native_partial.npz" if native else "coarse_partial.npz"))
    assert len(saved["features"]) >= 8
    resumed = Analyzer(config, interrupted.job_dir)
    resumed._native_source = synthetic_video if native else None
    original_capture = cv2.VideoCapture
    seeked = []

    class ObservedCapture:
        def __init__(self, path, **kwargs):
            self.cap = original_capture(str(path))

        def __getattr__(self, name):
            return getattr(self.cap, name)

        def set(self, prop, value):
            seeked.append((prop, value))
            return self.cap.set(prop, value)

    monkeypatch.setattr("snooker_ai.pipeline.analyzer.open_capture", ObservedCapture)
    actual, scenes, _ = resumed._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    assert seeked[-1] == (cv2.CAP_PROP_POS_FRAMES, saved["next_idx"])
    assert saved["next_idx"] > 0
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]
    assert scenes == expected_scenes


def test_atomic_checkpoint_write_failure_keeps_previous_valid_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "state.npz"
    save_snapshot(path, {"state": np.arange(10)})
    original_bytes = path.read_bytes()

    def interrupted_write(stream, **kwargs):
        stream.write(b"unfinished archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", interrupted_write)
    with pytest.raises(OSError, match="disk full"):
        save_snapshot(path, {"state": np.arange(20)})
    assert path.read_bytes() == original_bytes
    np.testing.assert_array_equal(load_snapshot(path)["state"], np.arange(10))


def test_finished_native_partial_does_not_decode_beyond_an_early_stop(
    config, tmp_path, synthetic_video, monkeypatch,
):
    config._data["device"] = "cpu"
    config._data["analysis"]["hwaccel_decode"] = False
    mapper = TimeMapper(7, 7, 15)
    kwargs = dict(sample_fps=15, start_time=2, end_time=6,
                  checkpoint_signature="same-window", stop_when=lambda features: len(features) >= 25)
    analyzer = Analyzer(config, tmp_path / "job")
    analyzer._native_source = synthetic_video
    expected, _, _ = analyzer._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    assert load_snapshot(analyzer.job_dir / "native_partial.npz")["complete"]
    resumed = Analyzer(config, analyzer.job_dir)
    resumed._native_source = synthetic_video

    def unexpected_decode(*args, **kwargs):
        pytest.fail("A completed early-stop snapshot must not decode more frames")

    monkeypatch.setattr("snooker_ai.pipeline.analyzer.sampled_frames", unexpected_decode)
    actual, _, _ = resumed._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]


@pytest.mark.parametrize("corruption", ["archive", "state"])
def test_corrupt_partial_checkpoint_restarts_safely_without_mixing_prefixes(
    config, tmp_path, synthetic_video, corruption,
):
    config._data["device"] = "cpu"
    config._data["analysis"]["sample_fps"] = 2
    config._data["analysis"]["hwaccel_decode"] = False
    mapper = TimeMapper(7, 7, 15)
    kwargs = dict(collect_scene_observations=True, checkpoint_signature="fixture")
    analyzer = Analyzer(config, tmp_path / "job")
    expected, expected_scenes, _ = analyzer._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    path = analyzer.job_dir / "coarse_partial.npz"
    if corruption == "archive":
        path.write_bytes(b"partial archive")
    else:
        snapshot = load_snapshot(path)
        snapshot["motion"] = (1,)  # invalid state after the prefix has been loaded
        save_snapshot(path, snapshot)
    fresh = Analyzer(config, analyzer.job_dir)
    actual, scenes, _ = fresh._extract_features(synthetic_video, None, mapper, 7, **kwargs)
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]
    assert scenes == expected_scenes


@pytest.mark.parametrize("problem", ["tail_missing", "gap", "empty"])
def test_partial_saved_analysis_cannot_resume_as_complete(config, tmp_path, monkeypatch, problem):
    from tests.test_analysis_resume import _saved_analysis
    source, analyzer, result = _saved_analysis(config, tmp_path)
    if problem == "tail_missing":
        result.features = result.features[:8]
    elif problem == "gap":
        result.features = result.features[:3] + result.features[12:]
    else:
        result.features = []
    analyzer._save_result(result)
    def reached_validation(*args, **kwargs):
        raise RuntimeError("incomplete result must be rebuilt")
    monkeypatch.setattr("snooker_ai.pipeline.analyzer.validate_video", reached_validation)
    with pytest.raises(RuntimeError, match="must be rebuilt"):
        analyzer.analyze(source, "job")


def setup_job(config, tmp_path):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = JobStore(config)
    job_id = store.create(source)
    store.update_progress(job_id, .45, "analyzing", "Halfway", shots_detected=12)
    (store.job_dir(job_id) / "checkpoint.json").write_text(
        json.dumps({"source_time": 42, "duration": 100}), encoding="utf-8")
    return store, job_id


def test_abandoned_job_shows_interrupted_and_continue_reuses_same_job(config, tmp_path, monkeypatch):
    store, job_id = setup_job(config, tmp_path)
    calls = []
    monkeypatch.setattr("apps.api.main._run_analysis", lambda *args: calls.append(args))
    with TestClient(create_app(config)) as client:
        progress = client.get(f"/api/jobs/{job_id}/progress").json()
        assert progress["status"] == "interrupted"
        assert progress["progress"] == .45
        assert progress["shots_detected"] == 12
        assert progress["can_continue"] is True
        assert progress["checkpoint"]["source_time"] == 42
        response = client.post(f"/api/jobs/{job_id}/continue")
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
    assert len(store.list_jobs()) == 1
    assert calls[0][0] == job_id
    assert calls[0][3:5] == (True, False)  # resume, no forced reanalysis


def test_active_worker_is_not_interrupted_or_started_twice(config, tmp_path):
    store, job_id = setup_job(config, tmp_path)
    lease = JobLease(store.job_dir(job_id))
    assert lease.acquire()
    try:
        with TestClient(create_app(config)) as client:
            progress = client.get(f"/api/jobs/{job_id}/progress").json()
            assert progress["status"] == "analyzing"
            assert progress["can_continue"] is False
            for route in ["continue", "restart"]:
                assert client.post(f"/api/jobs/{job_id}/{route}").status_code == 409
            assert client.post("/api/jobs", json={"job_id": job_id}).status_code == 409
        assert store.get_meta(job_id)["progress"] == .45
    finally:
        lease.release()


def test_decoder_failure_keeps_progress_and_does_not_export_partial_result(config, tmp_path, monkeypatch):
    from apps.api.main import _run_analysis
    store, job_id = setup_job(config, tmp_path)
    def fail(*args, **kwargs):
        raise IncompleteVideoError("Video decoding stopped at 42s of 100s")
    monkeypatch.setattr("apps.api.main.Analyzer.analyze", fail)
    monkeypatch.setattr("apps.api.main.Exporter.export", lambda *args: pytest.fail("partial analysis cannot export"))
    _run_analysis(job_id, store.get_meta(job_id)["source_path"], "strict", True, config=config)
    meta = store.get_meta(job_id)
    assert meta["status"] == "failed"
    assert meta["progress"] == .45
    assert "42s of 100s" in meta["error"]
    assert (store.job_dir(job_id) / "checkpoint.json").exists()


def test_process_termination_releases_job_lock(tmp_path):
    script = (
        "import sys,time,os; from pathlib import Path; from snooker_ai.jobs.lease import JobLease; "
        "lease=JobLease(Path(sys.argv[1])); assert lease.acquire(); print(os.getpid(),flush=True); time.sleep(30)"
    )
    worker = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        actual_pid = int(worker.stdout.readline().strip())
        assert JobLease.is_running(tmp_path)
        # Windows venv python.exe can launch a worker with a separate PID.
        os.kill(actual_pid, signal.SIGTERM)
        worker.wait(timeout=10)
        assert not JobLease.is_running(tmp_path)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
        worker.stdout.close()
