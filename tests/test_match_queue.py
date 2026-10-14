"""Real worker overlap, bounded capacity, recovery and request isolation."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.queue import MatchQueue
from snooker_ai.jobs.store import JobStore
from snooker_ai.types import EditMode


def prepare(config, tmp_path, workers=2):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    config._data["jobs"]["max_workers"] = workers
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return JobStore(config), source


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), "worker did not reach the expected state"


def test_two_matches_overlap_third_waits_and_failure_releases_slot(config, tmp_path):
    store, source = prepare(config, tmp_path)
    jobs = [store.create(source) for _ in range(4)]
    release = threading.Event()
    seen = []
    lock = threading.Lock()

    def run(job_id, *_args):
        with lock:
            seen.append(job_id)
        if job_id in jobs[:2]:
            assert release.wait(3)
        if job_id == jobs[0]:
            raise RuntimeError("bad video")
        store.update_progress(job_id, 1, "completed")

    queue = MatchQueue(config, run)
    queue.start()
    try:
        for job in jobs:
            assert queue.submit(job, str(source), EditMode.STRICT, True)
        until(lambda: queue.snapshot()["processing"] == 2)
        assert set(seen) == set(jobs[:2])
        assert queue.snapshot() == {"max_workers": 2, "processing": 2, "waiting": 2}
        assert queue.job_state(jobs[2]) == {"queued": True, "queue_position": 1}
        assert queue.job_state(jobs[3])["queue_position"] == 2
        assert not queue.submit(jobs[0], str(source), EditMode.STRICT, True)
        release.set()
        assert queue.wait_idle()
        assert set(seen) == set(jobs)
        assert store.get_meta(jobs[0])["status"] == "failed"
        assert all(store.get_meta(job)["status"] == "completed" for job in jobs[1:])
        assert not any(store.root.glob("*/queued.json"))
    finally:
        release.set()
        queue.close()


def test_shared_slots_and_queued_job_lease_bound_multiple_servers(config, tmp_path):
    store, source = prepare(config, tmp_path, 1)
    first, second = store.create(source), store.create(source)
    release = threading.Event()
    started = []

    def run(job_id, *_args):
        started.append(job_id)
        assert release.wait(4)
        store.update_progress(job_id, 1, "completed")

    one, two = MatchQueue(config, run), MatchQueue(config, run)
    one.start()
    two.start()
    try:
        assert one.submit(first, str(source), EditMode.STRICT, True)
        until(lambda: started == [first])
        assert not two.submit(first, str(source), EditMode.STRICT, True)
        assert two.submit(second, str(source), EditMode.STRICT, True)
        time.sleep(.35)  # allow another server to attempt an occupied OS slot
        assert started == [first]
        assert two.job_state(second)["queued"]
        release.set()
        assert one.wait_idle() and two.wait_idle()
        assert started == [first, second]
    finally:
        release.set()
        one.close()
        two.close()


def test_waiting_requests_recover_with_original_settings(config, tmp_path):
    store, source = prepare(config, tmp_path, 1)
    jobs = [store.create(source) for _ in range(2)]
    # Simulate server shutdown with waiting requests; no worker has started.
    old = MatchQueue(config, lambda *args: pytest.fail("no worker started"))
    assert old.submit(jobs[0], str(source), EditMode.STRICT, False, True, False)
    assert old.submit(jobs[1], str(source), EditMode.STRICT, True)
    old.close()
    calls = []

    def recovered(job_id, *args):
        calls.append((job_id, *args))
        store.update_progress(job_id, 1, "completed")

    new = MatchQueue(config, recovered)
    new.start()
    try:
        assert new.wait_idle()
        assert [call[0] for call in calls] == jobs
        assert calls[0][2:5] == (EditMode.STRICT, False, True)
        assert calls[0][-1] is False
        assert not any(store.root.glob("*/queued.json"))
    finally:
        new.close()


def test_finished_request_is_not_restarted_from_stale_queue_file(config, tmp_path):
    store, source = prepare(config, tmp_path)
    job = store.create(source)
    (store.job_dir(job) / "queued.json").write_text(json.dumps({"job_id": job}), "utf-8")
    store.update_progress(job, 1, "completed")
    queue = MatchQueue(config, lambda *args: pytest.fail("completed video must stay finished"))
    queue.start()
    assert queue.wait_idle()
    queue.close()
    assert store.get_meta(job)["status"] == "completed"


def test_api_accepts_batch_immediately_and_blocks_edits_of_waiting_matches(config, tmp_path, monkeypatch):
    store, source = prepare(config, tmp_path)
    release = threading.Event()

    def run(job_id, *_args):
        assert release.wait(4)
        store.update_progress(job_id, 1, "ready_for_review")

    monkeypatch.setattr("apps.api.main._run_analysis", run)
    app = create_app(config)
    with TestClient(app) as client:
        try:
            jobs = [client.post("/api/jobs", json={"source_path": str(source), "auto_export": False}).json()["job_id"] for _ in range(3)]
            until(lambda: app.state.match_queue.snapshot()["processing"] == 2)
            assert client.get("/api/queue").json() == {"max_workers": 2, "processing": 2, "waiting": 1}
            waiting = client.get(f"/api/jobs/{jobs[2]}/progress").json()
            assert waiting["status"] == "pending"
            assert waiting["queued"] and waiting["running"]
            assert waiting["queue_position"] == 1 and not waiting["can_continue"]
            for suffix, body in [("restart", {}), ("continue", {}), ("export", {}), ("shots", {"cue_strike": 3, "clip_start": 2, "clip_end": 5})]:
                assert client.post(f"/api/jobs/{jobs[2]}/{suffix}", json=body).status_code == 409
        finally:
            release.set()
        assert app.state.match_queue.wait_idle()
        assert all(client.get(f"/api/jobs/{job}/progress").json()["status"] == "ready_for_review" for job in jobs)


def test_submission_token_reuses_same_job_after_lost_response(config, tmp_path, monkeypatch):
    store, source = prepare(config, tmp_path)
    calls = []

    def run(job_id, *_args):
        calls.append(job_id)
        store.update_progress(job_id, 1, "ready_for_review")

    monkeypatch.setattr("apps.api.main._run_analysis", run)
    app = create_app(config)
    body = {"job_id": "batch-once", "submission_token": "once", "source_path": str(source), "auto_export": False}
    with TestClient(app) as client:
        assert client.post("/api/jobs", json=body).status_code == 200
        assert app.state.match_queue.wait_idle()
        assert client.post("/api/jobs", json=body).json()["job_id"] == "batch-once"
        assert client.post("/api/jobs", json={**body, "auto_export": True}).status_code == 409
        assert app.state.match_queue.wait_idle()
    assert calls == ["batch-once"]
    assert len(store.list_jobs()) == 1
    assert store.get_meta("batch-once")["created_at"]


def test_invalid_waiting_request_offers_continue_instead_of_staying_queued(config, tmp_path):
    store, source = prepare(config, tmp_path)
    job = store.create(source)
    (store.job_dir(job) / "queued.json").write_text("[]", "utf-8")
    with TestClient(create_app(config)) as client:
        meta = client.get(f"/api/jobs/{job}/progress").json()
        assert meta["status"] == "interrupted"
        assert meta["can_continue"] and not meta["queued"]


def test_submission_retry_repairs_request_not_yet_persisted(config, tmp_path, monkeypatch):
    store, source = prepare(config, tmp_path)
    job = store.create(source)
    meta = store.get_meta(job)
    meta.update(submission_token="saved-before-queue", auto_export=False)
    store._write_meta(job, meta)
    calls = []
    monkeypatch.setattr("apps.api.main._run_analysis", lambda *args: calls.append(args))
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"job_id": job, "submission_token": "saved-before-queue", "auto_export": False})
        assert response.status_code == 200
        assert app.state.match_queue.wait_idle()
    assert len(calls) == 1 and calls[0][0] == job


@pytest.mark.parametrize("workers", [0, 5, -1])
def test_invalid_concurrency_is_rejected(config, tmp_path, workers):
    prepare(config, tmp_path, workers)
    with pytest.raises(ValueError, match="max_workers"):
        MatchQueue(config, lambda *args: None)
