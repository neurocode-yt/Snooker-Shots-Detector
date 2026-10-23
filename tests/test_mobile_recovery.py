"""Phone recovery preserves committed detector state, edits and user pauses."""
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.pipeline.checkpoints import load_snapshot, save_snapshot
from snooker_ai.utils.persistence import RecoverySaveError, atomic_text, set_publish_guard
from snooker_ai.utils.work_control import WorkControl
from tests.test_work_pause import work_case  # noqa: F401 -- shared project fixture


@pytest.fixture
def mobile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "apps/android/app/src/main/python/mobile"))
    module = importlib.import_module("mobile_backup")
    yield module
    set_publish_guard(None)


@pytest.fixture
def recovery(mobile, tmp_path):
    root = tmp_path / "phone"
    (root / "jobs" / "match").mkdir(parents=True)
    store = mobile.RecoveryStore(root, tmp_path / "recovery")
    set_publish_guard(store.publish)
    yield store, root / "jobs/match"
    store.db.close()


def test_corrupt_checkpoint_and_edits_recover_latest_valid_generation(recovery):
    store, directory = recovery
    checkpoint = directory / "native_partial.npz"
    edits = directory / "analysis.json"
    save_snapshot(checkpoint, {"next_idx": 100, "track": np.arange(8)})
    atomic_text(edits, '{"shots": [1, 2]}')
    save_snapshot(checkpoint, {"next_idx": 200, "track": np.arange(9)})
    checkpoint.write_bytes(b"interrupted archive")
    edits.unlink()
    assert len(store.recover()) == 2
    assert load_snapshot(checkpoint)["next_idx"] == 200
    np.testing.assert_array_equal(load_snapshot(checkpoint)["track"], np.arange(9))
    assert json.loads(edits.read_text())["shots"] == [1, 2]


def test_bad_latest_recovery_object_falls_back_and_history_is_bounded(recovery):
    store, directory = recovery
    path = directory / "checkpoint.json"
    for value in range(6):
        atomic_text(path, json.dumps({"value": value}))
    records = dict(store.rows())["jobs/match/checkpoint.json"]
    assert len(records) == 3
    assert len(list(store.objects.glob("*"))) == 3
    (store.objects / records[0]["sha256"]).write_bytes(b"bad latest backup")
    path.write_bytes(b"bad primary")
    store.recover()
    assert json.loads(path.read_text())["value"] == 4


def test_failed_protection_cannot_publish_or_destroy_previous_checkpoint(recovery, monkeypatch):
    store, directory = recovery
    path = directory / "checkpoint.json"
    atomic_text(path, '{"frame": 20}')

    def full(*args):
        raise OSError("storage full")

    monkeypatch.setattr(store, "capture", full)
    with pytest.raises(RecoverySaveError, match="storage full"):
        atomic_text(path, '{"frame": 40}')
    assert json.loads(path.read_text())["frame"] == 20
    assert "storage full" in store.status()["error"]


def test_recovery_does_not_resurrect_deleted_jobs_or_queue_requests(recovery):
    store, directory = recovery
    atomic_text(directory / "job.json", '{}')
    atomic_text(directory / "queued.json", '{}')
    assert all(not path.endswith("queued.json") for path, _ in store.rows())
    (directory / "job.json").unlink()
    (directory / "queued.json").unlink()
    directory.rmdir()
    assert store.recover() == []
    assert not directory.exists()


@pytest.mark.parametrize("status,system_pause,pause_requested,expected", [
    ("analyzing", False, False, True), ("paused", False, True, False),
    ("paused", True, True, True), ("analyzing", False, True, False),
])
def test_automatic_recovery_uses_real_continue_route_and_respects_manual_pause(
    mobile, work_case, monkeypatch, status, system_pause, pause_requested, expected,  # noqa: F811
):
    runtime = importlib.import_module("mobile_runtime")
    config, store, result = work_case
    store.update_progress(result.job_id, .4, status)
    meta = store.get_meta(result.job_id)
    meta["auto_export"] = False
    store._write_meta(result.job_id, meta)
    marker = store.job_dir(result.job_id) / "mobile_system_resume"
    if system_pause:
        marker.write_text("1")
    if pause_requested:
        WorkControl(store.job_dir(result.job_id)).request()
    monkeypatch.setattr(runtime, "_config", config)
    monkeypatch.setattr(runtime, "_url", "http://testserver")
    monkeypatch.setattr(runtime, "_token", "test-only")
    app = create_app(config)
    calls = []
    # Stop at the queue boundary; this tests the real stale-lease detection,
    # Continue validation and recovery request without running a fake video.
    monkeypatch.setattr(app.state.match_queue, "submit", lambda *args, **kwargs: calls.append((args, kwargs)) or True)
    with TestClient(app) as client:
        def request(req, **kwargs):
            response = client.request(req.method, req.full_url)
            response.raise_for_status()
            return io.BytesIO(response.content)
        monkeypatch.setattr("urllib.request.urlopen", request)
        resumed = json.loads(runtime.recover_interrupted())
    assert bool(resumed) == expected
    assert bool(calls) == expected
    if expected:
        # Continue keeps resume=True and the original no-auto-export choice.
        assert calls[0][0][3] is True
        assert calls[0][0][5] is False
        assert not marker.exists()
