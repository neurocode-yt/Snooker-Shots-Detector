"""Storage failures must not consume the remaining drive or publish broken videos."""

import asyncio
import errno
import io
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.uploads import save_upload


@pytest.fixture
def upload_config(config, tmp_path):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    config._data["api"]["upload_reserve_mb"] = 0
    return config


def test_complete_upload_is_saved_exactly_and_has_no_partial(upload_config):
    content = b"source video\0" * 100
    with TestClient(create_app(upload_config)) as client:
        response = client.post("/api/upload", files={"file": ("match one.mp4", content)})
    assert response.status_code == 200
    result = response.json()
    saved = Path(result["path"])
    assert saved.read_bytes() == content
    assert result["filename"] == "match one.mp4"
    assert result["size"] == len(content)
    assert "match_one.mp4" in saved.name
    assert list(saved.parent.iterdir()) == [saved]


def test_upload_rejects_insufficient_space_before_writing_and_preserves_existing(
    upload_config, monkeypatch
):
    app = create_app(upload_config)
    directory = Path(upload_config.get("paths.uploads_dir"))
    previous = directory / "previous.mp4"
    previous.write_bytes(b"keep this")
    monkeypatch.setattr("apps.api.uploads.shutil.disk_usage", lambda _: SimpleNamespace(free=4))
    with TestClient(app) as client:
        response = client.post("/api/upload", files={"file": ("new.mp4", b"12345")})
    assert response.status_code == 507
    assert "Free space on that drive" in response.json()["detail"]
    assert str(directory) in response.json()["detail"]
    assert list(directory.iterdir()) == [previous]
    assert previous.read_bytes() == b"keep this"


def test_upload_reserves_configured_working_space(upload_config, monkeypatch):
    upload_config._data["api"]["upload_reserve_mb"] = 1
    monkeypatch.setattr(
        "apps.api.uploads.shutil.disk_usage", lambda _: SimpleNamespace(free=1024**2 + 3)
    )
    with TestClient(create_app(upload_config)) as client:
        response = client.post("/api/upload", files={"file": ("new.mp4", b"1234")})
    assert response.status_code == 507
    assert not list(Path(upload_config.get("paths.uploads_dir")).iterdir())


def test_space_consumed_by_another_task_removes_the_partial_and_allows_retry(
    upload_config, monkeypatch
):
    # Preflight and the first chunk fit; another writer fills the drive next.
    monkeypatch.setattr("apps.api.uploads._CHUNK_SIZE", 4)
    free = iter([100, 100, 0])
    monkeypatch.setattr(
        "apps.api.uploads.shutil.disk_usage", lambda _: SimpleNamespace(free=next(free))
    )
    with TestClient(create_app(upload_config)) as client:
        response = client.post("/api/upload", files={"file": ("new.mp4", b"12345678")})
        assert response.status_code == 507
        assert not list(Path(upload_config.get("paths.uploads_dir")).iterdir())
        monkeypatch.setattr(
            "apps.api.uploads.shutil.disk_usage", lambda _: SimpleNamespace(free=100)
        )
        retried = client.post("/api/upload", files={"file": ("new.mp4", b"12345678")})
    assert retried.status_code == 200
    assert Path(retried.json()["path"]).read_bytes() == b"12345678"


@pytest.mark.parametrize("failure_at", ["write", "close", "publish"])
@pytest.mark.parametrize("windows_error", [False, True])
def test_full_disk_during_write_flush_or_publication_is_json_and_cleans_up(
    upload_config, monkeypatch, failure_at, windows_error
):
    original_open, original_replace = Path.open, Path.replace
    error = OSError(errno.ENOSPC, "No space left on device")
    if windows_error:
        error = OSError(errno.EINVAL, "There is not enough space on the disk")
        error.winerror = 112

    class FailingOutput:
        def __init__(self, output):
            self.output = output

        def __enter__(self):
            return self

        def write(self, chunk):
            self.output.write(chunk[:2])
            if failure_at == "write":
                raise error

        def __exit__(self, *_):
            self.output.close()
            if failure_at == "close":
                raise error

    def failing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailingOutput(stream) if path.name.endswith(".part") else stream

    def failing_publish(path, target):
        if path.name.endswith(".part") and failure_at == "publish":
            raise error
        return original_replace(path, target)

    monkeypatch.setattr(Path, "open", failing_open)
    monkeypatch.setattr(Path, "replace", failing_publish)
    with TestClient(create_app(upload_config)) as client:
        response = client.post("/api/upload", files={"file": ("new.mp4", b"12345678")})
    assert response.status_code == 507
    assert "ran out of space" in response.json()["detail"]
    assert not list(Path(upload_config.get("paths.uploads_dir")).iterdir())


def test_known_oversize_upload_rejected_without_using_disk(upload_config, monkeypatch):
    upload_config._data["api"]["max_upload_gb"] = 4 / 1024**3
    monkeypatch.setattr(
        "apps.api.uploads.shutil.disk_usage", lambda _: pytest.fail("must reject first")
    )
    with TestClient(create_app(upload_config)) as client:
        response = client.post("/api/upload", files={"file": ("big.mp4", b"12345")})
    assert response.status_code == 413
    assert not list(Path(upload_config.get("paths.uploads_dir")).iterdir())


def test_unknown_size_still_enforces_limit_and_removes_partial(tmp_path):
    file = UploadFile(io.BytesIO(b"12345"), filename="big.mp4")
    with pytest.raises(HTTPException) as exc:
        save_upload(file, tmp_path, max_gb=4 / 1024**3, reserve_mb=0)
    assert exc.value.status_code == 413
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [OSError(errno.EIO, "Read failed"), KeyboardInterrupt()])
def test_interrupted_copy_removes_only_its_partial(tmp_path, monkeypatch, failure):
    previous = tmp_path / "previous.mp4"
    previous.write_bytes(b"keep")

    class BrokenInput(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise failure
            return super().read(size)

    monkeypatch.setattr("apps.api.uploads._CHUNK_SIZE", 2)
    file = UploadFile(BrokenInput(b"1234"), filename="broken.mp4")
    with pytest.raises(type(failure)):
        save_upload(file, tmp_path, max_gb=1, reserve_mb=0)
    assert list(tmp_path.iterdir()) == [previous]
    assert previous.read_bytes() == b"keep"


def test_incomplete_copy_never_publishes_a_video(tmp_path):
    file = UploadFile(io.BytesIO(b"1234"), size=8, filename="broken.mp4")
    with pytest.raises(HTTPException) as exc:
        save_upload(file, tmp_path, max_gb=1, reserve_mb=0)
    assert exc.value.status_code == 400
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_slow_upload_copy_does_not_block_health_requests(upload_config, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = save_upload

    def slow_copy(*args):
        started.set()
        if not release.wait(3):
            raise RuntimeError("copy blocked the event loop")
        return original(*args)

    monkeypatch.setattr("apps.api.main.save_upload", slow_copy)
    transport = httpx.ASGITransport(app=create_app(upload_config))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(
            client.post("/api/upload", files={"file": ("match.mp4", b"video")})
        )
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert (await asyncio.wait_for(client.get("/health"), 1)).status_code == 200
            assert not task.done()
        finally:
            release.set()
            response = await task
        assert response.status_code == 200
