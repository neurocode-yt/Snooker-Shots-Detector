"""CapCut draft interoperability, source trims, and local handoff API."""

import asyncio
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.main import _locks, create_app
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.capcut import (
    CapCutInstallation, CapCutUnavailable, create_capcut_project, find_capcut,
)
from snooker_ai.types import AnalysisResult, ShotRecord, VideoMetadata


@pytest.fixture
def capcut_case(config, tmp_path):
    source = tmp_path / "20261011-120000_123456ab_Match_cleaned_123456ab.mp4"
    source.write_bytes(b"original media")
    executable = tmp_path / "Apps" / "CapCut.exe"
    executable.parent.mkdir()
    executable.touch()
    installation = CapCutInstallation(executable, tmp_path / "drafts", "9.5.0.4050")
    result = AnalysisResult(
        job_id="capcut-test", source_path=str(source),
        metadata=VideoMetadata(
            path=str(source), duration=20.0, fps=25.0, width=1280, height=720, has_audio=True,
        ),
        shots=[
            ShotRecord(shot_id=3, cue_strike=8.6, clip_start=8.13, clip_end=9.87),
            ShotRecord(shot_id=1, cue_strike=2.0, clip_start=1.03, clip_end=4.59),
            ShotRecord(shot_id=2, cue_strike=5.0, clip_start=4.6, clip_end=6.0, included=False),
            ShotRecord(shot_id=4, cue_strike=7.0, clip_start=6.5, clip_end=7.5, possible_replay=True),
        ],
    )
    return result, installation


@pytest.mark.parametrize("fps,has_audio", [(25.0, True), (30000 / 1001, True), (50.0, False)])
def test_capcut_has_separate_ordered_clips_and_full_source_handles(
    config, capcut_case, fps, has_audio,
):
    result, installation = capcut_case
    result.metadata.fps, result.metadata.has_audio = fps, has_audio
    before = result.model_dump()
    payload = create_capcut_project(result, config, installation)
    folder = Path(payload["project_path"])
    content = json.loads((folder / "draft_content.json").read_text())
    segments = content["tracks"][0]["segments"]
    assert len(segments) == payload["shot_count"] == 2
    assert payload["project_name"].startswith("Snooker - Match - ")
    assert [s["desc"] for s in segments] == ["Shot 1", "Shot 3"]
    assert segments[0]["target_timerange"]["start"] == 0
    assert segments[1]["target_timerange"]["start"] == segments[0]["target_timerange"]["duration"]
    assert content["duration"] == sum(s["target_timerange"]["duration"] for s in segments)
    assert content["fps"] == fps
    assert content["canvas_config"]["width"] == 1280
    media = content["materials"]["videos"][0]
    assert media["path"] == Path(result.source_path).as_posix()
    assert media["duration"] == 20_000_000  # Extending a trim has full source handles.
    assert media["has_audio"] == has_audio
    assert segments[0]["volume"] == (1.0 if has_audio else 0.0)
    assert abs(segments[0]["source_timerange"]["start"] / 1e6 - 1.03) <= 0.5 / fps + 1e-6
    all_ids = {item["id"] for values in content["materials"].values() for item in values}
    for segment in segments:
        assert segment["material_id"] in all_ids
        assert set(segment["extra_material_refs"]).issubset(all_ids)
        assert segment["source_timerange"]["duration"] == segment["target_timerange"]["duration"]
    project = json.loads((folder / "Timelines" / "project.json").read_text())
    assert project["main_timeline_id"] == content["id"]
    mirror = folder / "Timelines" / content["id"] / "draft_content.json"
    assert mirror.read_bytes() == (folder / "draft_content.json").read_bytes()
    assert result.model_dump() == before
    assert Path(result.source_path).read_bytes() == b"original media"
    assert not list(folder.rglob("*.mp4"))


def test_capcut_preserves_existing_index_and_edits_on_repeated_handoffs(config, capcut_case):
    result, installation = capcut_case
    installation.projects_dir.mkdir()
    index_path = installation.projects_dir / "root_meta_info.json"
    original = {"all_draft_store": [{"draft_id": "existing", "custom": [42]}], "draft_ids": 17, "custom": "keep"}
    original_bytes = json.dumps(original).encode()
    index_path.write_bytes(original_bytes)
    first = create_capcut_project(result, config, installation)
    edit = Path(first["project_path"]) / "draft_content.json"
    edit.write_text("user edit")
    second = create_capcut_project(result, config, installation)
    index = json.loads(index_path.read_text())
    assert first["project_path"] != second["project_path"]
    assert first["project_id"] != second["project_id"]
    assert edit.read_text() == "user edit"
    assert index["all_draft_store"][-1] == original["all_draft_store"][0]
    assert index["draft_ids"] == 19
    assert index["custom"] == "keep"
    assert any(p.read_bytes() == original_bytes for p in installation.projects_dir.glob("*.bak"))
    assert Path(index["all_draft_store"][0]["draft_json_file"]).is_file()


def test_corrupt_capcut_index_is_untouched_and_new_folder_is_rolled_back(config, capcut_case):
    result, installation = capcut_case
    installation.projects_dir.mkdir()
    index = installation.projects_dir / "root_meta_info.json"
    index.write_bytes(b"unreadable index")
    with pytest.raises(CapCutUnavailable, match="left untouched"):
        create_capcut_project(result, config, installation)
    assert index.read_bytes() == b"unreadable index"
    assert list(installation.projects_dir.iterdir()) == [index]


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, 21), (float("nan"), 2), (1, float("inf"))])
def test_invalid_capcut_trims_do_not_create_a_project(config, capcut_case, start, end):
    result, installation = capcut_case
    result.shots = [ShotRecord(shot_id=1, cue_strike=1, clip_start=start, clip_end=end)]
    with pytest.raises(ValueError, match="invalid source boundaries"):
        create_capcut_project(result, config, installation)
    assert not installation.projects_dir.exists()


def test_empty_selection_and_missing_source_fail_without_writing(config, capcut_case):
    result, installation = capcut_case
    result.shots = []
    with pytest.raises(ValueError, match="No included live shots"):
        create_capcut_project(result, config, installation)
    Path(result.source_path).unlink()
    with pytest.raises(ValueError, match="source video is missing"):
        create_capcut_project(result, config, installation)
    assert not installation.projects_dir.exists()


def test_unsupported_capcut_version_reports_reason(config, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    apps = tmp_path / "CapCut" / "Apps"
    (apps / "9.6.0.1000").mkdir(parents=True)
    (apps / "CapCut.exe").touch()
    monkeypatch.setattr("snooker_ai.rendering.capcut.os", SimpleNamespace(name="nt", environ=os.environ))
    with pytest.raises(CapCutUnavailable, match="unverified project format"):
        find_capcut(config)


def _saved_case(config, capcut_case, tmp_path, monkeypatch):
    result, installation = capcut_case
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    store = JobStore(config)
    result.job_id = store.create(result.source_path)
    store.save_analysis(result)
    monkeypatch.setattr("apps.api.main.find_capcut", lambda _cfg: installation)
    return result, installation, store


def test_capcut_api_uses_saved_edits_and_preserves_job_state(config, capcut_case, tmp_path, monkeypatch):
    result, installation, store = _saved_case(config, capcut_case, tmp_path, monkeypatch)
    launched = []
    monkeypatch.setattr("apps.api.main.launch_capcut", launched.append)
    before = store.get_meta(result.job_id)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/capcut").json()["available"] is True
        response = client.post(f"/api/jobs/{result.job_id}/open-capcut")
    assert response.status_code == 200
    payload = response.json()
    assert payload["shot_count"] == 2
    assert payload["launched"] is True
    assert "Home screen" in payload["message"]
    assert launched == [installation]
    assert store.get_meta(result.job_id) == before
    assert (store.job_dir(result.job_id) / "export" / "capcut_project.json").is_file()


def test_capcut_launch_failure_keeps_the_created_project(config, capcut_case, tmp_path, monkeypatch):
    result, _, _ = _saved_case(config, capcut_case, tmp_path, monkeypatch)
    def fail(_installation):
        raise OSError("launch unavailable")
    monkeypatch.setattr("apps.api.main.launch_capcut", fail)
    with TestClient(create_app(config)) as client:
        response = client.post(f"/api/jobs/{result.job_id}/open-capcut")
    assert response.status_code == 200
    payload = response.json()
    assert payload["launched"] is False
    assert payload["launch_error"] == "launch unavailable"
    assert Path(payload["project_path"]).is_dir()
    assert "manually" in payload["message"]


def test_capcut_api_rejects_missing_analysis_busy_job_and_cross_site_origin(config, capcut_case, tmp_path, monkeypatch):
    result, installation, _ = _saved_case(config, capcut_case, tmp_path, monkeypatch)
    lock = _locks.setdefault(result.job_id, threading.Lock())
    with TestClient(create_app(config)) as client:
        assert client.post("/api/jobs/missing/open-capcut").status_code == 404
        assert client.post(f"/api/jobs/{result.job_id}/open-capcut", headers={"Origin": "https://example.org"}).status_code == 403
        lock.acquire()
        try:
            assert client.post(f"/api/jobs/{result.job_id}/open-capcut").status_code == 409
        finally:
            lock.release()
    assert not installation.projects_dir.exists()


def test_capcut_api_reports_unavailable_and_invalid_source(config, capcut_case, tmp_path, monkeypatch):
    result, installation, _ = _saved_case(config, capcut_case, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        Path(result.source_path).unlink()
        assert client.post(f"/api/jobs/{result.job_id}/open-capcut").status_code == 400
        def unavailable(_cfg):
            raise CapCutUnavailable("CapCut not found")
        monkeypatch.setattr("apps.api.main.find_capcut", unavailable)
        assert client.get("/api/capcut").json() == {"available": False, "reason": "CapCut not found"}
        assert client.post(f"/api/jobs/{result.job_id}/open-capcut").status_code == 409
    assert not installation.projects_dir.exists()


@pytest.mark.asyncio
async def test_capcut_project_preparation_does_not_block_progress(config, capcut_case, tmp_path, monkeypatch):
    result, _, _ = _saved_case(config, capcut_case, tmp_path, monkeypatch)
    started, release = threading.Event(), threading.Event()
    original = create_capcut_project
    def slow_create(*args):
        started.set()
        if not release.wait(3):
            raise RuntimeError("project creation blocked the web request loop")
        return original(*args)
    monkeypatch.setattr("apps.api.main.create_capcut_project", slow_create)
    monkeypatch.setattr("apps.api.main.launch_capcut", lambda _: None)
    transport = httpx.ASGITransport(app=create_app(config))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.post(f"/api/jobs/{result.job_id}/open-capcut"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            progress = await asyncio.wait_for(client.get(f"/api/jobs/{result.job_id}/progress"), 1)
            assert progress.status_code == 200
            assert not task.done()
        finally:
            release.set()
            response = await task
        assert response.status_code == 200
