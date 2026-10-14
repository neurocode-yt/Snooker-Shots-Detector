"""Independent selection, durable exports, API boundaries and actual MP4 output."""

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.shorts import (
    PRESETS,
    ShortsManager,
    ShortsSettings,
    geometry,
    preview_png,
    render_short,
    selected_shots,
    shot_filename,
    shot_title,
    title_overlay,
)
from snooker_ai.types import AnalysisResult, ShotRecord, ShotUpdate, VideoMetadata
from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg, run_command


@pytest.fixture
def saved_shorts(config, tmp_path):
    config._data["device"] = "cpu"
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake source")
    store = JobStore(config)
    job = store.create(source)
    result = AnalysisResult(
        job_id=job,
        source_path=str(source),
        original_duration=4,
        metadata=VideoMetadata(path=str(source), duration=4, fps=25, width=640, height=360),
        shots=[
            ShotRecord(shot_id=1, clip_start=0.2, cue_strike=0.4, clip_end=0.9),
            ShotRecord(shot_id=2, clip_start=1, cue_strike=1.3, clip_end=1.8, included=False),
            ShotRecord(shot_id=3, clip_start=2, cue_strike=2.2, clip_end=2.9),
            ShotRecord(shot_id=4, clip_start=3, cue_strike=3.2, clip_end=3.8, possible_replay=True),
        ],
    )
    store.save_analysis(result)
    store.update_progress(job, 1, "ready_for_review", "Saved shots", shots_detected=4)
    return config, store, result


def wait_batch(manager, job, batch_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        batch = manager.read(job, batch_id)
        if batch["status"] not in {"queued", "rendering"} and not JobLease.is_running(
            manager.root(job), ".shorts.lock"
        ):
            return batch
        time.sleep(0.02)
    pytest.fail("Shorts batch did not finish")


def fake_render(result, shot, settings, output, config):
    output.write_bytes(f"rendered {shot.shot_id} {shot.clip_start} {shot.clip_end}".encode())


def test_saved_selection_keeps_live_included_shots_without_mutation(saved_shorts):
    config, store, result = saved_shorts
    before = result.model_dump()
    assert [s.shot_id for s in selected_shots(result, config)] == [1, 3]
    assert result.model_dump() == before
    result.shots[2].camera_views = ["slow_motion_replay"]
    assert [s.shot_id for s in selected_shots(result, config)] == [1]
    result.shots[0].linked_live_shot_id = 9
    with pytest.raises(ValueError, match="Include at least"):
        selected_shots(result, config)


@pytest.mark.parametrize(
    "patch", [{"clip_start": -1}, {"clip_end": 10}, {"clip_end": 0.1}, {"clip_start": float("nan")}]
)
def test_invalid_boundaries_cannot_generate_short(saved_shorts, patch):
    config, _, result = saved_shorts
    for key, value in patch.items():
        setattr(result.shots[0], key, value)
    with pytest.raises(ValueError, match="invalid saved boundaries"):
        selected_shots(result, config)


@pytest.mark.parametrize(
    "patch",
    [
        {"player_names": "  "},
        {"background": "red;movie=secret"},
        {"video_zoom": float("nan")},
        {"video_zoom": 9},
        {"title_y": 1900},
        {"shot_names": {1: "a" * 161}},
        {"voiceover": True},
    ],
)
def test_settings_are_bounded_and_only_support_visual_work(saved_shorts, patch):
    with pytest.raises(ValidationError):
        ShortsSettings.model_validate({"player_names": "Judd", **patch})


def test_names_are_literal_unicode_and_filenames_are_safe():
    settings = ShortsSettings(
        player_names='Ronnie / Judd: <final> "match"', shot_names={3: "赵心童"}
    )
    assert shot_title(settings, 3) == "赵心童 - Shot 03"
    assert shot_filename(settings, 3) == "赵心童 - Shot 03.mp4"
    filename = shot_filename(settings, 1)
    assert not any(c in filename for c in '<>:"/\\|?*')
    assert filename.endswith("Shot 01.mp4")
    assert shot_filename(ShortsSettings(player_names="CON"), 1) == "CON - Shot 01.mp4"


def test_preview_and_overlay_keep_table_geometry_and_fit_long_names():
    settings = ShortsSettings(player_names="Very long player name " * 7)
    assert geometry((1920, 1080), settings) == (1080, 608, 0, 656)
    assert geometry((1080, 1920), settings) == (1080, 1920, 0, 0)
    overlay = title_overlay(settings, 3)
    assert overlay.size == (1080, 1920)
    bounds = overlay.getbbox()
    assert bounds and 0 <= bounds[0] < bounds[2] <= 1080
    assert bounds[1] >= settings.title_y and bounds[3] < 600


def test_failure_resume_keeps_completed_clips_and_zip_only_contains_selected_videos(
    saved_shorts, monkeypatch
):
    config, store, result = saved_shorts
    settings = ShortsSettings(player_names="Judd Trump", shot_names={3: "Mark Allen"})
    calls = []

    def fail_once(result, shot, settings, output, config):
        calls.append(shot.shot_id)
        if shot.shot_id == 3 and calls.count(3) == 1:
            raise RuntimeError("Simulated interrupted encoder")
        fake_render(result, shot, settings, output, config)

    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", fail_once)
    monkeypatch.setattr(
        store, "load_analysis", lambda *a: pytest.fail("Must use the compact timeline")
    )
    parent = store.get_meta(result.job_id)
    manager = ShortsManager(config, store)
    try:
        started = manager.submit(result.job_id, settings)
        first = wait_batch(manager, result.job_id, started["batch_id"])
        assert first["status"] == "failed" and first["completed"] == 1
        assert "Simulated" in first["error"]
        complete = wait_batch(
            manager, result.job_id, manager.submit(result.job_id, settings)["batch_id"]
        )
        assert complete["status"] == "completed" and complete["completed"] == 2
        assert calls == [1, 3, 3]
        with zipfile.ZipFile(
            manager.directory(result.job_id, complete["batch_id"]) / "shorts.zip"
        ) as archive:
            assert archive.namelist() == ["Judd Trump - Shot 01.mp4", "Mark Allen - Shot 03.mp4"]
        assert manager.submit(result.job_id, settings)["status"] == "completed"
        assert calls == [1, 3, 3]
        assert store.get_meta(result.job_id) == parent
        assert store.load_selection_analysis(result.job_id).model_dump() == result.model_dump()
    finally:
        manager.close()


def test_duplicate_batches_block_and_existing_snapshot_survives_later_edits(
    saved_shorts, monkeypatch
):
    config, store, result = saved_shorts
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        fake_render(*args)

    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", blocked)
    manager = ShortsManager(config, store)
    try:
        started = manager.submit(result.job_id, ShortsSettings(player_names="Player"))
        assert entered.wait(5)
        with pytest.raises(ValueError, match="already queued"):
            manager.submit(result.job_id, ShortsSettings(player_names="Other"))
        store.update_shot(result.job_id, 3, ShotUpdate(included=False))
        release.set()
        complete = wait_batch(manager, result.job_id, started["batch_id"])
        assert complete["completed"] == 2
        assert len(selected_shots(store.load_selection_analysis(result.job_id), config)) == 1
    finally:
        release.set()
        manager.close()


def test_changed_layout_and_corrupt_clip_do_not_reuse_wrong_exports(saved_shorts, monkeypatch):
    config, store, result = saved_shorts
    calls = []

    def record(*args):
        calls.append(args[1].shot_id)
        fake_render(*args)

    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", record)
    manager = ShortsManager(config, store)
    try:
        settings = ShortsSettings(player_names="Player")
        batch = wait_batch(
            manager, result.job_id, manager.submit(result.job_id, settings)["batch_id"]
        )
        (
            manager.directory(result.job_id, batch["batch_id"]) / batch["clips"][0]["filename"]
        ).write_bytes(b"broken")
        again = wait_batch(
            manager, result.job_id, manager.submit(result.job_id, settings)["batch_id"]
        )
        assert again["status"] == "completed" and calls == [1, 3, 1]
        changed = manager.submit(
            result.job_id, settings.model_copy(update={"background": "#112233"})
        )
        assert changed["batch_id"] != batch["batch_id"]
        assert wait_batch(manager, result.job_id, changed["batch_id"])["status"] == "completed"
    finally:
        manager.close()


def test_abandoned_batch_is_resumable_after_process_exit(saved_shorts, monkeypatch):
    config, store, result = saved_shorts
    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", fake_render)
    manager = ShortsManager(config, store)
    try:
        batch = wait_batch(
            manager,
            result.job_id,
            manager.submit(result.job_id, ShortsSettings(player_names="Player"))["batch_id"],
        )
        directory = manager.directory(result.job_id, batch["batch_id"])
        batch.update(status="rendering")
        manager.save(directory, batch)
        assert manager.read(result.job_id, batch["batch_id"])["status"] == "interrupted"
        assert not JobLease.is_running(manager.root(result.job_id), ".shorts.lock")
    finally:
        manager.close()


def test_progress_read_cannot_overwrite_a_just_completed_batch(saved_shorts, monkeypatch):
    config, store, result = saved_shorts
    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", fake_render)
    manager = ShortsManager(config, store)
    try:
        batch = wait_batch(
            manager,
            result.job_id,
            manager.submit(result.job_id, ShortsSettings(player_names="Player"))["batch_id"],
        )
        manifest = manager.directory(result.job_id, batch["batch_id"]) / "manifest.json"
        original_read = Path.read_text
        reads = 0

        def finish_between_read_and_lock(path, *args, **kwargs):
            nonlocal reads
            if path == manifest:
                reads += 1
                if reads == 1:
                    return json.dumps({**batch, "status": "rendering"})
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", finish_between_read_and_lock)
        assert manager.read(result.job_id, batch["batch_id"])["status"] == "completed"
        assert json.loads(original_read(manifest))["status"] == "completed"
        assert reads == 2
    finally:
        manager.close()


def test_source_changes_and_unknown_name_overrides_are_rejected(saved_shorts):
    config, store, result = saved_shorts
    manager = ShortsManager(config, store)
    try:
        with pytest.raises(ValueError, match="included live shots"):
            manager.submit(
                result.job_id, ShortsSettings(player_names="Player", shot_names={2: "Excluded"})
            )
        Path(result.source_path).write_bytes(b"replaced source")
        with pytest.raises(ValueError, match="changed"):
            manager.submit(result.job_id, ShortsSettings(player_names="Player"))
    finally:
        manager.close()


def test_api_setup_export_and_download_keep_original_job_intact(saved_shorts, monkeypatch):
    config, store, result = saved_shorts
    monkeypatch.setattr("snooker_ai.rendering.shorts.render_short", fake_render)
    parent = store.get_meta(result.job_id)
    with TestClient(create_app(config)) as client:
        base = f"/api/jobs/{result.job_id}/shorts"
        setup = client.get(base).json()
        assert [s["shot_id"] for s in setup["shots"]] == [1, 3]
        assert set(setup["presets"]) == set(PRESETS)
        assert (
            client.post(
                base, json={"player_names": "Player"}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        response = client.post(base, json={"player_names": "Judd Trump"})
        assert response.status_code == 202
        batch = wait_batch(
            client.app.state.shorts_manager, result.job_id, response.json()["batch_id"]
        )
        batch_url = f"{base}/{batch['batch_id']}"
        assert client.get(batch_url).json()["status"] == "completed"
        assert client.get(batch_url + "/zip").status_code == 200
        assert client.get(batch_url + "/clips/1").content.startswith(b"rendered 1")
        assert client.get(batch_url + "/clips/2").status_code == 404
        assert client.get(base + "/not-a-batch").status_code == 404
        assert client.get(base).json()["batch"]["completed"] == 2
        assert store.get_meta(result.job_id) == parent


@pytest.mark.parametrize("audio", [False, True])
def test_real_ffmpeg_short_and_preview_have_expected_geometry_duration_and_audio(
    saved_shorts, audio
):
    config, store, result = saved_shorts
    source = Path(result.source_path)
    command = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=25:duration=4",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4"]
    run_command(
        command
        + [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    result.metadata.has_audio = audio
    settings = ShortsSettings(player_names="Judd Trump", shot_names={3: "Mark Allen"})
    output = source.parent / "test-short.mp4"
    render_short(result, result.shots[0], settings, output, config)
    raw = ffprobe_json(output)
    video = next(s for s in raw["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert abs(float(video["duration"]) - 0.7) < 0.09
    assert any(s["codec_type"] == "audio" for s in raw["streams"]) == audio
    with Image.open(io.BytesIO(preview_png(result, settings, result.shots[0]))) as image:
        assert image.size == (432, 768)
        assert image.getpixel((0, 0)) == (0, 0, 0)
    assert not output.with_suffix(".png").exists()
