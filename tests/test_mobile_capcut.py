"""Phone clip handoff keeps saved selection, cuts, order and per-clip checkpoints."""
import importlib
import json
from pathlib import Path

import cv2
import pytest

from snooker_ai.jobs.store import JobStore
from snooker_ai.types import AnalysisResult, ShotRecord, VideoMetadata
from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg, run_command


@pytest.fixture
def case(config, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "apps/android/app/src/main/python/mobile"))
    module = importlib.import_module("mobile_capcut")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    store = JobStore(config)
    job = store.create(source)
    result = AnalysisResult(job_id=job, source_path=str(source), original_duration=10,
        metadata=VideoMetadata(path=str(source), duration=10, fps=30, width=320, height=180, has_audio=True),
        shots=[ShotRecord(shot_id=4, cue_strike=5, clip_start=4, clip_end=7),
               ShotRecord(shot_id=1, cue_strike=1, clip_start=0, clip_end=3),
               ShotRecord(shot_id=2, cue_strike=3, clip_start=2, clip_end=4, included=False),
               ShotRecord(shot_id=3, cue_strike=4, clip_start=3, clip_end=5, possible_replay=True)])
    store.save_analysis(result)
    store.update_progress(job, 1, "ready_for_review", shots_detected=4)
    manager = module.CapCutPreparation(config)
    calls = []

    def render(source, shot, destination, metadata):
        calls.append(shot.shot_id)
        destination.write_bytes(f"shot-{shot.shot_id}".encode())

    monkeypatch.setattr(manager, "render", render)
    return manager, job, result, calls


def test_interruption_reuses_completed_clips_and_rejects_damaged_output(case, monkeypatch):
    manager, job, _, calls = case
    render = manager.render

    def fail_second(*args):
        if len(calls) == 1:
            raise OSError("process interrupted")
        render(*args)

    monkeypatch.setattr(manager, "render", fail_second)
    with pytest.raises(OSError):
        manager.prepare(job)
    monkeypatch.setattr(manager, "render", render)
    state = manager.prepare(job)
    assert calls == [1, 4]  # first completed clip is not rendered again
    assert [f["shot_id"] for f in state["files"]] == [1, 4]
    directory = manager.directory(job) / state["generation"]
    (directory / "shot_0001.mp4").write_bytes(b"corrupt")
    manager.prepare(job)
    assert calls == [1, 4, 1]


def test_saved_trim_changes_create_a_new_generation_without_overwriting_prior_media(case):
    manager, job, result, calls = case
    first = manager.prepare(job)
    result.shots[0].clip_end = 6
    result.shots[0].clip_end_timestamp = 6
    manager.store.save_analysis(result)
    second = manager.prepare(job)
    assert first["generation"] != second["generation"]
    assert (manager.directory(job) / first["generation"] / "shot_0002.mp4").exists()
    assert calls == [1, 4, 1, 4]


def test_handoff_never_rewrites_saved_analysis_and_validates_job_paths(case):
    manager, job, result, _ = case
    path = manager.store.job_dir(job) / "analysis.json"
    before = path.read_bytes()
    state = manager.prepare(job)
    assert state["total"] == 2 and path.read_bytes() == before
    for value in ["../job", "a/b", "..", "", "x" * 161]:
        with pytest.raises(ValueError):
            manager.directory(value)
    assert json.loads((manager.directory(job) / "status.json").read_text())["status"] == "ready"


def test_dead_worker_cannot_leave_native_handoff_waiting_forever(case):
    manager, job, _, _ = case
    manager.save(job, {"status": "preparing", "completed": 1, "total": 2, "message": "Saved one clip"})
    state = manager.status(job)
    assert state["status"] == "interrupted" and state["completed"] == 1


def test_real_mobile_clip_preserves_requested_source_frames_audio_and_duration(case, monkeypatch):
    manager, _, result, _ = case
    source = Path(result.source_path)
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=5",
                 "-f", "lavfi", "-i", "sine=frequency=700:duration=5", "-c:v", "libx264",
                 "-g", "90", "-c:a", "aac", "-shortest", str(source)])
    shot = ShotRecord(shot_id=1, cue_strike=2, clip_start=1.1, clip_end=3.1)
    output = source.with_name("clip.mp4")
    module = importlib.import_module("mobile_capcut")
    module.CapCutPreparation.render(manager, source, shot, output, result.metadata)
    probe = ffprobe_json(output)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 60
    assert float(video["duration"]) == pytest.approx(2, abs=.001)
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    expected, actual = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    try:
        expected.set(cv2.CAP_PROP_POS_FRAMES, 33)
        ok, first = expected.read()
        ok2, exported = actual.read()
        assert ok and ok2
        assert cv2.PSNR(first, exported) > 32
    finally:
        expected.release()
        actual.release()
