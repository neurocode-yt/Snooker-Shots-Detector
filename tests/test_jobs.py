import json
from pathlib import Path

from snooker_ai.jobs.store import JobStore
from snooker_ai.types import (
    AnalysisResult,
    EditMode,
    ShotRecord,
    ShotUpdate,
    VideoMetadata,
)


def test_job_store_roundtrip(config, tmp_path):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    store = JobStore(config, root=tmp_path / "jobs")
    jid = store.create(tmp_path / "fake.mp4", mode="natural")
    meta = store.get_meta(jid)
    assert meta["job_id"] == jid

    result = AnalysisResult(
        job_id=jid,
        source_path=str(tmp_path / "fake.mp4"),
        metadata=VideoMetadata(path=str(tmp_path / "fake.mp4"), duration=60, width=1280, height=720, fps=25),
        shots=[
            ShotRecord(
                shot_id=1,
                cue_strike=5.0,
                clip_start=3.0,
                clip_end=8.0,
                ball_motion_end=7.0,
                shot_confidence=0.8,
            )
        ],
        mode=EditMode.NATURAL,
        original_duration=60,
        edited_duration=5,
    )
    store.save_analysis(result)
    loaded = store.load_analysis(jid)
    assert len(loaded.shots) == 1

    updated = store.update_shot(jid, 1, ShotUpdate(clip_start=2.5, clip_end=9.0))
    assert updated.clip_start == 2.5
    assert updated.user_modified is True

    store.add_shot(
        jid,
        ShotRecord(
            shot_id=0,
            cue_strike=12.0,
            clip_start=10.0,
            clip_end=15.0,
            ball_motion_end=14.0,
            shot_confidence=0.9,
        ),
    )
    loaded = store.load_analysis(jid)
    assert len(loaded.shots) == 2

    split = store.split_shot(jid, 1, 5.0)
    assert len(split) == 3
    merged = store.merge_shots(jid, [1, 2])
    assert merged.clip_start <= 5.0
    assert len(store.load_analysis(jid).shots) == 2


def test_progress_metadata_is_published_atomically(config, tmp_path, monkeypatch):
    store = JobStore(config, root=tmp_path / "jobs")
    jid = store.create(tmp_path / "input.mp4", mode="strict")
    meta_path = store._meta_path(jid)
    original_write_text = Path.write_text
    observed_existing_meta = []

    def inspect_temp_write(path, data, *args, **kwargs):
        if path.suffix == ".tmp":
            # The public path retains a complete document until the fully
            # written temporary file is atomically published.
            observed_existing_meta.append(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", inspect_temp_write)
    store.update_progress(jid, 0.5, "analyzing", "Halfway")

    assert observed_existing_meta
    assert json.loads(meta_path.read_text(encoding="utf-8"))["message"] == "Halfway"


def test_progress_metadata_retries_windows_sharing_violation(config, tmp_path, monkeypatch):
    store = JobStore(config, root=tmp_path / "jobs")
    jid = store.create(tmp_path / "input.mp4", mode="strict")
    original_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(path, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "Access is denied")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    store.update_progress(jid, 0.5, "analyzing", "Halfway")

    assert attempts["count"] == 3
    assert store.get_meta(jid)["message"] == "Halfway"
