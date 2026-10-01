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
    jid = store.create(tmp_path / "fake.mp4", mode="strict")
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
        mode=EditMode.STRICT,
        original_duration=60,
        edited_duration=5,
    )
    store.save_analysis(result)
    loaded = store.load_analysis(jid)
    assert len(loaded.shots) == 1

    updated = store.update_shot(jid, 1, ShotUpdate(clip_start=2.5, clip_end=9.0))
    assert updated.clip_start == 2.5
    assert updated.clip_start_timestamp == 2.5
    assert updated.clip_end_timestamp == 9.0
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


def test_restart_analysis_api_endpoint(config, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from apps.api.main import create_app

    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video content")

    store = JobStore(config, root=tmp_path / "jobs")
    jid = store.create(source, mode="strict")

    runs = []

    def fake_run_analysis(job_id, source_path, mode, resume, force_reanalyze=False, config=None):
        runs.append((job_id, source_path, mode, resume, force_reanalyze))

    monkeypatch.setattr("apps.api.main._run_analysis", fake_run_analysis)

    with TestClient(create_app(config)) as client:
        res = client.post(f"/api/jobs/{jid}/restart")

    assert res.status_code == 200
    assert res.json()["status"] == "restarted"
    assert len(runs) == 1
    assert runs[0] == (jid, str(source.resolve()), EditMode.STRICT, True, True)


def test_analyzer_preserves_user_edits_on_restart(config, tmp_path, monkeypatch):
    from snooker_ai.pipeline.analyzer import Analyzer

    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    job_dir = tmp_path / "jobs" / "test_job"
    job_dir.mkdir(parents=True)

    analyzer = Analyzer(config, job_dir=job_dir)

    # Save initial corrections with a user modified shot
    corrections = {
        "job_id": "test_job",
        "source_path": "video.mp4",
        "shots": [
            {
                "shot_id": 1,
                "cue_strike": 5.0,
                "cue_strike_timestamp": 5.0,
                "clip_start": 2.0,
                "clip_start_timestamp": 2.0,
                "clip_end": 10.0,
                "clip_end_timestamp": 10.0,
                "included": True,
                "user_modified": True,
            }
        ],
    }
    (job_dir / "corrections.json").write_text(json.dumps(corrections), encoding="utf-8")

    # Newly built auto shots from detector
    new_shots = [
        ShotRecord(
            shot_id=1,
            cue_strike=5.0,
            clip_start=3.0,
            clip_end=8.0,
            included=True,
            user_modified=False,
        )
    ]

    preserved = analyzer._preserve_user_edits(new_shots, "test_job")
    assert len(preserved) == 1
    assert preserved[0].user_modified is True
    assert preserved[0].clip_start == 2.0
    assert preserved[0].clip_end == 10.0


def test_analyzer_coarse_and_dense_cache_roundtrip(config, tmp_path):
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.types import FrameFeatures, SceneSegment, StrikeCandidate

    job_dir = tmp_path / "cache_job"
    job_dir.mkdir(parents=True)
    source = tmp_path / "test.mp4"
    source.write_bytes(b"test data")

    analyzer = Analyzer(config, job_dir=job_dir)
    sig = analyzer._analysis_signature(source)
    assert isinstance(sig, str) and len(sig) > 0

    features = [FrameFeatures(t=0.0, table_confidence=0.9)]
    scenes = [SceneSegment(start=0.0, end=10.0)]
    candidates = [StrikeCandidate(timestamp=2.0, confidence=0.8)]

    # Coarse cache roundtrip
    assert analyzer._load_coarse_cache(sig) is None
    analyzer._save_coarse_cache(sig, features, scenes, candidates)
    loaded = analyzer._load_coarse_cache(sig)
    assert loaded is not None
    loaded_feats, loaded_scenes, loaded_cands = loaded
    assert len(loaded_feats) == 1
    assert loaded_feats[0].t == 0.0
    assert len(loaded_scenes) == 1
    assert len(loaded_cands) == 1

    # Dense cache roundtrip
    assert analyzer._load_dense_window(sig, 0, 1.0, 5.0) is None
    analyzer._save_dense_window(sig, 0, 1.0, 5.0, features)
    dense_loaded = analyzer._load_dense_window(sig, 0, 1.0, 5.0)
    assert dense_loaded is not None
    assert len(dense_loaded) == 1

