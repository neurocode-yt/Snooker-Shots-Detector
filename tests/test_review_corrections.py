"""Review corrections must survive exports and subsequent detection runs."""

import json

from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import AnalysisResult, ShotRecord, ShotUpdate, VideoMetadata


def _job(config, tmp_path):
    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    store = JobStore(config)
    job_id = store.create(tmp_path / "source.mp4")
    shots = [
        ShotRecord(shot_id=1, cue_strike=5, clip_start=3, clip_end=9, ball_motion_end=9),
        ShotRecord(shot_id=2, cue_strike=15, clip_start=13, clip_end=19, ball_motion_end=19),
    ]
    result = AnalysisResult(
        job_id=job_id, source_path="source.mp4", shots=shots,
        metadata=VideoMetadata(path="source.mp4", duration=25), original_duration=25,
    )
    store.save_analysis(result)
    return store, job_id, shots


def _validate_export(config, shots):
    Exporter(config)._validate_strict_boundaries(shots, source_duration=25.0, source_fps=30.0)


def test_manual_boundary_edits_are_exportable(config, tmp_path):
    store, job_id, _ = _job(config, tmp_path)
    shot = store.update_shot(job_id, 1, ShotUpdate(clip_start=2, clip_end=10, ball_motion_end=8))
    assert shot.physical_stop_timestamp == 8
    _validate_export(config, [shot])


def test_merged_and_split_shots_have_consistent_export_timestamps(config, tmp_path):
    store, job_id, _ = _job(config, tmp_path)
    merged = store.merge_shots(job_id, [1, 2])
    _validate_export(config, [merged])
    # Splitting in the preparation interval is a supported manual edit too.
    split = store.split_shot(job_id, merged.shot_id, 4)
    _validate_export(config, split)


def test_deleted_false_shot_stays_deleted_after_reanalysis_and_other_edits(config, tmp_path):
    store, job_id, shots = _job(config, tmp_path)
    store.delete_shot(job_id, 1)
    store.update_shot(job_id, 1, ShotUpdate(clip_end=20))
    corrections = json.loads((store.job_dir(job_id) / "corrections.json").read_text())
    assert corrections["deleted_strikes"] == [5]
    restored = Analyzer(config, store.job_dir(job_id))._preserve_user_edits(shots, job_id)
    assert len(restored) == 1
    assert restored[0].cue_strike == 15
    assert restored[0].clip_end_timestamp == 20


def test_deleting_the_only_shot_is_preserved(config, tmp_path):
    store, job_id, shots = _job(config, tmp_path)
    store.delete_shot(job_id, 1)
    store.delete_shot(job_id, 1)
    restored = Analyzer(config, store.job_dir(job_id))._preserve_user_edits(shots, job_id)
    assert restored == []


def test_explicitly_adding_back_a_deleted_shot_restores_it(config, tmp_path):
    store, job_id, shots = _job(config, tmp_path)
    store.delete_shot(job_id, 1)
    store.add_shot(job_id, shots[0])
    restored = Analyzer(config, store.job_dir(job_id))._preserve_user_edits(shots, job_id)
    assert len(restored) == 2


def test_old_manual_edits_with_stale_aliases_can_be_exported(config, tmp_path):
    store, job_id, _ = _job(config, tmp_path)
    path = store.job_dir(job_id) / "analysis.json"
    payload = json.loads(path.read_text())
    payload["shots"][0].update(user_modified=True, clip_start=2, clip_end=10)
    path.write_text(json.dumps(payload))
    result = store.load_analysis(job_id)
    _validate_export(config, [result.shots[0]])


def test_old_automatic_include_decision_cannot_override_fresh_replay_detection(config, tmp_path):
    store, job_id, shots = _job(config, tmp_path)
    fresh = shots[0].model_copy(update={"included": False, "possible_replay": True}, deep=True)
    restored = Analyzer(config, store.job_dir(job_id))._preserve_user_edits([fresh], job_id)
    assert len(restored) == 1
    assert restored[0].included is False
