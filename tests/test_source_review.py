"""Source reviews are explicit, bounded edits of a preserved parent version."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from snooker_ai.evaluation.source_review import apply_source_review
from snooker_ai.jobs.store import JobStore
from snooker_ai.types import AnalysisResult, PotStatus, ShotRecord, TimelineEvent, VideoMetadata
from tools.rebuild_source_review import rebuild, sha256


@pytest.fixture
def reviewed_case(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"unchanged source fixture")
    result = AnalysisResult(
        job_id="parent",
        source_path=str(source),
        original_duration=90,
        metadata=VideoMetadata(path=str(source), duration=90, fps=25),
        shots=[
            ShotRecord(
                shot_id=i,
                cue_strike=t,
                clip_start=t - 2,
                clip_end=t + 5,
                included=False,
                pot_status=PotStatus.UNKNOWN,
            )
            for i, t in enumerate([10, 30, 70], 1)
        ],
        events=[TimelineEvent(event_type="cue_strike", timestamp=70, metadata={"shot_id": 3})],
    )
    manifest = {
        "name": "explicit review",
        "source_reviewed": True,
        "source_duration_seconds": 90,
        "updates": [{"shot_id": 2, "strike": 30, "status": "pot", "reason": "Red enters pocket"}],
        "additions": [
            {
                "contact_interval": [49.8, 50.2],
                "clip": [47.8, 56.2],
                "status": "pot",
                "reason": "Observed address, departure and pink pot",
            }
        ],
        "frames": [{"first_shot_id": 1, "last_shot_id": 3}],
    }
    return result, manifest


def test_import_preserves_parent_cuts_and_remaps_events_without_inventing_stops(reviewed_case):
    original, manifest = reviewed_case
    snapshot = original.model_dump_json()
    result = apply_source_review(original, manifest)
    assert original.model_dump_json() == snapshot
    assert [s.cue_strike for s in result.shots] == [10, 30, 50, 70]
    assert all(s.included for s in result.shots)
    assert [(s.clip_start, s.clip_end) for s in result.shots if s.cue_strike != 50] == [
        (s.clip_start, s.clip_end) for s in original.shots
    ]
    assert result.shots[0].pot_status is PotStatus.UNKNOWN  # opening context stays context
    added = result.shots[2]
    assert added.user_modified and added.manual_review_required
    assert added.evidence["stop_confirmed"] is False
    assert added.evidence["source_review"]["contact_interval"] == [49.8, 50.2]
    assert result.events[-1].metadata["shot_id"] == 4
    assert len(result.events) == 2  # an editing boundary is not an observed ball-stop event
    assert result.pot_selection_summary["non_scoring_context_shots"] == 2


def test_reclassifying_the_only_foul_clears_the_old_summary_count(reviewed_case):
    original, manifest = reviewed_case
    original.shots[1].pot_status = PotStatus.FOUL
    original.pot_selection_summary = {"foul": 1, "points": 7}
    result = apply_source_review(original, manifest)
    assert result.pot_selection_summary["foul"] == 0
    assert result.pot_selection_summary["points"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "unreviewed",
        "duration",
        "duplicate",
        "moved",
        "nan",
        "overlap",
        "bounds",
        "empty_reason",
        "replay",
    ],
)
def test_invalid_review_fails_before_changing_the_parent(reviewed_case, defect):
    original, manifest = reviewed_case
    if defect == "unreviewed":
        manifest["source_reviewed"] = False
    elif defect == "duration":
        manifest["source_duration_seconds"] = 91
    elif defect == "duplicate":
        manifest["updates"] *= 2
    elif defect == "moved":
        manifest["updates"][0]["strike"] = 31
    elif defect == "nan":
        manifest["additions"][0]["contact_interval"][0] = float("nan")
    elif defect == "overlap":
        manifest["additions"][0].update(contact_interval=[29.8, 30.2], clip=[27.8, 36.2])
    elif defect == "bounds":
        manifest["additions"][0]["clip"][1] = 91
    elif defect == "empty_reason":
        manifest["updates"][0]["reason"] = ""
    else:
        original.shots[1].possible_replay = True
    snapshot = original.model_dump_json()
    with pytest.raises(ValueError):
        apply_source_review(original, manifest)
    assert original.model_dump_json() == snapshot


def prepare_job(config, tmp_path, monkeypatch, reviewed_case):
    from tools import rebuild_source_review as command

    config._data["paths"]["jobs_dir"] = str(tmp_path / "jobs")
    config._data["paths"]["uploads_dir"] = str(tmp_path / "uploads")
    monkeypatch.setattr(command, "load_config", lambda: config)
    result, manifest = reviewed_case
    store = JobStore(config)
    parent = store.create(result.source_path, selection="pots_and_fouls")
    result.job_id = parent
    store.save_analysis(result)
    parent_path = store.job_dir(parent) / "analysis.json"
    manifest = copy.deepcopy(manifest)
    manifest.update(
        parent_job_id=parent,
        parent_analysis_sha256=sha256(parent_path),
        source_size_bytes=Path(result.source_path).stat().st_size,
        source_sha256=sha256(result.source_path),
    )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(manifest))
    return store, parent, path


def test_rebuild_creates_separate_version_and_preserves_parent_files(
    config, tmp_path, monkeypatch, reviewed_case
):
    store, parent, path = prepare_job(config, tmp_path, monkeypatch, reviewed_case)
    snapshot = {p.name: p.read_bytes() for p in store.job_dir(parent).iterdir() if p.is_file()}
    child = rebuild(path)
    assert child != parent
    assert store.get_meta(child)["derived_from_job_id"] == parent
    assert store.get_meta(child)["status"] == "ready_for_review"
    assert len(store.load_analysis(child).shots) == 4
    assert snapshot == {
        p.name: p.read_bytes() for p in store.job_dir(parent).iterdir() if p.is_file()
    }


@pytest.mark.parametrize("which", ["source", "parent"])
def test_changed_fingerprint_cannot_create_a_repair_job(
    config, tmp_path, monkeypatch, reviewed_case, which
):
    store, parent, path = prepare_job(config, tmp_path, monkeypatch, reviewed_case)
    manifest = json.loads(path.read_text())
    manifest["source_sha256" if which == "source" else "parent_analysis_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    before = set(store.root.iterdir())
    with pytest.raises(ValueError, match="changed|fingerprint"):
        rebuild(path)
    assert set(store.root.iterdir()) == before


def test_render_failure_leaves_recoverable_new_job_and_parent_intact(
    config, tmp_path, monkeypatch, reviewed_case
):
    from tools import rebuild_source_review as command

    store, parent, path = prepare_job(config, tmp_path, monkeypatch, reviewed_case)
    original = (store.job_dir(parent) / "analysis.json").read_bytes()
    monkeypatch.setattr(
        command,
        "Exporter",
        lambda config: SimpleNamespace(
            export=lambda *args: (_ for _ in ()).throw(RuntimeError("render interrupted"))
        ),
    )
    with pytest.raises(RuntimeError, match="render interrupted"):
        rebuild(path, render=True)
    children = [p for p in store.root.iterdir() if p.is_dir() and p.name != parent]
    assert len(children) == 1
    assert store.get_meta(children[0].name)["status"] == "failed"
    assert (children[0] / "analysis.json").is_file()
    assert (store.job_dir(parent) / "analysis.json").read_bytes() == original
