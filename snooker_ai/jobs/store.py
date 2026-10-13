"""Filesystem-backed job store with resume support."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from snooker_ai.config import Config
from snooker_ai.pipeline.algorithms import Algorithm
from snooker_ai.types import (
    AnalysisResult,
    JobProgress,
    JobStatus,
    ShotRecord,
    ShotUpdate,
    ShotSelection,
)
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.persistence import atomic_text

logger = get_logger("jobs")


class JobStore:
    def __init__(self, config: Config, root: Optional[Path] = None):
        self.config = config
        self.root = Path(root) if root else Path(config.get("paths.jobs_dir", "data/jobs"))
        self.root.mkdir(parents=True, exist_ok=True)
        uploads = Path(config.get("paths.uploads_dir", "data/uploads"))
        uploads.mkdir(parents=True, exist_ok=True)
        self.uploads = uploads

    def create(
        self,
        source_path: str | Path,
        mode: str = "strict",
        algorithm: str | Algorithm = Algorithm.CLASSIC,
        selection: str | ShotSelection = ShotSelection.ALL,
    ) -> str:
        selected_algorithm = Algorithm.parse(algorithm)
        selected_selection = ShotSelection(selection)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "job_id": job_id,
            "source_path": str(Path(source_path).resolve()),
            "mode": mode,
            "algorithm": selected_algorithm.value,
            "selection": selected_selection.value,
            "created_at": time.time(),
            "status": JobStatus.PENDING.value,
            "progress": 0.0,
            "stage": "pending",
            "message": "Created",
        }
        try:
            meta["source_fingerprint"] = self.source_fingerprint(source_path)
        except OSError:
            pass
        self._write_meta(job_id, meta)
        self.update_progress(job_id, 0.0, JobStatus.PENDING, "Created")
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _meta_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _write_meta(self, job_id: str, meta: dict[str, Any]) -> None:
        path = self._meta_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API/review UI may read progress while an analyzer is updating it.
        # Writing directly to job.json exposes a brief empty/partial document
        # and can make a concurrent reader fail JSON decoding. A unique temp
        # file also keeps simultaneous analyzer processes from sharing scratch
        # state; replace publishes only a complete JSON document.
        atomic_text(path, json.dumps(meta, indent=2))

    def get_meta(self, job_id: str) -> dict[str, Any]:
        path = self._meta_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("algorithm", Algorithm.CLASSIC.value)
        meta.setdefault("selection", ShotSelection.ALL.value)
        return meta

    def update_progress(
        self,
        job_id: str,
        progress: float,
        status: JobStatus | str,
        message: str = "",
        error: Optional[str] = None,
        shots_detected: int = 0,
    ) -> JobProgress:
        meta = self.get_meta(job_id)
        status_val = status.value if isinstance(status, JobStatus) else status
        meta.update(
            {
                "progress": float(progress),
                "status": status_val,
                "stage": status_val,
                "message": message,
                "error": error,
                "shots_detected": shots_detected,
                "updated_at": time.time(),
            }
        )
        self._write_meta(job_id, meta)
        return JobProgress(
            job_id=job_id,
            status=JobStatus(status_val),
            progress=float(progress),
            stage=status_val,
            message=message,
            error=error,
            shots_detected=shots_detected,
            updated_at=meta["updated_at"],
        )

    def load_analysis(self, job_id: str) -> AnalysisResult:
        path = self.job_dir(job_id) / "analysis.json"
        if not path.exists():
            raise FileNotFoundError(f"Analysis not found for job {job_id}")
        result = AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
        self._sync_review_timestamps(result)
        return result

    @staticmethod
    def source_fingerprint(source: str | Path) -> dict[str, Any]:
        path = Path(source).resolve()
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def load_selection_analysis(self, job_id: str) -> AnalysisResult:
        """Read the saved shot timeline without loading dense detector vectors."""
        try:
            data = json.loads((self.job_dir(job_id) / "timeline.json").read_text("utf-8"))
            if not all(key in data for key in ("source_path", "metadata", "shots")):
                raise ValueError("Incomplete saved timeline")
            result = AnalysisResult.model_validate(data)
            self._sync_review_timestamps(result)
            return result
        except (OSError, ValueError, TypeError):
            return self.load_analysis(job_id)

    def validate_saved_source(self, job_id: str) -> dict[str, Any]:
        meta = self.get_meta(job_id)
        current = self.source_fingerprint(meta["source_path"])
        bound = meta.get("source_fingerprint")
        if bound is None:
            proxy_dir = self.job_dir(job_id) / str(self.config.get("paths.proxy_subdir", "proxy"))
            try:
                values = json.loads((proxy_dir / "proxy.signature.json").read_text("utf-8"))[
                    "video"
                ]["source"]
                bound = {
                    "path": str(Path(values[0]).resolve()),
                    "size": values[1],
                    "mtime_ns": values[2],
                }
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                # Legacy jobs have no source fingerprint. Reject a source
                # changed after their saved analysis rather than reusing it.
                saved = self.job_dir(job_id) / "analysis.json"
                if not saved.is_file():
                    saved = self.job_dir(job_id) / "timeline.json"
                if current["mtime_ns"] > saved.stat().st_mtime_ns:
                    raise ValueError("Source video changed since detection; analyze it again first")
        if bound is not None and current != bound:
            raise ValueError("Source video changed since detection; analyze it again first")
        return current

    def seed_selection_analysis(self, job_id: str, parent_id: str) -> AnalysisResult:
        result = self.load_selection_analysis(parent_id)
        result.job_id = job_id
        result.selection = ShotSelection.ALL
        result.pot_selection_summary = {}
        for shot in result.shots:
            base_included = shot.evidence.pop("pots_only_base_included", shot.included)
            if not shot.evidence.get("pots_only_manual_override"):
                shot.included = bool(base_included)
            shot.manual_review_required = bool(
                shot.evidence.pop("pots_only_base_review_required", shot.manual_review_required)
            )
            shot.evidence.pop("pot_confirmation", None)
            shot.pot_status = None
        self.save_analysis(result)
        # Score observations belong to the source/window, not the selection
        # policy. The selector validates signatures/ranges before reuse.
        for path in (self.job_dir(parent_id) / "pot_score_cache").glob("window_*.json"):
            atomic_text(
                self.job_dir(job_id) / "pot_score_cache" / path.name, path.read_text("utf-8")
            )
        return result

    @staticmethod
    def _sync_review_timestamps(result: AnalysisResult) -> None:
        for shot in result.shots:
            if shot.user_modified:
                # The review API edits legacy names; exports validate both
                # representations. Keep them aligned after edits/merge/split.
                shot.clip_start_timestamp = shot.clip_start
                shot.clip_end_timestamp = shot.clip_end
                shot.cue_strike_timestamp = shot.cue_strike
                shot.physical_stop_timestamp = shot.ball_motion_end
                shot.last_ball_motion_timestamp = min(
                    shot.last_ball_motion_timestamp, shot.physical_stop_timestamp
                )
                shot.stop_confirmation_timestamp = max(
                    shot.stop_confirmation_timestamp, shot.physical_stop_timestamp
                )

    def save_analysis(self, result: AnalysisResult) -> None:
        self._sync_review_timestamps(result)
        if result.pot_selection_summary:
            result.pot_selection_summary["included"] = sum(shot.included for shot in result.shots)
        path = self.job_dir(result.job_id) / "analysis.json"
        atomic_text(path, result.model_dump_json())
        slim = {
            "job_id": result.job_id,
            "source_path": result.source_path,
            "mode": result.mode.value,
            "selection": result.selection.value,
            "pot_selection_summary": result.pot_selection_summary,
            "original_duration": result.original_duration,
            "edited_duration": result.edited_duration,
            "pause_removed_seconds": result.pause_removed_seconds,
            "shots": [s.model_dump() for s in result.shots],
            "scenes": [s.model_dump() for s in result.scenes],
            "metadata": result.metadata.model_dump(),
        }
        atomic_text(self.job_dir(result.job_id) / "timeline.json", json.dumps(slim, indent=2))

    def update_shot(self, job_id: str, shot_id: int, update: ShotUpdate) -> ShotRecord:
        result = self.load_analysis(job_id)
        found = None
        for s in result.shots:
            if s.shot_id == shot_id:
                data = update.model_dump(exclude_unset=True)
                if "included" in data and result.selection is not ShotSelection.ALL:
                    s.evidence["pots_only_manual_override"] = True
                for k, v in data.items():
                    setattr(s, k, v)
                s.user_modified = True
                s.manual_review_required = False
                found = s
                break
        if found is None:
            raise KeyError(f"Shot {shot_id} not found")
        # Recompute edited duration
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        self.save_analysis(result)
        self._export_corrections(job_id, result)
        return found

    def delete_shot(self, job_id: str, shot_id: int) -> None:
        result = self.load_analysis(job_id)
        deleted = [shot.cue_strike_timestamp for shot in result.shots if shot.shot_id == shot_id]
        result.shots = [s for s in result.shots if s.shot_id != shot_id]
        for i, s in enumerate(result.shots, start=1):
            s.shot_id = i
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        self.save_analysis(result)
        self._export_corrections(job_id, result, deleted_strikes=deleted)

    def add_shot(self, job_id: str, shot: ShotRecord) -> ShotRecord:
        result = self.load_analysis(job_id)
        if result.selection is not ShotSelection.ALL:
            shot.evidence["pots_only_manual_override"] = True
        next_id = max((s.shot_id for s in result.shots), default=0) + 1
        shot.shot_id = next_id
        shot.user_modified = True
        result.shots.append(shot)
        result.shots.sort(key=lambda s: s.clip_start)
        for i, s in enumerate(result.shots, start=1):
            s.shot_id = i
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        self.save_analysis(result)
        self._export_corrections(job_id, result)
        return shot

    def merge_shots(self, job_id: str, shot_ids: list[int]) -> ShotRecord:
        """Merge two or more shots into one continuous clip."""
        if len(shot_ids) < 2:
            raise ValueError("Need at least two shot ids to merge")
        result = self.load_analysis(job_id)
        selected = [s for s in result.shots if s.shot_id in set(shot_ids)]
        if len(selected) < 2:
            raise KeyError(f"Could not find shots to merge: {shot_ids}")
        selected.sort(key=lambda s: s.clip_start)
        keep = selected[0]
        keep.clip_start = min(s.clip_start for s in selected)
        keep.clip_end = max(s.clip_end for s in selected)
        keep.cue_strike = selected[0].cue_strike
        keep.preparation_start = min(s.preparation_start for s in selected)
        keep.ball_motion_start = min(s.ball_motion_start for s in selected)
        keep.ball_motion_end = max(s.ball_motion_end for s in selected)
        keep.shot_confidence = max(s.shot_confidence for s in selected)
        keep.possible_replay = any(s.possible_replay for s in selected)
        keep.included = any(s.included for s in selected)
        keep.user_modified = True
        if result.selection is not ShotSelection.ALL:
            keep.evidence["pots_only_manual_override"] = True
        keep.manual_review_required = False
        drop = {s.shot_id for s in selected[1:]}
        result.shots = [s for s in result.shots if s.shot_id not in drop]
        for i, s in enumerate(sorted(result.shots, key=lambda x: x.clip_start), start=1):
            s.shot_id = i
        result.shots.sort(key=lambda s: s.clip_start)
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        self.save_analysis(result)
        self._export_corrections(job_id, result)
        # Return the merged shot (id may have renumbered)
        for s in result.shots:
            if (
                abs(s.clip_start - keep.clip_start) < 1e-6
                and abs(s.clip_end - keep.clip_end) < 1e-6
            ):
                return s
        return keep

    def split_shot(self, job_id: str, shot_id: int, at_time: float) -> list[ShotRecord]:
        """Split a shot into two at the given timestamp."""
        result = self.load_analysis(job_id)
        target = next((s for s in result.shots if s.shot_id == shot_id), None)
        if target is None:
            raise KeyError(f"Shot {shot_id} not found")
        if not (target.clip_start + 0.05 < at_time < target.clip_end - 0.05):
            raise ValueError("Split time must be strictly inside the clip bounds")
        left = target.model_copy(deep=True)
        right = target.model_copy(deep=True)
        left.clip_end = at_time
        left.cue_strike = min(left.cue_strike, at_time)
        left.ball_motion_start = min(left.ball_motion_start, at_time)
        left.ball_motion_end = min(left.ball_motion_end, at_time)
        left.user_modified = True
        if result.selection is not ShotSelection.ALL:
            left.evidence["pots_only_manual_override"] = True
            right.evidence["pots_only_manual_override"] = True
        left.manual_review_required = False
        right.clip_start = at_time
        right.cue_strike = max(at_time, min(right.cue_strike, right.clip_end))
        right.ball_motion_start = max(right.ball_motion_start, at_time)
        right.preparation_start = at_time
        right.user_modified = True
        right.manual_review_required = False
        result.shots = [s for s in result.shots if s.shot_id != shot_id] + [left, right]
        result.shots.sort(key=lambda s: s.clip_start)
        for i, s in enumerate(result.shots, start=1):
            s.shot_id = i
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)
        self.save_analysis(result)
        self._export_corrections(job_id, result)
        return result.shots

    def _recompute_durations(self, result: AnalysisResult) -> None:
        result.edited_duration = sum(s.duration() for s in result.shots if s.included)
        result.pause_removed_seconds = max(0.0, result.original_duration - result.edited_duration)

    def _export_corrections(
        self, job_id: str, result: AnalysisResult, *, deleted_strikes: Optional[list[float]] = None
    ) -> None:
        meta = self.get_meta(job_id)
        meta["export_needs_update"] = True
        meta["shots_detected"] = len(result.shots)
        if result.pot_selection_summary:
            meta["pot_selection_summary"] = result.pot_selection_summary
        self._write_meta(job_id, meta)
        path = self.job_dir(job_id) / "corrections.json"
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            deleted = [float(t) for t in previous.get("deleted_strikes", [])]
        except (OSError, ValueError, TypeError):
            deleted = []
        deleted = sorted(set(deleted + (deleted_strikes or [])))
        # Explicitly adding a shot back restores it, including on re-analysis.
        deleted = [
            t
            for t in deleted
            if not any(
                shot.user_modified and abs(shot.cue_strike - t) <= 1.5 for shot in result.shots
            )
        ]
        path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "source_path": result.source_path,
                    "shots": [s.model_dump() for s in result.shots],
                    "deleted_strikes": deleted,
                    "exported_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for p in sorted(self.root.iterdir(), reverse=True):
            if (p / "job.json").exists():
                try:
                    jobs.append(self.get_meta(p.name))
                except Exception:
                    continue
        return jobs

    def copy_upload(self, upload_path: Path, original_name: str) -> Path:
        dest = self.uploads / f"{uuid.uuid4().hex[:10]}_{original_name}"
        shutil.copy2(upload_path, dest)
        return dest
