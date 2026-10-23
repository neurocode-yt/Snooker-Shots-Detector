"""Bounded, resumable per-shot media preparation for CapCut's Android importer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import ExportRequest
from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command
from snooker_ai.utils.persistence import atomic_text
from snooker_ai.utils.work_control import WorkControl, WorkPaused, current_control, check_pause

_manager = None


class CapCutPreparation:
    def __init__(self, config):
        self.config = config
        self.store = JobStore(config)
        self.lock = threading.RLock()
        self.threads = {}

    def directory(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", job_id):
            raise ValueError("Invalid saved video")
        return self.store.job_dir(job_id) / "export/mobile_capcut"

    def status(self, job_id):
        path = self.directory(job_id) / "status.json"
        try:
            state = json.loads(path.read_text())
            thread = self.threads.get(job_id)
            if state["status"] == "preparing" and not (thread and thread.is_alive()):
                return {**state, "status": "interrupted", "message": "CapCut preparation stopped. Saved clips are preserved; tap Open in CapCut to continue"}
            return state
        except FileNotFoundError:
            return {"status": "pending", "message": "Preparing editable clips", "completed": 0, "total": 0}

    def save(self, job_id, state):
        atomic_text(self.directory(job_id) / "status.json", json.dumps(state))

    def start(self, job_id):
        self.directory(job_id)  # validate before touching a path
        with self.lock:
            if self.threads.get(job_id) and self.threads[job_id].is_alive():
                return self.status(job_id)
            if any(thread.is_alive() for thread in self.threads.values()):
                raise ValueError("Another video's CapCut clips are still being prepared")
            meta = self.store.get_meta(job_id)
            if meta["status"] not in {"completed", "ready_for_review"}:
                raise ValueError("Finish or resume this video before opening CapCut")
            WorkControl(self.directory(job_id)).clear()
            self.save(job_id, {"status": "preparing", "message": "Preparing editable clips", "completed": 0, "total": 0})
            thread = threading.Thread(target=self.run, args=(job_id,), daemon=True, name="capcut-clips")
            self.threads[job_id] = thread
            thread.start()
            return self.status(job_id)

    def render(self, source, shot, destination, metadata):
        # Transcode with accurate input seeking: decode the preceding keyframe
        # and discard to the requested PTS. Never use keyframe-only stream copy.
        # One encoder at a time avoids a whole-match fan-out on a phone.
        args = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
                "-ss", f"{shot.clip_start:.9f}", "-i", str(source),
                "-t", f"{shot.duration():.9f}", "-map", "0:v:0",
                *(["-map", "0:a:0"] if metadata.has_audio else []),
                "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                "-crf", str(self.config.get("export.crf", 18)), "-pix_fmt", "yuv420p",
                "-fps_mode:v", "passthrough",
                *(["-c:a", "aac", "-b:a", "192k"] if metadata.has_audio else ["-an"]),
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(destination)]
        run_command(args, timeout=max(120, shot.duration() * 30))
        Exporter(self.config)._verify_clip(destination, shot, source_has_audio=metadata.has_audio,
                                            source_fps=metadata.fps, verify_timing=True)

    def prepare(self, job_id):
        result = self.store.load_selection_analysis(job_id)
        exporter = Exporter(self.config)
        shots = sorted(exporter._filter_shots(result.shots, ExportRequest()), key=lambda shot: shot.clip_start)
        if not shots:
            raise ValueError("Include at least one live shot before opening CapCut")
        source = Path(result.source_path)
        fingerprint = {"source": self.store.source_fingerprint(source), "version": 1,
                       "crf": self.config.get("export.crf", 18),
                       "shots": [[s.shot_id, s.clip_start, s.clip_end] for s in shots]}
        generation = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        directory = self.directory(job_id) / generation
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "clips.json"
        try:
            completed = json.loads(checkpoint.read_text())
        except (FileNotFoundError, ValueError):
            completed = {}
        files = []
        for index, shot in enumerate(shots):
            check_pause()
            name = f"shot_{index + 1:04d}.mp4"
            path = directory / name
            entry = completed.get(name, {})
            valid = path.is_file() and entry.get("sha256") == digest(path)
            if not valid:
                temporary = path.with_suffix("." + uuid.uuid4().hex + ".partial.mp4")
                self.render(source, shot, temporary, result.metadata)
                with temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                temporary.replace(path)
                completed[name] = {"sha256": digest(path), "shot_id": shot.shot_id,
                                   "clip_start": shot.clip_start, "clip_end": shot.clip_end}
                atomic_text(checkpoint, json.dumps(completed))
            files.append({"filename": name, "shot_id": shot.shot_id})
            self.save(job_id, {"status": "preparing", "message": f"CapCut: {index + 1}/{len(shots)} clips saved",
                               "completed": index + 1, "total": len(shots)})
        # Edits on another screen cannot silently hand off an older selection.
        latest = self.store.load_selection_analysis(job_id)
        selected = sorted(exporter._filter_shots(latest.shots, ExportRequest()), key=lambda s: s.clip_start)
        if [[s.shot_id, s.clip_start, s.clip_end] for s in selected] != fingerprint["shots"]:
            raise ValueError("Shots changed during preparation. Tap Open in CapCut again to use the saved edits")
        state = {"status": "ready", "message": f"{len(files)} editable clips ready for CapCut",
                 "completed": len(files), "total": len(files), "generation": generation, "files": files}
        self.save(job_id, state)
        return state

    def run(self, job_id):
        token = current_control.set(WorkControl(self.directory(job_id)))
        try:
            self.prepare(job_id)
        except WorkPaused:
            state = self.status(job_id)
            self.save(job_id, {**state, "status": "interrupted", "message": "CapCut preparation saved. Reopen to continue"})
        except Exception as error:
            state = self.status(job_id)
            self.save(job_id, {**state, "status": "failed", "message": str(error)})
        finally:
            current_control.reset(token)

    def pause(self):
        with self.lock:
            for job_id, thread in self.threads.items():
                if thread.is_alive():
                    WorkControl(self.directory(job_id)).request()

    def recover(self):
        for path in self.store.root.glob("*/export/mobile_capcut/status.json"):
            job_id = path.parents[2].name
            try:
                if self.status(job_id)["status"] in {"preparing", "interrupted"}:
                    self.start(job_id)
            except (OSError, ValueError):
                # A missing/deleted original must not prevent detector recovery.
                continue

    def work_items(self):
        with self.lock:
            items = []
            for job_id, thread in self.threads.items():
                if not thread.is_alive():
                    continue
                state = self.status(job_id)
                items.append({"task_id": "capcut-" + job_id, "kind": "capcut", "job_id": job_id,
                              "source_path": self.store.get_meta(job_id)["source_path"],
                              "has_saved_detection": True,
                              "completed": state["completed"], "total": state["total"],
                              "status": "preparing", "message": state["message"],
                              "progress": state["completed"] / max(1, state["total"]),
                              "can_pause": False, "can_resume": False})
            return items


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def configure(config):
    global _manager
    _manager = CapCutPreparation(config)


def start(job_id):
    return json.dumps(_manager.start(job_id))


def status(job_id):
    return json.dumps(_manager.status(job_id))
