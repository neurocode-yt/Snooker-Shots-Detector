"""REST API and review UI for snooker-ai."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from snooker_ai.config import Config, load_config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.jobs.store import JobStore
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.queue import MatchQueue
from snooker_ai.pipeline.analyzer import Analyzer as Analyzer
from snooker_ai.pipeline.algorithms import Algorithm, algorithm_capabilities, create_analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.rendering.capcut import (
    CapCutUnavailable,
    create_capcut_project,
    find_capcut,
    launch_capcut,
)
from snooker_ai.rendering.preprocessor import KeepRange, VideoPreprocessor
from snooker_ai.types import (
    EditMode,
    ExportRequest,
    JobStatus,
    ShotRecord,
    ShotUpdate,
    ShotSelection,
)
from snooker_ai.utils.logging import get_logger, setup_logging

logger = get_logger("api")

# In-process job locks
_locks: dict[str, threading.Lock] = {}
_ACTIVE_STATUSES = {status.value for status in (
    JobStatus.PENDING, JobStatus.VALIDATING, JobStatus.PROXY, JobStatus.ANALYZING,
    JobStatus.DETECTING, JobStatus.REFINING, JobStatus.SEGMENTING, JobStatus.EXPORTING,
)}


class VideoFileResponse(FileResponse):
    """Range-capable media response with efficient local streaming chunks."""

    # Starlette's 64 KiB default causes one async file/thread hop per chunk.
    # On Windows that throttles localhost playback below the source bitrate.
    chunk_size = 1024 * 1024


def _explorer_window_handles() -> list[int]:
    """Return top-level Windows Explorer folder-window handles."""

    if os.name != "nt":
        return []
    import ctypes

    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def collect(hwnd: int, _lparam: int) -> bool:
        class_name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, class_name, len(class_name))
        if class_name.value in {"CabinetWClass", "ExploreWClass"}:
            handles.append(int(hwnd))
        return True

    user32.EnumWindows(collect, 0)
    return handles


def _focus_window(hwnd: int) -> bool:
    """Restore and foreground a window, including from a web-server thread."""

    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    foreground = int(user32.GetForegroundWindow())
    current_thread = int(kernel32.GetCurrentThreadId())
    foreground_thread = int(user32.GetWindowThreadProcessId(foreground, None))
    attached = bool(
        foreground_thread
        and foreground_thread != current_thread
        and user32.AttachThreadInput(current_thread, foreground_thread, True)
    )
    try:
        user32.ShowWindowAsync(hwnd, 9)  # SW_RESTORE
        user32.BringWindowToTop(hwnd)
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
        return bool(user32.SetForegroundWindow(hwnd))
    finally:
        if attached:
            user32.AttachThreadInput(current_thread, foreground_thread, False)


def _open_folder(path: Path) -> bool:
    """Open a folder and, on Windows, make the new Explorer window visible."""

    resolved = path.resolve()
    if os.name != "nt":
        subprocess.Popen(["xdg-open", str(resolved)])
        return True

    existing = set(_explorer_window_handles())
    # /n is important: without it Explorer can silently reuse a background window.
    subprocess.Popen(["explorer.exe", "/n,", str(resolved)])
    deadline = time.monotonic() + 3.0
    target: int | None = None
    while time.monotonic() < deadline:
        current = _explorer_window_handles()
        new_handles = [hwnd for hwnd in current if hwnd not in existing]
        if new_handles:
            target = new_handles[-1]
            break
        time.sleep(0.05)
    return _focus_window(target) if target is not None else False


class AnalyzeBody(BaseModel):
    source_path: Optional[str] = None
    job_id: Optional[str] = None
    mode: str = "strict"
    algorithm: Optional[Algorithm] = None
    selection: Optional[ShotSelection] = None
    resume: bool = True
    auto_export: bool = True
    submission_token: Optional[str] = Field(default=None, max_length=128)


class RestartBody(BaseModel):
    mode: Optional[str] = None
    algorithm: Optional[Algorithm] = None
    selection: Optional[ShotSelection] = None
    force: bool = True
    auto_export: Optional[bool] = None


class ExportBody(BaseModel):
    output_name: str = "highlights.mp4"
    mode: Optional[str] = None
    include_replays: bool = False
    accurate: bool = True
    export_clips: bool = True
    export_joined: bool = True
    min_confidence: float = 0.0
    min_importance: float = 0.0


class AddShotBody(BaseModel):
    cue_strike: float
    clip_start: float
    clip_end: float
    ball_motion_end: Optional[float] = None
    preparation_start: Optional[float] = None
    shot_confidence: float = 1.0


class MergeShotsBody(BaseModel):
    shot_ids: list[int] = Field(..., min_length=2)


class SplitShotBody(BaseModel):
    at_time: float


class SourceRangeBody(BaseModel):
    start: float
    end: float


class PreprocessBody(BaseModel):
    source_path: str
    ranges: list[SourceRangeBody] = Field(..., min_length=1, max_length=200)


def _run_analysis(
    job_id: str,
    source: str,
    mode: EditMode,
    resume: bool,
    force_reanalyze: bool = False,
    config: Optional[Config] = None,
    auto_export: bool = True,
) -> None:
    cfg = config or load_config()
    store = JobStore(cfg)
    lock = _locks.setdefault(job_id, threading.Lock())
    if not lock.acquire(blocking=False):
        logger.warning("Job %s already running", job_id)
        return
    lease = JobLease(store.job_dir(job_id))
    if not lease.acquire():
        lock.release()
        logger.warning("Job %s is running in another process", job_id)
        return
    try:
        job_meta = store.get_meta(job_id)
        algorithm = Algorithm.parse(job_meta.get("algorithm"))
        selection = ShotSelection(job_meta.get("selection", "all"))
        reused_result = None
        fast_variant = bool(job_meta.get("reuse_saved_detection") and not force_reanalyze)
        if fast_variant:
            if store.source_fingerprint(source) != job_meta.get("source_fingerprint"):
                raise ValueError("Source video changed; create fresh highlights from a new analysis")
            if (store.job_dir(job_id)/"analysis.json").is_file():
                reused_result = store.load_analysis(job_id)
            else:
                parent_id = job_meta["derived_from_job_id"]
                parent_path = store.job_dir(parent_id)/job_meta.get("parent_analysis_file", "analysis.json")
                parent_stat = parent_path.stat()
                if [parent_stat.st_size, parent_stat.st_mtime_ns] != job_meta.get("parent_analysis_identity"):
                    raise ValueError("Saved analysis changed; click the highlights button again")
                reused_result = store.seed_selection_analysis(job_id, parent_id)
            store.update_progress(job_id, .75, JobStatus.SEGMENTING,
                                  "Reusing saved shots; checking points without rerunning detection")
        elif force_reanalyze and job_meta.get("reuse_saved_detection"):
            job_meta["reuse_saved_detection"] = False
            store._write_meta(job_id, job_meta)

        # Selection-only variants can reuse the parent's completed detector
        # output in a separate job; its source, edits and export stay intact.
        parent_id = job_meta.get("derived_from_job_id")
        if not fast_variant and parent_id and not (store.job_dir(job_id) / "analysis.json").exists():
            try:
                parent_meta = store.get_meta(parent_id)
                parent = store.load_analysis(parent_id)
                if (Algorithm.parse(parent_meta.get("algorithm")) is algorithm
                        and Path(parent.source_path).resolve() == Path(source).resolve()):
                    parent.job_id = job_id
                    parent.selection = ShotSelection.ALL
                    parent.pot_selection_summary = {}
                    for shot in parent.shots:
                        shot.included = bool(shot.evidence.pop("pots_only_base_included", shot.included))
                        shot.manual_review_required = bool(shot.evidence.pop("pots_only_base_review_required", shot.manual_review_required))
                        shot.evidence.pop("pots_only_manual_override", None)
                        shot.evidence.pop("pot_confirmation", None)
                        shot.pot_status = None
                    store.save_analysis(parent)
            except (OSError, ValueError):
                logger.info("Parent analysis unavailable; running a fresh detector scan")

        def prog(p: float, stage: str, msg: str) -> None:
            if selection is not ShotSelection.ALL:
                p *= .75
                if stage == JobStatus.READY_FOR_REVIEW.value:
                    stage = JobStatus.SEGMENTING.value
                    msg = "Detection complete; checking scoring shots"
            elif auto_export:
                p *= 0.90
                if stage == JobStatus.READY_FOR_REVIEW.value:
                    stage = JobStatus.SEGMENTING.value
                    msg = "Detection complete; preparing automatic export"
            store.update_progress(job_id, p, stage, msg)

        if reused_result is not None:
            result = reused_result
        else:
            analyzer = create_analyzer(cfg, store.job_dir(job_id), algorithm)
            result = analyzer.analyze(
                source, job_id, mode=mode, progress=prog, resume=resume,
                force_reanalyze=force_reanalyze,
            )
        if selection is not ShotSelection.ALL:
            from snooker_ai.pot_detection.selector import PotSelector
            result = PotSelector(cfg, store.job_dir(job_id)).apply(result, progress=lambda frac, msg:
                store.update_progress(job_id, .75+.15*frac, JobStatus.SEGMENTING, msg), selection=selection)
            store.save_analysis(result)
            meta = store.get_meta(job_id)
            meta["pot_selection_summary"] = result.pot_selection_summary
            store._write_meta(job_id, meta)
        n_shots = len(result.shots) if result and result.shots is not None else 0
        kept = sum(shot.included for shot in result.shots)
        unknown = int(result.pot_selection_summary.get("unknown", 0))
        if auto_export:
            request = ExportRequest(mode=mode, output_path="highlights.mp4", export_clips=False)
            exporter = Exporter(cfg)
            if exporter._filter_shots(result.shots, request):
                store.update_progress(
                    job_id, 0.92, JobStatus.EXPORTING,
                    "Creating finished video automatically", shots_detected=n_shots,
                )
                rendered = exporter.export(result, store.job_dir(job_id) / "export", request)
                if not rendered.joined_path or not rendered.joined_path.is_file():
                    raise RuntimeError("The finished video was not created. Use Continue to retry export.")
                meta = store.get_meta(job_id)
                meta["auto_export_ready"] = bool(rendered.joined_path and rendered.joined_path.is_file())
                meta["export_needs_update"] = False
                store._write_meta(job_id, meta)
                label = "Pots + fouls highlights" if selection is ShotSelection.POTS_AND_FOULS else "Pots-only video"
                message = (f"{label} ready: {kept} kept from {n_shots} shots; {unknown} need review"
                           if selection is not ShotSelection.ALL
                           else f"Finished video ready: {n_shots} detected shots")
            else:
                if selection is not ShotSelection.ALL and unknown:
                    store.update_progress(job_id, 1., JobStatus.READY_FOR_REVIEW,
                        f"No confirmed scoring shots; {unknown} shots need review because scores were inconclusive",
                        shots_detected=n_shots)
                    return
                message = ("Analysis complete; no scoring shots detected" if selection is not ShotSelection.ALL
                           else "Analysis complete; no playable shots detected")
            store.update_progress(
                job_id, 1.0, JobStatus.COMPLETED, message, shots_detected=n_shots,
            )
            return
        store.update_progress(
            job_id,
            1.0,
            JobStatus.READY_FOR_REVIEW,
            (f"{'Pots + fouls' if selection is ShotSelection.POTS_AND_FOULS else 'Pots only'}: {kept} kept from {n_shots} shots; {unknown} need review"
             if selection is not ShotSelection.ALL else f"Detected {n_shots} shots"),
            shots_detected=n_shots,
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        meta = store.get_meta(job_id)
        store.update_progress(
            job_id, min(.99, float(meta.get("progress", 0))), JobStatus.FAILED,
            "Processing stopped. Saved work is preserved; use Continue to retry.",
            error=str(exc), shots_detected=int(meta.get("shots_detected", 0)),
        )
    finally:
        lease.release()
        lock.release()


def _job_busy(store: JobStore, job_id: str) -> bool:
    return (_locks.setdefault(job_id, threading.Lock()).locked()
            or JobLease.is_running(store.job_dir(job_id), ".queue.lock")
            or JobLease.is_running(store.job_dir(job_id)))


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or load_config()
    setup_logging(str(cfg.get("log_level", "INFO")))
    cfg.ensure_dirs()
    store = JobStore(cfg)

    queue = MatchQueue(cfg, lambda *args: _run_analysis(*args))

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(queue.start)
        try:
            yield
        finally:
            await asyncio.to_thread(queue.close)

    app = FastAPI(
        title="Snooker AI API",
        version="0.1.0",
        description="Automatic snooker shot detection and video editing",
        lifespan=lifespan,
    )
    app.state.match_queue = queue

    def enqueue(job_id, source, mode, resume, force=False, auto_export=True):
        if not queue.submit(job_id, source, mode, resume, force, auto_export):
            raise HTTPException(409, f"Job {job_id} is already queued or processing")

    @app.get("/api/queue")
    async def queue_status() -> dict[str, Any]:
        return queue.snapshot()
    origins = cfg.get("api.cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = web_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Snooker AI API</h1><p>See /docs</p>")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        max_gb = float(cfg.get("api.max_upload_gb", 50.0))
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        stem = Path(file.filename or "video").stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
        dest = (
            store.uploads
            / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem}{suffix}"
        )
        # stream to disk
        size = 0
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_gb * (1024**3):
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"File exceeds {max_gb} GB limit")
                out.write(chunk)
        return {"path": str(dest.resolve()), "filename": file.filename, "size": size}

    @app.get("/api/algorithms")
    async def get_algorithms() -> dict[str, Any]:
        return {"algorithms": algorithm_capabilities(cfg)}

    @app.get("/api/shot-selections")
    async def get_shot_selections() -> dict[str, Any]:
        return {"selections": [
            {"id": "all", "label": "All shots", "available": True},
            {"id": "pots_only", "label": "Potted shots only", "available": True,
             "reason": "Requires readable scores; ambiguous shots can be reviewed"},
            {"id": "pots_and_fouls", "label": "Pots + fouls highlights", "available": True,
             "reason": "Keep confirmed pots, fouls and points increases"},
        ]}

    @app.get("/api/capcut")
    async def capcut_capability() -> dict[str, Any]:
        try:
            installation = find_capcut(cfg)
        except CapCutUnavailable as exc:
            return {"available": False, "reason": str(exc)}
        return {"available": True, "version": installation.version}

    def require_algorithm(algorithm: Algorithm) -> None:
        if algorithm is Algorithm.CLASSIC:
            return
        capability = next(
            item for item in algorithm_capabilities(cfg) if item["id"] == algorithm.value
        )
        if not capability.get("available", False):
            raise HTTPException(409, capability.get("reason") or "DL Algo is not ready")

    @app.post("/api/jobs")
    async def create_job(body: AnalyzeBody) -> dict[str, Any]:
        if not body.source_path and not body.job_id:
            raise HTTPException(400, "source_path required")
        existing_meta = None
        if body.job_id:
            try:
                existing_meta = store.get_meta(body.job_id)
            except FileNotFoundError:
                pass
        source = body.source_path or (existing_meta or {}).get("source_path")
        if not source or not Path(source).is_file():
            raise HTTPException(400, f"source not found: {source}")
        mode = EditMode.from_string(body.mode)
        algorithm = body.algorithm or Algorithm.parse((existing_meta or {}).get("algorithm"))
        selection = body.selection or ShotSelection((existing_meta or {}).get("selection", "all"))
        if selection is not ShotSelection.ALL and algorithm is not Algorithm.CLASSIC:
            raise HTTPException(400, "Potted shots only uses the current algorithm")
        require_algorithm(algorithm)
        switched_algorithm = (
            existing_meta is not None
            and algorithm != Algorithm.parse(existing_meta.get("algorithm"))
        )
        switched_selection = existing_meta is not None and selection.value != existing_meta.get("selection", "all")
        separate_job = switched_algorithm or switched_selection
        if existing_meta and body.submission_token and existing_meta.get("submission_token") == body.submission_token:
            if (Path(source).resolve() != Path(existing_meta["source_path"]).resolve()
                    or separate_job or mode.value != existing_meta.get("mode", "strict")
                    or body.auto_export != existing_meta.get("auto_export", True)):
                raise HTTPException(409, "This submission token already belongs to a different request")
            if (existing_meta["status"] != "pending" or _job_busy(store, body.job_id)
                    or (store.job_dir(body.job_id) / "queued.json").exists()):
                return {"job_id": body.job_id, "status": existing_meta["status"]}
        if body.job_id and not separate_job:
            job_id = body.job_id
            job_dir = store.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            try:
                store.get_meta(job_id)
            except FileNotFoundError:
                store._write_meta(
                    job_id,
                    {
                        "job_id": job_id,
                        "source_path": str(Path(source).resolve()),
                        "mode": mode.value,
                        "algorithm": algorithm.value,
                        "selection": selection.value,
                        "created_at": time.time(),
                        "source_fingerprint": store.source_fingerprint(source),
                        "status": JobStatus.PENDING.value,
                        "progress": 0.0,
                        "stage": "pending",
                        "message": "Created",
                    },
                )
        else:
            job_id = store.create(source, mode=mode.value, algorithm=algorithm, selection=selection)
        if _job_busy(store, job_id):
            raise HTTPException(409, f"Job {job_id} is already running")
        meta = store.get_meta(job_id)
        if separate_job:
            meta["derived_from_job_id"] = body.job_id
        meta["auto_export"] = body.auto_export
        if body.submission_token:
            meta["submission_token"] = body.submission_token
        if body.auto_export:
            meta["auto_export_ready"] = False
        store._write_meta(job_id, meta)
        store.update_progress(job_id, float(meta.get("progress", 0)), JobStatus.PENDING, "Queued for processing")
        enqueue(job_id, source, mode, body.resume, auto_export=body.auto_export)
        return {
            "job_id": job_id, "status": "started", "mode": mode.value,
            "algorithm": algorithm.value,
            "selection": selection.value,
            "derived_from_job_id": body.job_id if separate_job else None,
        }

    @app.post("/api/jobs/{job_id}/restart")
    async def restart_job(
        job_id: str,
        body: Optional[RestartBody] = None,
    ) -> dict[str, Any]:
        try:
            meta = store.get_meta(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

        source = meta.get("source_path")
        if not source or not Path(source).is_file():
            raise HTTPException(400, f"Source video not found: {source}")

        mode_str = (body.mode if body and body.mode else None) or meta.get("mode", "strict")
        mode = EditMode.from_string(mode_str)
        algorithm = (body.algorithm if body else None) or Algorithm.parse(meta.get("algorithm"))
        selection = (body.selection if body else None) or ShotSelection(meta.get("selection", "all"))
        if selection is not ShotSelection.ALL and algorithm is not Algorithm.CLASSIC:
            raise HTTPException(400, "Potted shots only uses the current algorithm")
        require_algorithm(algorithm)
        switched_algorithm = algorithm != Algorithm.parse(meta.get("algorithm"))
        separate_job = switched_algorithm or selection.value != meta.get("selection", "all")
        previous_job_id = job_id
        if not separate_job:
            if _job_busy(store, job_id):
                raise HTTPException(409, f"Job {job_id} is already running")
        force = body.force if body is not None else True
        if separate_job and not switched_algorithm and (body is None or "force" not in body.model_fields_set):
            force = False
        auto_export = (
            body.auto_export if body and body.auto_export is not None
            else bool(meta.get("auto_export", True))
        )
        if separate_job:
            job_id = store.create(source, mode=mode.value, algorithm=algorithm, selection=selection)
            meta = store.get_meta(job_id)
            meta["derived_from_job_id"] = previous_job_id
        meta["auto_export"] = auto_export
        if auto_export:
            meta["auto_export_ready"] = False
        store._write_meta(job_id, meta)

        store.update_progress(
            job_id,
            0.01,
            JobStatus.ANALYZING,
            "Starting separate analysis..." if separate_job
            else "Restarting analysis from last valid checkpoint...",
            error=None,
        )
        enqueue(job_id, source, mode, True, force, auto_export)
        return {
            "job_id": job_id,
            "status": "started" if separate_job else "restarted",
            "mode": mode.value,
            "algorithm": algorithm.value,
            "selection": selection.value,
            "derived_from_job_id": previous_job_id if separate_job else None,
            "message": "Starting separate analysis" if separate_job
            else "Restarting analysis from last checkpoint",
        }

    @app.post("/api/jobs/{job_id}/points-highlights")
    async def points_highlights(job_id: str) -> dict[str, Any]:
        try:
            parent = store.get_meta(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if Algorithm.parse(parent.get("algorithm")) is not Algorithm.CLASSIC:
            raise HTTPException(400, "Points highlights uses saved results from the current algorithm")
        if parent.get("status") not in {JobStatus.COMPLETED.value, JobStatus.READY_FOR_REVIEW.value}:
            raise HTTPException(409, "Wait until this video's analysis has finished")
        if _job_busy(store, job_id):
            raise HTTPException(409, "This video is still processing")
        analysis_path = store.job_dir(job_id)/"analysis.json"
        if not analysis_path.is_file():
            analysis_path = store.job_dir(job_id)/"timeline.json"
        if not analysis_path.is_file():
            raise HTTPException(409, "No saved shot detection is available for this video")
        try:
            fingerprint = store.validate_saved_source(job_id)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
        if (parent.get("selection") == ShotSelection.POTS_AND_FOULS.value
                and (store.job_dir(job_id)/"export"/"highlights.mp4").is_file()
                and parent.get("auto_export_ready") is not False and not parent.get("export_needs_update")):
            return {"job_id": job_id, "status": "ready",
                    "download_url": f"/api/jobs/{job_id}/download/highlights"}
        stat = analysis_path.stat()
        identity = [stat.st_size, stat.st_mtime_ns]
        templates = Path(__file__).resolve().parents[2]/"snooker_ai"/"pot_detection"/"assets"/"score_digits.npz"
        key = hashlib.sha256(json.dumps({
            "version": 1, "parent": job_id, "analysis": identity, "source": fingerprint,
            "scores": cfg.get("pot_selection", {}), "export": cfg.get("export", {}),
            "templates": hashlib.sha256(templates.read_bytes()).hexdigest(),
        }, sort_keys=True, default=str).encode()).hexdigest()
        existing = next((meta for meta in store.list_jobs() if meta.get("points_variant_key") == key), None)
        if existing is not None:
            variant_id = existing["job_id"]
            if (existing.get("status") == JobStatus.COMPLETED.value
                    and existing.get("auto_export_ready")
                    and not existing.get("export_needs_update")
                    and (store.job_dir(variant_id)/"export"/"highlights.mp4").is_file()):
                return {"job_id": variant_id, "status": "ready",
                        "download_url": f"/api/jobs/{variant_id}/download/highlights"}
            if _job_busy(store, variant_id):
                return {"job_id": variant_id, "status": "running"}
            meta = existing
        else:
            variant_id = store.create(fingerprint["path"], mode=parent.get("mode", "strict"),
                                      selection=ShotSelection.POTS_AND_FOULS)
            meta = store.get_meta(variant_id)
            meta.update(derived_from_job_id=job_id, reuse_saved_detection=True,
                        parent_analysis_identity=identity, parent_analysis_file=analysis_path.name,
                        points_variant_key=key)
        meta.update(auto_export=True, auto_export_ready=False)
        store._write_meta(variant_id, meta)
        store.update_progress(variant_id, min(.9, float(meta.get("progress", 0))), JobStatus.PENDING,
                              "Creating pots + fouls highlights from saved detection")
        enqueue(variant_id, fingerprint["path"], EditMode.from_string(meta.get("mode", "strict")), True)
        return {"job_id": variant_id, "status": "started", "derived_from_job_id": job_id}

    @app.post("/api/jobs/{job_id}/continue")
    async def continue_job(job_id: str) -> dict[str, Any]:
        try:
            meta = store.get_meta(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if _job_busy(store, job_id):
            raise HTTPException(409, f"Job {job_id} is already running")
        source = meta.get("source_path")
        if not source or not Path(source).is_file():
            raise HTTPException(400, f"Source video not found: {source}")
        algorithm = Algorithm.parse(meta.get("algorithm"))
        require_algorithm(algorithm)
        mode = EditMode.from_string(meta.get("mode", "strict"))
        auto_export = bool(meta.get("auto_export", True))
        meta["auto_export_ready"] = False
        store._write_meta(job_id, meta)
        store.update_progress(job_id, min(.99, float(meta.get("progress", 0))),
                              JobStatus.PENDING, "Continuing from last saved checkpoint", error=None)
        enqueue(job_id, source, mode, True, auto_export=auto_export)
        return {"job_id": job_id, "status": "continued", "message": "Continuing from last saved checkpoint"}

    @app.post("/api/preprocess")
    async def preprocess_video(body: PreprocessBody) -> dict[str, Any]:
        source = Path(body.source_path).resolve()
        uploads_root = store.uploads.resolve()
        if not source.is_file() or not source.is_relative_to(uploads_root):
            raise HTTPException(400, "Preprocessing is limited to uploaded videos")

        safe_stem = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in source.stem
        )[:80]
        output = uploads_root / f"{safe_stem}_cleaned_{uuid.uuid4().hex[:8]}.mp4"
        ranges = [KeepRange(item.start, item.end) for item in body.ranges]
        try:
            rendered = await asyncio.to_thread(
                VideoPreprocessor(cfg).render,
                source,
                ranges,
                output,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Pre-analysis edit failed for %s", source)
            raise HTTPException(500, f"Could not create cleaned video: {exc}") from exc
        metadata = probe_video(rendered)
        return {
            "path": str(rendered),
            "duration": metadata.duration,
            "original_path": str(source),
            "kept_sections": len(ranges),
        }

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        return [with_download(meta) for meta in store.list_jobs()]

    def with_download(meta: dict[str, Any]) -> dict[str, Any]:
        job_id = meta["job_id"]
        running = _job_busy(store, job_id)
        queue_state = queue.job_state(job_id)
        if meta.get("status") in _ACTIVE_STATUSES and not running:
            store.update_progress(
                job_id, min(.99, float(meta.get("progress", 0))), JobStatus.INTERRUPTED,
                "Processing was interrupted. Use Continue to resume saved work.",
                error="The processing worker stopped before completion.",
                shots_detected=int(meta.get("shots_detected", 0)),
            )
            meta = store.get_meta(job_id)
        checkpoint = {}
        try:
            checkpoint = json.loads((store.job_dir(job_id) / "checkpoint.json").read_text("utf-8"))
        except (OSError, ValueError):
            pass
        available = (
            meta.get("status") == JobStatus.COMPLETED.value
            and meta.get("auto_export_ready") is not False
            and not meta.get("export_needs_update")
            and (store.job_dir(job_id) / "export" / "highlights.mp4").is_file()
        )
        timeline_revision = None
        for name in ("timeline.json", "analysis.json"):
            try:
                snapshot = (store.job_dir(job_id) / name).stat()
                timeline_revision = f"{snapshot.st_mtime_ns}:{snapshot.st_size}"
                break
            except OSError:
                continue
        saved = timeline_revision is not None
        can_points = (not running and saved and Algorithm.parse(meta.get("algorithm")) is Algorithm.CLASSIC
                      and meta.get("status") in {JobStatus.COMPLETED.value, JobStatus.READY_FOR_REVIEW.value})
        return {**meta, **queue_state, "running": running, "can_continue": not running,
                "has_saved_detection": saved, "can_create_points": can_points,
                "timeline_revision": timeline_revision,
                "checkpoint": checkpoint,
                "download_url": f"/api/jobs/{job_id}/download/highlights" if available else None}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return with_download(store.get_meta(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/progress")
    async def get_progress(job_id: str) -> dict[str, Any]:
        try:
            return with_download(store.get_meta(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/shots")
    async def get_shots(job_id: str) -> dict[str, Any]:
        # The review page only needs the compact timeline.  Loading the full
        # analysis (137 MB for a two-hour match) made the UI show a misleading
        # temporary "0 shots" state while thousands of feature rows parsed.
        timeline_path = store.job_dir(job_id) / "timeline.json"
        if timeline_path.exists():
            try:
                timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
                return {
                    "job_id": job_id,
                    "mode": timeline.get("mode", "strict"),
                    "selection": timeline.get("selection", "all"),
                    "pot_selection_summary": timeline.get("pot_selection_summary", {}),
                    "original_duration": timeline.get("original_duration", 0.0),
                    "edited_duration": timeline.get("edited_duration", 0.0),
                    "shots": timeline.get("shots", []),
                }
            except (OSError, ValueError, TypeError):
                logger.warning("Could not load compact timeline for %s", job_id)
        try:
            result = store.load_analysis(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "job_id": job_id,
            "mode": result.mode.value,
            "selection": result.selection.value,
            "pot_selection_summary": result.pot_selection_summary,
            "original_duration": result.original_duration,
            "edited_duration": result.edited_duration,
            "shots": [s.model_dump() for s in result.shots],
        }

    @app.get("/api/jobs/{job_id}/timeline")
    async def get_timeline(job_id: str) -> Any:
        path = store.job_dir(job_id) / "timeline.json"
        if not path.exists():
            raise HTTPException(404, "timeline not found")
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    def edit_review(job_id: str, operation: Callable[[], Any]) -> Any:
        # Another tab or server process must not change the snapshot being
        # analyzed/exported. Use the same OS lease as those workers.
        store.get_meta(job_id)
        lock = _locks.setdefault(job_id, threading.Lock())
        if _job_busy(store, job_id) or not lock.acquire(blocking=False):
            raise HTTPException(409, "This video is processing. Wait for it to finish before editing shots.")
        lease = JobLease(store.job_dir(job_id))
        if not lease.acquire():
            lock.release()
            raise HTTPException(409, "This video is processing. Wait for it to finish before editing shots.")
        try:
            return operation()
        finally:
            lease.release()
            lock.release()

    @app.patch("/api/jobs/{job_id}/shots/{shot_id}")
    async def patch_shot(job_id: str, shot_id: int, body: ShotUpdate) -> dict[str, Any]:
        try:
            s = edit_review(job_id, lambda: store.update_shot(job_id, shot_id, body))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return s.model_dump()

    @app.delete("/api/jobs/{job_id}/shots/{shot_id}")
    async def remove_shot(job_id: str, shot_id: int) -> dict[str, str]:
        try:
            edit_review(job_id, lambda: store.delete_shot(job_id, shot_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"status": "deleted", "shot_id": str(shot_id)}

    @app.post("/api/jobs/{job_id}/shots")
    async def add_shot(job_id: str, body: AddShotBody) -> dict[str, Any]:
        shot = ShotRecord(
            shot_id=0,
            preparation_start=body.preparation_start or body.clip_start,
            cue_strike=body.cue_strike,
            ball_motion_start=body.cue_strike,
            ball_motion_end=body.ball_motion_end or body.clip_end,
            clip_start=body.clip_start,
            clip_end=body.clip_end,
            shot_confidence=body.shot_confidence,
            start_confidence=1.0,
            end_confidence=1.0,
            manual_review_required=False,
            user_modified=True,
            included=True,
        )
        try:
            s = edit_review(job_id, lambda: store.add_shot(job_id, shot))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return s.model_dump()

    @app.post("/api/jobs/{job_id}/shots/merge")
    async def merge_shots(job_id: str, body: MergeShotsBody) -> dict[str, Any]:
        try:
            s = edit_review(job_id, lambda: store.merge_shots(job_id, body.shot_ids))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.model_dump()

    @app.post("/api/jobs/{job_id}/shots/{shot_id}/split")
    async def split_shot(job_id: str, shot_id: int, body: SplitShotBody) -> dict[str, Any]:
        try:
            shots = edit_review(job_id, lambda: store.split_shot(job_id, shot_id, body.at_time))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"shots": [s.model_dump() for s in shots]}

    @app.post("/api/jobs/{job_id}/export")
    async def export_job(job_id: str, body: ExportBody) -> dict[str, Any]:
        if not body.export_clips and not body.export_joined:
            raise HTTPException(400, "Select combined video, individual clips, or both")
        lock = _locks.setdefault(job_id, threading.Lock())
        if JobLease.is_running(store.job_dir(job_id), ".queue.lock"):
            raise HTTPException(409, f"Job {job_id} is queued or processing")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, f"Job {job_id} is already processing")
        lease = JobLease(store.job_dir(job_id))
        if JobLease.is_running(store.job_dir(job_id), ".queue.lock") or not lease.acquire():
            lock.release()
            raise HTTPException(409, f"Job {job_id} is already processing")
        try:
            try:
                result = store.load_analysis(job_id)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            if body.mode:
                requested_mode = EditMode.from_string(body.mode)
                # The editor sends its current mode with every export. Rebuilding
                # in that case used to discard persisted include/exclude edits.
                if requested_mode != result.mode:
                    analyzer = create_analyzer(
                        cfg, store.job_dir(job_id), store.get_meta(job_id).get("algorithm"),
                    )
                    result = analyzer.resegment(result, requested_mode)
            out_dir = store.job_dir(job_id) / "export"
            req = ExportRequest(
                mode=result.mode,
                output_path=body.output_name,
                export_clips=body.export_clips,
                export_joined=body.export_joined,
                include_replays=body.include_replays,
                accurate=body.accurate,
                min_confidence=body.min_confidence,
                min_importance=body.min_importance,
            )
            n_shots = len(result.shots) if result and result.shots is not None else 0
            store.update_progress(
                job_id,
                0.1,
                JobStatus.EXPORTING,
                "Exporting",
                shots_detected=n_shots,
            )
            try:
                er = await asyncio.to_thread(Exporter(cfg).export, result, out_dir, req)
            except Exception as exc:
                current = store.get_meta(job_id)
                store.update_progress(job_id, current.get("progress", 0), JobStatus.FAILED,
                                      "Export stopped. Use Continue to retry saved work.", error=str(exc))
                raise HTTPException(500, str(exc)) from exc
            clip_paths = er.clip_paths if er and er.clip_paths is not None else []
            meta = store.get_meta(job_id)
            if body.export_joined:
                meta["auto_export_ready"] = bool(er.joined_path)
            if er.joined_path:
                meta["export_needs_update"] = False
            store._write_meta(job_id, meta)
            store.update_progress(
                job_id,
                1.0,
                JobStatus.COMPLETED,
                "Export complete",
                shots_detected=n_shots,
            )
            return {
                "joined": str(er.joined_path) if er.joined_path else None,
                "download_url": (
                    f"/api/jobs/{job_id}/download/highlights" if er.joined_path else None
                ),
                "clips": [str(p) for p in clip_paths],
                "clips_dir": str(out_dir / "clips"),
                "clip_count": len(clip_paths),
                "csv": str(er.csv_path) if er.csv_path else None,
                "edl": str(er.edl_path) if er.edl_path else None,
                "metadata": str(er.metadata_path) if er.metadata_path else None,
                "training_labels": str(er.training_labels_path) if er.training_labels_path else None,
            }
        finally:
            lease.release()
            lock.release()

    @app.post("/api/jobs/{job_id}/open-capcut")
    async def open_capcut(job_id: str, request: Request) -> dict[str, Any]:
        # A cross-site browser request must not create local desktop projects.
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Open in CapCut must be requested from Snooker AI.")
        lock = _locks.setdefault(job_id, threading.Lock())
        if _job_busy(store, job_id) or not lock.acquire(blocking=False):
            raise HTTPException(409, f"Job {job_id} is already processing")
        try:
            try:
                result = await asyncio.to_thread(store.load_analysis, job_id)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            try:
                installation = find_capcut(cfg)
                project = await asyncio.to_thread(create_capcut_project, result, cfg, installation)
            except CapCutUnavailable as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(500, f"Could not create CapCut project: {exc}") from exc
            out_dir = store.job_dir(job_id) / "export"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "capcut_project.json").write_text(
                json.dumps(project, indent=2), encoding="utf-8",
            )
            launched, launch_error = True, None
            try:
                await asyncio.to_thread(launch_capcut, installation)
            except OSError as exc:
                launched, launch_error = False, str(exc)
            message = (
                f"{project['shot_count']} editable shots ready. "
                f"Open \u201c{project['project_name']}\u201d on CapCut\u2019s Home screen."
            )
            if not launched:
                message = "Project created; start CapCut manually. " + message
            return {**project, "launched": launched, "launch_error": launch_error, "message": message}
        finally:
            lock.release()

    @app.post("/api/jobs/{job_id}/open-clips-folder")
    async def open_clips_folder(job_id: str) -> dict[str, Any]:
        """Open the local exported-clips directory in the system file manager."""

        job_dir = store.job_dir(job_id)
        if not job_dir.is_dir():
            raise HTTPException(404, f"Job not found: {job_id}")
        clips_dir = job_dir / "export" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        try:
            focused = _open_folder(clips_dir)
        except OSError as exc:
            raise HTTPException(500, f"Could not open clips folder: {exc}") from exc
        clip_count = sum(1 for path in clips_dir.glob("shot_*.mp4") if path.is_file())
        return {
            "folder": str(clips_dir.resolve()),
            "clip_count": clip_count,
            "opened": True,
            "focused": focused,
        }

    @app.post("/api/jobs/{job_id}/open-export-folder")
    async def open_export_folder(job_id: str) -> dict[str, Any]:
        """Open the export root so combined output is visible to the user."""

        job_dir = store.job_dir(job_id)
        if not job_dir.is_dir():
            raise HTTPException(404, f"Job not found: {job_id}")
        export_dir = job_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            focused = _open_folder(export_dir)
        except OSError as exc:
            raise HTTPException(500, f"Could not open export folder: {exc}") from exc
        clips_dir = export_dir / "clips"
        clip_count = (
            sum(1 for path in clips_dir.glob("shot_*.mp4") if path.is_file())
            if clips_dir.is_dir()
            else 0
        )
        combined = export_dir / "highlights.mp4"
        return {
            "folder": str(export_dir.resolve()),
            "clip_count": clip_count,
            "combined_exists": combined.is_file(),
            "combined_video": str(combined.resolve()) if combined.is_file() else None,
            "opened": True,
            "focused": focused,
        }

    @app.get("/api/jobs/{job_id}/download/{kind}")
    async def download(job_id: str, kind: str) -> FileResponse:
        base = store.job_dir(job_id)
        mapping = {
            "analysis": base / "analysis.json",
            "timeline": base / "timeline.json",
            "corrections": base / "corrections.json",
            "highlights": base / "export" / "highlights.mp4",
            "csv": base / "export" / "shots.csv",
            "edl": base / "export" / "timeline.edl",
            "training": base / "export" / "training_labels.json",
        }
        path = mapping.get(kind)
        if kind == "highlights":
            try:
                meta = store.get_meta(job_id)
                if meta.get("export_needs_update"):
                    raise HTTPException(409, "Review changes are saved. Export video to update your highlights.")
                if meta.get("auto_export_ready") is False:
                    raise HTTPException(404, "No finished video for the current analysis")
            except FileNotFoundError:
                pass
        if not path or not path.exists():
            raise HTTPException(404, f"Artifact '{kind}' not found")
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/video")
    async def stream_source(job_id: str) -> FileResponse:
        meta = store.get_meta(job_id)
        source_path = Path(meta["source_path"])
        path: Path | None = None

        # Review against the timestamp-aligned, fast-start proxy.  It is much
        # smaller than the camera original and is encoded with frequent
        # keyframes specifically so selecting a distant shot is immediate.
        try:
            result = store.load_analysis(job_id)
            if result.proxy_path:
                proxy_path = Path(result.proxy_path)
                if proxy_path.exists():
                    path = proxy_path
            if path is None and Path(result.source_path).exists():
                source_path = Path(result.source_path)
        except FileNotFoundError:
            pass

        # Convention fallback supports older analysis files that did not save
        # proxy_path.  The original remains the final compatibility fallback.
        conventional_proxy = store.job_dir(job_id) / "proxy" / "proxy.mp4"
        if path is None and conventional_proxy.exists():
            path = conventional_proxy
        if path is None and source_path.exists():
            path = source_path
        if path is None or not path.exists():
            raise HTTPException(404, "source video missing")
        return VideoFileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/review/{job_id}", response_class=HTMLResponse)
    async def review_page(job_id: str) -> HTMLResponse:
        page = web_dir / "review.html"
        if not page.exists():
            raise HTTPException(404, "review UI missing")
        html = page.read_text(encoding="utf-8").replace("{{JOB_ID}}", job_id)
        return HTMLResponse(html)

    return app


# Module-level app for uvicorn apps.api.main:app
app = create_app()
