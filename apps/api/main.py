"""REST API and review UI for snooker-ai."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from snooker_ai.config import Config, load_config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.rendering.preprocessor import KeepRange, VideoPreprocessor
from snooker_ai.types import (
    EditMode,
    ExportRequest,
    JobStatus,
    ShotRecord,
    ShotUpdate,
)
from snooker_ai.utils.logging import get_logger, setup_logging

logger = get_logger("api")

# In-process job locks
_locks: dict[str, threading.Lock] = {}


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
    resume: bool = True


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


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or load_config()
    setup_logging(str(cfg.get("log_level", "INFO")))
    cfg.ensure_dirs()
    store = JobStore(cfg)

    app = FastAPI(
        title="Snooker AI API",
        version="0.1.0",
        description="Automatic snooker shot detection and video editing",
    )
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

    def _run_analysis(job_id: str, source: str, mode: EditMode, resume: bool) -> None:
        lock = _locks.setdefault(job_id, threading.Lock())
        if not lock.acquire(blocking=False):
            logger.warning("Job %s already running", job_id)
            return
        try:
            analyzer = Analyzer(cfg, store.job_dir(job_id))

            def prog(p: float, stage: str, msg: str) -> None:
                store.update_progress(job_id, p, stage, msg)

            result = analyzer.analyze(source, job_id, mode=mode, progress=prog, resume=resume)
            store.update_progress(
                job_id,
                1.0,
                JobStatus.READY_FOR_REVIEW,
                f"Detected {len(result.shots)} shots",
                shots_detected=len(result.shots),
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            store.update_progress(job_id, 0.0, JobStatus.FAILED, str(exc), error=str(exc))
        finally:
            lock.release()

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

    @app.post("/api/jobs")
    async def create_job(body: AnalyzeBody, background: BackgroundTasks) -> dict[str, Any]:
        if not body.source_path and not body.job_id:
            raise HTTPException(400, "source_path required")
        source = body.source_path
        if not source or not Path(source).is_file():
            raise HTTPException(400, f"source not found: {source}")
        mode = EditMode.from_string(body.mode)
        if body.job_id:
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
                        "status": JobStatus.PENDING.value,
                        "progress": 0.0,
                        "stage": "pending",
                        "message": "Created",
                    },
                )
        else:
            job_id = store.create(source, mode=mode.value)
        background.add_task(_run_analysis, job_id, source, mode, body.resume)
        return {"job_id": job_id, "status": "started", "mode": mode.value}

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
        return store.list_jobs()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get_meta(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/progress")
    async def get_progress(job_id: str) -> dict[str, Any]:
        try:
            return store.get_meta(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/shots")
    async def get_shots(job_id: str) -> dict[str, Any]:
        try:
            result = store.load_analysis(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "job_id": job_id,
            "mode": result.mode.value,
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

    @app.patch("/api/jobs/{job_id}/shots/{shot_id}")
    async def patch_shot(job_id: str, shot_id: int, body: ShotUpdate) -> dict[str, Any]:
        try:
            s = store.update_shot(job_id, shot_id, body)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return s.model_dump()

    @app.delete("/api/jobs/{job_id}/shots/{shot_id}")
    async def remove_shot(job_id: str, shot_id: int) -> dict[str, str]:
        try:
            store.delete_shot(job_id, shot_id)
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
            s = store.add_shot(job_id, shot)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return s.model_dump()

    @app.post("/api/jobs/{job_id}/shots/merge")
    async def merge_shots(job_id: str, body: MergeShotsBody) -> dict[str, Any]:
        try:
            s = store.merge_shots(job_id, body.shot_ids)
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
            shots = store.split_shot(job_id, shot_id, body.at_time)
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
        try:
            result = store.load_analysis(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if body.mode:
            requested_mode = EditMode.from_string(body.mode)
            # The editor sends its current mode with every export. Rebuilding
            # in that case used to discard persisted include/exclude edits.
            if requested_mode != result.mode:
                analyzer = Analyzer(cfg, store.job_dir(job_id))
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
        store.update_progress(
            job_id,
            0.1,
            JobStatus.EXPORTING,
            "Exporting",
            shots_detected=len(result.shots),
        )
        try:
            er = Exporter(cfg).export(result, out_dir, req)
        except Exception as exc:
            store.update_progress(job_id, 0.0, JobStatus.FAILED, str(exc), error=str(exc))
            raise HTTPException(500, str(exc)) from exc
        store.update_progress(
            job_id,
            1.0,
            JobStatus.COMPLETED,
            "Export complete",
            shots_detected=len(result.shots),
        )
        return {
            "joined": str(er.joined_path) if er.joined_path else None,
            "download_url": (
                f"/api/jobs/{job_id}/download/highlights" if er.joined_path else None
            ),
            "clips": [str(p) for p in er.clip_paths],
            "clips_dir": str(out_dir / "clips"),
            "clip_count": len(er.clip_paths),
            "csv": str(er.csv_path) if er.csv_path else None,
            "edl": str(er.edl_path) if er.edl_path else None,
            "metadata": str(er.metadata_path) if er.metadata_path else None,
            "training_labels": str(er.training_labels_path) if er.training_labels_path else None,
        }

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
