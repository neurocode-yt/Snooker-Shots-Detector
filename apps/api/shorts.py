"""Saved-shot Shorts endpoints, independent of detection and combined exports."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from snooker_ai.utils.models import BaseModel

from snooker_ai.rendering.shorts import PRESETS, ShortsSettings, preview_png, selection_key


class PreviewBody(BaseModel):
    settings: ShortsSettings
    shot_id: int
    batch_id: str | None = None


def public_batch(batch):
    if batch is None:
        return None
    return {key: value for key, value in batch.items() if key not in {"source_fingerprint"}} | {
        "clips": [
            {key: value for key, value in clip.items() if key != "file_identity"}
            for clip in batch["clips"]
        ]
    }


def install_shorts_routes(app, manager, edit_review, open_folder):
    async def call(operation):
        try:
            return await asyncio.to_thread(operation)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def same_origin(request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Request Shorts from Snooker AI")

    @app.get("/api/jobs/{job_id}/shorts")
    async def setup(job_id: str, batch_id: str | None = None):
        def load():
            result, shots = (
                manager.load_batch_source(job_id, batch_id)
                if batch_id
                else manager.load_source(job_id)
            )
            batch = manager.read(job_id, batch_id) if batch_id else manager.latest(job_id)
            return {
                "presets": PRESETS,
                "defaults": ShortsSettings(player_names="Player name").model_dump(),
                "shots": [
                    {
                        "shot_id": s.shot_id,
                        "start": s.clip_start,
                        "end": s.clip_end,
                        "needs_review": s.manual_review_required,
                    }
                    for s in shots
                ],
                "batch": public_batch(batch),
                "selection_changed": bool(
                    batch and batch["selection_key"] != selection_key(result, shots)
                ),
            }

        return await call(load)

    @app.post("/api/jobs/{job_id}/shorts/preview")
    async def preview(job_id: str, body: PreviewBody):
        def render():
            result, shots = (
                manager.load_batch_source(job_id, body.batch_id)
                if body.batch_id
                else manager.load_source(job_id)
            )
            shot = next((s for s in shots if s.shot_id == body.shot_id), None)
            if shot is None:
                raise FileNotFoundError("Included live shot not found")
            return preview_png(result, body.settings, shot)

        return Response(
            await call(render), media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/jobs/{job_id}/shorts", status_code=202)
    async def start(job_id: str, body: ShortsSettings, request: Request):
        same_origin(request)
        return public_batch(
            await call(lambda: edit_review(job_id, lambda: manager.submit(job_id, body)))
        )

    @app.get("/api/jobs/{job_id}/shorts/{batch_id}")
    async def status(job_id: str, batch_id: str):
        return public_batch(await call(lambda: manager.read(job_id, batch_id)))

    @app.post("/api/jobs/{job_id}/shorts/{batch_id}/pause")
    async def pause(job_id: str, batch_id: str, request: Request):
        same_origin(request)
        return public_batch(await call(lambda: manager.pause(job_id, batch_id)))

    @app.post("/api/jobs/{job_id}/shorts/{batch_id}/resume", status_code=202)
    async def resume(job_id: str, batch_id: str, request: Request):
        same_origin(request)
        return public_batch(await call(lambda: manager.resume(job_id, batch_id)))

    @app.get("/api/jobs/{job_id}/shorts/{batch_id}/zip")
    async def download_zip(job_id: str, batch_id: str):
        batch = await call(lambda: manager.read(job_id, batch_id))
        if batch["status"] != "completed":
            raise HTTPException(409, "Wait until this Shorts batch has finished")
        path = manager.directory(job_id, batch_id) / "shorts.zip"
        if not path.is_file():
            raise HTTPException(404, "ZIP is missing. Create this batch again to rebuild it.")
        return FileResponse(
            path, filename=f"snooker-shorts-{batch_id}.zip", media_type="application/zip"
        )

    @app.get("/api/jobs/{job_id}/shorts/{batch_id}/clips/{shot_id}")
    async def download_clip(job_id: str, batch_id: str, shot_id: int):
        batch = await call(lambda: manager.read(job_id, batch_id))
        clip = next((c for c in batch["clips"] if c["shot_id"] == shot_id), None)
        if clip is None:
            raise HTTPException(404, "This Short is not ready yet")
        directory = manager.directory(job_id, batch_id).resolve()
        path = (directory / clip["filename"]).resolve()
        if path.parent != directory or not path.is_file():
            raise HTTPException(404, "Short not found")
        return FileResponse(path, filename=path.name, media_type="video/mp4")

    @app.post("/api/jobs/{job_id}/shorts/{batch_id}/open-folder")
    async def show_folder(job_id: str, batch_id: str, request: Request):
        same_origin(request)
        await call(lambda: manager.read(job_id, batch_id))
        path = manager.directory(job_id, batch_id)
        await call(lambda: open_folder(Path(path)))
        return {"message": "Opened the Shorts folder"}
