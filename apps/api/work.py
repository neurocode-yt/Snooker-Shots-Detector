"""One compact activity feed for match processing, exports and Shorts batches."""

import asyncio

from fastapi import HTTPException, Request

from snooker_ai.types import JobStatus


def install_work_routes(app, store, queue, shorts, with_download, job_busy, active_statuses):
    @app.get("/api/work")
    async def current_work():
        def collect():
            items = []
            for meta in store.list_jobs():
                if meta.get("algorithm") == "dl_algo":
                    continue
                if meta.get("status") not in active_statuses | {
                    JobStatus.PAUSED.value,
                    JobStatus.INTERRUPTED.value,
                }:
                    continue
                job = with_download(meta)
                if job["status"] not in active_statuses | {
                    JobStatus.PAUSED.value,
                    JobStatus.INTERRUPTED.value,
                }:
                    continue
                paused = job["status"] == JobStatus.PAUSED.value
                base = f"/api/jobs/{job['job_id']}"
                items.append(
                    {
                        "task_id": f"match-{job['job_id']}",
                        "kind": "match",
                        "job_id": job["job_id"],
                        "has_saved_detection": job["has_saved_detection"],
                        "source_path": job["source_path"],
                        "status": "pausing" if job["pause_requested"] else job["status"],
                        "queued": job.get("queued", False),
                        "queue_position": job.get("queue_position"),
                        "message": job.get("message", ""),
                        "progress": job.get("progress", 0),
                        "can_pause": job["status"] in active_statuses
                        and not job["pause_requested"],
                        "can_resume": not job["running"]
                        and (paused or job["status"] == "interrupted"),
                        "pause_url": base + "/pause",
                        "resume_url": base + "/continue",
                    }
                )
            for batch in shorts.work_items():
                meta = store.get_meta(batch["job_id"])
                base = f"/api/jobs/{batch['job_id']}/shorts/{batch['batch_id']}"
                items.append(
                    {
                        "task_id": f"shorts-{batch['job_id']}-{batch['batch_id']}",
                        "kind": "shorts",
                        "job_id": batch["job_id"],
                        "batch_id": batch["batch_id"],
                        "source_path": meta["source_path"],
                        "status": "pausing" if batch.get("pause_requested") else batch["status"],
                        "message": batch["message"],
                        "completed": batch["completed"],
                        "total": batch["total"],
                        "progress": batch["completed"] / max(1, batch["total"]),
                        "can_pause": batch["status"] in {"queued", "rendering"}
                        and not batch.get("pause_requested"),
                        "can_resume": batch["status"] in {"paused", "interrupted"},
                        "pause_url": base + "/pause",
                        "resume_url": base + "/resume",
                    }
                )
            return {"items": items}

        return await asyncio.to_thread(collect)

    @app.post("/api/jobs/{job_id}/pause")
    async def pause(job_id: str, request: Request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Pause work from Snooker AI")

        def stop():
            try:
                meta = store.get_meta(job_id)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            if meta["status"] == JobStatus.PAUSED.value:
                return with_download(meta)
            if meta["status"] not in active_statuses or not job_busy(store, job_id):
                raise HTTPException(409, "This video is not processing")
            queue.pause(job_id)
            return with_download(store.get_meta(job_id))

        return await asyncio.to_thread(stop)
