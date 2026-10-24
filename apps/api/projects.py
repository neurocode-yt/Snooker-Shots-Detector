"""Library project naming and confirmed deletion of complete source families."""

import asyncio
import threading
from contextlib import ExitStack

from fastapi import HTTPException, Request

from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.projects import ProjectConflict, ProjectLibrary
from snooker_ai.rendering.capcut import capcut_index_lease
from pathlib import Path
from snooker_ai.utils.models import BaseModel, Field


class ProjectNameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class ProjectSelectionBody(BaseModel):
    job_ids: list[str] = Field(..., min_length=1, max_length=500)
    token: str | None = None


def install_project_routes(app, store, locks):
    library = ProjectLibrary(store)

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Manage projects from Snooker AI")

    async def call(operation):
        def guarded():
            lease = JobLease(store.root / ".library", ".projects.lock")
            if not lease.acquire():
                raise ProjectConflict("Another project action is running. Try again shortly.")
            try:
                return operation()
            finally:
                lease.release()

        try:
            return await asyncio.to_thread(guarded)
        except ProjectConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                409, "Could not access a project resource. Close apps using it and retry."
            ) from exc

    @app.patch("/api/projects/{job_id}")
    async def rename_project(job_id: str, body: ProjectNameBody, request: Request):
        same_origin(request)
        return await call(lambda: library.rename(job_id, body.name))

    @app.post("/api/projects/delete-preview")
    async def preview_delete(body: ProjectSelectionBody, request: Request):
        same_origin(request)

        def preview():
            plan = library.plan(body.job_ids)
            return {key: value for key, value in plan.items() if key != "indexes"}

        return await call(preview)

    @app.post("/api/projects/delete")
    async def delete_projects(body: ProjectSelectionBody, request: Request):
        same_origin(request)
        if not body.token:
            raise HTTPException(400, "Check the deletion details before confirming")

        def delete():
            plan = library.plan(body.job_ids)
            if plan["issues"]:
                raise ProjectConflict(plan["issues"][0])
            with ExitStack() as stack:
                leases = []
                for index in sorted({p["index"] for p in plan["capcut"]}):
                    lease = capcut_index_lease(Path(index).parent)
                    if not lease.acquire():
                        raise ProjectConflict(
                            "CapCut's project list is being updated. Try again shortly."
                        )
                    stack.callback(lease.release)
                for job_id in plan["job_ids"]:
                    lock = locks.setdefault(job_id, threading.Lock())
                    if not lock.acquire(blocking=False):
                        raise ProjectConflict(
                            "A selected project is in use. Retry after it finishes."
                        )
                    stack.callback(lock.release)
                    directory = store.job_dir(job_id)
                    for parent, name in [
                        (directory, ".queue.lock"),
                        (directory, ".processing.lock"),
                        (directory, ".capcut.lock"),
                        (directory / "shorts", ".shorts.lock"),
                    ]:
                        lease = JobLease(parent, name)
                        if not lease.acquire():
                            raise ProjectConflict(
                                "Pause active processing and Shorts before deleting."
                            )
                        leases.append(lease)
                        stack.callback(lease.release)
                    marker = store.root / ".deleting" / job_id
                    marker.parent.mkdir(exist_ok=True)
                    marker.touch(exist_ok=False)
                    stack.callback(marker.unlink, missing_ok=True)
                # Prevent new workers while releasing Windows file handles and moving folders.
                for lease in leases:
                    lease.release()
                return library.delete(plan, body.token)

        return await call(delete)

    @app.post("/api/projects/cleanup")
    async def retry_cleanup(request: Request):
        same_origin(request)
        return await call(lambda: {"cleanup_pending": library.cleanup()})

    @app.get("/api/projects/cleanup")
    async def cleanup_status():
        return {
            "cleanup_pending": len(list((store.root / ".deletions").glob("*.json")))
            + len(list((store.root / ".deleting").glob("*")))
        }

    return library
