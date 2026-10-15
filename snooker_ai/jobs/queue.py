"""Bounded match workers with durable waiting requests and OS-owned slots."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Callable

from snooker_ai.config import Config
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.utils.work_control import WorkControl
from snooker_ai.types import EditMode, JobStatus
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.persistence import atomic_text

logger = get_logger("queue")


class MatchQueue:
    """Queue entire analyses, including automatic export, without holding HTTP requests.

    Waiting requests survive a restart. Once a match starts its request is removed:
    an interrupted active match uses the existing Continue/checkpoint workflow.
    Queue leases prevent duplicate ownership across server processes; worker-slot
    leases share the concurrency limit when servers use the same jobs directory.
    """

    def __init__(self, config: Config, runner: Callable):
        self.config = config
        self.store = JobStore(config)
        self.max_workers = int(config.get("jobs.max_workers", 2))
        if not 1 <= self.max_workers <= 4:
            raise ValueError("jobs.max_workers must be between 1 and 4")
        self.runner = runner
        self._condition = threading.Condition()
        self._waiting: deque[tuple[dict, JobLease]] = deque()
        self._active: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._closed = False
        self._started = False

    def processing_order(self) -> str:
        """Shared, durable choice: sequential queue or configured parallel capacity."""
        try:
            order = json.loads((self.store.root / "queue-settings.json").read_text("utf-8"))["processing_order"]
            if order in {"sequential", "parallel"}:
                return order
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return "parallel"

    def set_processing_order(self, order: str) -> None:
        if order not in {"sequential", "parallel"}:
            raise ValueError("Processing order must be sequential or parallel")
        with self._condition:
            atomic_text(self.store.root / "queue-settings.json", json.dumps({"processing_order": order}))
            self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._started or self._closed:
                return
            self._started = True
        requests = []
        for path in self.store.root.glob("*/queued.json"):
            try:
                request = json.loads(path.read_text("utf-8"))
                if not isinstance(request, dict) or request.get("job_id") != path.parent.name:
                    raise ValueError("Waiting request does not match its job directory")
                requests.append((path.stat().st_mtime_ns, path, request))
            except FileNotFoundError:
                continue  # another server has already started this request
            except (OSError, ValueError):
                logger.exception("Could not read waiting match %s", path)
        for _, path, request in sorted(requests, key=lambda item: item[0]):
            try:
                # Never automatically restart a failed/finished job from stale queue data.
                if self.store.get_meta(request["job_id"])["status"] != JobStatus.PENDING.value:
                    continue
                self.submit(request["job_id"], request["source"], EditMode(request["mode"]),
                            request["resume"], request["force"], request["auto_export"], recover=True)
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Could not recover waiting match %s", path)
        with self._condition:
            if self._closed:
                return
            for index in range(self.max_workers):
                thread = threading.Thread(target=self._worker, name=f"match-worker-{index + 1}", daemon=True)
                self._threads.append(thread)
                thread.start()

    def submit(self, job_id: str, source: str, mode: EditMode, resume: bool,
               force: bool = False, auto_export: bool = True, *, recover: bool = False) -> bool:
        request = dict(job_id=job_id, source=source, mode=mode.value, resume=resume,
                       force=force, auto_export=auto_export)
        with self._condition:
            if self._closed:
                raise RuntimeError("The processing queue is shutting down")
            lease = JobLease(self.store.job_dir(job_id), ".queue.lock")
            if not lease.acquire():
                return False
            try:
                if JobLease.is_running(self.store.job_dir(job_id)):
                    lease.release()
                    return False
                WorkControl(self.store.job_dir(job_id)).clear()
                if not recover:
                    atomic_text(lease.path.parent / "queued.json", json.dumps(request))
                meta = self.store.get_meta(job_id)
                self.store.update_progress(job_id, float(meta.get("progress", 0)), JobStatus.PENDING,
                                           "Queued; waiting for a processing slot", error=None,
                                           shots_detected=int(meta.get("shots_detected", 0)))
                self._waiting.append((request, lease))
            except BaseException:
                lease.release()
                raise
            self._condition.notify_all()
        return True

    def job_state(self, job_id: str) -> dict:
        with self._condition:
            for position, (request, _) in enumerate(self._waiting, 1):
                if request["job_id"] == job_id:
                    return {"queued": True, "queue_position": position}
            if job_id in self._active:
                return {"queued": False, "queue_position": None}
        # Queued in another server process sharing this directory.
        waiting = ((self.store.job_dir(job_id) / "queued.json").exists()
                   and JobLease.is_running(self.store.job_dir(job_id), ".queue.lock"))
        return {"queued": waiting, "queue_position": None}

    def pause(self, job_id: str) -> None:
        with self._condition:
            WorkControl(self.store.job_dir(job_id)).request()
            for entry in list(self._waiting):
                if entry[0]["job_id"] == job_id:
                    self._waiting.remove(entry)
                    self._mark_paused(job_id)
                    entry[1].release()
                    break
            self._condition.notify_all()

    def _mark_paused(self, job_id: str):
        meta = self.store.get_meta(job_id)
        self.store.update_progress(job_id, float(meta.get("progress", 0)), JobStatus.PAUSED,
                                   "Paused. Saved work is preserved; use Resume to continue.",
                                   shots_detected=int(meta.get("shots_detected", 0)))

    def snapshot(self) -> dict:
        with self._condition:
            order = self.processing_order()
            return {"max_workers": self.max_workers, "processing": len(self._active),
                    "waiting": len(self._waiting), "processing_order": order,
                    "concurrency_limit": 1 if order == "sequential" else self.max_workers}

    def _slot(self) -> JobLease | None:
        # Probe all slots under a shared dispatch lease. When switching from two
        # active matches to sequential, a free slot must wait for the other one
        # to finish too; selecting only slot 0 would accidentally run two again.
        gate = JobLease(self.store.root / ".workers", ".dispatch.lock")
        if not gate.acquire():
            return None
        free = []
        try:
            for index in range(self.max_workers):
                lease = JobLease(self.store.root / ".workers" / str(index))
                if lease.acquire():
                    free.append(lease)
            limit = 1 if self.processing_order() == "sequential" else self.max_workers
            if self.max_workers - len(free) < limit and free:
                return free.pop(0)
            return None
        finally:
            for lease in free:
                lease.release()
            gate.release()

    def _worker(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                for entry in list(self._waiting):
                    if WorkControl(self.store.job_dir(entry[0]["job_id"])).requested():
                        self._waiting.remove(entry)
                        self._mark_paused(entry[0]["job_id"])
                        entry[1].release()
                slot = self._slot() if self._waiting else None
                if slot is None:
                    self._condition.wait(.25)
                    continue
                request, lease = self._waiting.popleft()
                job_id = request["job_id"]
                self._active.add(job_id)
            try:
                # A crash after this point becomes interrupted, with saved checkpoints.
                for attempt in range(8):
                    try:
                        (self.store.job_dir(job_id) / "queued.json").unlink(missing_ok=True)
                        break
                    except PermissionError:
                        if attempt == 7:
                            raise
                        time.sleep(min(.1, .005 * 2**attempt))
                self.runner(job_id, request["source"], EditMode(request["mode"]), request["resume"],
                            request["force"], self.config, request["auto_export"])
            except Exception as exc:
                logger.exception("Match worker failed for %s", job_id)
                try:
                    meta = self.store.get_meta(job_id)
                    self.store.update_progress(job_id, min(.99, float(meta.get("progress", 0))),
                                               JobStatus.FAILED, "Processing stopped; use Continue to retry",
                                               error=str(exc), shots_detected=int(meta.get("shots_detected", 0)))
                except Exception:
                    logger.exception("Could not save failure status for %s; checkpoints remain on disk", job_id)
            finally:
                with self._condition:
                    self._active.discard(job_id)
                    lease.release()
                    slot.release()
                    self._condition.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Bounded drain for checks/shutdown; never wait hours for a match."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._waiting or self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    def close(self) -> None:
        # Let short tasks finish; retain all waiting requests if work is long.
        self.wait_idle(2.0)
        with self._condition:
            self._closed = True
            while self._waiting:
                _, lease = self._waiting.popleft()
                lease.release()
            self._condition.notify_all()
