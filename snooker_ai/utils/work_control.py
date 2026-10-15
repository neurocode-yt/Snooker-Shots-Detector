"""Dependency-light cooperative pause requests scoped to an owned unit of work."""

from __future__ import annotations

import time
from contextvars import ContextVar
from pathlib import Path

from snooker_ai.utils.persistence import atomic_text


class WorkPaused(BaseException):
    """Control flow, like cancellation: never turn a pause into a codec retry."""


current_control: ContextVar[WorkControl | None] = ContextVar("work_control", default=None)


class WorkControl:
    def __init__(self, directory: Path):
        self.path = Path(directory) / "pause.requested"
        self.last_check = 0.0

    def request(self):
        atomic_text(self.path, str(time.time()))

    def clear(self):
        self.path.unlink(missing_ok=True)

    def requested(self) -> bool:
        return self.path.is_file()

    def check(self):
        now = time.monotonic()
        if now - self.last_check < 0.1:
            return
        self.last_check = now
        if self.requested():
            raise WorkPaused()


def check_pause():
    control = current_control.get()
    if control is not None:
        control.check()
