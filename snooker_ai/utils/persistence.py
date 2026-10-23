"""Publish complete, durable files without exposing partially written data."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_before_publish = None


class RecoverySaveError(RuntimeError):
    """A required Android recovery copy could not be made durable."""


def set_publish_guard(callback=None):
    """Optionally preserve a durable recovery copy before publishing a write.

    The Android host installs this guard; desktop behavior stays unchanged.
    A guard failure must propagate so a save is never acknowledged as protected.
    """
    global _before_publish
    _before_publish = callback


@contextmanager
def atomic_output(path: Path, *, binary: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if _before_publish is not None:
            _before_publish(path, temporary)
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.25, .01 * 2**attempt))
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    with atomic_output(path) as stream:
        stream.write(text)
