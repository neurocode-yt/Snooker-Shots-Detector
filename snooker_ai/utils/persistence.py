"""Publish complete, durable files without exposing partially written data."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


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
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.25, .01 * 2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    with atomic_output(path) as stream:
        stream.write(text)
