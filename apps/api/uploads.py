"""Save complete uploads without leaving broken videos when storage fills."""

from __future__ import annotations

import errno
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from snooker_ai.utils.logging import get_logger

logger = get_logger("api.uploads")
_CHUNK_SIZE = 4 * 1024 * 1024


def _check_space(directory: Path, incoming: int, reserve: int) -> None:
    free = shutil.disk_usage(directory).free
    required = incoming + reserve
    if free < required:
        raise HTTPException(
            507,
            f"Not enough free space in {directory.resolve()}: "
            f"{required / 1024**3:.2f} GiB needed including working space, "
            f"{free / 1024**3:.2f} GiB available. Free space on that drive, then retry.",
        )


def save_upload(
    file: UploadFile, directory: Path, max_gb: float, reserve_mb: float = 256
) -> dict[str, Any]:
    """Run in a worker: validate capacity, copy to a private part, then publish."""
    limit = max_gb * 1024**3
    expected = file.size
    if expected is not None and expected > limit:
        raise HTTPException(413, f"File exceeds {max_gb} GB limit")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    stem = Path(file.filename or "video").stem
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
    dest = directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex}_{safe_stem}{suffix}"
    partial = dest.with_name(dest.name + ".part")
    reserve = max(0, int(reserve_mb * 1024**2))
    size = 0
    created = False
    try:
        _check_space(directory, expected or 0, reserve)
        with partial.open("xb") as out:
            created = True
            while chunk := file.file.read(_CHUNK_SIZE):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds {max_gb} GB limit")
                # Other uploads or exports can consume space after preflight.
                _check_space(directory, len(chunk), reserve)
                out.write(chunk)
        if expected is not None and size != expected:
            raise HTTPException(400, "The upload was incomplete. Select the video and retry.")
        # Closing flushes buffered writes before a complete video is published.
        partial.replace(dest)
    except OSError as exc:
        if exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)} or (
            getattr(exc, "winerror", None) in {39, 112}
        ):
            raise HTTPException(
                507,
                f"The drive containing {directory.resolve()} ran out of space while saving "
                "the upload. Free space on that drive, then retry.",
            ) from exc
        raise
    finally:
        if created:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove incomplete upload %s", partial, exc_info=True)
    return {"path": str(dest.resolve()), "filename": file.filename, "size": size}
