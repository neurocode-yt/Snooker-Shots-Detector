"""FFmpeg / FFprobe process helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from snooker_ai.utils.logging import get_logger

logger = get_logger("ffmpeg")


class FFmpegError(RuntimeError):
    def __init__(self, message: str, returncode: int = 1, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Common Windows install locations
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    raise FFmpegError(
        "ffmpeg not found on PATH. Install FFmpeg and ensure ffmpeg/ffprobe are available."
    )


def find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    candidates = [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"),
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    raise FFmpegError(
        "ffprobe not found on PATH. Install FFmpeg and ensure ffmpeg/ffprobe are available."
    )


@lru_cache(maxsize=8)
def supports_encoder(encoder: str, ffmpeg: str | None = None) -> bool:
    """Return whether the selected FFmpeg binary exposes an encoder."""

    binary = ffmpeg or find_ffmpeg()
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and encoder in (result.stdout or "")


def run_command(
    args: Sequence[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: Optional[float] = None,
    cwd: Optional[str | Path] = None,
) -> subprocess.CompletedProcess[str]:
    logger.debug("Running: %s", " ".join(str(a) for a in args))
    try:
        result = subprocess.run(
            list(args),
            check=False,
            capture_output=capture,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"Executable not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"Command timed out after {timeout}s: {args[0]}") from exc

    if check and result.returncode != 0:
        stderr = result.stderr or ""
        raise FFmpegError(
            f"Command failed ({result.returncode}): {' '.join(str(a) for a in args[:6])}...",
            returncode=result.returncode,
            stderr=stderr,
        )
    return result


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    ffprobe = find_ffprobe()
    args = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    result = run_command(args)
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"Invalid ffprobe JSON for {path}") from exc
