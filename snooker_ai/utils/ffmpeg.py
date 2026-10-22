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
from snooker_ai.utils.work_control import current_control, check_pause

logger = get_logger("ffmpeg")


@lru_cache(maxsize=32)
def _supports_hide_banner(binary: str) -> bool:
    """Skip stale PATH shims that cannot run the flags used by this app."""
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _first_compatible(candidates: Sequence[str]) -> str | None:
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).is_file() and _supports_hide_banner(candidate):
            return candidate
    return None


class FFmpegError(RuntimeError):
    def __init__(self, message: str, returncode: int = 1, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def find_ffmpeg() -> str:
    bundled = os.environ.get("SNOOKER_FFMPEG")
    if bundled and Path(bundled).is_file():
        return bundled
    candidates = []
    path = shutil.which("ffmpeg")
    if path:
        candidates.append(path)
    candidates.extend(
        [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        ]
    )
    winget_pkg = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(winget_pkg):
        for root, _, files in os.walk(winget_pkg):
            for file in files:
                if file.lower() == "ffmpeg.exe":
                    candidates.append(os.path.join(root, file))

    return _first_compatible(candidates) or "ffmpeg"


def find_ffprobe() -> str:
    bundled = os.environ.get("SNOOKER_FFPROBE")
    if bundled and Path(bundled).is_file():
        return bundled
    candidates = []
    path = shutil.which("ffprobe")
    if path:
        candidates.append(path)
    candidates.extend(
        [
            r"C:\ffmpeg\bin\ffprobe.exe",
            r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"),
        ]
    )
    winget_pkg = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(winget_pkg):
        for root, _, files in os.walk(winget_pkg):
            for file in files:
                if file.lower() == "ffprobe.exe":
                    candidates.append(os.path.join(root, file))

    return _first_compatible(candidates) or "ffprobe"


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


def _run_process(args, *, capture, timeout, cwd):
    options = dict(text=True, cwd=str(cwd) if cwd else None, encoding="utf-8", errors="replace")
    if current_control.get() is None:
        return subprocess.run(list(args), check=False, capture_output=capture, timeout=timeout, **options)
    check_pause()
    import time
    started = time.monotonic()
    process = subprocess.Popen(list(args), stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.PIPE if capture else None, **options)
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=.15)
                check_pause()
                return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                check_pause()
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise


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
        result = _run_process(args, capture=capture, timeout=timeout, cwd=cwd)
    except FileNotFoundError as exc:
        if len(args) > 0 and (Path(args[0]).is_absolute() or "\\" in str(args[0])):
            fallback_binary = "ffmpeg" if "ffmpeg" in Path(args[0]).name.lower() else "ffprobe"
            new_args = [fallback_binary] + list(args[1:])
            try:
                result = _run_process(new_args, capture=capture, timeout=timeout, cwd=cwd)
            except FileNotFoundError:
                raise FFmpegError(f"Executable not found: {args[0]}") from exc
        else:
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
