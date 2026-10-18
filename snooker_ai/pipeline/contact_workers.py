"""Isolated CPU workers for independent, unchanged native contact windows."""

from __future__ import annotations

import copy
import ctypes
import os
from pathlib import Path

import cv2

from snooker_ai.utils.work_control import WorkControl, check_pause, current_control


def available_memory() -> int | None:
    """Available physical RAM without requiring another runtime dependency."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total", "available", "page_total", "page_available",
                        "virtual_total", "virtual_available", "extended_available",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        else:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return None


def contact_worker_count(requested, backend: str) -> int:
    """Bound outer parallelism; leave CPU and RAM for the chronological parent."""
    logical = max(1, os.cpu_count() or 1)
    cpu_limit = max(1, logical // 2)
    automatic = min(4, cpu_limit)
    count = automatic if str(requested).lower() == "auto" else max(1, int(requested))
    count = min(count, cpu_limit, 4 if backend == "process" else 2)
    free = available_memory()
    if free is not None:
        # The parent retains the growing match timeline. Each persistent child
        # also owns a decoder, OpenCV buffers and a bounded pure-pixel cache.
        count = min(count, max(1, (free - 1536 * 1024**2) // (768 * 1024**2)))
    else:
        count = min(count, 2)
    return int(count)


_worker = None
_settings = None


def init_contact_process(config_data, directory, source, proxy, audio, mapper, duration, pause_directory):
    # Outer processes already occupy separate cores. Eight inner OpenCV threads
    # per child create contention on laptop CPUs, without adding observations.
    cv2.setNumThreads(1)
    from snooker_ai.config import Config
    from snooker_ai.pipeline.analyzer import Analyzer

    global _worker, _settings
    _worker = Analyzer(Config(config_data), Path(directory) / str(os.getpid()))
    _worker._native_source = source
    _settings = (proxy, audio, mapper, duration)
    if pause_directory is not None:
        current_control.set(WorkControl(Path(pause_directory)))


def extract_contact_process(key, context_guard, reference, target_known):
    return extract_contact(_worker, _settings, key, context_guard, reference, target_known)


def extract_contact(worker, settings, key, context_guard, reference, target_known):
    """Keep temporal state inside one window; return observations in source order."""
    from snooker_ai.types import FrameFeatures

    check_pause()
    start, end, fps = key
    proxy, audio, mapper, duration = settings
    # Keep the exact float32 template; restore_target would normalize it again.
    worker.broadcast_context = copy.deepcopy(context_guard)
    worker._confirmed_target_returns = []
    worker._coarse_context_reference = [
        FrameFeatures(t=t, match_context_valid=valid) for t, valid in reference
    ]
    features, _, _ = worker._extract_features(
        proxy, audio, mapper, duration, sample_fps=fps, start_time=start, end_time=end,
        collect_scene_observations=False, checkpoint_stage="contact_prefetch",
    )
    if worker._confirmed_target_returns or not target_known and worker.broadcast_context.target_ready:
        return None  # apply new identity evidence in parent order via normal extraction
    check_pause()
    return features
