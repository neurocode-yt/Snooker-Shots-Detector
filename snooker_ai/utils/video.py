"""Video decode helpers: GPU-accelerated capture and decode-ahead sampling.

The analysis loop previously decoded every proxy frame on the CPU inside the
same thread that ran feature extraction.  Two independent improvements live
here:

* ``open_capture`` asks OpenCV's FFmpeg backend for a hardware decoder
  (NVDEC/D3D11 on Windows) with an automatic software fallback, so decode
  moves off the CPU whenever the driver allows it.
* ``sampled_frames`` runs the grab/read/skip sampling loop on a background
  thread with a small bounded queue.  OpenCV releases the GIL inside
  ``grab``/``read``, so frame decode genuinely overlaps the OpenCL/NumPy
  feature work in the consumer.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from snooker_ai.utils.logging import get_logger

logger = get_logger("utils.video")

_hw_state_logged = False


def open_capture(path: str | Path, *, prefer_hwaccel: bool = True) -> cv2.VideoCapture:
    """Open a video, preferring hardware decode with a software fallback.

    ``VIDEO_ACCELERATION_ANY`` lets the FFmpeg backend select a hardware
    decoder and silently fall back to software decode, so the returned capture
    works identically on machines without a compatible GPU/driver.  A plain
    re-open guards against builds whose backend rejects acceleration params.
    """
    global _hw_state_logged
    if prefer_hwaccel and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        try:
            cap = cv2.VideoCapture(
                str(path),
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, int(cv2.VIDEO_ACCELERATION_ANY)],
            )
            if cap.isOpened():
                if not _hw_state_logged:
                    accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
                    logger.info(
                        "Video decode backend: %s",
                        "hardware-accelerated" if accel else "software",
                    )
                    _hw_state_logged = True
                return cap
            cap.release()
        except Exception as exc:  # pragma: no cover - build/driver specific
            logger.debug("Hardware-accelerated capture unavailable: %s", exc)
    return cv2.VideoCapture(str(path))


def sampled_frames(
    cap: cv2.VideoCapture,
    *,
    proxy_fps: float,
    to_source: Callable[[float], float],
    start_time: float,
    end_time: float,
    sample_period: float,
    start_idx: int = 0,
    prefetch: int = 8,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, source_t, frame)`` for each sampled frame.

    Sampling decisions are identical to the previous inline analyzer loop:
    unsampled frames are ``grab``-advanced without BGR retrieval, and the
    next-sample clock only moves forward from ``start_time`` in
    ``sample_period`` steps.  The caller keeps ownership of ``cap`` and must
    release it after the iterator is exhausted or closed.
    """
    out: queue.Queue = queue.Queue(maxsize=max(2, int(prefetch)))
    stop = threading.Event()
    error: list[BaseException] = []

    def produce() -> None:
        idx = start_idx
        produced = 0
        next_sample_t = start_time
        try:
            while not stop.is_set():
                video_t = (
                    idx / proxy_fps
                    if proxy_fps > 0
                    else start_time + produced * sample_period
                )
                source_t = to_source(video_t)
                if source_t > end_time + 1e-6:
                    break
                if source_t < start_time - 1e-6 or source_t + 1e-9 < next_sample_t:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                while next_sample_t <= source_t + 1e-9:
                    next_sample_t += sample_period
                out.put((idx, source_t, frame))
                idx += 1
                produced += 1
        except BaseException as exc:  # re-raised in the consumer
            error.append(exc)
        finally:
            out.put(None)

    worker = threading.Thread(target=produce, name="decode-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = out.get()
            if item is None:
                break
            yield item
        if error:
            raise error[0]
    finally:
        stop.set()
        # Unblock a producer waiting on a full queue, then let it finish.
        while worker.is_alive():
            try:
                out.get(timeout=0.05)
            except queue.Empty:
                continue
        worker.join(timeout=2.0)
