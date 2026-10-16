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


class IncompleteVideoError(RuntimeError):
    """A decoder ended before the requested source interval was read."""


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
    next_sample_time: float | None = None,
    presentation_timestamps: bool = False,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, source_t, frame)`` for each sampled frame.

    Proxy sampling retains the previous inline analyzer decisions:
    unsampled frames are ``grab``-advanced without BGR retrieval, and the
    next-sample clock only moves forward from ``start_time`` in
    ``sample_period`` steps.  The caller keeps ownership of ``cap`` and must
    release it after the iterator is exhausted or closed. Native observations
    opt into decoded presentation timestamps; at source cadence every original
    image is retained, including small timestamp quantization differences.
    """
    out: queue.Queue = queue.Queue(maxsize=max(2, int(prefetch)))
    stop = threading.Event()
    error: list[BaseException] = []

    def produce() -> None:
        idx = start_idx
        produced = 0
        next_sample_t = start_time if next_sample_time is None else next_sample_time
        ended_early = False
        previous_pts = None
        source_t = start_time
        try:
            while not stop.is_set():
                if presentation_timestamps:
                    if not cap.grab():
                        ended_early = True
                        break
                    pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
                    if (not np.isfinite(pts) or pts < 0
                            or previous_pts is not None and pts <= previous_pts):
                        raise IncompleteVideoError("Native video presentation timestamps are invalid; saved work is preserved.")
                    previous_pts = pts
                    source_t = to_source(pts)
                    if source_t > end_time+1e-6:
                        break
                    every_native_frame = proxy_fps > 0 and sample_period <= 1.001/proxy_fps
                    if source_t < start_time-1e-6 or (not every_native_frame and source_t+1e-6 < next_sample_t):
                        idx += 1
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        ended_early = True
                        break
                    while next_sample_t <= source_t+1e-9:
                        next_sample_t += sample_period
                    out.put((idx, source_t, frame))
                    idx += 1
                    produced += 1
                    continue
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
                        ended_early = True
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    ended_early = True
                    break
                while next_sample_t <= source_t + 1e-9:
                    next_sample_t += sample_period
                out.put((idx, source_t, frame))
                idx += 1
                produced += 1
            if ended_early and not stop.is_set():
                reached = (source_t if presentation_timestamps else
                           to_source(idx / proxy_fps) if proxy_fps > 0 else source_t)
                frame_period = abs(to_source(1 / proxy_fps) - to_source(0)) if proxy_fps > 0 else sample_period
                if reached + max(.1, 2 * frame_period) < end_time:
                    raise IncompleteVideoError(
                        f"Video decoding stopped at {reached:.2f}s of {end_time:.2f}s. "
                        "Processing is incomplete; saved checkpoints are preserved. "
                        "Use Continue to retry from the last checkpoint."
                    )
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
