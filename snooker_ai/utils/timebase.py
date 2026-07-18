"""Timestamp helpers and proxy↔source time mapping."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


@dataclass
class TimeMapper:
    """
    Maps analysis/proxy timestamps to original source timestamps.

    Phase 1 assumes a shared presentation timeline (proxy is re-encoded with
    the same duration). For VFR sources, cuts use absolute seconds with
    FFmpeg -ss/-to, which is presentation-time based.
    """

    source_duration: float
    proxy_duration: float | None = None
    source_fps: float = 25.0
    analysis_fps: float = 10.0

    def to_source(self, t_proxy: float) -> float:
        if self.proxy_duration and self.proxy_duration > 0 and self.source_duration > 0:
            # Linear map if durations differ slightly after re-encode
            ratio = self.source_duration / self.proxy_duration
            return clamp(t_proxy * ratio, 0.0, self.source_duration)
        return clamp(t_proxy, 0.0, self.source_duration)

    def to_proxy(self, t_source: float) -> float:
        if self.proxy_duration and self.proxy_duration > 0 and self.source_duration > 0:
            ratio = self.proxy_duration / self.source_duration
            return clamp(t_source * ratio, 0.0, self.proxy_duration)
        return clamp(t_source, 0.0, self.source_duration)

    def frame_to_time(self, frame_idx: int, fps: float | None = None) -> float:
        rate = fps if fps is not None else self.analysis_fps
        if rate <= 0:
            return 0.0
        return frame_idx / rate

    def time_to_frame(self, t: float, fps: float | None = None) -> int:
        rate = fps if fps is not None else self.analysis_fps
        if rate <= 0:
            return 0
        return int(round(t * rate))
