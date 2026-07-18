"""Video metadata inspection via FFprobe."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from snooker_ai.types import VideoMetadata
from snooker_ai.utils.ffmpeg import ffprobe_json
from snooker_ai.utils.logging import get_logger

logger = get_logger("ingestion.probe")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}


def _parse_fraction(value: Optional[str]) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except ValueError:
            return 0.0


def _stream_by_type(probe: dict[str, Any], codec_type: str) -> Optional[dict[str, Any]]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _count_streams(probe: dict[str, Any], codec_type: str) -> int:
    return sum(1 for s in probe.get("streams", []) if s.get("codec_type") == codec_type)


def _rotation_from_stream(stream: dict[str, Any]) -> float:
    # side_data_list rotation or tags.rotate
    for item in stream.get("side_data_list", []) or []:
        if "rotation" in item:
            try:
                return float(item["rotation"])
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return float(tags["rotate"])
        except (TypeError, ValueError):
            pass
    return 0.0


def _is_vfr(stream: dict[str, Any]) -> bool:
    avg = _parse_fraction(stream.get("avg_frame_rate"))
    r = _parse_fraction(stream.get("r_frame_rate"))
    if avg <= 0 or r <= 0:
        return True
    # Significant mismatch suggests VFR
    return abs(avg - r) / max(r, 1e-6) > 0.02


def probe_video(path: str | Path) -> VideoMetadata:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    raw = ffprobe_json(path)
    fmt = raw.get("format") or {}
    vstream = _stream_by_type(raw, "video")
    astream = _stream_by_type(raw, "audio")

    if not vstream:
        raise ValueError(f"No video stream found in {path}")

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    fps = _parse_fraction(vstream.get("avg_frame_rate")) or _parse_fraction(
        vstream.get("r_frame_rate")
    )
    duration = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    if duration <= 0 and vstream.get("nb_frames") and fps > 0:
        try:
            duration = float(vstream["nb_frames"]) / fps
        except (ValueError, ZeroDivisionError):
            pass

    dar = vstream.get("display_aspect_ratio")
    if not dar or dar == "0:1":
        if width and height:
            dar = f"{width}:{height}"

    audio_rate = None
    audio_channels = None
    audio_codec = None
    if astream:
        audio_codec = astream.get("codec_name")
        try:
            audio_rate = int(float(astream.get("sample_rate") or 0)) or None
        except (TypeError, ValueError):
            audio_rate = None
        audio_channels = astream.get("channels")

    bit_rate = None
    try:
        bit_rate = int(fmt.get("bit_rate") or 0) or None
    except (TypeError, ValueError):
        pass

    meta = VideoMetadata(
        path=str(path.resolve()),
        duration=duration,
        width=width,
        height=height,
        display_aspect_ratio=dar,
        fps=fps,
        is_variable_frame_rate=_is_vfr(vstream),
        video_codec=vstream.get("codec_name"),
        audio_codec=audio_codec,
        audio_sample_rate=audio_rate,
        audio_channels=audio_channels,
        num_audio_streams=_count_streams(raw, "audio"),
        rotation=_rotation_from_stream(vstream),
        time_base=vstream.get("time_base"),
        bit_rate=bit_rate,
        format_name=fmt.get("format_name"),
        has_audio=astream is not None,
        probe_raw={
            "format_name": fmt.get("format_name"),
            "nb_streams": fmt.get("nb_streams"),
            "video_pix_fmt": vstream.get("pix_fmt"),
            "video_profile": vstream.get("profile"),
            "r_frame_rate": vstream.get("r_frame_rate"),
            "avg_frame_rate": vstream.get("avg_frame_rate"),
        },
    )
    logger.info(
        "Probed %s: %dx%d @ %.3ffps, %.2fs, v=%s a=%s vfr=%s",
        path.name,
        meta.width,
        meta.height,
        meta.fps,
        meta.duration,
        meta.video_codec,
        meta.audio_codec,
        meta.is_variable_frame_rate,
    )
    return meta


def validate_video(
    path: str | Path,
    *,
    max_hours: float = 12.0,
    require_video: bool = True,
) -> VideoMetadata:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    meta = probe_video(path)
    if require_video and (meta.width <= 0 or meta.height <= 0):
        raise ValueError("Invalid video dimensions")
    if meta.duration <= 0:
        raise ValueError("Could not determine video duration")
    if meta.duration > max_hours * 3600:
        raise ValueError(
            f"Video duration {meta.duration / 3600:.1f}h exceeds max {max_hours}h"
        )
    return meta
