"""Lower-resolution analysis proxy generation with timestamp mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from snooker_ai.config import Config
from snooker_ai.types import VideoMetadata
from snooker_ai.utils.acceleration import acceleration_enabled
from snooker_ai.utils.ffmpeg import FFmpegError, find_ffmpeg, run_command, supports_encoder
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.timebase import TimeMapper

logger = get_logger("ingestion.proxy")


@dataclass
class ProxyResult:
    proxy_path: Path
    audio_path: Optional[Path]
    mapper: TimeMapper
    width: int
    height: int
    fps: float


def _compute_proxy_size(
    width: int, height: int, max_w: int, max_h: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return max_w, max_h
    scale = min(max_w / width, max_h / height, 1.0)
    pw = max(2, int(round(width * scale / 2) * 2))
    ph = max(2, int(round(height * scale / 2) * 2))
    return pw, ph


def generate_proxy(
    source: str | Path,
    output_dir: str | Path,
    metadata: VideoMetadata,
    config: Config,
    *,
    force: bool = False,
) -> ProxyResult:
    """
    Create a seek-friendly review/analysis proxy and optional mono WAV features.

    Cuts/export always use the original source; analysis uses the proxy.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_cfg = config.section("proxy")
    max_w = int(proxy_cfg.get("max_width", 960))
    max_h = int(proxy_cfg.get("max_height", 540))
    target_fps = float(proxy_cfg.get("target_fps", 15.0))
    crf = int(proxy_cfg.get("crf", 28))
    preset = str(proxy_cfg.get("preset", "veryfast"))
    keyframe_s = max(0.25, float(proxy_cfg.get("seek_keyframe_seconds", 2.0)))
    gop_frames = max(1, int(round(target_fps * keyframe_s)))
    review_audio_bitrate = str(proxy_cfg.get("review_audio_bitrate", "96k"))
    audio_sr = int(proxy_cfg.get("audio_sample_rate", 16000))
    extract_audio = bool(proxy_cfg.get("extract_audio", True))

    pw, ph = _compute_proxy_size(metadata.width, metadata.height, max_w, max_h)
    proxy_path = output_dir / "proxy.mp4"
    backend_marker = output_dir / "proxy.backend"
    audio_path = output_dir / "audio.wav"

    reuse_proxy = proxy_path.exists() and not force
    # Upgrade an older CPU proxy once when NVIDIA acceleration is now available.
    # New jobs have no marker and therefore take the GPU path on first decode.
    if reuse_proxy and acceleration_enabled(config):
        try:
            if backend_marker.read_text(encoding="utf-8").strip() != "nvidia_nvenc":
                reuse_proxy = False
        except OSError:
            reuse_proxy = False
    if reuse_proxy:
        # A stale low-FPS proxy silently quantises cue contact even when the
        # caller has raised ``proxy.target_fps``.  Regenerate when cadence or
        # dimensions do not match the requested analysis contract.
        try:
            from snooker_ai.ingestion.probe import probe_video

            existing = probe_video(proxy_path)
            if (
                abs(float(existing.fps or 0.0) - target_fps) > 0.75
                or int(existing.width or 0) != pw
                or int(existing.height or 0) != ph
                or bool(existing.has_audio) != bool(metadata.has_audio)
            ):
                reuse_proxy = False
                logger.info(
                    "Regenerating stale proxy (found %sx%s @ %.2ffps; need %sx%s @ %.2ffps)",
                    existing.width,
                    existing.height,
                    existing.fps,
                    pw,
                    ph,
                    target_fps,
                )
        except Exception as exc:
            logger.warning("Could not validate existing proxy; regenerating: %s", exc)
            reuse_proxy = False

    if reuse_proxy:
        logger.info("Reusing existing proxy: %s", proxy_path)
    else:
        ffmpeg = find_ffmpeg()
        use_nvenc = acceleration_enabled(config) and supports_encoder("h264_nvenc", ffmpeg)
        audio_args = (
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:a",
                "aac",
                "-b:a",
                review_audio_bitrate,
            ]
            if metadata.has_audio
            else ["-map", "0:v:0", "-an"]
        )
        if use_nvenc:
            args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(source),
                "-vf",
                f"scale_cuda={pw}:{ph}:format=nv12",
                "-r",
                str(target_fps),
                "-fps_mode",
                "cfr",
                "-c:v",
                "h264_nvenc",
                "-preset",
                str(proxy_cfg.get("nvenc_preset", "p4")),
                "-cq",
                str(crf),
                "-g",
                str(gop_frames),
                # scale_cuda already emits NV12 frames for NVENC; forcing a
                # software yuv420p conversion would move every frame back to
                # the CPU and defeat the GPU path.
                *audio_args,
                "-movflags",
                "+faststart",
                str(proxy_path),
            ]
        else:
            args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"scale={pw}:{ph}:flags=bicubic,fps={target_fps}",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(gop_frames),
                "-keyint_min",
                str(gop_frames),
                "-sc_threshold",
                "0",
                *audio_args,
                "-movflags",
                "+faststart",
                str(proxy_path),
            ]
        logger.info("Generating proxy %dx%d @ %.1ffps → %s", pw, ph, target_fps, proxy_path)
        try:
            run_command(args, timeout=max(600.0, metadata.duration * 2))
        except FFmpegError:
            if not use_nvenc:
                raise
            logger.warning("NVIDIA proxy path failed; retrying with CPU FFmpeg")
            cpu_args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"scale={pw}:{ph}:flags=bicubic,fps={target_fps}",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(gop_frames),
                "-keyint_min",
                str(gop_frames),
                "-sc_threshold",
                "0",
                *audio_args,
                "-movflags",
                "+faststart",
                str(proxy_path),
            ]
            run_command(cpu_args, timeout=max(600.0, metadata.duration * 2))
        backend_marker.write_text(
            "nvidia_nvenc" if use_nvenc and proxy_path.exists() else "cpu",
            encoding="utf-8",
        )

    audio_out: Optional[Path] = None
    if extract_audio and metadata.has_audio:
        if audio_path.exists() and not force:
            audio_out = audio_path
            logger.info("Reusing existing audio extract: %s", audio_path)
        else:
            ffmpeg = find_ffmpeg()
            args = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(audio_sr),
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ]
            logger.info("Extracting mono audio → %s", audio_path)
            run_command(args, timeout=max(600.0, metadata.duration * 2))
            audio_out = audio_path
    elif not metadata.has_audio:
        logger.info("Source has no audio; skipping audio extract")

    # Probe proxy duration for mapper (may differ by a few frames)
    from snooker_ai.ingestion.probe import probe_video

    try:
        proxy_meta = probe_video(proxy_path)
        proxy_duration = proxy_meta.duration
    except Exception:
        proxy_duration = metadata.duration

    mapper = TimeMapper(
        source_duration=metadata.duration,
        proxy_duration=proxy_duration,
        source_fps=metadata.fps or 25.0,
        analysis_fps=target_fps,
    )

    return ProxyResult(
        proxy_path=proxy_path,
        audio_path=audio_out,
        mapper=mapper,
        width=pw,
        height=ph,
        fps=target_fps,
    )
