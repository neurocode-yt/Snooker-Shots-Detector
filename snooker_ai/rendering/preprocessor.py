"""Pre-analysis removal of user-selected match breaks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from snooker_ai.config import Config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command


@dataclass(frozen=True)
class KeepRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_keep_ranges(
    ranges: Iterable[KeepRange],
    source_duration: float,
    *,
    max_sections: int = 200,
    minimum_seconds: float = 0.05,
) -> list[KeepRange]:
    """Validate, sort, and merge touching source ranges."""

    duration = max(0.0, float(source_duration))
    ordered = sorted(
        (KeepRange(float(item.start), float(item.end)) for item in ranges),
        key=lambda item: item.start,
    )
    if not ordered:
        raise ValueError("At least one section must be kept")
    if len(ordered) > max_sections:
        raise ValueError(f"Too many sections; maximum is {max_sections}")

    merged: list[KeepRange] = []
    tolerance = 1e-3
    for item in ordered:
        if item.start < -tolerance or item.end > duration + tolerance:
            raise ValueError("A section is outside the source video duration")
        start = max(0.0, item.start)
        end = min(duration, item.end)
        if end - start < minimum_seconds:
            raise ValueError("Every kept section must be at least 0.05 seconds")
        if merged and start < merged[-1].end - tolerance:
            raise ValueError("Kept sections cannot overlap")
        if merged and start <= merged[-1].end + tolerance:
            merged[-1] = KeepRange(merged[-1].start, max(merged[-1].end, end))
        else:
            merged.append(KeepRange(start, end))
    return merged


class VideoPreprocessor:
    """Render kept source ranges into one timestamp-continuous MP4."""

    def __init__(self, config: Config):
        self.config = config
        self.pcfg = config.section("preprocess")

    def render(
        self,
        source: str | Path,
        ranges: Iterable[KeepRange],
        output: str | Path,
    ) -> Path:
        source = Path(source).resolve()
        output = Path(output).resolve()
        metadata = probe_video(source)
        keep = normalize_keep_ranges(
            ranges,
            metadata.duration,
            max_sections=int(self.pcfg.get("max_sections", 200)),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        working = output.with_name(f"{output.stem}.working{output.suffix}")
        working.unlink(missing_ok=True)

        exporter = Exporter(self.config)
        ffmpeg = find_ffmpeg()
        codec, _use_gpu_decode = exporter._video_codec(ffmpeg)
        crf = str(self.pcfg.get("crf", 18))
        preset = str(self.pcfg.get("preset", "veryfast"))
        if codec == "h264_nvenc":
            preset = str(self.pcfg.get("nvenc_preset", "p4"))
        audio_codec = str(self.pcfg.get("audio_codec", "aac"))
        audio_bitrate = str(self.pcfg.get("audio_bitrate", "192k"))
        pixel_format = str(self.pcfg.get("pixel_format", "yuv420p"))

        filters: list[str] = []
        concat_inputs: list[str] = []
        for index, item in enumerate(keep):
            filters.append(
                f"[0:v:0]trim=start={item.start:.9f}:end={item.end:.9f},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
            concat_inputs.append(f"[v{index}]")
            if metadata.has_audio:
                filters.append(
                    f"[0:a:0]atrim=start={item.start:.9f}:end={item.end:.9f},"
                    f"asetpts=PTS-STARTPTS[a{index}]"
                )
                concat_inputs.append(f"[a{index}]")

        if metadata.has_audio:
            filters.append(
                "".join(concat_inputs)
                + f"concat=n={len(keep)}:v=1:a=1[outv][outa]"
            )
        else:
            filters.append(
                "".join(concat_inputs) + f"concat=n={len(keep)}:v=1:a=0[outv]"
            )

        args = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            *( ["-map", "[outa]"] if metadata.has_audio else [] ),
            "-c:v",
            codec,
            "-preset",
            preset,
            *exporter._video_quality(codec, crf),
            *( ["-pix_fmt", pixel_format] if codec != "h264_nvenc" else [] ),
            *( ["-c:a", audio_codec, "-b:a", audio_bitrate] if metadata.has_audio else ["-an"] ),
            "-movflags",
            "+faststart",
            str(working),
        ]
        kept_duration = sum(item.duration for item in keep)
        try:
            run_command(args, timeout=max(600.0, kept_duration * 10.0))
            rendered = probe_video(working)
            duration_tolerance = max(0.5, 2.0 / max(metadata.fps, 1.0))
            if abs(rendered.duration - kept_duration) > duration_tolerance:
                raise RuntimeError(
                    "Cleaned video duration does not match the selected sections"
                )
            if metadata.has_audio and not rendered.has_audio:
                raise RuntimeError("Cleaned video is missing source audio")
            os.replace(working, output)
        except Exception:
            working.unlink(missing_ok=True)
            raise
        return output
