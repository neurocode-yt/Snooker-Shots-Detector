"""FFmpeg-based accurate and fast export of shot clips and joined video."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from snooker_ai.config import Config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.types import AnalysisResult, EditMode, ExportRequest, ShotRecord
from snooker_ai.utils.acceleration import acceleration_enabled
from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg, run_command, supports_encoder
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.timebase import format_timestamp

logger = get_logger("rendering")


@dataclass
class ExportResult:
    joined_path: Optional[Path] = None
    clip_paths: list[Path] = field(default_factory=list)
    csv_path: Optional[Path] = None
    edl_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    training_labels_path: Optional[Path] = None


class Exporter:
    def __init__(self, config: Config):
        self.config = config
        self.ecfg = config.section("export")
        self.use_gpu = acceleration_enabled(config)

    def _video_codec(self, ffmpeg: str) -> tuple[str, bool]:
        """Resolve ``auto`` to NVENC and report whether CUDA decode is safe."""

        configured = str(self.ecfg.get("video_codec", "auto")).strip().lower()
        if configured in {"auto", "gpu", "cuda"}:
            if self.use_gpu and supports_encoder("h264_nvenc", ffmpeg):
                return "h264_nvenc", True
            return "libx264", False
        return configured, configured == "h264_nvenc" and self.use_gpu

    @staticmethod
    def _video_quality(codec: str, crf: str) -> list[str]:
        return ["-cq", crf] if codec == "h264_nvenc" else ["-crf", crf]

    def export(
        self,
        result: AnalysisResult,
        output_dir: str | Path,
        request: Optional[ExportRequest] = None,
    ) -> ExportResult:
        request = request or ExportRequest(mode=result.mode)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        shots = self._filter_shots(result.shots, request)
        if not shots:
            raise ValueError("No shots selected for export")

        strict_export = result.mode == EditMode.STRICT
        if strict_export:
            self._validate_strict_boundaries(
                shots,
                source_duration=result.metadata.duration or result.original_duration,
                source_fps=result.metadata.fps,
            )

        out = ExportResult()
        accurate = request.accurate if request.accurate is not None else (
            str(self.ecfg.get("mode", "accurate")) == "accurate"
        )
        # Stream-copy cuts are keyframe aligned and cannot satisfy the strict
        # cue-contact/physical-stop boundary contract.  A strict analysis is
        # therefore always decoded and re-encoded, even if a caller asks for
        # the faster approximate path.
        if strict_export:
            accurate = True

        if request.export_clips:
            out.clip_paths = self._export_clips(
                result.source_path,
                shots,
                clips_dir,
                accurate=accurate,
                source_has_audio=result.metadata.has_audio,
                source_fps=result.metadata.fps,
            )

        if request.export_joined:
            joined_name = request.output_path
            if joined_name:
                joined_path = Path(joined_name)
                if not joined_path.is_absolute():
                    joined_path = output_dir / joined_path
            else:
                joined_path = output_dir / "highlights.mp4"
            # Ensure clips exist for concat
            if not out.clip_paths:
                out.clip_paths = self._export_clips(
                    result.source_path,
                    shots,
                    clips_dir,
                    accurate=accurate,
                    source_has_audio=result.metadata.has_audio,
                    source_fps=result.metadata.fps,
                )
            out.joined_path = self._concat_clips(
                out.clip_paths,
                joined_path,
                accurate=accurate,
                has_audio=result.metadata.has_audio,
                source_fps=result.metadata.fps,
            )

        if request.export_csv:
            out.csv_path = self._write_csv(shots, output_dir / "shots.csv")

        if request.export_edl:
            out.edl_path = self._write_edl(shots, output_dir / "timeline.edl", result.source_path)

        # Chapter markers (ffmpeg metadata file + human-readable list)
        self._write_chapters(shots, output_dir / "chapters.ffmeta")
        self._write_chapter_list(shots, output_dir / "chapters.txt")

        out.metadata_path = output_dir / "export_metadata.json"
        out.metadata_path.write_text(
            json.dumps(
                {
                    "job_id": result.job_id,
                    "mode": result.mode.value,
                    "shots": [s.model_dump() for s in shots],
                    "original_duration": result.original_duration,
                    "edited_duration": sum(s.duration() for s in shots),
                    "export_accurate": accurate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Training labels from final (possibly user-corrected) shots
        out.training_labels_path = output_dir / "training_labels.json"
        out.training_labels_path.write_text(
            json.dumps(self._training_labels(result, shots), indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Export complete: %d clips, joined=%s",
            len(out.clip_paths),
            out.joined_path,
        )
        return out

    def _filter_shots(self, shots: list[ShotRecord], request: ExportRequest) -> list[ShotRecord]:
        selected = []
        for s in shots:
            # Replays are excluded conservatively by default.  Explicit replay
            # views and links to an earlier live shot count even if an upstream
            # producer omitted ``possible_replay``.  --include-replays re-admits
            # them even when the segment builder marked them included=False.
            is_replay = (
                s.possible_replay
                or s.linked_live_shot_id is not None
                or any(
                    view in {"replay", "slow_motion_replay"} for view in s.camera_views
                )
            )
            if is_replay:
                if not request.include_replays:
                    continue
            elif request.only_included and not s.included:
                continue
            if s.shot_confidence < request.min_confidence:
                continue
            if s.importance < request.min_importance:
                continue
            selected.append(s)
        return selected

    def _validate_strict_boundaries(
        self,
        shots: list[ShotRecord],
        *,
        source_duration: float,
        source_fps: float,
    ) -> None:
        """Reject strict records whose stored aliases or boundaries diverge.

        Export is deliberately validation-only here: silently moving a
        boundary would hide an upstream detection error and could include the
        0.5-second stop-confirmation look-ahead in the rendered clip.
        """

        tolerance = 1e-6
        end_tolerance = max(tolerance, 1.0 / max(float(source_fps or 30.0), 1.0))
        pre_roll = float(
            self.config.mode_settings(EditMode.STRICT).get("pre_roll", 2.0)
        )
        min_after = float(
            self.config.mode_settings(EditMode.STRICT).get(
                "min_seconds_after_strike", 0.0
            )
        )
        for shot in shots:
            fields = {
                "cue_strike/cue_strike_timestamp": (
                    shot.cue_strike,
                    shot.cue_strike_timestamp,
                ),
                "clip_start/clip_start_timestamp": (
                    shot.clip_start,
                    shot.clip_start_timestamp,
                ),
                "clip_end/clip_end_timestamp": (
                    shot.clip_end,
                    shot.clip_end_timestamp,
                ),
            }
            for label, (legacy, canonical) in fields.items():
                if abs(legacy - canonical) > tolerance:
                    raise ValueError(
                        f"Shot {shot.shot_id} has inconsistent {label}: "
                        f"{legacy:.9f} != {canonical:.9f}"
                    )

            expected_start = max(
                0.0, shot.cue_strike_timestamp - pre_roll
            )
            if abs(shot.clip_start_timestamp - expected_start) > tolerance:
                raise ValueError(
                    f"Shot {shot.shot_id} violates strict start boundary: "
                    f"expected {expected_start:.9f}, got {shot.clip_start_timestamp:.9f}"
                )
            expected_end = min(
                source_duration if source_duration > 0 else float("inf"),
                max(
                    shot.physical_stop_timestamp,
                    shot.cue_strike_timestamp + min_after,
                ),
            )
            if abs(shot.clip_end_timestamp - expected_end) > tolerance:
                raise ValueError(
                    f"Shot {shot.shot_id} violates strict end boundary: "
                    f"clip_end={shot.clip_end_timestamp:.9f}, "
                    f"expected={expected_end:.9f}"
                )
            if shot.stop_confirmation_timestamp + tolerance < shot.physical_stop_timestamp:
                raise ValueError(
                    f"Shot {shot.shot_id} stop confirmation precedes physical stop"
                )
            if shot.last_ball_motion_timestamp > shot.physical_stop_timestamp + tolerance:
                raise ValueError(
                    f"Shot {shot.shot_id} last ball motion follows physical stop"
                )
            if shot.clip_start_timestamp < -tolerance or shot.clip_end_timestamp <= (
                shot.clip_start_timestamp + tolerance
            ):
                raise ValueError(f"Shot {shot.shot_id} has an invalid clip interval")
            if not (
                shot.clip_start_timestamp - tolerance
                <= shot.cue_strike_timestamp
                <= shot.clip_end_timestamp + tolerance
            ):
                raise ValueError(f"Shot {shot.shot_id} strike is outside its clip interval")
            if source_duration > 0 and shot.clip_end_timestamp > source_duration + end_tolerance:
                raise ValueError(
                    f"Shot {shot.shot_id} ends beyond source duration "
                    f"({shot.clip_end_timestamp:.9f} > {source_duration:.9f})"
                )

    def _export_clips(
        self,
        source: str,
        shots: list[ShotRecord],
        clips_dir: Path,
        *,
        accurate: bool,
        source_has_audio: bool = True,
        source_fps: float = 30.0,
    ) -> list[Path]:
        ffmpeg = find_ffmpeg()
        paths: list[Path] = []
        codec, use_gpu = self._video_codec(ffmpeg)
        acodec = str(self.ecfg.get("audio_codec", "aac"))
        crf = str(self.ecfg.get("crf", 18))
        preset = str(self.ecfg.get("preset", "medium"))
        if codec == "h264_nvenc" and preset not in {f"p{i}" for i in range(1, 8)}:
            preset = str(self.ecfg.get("nvenc_preset", "p4"))
        abitrate = str(self.ecfg.get("audio_bitrate", "192k"))
        pix = str(self.ecfg.get("pixel_format", "yuv420p"))

        for s in shots:
            out = clips_dir / f"shot_{s.shot_id:04d}.mp4"
            duration = s.clip_end - s.clip_start
            if duration <= 0:
                raise ValueError(
                    f"Shot {s.shot_id} has non-positive export duration "
                    f"({s.clip_start:.9f} -> {s.clip_end:.9f})"
                )
            if accurate:
                # Accurate: seek after -i so decode lands on exact timestamps.
                args = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    *(("-hwaccel", "cuda", "-hwaccel_output_format", "cuda") if use_gpu else ()),
                    "-i",
                    source,
                    "-ss",
                    f"{s.clip_start:.9f}",
                    "-t",
                    f"{duration:.9f}",
                    "-map",
                    "0:v:0",
                    *(["-map", "0:a:0"] if source_has_audio else []),
                    "-c:v",
                    codec,
                    "-preset",
                    preset,
                    *self._video_quality(codec, crf),
                    *( ["-pix_fmt", pix] if not use_gpu else [] ),
                    "-fps_mode:v",
                    "passthrough",
                    *(["-c:a", acodec, "-b:a", abitrate] if source_has_audio else ["-an"]),
                    "-avoid_negative_ts",
                    "make_zero",
                    "-fflags",
                    "+genpts",
                    "-movflags",
                    "+faststart",
                    str(out),
                ]
            else:
                # Fast stream copy (keyframe-aligned, may be less precise)
                args = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{s.clip_start:.9f}",
                    "-i",
                    source,
                    "-t",
                    f"{duration:.9f}",
                    "-map",
                    "0:v:0",
                    *(["-map", "0:a:0"] if source_has_audio else []),
                    "-c",
                    "copy",
                    *(["-an"] if not source_has_audio else []),
                    "-avoid_negative_ts",
                    "make_zero",
                    str(out),
                ]
            try:
                run_command(args, timeout=max(120.0, duration * 30))
                # Audio presence is never optional when the source has audio.
                # Full timing verification may be disabled for non-strict/fast
                # workflows, but silent output must never look successful.
                verify_timing = bool(self.ecfg.get("verify_sync", True)) or accurate
                self._verify_clip(
                    out,
                    s,
                    source_has_audio=source_has_audio,
                    source_fps=source_fps,
                    verify_timing=verify_timing,
                )
            except Exception as exc:
                out.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Export failed validation for shot {s.shot_id}; "
                    "no audio-less or boundary-shifting fallback was used"
                ) from exc
            paths.append(out)
            logger.info("Exported %s (%.2fs–%.2fs)", out.name, s.clip_start, s.clip_end)
        return paths

    def _concat_clips(
        self,
        clip_paths: list[Path],
        output: Path,
        *,
        accurate: bool,
        has_audio: bool = True,
        source_fps: float = 30.0,
        expected_duration: Optional[float] = None,
    ) -> Path:
        ffmpeg = find_ffmpeg()
        output.parent.mkdir(parents=True, exist_ok=True)
        list_file = output.parent / "concat_list.txt"
        # Use absolute paths escaped for concat demuxer
        lines = []
        for p in clip_paths:
            ap = p.resolve().as_posix().replace("'", "'\\''")
            lines.append(f"file '{ap}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        codec, use_gpu = self._video_codec(ffmpeg)
        acodec = str(self.ecfg.get("audio_codec", "aac"))
        crf = str(self.ecfg.get("crf", 18))
        preset = str(self.ecfg.get("preset", "medium"))
        if codec == "h264_nvenc" and preset not in {f"p{i}" for i in range(1, 8)}:
            preset = str(self.ecfg.get("nvenc_preset", "p4"))
        abitrate = str(self.ecfg.get("audio_bitrate", "192k"))
        pix = str(self.ecfg.get("pixel_format", "yuv420p"))

        # Re-encode on concat for timestamp continuity (default accurate path)
        args = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            *(("-hwaccel", "cuda", "-hwaccel_output_format", "cuda") if use_gpu else ()),
            "-i",
            str(list_file),
            "-map",
            "0:v:0",
            *(["-map", "0:a:0"] if has_audio else []),
            "-c:v",
            codec,
            "-preset",
            preset,
            *self._video_quality(codec, crf),
            *( ["-pix_fmt", pix] if not use_gpu else [] ),
            *(["-c:a", acodec, "-b:a", abitrate] if has_audio else ["-an"]),
            "-movflags",
            "+faststart",
            "-fflags",
            "+genpts",
            str(output),
        ]
        if expected_duration is None:
            expected_duration = sum(probe_video(path).duration for path in clip_paths)
        try:
            run_command(args, timeout=max(600.0, 60.0 * len(clip_paths)))
            self._verify_media(
                output,
                expected_duration=expected_duration,
                source_has_audio=has_audio,
                source_fps=source_fps,
                verify_timing=bool(self.ecfg.get("verify_sync", True)) or accurate,
            )
        except Exception:
            output.unlink(missing_ok=True)
            raise

        logger.info("Joined highlights → %s", output)
        return output

    @staticmethod
    def _verify_clip(
        path: Path,
        shot: ShotRecord,
        *,
        source_has_audio: bool,
        source_fps: float,
        verify_timing: bool = True,
    ) -> None:
        """Verify one encoded clip without changing its requested boundaries."""

        expected = shot.clip_end - shot.clip_start
        Exporter._verify_media(
            path,
            expected_duration=expected,
            source_has_audio=source_has_audio,
            source_fps=source_fps,
            verify_timing=verify_timing,
        )

    @staticmethod
    def _verify_media(
        path: Path,
        *,
        expected_duration: float,
        source_has_audio: bool,
        source_fps: float,
        verify_timing: bool = True,
    ) -> None:
        """Verify duration plus A/V stream alignment on a rendered media file."""

        meta = probe_video(path)
        if source_has_audio and not meta.has_audio:
            raise RuntimeError(f"Audio stream missing from exported clip {path}")
        if not verify_timing:
            return

        frame_tol = 1.0 / max(float(source_fps or 30.0), 1.0)
        # Encoders quantize presentation timestamps to whole video frames. A
        # non-frame-aligned strict boundary can therefore differ by up to two
        # frames even though ffmpeg honored the requested -ss/-t interval. Do
        # not reject that valid output as a false boundary shift.
        duration_tolerance_frames = 2.0
        if abs(meta.duration - expected_duration) > frame_tol * duration_tolerance_frames:
            raise RuntimeError(
                f"A/V duration mismatch for {path.name}: expected {expected_duration:.4f}s, "
                f"encoded {meta.duration:.4f}s (tolerance {frame_tol * duration_tolerance_frames:.4f}s)"
            )

        if not source_has_audio:
            return

        raw = ffprobe_json(path)
        video = next(
            (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        audio = next(
            (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"),
            None,
        )
        if video is None or audio is None:
            raise RuntimeError(f"Expected audio/video streams in exported clip {path}")

        def _number(stream: dict, key: str) -> Optional[float]:
            try:
                value = stream.get(key)
                return float(value) if value not in (None, "N/A") else None
            except (TypeError, ValueError):
                return None

        video_start = _number(video, "start_time") or 0.0
        audio_start = _number(audio, "start_time") or 0.0
        sample_rate = _number(audio, "sample_rate") or 48000.0
        audio_packet = 1024.0 / max(sample_rate, 1.0)
        # NVENC may expose a small encoder-delay offset between the first video
        # frame and the first AAC packet. Keep the A/V alignment contract tight,
        # but allow the same bounded two-frame timestamp quantization as the
        # duration check above.
        sync_tolerance = max(frame_tol, audio_packet) * 2.0
        if abs(video_start - audio_start) > sync_tolerance:
            raise RuntimeError(
                f"A/V start-time mismatch for {path.name}: video={video_start:.6f}s, "
                f"audio={audio_start:.6f}s"
            )

        video_duration = _number(video, "duration")
        audio_duration = _number(audio, "duration")
        if (
            video_duration is not None
            and audio_duration is not None
            and abs(video_duration - audio_duration) > sync_tolerance * 2.0
        ):
            raise RuntimeError(
                f"A/V stream-duration mismatch for {path.name}: "
                f"video={video_duration:.6f}s, audio={audio_duration:.6f}s"
            )

    def _write_csv(self, shots: list[ShotRecord], path: Path) -> Path:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "shot_id",
                    "clip_start",
                    "cue_strike",
                    "clip_end",
                    "duration",
                    "confidence",
                    "importance",
                    "possible_replay",
                    "review_required",
                    "included",
                ]
            )
            for s in shots:
                w.writerow(
                    [
                        s.shot_id,
                        f"{s.clip_start:.3f}",
                        f"{s.cue_strike:.3f}",
                        f"{s.clip_end:.3f}",
                        f"{s.duration():.3f}",
                        f"{s.shot_confidence:.3f}",
                        f"{s.importance:.3f}",
                        s.possible_replay,
                        s.manual_review_required,
                        s.included,
                    ]
                )
        return path

    def _write_edl(self, shots: list[ShotRecord], path: Path, source: str) -> Path:
        """Simple CMX3600-style EDL."""
        lines = ["TITLE: Snooker AI Export", "FCM: NON-DROP FRAME", ""]
        rec_in = 0.0
        for i, s in enumerate(shots, start=1):
            dur = s.duration()
            src_in = format_timestamp(s.clip_start).replace(".", ":")
            src_out = format_timestamp(s.clip_end).replace(".", ":")
            rec_out_t = rec_in + dur
            rec_in_s = format_timestamp(rec_in).replace(".", ":")
            rec_out_s = format_timestamp(rec_out_t).replace(".", ":")
            lines.append(
                f"{i:03d}  AX       V     C        {src_in} {src_out} {rec_in_s} {rec_out_s}"
            )
            lines.append(f"* FROM CLIP NAME: shot_{s.shot_id:04d}")
            lines.append(f"* SOURCE FILE: {source}")
            rec_in = rec_out_t
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_chapters(self, shots: list[ShotRecord], path: Path) -> Path:
        """FFmpeg metadata chapters for the joined export timeline."""
        lines = [";FFMETADATA1"]
        t = 0.0
        for s in shots:
            start_ms = int(round(t * 1000))
            end_ms = int(round((t + s.duration()) * 1000))
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={start_ms}")
            lines.append(f"END={end_ms}")
            lines.append(f"title=Shot {s.shot_id:04d} @ {s.cue_strike:.3f}s")
            t += s.duration()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_chapter_list(self, shots: list[ShotRecord], path: Path) -> Path:
        lines = ["# edited_time\tsource_start\tsource_end\tshot_id\tcue_strike"]
        t = 0.0
        for s in shots:
            lines.append(
                f"{t:.3f}\t{s.clip_start:.3f}\t{s.clip_end:.3f}\t{s.shot_id}\t{s.cue_strike:.3f}"
            )
            t += s.duration()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _training_labels(self, result: AnalysisResult, shots: list[ShotRecord]) -> dict:
        return {
            "source_path": result.source_path,
            "job_id": result.job_id,
            "version": result.analysis_version,
            "labels": [
                {
                    "shot_id": s.shot_id,
                    "preparation_start": s.preparation_start,
                    "cue_strike": s.cue_strike,
                    "ball_motion_start": s.ball_motion_start,
                    "ball_motion_end": s.ball_motion_end,
                    "reaction_end": s.clip_end,
                    "possible_replay": s.possible_replay,
                    "user_modified": s.user_modified,
                    "confidence": s.shot_confidence,
                }
                for s in shots
            ],
        }
