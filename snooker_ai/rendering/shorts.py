"""Independent vertical shorts from saved live-shot selections; no AI calls."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
from PIL import Image, ImageDraw, ImageFont
from snooker_ai.utils.models import BaseModel, ConfigDict, Field, field_validator

from snooker_ai.config import Config
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.utils.work_control import WorkControl, WorkPaused, current_control, check_pause
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import AnalysisResult, ExportRequest, JobStatus, ShotRecord
from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg, run_command
from snooker_ai.utils.persistence import atomic_text

WIDTH, HEIGHT = 1080, 1920
PRESETS = {
    "classic": {
        "label": "Editor classic",
        "background": "#000000",
        "name_color": "#FFD700",
        "shot_color": "#FFFFFF",
        "video_zoom": 1.0,
        "video_y": 0.5,
        "title_y": 120,
        "name_size": 92,
        "shot_size": 72,
        "line_spacing": 16,
    },
    "snooker": {
        "label": "Snooker green",
        "background": "#0B1F14",
        "name_color": "#F5E96B",
        "shot_color": "#FFFFFF",
        "video_zoom": 1.0,
        "video_y": 0.5,
        "title_y": 120,
        "name_size": 92,
        "shot_size": 72,
        "line_spacing": 16,
    },
    "bright": {
        "label": "Bright yellow",
        "background": "#F2FF06",
        "name_color": "#06BF22",
        "shot_color": "#FFFFFF",
        "video_zoom": 1.5,
        "video_y": 0.5,
        "title_y": 155,
        "name_size": 162,
        "shot_size": 100,
        "line_spacing": 38,
    },
}


class ShortsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    player_names: str = Field(min_length=1, max_length=160)
    shot_names: dict[int, str] = Field(default_factory=dict)
    preset: Literal["classic", "snooker", "bright"] = "classic"
    background: str = "#000000"
    name_color: str = "#FFD700"
    shot_color: str = "#FFFFFF"
    video_zoom: float = Field(default=1, ge=0.5, le=2)
    video_y: float = Field(default=0.5, ge=0, le=1)
    title_y: int = Field(default=120, ge=0, le=1650)
    name_size: int = Field(default=92, ge=32, le=180)
    shot_size: int = Field(default=72, ge=24, le=140)
    line_spacing: int = Field(default=16, ge=0, le=80)

    @field_validator("player_names")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("Enter the player name or match names")
        return value

    @field_validator("shot_names")
    @classmethod
    def clean_overrides(cls, value: dict[int, str]) -> dict[int, str]:
        if len(value) > 10000:
            raise ValueError("At most 10000 shot names are allowed")
        if any(len(name) > 160 for name in value.values()):
            raise ValueError("Shot names must be at most 160 characters")
        return {key: cls.clean_name(name) for key, name in value.items() if name.strip()}

    @field_validator("background", "name_color", "shot_color")
    @classmethod
    def color(cls, value: str) -> str:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError("Use a six-digit hex color")
        return value.upper()


def shot_title(settings: ShortsSettings, shot_id: int) -> str:
    return f"{settings.shot_names.get(shot_id, settings.player_names)} - Shot {shot_id:02d}"


def shot_filename(settings: ShortsSettings, shot_id: int) -> str:
    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]', "_", settings.shot_names.get(shot_id, settings.player_names)
    )
    name = name.strip(" .")[:100] or "Player"
    # The suffix also makes Windows reserved names safe and keeps IDs distinct.
    return f"{name} - Shot {shot_id:02d}.mp4"


def geometry(size: tuple[int, int], settings: ShortsSettings) -> tuple[int, int, int, int]:
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("Source video dimensions are invalid")
    scale = min(WIDTH / w, HEIGHT / h) * settings.video_zoom
    fit_w, fit_h = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
    return fit_w, fit_h, (WIDTH - fit_w) // 2, round((HEIGHT - fit_h) * settings.video_y)


def _font(size: int):
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def title_overlay(settings: ShortsSettings, shot_id: int) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    lines = [
        (
            settings.shot_names.get(shot_id, settings.player_names),
            settings.name_size,
            settings.name_color,
        ),
        (f"Shot {shot_id:02d}", settings.shot_size, settings.shot_color),
    ]
    y = settings.title_y
    for text, requested, color in lines:
        size = requested
        font = _font(size)
        while draw.textbbox((0, 0), text, font=font, stroke_width=4)[2] > WIDTH - 96 and size > 8:
            size -= 1
            font = _font(size)
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=4)
        x = (WIDTH - (bounds[2] - bounds[0])) // 2 - bounds[0]
        draw.text((x + 3, y - bounds[1] + 3), text, font=font, fill="#00000099", stroke_width=4)
        draw.text(
            (x, y - bounds[1]), text, font=font, fill=color, stroke_width=4, stroke_fill="#000000"
        )
        y += bounds[3] - bounds[1] + settings.line_spacing
    return image


def preview_png(result: AnalysisResult, settings: ShortsSettings, shot: ShotRecord) -> bytes:
    cap = cv2.VideoCapture(result.source_path)
    try:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            max(shot.clip_start, min(shot.cue_strike, shot.clip_end - 0.05)) * 1000,
        )
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Could not read this shot from the source video")
    finally:
        cap.release()
    source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h, x, y = geometry(source.size, settings)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), settings.background)
    canvas.paste(source.resize((w, h), Image.Resampling.BICUBIC), (x, y))
    overlay = title_overlay(settings, shot.shot_id)
    canvas.paste(overlay, (0, 0), overlay)
    canvas.thumbnail((432, 768), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def selected_shots(result: AnalysisResult, config: Config) -> list[ShotRecord]:
    shots = Exporter(config)._filter_shots(result.shots, ExportRequest())
    shots.sort(key=lambda shot: (shot.clip_start, shot.shot_id))
    if not shots:
        raise ValueError("Include at least one live shot in the editor first")
    if len({shot.shot_id for shot in shots}) != len(shots):
        raise ValueError("Shot numbers must be unique")
    duration = result.metadata.duration or result.original_duration
    for shot in shots:
        if (
            not all(math.isfinite(t) for t in (shot.clip_start, shot.clip_end))
            or not 0 <= shot.clip_start < shot.clip_end <= duration + 0.05
        ):
            raise ValueError(
                f"Shot {shot.shot_id} has invalid saved boundaries; fix its trim in the editor"
            )
    return shots


def selection_key(result: AnalysisResult, shots: list[ShotRecord]) -> str:
    return hashlib.sha256(
        json.dumps(
            [result.source_path, [[s.shot_id, s.clip_start, s.clip_end] for s in shots]],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def render_short(
    result: AnalysisResult, shot: ShotRecord, settings: ShortsSettings, output: Path, config: Config
) -> None:
    ffmpeg = find_ffmpeg()
    duration = shot.clip_end - shot.clip_start
    fps = min(60, max(1, result.metadata.fps or 30))
    overlay = output.with_suffix(".png")
    title_overlay(settings, shot.shot_id).save(overlay)
    size = (result.metadata.width, result.metadata.height)
    if abs(result.metadata.rotation) % 180 == 90:
        size = size[::-1]
    w, h, x, y = geometry(size, settings)
    graph = (
        f"[0:v:0]scale={w}:{h}:flags=bicubic,setsar=1,fps={fps}[clip];"
        f"color=c={settings.background}:s={WIDTH}x{HEIGHT}:r={fps}:d={duration}[bg];"
        f"[bg][clip]overlay={x}:{y}:shortest=1[base];"
        "[base][1:v]overlay=0:0:format=auto,format=yuv420p[v]"
    )
    exporter = Exporter(config)
    codec, _ = exporter._video_codec(ffmpeg)
    temporary = output.with_suffix(".partial.mp4")
    base = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(shot.clip_start),
        "-i",
        result.source_path,
        "-i",
        str(overlay),
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "0:a:0?",
        "-t",
        str(duration),
    ]

    def encode(selected_codec):
        run_command(
            base
            + [
                "-c:v",
                selected_codec,
                "-preset",
                "fast",
                *exporter._video_quality(selected_codec, "20"),
                "-threads",
                "4",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )

    try:
        try:
            encode(codec)
        except RuntimeError:
            if codec != "h264_nvenc":
                raise
            encode("libx264")
        raw = ffprobe_json(temporary)
        video = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
        actual_duration = float(video.get("duration", raw.get("format", {}).get("duration", 0)))
        if (video.get("width"), video.get("height")) != (WIDTH, HEIGHT) or abs(
            actual_duration - duration
        ) > max(0.2, 2 / fps):
            raise RuntimeError(f"Shot {shot.shot_id} did not render completely; resume to retry it")
        if result.metadata.has_audio and not any(
            s.get("codec_type") == "audio" for s in raw.get("streams", [])
        ):
            raise RuntimeError(f"Original audio is missing from shot {shot.shot_id}")
        temporary.replace(output)
    finally:
        overlay.unlink(missing_ok=True)


class ShortsManager:
    """One render at a time, immutable snapshots and a durable checkpoint per clip."""

    def __init__(self, config: Config, store: JobStore):
        self.config, self.store = config, store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shorts")
        self.stopping = threading.Event()
        self.pending = {}
        self.pending_lock = threading.RLock()
        self.indexed = False

    def root(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise ValueError("Invalid job ID")
        self.store.get_meta(job_id)
        return self.store.job_dir(job_id) / "export" / "shorts"

    def directory(self, job_id: str, batch_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{16}", batch_id):
            raise FileNotFoundError("Shorts batch not found")
        return self.root(job_id) / batch_id

    def read(self, job_id: str, batch_id: str) -> dict:
        directory = self.directory(job_id, batch_id)
        batch = self.read_manifest(directory)
        if batch["status"] in {"queued", "rendering"}:
            lease = JobLease(self.root(job_id), ".shorts.lock")
            if lease.acquire():
                try:
                    # The worker may have completed after our first read. Re-read
                    # while holding its lease before declaring work abandoned.
                    batch = self.read_manifest(directory)
                    if batch["status"] in {"queued", "rendering"}:
                        batch.update(
                            status="interrupted",
                            message="Shorts stopped. Resume to keep completed clips and finish the rest.",
                        )
                        self.save(directory, batch)
                finally:
                    lease.release()
        batch["pause_requested"] = (
            batch["status"] in {"queued", "rendering"} and WorkControl(directory).requested()
        )
        return batch

    @staticmethod
    def read_manifest(directory: Path) -> dict:
        # NTFS/antivirus can briefly deny opening a file during atomic replace.
        for attempt in range(8):
            try:
                return json.loads((directory / "manifest.json").read_text("utf-8"))
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05, 0.005 * 2**attempt))

    def save(self, directory: Path, batch: dict):
        batch["updated_at"] = time.time()
        atomic_text(directory / "manifest.json", json.dumps(batch, indent=2))
        self.update_index(batch)

    def update_index(self, batch):
        index = self.store.root / ".work" / f"shorts-{batch['job_id']}-{batch['batch_id']}.json"
        if batch["status"] in {"queued", "rendering", "paused", "interrupted"}:
            if not index.exists():
                atomic_text(
                    index, json.dumps({"job_id": batch["job_id"], "batch_id": batch["batch_id"]})
                )
        else:
            for attempt in range(8):
                try:
                    index.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(min(.05, .005 * 2**attempt))

    def work_items(self) -> list[dict]:
        with self.pending_lock:
            if not self.indexed:
                for path in self.store.root.glob("*/export/shorts/*/manifest.json"):
                    batch = self.read_manifest(path.parent)
                    if batch["status"] in {"queued", "rendering", "paused", "interrupted"}:
                        self.update_index(self.read(batch["job_id"], batch["batch_id"]))
                self.indexed = True
        items = []
        for path in (self.store.root / ".work").glob("shorts-*.json"):
            try:
                entry = json.loads(path.read_text("utf-8"))
                batch = self.read(entry["job_id"], entry["batch_id"])
                if batch["status"] in {"queued", "rendering", "paused", "interrupted"}:
                    items.append(batch)
                else:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
        return items

    def latest(self, job_id: str) -> dict | None:
        manifests = list(self.root(job_id).glob("*/manifest.json"))
        if not manifests:
            return None
        path = max(manifests, key=lambda item: item.stat().st_mtime_ns)
        return self.read(job_id, path.parent.name)

    def load_source(self, job_id: str) -> tuple[AnalysisResult, list[ShotRecord]]:
        meta = self.store.get_meta(job_id)
        if meta.get("status") not in {JobStatus.COMPLETED.value, JobStatus.READY_FOR_REVIEW.value}:
            raise ValueError("Finish shot detection before making Shorts")
        self.store.validate_saved_source(job_id)
        result = self.store.load_selection_analysis(job_id)
        if Path(result.source_path).resolve() != Path(meta["source_path"]).resolve():
            raise ValueError("Saved shots do not match this source video")
        return result, selected_shots(result, self.config)

    def submit(self, job_id: str, settings: ShortsSettings) -> dict:
        if self.stopping.is_set():
            raise ValueError("The server is stopping; retry after restarting it")
        lease = JobLease(self.root(job_id), ".shorts.lock")
        if not lease.acquire():
            raise ValueError(
                "Shorts are already queued or rendering for this video. Follow their progress below."
            )
        try:
            result, shots = self.load_source(job_id)
            if set(settings.shot_names) - {shot.shot_id for shot in shots}:
                raise ValueError("Player name overrides must refer to included live shots")
            fingerprint = self.store.source_fingerprint(result.source_path)
            signature = hashlib.sha256(
                json.dumps(
                    [
                        1,
                        settings.model_dump(mode="json"),
                        fingerprint,
                        selection_key(result, shots),
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
            directory = self.directory(job_id, signature)
            directory.mkdir(parents=True, exist_ok=True)
            manifest = directory / "manifest.json"
            if manifest.exists():
                batch = self.read_manifest(directory)
                batch["clips"] = [
                    clip for clip in batch["clips"] if self.clip_valid(directory, clip)
                ]
                if (
                    batch["status"] == "completed"
                    and len(batch["clips"]) == len(shots)
                    and (directory / "shorts.zip").is_file()
                ):
                    lease.release()
                    return batch
            else:
                batch = {
                    "batch_id": signature,
                    "job_id": job_id,
                    "created_at": time.time(),
                    "settings": settings.model_dump(mode="json"),
                    "selection_key": selection_key(result, shots),
                    "source_fingerprint": fingerprint,
                    "total": len(shots),
                    "clips": [],
                }
            # Keep only the compact immutable source/selection snapshot.
            snapshot = result.model_copy(update={"shots": shots, "features": []})
            atomic_text(directory / "snapshot.json", snapshot.model_dump_json())
            return self.enqueue(directory, batch, lease)
        except BaseException:
            lease.release()
            raise

    def enqueue(self, directory, batch, lease):
        WorkControl(directory).clear()
        batch.update(
            status="queued",
            error=None,
            pause_requested=False,
            completed=len(batch["clips"]),
            message="Waiting to render Shorts",
        )
        self.save(directory, batch)
        response = json.loads(json.dumps(batch))
        key = (batch["job_id"], batch["batch_id"])
        with self.pending_lock:
            future = self.pool.submit(self._run, directory, batch, lease)
            self.pending[key] = (future, lease)

            def finished(_future):
                with self.pending_lock:
                    self.pending.pop(key, None)

            future.add_done_callback(finished)
        return response

    def pause(self, job_id, batch_id):
        batch = self.read(job_id, batch_id)
        if batch["status"] == "paused":
            return batch
        if batch["status"] not in {"queued", "rendering"}:
            raise ValueError("This Shorts batch is not running")
        directory = self.directory(job_id, batch_id)
        WorkControl(directory).request()
        with self.pending_lock:
            entry = self.pending.get((job_id, batch_id))
            if entry and entry[0].cancel():
                batch.update(status="paused", message="Shorts paused. Resume to continue.")
                self.save(directory, batch)
                entry[1].release()
        return self.read(job_id, batch_id)

    def load_batch_source(self, job_id, batch_id):
        batch = self.read(job_id, batch_id)
        result = AnalysisResult.model_validate_json(
            (self.directory(job_id, batch_id) / "snapshot.json").read_text("utf-8")
        )
        if self.store.source_fingerprint(result.source_path) != batch["source_fingerprint"]:
            raise ValueError("Source video changed; analyze it again before resuming Shorts")
        return result, selected_shots(result, self.config)

    def resume(self, job_id, batch_id):
        if self.stopping.is_set():
            raise ValueError("The server is stopping; retry after restart")
        directory = self.directory(job_id, batch_id)
        lease = JobLease(self.root(job_id), ".shorts.lock")
        if not lease.acquire():
            raise ValueError("Shorts are already queued or running for this video")
        try:
            batch = self.read_manifest(directory)
            if batch["status"] == "completed":
                lease.release()
                return batch
            self.load_batch_source(job_id, batch_id)
            batch["clips"] = [clip for clip in batch["clips"] if self.clip_valid(directory, clip)]
            return self.enqueue(directory, batch, lease)
        except BaseException:
            lease.release()
            raise

    @staticmethod
    def clip_valid(directory: Path, clip: dict) -> bool:
        path = directory / clip["filename"]
        try:
            stat = path.stat()
            return stat.st_size > 0 and [stat.st_size, stat.st_mtime_ns] == clip["file_identity"]
        except OSError:
            return False

    def _run(self, directory: Path, batch: dict, lease: JobLease):
        token = current_control.set(WorkControl(directory))
        try:
            check_pause()
            result = AnalysisResult.model_validate_json(
                (directory / "snapshot.json").read_text("utf-8")
            )
            settings = ShortsSettings.model_validate(batch["settings"])
            for shot in result.shots:
                check_pause()
                if self.stopping.is_set():
                    batch.update(
                        status="interrupted",
                        message="Shorts paused when the server stopped. Resume to finish.",
                    )
                    self.save(directory, batch)
                    return
                if self.store.source_fingerprint(result.source_path) != batch["source_fingerprint"]:
                    raise ValueError("Source video changed; analyze it again before making Shorts")
                if any(clip["shot_id"] == shot.shot_id for clip in batch["clips"]):
                    continue
                batch.update(
                    status="rendering",
                    message=f"Rendering shot {shot.shot_id:02d} — {len(batch['clips'])} of {batch['total']} ready",
                )
                self.save(directory, batch)
                filename = shot_filename(settings, shot.shot_id)
                path = directory / filename
                render_short(result, shot, settings, path, self.config)
                stat = path.stat()
                batch["clips"].append(
                    {
                        "shot_id": shot.shot_id,
                        "filename": filename,
                        "title": shot_title(settings, shot.shot_id),
                        "duration": shot.duration(),
                        "file_identity": [stat.st_size, stat.st_mtime_ns],
                    }
                )
                batch["completed"] = len(batch["clips"])
                self.save(directory, batch)
            check_pause()
            temporary = directory / "shorts.partial.zip"
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                for clip in batch["clips"]:
                    check_pause()
                    archive.write(directory / clip["filename"], arcname=clip["filename"])
            temporary.replace(directory / "shorts.zip")
            batch.update(
                status="completed", message=f"{len(batch['clips'])} Shorts ready", error=None
            )
            self.save(directory, batch)
        except WorkPaused:
            batch.update(
                status="paused",
                pause_requested=False,
                message="Shorts paused. Completed clips are saved; resume to continue.",
            )
            self.save(directory, batch)
        except Exception as exc:
            batch.update(
                status="failed",
                error=str(exc),
                message="Shorts stopped. Completed clips are saved; resume to retry.",
            )
            self.save(directory, batch)
        finally:
            current_control.reset(token)
            lease.release()

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True)
