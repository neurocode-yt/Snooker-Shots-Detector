"""Bounded-memory cross-dissolves that seek past omitted match intervals."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from snooker_ai.types import ShotRecord
from snooker_ai.utils.ffmpeg import run_command


@dataclass(frozen=True)
class MixPlan:
    fps: float
    durations: list[float]
    overlaps: list[float]
    starts: list[float]

    @property
    def duration(self) -> float:
        return sum(self.durations) - sum(self.overlaps)


def plan_mix(shots: list[ShotRecord], fps: float, seconds: float) -> MixPlan:
    if any(s.duration() <= 0 for s in shots):
        raise ValueError("Cannot mix a non-positive clip duration")
    fps = max(1.0, float(fps or 30))
    fps = float(round(fps)) if abs(fps - round(fps)) < 0.001 else float(Fraction(fps).limit_denominator(1001))
    frames = [max(1, int(s.duration() * fps + 1e-6)) for s in shots]
    durations = [n / fps for n in frames]
    overlaps = []
    for i, (left, right) in enumerate(zip(shots, shots[1:])):
        # Avoid covering cue impact on unusually short clips.
        after_impact = max(0, int((left.clip_start + durations[i] - left.cue_strike - 1 / fps) * fps + 1e-6))
        before_impact = max(0, int((right.cue_strike - right.clip_start - 1 / fps) * fps + 1e-6))
        n = min(max(0, round(seconds * fps)), frames[i] // 3, frames[i + 1] // 3,
                after_impact, before_impact)
        overlaps.append(n / fps)
    starts = [0.0] if shots else []
    for i in range(1, len(shots)):
        starts.append(starts[-1] + durations[i - 1] - overlaps[i - 1])
    return MixPlan(fps, durations, overlaps, starts)


def render_mix(
    source: str, shots: list[ShotRecord], output: Path, *, plan: MixPlan,
    ffmpeg: str, video_args: list[str], has_audio: bool, audio_args: list[str],
    batch_size: int = 8,
) -> None:
    """Render short bodies and overlaps in batches, then copy encoded video.

    Each input is accurately seeked to a kept interval. No decoder traverses
    a ten-minute break, and only a bounded number of streams is open at once.
    Temporary audio is PCM; AAC is encoded once to avoid priming gaps at joins.
    """
    pieces: list[list[tuple[float, float]]] = []
    for i, shot in enumerate(shots):
        head = plan.overlaps[i - 1] if i else 0.0
        tail = plan.overlaps[i] if i < len(plan.overlaps) else 0.0
        body = plan.durations[i] - head - tail
        if body > 1e-8:
            pieces.append([(shot.clip_start + head, body)])
        if tail > 0:
            pieces.append([
                (shot.clip_start + plan.durations[i] - tail, tail),
                (shots[i + 1].clip_start, tail),
            ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".mix-", dir=output.parent) as temp:
        root = Path(temp)
        batches = []
        for batch_index, offset in enumerate(range(0, len(pieces), max(1, batch_size))):
            group = pieces[offset:offset + max(1, batch_size)]
            args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
            filters, joins = [], []
            input_index = 0
            duration = sum(piece[0][1] for piece in group)
            for j, piece in enumerate(group):
                labels = []
                for start, length in piece:
                    k = input_index
                    args += ["-ss", f"{start:.9f}", "-t", f"{length:.9f}", "-i", source]
                    filters.append(f"[{k}:v]fps={plan.fps:.9f},settb=AVTB,setpts=PTS-STARTPTS,"
                                   f"trim=duration={length:.9f},format=yuv420p[v{k}]")
                    if has_audio:
                        filters.append(f"[{k}:a]asetpts=PTS-STARTPTS,aresample=48000,"
                                       f"apad,atrim=duration={length:.9f}[a{k}]")
                    labels.append(k)
                    input_index += 1
                if len(labels) == 2:
                    a, b = labels
                    d = piece[0][1]
                    filters.append(f"[v{a}][v{b}]xfade=transition=fade:duration={d:.9f}:offset=0,"
                                   f"trim=duration={d:.9f},setpts=PTS-STARTPTS[mv{j}]")
                    if has_audio:
                        filters.append(f"[a{a}][a{b}]acrossfade=d={d:.9f}:c1=tri:c2=tri[ma{j}]")
                    joins.append(f"[mv{j}]" + (f"[ma{j}]" if has_audio else ""))
                else:
                    k = labels[0]
                    joins.append(f"[v{k}]" + (f"[a{k}]" if has_audio else ""))
            filters.append(f"{''.join(joins)}concat=n={len(group)}:v=1:a={int(has_audio)}[outv]"
                           + ("[outa]" if has_audio else ""))
            batch = root / f"batch_{batch_index:05}.mkv"
            args += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
                     "-map", "[outv]", *(["-map", "[outa]"] if has_audio else []),
                     *video_args, "-pix_fmt", "yuv420p", "-r", f"{plan.fps:.9f}",
                     *(["-c:a", "pcm_s16le"] if has_audio else ["-an"]), str(batch)]
            run_command(args, timeout=max(300, duration * 15))
            batches.append(batch)
        listing = root / "concat.txt"
        listing.write_text("".join("file '" + p.as_posix().replace("'", "'\\''") + "'\n" for p in batches), encoding="utf-8")
        run_command([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                     "-i", str(listing), "-map", "0:v:0", "-c:v", "copy",
                     *(["-map", "0:a:0", *audio_args] if has_audio else ["-an"]),
                     "-movflags", "+faststart", str(output)], timeout=max(300, plan.duration * 2))
