from pathlib import Path

import cv2
import pytest

from snooker_ai.ingestion.probe import probe_video
from snooker_ai.rendering.mix import plan_mix, render_mix
from snooker_ai.types import ShotRecord
from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command


def shots():
    return [ShotRecord(shot_id=i + 1, clip_start=t, clip_end=t + 3, cue_strike=t + 1)
            for i, t in enumerate([0, 5, 10])]


def test_mix_keeps_cue_strikes_out_of_overlaps():
    items = shots()
    items[0].clip_end = 1.1
    plan = plan_mix(items, 25, .24)
    assert 0 <= plan.overlaps[0] <= .06
    assert plan.duration == pytest.approx(sum(plan.durations) - sum(plan.overlaps))
    assert plan.starts[1] == pytest.approx(plan.durations[0] - plan.overlaps[0])


@pytest.mark.parametrize("has_audio", [True, False])
def test_real_mix_blends_pixels_keeps_audio_and_matches_timing(tmp_path: Path, has_audio):
    ffmpeg = find_ffmpeg()
    source, output = tmp_path / "colors.mp4", tmp_path / "mixed.mp4"
    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for color in ["red", "blue", "white"]:
        args += ["-f", "lavfi", "-i", f"color={color}:s=160x90:r=25:d=5"]
    if has_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=15"]
    args += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
             *(["-map", "3:a", "-c:a", "aac"] if has_audio else ["-an"]),
             "-c:v", "libx264", "-preset", "ultrafast", str(source)]
    run_command(args, timeout=60)
    items = shots()
    plan = plan_mix(items, 25, .24)
    render_mix(str(source), items, output, plan=plan, ffmpeg=ffmpeg,
               video_args=["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"],
               has_audio=has_audio, audio_args=["-c:a", "aac"], batch_size=2)
    metadata = probe_video(output)
    assert metadata.has_audio is has_audio
    assert metadata.duration == pytest.approx(8.52, abs=.04)
    cap = cv2.VideoCapture(str(output))
    cap.set(cv2.CAP_PROP_POS_MSEC, 2880)
    ok, frame = cap.read()
    cap.release()
    assert ok
    blue, _, red = frame[45, 80]
    assert blue > 60 and red > 60  # actual blended frame, not a hard cut
    assert not list(tmp_path.glob(".mix-*"))
