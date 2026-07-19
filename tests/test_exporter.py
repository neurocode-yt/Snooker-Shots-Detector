"""Focused checks for strict clip rendering and A/V preservation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import snooker_ai.rendering.exporter as exporter_module
from snooker_ai.config import load_config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import (
    AnalysisResult,
    EditMode,
    ExportRequest,
    ShotRecord,
    VideoMetadata,
)
from snooker_ai.utils.ffmpeg import run_command


def _strict_shot(
    *,
    shot_id: int = 1,
    strike: float = 2.125,
    clip_start: float | None = None,
    physical_stop: float = 5.250,
    confirmation: float = 5.750,
    **updates,
) -> ShotRecord:
    clip_start = max(0.0, strike - 2.0) if clip_start is None else clip_start
    values = {
        "shot_id": shot_id,
        "cue_strike": strike,
        "clip_start": clip_start,
        "clip_end": max(physical_stop, strike + 4.0),
        "ball_motion_end": physical_stop,
        "last_ball_motion_timestamp": max(strike, physical_stop - 0.033333),
        "physical_stop_timestamp": physical_stop,
        "stop_confirmation_timestamp": confirmation,
        "shot_confidence": 0.9,
        "included": True,
    }
    values.update(updates)
    return ShotRecord(**values)


def _result(shot: ShotRecord, *, source: str = "input.mp4", duration: float = 10.0):
    return AnalysisResult(
        job_id="export-test",
        source_path=source,
        metadata=VideoMetadata(
            path=source,
            duration=duration,
            width=320,
            height=180,
            fps=30.0,
            has_audio=True,
        ),
        shots=[shot],
        mode=EditMode.STRICT,
        original_duration=duration,
    )


def test_strict_export_forces_accurate_and_keeps_confirmation_out(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = Exporter(config)
    shot = _strict_shot()
    calls: list[dict] = []

    def fake_export_clips(source, shots, clips_dir, **kwargs):
        calls.append(kwargs)
        path = clips_dir / "shot_0001.mp4"
        path.touch()
        return [path]

    monkeypatch.setattr(exporter, "_export_clips", fake_export_clips)
    out = exporter.export(
        _result(shot),
        tmp_path,
        ExportRequest(
            accurate=False,
            export_joined=False,
            export_csv=False,
            export_edl=False,
        ),
    )

    assert calls[0]["accurate"] is True
    payload = json.loads(out.metadata_path.read_text(encoding="utf-8"))
    assert payload["export_accurate"] is True
    assert payload["edited_duration"] == pytest.approx(
        shot.clip_end_timestamp - shot.clip_start_timestamp
    )
    assert payload["edited_duration"] != pytest.approx(
        shot.stop_confirmation_timestamp - shot.clip_start_timestamp
    )


def test_combined_only_export_does_not_leave_numbered_clips(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = Exporter(config)
    output_dir = tmp_path / "export"
    temporary_dirs: list[Path] = []

    def fake_export_clips(_source, _shots, clips_dir, **_kwargs):
        clips_dir = Path(clips_dir)
        temporary_dirs.append(clips_dir)
        part = clips_dir / "shot_0001.mp4"
        part.touch()
        return [part]

    def fake_concat(clip_paths, output, **_kwargs):
        assert clip_paths[0].parent.name.startswith(".combined-parts-")
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "concat_list.txt").write_text("temporary", encoding="utf-8")
        output.touch()
        return output

    monkeypatch.setattr(exporter, "_export_clips", fake_export_clips)
    monkeypatch.setattr(exporter, "_concat_clips", fake_concat)

    result = exporter.export(
        _result(_strict_shot()),
        output_dir,
        ExportRequest(
            export_clips=False,
            export_joined=True,
            export_csv=False,
            export_edl=False,
        ),
    )

    assert result.joined_path == output_dir / "highlights.mp4"
    assert result.joined_path.is_file()
    assert result.clip_paths == []
    assert temporary_dirs and not temporary_dirs[0].exists()
    assert list((output_dir / "clips").glob("shot_*.mp4")) == []
    assert not (output_dir / "concat_list.txt").exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"clip_start": 1.0, "clip_start_timestamp": 1.0},
            "violates strict start boundary",
        ),
        (
            {"clip_end": 5.75, "clip_end_timestamp": 5.75},
            "strict end boundary",
        ),
        (
            {"stop_confirmation_timestamp": 5.0},
            "confirmation precedes physical stop",
        ),
    ],
)
def test_strict_export_rejects_shifted_contract_boundaries(config, updates, message):
    shot = _strict_shot(**updates)
    with pytest.raises(ValueError, match=message):
        Exporter(config)._validate_strict_boundaries(
            [shot], source_duration=10.0, source_fps=30.0
        )


def test_replay_filter_is_conservative(config):
    live = _strict_shot(shot_id=1)
    flagged = _strict_shot(shot_id=2, possible_replay=True, included=True)
    linked = _strict_shot(shot_id=3, linked_live_shot_id=1, included=True)
    view_only = _strict_shot(shot_id=4, camera_views=["slow_motion_replay"])
    excluded_live = _strict_shot(shot_id=5, included=False)
    exporter = Exporter(config)

    default = exporter._filter_shots(
        [live, flagged, linked, view_only, excluded_live], ExportRequest()
    )
    assert [shot.shot_id for shot in default] == [1]

    with_replays = exporter._filter_shots(
        [live, flagged, linked, view_only, excluded_live],
        ExportRequest(include_replays=True),
    )
    assert [shot.shot_id for shot in with_replays] == [1, 2, 3, 4]


def test_accurate_command_keeps_sub_millisecond_precision_and_maps_audio(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = Exporter(config)
    shot = _strict_shot(
        strike=2.123456789,
        clip_start=1.123456789,
        physical_stop=3.987654321,
        confirmation=4.5,
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(exporter_module, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        commands.append(list(args))
        Path(args[-1]).touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exporter_module, "run_command", fake_run)
    monkeypatch.setattr(Exporter, "_verify_clip", staticmethod(lambda *args, **kwargs: None))

    clips = tmp_path / "clips"
    clips.mkdir()
    exporter._export_clips(
        "source.mp4",
        [shot],
        clips,
        accurate=True,
        source_has_audio=True,
        source_fps=30.0,
    )

    args = commands[0]
    assert args[args.index("-ss") + 1] == "1.123456789"
    assert args[args.index("-t") + 1] == f"{shot.duration():.9f}"
    assert [args[i + 1] for i, value in enumerate(args) if value == "-map"] == [
        "0:v:0",
        "0:a:0",
    ]
    assert "-c:a" in args
    assert "-an" not in args


def test_non_positive_interval_is_rejected_not_silently_extended(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exporter = Exporter(config)
    shot = _strict_shot(
        strike=2.0,
        clip_start=1.0,
        physical_stop=1.0,
        confirmation=1.0,
        last_ball_motion_timestamp=1.0,
        clip_end=1.0,
        clip_end_timestamp=1.0,
    )
    monkeypatch.setattr(exporter_module, "find_ffmpeg", lambda: "ffmpeg")
    clips = tmp_path / "clips"
    clips.mkdir()

    with pytest.raises(ValueError, match="non-positive export duration"):
        exporter._export_clips(
            "source.mp4",
            [shot],
            clips,
            accurate=True,
            source_has_audio=False,
        )


def test_audio_presence_is_checked_even_when_timing_verification_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "silent.mp4"
    path.touch()
    monkeypatch.setattr(
        exporter_module,
        "probe_video",
        lambda _path: SimpleNamespace(has_audio=False, duration=2.0),
    )

    with pytest.raises(RuntimeError, match="Audio stream missing"):
        Exporter._verify_media(
            path,
            expected_duration=2.0,
            source_has_audio=True,
            source_fps=30.0,
            verify_timing=False,
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)
def test_ffmpeg_strict_export_preserves_audio_and_excludes_confirmation_tail(
    tmp_path: Path,
):
    source = tmp_path / "source.mp4"
    run_command(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=4",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        timeout=60,
    )
    source_meta = probe_video(source)
    shot = _strict_shot(
        strike=2.5,
        clip_start=0.5,
        physical_stop=2.5,
        confirmation=3.0,
        clip_end=4.0,
        clip_end_timestamp=4.0,
    )
    result = _result(shot, source=str(source), duration=source_meta.duration)
    result.metadata = source_meta
    exporter = Exporter(
        load_config(overrides={"export": {"preset": "ultrafast", "crf": 28}})
    )

    output = exporter.export(
        result,
        tmp_path / "export",
        ExportRequest(
            accurate=False,
            export_joined=True,
            export_csv=False,
            export_edl=False,
        ),
    )

    assert len(output.clip_paths) == 1
    for path in [output.clip_paths[0], output.joined_path]:
        meta = probe_video(path)
        assert meta.has_audio is True
        assert meta.duration == pytest.approx(3.5, abs=1.0 / 24.0)
