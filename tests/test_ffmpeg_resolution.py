"""Executable discovery skips stale FFmpeg shims earlier on PATH."""

import pytest

import snooker_ai.utils.ffmpeg as ffmpeg


@pytest.mark.parametrize(
    ("tool", "resolver"),
    [("ffmpeg", ffmpeg.find_ffmpeg), ("ffprobe", ffmpeg.find_ffprobe)],
)
def test_resolver_skips_incompatible_path_binary(tool, resolver, tmp_path, monkeypatch):
    stale = tmp_path / f"old-{tool}.exe"
    compatible = tmp_path / f"new-{tool}.exe"
    stale.touch()
    compatible.touch()

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: str(stale) if name == tool else None)
    monkeypatch.setattr(
        ffmpeg.os.path,
        "expandvars",
        lambda value: str(compatible) if "WinGet\\Links" in value else str(tmp_path / "missing"),
    )
    monkeypatch.setattr(ffmpeg.os.path, "isdir", lambda _path: False)
    monkeypatch.setattr(
        ffmpeg,
        "_supports_hide_banner",
        lambda binary: binary == str(compatible),
    )

    assert resolver() == str(compatible)
