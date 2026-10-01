"""Proxy and audio must belong to the current source and sampling settings."""

from pathlib import Path

from snooker_ai.ingestion.proxy import generate_proxy
from snooker_ai.types import VideoMetadata
from snooker_ai.utils.ffmpeg import FFmpegError


def _setup(config, tmp_path, monkeypatch, gpu=False):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    output = tmp_path / "proxy"
    metadata = VideoMetadata(path=str(source), width=1920, height=1080, duration=5, fps=30, has_audio=True)
    proxy_metadata = VideoMetadata(path="proxy.mp4", width=960, height=540, duration=5, fps=30, has_audio=True)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"generated media")

    monkeypatch.setattr("snooker_ai.ingestion.proxy.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("snooker_ai.ingestion.proxy.acceleration_enabled", lambda cfg: gpu)
    monkeypatch.setattr("snooker_ai.ingestion.proxy.supports_encoder", lambda *args: gpu)
    monkeypatch.setattr("snooker_ai.ingestion.proxy.run_command", run)
    monkeypatch.setattr("snooker_ai.ingestion.probe.probe_video", lambda path: proxy_metadata)
    return source, output, metadata, calls, run


def test_proxy_reuses_only_matching_source_and_audio_rate(config, tmp_path, monkeypatch):
    config._data["proxy"]["extract_audio"] = True  # optional legacy WAV extraction
    source, output, metadata, calls, _ = _setup(config, tmp_path, monkeypatch)
    generate_proxy(source, output, metadata, config)
    assert len(calls) == 2
    calls.clear()
    generate_proxy(source, output, metadata, config)
    assert calls == []

    config._data["proxy"]["audio_sample_rate"] = 8000
    generate_proxy(source, output, metadata, config)
    assert len(calls) == 1
    assert calls[0][-1].endswith("audio.wav")
    calls.clear()
    source.write_bytes(b"a different video with the same dimensions and frame rate")
    generate_proxy(source, output, metadata, config)
    assert len(calls) == 2


def test_default_proxy_keeps_preview_sound_without_extracting_wav(config, tmp_path, monkeypatch):
    source, output, metadata, calls, _ = _setup(config, tmp_path, monkeypatch)
    result = generate_proxy(source, output, metadata, config)
    assert len(calls) == 1
    assert "-an" not in calls[0]
    assert result.audio_path is None
    assert not (output / "audio.wav").exists()


def test_proxy_records_cpu_fallback_and_reuses_it(config, tmp_path, monkeypatch):
    source, output, metadata, calls, run = _setup(config, tmp_path, monkeypatch, gpu=True)

    def fail_gpu(args, **kwargs):
        if "scale_cuda=960:540:format=nv12" in args:
            calls.append(args)
            raise FFmpegError("GPU unavailable")
        run(args, **kwargs)

    monkeypatch.setattr("snooker_ai.ingestion.proxy.run_command", fail_gpu)
    generate_proxy(source, output, metadata, config)
    assert (output / "proxy.backend").read_text() == "cpu"
    calls.clear()
    generate_proxy(source, output, metadata, config)
    assert calls == []
