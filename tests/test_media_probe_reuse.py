"""Probe reuse retains all timing failures and always reads fresh exported media."""
import copy

import pytest

from snooker_ai.ingestion.probe import probe_video
from snooker_ai.rendering.exporter import Exporter


def raw_probe():
    return {"format": {"duration": "2.0", "format_name": "mov,mp4", "nb_streams": 2}, "streams": [
        {"codec_type": "video", "width": 640, "height": 360, "avg_frame_rate": "25/1",
         "r_frame_rate": "25/1", "codec_name": "h264", "duration": "2.0", "start_time": "0.0"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2,
         "duration": "2.0", "start_time": "0.0"},
    ]}


def test_parsing_previously_read_probe_retains_every_metadata_field(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.touch()
    raw = raw_probe()
    monkeypatch.setattr("snooker_ai.ingestion.probe.ffprobe_json", lambda _path: raw)
    expected = probe_video(path)
    monkeypatch.setattr("snooker_ai.ingestion.probe.ffprobe_json", lambda _path: pytest.fail("must reuse provided probe"))
    assert probe_video(path, probe_data=raw) == expected
    assert raw == raw_probe()  # parsing does not mutate the caller's snapshot


@pytest.mark.parametrize("problem, error", [("missing_audio", "Audio stream missing"),
                                            ("duration", "duration mismatch"),
                                            ("offset", "start-time mismatch"),
                                            ("audio_duration", "stream-duration mismatch"),
                                            (None, None)])
def test_single_probe_preserves_all_export_timing_checks(tmp_path, monkeypatch, problem, error):
    path = tmp_path / "export.mp4"
    path.touch()
    raw = raw_probe()
    if problem == "missing_audio":
        raw["streams"].pop()
    elif problem == "duration":
        raw["format"]["duration"] = "3.0"
    elif problem == "offset":
        raw["streams"][1]["start_time"] = ".5"
    elif problem == "audio_duration":
        raw["streams"][1]["duration"] = "3.0"
    calls = []
    monkeypatch.setattr("snooker_ai.rendering.exporter.ffprobe_json", lambda _path: calls.append(_path) or copy.deepcopy(raw))
    monkeypatch.setattr("snooker_ai.ingestion.probe.ffprobe_json", lambda _path: pytest.fail("duplicate FFprobe process"))
    kwargs = dict(expected_duration=2, source_has_audio=True, source_fps=25)
    if error:
        with pytest.raises(RuntimeError, match=error):
            Exporter._verify_media(path, **kwargs)
    else:
        Exporter._verify_media(path, **kwargs)
    assert calls == [path]


def test_verification_reprobes_when_output_file_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "same-export.mp4"
    path.touch()
    responses = [raw_probe(), raw_probe()]
    responses[1]["streams"][1]["start_time"] = "1.0"
    monkeypatch.setattr("snooker_ai.rendering.exporter.ffprobe_json", lambda _path: responses.pop(0))
    kwargs = dict(expected_duration=2, source_has_audio=True, source_fps=25)
    Exporter._verify_media(path, **kwargs)
    path.write_bytes(b"replacement export")
    with pytest.raises(RuntimeError, match="start-time mismatch"):
        Exporter._verify_media(path, **kwargs)
    assert not responses
