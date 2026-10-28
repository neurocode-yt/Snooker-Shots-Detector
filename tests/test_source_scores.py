"""Full-source observations must survive pauses and derived versions safely."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from snooker_ai.pot_detection import source_scores as scan
from snooker_ai.pot_detection.scoreboard import ScoreReading


def test_absent_source_cache_never_opens_a_legacy_source(config, tmp_path):
    assert scan.cached_source_scores(tmp_path / "missing.mp4", 10, config, tmp_path) is None


def test_full_resolution_scan_resumes_without_reusing_partial_or_changed_sources(
    config, tmp_path, monkeypatch
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video1")
    directory = tmp_path / "job"
    directory.mkdir()
    complete = False
    starts = []
    observed = []

    class Capture:
        def isOpened(self):
            return True

        def set(self, key, value):
            starts.append(value / 1000)

        def get(self, key):
            return 25

        def release(self):
            pass

    def frames(*args, **kwargs):
        for t in np.arange(kwargs["start_time"], 3, 0.5):
            if not complete and t == 1:
                raise RuntimeError("pause or interrupted decoding")
            yield 0, float(t), np.zeros((720, 1280, 3), np.uint8)

    class Reader:
        def __init__(self, *args):
            pass

        def read(self, image, t):
            observed.append(image.shape)
            return ScoreReading(t, int(t), 0, 0, 0.99, "paired_boxes")

    monkeypatch.setattr(scan, "open_capture", lambda *a, **kw: Capture())
    monkeypatch.setattr(scan, "sampled_frames", frames)
    monkeypatch.setattr(scan, "ScoreboardReader", Reader)
    monkeypatch.setattr(scan.ViewClassifier, "replay_stinger_signature", lambda image: [1.0] * 196)
    with pytest.raises(RuntimeError, match="interrupted"):
        scan.scan_source_scores(source, 3, config, directory)
    assert scan.cached_source_scores(source, 3, config, directory) is None
    saved = json.loads((directory / "source_scores.json").read_text())
    assert not saved["complete"] and saved["next_sample"] == 1
    complete = True
    found = scan.scan_source_scores(source, 3, config, directory)
    assert [r.t for r in found] == [0, 0.5, 1, 1.5, 2, 2.5]
    assert starts == [0, 1]
    assert all(shape == (720, 1280, 3) for shape in observed)
    assert [row["t"] for row in scan.cached_source_markers(source, 3, config, directory)] == [
        0,
        0.5,
        1,
        1.5,
        2,
        2.5,
    ]
    monkeypatch.setattr(
        scan, "open_capture", lambda *a, **kw: pytest.fail("complete cache decoded again")
    )
    assert [asdict(r) for r in scan.scan_source_scores(source, 3, config, directory)] == [
        asdict(r) for r in found
    ]
    assert scan.cached_source_scores(source, 4, config, directory) is None
    source.write_bytes(b"changed video")
    assert scan.cached_source_scores(source, 3, config, directory) is None
