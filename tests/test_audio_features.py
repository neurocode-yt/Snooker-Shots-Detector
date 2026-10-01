"""Audio lookup correctness and long-recording performance regressions."""

import numpy as np
import pytest

from snooker_ai.audio.features import AudioFeatures


def _audio(times):
    times = np.asarray(times, dtype=np.float64)
    values = np.arange(len(times), dtype=np.float64)
    return AudioFeatures(times, values, values, values, values, 16000)


def test_audio_lookups_match_full_scan_on_irregular_timestamps():
    rng = np.random.default_rng(123)
    audio = _audio(np.cumsum(rng.uniform(0.01, 0.08, 300)))
    for timestamp in np.linspace(-1, audio.times[-1] + 1, 80):
        nearest = np.argmin(np.abs(audio.times - timestamp))
        assert audio.value_at(timestamp, audio.rms) == audio.rms[nearest]
        for radius in (0.0, 0.12, 0.5):
            mask = np.abs(audio.times - timestamp) <= radius
            expected = np.max(audio.onset_env[mask]) if mask.any() else audio.onset_env[nearest]
            assert audio.peak_near(timestamp, radius) == expected


def test_audio_nearest_lookup_chooses_earlier_sample_on_tie():
    audio = _audio([0.0, 2.0, 4.0])
    assert audio.value_at(1.0, audio.rms) == 0.0
    assert audio.value_at(3.0, audio.rms) == 1.0


def test_audio_empty_lookup():
    audio = _audio([])
    assert audio.value_at(2.0, audio.rms) == 0.0
    assert audio.peak_near(2.0) == 0.0


def test_audio_lookup_does_not_scan_entire_long_recording(monkeypatch):
    audio = _audio(np.arange(450000) * 0.032)
    original_abs = np.abs
    sizes = []

    def measured_abs(array):
        sizes.append(array.size)
        return original_abs(array)

    monkeypatch.setattr(np, "abs", measured_abs)
    assert audio.value_at(6000.0, audio.rms) == pytest.approx(187500)
    audio.peak_near(6000.0, 0.12)
    assert sizes and max(sizes) < 20


def test_cue_peak_spacing_matches_strongest_first_reference():
    rng = np.random.default_rng(42)
    audio = _audio(np.cumsum(rng.uniform(0.02, 0.04, 1000)))
    audio.onset_env = rng.random(1000).astype(np.float32)
    audio.highband = np.zeros(1000, dtype=np.float32)
    score = audio.onset_env
    indices = np.flatnonzero(
        (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:]) & (score[1:-1] >= 0.3)
    ) + 1
    selected = []
    for idx in sorted(indices, key=lambda i: float(score[i]), reverse=True):
        if all(abs(audio.times[idx] - audio.times[other]) >= 1.2 for other in selected):
            selected.append(idx)
    assert [peak[0] for peak in audio.cue_peaks()] == [audio.times[i] for i in sorted(selected)]
