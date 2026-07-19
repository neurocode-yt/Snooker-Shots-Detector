"""Onset / RMS / band energy features for cue-contact support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from snooker_ai.config import Config
from snooker_ai.utils.logging import get_logger

logger = get_logger("audio")


@dataclass
class AudioFeatures:
    times: np.ndarray  # seconds
    onset_env: np.ndarray  # 0..1-ish
    rms: np.ndarray
    highband: np.ndarray  # cue-like transient band
    midband: np.ndarray  # collision band
    sample_rate: int

    def value_at(self, t: float, series: np.ndarray) -> float:
        if self.times.size == 0:
            return 0.0
        idx = int(np.argmin(np.abs(self.times - t)))
        return float(series[idx])

    def peak_near(self, t: float, radius: float = 0.15) -> float:
        if self.times.size == 0:
            return 0.0
        mask = np.abs(self.times - t) <= radius
        if not np.any(mask):
            return self.value_at(t, self.onset_env)
        return float(np.max(self.onset_env[mask]))

    def cue_peaks(
        self,
        min_score: float = 0.30,
        min_distance: float = 1.2,
        max_peaks: int = 0,
    ) -> list[tuple[float, float, float, float]]:
        """Return isolated transient peaks suitable for recovery windows.

        The coarse visual pass is intentionally sparse, so a short cue impact
        can fall between its samples and never reach native-rate refinement.
        This method exposes the audio timeline as an independent proposal
        source.  It is deliberately conservative about spacing, but does not
        require a visual feature to exist at the same timestamp.

        Each tuple is ``(time, combined_score, onset_score, highband_score)``.
        Peaks are selected by descending strength and returned chronologically
        so nearby applause/commentary transients collapse to one window.
        """
        if self.times.size < 3:
            return []
        onset = np.asarray(self.onset_env, dtype=np.float32)
        high = np.asarray(self.highband, dtype=np.float32)
        score = np.maximum(onset, high)
        local_max = (
            (score[1:-1] >= score[:-2])
            & (score[1:-1] > score[2:])
            & (score[1:-1] >= float(min_score))
        )
        indices = np.flatnonzero(local_max) + 1
        if indices.size == 0:
            return []

        # Strongest-first NMS is more stable than taking the first peak in a
        # noisy burst, while the final chronological sort keeps callers simple.
        ranked = sorted(indices.tolist(), key=lambda idx: float(score[idx]), reverse=True)
        selected: list[int] = []
        separation = max(0.0, float(min_distance))
        for idx in ranked:
            if all(abs(float(self.times[idx] - self.times[other])) >= separation for other in selected):
                selected.append(idx)
                if max_peaks > 0 and len(selected) >= max_peaks:
                    break
        selected.sort()
        return [
            (
                float(self.times[idx]),
                float(score[idx]),
                float(onset[idx]),
                float(high[idx]),
            )
            for idx in selected
        ]


class AudioFeatureExtractor:
    def __init__(self, config: Config):
        cfg = config.section("audio")
        self.sr = int(cfg.get("sample_rate", 16000))
        self.hop = int(cfg.get("hop_length", 512))
        self.n_fft = int(cfg.get("n_fft", 2048))
        self.onset_delta = float(cfg.get("onset_delta", 0.15))
        band = cfg.get("cue_transient_band_hz", [2000, 8000])
        self.high_band = (float(band[0]), float(band[1]))
        mid = cfg.get("collision_band_hz", [500, 4000])
        self.mid_band = (float(mid[0]), float(mid[1]))
        self.max_weight = float(cfg.get("max_audio_weight", 0.25))

    def extract(self, audio_path: Optional[str | Path]) -> Optional[AudioFeatures]:
        if not audio_path or not Path(audio_path).is_file():
            logger.info("No audio file for feature extraction")
            return None
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("librosa is required for audio analysis") from exc

        logger.info("Extracting audio features from %s", audio_path)
        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        if y.size == 0:
            return None

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop, aggregate=np.median
        )
        # Normalise onset envelope
        if onset_env.max() > 0:
            onset_norm = onset_env / (onset_env.max() + 1e-8)
        else:
            onset_norm = onset_env

        rms = librosa.feature.rms(y=y, hop_length=self.hop, frame_length=self.n_fft)[0]
        if rms.max() > 0:
            rms = rms / (rms.max() + 1e-8)

        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        def band_energy(lo: float, hi: float) -> np.ndarray:
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            if idx.size == 0:
                return np.zeros(S.shape[1], dtype=np.float32)
            e = S[idx].mean(axis=0)
            if e.max() > 0:
                e = e / (e.max() + 1e-8)
            return e.astype(np.float32)

        high = band_energy(*self.high_band)
        mid = band_energy(*self.mid_band)
        times = librosa.frames_to_time(
            np.arange(len(onset_norm)), sr=sr, hop_length=self.hop
        )

        # Align lengths
        n = min(len(times), len(onset_norm), len(rms), len(high), len(mid))
        return AudioFeatures(
            times=times[:n],
            onset_env=onset_norm[:n].astype(np.float32),
            rms=rms[:n].astype(np.float32),
            highband=high[:n],
            midband=mid[:n],
            sample_rate=sr,
        )
