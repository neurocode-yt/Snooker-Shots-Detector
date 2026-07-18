"""Phase 1 replay heuristics: graphics, timing, motion signature matching."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.logging import get_logger

logger = get_logger("replay")


class ReplayDetector:
    def __init__(self, config: Config):
        cfg = config.section("replay")
        self.enabled = bool(cfg.get("enabled", True))
        self.min_after = float(cfg.get("min_seconds_after_live", 1.0))
        self.max_after = float(cfg.get("max_seconds_after_live", 90.0))
        self.sim_thr = float(cfg.get("embedding_similarity", 0.88))

    def mark_candidates(
        self,
        candidates: list[StrikeCandidate],
        features: list[FrameFeatures],
    ) -> list[StrikeCandidate]:
        if not self.enabled or not candidates:
            return candidates

        times = [f.t for f in features]

        def feature_window(start: float, end: float) -> list[FrameFeatures]:
            return features[bisect_left(times, start) : bisect_right(times, end)]

        # Build simple motion signatures for each candidate
        sigs: list[np.ndarray] = []
        for c in candidates:
            sigs.append(self._signature(c.timestamp, features, times=times))

        live_candidates: list[tuple[float, int]] = []
        for i, c in enumerate(candidates):
            # Already flagged by view classifier
            near_features = feature_window(c.timestamp - 1.0, c.timestamp + 1.0)
            explicit_view = any(
                f.view_type in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY)
                for f in near_features
            )
            if explicit_view:
                c.possible_replay = True
                c.evidence = {**c.evidence, "replay_explicit_view": 1.0}

            # A repeated motion signature on the same uninterrupted table view
            # is not enough: consecutive live shots often look alike.  Require a
            # broadcast transition/scene-cut (or an explicit replay view) before
            # linking a signature as a replay.
            transition_near = any(
                f.scene_cut_score >= 0.5
                for f in feature_window(c.timestamp - 0.8, c.timestamp + 0.8)
            )

            # Temporal proximity: shortly after a high-confidence live strike
            for lt, j in live_candidates:
                dt = c.timestamp - lt
                if self.min_after <= dt <= self.max_after:
                    # Require strong signature match (was *0.7 → too many false replays)
                    if (
                        (transition_near or explicit_view)
                        and self._similarity(sigs[i], sigs[j]) >= self.sim_thr
                    ):
                        c.possible_replay = True
                        c.evidence = {
                            **c.evidence,
                            "replay_match_to": lt,
                            "replay_signature_confirmed": 1.0,
                        }

            if not c.possible_replay and c.confidence >= 0.40:
                live_candidates.append((c.timestamp, i))

        n_rep = sum(1 for c in candidates if c.possible_replay)
        logger.info("Marked %d/%d candidates as possible replays", n_rep, len(candidates))
        return candidates

    def _signature(
        self,
        t: float,
        features: list[FrameFeatures],
        window: float = 3.0,
        times: list[float] | None = None,
    ) -> np.ndarray:
        if times is None:
            times = [f.t for f in features]
        lo = bisect_left(times, t)
        hi = bisect_right(times, t + window)
        samples = [f.motion_score for f in features[lo:hi]]
        if not samples:
            return np.zeros(16, dtype=np.float32)
        # Resample to fixed length
        x = np.array(samples, dtype=np.float32)
        target = 16
        idx = np.linspace(0, len(x) - 1, target)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
