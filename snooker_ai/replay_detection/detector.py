"""Conservative replay association across broadcast camera changes.

A camera cut and a familiar motion curve are normal during live snooker.
Automatic duplicate exclusion therefore requires repeated ball positions and
their movement over time, or an explicit broadcast replay marker.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque

import numpy as np

from snooker_ai.config import Config
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.logging import get_logger

logger = get_logger("replay")
_REPLAY_VIEWS = (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY)


class ReplayDetector:
    # Preparation plus a moving trajectory: a still layout alone could be
    # another genuine live attempt.
    _LAYOUT_ANCHORS = (-0.5, 0.5, 1.5, 2.5)
    _OWNED_EVIDENCE = (
        "replay_explicit_view", "replay_match_to", "replay_signature_confirmed",
        "replay_layout_confirmed", "replay_layout_similarity",
        "replay_appearance_similarity", "replay_broadcast_marker",
        "replay_stinger_confirmed",
    )

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

        # Native refinement windows can arrive out of order. Associate on the
        # timeline while retaining the caller's candidate objects and ordering.
        if any(a.t > b.t for a, b in zip(features, features[1:])):
            features = sorted(features, key=lambda f: f.t)
        times = [f.t for f in features]
        for feature in features:
            if feature.replay_stinger_annotated:
                feature.broadcast_replay = feature.replay_stinger_original_marker
                feature.replay_stinger_annotated = False

        def feature_window(start: float, end: float) -> list[FrameFeatures]:
            return features[bisect_left(times, start):bisect_right(times, end)]

        ordered = sorted(candidates, key=lambda c: c.timestamp)
        stinger_intervals = self._stinger_intervals(features, ordered)
        for start, end in stinger_intervals:
            for feature in feature_window(start, end):
                feature.replay_stinger_original_marker = feature.broadcast_replay
                feature.replay_stinger_annotated = True
                feature.broadcast_replay = True
        signatures = [self._signature(c.timestamp, features, times=times) for c in ordered]
        layouts = [self._layout_sequence(c.timestamp, features, times) for c in ordered]
        live_candidates: deque[tuple[float, int]] = deque()
        for i, candidate in enumerate(ordered):
            # Re-evaluate flags from saved analyses made by earlier heuristics.
            # Flags set independently by callers are preserved.
            if any(key in candidate.evidence for key in self._OWNED_EVIDENCE):
                candidate.possible_replay = False
                candidate.evidence = {
                    key: value for key, value in candidate.evidence.items()
                    if key not in self._OWNED_EVIDENCE
                }

            near = feature_window(candidate.timestamp - 0.6, candidate.timestamp + 0.6)
            nearest = min(near, key=lambda f: abs(f.t - candidate.timestamp), default=None)
            marker = bool(nearest is not None and abs(nearest.t - candidate.timestamp) <= 0.35
                          and getattr(nearest, "broadcast_replay", False))
            # An explicit candidate view is a trusted caller annotation. Do not
            # infer it from a lone nearby frame: that might be a replay ending
            # just before the next live shot, or a normal colourful close-up.
            if marker or candidate.camera_view in _REPLAY_VIEWS:
                candidate.possible_replay = True
                candidate.evidence = {
                    **candidate.evidence,
                    "replay_explicit_view": 1.0,
                    "replay_broadcast_marker": float(marker),
                }
            if any(start <= candidate.timestamp <= end for start, end in stinger_intervals):
                candidate.possible_replay = True
                candidate.evidence.update(replay_stinger_confirmed=1.0, replay_signature_confirmed=1.0)

            while live_candidates and candidate.timestamp - live_candidates[0][0] > self.max_after:
                live_candidates.popleft()

            transition = any(
                f.scene_cut_score >= 0.5
                for f in feature_window(candidate.timestamp - 3.0, candidate.timestamp + 0.6)
            )
            if transition:
                # Most recent matches first; names/scoreboard appearance are
                # only supporting context and can never establish duplication.
                for live_time, j in reversed(live_candidates):
                    if candidate.timestamp - live_time < self.min_after:
                        continue
                    if self._similarity(signatures[i], signatures[j]) < self.sim_thr:
                        continue
                    layout_score = self._layout_repeat_score(layouts[i], layouts[j])
                    if layout_score <= 0.0:
                        continue
                    candidate.possible_replay = True
                    candidate.evidence = {
                        **candidate.evidence,
                        "replay_match_to": live_time,
                        "replay_signature_confirmed": 1.0,
                        "replay_layout_confirmed": 1.0,
                        "replay_layout_similarity": layout_score,
                    }
                    break

            if not candidate.possible_replay and candidate.confidence >= 0.40:
                live_candidates.append((candidate.timestamp, i))

        n_replay = sum(1 for c in candidates if c.possible_replay)
        logger.info("Marked %d/%d candidates as possible replays", n_replay, len(candidates))
        return candidates

    def _stinger_intervals(
        self, features: list[FrameFeatures], candidates: list[StrikeCandidate],
    ) -> list[tuple[float, float]]:
        groups: list[list[FrameFeatures]] = []
        for feature in features:
            if len(feature.appearance_signature) not in (192, 193):
                continue
            if (groups and feature.t-groups[-1][-1].t <= .8
                    and len(feature.appearance_signature) == len(groups[-1][-1].appearance_signature)):
                groups[-1].append(feature)
            else:
                groups.append([feature])
        intervals = []
        index = 0
        while index + 1 < len(groups):
            opening, closing = groups[index:index+2]
            index += 1
            start, end = opening[0].t, closing[-1].t
            if not 2 <= closing[0].t-opening[-1].t <= 20:
                continue
            if not any(self._cosine(np.asarray(a.appearance_signature), np.asarray(b.appearance_signature)) >= .92
                       for a in opening for b in closing):
                continue
            # Pair ownership is graphical evidence. A replay contact may be
            # occluded or already filtered from native candidates; its closing
            # wipe still cannot open a package spanning the next live shot.
            index += 1
            if not any(c.confidence >= .40 and not c.possible_replay
                       and start-self.max_after <= c.timestamp < start-self.min_after
                       and not any(lo <= c.timestamp <= hi for lo, hi in intervals)
                       for c in candidates):
                continue
            intervals.append((start, end))
            # A closing wipe cannot also open the next replay package. Reusing
            # it would incorrectly erase live play between consecutive replays.
        return intervals

    def returning_live_candidates(
        self,
        candidates: list[StrikeCandidate],
        features: list[FrameFeatures],
        completed_live_stops: dict[float, float],
    ) -> list[StrikeCandidate]:
        """Recover newly visible live rolls whose contact was hidden by a wipe.

        These are additional shot-in-progress proposals, not exact cue strikes.
        A verified stop before the replay opening prevents the same ongoing live
        shot from being counted again when the broadcast returns to it. Inputs
        remain untouched; the caller can merge these proposals into its timeline.
        """
        if not self.enabled or not completed_live_stops:
            return []
        ordered_features = sorted(features, key=lambda f: f.t)
        ordered_candidates = sorted(candidates, key=lambda c: c.timestamp)
        intervals = self._stinger_intervals(ordered_features, ordered_candidates)
        times = [f.t for f in ordered_features]
        recovered: list[StrikeCandidate] = []

        def reliable_cue(f: FrameFeatures) -> bool:
            return bool(
                f.observation_fps >= 10 and f.table_full_view and f.table_observable
                and f.match_context_valid and not f.table_handling and not f.broadcast_replay
                and f.view_type not in _REPLAY_VIEWS and not f.appearance_signature
                and f.cue_ball_detected and f.cue_ball_track_confidence >= .70
                and f.ball_diameter_px > .5
                and f.cue_ball_x is not None and f.cue_ball_y is not None
                and np.isfinite(f.cue_ball_x) and np.isfinite(f.cue_ball_y)
            )

        def speed(f: FrameFeatures) -> float:
            value = f.cue_ball_stable_normalized_speed
            return float(value) if value is not None and np.isfinite(value) else 0.0

        for opening_start, closing_end in intervals:
            prior = next((c for c in reversed(ordered_candidates)
                          if c.timestamp < opening_start and c.confidence >= .40
                          and not c.possible_replay
                          and not any(lo <= c.timestamp <= hi for lo, hi in intervals)), None)
            if prior is None:
                continue
            prior_stop = completed_live_stops.get(prior.timestamp)
            if (prior_stop is None or not np.isfinite(prior_stop)
                    or not prior.timestamp <= prior_stop < opening_start):
                continue
            # The opening/closing association has already been confirmed. Find
            # the first retained image of its closing graphic for the uncertainty
            # interval, without treating a graphic as an observed contact.
            graphics = [f for f in ordered_features[bisect_left(times, opening_start):bisect_right(times, closing_end)]
                        if len(f.appearance_signature) in (192, 193)]
            closing_start = closing_end
            for graphic in reversed(graphics):
                if closing_start-graphic.t > .8:
                    break
                closing_start = graphic.t
            window = ordered_features[bisect_right(times, closing_end):bisect_right(times, closing_end+.60)]
            first = next((f for f in window if reliable_cue(f)), None)
            if first is None:
                continue
            if first.observation_valid and speed(first) < .75:
                # Only an explicit reset image has an unmeasured warmup speed.
                # A trustworthy quiet white shows preparation after the return.
                continue
            if any(f.table_handling or not f.match_context_valid or f.broadcast_replay
                   or f.view_type in _REPLAY_VIEWS for f in window if f.t <= first.t):
                continue
            existing = ordered_candidates + recovered
            if any(not c.possible_replay and abs(c.timestamp-first.t) <= 1.0
                   and not any(lo <= c.timestamp <= hi for lo, hi in intervals)
                   and (c.evidence.get("cue_ball_motion_confirmed", 0) >= .5
                        or c.evidence.get("dense_transition_confirmed", 0) >= .5)
                   for c in existing):
                continue

            # Returning wipes can produce consecutive tracker resets while the
            # transparent graphic clears. Only this short initial warmup may
            # contain cuts. All movement proof belongs to the final returned view.
            anchor = first
            warmup = [first]
            proof = [anchor]
            valid_count = 0
            previous = first
            failed = False
            proof_window = ordered_features[bisect_left(times, first.t):bisect_right(times, first.t+.35)]
            for f in proof_window:
                if f.t <= first.t:
                    continue
                if f.t-first.t > .35:
                    break
                gap_limit = 1.75 / min(previous.observation_fps, f.observation_fps) if f.observation_fps >= 10 else 0.0
                if not reliable_cue(f) or f.t-previous.t > gap_limit+1e-9:
                    failed = True
                    break
                changed = f.camera_scene_id != anchor.camera_scene_id
                if not f.observation_valid or f.scene_cut_score >= .42 or changed:
                    if valid_count or f.t-first.t > .10:
                        failed = True
                        break
                    warmup.append(f)
                    anchor = f
                    proof = [f]
                else:
                    # A quiet native image followed by a later launch is ordinary
                    # preparation, not movement already underway behind the wipe.
                    if valid_count == 0 and speed(f) < .75:
                        failed = True
                        break
                    proof.append(f)
                    valid_count += 1
                    if valid_count == 3:
                        break
                previous = f
            if failed or valid_count < 3:
                continue
            diameter = float(np.median([f.ball_diameter_px for f in proof]))
            xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in proof], dtype=float)
            legs = np.diff(xy, axis=0)
            lengths = np.linalg.norm(legs, axis=1)
            net = xy[-1]-xy[0]
            distance = float(np.linalg.norm(net))
            consistency = distance / max(float(np.sum(lengths)), 1e-9)
            peak = max(speed(f) for f in proof[1:])
            if distance/diameter < .50 or consistency < .80 or peak < .75:
                continue
            if len(warmup) > 1:
                # Do not backdate onset across a camera-induced identity jump.
                # Reset images must remain close and move in the proven direction.
                warm_xy = np.array([(f.cue_ball_x, f.cue_ball_y) for f in warmup], dtype=float)
                warm_legs = np.diff(warm_xy, axis=0)
                if any(np.linalg.norm(leg) > 2*diameter
                       or np.dot(leg, net) < .80*np.linalg.norm(leg)*distance
                       for leg in warm_legs):
                    continue
            recovered.append(StrikeCandidate(
                timestamp=first.t, confidence=.70,
                uncertainty_start=closing_start, uncertainty_end=first.t,
                camera_view=proof[-1].view_type,
                evidence={
                    "return_from_replay_visible_roll": 1.0,
                    "contact_time_upper_bound": 1.0,
                    "cue_ball_motion_confirmed": 1.0,
                    "replay_return_closing_start": closing_start,
                    "replay_return_closing_end": closing_end,
                    "replay_return_proof_timestamp": proof[-1].t,
                    "replay_return_previous_live_contact": prior.timestamp,
                    "replay_return_previous_live_stop": float(prior_stop),
                    "cue_displacement_diameters": distance/diameter,
                    "cue_direction_consistency": consistency,
                    "post_peak_cue_speed": peak,
                    "track_confidence": min(f.cue_ball_track_confidence for f in proof),
                },
            ))
        return recovered

    @staticmethod
    def _cosine(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            return 0.0
        denom = float(np.linalg.norm(first)*np.linalg.norm(second))
        return float(np.dot(first, second)/denom) if denom > 1e-9 else 0.0

    @staticmethod
    def _usable_table_feature(feature: FrameFeatures) -> bool:
        return bool(
            feature.table_observable and feature.observation_valid
            and feature.scene_cut_score < 0.5
            and getattr(feature, "match_context_valid", True)
            and not getattr(feature, "table_handling", False)
            and not feature.rack_idle
        )

    def _layout_sequence(
        self, timestamp: float, features: list[FrameFeatures], times: list[float],
    ) -> list[np.ndarray | None]:
        sequence: list[np.ndarray | None] = []
        for offset in self._LAYOUT_ANCHORS:
            target = timestamp + offset
            lo = bisect_left(times, target - 0.35)
            hi = bisect_right(times, target + 0.35)
            available = [
                feature for feature in features[lo:hi]
                if self._usable_table_feature(feature)
                and getattr(feature, "table_full_view", True)
                and len(getattr(feature, "ball_layout_signature", [])) == 14
            ]
            if not available:
                sequence.append(None)
                continue
            nearest = min(available, key=lambda f: abs(f.t - target))
            sequence.append(np.asarray(nearest.ball_layout_signature, dtype=np.float32))
        return sequence

    @staticmethod
    def _layout_repeat_score(
        first: list[np.ndarray | None], second: list[np.ndarray | None],
    ) -> float:
        matched: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for index, (a, b) in enumerate(zip(first, second)):
            if a is None or b is None or a.size != 14 or b.size != 14:
                continue
            a_pairs, b_pairs = a.reshape(7, 2), b.reshape(7, 2)
            valid_a = np.all(np.isfinite(a_pairs) & (a_pairs >= 0) & (a_pairs <= 1), axis=1)
            valid_b = np.all(np.isfinite(b_pairs) & (b_pairs >= 0) & (b_pairs <= 1), axis=1)
            # Match presence as well as position; potting a colour changes the
            # layout even when the remaining colour spots are identical.
            if not np.array_equal(valid_a, valid_b) or not valid_a[0] or np.sum(valid_a) < 3:
                continue
            difference = np.abs(a_pairs[valid_a] - b_pairs[valid_a])
            if np.max(difference) > 0.025 or np.mean(difference) > 0.010:
                continue
            matched.append((index, a_pairs[0], b_pairs[0], float(np.mean(difference))))

        # Preparation and at least two later positions must repeat. A single
        # returning table shot cannot supply a cross-view duplicate fingerprint.
        if len(matched) < 3 or matched[0][0] != 0:
            return 0.0
        cue_path_a = np.array([entry[1] for entry in matched])
        cue_path_b = np.array([entry[2] for entry in matched])
        if (
            np.max(np.linalg.norm(cue_path_a - cue_path_a[0], axis=1)) < 0.04
            or np.max(np.linalg.norm(cue_path_b - cue_path_b[0], axis=1)) < 0.04
        ):
            return 0.0
        return max(0.0, 1.0 - float(np.mean([entry[3] for entry in matched])) / 0.025)

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
        # Cuts/handling/unknown views do not become motion evidence. Resampling
        # uses actual timestamps so missing frames do not compress time.
        samples = [f for f in features[lo:hi] if self._usable_table_feature(f)]
        if len(samples) < 2:
            return np.zeros(16, dtype=np.float32)
        values = np.array([f.motion_score for f in samples], dtype=np.float32)
        sample_times = np.array([f.t - t for f in samples], dtype=np.float32)
        target_times = np.linspace(0.0, window, 16)
        return np.interp(target_times, sample_times, values).astype(np.float32)

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
