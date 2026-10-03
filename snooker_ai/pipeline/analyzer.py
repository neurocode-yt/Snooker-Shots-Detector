"""
Main analysis pipeline.

Phase 1 flow:
  validate → proxy → sample frames → table/motion/scene features
  → strike fusion → replay filter → state machine → segments
"""

from __future__ import annotations

import hashlib
import json
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from snooker_ai.config import Config
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.event_fusion.rack_idle import RackIdleGate
from snooker_ai.ingestion.probe import validate_video
from snooker_ai.ingestion.proxy import generate_proxy
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.detector import SceneDetector, SceneObservation
from snooker_ai.scene_detection.broadcast_context import BroadcastContextGuard
from snooker_ai.scene_detection.table_context import (
    TableInteractionDetector, ViewGeometry, ball_layout, view_geometry,
)
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.table_detection.localizer import TableLocalizer, TableObservation
from snooker_ai.temporal_model.state_machine import ShotStateMachine
from snooker_ai.tracking.tracker import BallTracker
from snooker_ai.motion.residual import ResidualMotionAnalyzer
from snooker_ai.types import (
    AnalysisResult,
    CameraViewType,
    EditMode,
    FrameFeatures,
    JobStatus,
    SceneSegment,
    ShotRecord,
    StrikeCandidate,
    TimelineEvent,
)
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.acceleration import configure_acceleration
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.utils.video import open_capture, sampled_frames

logger = get_logger("pipeline")

ProgressCb = Callable[[float, str, str], None]
_CACHE_VERSION = 22


class Analyzer:
    def __init__(self, config: Config, job_dir: Path):
        self.config = config
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.acceleration = configure_acceleration(config)
        logger.info(
            "Analysis backend: %s (%s)",
            self.acceleration.backend,
            self.acceleration.device_name,
        )
        self.table = TableLocalizer(config)
        self.motion = ResidualMotionAnalyzer(config)
        self.scene_det = SceneDetector(config)
        self.strike_det = StrikeDetector(config)
        self.replay_det = ReplayDetector(config)
        self.state_machine = ShotStateMachine(config)
        self.segmenter = SegmentBuilder(config)
        self.objects = ObjectDetector(config)
        self.tracker = BallTracker()
        self._last_cue_tip: Optional[tuple[float, float, float]] = None
        self.broadcast_context = BroadcastContextGuard(config)
        self._coarse_context_reference: list[FrameFeatures] = []
        self._confirmed_target_returns: list[tuple[float, float]] = []
        self._detection_diagnostics: dict = {"stages": [], "native_proposals": []}

    def analyze(
        self,
        source: str | Path,
        job_id: str,
        mode: EditMode = EditMode.STRICT,
        progress: Optional[ProgressCb] = None,
        resume: bool = True,
        force_reanalyze: bool = False,
    ) -> AnalysisResult:
        def report(p: float, stage: str, msg: str = "") -> None:
            if progress:
                progress(p, stage, msg)
            logger.info("[%.0f%%] %s — %s", p * 100, stage, msg)

        checkpoint_path = self.job_dir / "checkpoint.json"
        analysis_path = self.job_dir / "analysis.json"
        source = Path(source)
        self.broadcast_context.reset_observations(preserve_target=False)
        self._coarse_context_reference = []
        self._confirmed_target_returns = []
        self._detection_diagnostics = {"stages": [], "native_proposals": []}
        result_signature = self._result_signature(source)
        prior_result = None

        # Resume completed analysis
        if resume and not force_reanalyze and analysis_path.exists():
            try:
                data = json.loads(analysis_path.read_text(encoding="utf-8"))
                result = AnalysisResult.model_validate(data)
                prior_result = result
                if result.analysis_signature == result_signature:
                    if result.mode != mode:
                        result = self._rebuild_segments(result, mode)
                        self._save_result(result)
                    report(1.0, JobStatus.READY_FOR_REVIEW.value, "Resumed from saved analysis")
                    return result
                logger.info("Saved analysis is stale; rebuilding with current detection settings")
            except Exception as exc:
                logger.warning("Could not resume analysis.json: %s", exc)

        report(0.02, JobStatus.VALIDATING.value, "Probing video")
        max_h = float(self.config.get("analysis.max_video_hours", 12.0))
        metadata = validate_video(source, max_hours=max_h)
        source = Path(metadata.path)
        analysis_signature = self._analysis_signature(source)

        report(0.08, JobStatus.PROXY.value, "Generating analysis proxy")
        proxy_dir = self.job_dir / str(self.config.get("paths.proxy_subdir", "proxy"))
        proxy = generate_proxy(source, proxy_dir, metadata, self.config)

        cached = self._load_coarse_cache(analysis_signature) if resume else None
        if cached is not None:
            features, scenes, candidates = cached
            if self._repair_pathological_replay_labels(features, scenes):
                self._annotate_scenes(features, scenes)
                features = self.strike_det.score_frames(features)
                features = self.state_machine.label(features)
                coarse_fps = float(self.config.get("analysis.sample_fps", 10.0))
                if coarse_fps <= 3.0:
                    candidates = self.strike_det.detect_sparse_candidates(features)
                else:
                    candidates = self.strike_det.detect_candidates(features)
                candidates = self.replay_det.mark_candidates(candidates, features)
                self._save_coarse_cache(
                    analysis_signature, features, scenes, candidates
                )
            report(
                0.82,
                JobStatus.DETECTING.value,
                f"Resumed {len(features)} coarse observations from checkpoint",
            )
        else:
            report(0.2, JobStatus.ANALYZING.value, "Extracting visual features")
            features, scene_observations, _ = self._extract_features(
                proxy.proxy_path,
                proxy.audio_path,
                proxy.mapper,
                metadata.duration,
                progress=lambda frac, msg: report(
                    0.2 + 0.45 * frac, JobStatus.ANALYZING.value, msg
                ),
                collect_scene_observations=True,
                checkpoint_stage="coarse_features",
            )

            report(0.68, JobStatus.DETECTING.value, "Detecting camera scenes")
            scenes = self.scene_det.detect_from_observations(
                scene_observations, metadata.duration
            )
            self._repair_pathological_replay_labels(features, scenes)
            self._annotate_scenes(features, scenes)

            report(0.75, JobStatus.DETECTING.value, "Scoring cue-strike candidates")
            features = self.strike_det.score_frames(features)
            features = self.state_machine.label(features)
            coarse_fps = float(self.config.get("analysis.sample_fps", 10.0))
            if coarse_fps <= 3.0:
                candidates = self.strike_det.detect_sparse_candidates(features)
            else:
                candidates = self.strike_det.detect_candidates(features)
            candidates = self.replay_det.mark_candidates(candidates, features)
            self._save_coarse_cache(
                analysis_signature, features, scenes, candidates
            )

        # Re-propose from observations even when resuming an older checkpoint:
        # commentary peaks saved by older versions are not shot candidates.
        candidates = self._visual_proposals(features)
        self._record_detection_stage("coarse_proposals", candidates, features)

        rack_waits = self._rack_wait_intervals(features)
        preparation_intervals = self._preparation_intervals(features)
        rack_reference = features
        self._coarse_context_reference = features
        # Only the interior of continuously observed waiting intervals is
        # suppressed. Keep their edges available for break-off recovery.
        candidates = [candidate for candidate in candidates if not any(
            start + 2 < candidate.timestamp < end - 2 for start, end in rack_waits
        ) and not any(
            start <= candidate.timestamp <= end for start, end in preparation_intervals
        )]
        for feature in features:
            if feature.rack_restart:
                candidates.append(StrikeCandidate(
                    timestamp=max(0.0, feature.t - 1), confidence=0.55,
                    uncertainty_start=max(0.0, feature.t - 4),
                    uncertainty_end=min(metadata.duration, feature.t + 1),
                    evidence={"rack_restart": 1.0},
                ))
        candidates = self._deduplicate_candidates(candidates)
        self._record_detection_stage("after_wait_filters", candidates)

        report(0.85, JobStatus.REFINING.value, "Refining strike boundaries")
        # Decode candidate windows at the configured refinement rate.  The same
        # dense observations are retained for stop detection, so strict ends are
        # frame-accurate rather than quantised to the coarse analysis cadence.
        candidates, dense_features = self._refine_candidate_windows(
            proxy.proxy_path,
            proxy.audio_path,
            proxy.mapper,
            metadata.duration,
            candidates,
            features,
            progress=lambda frac, msg: report(
                0.85 + 0.05 * frac, JobStatus.REFINING.value, msg
            ),
            signature=analysis_signature,
            resume=resume,
            existing_dense=(prior_result.features if prior_result is not None
                            and prior_result.analysis_feature_signature == analysis_signature else None),
        )
        if dense_features:
            self._annotate_scenes(dense_features, scenes)
            self._apply_match_context(dense_features, rack_reference)
            dense_features = self.strike_det.score_frames(dense_features)
            features = self._merge_feature_layers(features, dense_features)
            # Dense ranges may reveal a strike that the sparse proposal pass
            # skipped (for example when the 2fps cadence lands on the player
            # before and after contact).  Re-detect on the merged timeline and
            # union those discoveries with the original proposals before
            # snapping boundaries and building segments.
            dense_candidates = self.strike_det.detect_candidates(features)
            candidates = self._deduplicate_candidates(candidates + dense_candidates)
            candidates = self.strike_det.refine_boundaries(candidates, dense_features)
            # Sparse entries are proposals only.  Do not let a noisy proposal
            # become an exported shot unless the native-rate pass confirmed a
            # cue transition (or the explicit impact-occlusion fallback).
            retained: list[StrikeCandidate] = []
            for candidate in candidates:
                confirmed = (
                    candidate.evidence.get("dense_transition_confirmed", 0.0) >= 0.5
                    or candidate.evidence.get("sparse_dense_transition", 0.0) >= 0.5
                    or (
                        candidate.evidence.get("occlusion_inferred", 0.0) >= 0.5
                        and candidate.evidence.get("ball_onset_run", 0.0) >= 2.0
                    )
                )
                in_preparation = any(start <= candidate.timestamp <= end for start, end in preparation_intervals)
                if confirmed and not in_preparation and self._rack_candidate_supported(candidate, rack_reference):
                    retained.append(candidate)
            candidates = retained
            self._record_detection_stage("native_contact_confirmation", candidates, dense_features)

            # A contact can become supported only after independently decoded
            # views have been combined. Its original proposal then never ran
            # travel tracking. Sparse quiet rows cannot confirm a physical stop,
            # so recover those missing intervals before constructing clips.
            unresolved = self._unresolved_stop_candidates(candidates, features, metadata.duration)
            if unresolved:
                _, recovered = self._refine_candidate_windows(
                    proxy.proxy_path, proxy.audio_path, proxy.mapper,
                    metadata.duration, unresolved, rack_reference,
                    progress=lambda frac, msg: report(
                        0.90 + 0.02 * frac, JobStatus.REFINING.value, msg
                    ),
                    signature=f"{analysis_signature}:stop-recovery",
                    resume=resume, existing_dense=dense_features,
                )
                if recovered:
                    self._annotate_scenes(recovered, scenes)
                    self._apply_match_context(recovered, rack_reference)
                    dense_features = self._merge_feature_layers(dense_features, recovered)
                    self.strike_det.score_frames(dense_features)
                    features = self._merge_feature_layers(rack_reference, dense_features)
                    candidates = self._deduplicate_candidates(
                        candidates + self.strike_det.detect_candidates(features)
                    )
                    candidates = self.strike_det.refine_boundaries(candidates, dense_features)
                    candidates = [c for c in candidates if self.segmenter._candidate_supported(c)
                                  and not any(lo <= c.timestamp <= hi for lo, hi in preparation_intervals)
                                  and self._rack_candidate_supported(c, rack_reference)]

            candidates = self.replay_det.mark_candidates(candidates, features)
            candidates = self._recover_replay_returns(candidates, features, metadata.duration)
            self._record_detection_stage("replay_and_live_returns", candidates)
            features = self.state_machine.label(features)

        report(0.92, JobStatus.SEGMENTING.value, "Building shot segments")
        shots = self.segmenter.build(candidates, features, metadata.duration, mode)
        shots = self._preserve_user_edits(shots, job_id)
        shots = self._score_importance(shots, features)
        self._record_detection_stage("segmentation", candidates, extra={
            "shot_records": len(shots), "included_shots": sum(s.included for s in shots),
        })

        edited, removed = self.segmenter.recompute_durations(shots, metadata.duration)

        events: list[TimelineEvent] = [
            TimelineEvent(event_type="rack_wait", timestamp=start, end=end,
                          metadata={"reason": "stationary_racked_table"})
            for start, end in rack_waits if end - start >= 5
        ]
        events.extend(TimelineEvent(event_type="table_preparation", timestamp=start, end=end,
                                   metadata={"reason": "reds_returned_then_racked"})
                      for start, end in preparation_intervals)
        for s in shots:
            events.append(
                TimelineEvent(
                    event_type="cue_strike",
                    timestamp=s.cue_strike,
                    confidence=s.shot_confidence,
                    metadata={
                        "shot_id": s.shot_id,
                        "clip_start_timestamp": s.clip_start,
                        "strike_confidence": s.strike_confidence,
                    },
                )
            )
            events.append(
                TimelineEvent(
                    event_type="ball_stop",
                    timestamp=s.physical_stop_timestamp,
                    confidence=s.end_confidence,
                    metadata={
                        "shot_id": s.shot_id,
                        "last_ball_motion_timestamp": s.last_ball_motion_timestamp,
                        "physical_stop_timestamp": s.physical_stop_timestamp,
                        "stop_confirmation_timestamp": s.stop_confirmation_timestamp,
                        "stop_confidence": s.stop_confidence,
                    },
                )
            )

        # Store compact features (drop huge arrays — already lightweight)
        result = AnalysisResult(
            job_id=job_id,
            source_path=str(source),
            proxy_path=str(proxy.proxy_path),
            audio_path=str(proxy.audio_path) if proxy.audio_path else None,
            metadata=metadata,
            scenes=scenes,
            features=features,
            strike_candidates=candidates,
            shots=shots,
            events=events,
            mode=mode,
            original_duration=metadata.duration,
            edited_duration=edited,
            pause_removed_seconds=removed,
            analysis_signature=result_signature,
            analysis_feature_signature=analysis_signature,
        )
        self._save_result(result)
        if checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)

        report(1.0, JobStatus.READY_FOR_REVIEW.value, f"Detected {len(shots)} shots")
        return result

    def _visual_proposals(self, features: list[FrameFeatures]) -> list[StrikeCandidate]:
        """Rebuild proposals independently of audio and stale cached scores."""
        features = self.strike_det.score_frames(features)
        if float(self.config.get("analysis.sample_fps", 2.0)) <= 3.0:
            candidates = self.strike_det.detect_sparse_candidates(features)
            candidates.extend(self._coverage_proposals(features, candidates))
            candidates = self._deduplicate_candidates(candidates)
        else:
            candidates = self.strike_det.detect_candidates(features)
        return self.replay_det.mark_candidates(candidates, features)

    def _coverage_proposals(
        self, features: list[FrameFeatures], candidates: list[StrikeCandidate],
    ) -> list[StrikeCandidate]:
        """Cover localized table action which a sparse onset cannot measure.

        Coarse camera fitting can fail throughout an otherwise useful close-up,
        and a short roll can disappear between its two samples. Those failures
        must open a native verification window rather than veto its extraction.
        Ball/address evidence or a verified target return is still required;
        neither aggregate motion nor a camera cut confirms a strike here.
        """
        forward = max(.5, float(self.config.get("analysis.strike_refine_post_seconds", 2)))
        intervals = [self._contact_window(candidate) for candidate in candidates]
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        ends = [end for _, end in merged]
        proposed = []
        covered_until = -1.0
        previous_t = None
        previous_scene = None
        for frame in features:
            if (not frame.match_context_valid or frame.broadcast_replay or frame.table_handling
                    or frame.rack_idle or not frame.table_observable or frame.table_confidence < .25
                    or frame.view_type in {CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY,
                                           CameraViewType.ADVERTISEMENT, CameraViewType.SCOREBOARD,
                                           CameraViewType.AUDIENCE, CameraViewType.PLAYER_CLOSEUP}):
                continue
            if (previous_t is None or frame.t-previous_t > .76
                    or previous_scene != frame.camera_scene_id):
                covered_until = -1.0
            previous_t, previous_scene = frame.t, frame.camera_scene_id
            diameter = max(1.0, frame.ball_diameter_px)
            addressed = bool(frame.cue_ball_detected and frame.cue_ball_track_confidence >= .45
                             and frame.cue_tip_visible and frame.cue_tip_distance_to_ball <= 4*diameter)
            localized_motion = bool(frame.ball_count > 0 and (
                frame.cue_ball_normalized_speed >= .5 or frame.max_ball_normalized_speed >= .6
                or frame.moving_ball_count > 0 or frame.ball_residual_motion >= .15))
            uncertain_closeup = bool(frame.view_type == CameraViewType.BALL_CLOSEUP
                                     and frame.ball_count > 0
                                     and (not frame.observation_valid or frame.motion_raw >= .12))
            returned = any(start <= frame.t <= end for start, end in self._confirmed_target_returns)
            if not (addressed or localized_motion or uncertain_closeup or returned):
                continue
            # Keep half a second on both sides of an action row. A row merely
            # touching another window's edge lacks reliable pre/post context.
            lo, hi = frame.t-.5, frame.t+.5
            index = bisect_left(ends, hi)
            already_covered = index < len(merged) and merged[index][0] <= lo
            if already_covered or hi <= covered_until:
                continue
            proposed.append(StrikeCandidate(
                timestamp=frame.t, confidence=.5, uncertainty_start=max(0, frame.t-1.5),
                uncertainty_end=frame.t+1.5, camera_view=frame.view_type,
                evidence={"sparse_proposal": 1., "coverage_proposal": 1.,
                          "coverage_cue_address": float(addressed),
                          "coverage_localized_motion": float(localized_motion),
                          "coverage_uncertain_closeup": float(uncertain_closeup),
                          "coverage_target_return": float(returned)},
            ))
            covered_until = frame.t+forward
        return proposed

    def _contact_window(self, candidate: StrikeCandidate) -> tuple[float, float]:
        """Decode the uncertainty union even after nearby proposals are merged."""
        backward = float(self.config.get("analysis.refine_backward_seconds", 2))
        forward = float(self.config.get("analysis.strike_refine_post_seconds", 2))
        start, end = candidate.timestamp-backward, candidate.timestamp+forward
        if candidate.evidence.get("sparse_proposal", 0) >= .5:
            if candidate.uncertainty_start > 0:
                start = min(start, candidate.uncertainty_start-min(backward, .75))
            end = max(end, candidate.uncertainty_end+.5)
        if candidate.evidence.get("rack_restart", 0) >= .5:
            start = min(start, candidate.uncertainty_start)
        return start, end

    def _record_detection_stage(
        self, stage: str, candidates: list[StrikeCandidate],
        features: list[FrameFeatures] | None = None, extra: dict | None = None,
    ) -> None:
        """Persist counts at each gate so lost contacts are diagnosable."""
        row = {"stage": stage, "candidates": len(candidates),
               "possible_replays": sum(c.possible_replay for c in candidates),
               "coverage_proposals": sum(c.evidence.get("coverage_proposal", 0) >= .5 for c in candidates),
               "timestamps": [round(c.timestamp, 6) for c in candidates]}
        if features is not None:
            row.update(observations=len(features),
                       table_observable=sum(f.table_observable for f in features),
                       valid_observations=sum(f.observation_valid for f in features),
                       foreign_observations=sum(not f.match_context_valid for f in features),
                       cue_ball_observations=sum(f.cue_ball_detected for f in features),
                       native_observations=sum(f.observation_fps >= 10 for f in features))
        row.update(extra or {})
        self._detection_diagnostics["stages"].append(row)
        self._detection_diagnostics["confirmed_target_returns"] = self._confirmed_target_returns
        try:
            self._write_json_atomic(self.job_dir/"detection_diagnostics.json", self._detection_diagnostics)
        except OSError as exc:
            logger.warning("Could not save detection diagnostics: %s", exc)

    def _recover_replay_returns(
        self, candidates: list[StrikeCandidate], features: list[FrameFeatures], duration: float,
    ) -> list[StrikeCandidate]:
        """Retain visible live rolls whose contact was hidden by a replay wipe."""
        intervals = self.replay_det._stinger_intervals(features, candidates)
        completed = {}
        ordered = sorted(candidates, key=lambda c: c.timestamp)
        times = [f.t for f in features]
        for opening, _ in intervals:
            prior = next((c for c in reversed(ordered) if c.timestamp < opening
                          and c.confidence >= .40 and not c.possible_replay
                          and not any(lo <= c.timestamp <= hi for lo, hi in intervals)), None)
            if prior is None or prior.timestamp in completed:
                continue
            evidence = prior.evidence
            end = float(evidence.get("refined_stop_timestamp", 0))
            if (prior.timestamp <= end < opening
                    and (evidence.get("refined_stop_confidence", 0) >= .70
                         or evidence.get("refined_stop_upper_bound", 0) >= .5)):
                completed[prior.timestamp] = end
                continue
            stop = self.segmenter.ball_stop.detect_stop(prior, features, duration, times=times)
            if (stop.confirmed and prior.timestamp <= stop.physical_stop_timestamp < opening
                    and (stop.end_confidence >= .70 or "upper_bound" in stop.reason)):
                completed[prior.timestamp] = stop.physical_stop_timestamp
        returning = self.replay_det.returning_live_candidates(candidates, features, completed)
        return self._deduplicate_candidates(candidates+returning) if returning else candidates

    @staticmethod
    def _preparation_intervals(features: list[FrameFeatures]) -> list[tuple[float, float]]:
        """Confirm preparation retrospectively when returned reds form a rack.

        Require an observed nearly cleared table followed by replenished reds.
        A camera cut or missing observation cannot establish the change.
        """
        intervals = []
        times = [f.t for f in features]
        for i, frame in enumerate(features):
            if not frame.red_rack_intact or frame.red_area_ratio <= 0:
                continue
            if i and features[i - 1].red_rack_intact:
                continue
            history = features[bisect_left(times, frame.t - 180):i]
            quiet_since = None
            last_low = None
            previous_t = None
            rise_run = 0
            preparation_start = None
            for item in history:
                if previous_t is not None and item.t - previous_t > 0.76:
                    quiet_since = None
                    last_low = None
                    preparation_start = None
                previous_t = item.t
                if not item.rack_observation_valid or item.scene_cut_score >= 0.5:
                    quiet_since = None
                    last_low = None
                    preparation_start = None
                    continue
                if preparation_start is not None:
                    continue
                if item.red_area_ratio <= frame.red_area_ratio * 0.12:
                    rise_run = 0
                    if quiet_since is None:
                        quiet_since = item.t
                    if item.t - quiet_since >= 3:
                        last_low = item.t
                else:
                    quiet_since = None
                    if last_low is not None and item.red_area_ratio >= frame.red_area_ratio * 0.25:
                        rise_run += 1
                        if rise_run >= 2:
                            # Later referee occlusion can hide the growing pack;
                            # do not mistake that for a newly cleared table.
                            preparation_start = max(history[0].t, last_low - 4)
                    else:
                        rise_run = 0
            if preparation_start is not None and frame.t - preparation_start >= 4:
                start = preparation_start
                if intervals and start <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], frame.t)
                else:
                    intervals.append((start, frame.t))
        return intervals

    @staticmethod
    def _rack_candidate_supported(candidate: StrikeCandidate, coarse: list[FrameFeatures]) -> bool:
        if candidate.evidence.get("cue_geometry_confirmed", 0) >= 0.5:
            return True
        times = [f.t for f in coarse]
        pre = coarse[bisect_left(times, candidate.timestamp - 1):bisect_right(times, candidate.timestamp)]
        post = coarse[bisect_left(times, candidate.timestamp + 0.5):bisect_right(times, candidate.timestamp + 2.5)]
        # A placed/picked-up white ball can have launch-like speed. Reject it
        # only with sustained intact-rack evidence on both sides and no cue
        # contact; missing or obscured observations do not prove preparation.
        return not (len(pre) >= 2 and len(post) >= 3
                    and all(f.red_rack_intact for f in pre + post))

    @staticmethod
    def _rack_wait_intervals(features: list[FrameFeatures]) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        active = False
        for feature in features:
            if not feature.rack_idle:
                active = False
                continue
            if active and intervals and feature.t - intervals[-1][1] <= 0.76:
                intervals[-1] = (intervals[-1][0], feature.t)
            else:
                intervals.append((feature.t, feature.t))
            active = True
        return intervals

    def _extract_features(
        self,
        proxy_path: Path,
        audio_path: Optional[Path],
        mapper: TimeMapper,
        duration: float,
        progress: Optional[Callable[[float, str], None]] = None,
        sample_fps: Optional[float] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        collect_scene_observations: bool = False,
        checkpoint_stage: str = "features",
        stop_when: Optional[Callable[[list[FrameFeatures]], bool]] = None,
    ) -> tuple[list[FrameFeatures], list[SceneObservation], list[float]]:
        sample_fps = float(sample_fps or self.config.get("analysis.sample_fps", 10.0))
        start_time = max(0.0, float(start_time))
        end_time = duration if end_time is None else min(duration, float(end_time))
        # Sound is preserved in preview/export, but never analysed for strikes.

        cap = open_capture(
            proxy_path,
            prefer_hwaccel=bool(self.config.get("analysis.hwaccel_decode", True)),
        )
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open proxy video: {proxy_path}")

        proxy_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.config.get("proxy.target_fps", 15.0))
        # Timestamp-based sampling avoids the old ``round(proxy_fps / fps)``
        # drift (15fps/10fps silently became 7.5fps).  The dense refinement pass
        # below can then use the native presentation timestamps.
        sample_period = 1.0 / max(sample_fps, 1e-6)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        features: list[FrameFeatures] = []
        scene_stream = (
            self.scene_det.start_stream() if collect_scene_observations else None
        )
        prev_gray: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        prev_image: Optional[np.ndarray] = None
        prev_sample_t: Optional[float] = None
        camera_scene_id = 0
        interaction = TableInteractionDetector()
        geometry = ViewGeometry()
        current_view = CameraViewType.OTHER
        current_view_extra: dict = {}
        self.broadcast_context.reset_observations(preserve_target=True)
        last_context_t: float | None = None
        context = None
        active_foreign = False
        context_reference_times = [f.t for f in self._coarse_context_reference]
        idx = 0
        kept = 0
        scene_step = max(1, int(round(sample_fps / 2)))  # ~2 fps for scene detect
        refine_fps = float(self.config.get("analysis.refine_fps", 30.0))
        native_target = min(refine_fps, proxy_fps, mapper.source_fps or refine_fps)
        dense_pass = sample_fps >= native_target * 0.90
        rack_gate = (
            RackIdleGate() if sample_fps <= 3
            and bool(self.config.get("analysis.skip_racked_waits", True)) else None
        )
        flow_refresh_fps = min(
            sample_fps,
            float(self.config.get("analysis.native_flow_fps", 10.0)) if dense_pass else sample_fps,
        )
        flow_refresh_period = 1.0 / max(flow_refresh_fps, 1e-6)
        last_flow_t: float | None = None
        table_key = (
            "analysis.refine_table_refresh_fps"
            if dense_pass
            else "analysis.coarse_table_refresh_fps"
        )
        table_refresh_fps = min(
            sample_fps,
            float(self.config.get(table_key, 10.0 if dense_pass else 2.0)),
        )
        table_refresh_period = 1.0 / max(table_refresh_fps, 1e-6)
        last_table_t: float | None = None
        table_obs = None
        self.motion.flow_scale = float(
            np.clip(
                self.config.get(
                    "motion.flow_scale"
                    if dense_pass
                    else "analysis.coarse_flow_scale",
                    0.33 if dense_pass else 0.25,
                ),
                0.25,
                1.0,
            )
        )
        hough_fps = (
            min(
                sample_fps,
                float(self.config.get("analysis.refine_hough_fps", 15.0)),
            )
            if dense_pass
            else min(
                sample_fps,
                float(self.config.get("analysis.coarse_hough_fps", 5.0)),
            )
        )
        hough_step = max(1, int(round(sample_fps / max(hough_fps, 1e-6))))
        # Seek close to the requested source interval for dense refinement.  We
        # still discard frames until the mapped presentation time reaches the
        # exact start boundary, so VFR/mapper rounding cannot leak earlier data.
        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, mapper.to_proxy(start_time) * 1000.0)
            idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))

        self.table.reset()
        self.motion.reset()
        self.objects.reset()
        self.tracker = BallTracker()
        self._last_cue_tip = None

        # Decode runs on a background thread (see utils.video) so FFmpeg/NVDEC
        # frame decode overlaps the OpenCL/CPU feature extraction below.
        # Unsampled proxy frames are ``grab``-advanced without BGR retrieval.
        frames = sampled_frames(
            cap,
            proxy_fps=proxy_fps,
            to_source=mapper.to_source,
            start_time=start_time,
            end_time=end_time,
            sample_period=sample_period,
            start_idx=idx,
            prefetch=int(self.config.get("analysis.decode_prefetch_frames", 8)),
        )
        try:
            for idx, t, frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect cuts before optical flow/tracking.  A cut is an unknown
                # observation, never a stationary frame, and view-local trackers are
                # reacquired without terminating the logical shot.
                hist = self.scene_det.histogram(frame)
                cut_image = self.scene_det.cut_thumbnail(frame)
                online_cut = 0.0
                if prev_hist is not None:
                    online_cut = self.scene_det.cut_score_simple(prev_hist, hist)
                if prev_image is not None:
                    online_cut = max(online_cut, self.scene_det.structural_cut_score(prev_image, cut_image))
                prev_image = cut_image
                cut_like = online_cut >= float(
                    self.config.get("scene_detection.hard_cut_threshold", 0.42)
                )
                if cut_like:
                    camera_scene_id = int(round(t * 1000)) + 1
                    interaction.reset()
                    self.table.reset()
                    self.motion.reset()
                    self.objects.reset()
                    self.tracker = BallTracker()
                    self._last_cue_tip = None
                    prev_gray = None
                    prev_sample_t = None
                    last_flow_t = None
                    table_obs = None
                    last_table_t = None

                if rack_gate is not None and rack_gate.observe(frame, t):
                    features.append(FrameFeatures(
                        t=t, rack_idle=True, red_rack_intact=True, observation_valid=False,
                        observation_fps=sample_fps,
                        red_area_ratio=rack_gate.red_area_ratio, rack_observation_valid=True,
                        table_observable=True, table_confidence=0.9,
                        green_ratio=rack_gate.table_ratio,
                        table_mask_area_ratio=rack_gate.table_ratio,
                        scene_cut_score=online_cut,
                        camera_scene_id=camera_scene_id,
                        table_full_view=False,
                    ))
                    if scene_stream is not None and kept % scene_step == 0:
                        scene_stream.observe(frame, t, histogram=hist)
                    prev_gray = None
                    prev_hist = hist
                    kept += 1
                    if progress and kept % 20 == 0:
                        progress(min(0.99, (t - start_time) / max(end_time - start_time, 1e-6)),
                                 "Skipping stationary racked-table wait")
                    continue
                if rack_gate is not None and rack_gate.just_released:
                    self.motion.reset()
                    self.objects.reset()
                    self.tracker = BallTracker()
                    self._last_cue_tip = None
                    prev_sample_t = None
                    last_flow_t = None

                if (
                    table_obs is None
                    or last_table_t is None
                    or t - last_table_t + 1e-9 >= table_refresh_period
                ):
                    prior_table = table_obs
                    table_obs = self.table.detect(frame)
                    if not cut_like and self._table_view_changed(prior_table, table_obs):
                        # Similar green histograms can conceal a close-up to
                        # overhead cut. A large playing-surface rearrangement
                        # still invalidates image coordinates and ball scale.
                        online_cut = max(online_cut, .5)
                        cut_like = True
                        camera_scene_id = int(round(t * 1000)) + 1
                        interaction.reset()
                        self.table.reset()
                        table_obs = self.table.detect(frame)
                        self.motion.reset()
                        self.objects.reset()
                        self.tracker = BallTracker()
                        self._last_cue_tip = None
                        prev_gray = None
                        prev_sample_t = None
                        last_flow_t = None
                    geometry = view_geometry(table_obs, frame.shape)
                    current_view, _, current_view_extra = self.scene_det.classifier.classify(frame)
                    if geometry.full_table:
                        current_view = CameraViewType.MAIN_TABLE
                    last_table_t = t
                if last_context_t is None or t - last_context_t >= 0.49:
                    context = self.broadcast_context.observe(frame, t, current_view)
                    last_context_t = t
                    if context.foreign_match:
                        active_foreign = True
                    elif context.identity_known and context.identity_similarity >= self.broadcast_context.same_threshold:
                        active_foreign = False
                    if context.target_interval_start is not None:
                        self._confirm_target_return(features, context.target_interval_start, t)
                    if context.foreign_match and context.foreign_interval_start is not None:
                        for earlier in reversed(features):
                            if earlier.t < context.foreign_interval_start:
                                break
                            earlier.match_context_valid = False
                context_valid = not active_foreign
                observable = bool(table_obs.confidence >= 0.25 and table_obs.area_ratio >= 0.03)
                dets = self.objects.detect(
                    frame,
                    table_obs.mask,
                    use_hough=kept % hough_step == 0,
                    partial_view=not geometry.full_table,
                    table_bounds=((table_obs.bbox[0], table_obs.bbox[1],
                                   table_obs.bbox[0]+table_obs.bbox[2], table_obs.bbox[1]+table_obs.bbox[3])
                                  if table_obs.bbox is not None else None),
                ) if observable and context_valid else []
                ball_regions = [
                    (d.cx, d.cy, d.diameter_px)
                    for d in dets
                    if d.confidence >= 0.30 and d.diameter_px > 1.0
                    and d.shape_confidence >= 0.48
                    and d.cloth_surround_confidence >= 0.45
                ]
                residual = None
                comparison_gray = prev_gray
                if prev_gray is not None and observable and context_valid:
                    dt = t - prev_sample_t if prev_sample_t is not None else 1.0 / max(sample_fps, 1.0)
                    refresh_flow = last_flow_t is None or t - last_flow_t + 1e-9 >= flow_refresh_period
                    residual = self.motion.analyze(
                        prev_gray,
                        gray,
                        table_obs.mask,
                        ball_regions=ball_regions,
                        frame_dt=dt,
                        refresh_flow=refresh_flow,
                    )
                    if refresh_flow:
                        last_flow_t = t
                prev_gray = gray if observable and context_valid else None
                prev_sample_t = t
                prev_hist = hist
                tracks = self.tracker.update(
                    t, dets, camera_transform=residual.camera_transform if residual else None,
                )
                cue = self.tracker.cue_ball_track()

                diameter = max(
                    self.tracker.estimated_ball_diameter(),
                    self.objects.estimated_ball_diameter(),
                    0.0,
                )
                stop_speed = float(
                    self.config.get("ball_stop.motion_stop_normalized_speed", 0.16)
                )
                visible_tracks = [tr for tr in tracks if tr.active and tr.visible]
                moving_count = sum(
                    1
                    for tr in visible_tracks
                    if diameter > 0.5
                    and self.tracker.stable_track_speed(tr, tr.diameter or diameter) >= stop_speed
                )
                cue_speed_px = cue.speed() if cue is not None and cue.visible else 0.0
                cue_diameter = cue.diameter if cue is not None and cue.diameter > 0.5 else diameter
                cue_speed_norm = cue_speed_px / cue_diameter if cue is not None and cue_diameter > 0.5 else 0.0
                cue_accel_norm = (
                    float(np.hypot(cue.ax, cue.ay) / cue_diameter)
                    if cue is not None and cue.visible and cue_diameter > 0.5
                    else 0.0
                )
                cue_geometry = self._cue_geometry(frame, table_obs.mask, cue, cue_diameter, t)
                handling, handling_score = interaction.observe(
                    frame, table_obs, dets, t,
                    cue_visible=bool(cue_geometry["visible"]) or cue_speed_norm >= 1.0,
                ) if observable and context_valid else (False, 0.0)
                table_observable = bool(
                    table_obs.confidence >= float(
                        self.config.get("table_detection.min_confidence", 0.25)
                    )
                    and table_obs.area_ratio >= 0.03
                )
                observation_valid = bool(
                    table_observable
                    and context_valid
                    and not cut_like
                    and (residual is None or residual.observation_valid)
                )


                feat = FrameFeatures(
                    t=t,
                    observation_fps=sample_fps,
                    camera_scene_id=camera_scene_id,
                    view_classified=True,
                    table_full_view=geometry.full_table,
                    table_handling=handling,
                    handling_score=handling_score,
                    match_context_valid=context_valid,
                    appearance_signature=current_view_extra.get("replay_stinger_signature", []),
                    ball_layout_signature=ball_layout(frame, dets, geometry) if kept % max(1, int(sample_fps / 5)) == 0 else [],
                    rack_restart=bool(rack_gate and rack_gate.just_released),
                    red_rack_intact=bool(rack_gate and rack_gate.racked),
                    red_area_ratio=rack_gate.red_area_ratio if rack_gate else 0.0,
                    rack_observation_valid=bool(rack_gate and rack_gate.table_ratio >= 0.12),
                    table_confidence=table_obs.confidence,
                    table_mask_area_ratio=table_obs.area_ratio,
                    residual_motion_mean=residual.residual_mean if residual else 0.0,
                    residual_motion_max=residual.residual_max if residual else 0.0,
                    motion_area_ratio=residual.motion_area_ratio if residual else 0.0,
                    camera_motion_magnitude=residual.camera_magnitude if residual else 0.0,
                    view_type=current_view,
                    green_ratio=table_obs.area_ratio,
                    ball_count=len([tr for tr in tracks if tr.active]),
                    cue_ball_detected=cue is not None and cue.visible,
                    max_ball_speed=float(self.tracker.max_speed()),
                    table_observable=table_observable,
                    observation_valid=observation_valid,
                    ball_diameter_px=diameter,
                    cue_ball_x=(cue.positions[-1][1] if cue is not None and cue.visible else None),
                    cue_ball_y=(cue.positions[-1][2] if cue is not None and cue.visible else None),
                    cue_ball_speed=cue_speed_px,
                    cue_ball_normalized_speed=cue_speed_norm,
                    cue_ball_stable_normalized_speed=(
                        self.tracker.stable_track_speed(cue, cue_diameter)
                        if cue is not None and cue.visible else None),
                    cue_ball_acceleration=cue_accel_norm,
                    cue_ball_track_confidence=(cue.confidence if cue is not None and cue.visible else 0.0),
                    cue_tip_visible=bool(cue_geometry["visible"]),
                    cue_tip_distance_to_ball=float(cue_geometry["distance"]),
                    cue_approach_speed=float(cue_geometry["approach_speed"]),
                    cue_forward_motion=float(cue_geometry["forward_motion"]),
                    cue_contact_score=float(cue_geometry["contact_score"]),
                    max_ball_normalized_speed=self.tracker.max_normalized_speed(),
                    ball_kinematics_valid=any(
                        tr.hits >= 2 and self.tracker.is_ball_quality_track(tr)
                        for tr in visible_tracks
                    ),
                    ambiguous_ball_motion=self.tracker.ambiguous_region_motion(
                        comparison_gray, gray, residual.camera_transform if residual else None),
                    moving_ball_count=moving_count,
                    occluded_ball_count=self.tracker.occluded_moving_count(
                        min_normalized_speed=stop_speed,
                        ball_diameter_px=diameter,
                    ),
                    ball_residual_motion=(residual.ball_residual_motion if residual else 0.0),
                    motion_score=residual.motion_score if residual else 0.0,
                    motion_raw=residual.motion_raw if residual else 0.0,
                    scene_cut_score=online_cut,
                )
                self._apply_match_context([feat], self._coarse_context_reference, context_reference_times)
                features.append(feat)

                if scene_stream is not None and kept % scene_step == 0:
                    scene_stream.observe(frame, t, histogram=hist)

                kept += 1
                if (
                    stop_when is not None
                    and kept % max(1, int(round(sample_fps * 0.5))) == 0
                    and stop_when(features)
                ):
                    break
                if progress and kept % 20 == 0:
                    interval = max(end_time - start_time, 1e-6)
                    fraction = (t - start_time) / interval
                    progress(
                        min(0.99, max(0.0, fraction)),
                        f"Frame {idx + 1}/{total_frames}",
                    )

                # Checkpoint periodically for long videos
                if kept % 500 == 0:
                    self._write_checkpoint(
                        {
                            "stage": checkpoint_stage,
                            "frame_idx": idx,
                            "source_time": t,
                            "features": len(features),
                        }
                    )

        finally:
            frames.close()
            cap.release()
        if progress:
            progress(1.0, f"Sampled {len(features)} frames")
        observations = scene_stream.observations if scene_stream is not None else []
        self._apply_match_context(features, self._coarse_context_reference)
        return features, observations, []

    @staticmethod
    def _apply_match_context(features: list[FrameFeatures], reference: list[FrameFeatures],
                             reference_times: list[float] | None = None) -> None:
        """Carry confirmed foreign-table spans into scoreboard-free dense views.

        A refinement seek must not learn a foreign scoreboard as a new target.
        Coarse decisions are reused only inside the observed invalid interval.
        """
        if not reference:
            return
        times = reference_times if reference_times is not None else [f.t for f in reference]
        for f in features:
            i = bisect_right(times, f.t) - 1
            if 0 <= i < len(reference) and f.t - reference[i].t <= 0.76:
                if not reference[i].match_context_valid:
                    f.match_context_valid = False
                    f.observation_valid = False

    def _confirm_target_return(
        self, features: list[FrameFeatures], start: float, end: float,
    ) -> None:
        """Reopen only the bounded table episode proved by the target overlay.

        Previously blocked rows contain no object evidence. Repair their match
        identity, retain their observation uncertainty, and queue native scans
        through the episode rather than inventing ball observations.
        """
        if not 0 < end-start <= 15:
            return
        self._confirmed_target_returns.append((start, end))
        for rows in (features, self._coarse_context_reference):
            times = [f.t for f in rows]
            for frame in rows[bisect_left(times, start):bisect_right(times, end)]:
                frame.match_context_valid = True

    @staticmethod
    def _table_view_changed(before: TableObservation | None, after: TableObservation) -> bool:
        """Detect a major view-local cloth change missed by image histograms."""
        if (before is None or before.contour is None or after.contour is None
                or before.confidence < .5 or after.confidence < .5
                or before.bbox is None or after.bbox is None):
            return False
        ax, ay, aw, ah = before.bbox
        bx, by, bw, bh = after.bbox
        area_a, area_b = aw*ah, bw*bh
        if min(area_a, area_b) <= 0:
            return False
        intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
        overlap = intersection / (area_a+area_b-intersection)
        scale = area_b / area_a
        return overlap < .35 or (overlap < .60 and (scale < .55 or scale > 1.8))

    def _unresolved_stop_candidates(
        self, candidates: list[StrikeCandidate], features: list[FrameFeatures], duration: float,
    ) -> list[StrikeCandidate]:
        """Find supported contacts whose forward stop observations are missing."""
        selected = []
        times = [f.t for f in features]
        ordered = sorted(candidates, key=lambda c: c.timestamp)
        for index, candidate in enumerate(ordered):
            if candidate.possible_replay or not self.segmenter._candidate_supported(candidate):
                continue
            evidence = candidate.evidence
            declared_stop = float(evidence.get("refined_stop_timestamp", duration))
            next_t = ordered[index+1].timestamp if index+1 < len(ordered) else None
            limit = min(declared_stop-.5, duration,
                        candidate.timestamp+(self.segmenter.ball_stop.max_after_strike or 60))
            baseline = self.segmenter.ball_stop._baseline(features, candidate.timestamp, times)
            quiet_start = None
            previous = None
            earlier_quiet = False
            sparse_run = False
            for f in features[bisect_left(times, candidate.timestamp+.2):bisect_left(times, limit)]:
                quiet = (self.segmenter.ball_stop._is_full_table_observation(f)
                         and not self.segmenter.ball_stop._moving_evidence(
                             f, already_moving=True, baseline=baseline))
                continuous = (previous is not None and f.t-previous.t <= .76
                              and f.camera_scene_id == previous.camera_scene_id)
                if not quiet or not continuous or quiet_start is None:
                    quiet_start = f.t if quiet else None
                    sparse_run = bool(quiet and 0 < f.observation_fps <= 3)
                else:
                    sparse_run = sparse_run or (0 < f.observation_fps <= 3)
                if (sparse_run and quiet_start is not None
                        and f.t-quiet_start >= max(.75, self.segmenter.ball_stop.confirm_s)):
                    # Sparse quiet samples are a decoding proposal, never stop
                    # proof. Revisit them rather than borrowing a later shot's
                    # native boundary beyond a hole in the original coverage.
                    earlier_quiet = True
                    break
                previous = f
            if earlier_quiet:
                selected.append(candidate)
                continue
            if (evidence.get("refined_stop_timestamp", 0) > candidate.timestamp
                    and (evidence.get("refined_stop_confidence", 0) >= .70
                         or evidence.get("refined_stop_upper_bound", 0) >= .5)):
                continue
            stop = self.segmenter.ball_stop.detect_stop(
                candidate, features, duration, times=times, next_strike_timestamp=next_t,
            )
            if not stop.confirmed and stop.reason != "unconfirmed_ball_handling_boundary":
                selected.append(candidate)
        return selected

    def _refine_candidate_windows(
        self,
        proxy_path: Path,
        audio_path: Optional[Path],
        mapper: TimeMapper,
        duration: float,
        candidates: list[StrikeCandidate],
        coarse_features: list[FrameFeatures],
        progress: Optional[Callable[[float, str], None]] = None,
        signature: str = "",
        resume: bool = True,
        existing_dense: list[FrameFeatures] | None = None,
    ) -> tuple[list[StrikeCandidate], list[FrameFeatures]]:
        """Extract dense observations around each candidate and its rough stop.

        Every interval is bounded by the same conservative maximum used by the
        stop detector.  An unresolved final candidate must never expand to the
        end of a multi-hour source; it is capped and marked for review later.
        """
        if not candidates:
            if progress:
                progress(1.0, "No candidate windows to refine")
            return candidates, []
        if self.config.get("analysis.adaptive_stop_tracking", True):
            return self._refine_adaptive_windows(
                proxy_path, audio_path, mapper, duration, candidates,
                progress=progress, signature=signature, resume=resume,
                existing_dense=existing_dense,
                rack_features=coarse_features,
            )
        dense_fps = float(
            self.config.get(
                "analysis.refine_fps",
                self.config.get("analysis.sample_fps", 10.0),
            )
        )
        dense_fps = min(dense_fps, mapper.source_fps or dense_fps,
                        float(self.config.get("proxy.target_fps", dense_fps)))
        merge_gap = float(
            self.config.get("analysis.refine_merge_gap_seconds", 0.25)
        )
        backward_seconds = max(
            0.5,
            float(self.config.get("analysis.refine_backward_seconds", 2.0)),
        )
        reacquisition_tail = max(
            0.0,
            float(self.config.get("analysis.reacquisition_tail_seconds", 2.0)),
        )
        dense: list[FrameFeatures] = []
        ranges: list[tuple[float, float, float]] = []
        coarse_times = [feature.t for feature in coarse_features]
        max_after = self.segmenter.ball_stop.max_after_strike
        if max_after is None:
            max_after = float(
                self.config.get("motion.max_ball_travel_seconds", 60.0)
            )
        confirmation_tail = max(
            0.8,
            float(self.segmenter.ball_stop.confirm_s) + 0.2,
        )
        for i, candidate in enumerate(candidates):
            rough = self.segmenter.ball_stop.detect_stop(
                candidate, coarse_features, duration, times=coarse_times
            )
            next_t = candidates[i + 1].timestamp if i + 1 < len(candidates) else duration
            hard_end = min(
                duration,
                candidate.timestamp + float(max_after) + confirmation_tail,
            )
            # Once a sparse proposal is found, decode continuously at the
            # refinement/native cadence from a little before contact until the
            # rough stop (plus a short tail for a following strike).  This is
            # what lets the dense pass recover impacts that fell between sparse
            # samples and observe the true movement-to-stop transition.
            start = max(0.0, candidate.timestamp - backward_seconds)
            if rough.confirmed:
                end = min(
                    hard_end,
                    max(
                        start + 1.0,
                        rough.physical_stop_timestamp
                        + confirmation_tail
                        + reacquisition_tail,
                    ),
                )
                if end > start:
                    ranges.append((start, end, dense_fps))
                continue
            if rough.reason == "max_duration_review_cap":
                end = hard_end
            elif next_t < hard_end:
                # A later valid strike is temporal evidence that this uncertain
                # shot ended before it; retain a generous reacquisition tail.
                end = min(hard_end, next_t + 1.0)
            else:
                end = hard_end
            if end <= start:
                end = min(hard_end, start + 1.0)
            if end > start:
                ranges.append((start, end, dense_fps))

        # Closely spaced shots often have overlapping strike/settling windows.
        # Decode and analyze each combined interval once instead of seeking,
        # rebuilding the tracker, and recomputing flow for every candidate.
        # This preserves the exact dense observations while cutting duplicated
        # work on full matches with clusters of safety exchanges.
        merged_ranges: list[tuple[float, float, float]] = []
        for start, end, scan_fps in sorted(ranges):
            if merged_ranges and start <= merged_ranges[-1][1] + merge_gap:
                previous_start, previous_end, previous_fps = merged_ranges[-1]
                merged_ranges[-1] = (
                    previous_start,
                    max(previous_end, end),
                    max(previous_fps, scan_fps),
                )
            else:
                merged_ranges.append((start, end, scan_fps))

        if len(merged_ranges) < len(ranges):
            logger.info(
                "Merged %d dense refinement windows into %d intervals",
                len(ranges),
                len(merged_ranges),
            )

        if not merged_ranges:
            if progress:
                progress(1.0, "No valid candidate windows to refine")
            return candidates, []

        total_ranges = max(1, len(merged_ranges))
        existing_times = [feature.t for feature in existing_dense or []]
        for range_idx, (start, end, scan_fps) in enumerate(merged_ranges):
            try:
                part = None
                if existing_dense and scan_fps >= dense_fps:
                    lo = bisect_left(existing_times, start)
                    hi = bisect_right(existing_times, end)
                    covered = existing_dense[lo:hi]
                    max_gap = 1.5 / max(dense_fps, 1.0)
                    if (
                        len(covered) >= 2
                        and covered[0].t - start <= max_gap
                        and end - covered[-1].t <= max_gap
                        and all(
                            right.t - left.t <= max_gap
                            for left, right in zip(covered, covered[1:])
                        )
                    ):
                        # Audio seeds merged into a visual interval were already
                        # observed at native cadence. Reuse that evidence rather
                        # than decoding and tracking the same shot a second time.
                        part = covered
                part = (
                    self._load_dense_window(
                        signature, range_idx, start, end
                    )
                    if part is None and resume and signature
                    else part
                )
                if part is None:
                    part, _, _ = self._extract_features(
                        proxy_path,
                        audio_path,
                        mapper,
                        duration,
                        progress=(
                            lambda frac, msg, base=range_idx: progress(
                                (base + frac) / total_ranges,
                                f"Window {base + 1}/{total_ranges}: {msg}",
                            )
                            if progress
                            else None
                        ),
                        sample_fps=scan_fps,
                        start_time=start,
                        end_time=end,
                        collect_scene_observations=False,
                        checkpoint_stage="dense_refinement",
                    )
                    if signature:
                        self._save_dense_window(
                            signature, range_idx, start, end, part
                        )
                elif progress:
                    progress(
                        (range_idx + 1) / total_ranges,
                        f"Resumed window {range_idx + 1}/{total_ranges}",
                    )
                dense.extend(part)
            except Exception as exc:  # pragma: no cover - codec-specific fallback
                logger.warning(
                    "Dense refinement failed for %.3f-%.3f: %s",
                    start,
                    end,
                    exc,
                )
            if progress:
                progress(
                    (range_idx + 1) / total_ranges,
                    f"Refined window {range_idx + 1}/{total_ranges}",
                )

        if not dense:
            return candidates, []
        # Keep the denser layer wherever it overlaps a coarse sample.  Rounded
        # presentation times de-duplicate VFR/proxy seek jitter deterministically.
        return candidates, dense

    def _refine_adaptive_windows(
        self, proxy_path: Path, audio_path: Optional[Path], mapper: TimeMapper,
        duration: float, candidates: list[StrikeCandidate], *,
        progress: Optional[Callable[[float, str], None]] = None,
        signature: str = "", resume: bool = True,
        existing_dense: list[FrameFeatures] | None = None,
        rack_features: list[FrameFeatures] | None = None,
    ) -> tuple[list[StrikeCandidate], list[FrameFeatures]]:
        """Track through the roll cheaply; spend native cadence on the boundaries.

        Stop tracking exits as soon as stationary evidence is confirmed. The
        long safety horizon therefore preserves genuine long rolls without
        decoding a minute at native rate for every proposed strike.
        """
        native_fps = min(float(self.config.get("analysis.refine_fps", 30.0)),
                         mapper.source_fps or 30.0,
                         float(self.config.get("proxy.target_fps", 30.0)))
        tracking_fps = min(native_fps, float(self.config.get("analysis.stop_tracking_fps", 10.0)))
        stop_warmup = float(self.config.get("analysis.stop_refine_window_seconds", 1.5))
        tail = float(self.config.get("analysis.stop_tracking_tail_seconds", 0.2))
        stop_detector = self.segmenter.ball_stop
        horizon = stop_detector.max_after_strike or float(self.config.get("motion.max_ball_travel_seconds", 60.0))
        known = sorted(existing_dense or [], key=lambda feature: feature.t)
        known_times = [feature.t for feature in known]
        priorities = {int(round(feature.t * 10000)): (
            4 if feature.contact_window else 3 if feature.view_classified
            and feature.observation_fps+1e-3 >= native_fps and 0 < i < len(known)-1
            and max(feature.t-known[i-1].t,known[i+1].t-feature.t) <= 2.0/native_fps else 1
        ) for i,feature in enumerate(known)}
        window_index = 0

        def remember(part: list[FrameFeatures], priority: int = 0) -> None:
            if not part:
                return
            lo = bisect_left(known_times, part[0].t - 1e-6)
            hi = bisect_right(known_times, part[-1].t + 1e-6)
            accepted = []
            for feature in part:
                key = int(round(feature.t * 10000))
                if priority >= priorities.get(key, -1):
                    accepted.append(feature)
                    priorities[key] = priority
            merged = self._merge_feature_layers(known[lo:hi], accepted)
            known[lo:hi] = merged
            known_times[lo:hi] = [feature.t for feature in merged]

        def settled(part: list[FrameFeatures], candidate: StrikeCandidate) -> bool:
            if not part or part[-1].t < candidate.timestamp + stop_detector.min_travel_s + stop_detector.confirm_s:
                return False
            stop = stop_detector.detect_stop(candidate, part, duration)
            bounded = stop.confirmed or stop.reason == "unconfirmed_ball_handling_boundary"
            return bounded and part[-1].t + 1e-6 >= stop.stop_confirmation_timestamp + tail

        def observe(start: float, end: float, fps: float, candidate: StrikeCandidate | None = None, priority: int = 0) -> list[FrameFeatures]:
            nonlocal window_index
            start, end = max(0.0, start), min(duration, end)
            index = window_index
            window_index += 1
            if end <= start:
                return []
            lo, hi = bisect_left(known_times, start - 1e-6), bisect_right(known_times, end + 1e-6)
            covered = known[lo:hi]
            # A previously observed stop may cover less than the planned safety
            # horizon. Reuse it only when the entire required interval is dense.
            required_end = end
            if candidate is not None and settled(covered, candidate):
                stop = stop_detector.detect_stop(candidate, covered, duration)
                required_end = min(end, stop.stop_confirmation_timestamp + tail)
                covered = covered[:bisect_right([feature.t for feature in covered], required_end + 2.0 / fps)]
            # Timestamp sampling from a faster proxy alternates adjacent and
            # skipped proxy images. Reuse this measured cadence rather than
            # decoding an already complete native window again.
            max_gap = 2.0 / max(fps, 1.0)
            complete = bool(
                len(covered) >= 2 and covered[0].t - start <= max_gap
                and required_end - covered[-1].t <= max_gap
                and all(right.t - left.t <= max_gap for left, right in zip(covered, covered[1:]))
                and all(f.observation_fps <= 0 or f.observation_fps+1e-3 >= fps for f in covered)
            )
            if complete and (
                priority < 3
                or all(priorities.get(int(round(feature.t * 10000)), -1) >= priority for feature in covered)
            ):
                return self._merge_feature_layers([], covered)
            part = self._load_dense_window(signature, index, start, end) if resume and signature else None
            if part is None:
                part, _, _ = self._extract_features(
                    proxy_path, audio_path, mapper, duration,
                    sample_fps=fps, start_time=start, end_time=end,
                    collect_scene_observations=False, checkpoint_stage="adaptive_refinement",
                    stop_when=(lambda features: settled(features, candidate)) if candidate is not None else None,
                )
                if signature:
                    self._save_dense_window(signature, index, start, end, part)
            remember(part, priority)
            return part

        total = max(1, len(candidates))
        for number, candidate in enumerate(candidates):
            start, contact_end = self._contact_window(candidate)
            contact = observe(start, contact_end, native_fps)
            contact = self.strike_det.score_frames(contact)
            if self.segmenter._candidate_supported(candidate):
                # A newer detector can require a few later frames to verify
                # the same launch. A previously single-frame proof interval
                # must not reject it before the native trajectory is measured.
                candidate.uncertainty_start = min(candidate.uncertainty_start, candidate.timestamp-.25)
                candidate.uncertainty_end = max(candidate.uncertainty_end, candidate.timestamp+.40)
            self.strike_det.refine_boundaries([candidate], contact)
            verified = bool(
                candidate.evidence.get("dense_transition_confirmed", 0.0) >= 0.5
                or candidate.evidence.get("sparse_dense_transition", 0.0) >= 0.5

            )
            if not verified:
                found = self.strike_det.detect_candidates(contact)
                fallback = [
                    item for item in found
                    if candidate.uncertainty_start <= item.timestamp <= candidate.uncertainty_end
                    and item.evidence.get("occlusion_inferred", 0.0) >= 0.5
                    and item.evidence.get("ball_onset_run", 0.0) >= 2.0
                ]
                if fallback:
                    launch = max(fallback, key=lambda item: item.confidence)
                    candidate.timestamp = launch.timestamp
                    candidate.confidence = launch.confidence
                    candidate.evidence.update(launch.evidence)
                    verified = True
            if verified and not self._rack_candidate_supported(candidate, rack_features or []):
                verified = False
                rejection_reason = "racked_table_without_cue_contact"
            else:
                rejection_reason = "native_transition_unconfirmed"
            self._detection_diagnostics["native_proposals"].append({
                "timestamp": round(candidate.timestamp, 6),
                "window_start": round(start, 6),
                "requested_window_end": round(contact_end, 6),
                "window_end": round(contact[-1].t, 6) if contact else start,
                "accepted": verified,
                "reason": "native_contact_confirmed" if verified else rejection_reason,
                "coverage_proposal": candidate.evidence.get("coverage_proposal", 0) >= .5,
                "observations": len(contact),
                "valid_observations": sum(f.observation_valid for f in contact),
                "foreign_observations": sum(not f.match_context_valid for f in contact),
                "cue_ball_observations": sum(f.cue_ball_detected for f in contact),
                "confirmation": {key: candidate.evidence.get(key, 0) for key in (
                    "dense_transition_confirmed", "sparse_dense_transition",
                    "occlusion_inferred", "ball_onset_run",
                )},
            })
            if not verified:
                if progress:
                    progress((number + 1) / total, f"Rejected non-strike proposal {number + 1}/{len(candidates)}")
                continue
            for feature in contact:
                feature.contact_window = True
            remember(contact, 4)
            tracking = observe(start, candidate.timestamp + horizon, tracking_fps, candidate, priority=1)
            # Restore native observations where the cheaper tracking layer used
            # the same timestamps, keeping exact contact evidence authoritative.
            remember(contact, 4)
            if tracking:
                stop = stop_detector.detect_stop(candidate, tracking, duration)
                if stop.confirmed:
                    # The native tracker can resolve a later stop than the
                    # cheap travel pass. Continue until its own confirmation
                    # instead of truncating at the rough boundary's tail.
                    stop_features = observe(
                        max(start, stop.physical_stop_timestamp - stop_warmup),
                        candidate.timestamp + horizon, native_fps, candidate, priority=3,
                    )
                    refined_stop = stop_detector.detect_stop(candidate, stop_features, duration)
                    if refined_stop.confirmed:
                        # Preserve the dedicated boundary observation. A later
                        # rejected proposal starts a different tracker and must
                        # not revise the already verified physical stop.
                        candidate.evidence.update({
                            "refined_stop_timestamp": refined_stop.physical_stop_timestamp,
                            "refined_stop_confirmation_timestamp": refined_stop.stop_confirmation_timestamp,
                            "refined_last_motion_timestamp": refined_stop.last_ball_motion_timestamp,
                            "refined_stop_confidence": refined_stop.end_confidence,
                            "refined_ball_motion_start": stop.motion_start,
                            "refined_stop_review_required": float(refined_stop.manual_review_required),
                            "refined_stop_upper_bound": float("upper_bound" in refined_stop.reason),
                        })
            if progress:
                progress((number + 1) / total, f"Tracked shot {number + 1}/{len(candidates)}")
        return candidates, self._merge_feature_layers([], known)

    @staticmethod
    def _merge_feature_layers(
        coarse: list[FrameFeatures], dense: list[FrameFeatures]
    ) -> list[FrameFeatures]:
        # Keep independently decoded contact windows intact: canonical IDs on
        # the combined timeline must never mutate the caller's tracker output.
        dense_times = [f.t for f in dense]
        coarse_times = [f.t for f in coarse]
        by_time: dict[int, FrameFeatures] = {}
        for f in coarse:
            i = bisect_right(dense_times, f.t)
            covered = 0 < i < len(dense) and dense[i].t-dense[i-1].t <= .16
            finer = (covered and f.t > dense[0].t+1e-6
                     and dense[i-1].observation_fps > f.observation_fps > 0
                     and f.match_context_valid)
            if not finer:
                by_time[int(round(f.t * 10000.0))] = f.model_copy()
        for i, original in enumerate(dense):
            j = bisect_right(coarse_times, original.t)
            if 0 < j < len(coarse):
                left, right = coarse[j-1], coarse[j]
                if (left.observation_fps > original.observation_fps > 0
                        and right.observation_fps == left.observation_fps
                        and right.t-left.t <= 1.5/left.observation_fps
                        and original.match_context_valid):
                    continue
            f = original.model_copy()
            key = int(round(f.t * 10000.0))
            prior = by_time.get(key)
            if prior is not None:
                if ((prior.contact_window and not f.contact_window)
                        or prior.observation_fps > f.observation_fps > 0):
                    prior.match_context_valid = prior.match_context_valid and f.match_context_valid
                    continue
                # A first seek image cannot measure its preceding cut. Inside
                # a continuous dense window, its measured cut timing wins over
                # a delayed coarse comparison of the same camera change.
                if i == 0 or f.t-dense[i-1].t > .16:
                    f.scene_cut_score = max(f.scene_cut_score, prior.scene_cut_score)
                f.match_context_valid = f.match_context_valid and prior.match_context_valid
            by_time[key] = f
        merged = [by_time[key] for key in sorted(by_time)]
        # Seek-local IDs cannot be compared between independently decoded
        # windows. Rebuild IDs from the observed cuts on the merged timeline.
        scene_id = 0
        previous_t = None
        for f in merged:
            if f.scene_cut_score >= 0.5 or (previous_t is not None and f.t-previous_t > 0.76):
                scene_id = int(round(f.t*10000)) + 1
            f.camera_scene_id = scene_id
            previous_t = f.t
        return merged

    def _deduplicate_candidates(
        self, candidates: list[StrikeCandidate]
    ) -> list[StrikeCandidate]:
        """Merge sparse and dense discoveries without double-counting a shot."""
        if not candidates:
            return []
        gap = float(
            self.config.get(
                "strike_fusion.candidate_peak_min_distance_seconds", 1.2
            )
        )
        ordered = sorted(candidates, key=lambda item: item.timestamp)
        kept: list[StrikeCandidate] = []
        for candidate in ordered:
            if not kept or candidate.timestamp - kept[-1].timestamp >= gap:
                kept.append(candidate)
                continue
            if candidate.confidence > kept[-1].confidence:
                previous = kept[-1]
                candidate.uncertainty_start = min(candidate.uncertainty_start, previous.uncertainty_start)
                candidate.uncertainty_end = max(candidate.uncertainty_end, previous.uncertainty_end)
                # Retain useful sparse evidence when a dense candidate replaces
                # it, while letting the dense timestamp/confidence win.
                candidate.evidence = {
                    **previous.evidence,
                    **candidate.evidence,
                }
                kept[-1] = candidate
            else:
                kept[-1].uncertainty_start = min(kept[-1].uncertainty_start, candidate.uncertainty_start)
                kept[-1].uncertainty_end = max(kept[-1].uncertainty_end, candidate.uncertainty_end)
                kept[-1].evidence = {
                    **candidate.evidence,
                    **kept[-1].evidence,
                }
        return kept

    @staticmethod
    def _annotate_scenes(
        features: list[FrameFeatures], scenes: list[SceneSegment]
    ) -> None:
        """Apply scene labels in one chronological pass.

        The former implementation searched every scene and every cut for every
        feature.  Long broadcasts can contain thousands of cuts, making that
        another avoidable quadratic step.
        """

        if not features:
            return
        if not scenes:
            return
        ordered_scenes = sorted(scenes, key=lambda item: item.start)
        cut_times = sorted(round(item.start, 3) for item in ordered_scenes)
        scene_idx = 0
        for feature in features:
            while (
                scene_idx + 1 < len(ordered_scenes)
                and feature.t >= ordered_scenes[scene_idx].end
            ):
                scene_idx += 1
            scene = ordered_scenes[scene_idx]
            if scene.start <= feature.t < scene.end or (
                scene_idx == len(ordered_scenes) - 1 and feature.t >= scene.start
            ):
                if not feature.view_classified:
                    feature.view_type = scene.view_type

            cut_idx = bisect_left(cut_times, feature.t)
            near_cut = any(
                abs(feature.t - cut_times[index]) < 0.08
                for index in (cut_idx - 1, cut_idx)
                if 0 <= index < len(cut_times)
            )
            if near_cut and not feature.view_classified:
                feature.scene_cut_score = max(feature.scene_cut_score, 1.0)

    def _analysis_signature(self, source: Path) -> str:
        """Fingerprint source plus analysis-relevant configuration for resume."""

        stat = source.stat()
        analysis_keys = (
            "device",
            "proxy",
            "analysis",
            "scene_detection",
            "camera_view",
            "broadcast_context",
            "table_detection",
            "camera_motion",
            "motion",
            "audio",
            "strike_fusion",
            "ball_stop",
            "replay",
            "object_detection",
            "temporal_model",
        )
        payload = {
            "cache_version": _CACHE_VERSION,
            "source": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            # Export/UI/mode settings do not affect extracted observations or
            # strike candidates and therefore must not discard expensive caches.
            "config": {key: self.config.get(key) for key in analysis_keys},
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _result_signature(self, source: Path) -> str:
        """Final clips also depend on segmentation settings, unlike features."""
        payload = {
            "analysis": self._analysis_signature(source),
            "result_policy_version": 16,
            "segmentation": {
                key: self.config.get(key) for key in ("modes", "confidence", "importance")
            },
            "transition_context": {
                "transition": self.config.get("export.transition", "cut"),
                "seconds": self.config.get("export.transition_seconds", 0.24),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _repair_pathological_replay_labels(
        self,
        features: list[FrameFeatures],
        scenes: list[SceneSegment],
    ) -> bool:
        """Recover a match globally mislabeled as replay by banner graphics.

        Replay broadcasts are short portions of a match.  If almost every
        observation is marked replay while a substantial share still contains
        main-table levels of green cloth, the label came from a persistent
        tournament banner rather than a replay transition.  This also repairs
        checkpoints created by the older classifier without repeating the
        expensive video scan.
        """

        if not features:
            return False
        replay_views = {
            CameraViewType.REPLAY,
            CameraViewType.SLOW_MOTION_REPLAY,
        }
        replay_fraction = sum(
            feature.view_type in replay_views for feature in features
        ) / len(features)
        main_ratio = float(
            self.config.get("camera_view.table_green_ratio_main", 0.18)
        )
        main_like_fraction = sum(
            feature.green_ratio >= main_ratio for feature in features
        ) / len(features)
        if replay_fraction < 0.90 or main_like_fraction < 0.20:
            return False

        for feature in features:
            if feature.green_ratio >= main_ratio:
                feature.view_type = CameraViewType.MAIN_TABLE
        for scene in scenes:
            if scene.table_ratio >= main_ratio:
                scene.view_type = CameraViewType.MAIN_TABLE
                scene.is_replay_candidate = False

        logger.warning(
            "Repaired pathological replay classification "
            "(%.1f%% replay, %.1f%% main-table observations)",
            replay_fraction * 100.0,
            main_like_fraction * 100.0,
        )
        return True

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        temporary.replace(path)

    def _coarse_cache_path(self) -> Path:
        return self.job_dir / "coarse_checkpoint.json"

    def _save_coarse_cache(
        self,
        signature: str,
        features: list[FrameFeatures],
        scenes: list[SceneSegment],
        candidates: list[StrikeCandidate],
    ) -> None:
        payload = {
            "cache_version": _CACHE_VERSION,
            "signature": signature,
            # The identity belongs to these source/configuration observations.
            # Commit it atomically with the matching coarse cache so a crash
            # cannot pair new observations with another source's old target.
            "broadcast_target": self.broadcast_context.export_target(),
            "confirmed_target_returns": self._confirmed_target_returns,
            "features": [item.model_dump(mode="json") for item in features],
            "scenes": [item.model_dump(mode="json") for item in scenes],
            "candidates": [item.model_dump(mode="json") for item in candidates],
        }
        try:
            self._write_json_atomic(self._coarse_cache_path(), payload)
            self._write_checkpoint(
                {
                    "stage": "coarse_complete",
                    "signature": signature,
                    "features": len(features),
                    "candidates": len(candidates),
                }
            )
        except OSError as exc:  # analysis can continue without a cache
            logger.warning("Could not save coarse checkpoint: %s", exc)

    def _load_coarse_cache(
        self, signature: str
    ) -> tuple[
        list[FrameFeatures], list[SceneSegment], list[StrikeCandidate]
    ] | None:
        path = self._coarse_cache_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_version") != _CACHE_VERSION
                or payload.get("signature") != signature
            ):
                logger.info("Ignoring stale coarse checkpoint")
                return None
            features = [
                FrameFeatures.model_validate(item)
                for item in payload.get("features", [])
            ]
            scenes = [
                SceneSegment.model_validate(item)
                for item in payload.get("scenes", [])
            ]
            candidates = [
                StrikeCandidate.model_validate(item)
                for item in payload.get("candidates", [])
            ]
            if not features:
                return None
            # Older caches may have no target. Never fall back to the former
            # unbound broadcast_target.json file, nor keep a previous source's
            # template on a reused Analyzer instance.
            self.broadcast_context.reset_observations(preserve_target=False)
            self.broadcast_context.restore_target(payload.get("broadcast_target", []))
            self._confirmed_target_returns = [tuple(span) for span in payload.get("confirmed_target_returns", [])
                                              if len(span) == 2 and 0 < span[1]-span[0] <= 15]
            logger.info(
                "Loaded coarse checkpoint (%d features, %d candidates)",
                len(features),
                len(candidates),
            )
            return features, scenes, candidates
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not load coarse checkpoint: %s", exc)
            return None

    def _dense_cache_path(self, index: int) -> Path:
        return self.job_dir / "refinement_cache" / f"window_{index:04d}.json"

    def _save_dense_window(
        self,
        signature: str,
        index: int,
        start: float,
        end: float,
        features: list[FrameFeatures],
    ) -> None:
        payload = {
            "cache_version": _CACHE_VERSION,
            "signature": signature,
            "start": start,
            "end": end,
            "features": [item.model_dump(mode="json") for item in features],
        }
        try:
            self._write_json_atomic(self._dense_cache_path(index), payload)
        except OSError as exc:
            logger.warning("Could not save refinement window %d: %s", index + 1, exc)

    def _load_dense_window(
        self, signature: str, index: int, start: float, end: float
    ) -> list[FrameFeatures] | None:
        path = self._dense_cache_path(index)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_version") != _CACHE_VERSION
                or payload.get("signature") != signature
                or abs(float(payload.get("start", -1.0)) - start) > 1e-6
                or abs(float(payload.get("end", -1.0)) - end) > 1e-6
            ):
                return None
            return [
                FrameFeatures.model_validate(item)
                for item in payload.get("features", [])
            ]
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Could not load refinement window %d: %s", index + 1, exc
            )
            return None

    def _cue_geometry(
        self,
        frame: np.ndarray,
        table_mask: Optional[np.ndarray],
        cue,
        diameter: float,
        t: float,
    ) -> dict[str, float | bool]:
        """Find optional cue-stick/tip evidence near the tracked white ball.

        This is intentionally a supporting signal, not a standalone detector:
        a line must be long, thin, inside the table, and pass through the ball
        edge.  If it is absent/occluded, stationary-ball acceleration remains the
        only admissible fallback and the resulting shot is reviewable.
        """
        empty: dict[str, float | bool] = {
            "visible": False,
            "distance": 0.0,
            "approach_speed": 0.0,
            "forward_motion": 0.0,
            "contact_score": 0.0,
        }
        if cue is None or not cue.visible or diameter <= 1.0:
            self._last_cue_tip = None
            return empty
        cx, cy = float(cue.positions[-1][1]), float(cue.positions[-1][2])
        h, w = frame.shape[:2]
        large_ball = diameter > 45
        # A 145px close-up white used to require a 507px line inside a 361px
        # crop. Show enough shaft around large balls, while keeping work bounded.
        half = int(np.clip(diameter * (3.0 if large_ball else 10.0), 45,
                           360 if large_ball else 180))
        x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
        y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
        if x1 <= x0 or y1 <= y0:
            self._last_cue_tip = None
            return empty
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if table_mask is not None and table_mask.shape[:2] == frame.shape[:2]:
            local_mask = table_mask[y0:y1, x0:x1]
            crop = cv2.bitwise_and(crop, crop, mask=local_mask)
        edges = cv2.Canny(crop, 50, 150, apertureSize=3)
        min_len = max(18, int(round(diameter * (1.0 if large_ball else 3.5))))
        lines = cv2.HoughLinesP(
            edges,
            1.0,
            np.pi / 180.0,
            threshold=max(12, int(round(min(diameter * 1.4, min_len * .45)))),
            minLineLength=min_len,
            maxLineGap=max(3, int(round(min(diameter * .8, min_len * .15)))) if large_ball
            else max(3, int(round(diameter * .8))),
        )
        if lines is None:
            self._last_cue_tip = None
            return empty

        best: tuple[float, float, float] | None = None  # score, tip_x, tip_y
        local_cx, local_cy = cx - x0, cy - y0
        for line in lines[:, 0, :]:
            ax, ay, bx, by = [float(v) for v in line]
            vx, vy = bx - ax, by - ay
            length = float(np.hypot(vx, vy))
            if length < min_len:
                continue
            # Distance from the ball centre to the line and to its nearest end.
            cross = abs(vx * (local_cy - ay) - vy * (local_cx - ax)) / max(length, 1e-6)
            da = float(np.hypot(local_cx - ax, local_cy - ay))
            db = float(np.hypot(local_cx - bx, local_cy - by))
            nearest = (ax, ay) if da <= db else (bx, by)
            endpoint_dist = min(da, db)
            if cross > diameter * (.35 if large_ball else 1.35) or endpoint_dist > diameter * (2.0 if large_ball else 3.2):
                continue
            if large_ball and (endpoint_dist < .30*diameter
                               or max(da, db) < endpoint_dist+.75*diameter):
                continue
            # The far endpoint must extend away from the ball; this rejects
            # short cushion/scoreboard edges crossing the crop.
            score = (length / max(diameter, 1.0)) * np.exp(-cross / max(diameter, 1.0))
            if best is None or score > best[0]:
                best = (score, nearest[0] + x0, nearest[1] + y0)
        if best is None:
            self._last_cue_tip = None
            return empty

        _, tip_x, tip_y = best
        distance = float(np.hypot(tip_x - cx, tip_y - cy))
        approach = 0.0
        forward = 0.0
        if self._last_cue_tip is not None:
            prev_x, prev_y, prev_t = self._last_cue_tip
            dt = max(1e-3, t - prev_t)
            prev_dist = float(np.hypot(prev_x - cx, prev_y - cy))
            approach = max(0.0, (prev_dist - distance) / dt / max(diameter, 1.0))
            forward = float(np.hypot(tip_x - prev_x, tip_y - prev_y) / dt / max(diameter, 1.0))
        self._last_cue_tip = (tip_x, tip_y, t)
        contact = float(
            np.clip(
                0.55 * np.clip((diameter * 1.25 - distance) / max(diameter, 1.0), 0, 1)
                + 0.45 * np.clip(approach / 2.0, 0, 1),
                0,
                1,
            )
        )
        return {
            "visible": True,
            "distance": distance,
            "approach_speed": approach,
            "forward_motion": forward,
            "contact_score": contact,
        }

    def _score_importance(self, shots, features):
        cfg = self.config.section("importance")
        w_multi = float(cfg.get("multi_ball_weight", 0.15))
        w_long = float(cfg.get("long_travel_weight", 0.1))
        w_replay = float(cfg.get("replay_shown_weight", 0.25))
        w_energy = float(cfg.get("motion_energy_weight", 0.2))
        times = [feature.t for feature in features]

        for s in shots:
            lo = bisect_left(times, s.clip_start)
            hi = bisect_right(times, s.clip_end)
            window = features[lo:hi]
            if not window:
                s.importance = 0.3
                continue
            energy = float(np.mean([f.motion_score for f in window]))
            multi = float(np.mean([f.motion_area_ratio for f in window]))
            travel = min(1.0, (s.ball_motion_end - s.ball_motion_start) / 15.0)
            replay = 1.0 if s.possible_replay else 0.0
            s.importance = float(
                np.clip(
                    w_energy * energy
                    + w_multi * min(1.0, multi / 0.02)
                    + w_long * travel
                    + w_replay * replay,
                    0,
                    1,
                )
            )
        return shots

    def _rebuild_segments(self, result: AnalysisResult, mode: EditMode) -> AnalysisResult:
        previous_shots = list(result.shots)
        shots = self.segmenter.build(
            result.strike_candidates,
            result.features,
            result.original_duration,
            mode,
        )
        # A mode change rebuilds boundaries, but it must not silently undo the
        # editor's include/exclude decisions. Match by strike time because shot
        # IDs can be renumbered when candidates are added or removed.
        unmatched = set(range(len(previous_shots)))
        for shot in shots:
            best_idx = min(
                unmatched,
                key=lambda idx: abs(
                    previous_shots[idx].cue_strike_timestamp - shot.cue_strike_timestamp
                ),
                default=None,
            )
            if best_idx is None:
                continue
            previous = previous_shots[best_idx]
            if abs(previous.cue_strike_timestamp - shot.cue_strike_timestamp) > 1.5:
                continue
            unmatched.remove(best_idx)
            if previous.user_modified:
                shot.included = previous.included
                shot.user_modified = True
                shot.linked_live_shot_id = previous.linked_live_shot_id
        shots = self._score_importance(shots, result.features)
        edited, removed = self.segmenter.recompute_durations(shots, result.original_duration)
        result.shots = shots
        result.events = []
        for shot in shots:
            result.events.append(
                TimelineEvent(
                    event_type="cue_strike",
                    timestamp=shot.cue_strike,
                    confidence=shot.shot_confidence,
                    metadata={
                        "shot_id": shot.shot_id,
                        "clip_start_timestamp": shot.clip_start,
                        "strike_confidence": shot.strike_confidence,
                    },
                )
            )
            result.events.append(
                TimelineEvent(
                    event_type="ball_stop",
                    timestamp=shot.physical_stop_timestamp,
                    confidence=shot.end_confidence,
                    metadata={
                        "shot_id": shot.shot_id,
                        "last_ball_motion_timestamp": shot.last_ball_motion_timestamp,
                        "physical_stop_timestamp": shot.physical_stop_timestamp,
                        "stop_confirmation_timestamp": shot.stop_confirmation_timestamp,
                        "stop_confidence": shot.stop_confidence,
                    },
                )
            )
        result.mode = mode
        result.edited_duration = edited
        result.pause_removed_seconds = removed
        return result

    def resegment(self, result: AnalysisResult, mode: EditMode) -> AnalysisResult:
        result = self._rebuild_segments(result, mode)
        self._save_result(result)
        return result

    def _save_result(self, result: AnalysisResult) -> None:
        path = self.job_dir / "analysis.json"
        # Full analysis (includes per-frame features for resume / resegment)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        # Slim timeline for the review UI (no dense feature vectors)
        slim = {
            "job_id": result.job_id,
            "source_path": result.source_path,
            "mode": result.mode.value,
            "original_duration": result.original_duration,
            "edited_duration": result.edited_duration,
            "pause_removed_seconds": result.pause_removed_seconds,
            "shots": [s.model_dump() for s in result.shots],
            "events": [e.model_dump() for e in result.events],
            "scenes": [s.model_dump() for s in result.scenes],
            "metadata": result.metadata.model_dump(),
            "strike_candidates": [c.model_dump() for c in result.strike_candidates],
            "analysis_version": result.analysis_version,
            "analysis_signature": result.analysis_signature,
        }
        (self.job_dir / "timeline.json").write_text(
            json.dumps(slim, indent=2), encoding="utf-8"
        )
        logger.info("Saved analysis to %s (%d shots)", path, len(result.shots))

    def _write_checkpoint(self, data: dict) -> None:
        data["updated_at"] = time.time()
        (self.job_dir / "checkpoint.json").write_text(
            json.dumps(data), encoding="utf-8"
        )


    def _preserve_user_edits(
        self, new_shots: list[ShotRecord], job_id: str
    ) -> list[ShotRecord]:
        previous_shots: list[ShotRecord] = []
        corrections_path = self.job_dir / "corrections.json"
        timeline_path = self.job_dir / "timeline.json"
        analysis_path = self.job_dir / "analysis.json"
        deleted_strikes: list[float] = []

        if corrections_path.exists():
            try:
                data = json.loads(corrections_path.read_text(encoding="utf-8"))
                previous_shots = [ShotRecord.model_validate(s) for s in data.get("shots", [])]
                deleted_strikes = [float(t) for t in data.get("deleted_strikes", [])]
            except Exception as exc:
                logger.warning("Could not read corrections.json for job %s: %s", job_id, exc)
        elif timeline_path.exists():
            try:
                data = json.loads(timeline_path.read_text(encoding="utf-8"))
                previous_shots = [ShotRecord.model_validate(s) for s in data.get("shots", [])]
            except Exception as exc:
                logger.warning("Could not read timeline.json for job %s: %s", job_id, exc)
        elif analysis_path.exists():
            try:
                data = json.loads(analysis_path.read_text(encoding="utf-8"))
                res = AnalysisResult.model_validate(data)
                previous_shots = res.shots
            except Exception as exc:
                logger.warning("Could not read analysis.json for job %s: %s", job_id, exc)

        new_shots = [
            shot for shot in new_shots
            if not any(abs(shot.cue_strike_timestamp - t) <= 1.5 for t in deleted_strikes)
        ]
        if not previous_shots:
            for i, shot in enumerate(new_shots, start=1):
                shot.shot_id = i
            return new_shots

        unmatched = set(range(len(previous_shots)))
        shots = list(new_shots)

        for shot_index, shot in enumerate(shots):
            best_idx = min(
                unmatched,
                key=lambda idx: abs(
                    previous_shots[idx].cue_strike_timestamp - shot.cue_strike_timestamp
                ),
                default=None,
            )
            if best_idx is None:
                continue
            previous = previous_shots[best_idx]
            if abs(previous.cue_strike_timestamp - shot.cue_strike_timestamp) > 1.5:
                continue
            unmatched.remove(best_idx)
            if previous.user_modified:
                shots[shot_index] = previous.model_copy(
                    update={"shot_id": shot.shot_id}, deep=True
                )

        for idx in sorted(unmatched):
            prev = previous_shots[idx]
            if prev.user_modified:
                shots.append(prev)

        shots.sort(key=lambda s: s.clip_start)
        for i, s in enumerate(shots, start=1):
            s.shot_id = i
        return shots
