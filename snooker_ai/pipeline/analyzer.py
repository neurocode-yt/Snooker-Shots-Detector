"""
Main analysis pipeline.

Phase 1 flow:
  validate → proxy → sample frames → table/motion/scene/audio features
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

from snooker_ai.audio.features import AudioFeatureExtractor
from snooker_ai.config import Config
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.ingestion.probe import validate_video
from snooker_ai.ingestion.proxy import generate_proxy
from snooker_ai.object_detection.detector import ObjectDetector
from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.detector import SceneDetector, SceneObservation
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.table_detection.localizer import TableLocalizer
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
    StrikeCandidate,
    TimelineEvent,
)
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.acceleration import configure_acceleration
from snooker_ai.utils.timebase import TimeMapper

logger = get_logger("pipeline")

ProgressCb = Callable[[float, str, str], None]
_CACHE_VERSION = 2


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
        self.audio_ext = AudioFeatureExtractor(config)
        self.strike_det = StrikeDetector(config)
        self.replay_det = ReplayDetector(config)
        self.state_machine = ShotStateMachine(config)
        self.segmenter = SegmentBuilder(config)
        self.objects = ObjectDetector(config)
        self.tracker = BallTracker()
        self._audio_feature_cache: dict[str, object] = {}
        self._last_cue_tip: Optional[tuple[float, float, float]] = None

    def analyze(
        self,
        source: str | Path,
        job_id: str,
        mode: EditMode = EditMode.STRICT,
        progress: Optional[ProgressCb] = None,
        resume: bool = True,
    ) -> AnalysisResult:
        def report(p: float, stage: str, msg: str = "") -> None:
            if progress:
                progress(p, stage, msg)
            logger.info("[%.0f%%] %s — %s", p * 100, stage, msg)

        checkpoint_path = self.job_dir / "checkpoint.json"
        analysis_path = self.job_dir / "analysis.json"

        # Resume completed analysis
        if resume and analysis_path.exists():
            try:
                data = json.loads(analysis_path.read_text(encoding="utf-8"))
                result = AnalysisResult.model_validate(data)
                if result.mode != mode:
                    result = self._rebuild_segments(result, mode)
                    self._save_result(result)
                report(1.0, JobStatus.READY_FOR_REVIEW.value, "Resumed from saved analysis")
                return result
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
            report(0.2, JobStatus.ANALYZING.value, "Extracting multimodal features")
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
        )
        if dense_features:
            self._annotate_scenes(dense_features, scenes)
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
            candidates = [
                candidate
                for candidate in candidates
                if candidate.evidence.get("dense_transition_confirmed", 0.0) >= 0.5
                or candidate.evidence.get("sparse_dense_transition", 0.0) >= 0.5
                or (
                    candidate.evidence.get("occlusion_inferred", 0.0) >= 0.5
                    and candidate.evidence.get("ball_onset_run", 0.0) >= 2.0
                )
            ]
            candidates = self.replay_det.mark_candidates(candidates, features)
            features = self.state_machine.label(features)

        report(0.9, JobStatus.SEGMENTING.value, "Building shot segments")
        shots = self.segmenter.build(candidates, features, metadata.duration, mode)
        shots = self._score_importance(shots, features)

        edited, removed = self.segmenter.recompute_durations(shots, metadata.duration)

        events: list[TimelineEvent] = []
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
        )
        self._save_result(result)
        if checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)

        report(1.0, JobStatus.READY_FOR_REVIEW.value, f"Detected {len(shots)} shots")
        return result

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
    ) -> tuple[list[FrameFeatures], list[SceneObservation], list[float]]:
        sample_fps = float(sample_fps or self.config.get("analysis.sample_fps", 10.0))
        start_time = max(0.0, float(start_time))
        end_time = duration if end_time is None else min(duration, float(end_time))
        audio_key = str(audio_path) if audio_path else ""
        if audio_key and audio_key in self._audio_feature_cache:
            audio_feats = self._audio_feature_cache[audio_key]
        else:
            audio_feats = self.audio_ext.extract(audio_path)
            if audio_key:
                self._audio_feature_cache[audio_key] = audio_feats

        cap = cv2.VideoCapture(str(proxy_path))
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
        prev_sample_t: Optional[float] = None
        idx = 0
        kept = 0
        scene_step = max(1, int(round(sample_fps / 2)))  # ~2 fps for scene detect
        refine_fps = float(self.config.get("analysis.refine_fps", 30.0))
        dense_pass = sample_fps >= refine_fps * 0.90
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
        next_sample_t = start_time
        # Seek close to the requested source interval for dense refinement.  We
        # still discard frames until the mapped presentation time reaches the
        # exact start boundary, so VFR/mapper rounding cannot leak earlier data.
        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, mapper.to_proxy(start_time) * 1000.0)
            idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))

        self.table.reset()
        self.motion.reset()
        self.tracker = BallTracker()
        self._last_cue_tip = None

        while True:
            video_t = idx / proxy_fps if proxy_fps > 0 else kept / sample_fps
            source_t = mapper.to_source(video_t)
            # Skip retrieval/conversion for unsampled proxy frames.  ``grab``
            # still advances the decoder correctly but avoids constructing a
            # full BGR array for the two out of every three frames discarded by
            # the normal 30fps-proxy/10fps-analysis path.
            if source_t < start_time - 1e-6:
                if not cap.grab():
                    break
                idx += 1
                continue
            if source_t > end_time + 1e-6:
                break
            if source_t + 1e-9 < next_sample_t:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            while next_sample_t <= source_t + 1e-9:
                next_sample_t += sample_period

            t = source_t
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect cuts before optical flow/tracking.  A cut is an unknown
            # observation, never a stationary frame, and view-local trackers are
            # reacquired without terminating the logical shot.
            hist = self.scene_det.histogram(frame)
            online_cut = 0.0
            if prev_hist is not None:
                online_cut = self.scene_det.cut_score_simple(prev_hist, hist)
            cut_like = online_cut >= float(
                self.config.get("scene_detection.hard_cut_threshold", 0.42)
            )
            if cut_like:
                self.table.reset()
                self.motion.reset()
                self.tracker = BallTracker()
                self._last_cue_tip = None
                prev_gray = None
                prev_sample_t = None
                table_obs = None
                last_table_t = None

            if (
                table_obs is None
                or last_table_t is None
                or t - last_table_t + 1e-9 >= table_refresh_period
            ):
                table_obs = self.table.detect(frame)
                last_table_t = t
            dets = self.objects.detect(
                frame,
                table_obs.mask,
                use_hough=kept % hough_step == 0,
            )
            ball_regions = [
                (d.cx, d.cy, d.diameter_px)
                for d in dets
                if d.confidence >= 0.30 and d.diameter_px > 1.0
            ]
            residual = None
            if prev_gray is not None:
                dt = t - prev_sample_t if prev_sample_t is not None else 1.0 / max(sample_fps, 1.0)
                residual = self.motion.analyze(
                    prev_gray,
                    gray,
                    table_obs.mask,
                    ball_regions=ball_regions,
                    frame_dt=dt,
                )
            prev_gray = gray
            prev_sample_t = t
            prev_hist = hist
            tracks = self.tracker.update(t, dets)
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
                and self.tracker.stable_track_speed(tr, diameter) >= stop_speed
            )
            cue_speed_px = cue.speed() if cue is not None and cue.visible else 0.0
            cue_speed_norm = cue_speed_px / diameter if cue is not None and diameter > 0.5 else 0.0
            cue_accel_norm = (
                float(np.hypot(cue.ax, cue.ay) / diameter)
                if cue is not None and cue.visible and diameter > 0.5
                else 0.0
            )
            cue_geometry = self._cue_geometry(frame, table_obs.mask, cue, diameter, t)
            table_observable = bool(
                table_obs.confidence >= float(
                    self.config.get("table_detection.min_confidence", 0.25)
                )
                and table_obs.area_ratio >= 0.03
            )
            observation_valid = bool(
                table_observable
                and not cut_like
                and (residual is None or residual.observation_valid)
            )

            audio_onset = audio_feats.peak_near(t, 0.12) if audio_feats else 0.0
            audio_rms = audio_feats.value_at(t, audio_feats.rms) if audio_feats else 0.0
            audio_hi = audio_feats.value_at(t, audio_feats.highband) if audio_feats else 0.0

            from snooker_ai.types import CameraViewType

            feat = FrameFeatures(
                t=t,
                table_confidence=table_obs.confidence,
                table_mask_area_ratio=table_obs.area_ratio,
                residual_motion_mean=residual.residual_mean if residual else 0.0,
                residual_motion_max=residual.residual_max if residual else 0.0,
                motion_area_ratio=residual.motion_area_ratio if residual else 0.0,
                camera_motion_magnitude=residual.camera_magnitude if residual else 0.0,
                view_type=CameraViewType.OTHER,
                green_ratio=table_obs.area_ratio,
                audio_onset=audio_onset,
                audio_rms=audio_rms,
                audio_highband=audio_hi,
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
                cue_ball_acceleration=cue_accel_norm,
                cue_ball_track_confidence=(cue.confidence if cue is not None and cue.visible else 0.0),
                cue_tip_visible=bool(cue_geometry["visible"]),
                cue_tip_distance_to_ball=float(cue_geometry["distance"]),
                cue_approach_speed=float(cue_geometry["approach_speed"]),
                cue_forward_motion=float(cue_geometry["forward_motion"]),
                cue_contact_score=float(cue_geometry["contact_score"]),
                max_ball_normalized_speed=self.tracker.max_normalized_speed(diameter),
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
            features.append(feat)

            if scene_stream is not None and kept % scene_step == 0:
                scene_stream.observe(frame, t, histogram=hist)

            kept += 1
            idx += 1
            if progress and kept % 20 == 0:
                interval = max(end_time - start_time, 1e-6)
                fraction = (t - start_time) / interval
                progress(
                    min(0.99, max(0.0, fraction)),
                    f"Frame {idx}/{total_frames}",
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

        cap.release()
        if progress:
            progress(1.0, f"Sampled {len(features)} frames")
        observations = scene_stream.observations if scene_stream is not None else []
        return features, observations, []

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
        dense_fps = float(
            self.config.get(
                "analysis.refine_fps",
                self.config.get("analysis.sample_fps", 10.0),
            )
        )
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
        ranges: list[tuple[float, float]] = []
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
                    ranges.append((start, end))
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
                ranges.append((start, end))

        # Closely spaced shots often have overlapping strike/settling windows.
        # Decode and analyze each combined interval once instead of seeking,
        # rebuilding the tracker, and recomputing flow for every candidate.
        # This preserves the exact dense observations while cutting duplicated
        # work on full matches with clusters of safety exchanges.
        merged_ranges: list[tuple[float, float]] = []
        for start, end in sorted(ranges):
            if merged_ranges and start <= merged_ranges[-1][1] + merge_gap:
                previous_start, previous_end = merged_ranges[-1]
                merged_ranges[-1] = (previous_start, max(previous_end, end))
            else:
                merged_ranges.append((start, end))

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
        for range_idx, (start, end) in enumerate(merged_ranges):
            try:
                part = (
                    self._load_dense_window(
                        signature, range_idx, start, end
                    )
                    if resume and signature
                    else None
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
                        sample_fps=dense_fps,
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

    @staticmethod
    def _merge_feature_layers(
        coarse: list[FrameFeatures], dense: list[FrameFeatures]
    ) -> list[FrameFeatures]:
        by_time: dict[int, FrameFeatures] = {
            int(round(f.t * 10000.0)): f for f in coarse
        }
        for f in dense:
            by_time[int(round(f.t * 10000.0))] = f
        return [by_time[key] for key in sorted(by_time)]

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
                # Retain useful sparse evidence when a dense candidate replaces
                # it, while letting the dense timestamp/confidence win.
                candidate.evidence = {
                    **previous.evidence,
                    **candidate.evidence,
                }
                kept[-1] = candidate
            else:
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
                feature.view_type = scene.view_type

            cut_idx = bisect_left(cut_times, feature.t)
            near_cut = any(
                abs(feature.t - cut_times[index]) < 0.08
                for index in (cut_idx - 1, cut_idx)
                if 0 <= index < len(cut_times)
            )
            if near_cut:
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
        half = int(np.clip(diameter * 10.0, 45, 180))
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
        min_len = max(18, int(round(diameter * 3.5)))
        lines = cv2.HoughLinesP(
            edges,
            1.0,
            np.pi / 180.0,
            threshold=max(12, int(round(diameter * 1.4))),
            minLineLength=min_len,
            maxLineGap=max(3, int(round(diameter * 0.8))),
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
            if cross > diameter * 1.35 or endpoint_dist > diameter * 3.2:
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
        w_crowd = float(cfg.get("crowd_reaction_weight", 0.2))
        w_multi = float(cfg.get("multi_ball_weight", 0.15))
        w_long = float(cfg.get("long_travel_weight", 0.1))
        w_replay = float(cfg.get("replay_shown_weight", 0.25))
        w_energy = float(cfg.get("motion_energy_weight", 0.2))
        w_audio = float(cfg.get("audio_excitement_weight", 0.1))
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
            audio = float(np.mean([f.audio_rms for f in window]))
            travel = min(1.0, (s.ball_motion_end - s.ball_motion_start) / 15.0)
            replay = 1.0 if s.possible_replay else 0.0
            # crowd approx: high rms after ball stop
            after_lo = bisect_left(times, s.ball_motion_end)
            after_hi = bisect_right(times, s.ball_motion_end + 3.0)
            after = features[after_lo:after_hi]
            crowd = float(np.mean([f.audio_rms for f in after])) if after else 0.0
            s.importance = float(
                np.clip(
                    w_energy * energy
                    + w_multi * min(1.0, multi / 0.02)
                    + w_long * travel
                    + w_replay * replay
                    + w_audio * audio
                    + w_crowd * crowd,
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
            shot.included = previous.included
            shot.user_modified = previous.user_modified
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
