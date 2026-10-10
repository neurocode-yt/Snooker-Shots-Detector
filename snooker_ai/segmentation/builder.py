"""Build shot records while preserving exact strict-mode boundaries."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from snooker_ai.config import Config
from snooker_ai.event_fusion.ball_stop import BallStopDetector, StopDetection
from snooker_ai.types import (
    CameraViewType,
    ConfidenceLevel,
    EditMode,
    FrameFeatures,
    ShotRecord,
    StrikeCandidate,
)
from snooker_ai.utils.logging import get_logger
from snooker_ai.utils.timebase import clamp

logger = get_logger("segmentation")


class SegmentBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.ball_stop = BallStopDetector(config)
        conf = config.section("confidence")
        self.high = float(conf.get("high", 0.70))
        self.medium = float(conf.get("medium", 0.50))
        self.low = float(conf.get("low", 0.35))
        self.fail_safe = float(conf.get("fail_safe_keep_extra_seconds", 1.0))
        mode_cfg = config.mode_settings(EditMode.STRICT)
        self.strict_pre_roll = float(mode_cfg.get("pre_roll", 2.0))
        self.min_shot_spacing = float(mode_cfg.get("min_shot_spacing_seconds", 4.0))
        self.conflict_confidence_margin = float(
            config.get("strike_fusion.conflict_confidence_margin", 0.05)
        )
        self.support_quiet_ratio = float(
            config.get("strike_fusion.fallback_pre_strike_ball_quiet_min_ratio", 0.50)
        )

    def _level(self, score: float) -> ConfidenceLevel:
        if score >= self.high:
            return ConfidenceLevel.HIGH
        if score >= self.medium:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _strict_start(self, strike_t: float, duration: float) -> float:
        """The immutable strict start rule from the editing contract."""
        return clamp(float(strike_t) - self.strict_pre_roll, 0.0, duration)

    def build(
        self,
        candidates: list[StrikeCandidate],
        features: list[FrameFeatures],
        duration: float,
        mode: EditMode = EditMode.STRICT,
    ) -> list[ShotRecord]:
        mode_cfg = self.config.mode_settings(EditMode.STRICT)
        retain_replays = bool(mode_cfg.get("retain_replays", False))
        minimum_clip = max(0.0, float(mode_cfg.get("minimum_clip_seconds", 0.0)))
        minimum_visibility = max(0.0, float(mode_cfg.get("minimum_strike_visibility_seconds", 0.1)))
        transition = (
            max(0.0, float(self.config.get("export.transition_seconds", 0.24)))
            if self.config.get("export.transition", "cut") == "mix" and minimum_clip > 0
            else 0.0
        )

        # Remove sub-frame/nearby duplicates before stop searches.  This never
        # changes the winning candidate's timestamp.
        ordered = self._deduplicate_candidates(candidates)
        shots: list[ShotRecord] = []
        feature_times = [feature.t for feature in features]
        unusable_spans = self._unusable_spans(features)
        usable_boundaries = sorted((start, reason) for start, _, reason in unusable_spans)
        boundary_times = [t for t, _ in usable_boundaries]

        for candidate_index, cand in enumerate(ordered):
            contact_index = bisect_left(feature_times, cand.timestamp)
            near = features[max(0, contact_index-1):contact_index+2]
            contact = min(near, key=lambda f: abs(f.t-cand.timestamp), default=None)
            if contact is not None and abs(contact.t-cand.timestamp) <= 0.2 and (
                not contact.match_context_valid or contact.table_handling
            ):
                continue
            # A separately verified next strike bounds unresolved tracking.
            # Sparse proposals and replay events cannot supply this boundary.
            next_strike = next((later.timestamp for later in ordered[candidate_index + 1:]
                                if not later.possible_replay
                                and self._candidate_supported(later)), None)
            stop = self.ball_stop.detect_stop(
                cand, features, duration, times=feature_times,
                next_strike_timestamp=next_strike,
            )
            refined_end = float(cand.evidence.get("refined_stop_timestamp", 0.0))
            refined_confirmation = float(cand.evidence.get("refined_stop_confirmation_timestamp", 0.0))
            refined_confidence = float(cand.evidence.get("refined_stop_confidence", 0.0))
            if (
                cand.timestamp <= refined_end <= refined_confirmation <= duration
                and refined_confirmation - refined_end + 1e-6 >= self.ball_stop.confirm_s
                and (next_strike is None or refined_confirmation < next_strike)
                and (refined_confidence >= 0.70 or cand.evidence.get("refined_stop_upper_bound", 0) >= 0.5)
            ):
                refined_motion_start = float(cand.evidence.get("refined_ball_motion_start", cand.timestamp))
                # A separate end-window tracker may reacquire the rolling ball
                # later than the native contact observations. Preserve that
                # earlier measured onset when adopting its refined stop; a
                # default/unconfirmed onset supplies no such evidence.
                if stop.start_confidence >= .70:
                    refined_motion_start = min(refined_motion_start, stop.motion_start)
                stop = StopDetection(
                    motion_start=refined_motion_start,
                    last_ball_motion_timestamp=float(cand.evidence.get("refined_last_motion_timestamp", refined_end)),
                    physical_stop_timestamp=refined_end,
                    stop_confirmation_timestamp=refined_confirmation,
                    end_confidence=refined_confidence, start_confidence=stop.start_confidence,
                    confirmed=True,
                    manual_review_required=bool(cand.evidence.get("refined_stop_review_required", 0)),
                    reason=("confirmed_stationary_after_unseen_interval_upper_bound"
                            if cand.evidence.get("refined_stop_upper_bound", 0) >= 0.5
                            else "confirmed_native_stop"),
                )

            # A practice stroke/feathering candidate with no sustained ball
            # motion is not a shot.  Ambiguous cases that did show movement are
            # retained conservatively and flagged by StopDetection.
            if stop.reason == "no_sustained_ball_motion":
                logger.debug("Rejected strike %.3f: no sustained ball motion", cand.timestamp)
                continue
            # A proposal cannot borrow movement from a later actual shot. This
            # also prevents a false preparation event from suppressing that
            # later shot as an apparent mid-roll duplicate during overlap repair.
            onset_timeout = float(self.config.get("strike_fusion.strike_candidate_timeout_seconds", 0.75))
            if stop.motion_start > cand.timestamp + onset_timeout:
                logger.debug("Rejected strike %.3f: ball motion starts too late (%.3f)",
                             cand.timestamp, stop.motion_start)
                continue

            physical_stop = stop.physical_stop_timestamp
            uncapped_physical_stop = physical_stop
            confirmation = stop.stop_confirmation_timestamp
            last_motion = stop.last_ball_motion_timestamp
            end_confidence = stop.end_confidence
            stop_confirmed = stop.confirmed
            stop_reason = stop.reason
            stop_review = stop.manual_review_required
            minimum_clip_end = physical_stop

            clip_start = self._strict_start(cand.timestamp, duration)
            usable_start = 0.0
            preparation_boundary_reason = ""
            for _, end, reason in unusable_spans:
                if usable_start < end <= cand.timestamp:
                    usable_start = end
                    preparation_boundary_reason = reason
            clip_start = max(clip_start, usable_start)
            prep_start = clip_start
            # The strict contract uses an actual physical stop whenever it
            # is observed early.  If tracking remains unresolved, cap the
            # exported shot at the configured post-strike horizon and mark
            # the boundary for review instead of allowing a long false
            # track to run away.
            max_after = mode_cfg.get("max_seconds_after_strike")
            if max_after is not None:
                shot_cap = clamp(
                    cand.timestamp + max(0.0, float(max_after)),
                    clip_start,
                    duration,
                )
                if physical_stop > shot_cap + 1e-9:
                    physical_stop = shot_cap
                    confirmation = shot_cap
                    last_motion = min(last_motion, shot_cap)
                    end_confidence = min(end_confidence, 0.20)
                    stop_confirmed = False
                    stop_reason = "max_seconds_after_strike_review_cap"
                    stop_review = True
            # Confirmation is look-ahead only.  No pad or confidence tail is
            # allowed to alter this boundary. A separate practical clip limit
            # prevents one unresolved track from producing a 40–50 second
            # segment.
            max_clip = float(mode_cfg.get("max_clip_seconds", 9.0))
            clip_cap = clamp(clip_start + max_clip, clip_start, duration)
            availability_reason = ""
            boundary_index = bisect_right(boundary_times, cand.timestamp)
            if boundary_index < len(usable_boundaries):
                boundary_t, boundary_reason = usable_boundaries[boundary_index]
                if boundary_t < clip_cap:
                    clip_cap = boundary_t
                    availability_reason = boundary_reason
            if stop.reason == "unconfirmed_ball_handling_boundary":
                clip_cap = min(clip_cap, physical_stop)
            visual_cap = float(cand.evidence.get("visual_hand_entry_clip_cap_timestamp", 0.))
            visual_entry = float(cand.evidence.get("visual_hand_entry_timestamp", 0.))
            visual_confirmation = float(cand.evidence.get("visual_hand_entry_confirmation_timestamp", 0.))
            visual_floor = max(clip_start + minimum_clip,
                               cand.timestamp + minimum_visibility)
            if (cand.evidence.get("visual_hand_entry_confidence", 0.) >= .8
                    and visual_floor-1e-6 <= visual_cap < visual_entry < visual_confirmation <= duration
                    and visual_cap < clip_cap):
                clip_cap = visual_cap
                availability_reason = "visual_hand_entry_clip_boundary"
            # A sustained broadcast cutaway limits usable edit footage. It is
            # not a measurement of where the balls physically stopped.
            cutaway_boundary = availability_reason in {
                "non_table_cutaway_clip_boundary", "visual_hand_entry_clip_boundary",
            }
            if physical_stop > clip_cap + 1e-9 and not cutaway_boundary:
                physical_stop = clip_cap
                confirmation = clip_cap
                last_motion = min(last_motion, clip_cap)
                end_confidence = min(end_confidence, 0.20)
                stop_confirmed = False
                stop_reason = availability_reason or "max_clip_duration_review_cap"
                stop_review = True
            min_after = max(
                0.0,
                float(mode_cfg.get("min_seconds_after_strike", 0.0)),
            )
            end_trim = (
                max(0.0, float(mode_cfg.get("end_before_ball_stop_seconds", 0.0)))
                if stop_confirmed else 0.0
            )
            if end_trim > 0:
                # Duration is cue contact to physical stop, excluding pre-roll
                # and confirmation look-ahead. Unresolved caps are not stops.
                shot_duration = max(0.0, physical_stop - cand.timestamp)
                long_threshold = float(mode_cfg.get("long_shot_threshold_seconds", 7.0))
                if shot_duration + 1e-9 >= long_threshold:
                    end_trim = max(0.0, float(mode_cfg.get(
                        "long_shot_end_before_ball_stop_seconds", end_trim,
                    )))
                # Early trimming yields to viewing time on short shots.
                min_after = minimum_visibility
            minimum_clip_end = clamp(
                max(cand.timestamp + min_after, clip_start + minimum_clip + 2 * transition),
                clip_start,
                min(duration, clip_cap),
            )
            object_motion_end = self._last_independent_object_motion(
                features, cand.timestamp, min(physical_stop, clip_cap), feature_times,
            )
            if object_motion_end > cand.timestamp:
                # Trimming settling time must not hide an object ball's final
                # approach to a pocket. Keep its supported travel and a short
                # outcome hold clear of the outgoing dissolve. Source handling
                # and replay limits remain authoritative.
                minimum_clip_end = max(minimum_clip_end, clamp(
                    object_motion_end + .40 + transition,
                    clip_start, min(duration, clip_cap),
                ))
            clip_end = clamp(
                max(physical_stop - end_trim, minimum_clip_end),
                clip_start,
                min(duration, clip_cap),
            )

            shot_conf = float(cand.confidence)
            level = self._level(shot_conf)
            review = (
                level != ConfidenceLevel.HIGH
                or stop_review
                or not stop_confirmed
                or float(cand.evidence.get("cue_geometry_confirmed", 1.0)) < 0.5
                or float(cand.evidence.get("contact_time_upper_bound", 0.0)) >= 0.5
                or clip_end-clip_start < minimum_clip
            )

            possible_replay = bool(cand.possible_replay)
            included = True
            if possible_replay and not retain_replays:
                # Only explicit replay camera classifications auto-exclude.  A
                # weak visual-signature guess remains reviewable and cannot
                # silently delete a live shot.
                if cand.camera_view in {
                    CameraViewType.REPLAY,
                    CameraViewType.SLOW_MOTION_REPLAY,
                } or float(cand.evidence.get("replay_signature_confirmed", 0.0)) >= 0.5:
                    included = False
                review = True

            evidence = dict(cand.evidence)
            evidence.update(
                {
                    # Used only to suppress mid-motion duplicate candidates;
                    # the exported strict boundary may be capped separately.
                    "uncapped_physical_stop_timestamp": uncapped_physical_stop,
                    "minimum_clip_end_timestamp": minimum_clip_end,
                    "last_independent_object_motion_timestamp": object_motion_end,
                    "minimum_clip_seconds": minimum_clip,
                    "minimum_strike_visibility_seconds": minimum_visibility,
                    "minimum_clip_transition_padding_seconds": 2 * transition,
                    "end_before_ball_stop_seconds": end_trim,
                    "shot_duration_for_end_trim_seconds": (
                        max(0.0, physical_stop - cand.timestamp) if stop_confirmed else None
                    ),
                    "last_ball_motion_timestamp": last_motion,
                    "physical_stop_timestamp": physical_stop,
                    "stop_confirmation_timestamp": confirmation,
                    "stop_confirmed": stop_confirmed,
                    "stop_reason": stop_reason,
                    "usable_source_end_timestamp": clip_cap,
                    "usable_source_end_reason": availability_reason,
                    "usable_source_start_timestamp": usable_start,
                    "usable_source_start_reason": preparation_boundary_reason,
                }
            )
            views = self._views_between(
                features, clip_start, clip_end, cand, times=feature_times
            )
            shots.append(
                ShotRecord(
                    shot_id=len(shots) + 1,
                    preparation_start=prep_start,
                    cue_strike=cand.timestamp,
                    cue_strike_timestamp=cand.timestamp,
                    ball_motion_start=stop.motion_start,
                    # Legacy field remains the physical all-ball stop.
                    ball_motion_end=physical_stop,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    clip_start_timestamp=clip_start,
                    clip_end_timestamp=clip_end,
                    shot_confidence=shot_conf,
                    start_confidence=stop.start_confidence,
                    end_confidence=end_confidence,
                    last_ball_motion_timestamp=last_motion,
                    physical_stop_timestamp=physical_stop,
                    stop_confirmation_timestamp=confirmation,
                    strike_confidence=shot_conf,
                    stop_confidence=end_confidence,
                    camera_views=views,
                    possible_replay=possible_replay,
                    manual_review_required=review,
                    evidence=evidence,
                    included=included,
                    confidence_level=level,
                )
            )

        shots = self._resolve_overlaps(shots, strict=True, source_duration=duration)
        logger.info("Built %d shot segments (strict)", len(shots))
        return shots

    def _unusable_boundaries(self, features: list[FrameFeatures]) -> list[tuple[float, str]]:
        """Bound padding by established handling or a confirmed foreign table.

        A physical stop can be confirmed before the referee reaches a ball.
        Minimum-viewing padding must not subsequently extend into that action.
        Handling flags need a sustained run; foreign flags already carry the
        broadcast guard's confirmation and retrospective start timestamp.
        """
        return sorted((start, reason) for start, _, reason in self._unusable_spans(features))

    def _unusable_spans(self, features: list[FrameFeatures]) -> list[tuple[float, float, str]]:
        """Known unusable action, ending at the next observed usable frame.

        Retain complete spans so preparation footage as well as minimum-viewing
        padding can be bounded. A brief handling flag still requires the same
        sustained confirmation used for end boundaries.
        """
        spans = []
        handling_start = None
        handling_confirmed = False
        previous = None
        foreign_start = None
        replay_start = None
        for index, f in enumerate(features):
            if f.broadcast_replay:
                if replay_start is None:
                    replay_start = self._replay_fade_start(features, index)
            elif replay_start is not None:
                spans.append((replay_start, f.t, "replay_clip_boundary"))
                replay_start = None
            if not f.match_context_valid:
                if foreign_start is None:
                    foreign_start = f.t
            elif foreign_start is not None:
                spans.append((foreign_start, f.t, "foreign_match_clip_boundary"))
                foreign_start = None
            usable_handling = (f.table_handling and f.match_context_valid
                               and not f.broadcast_replay and f.view_type not in {
                                   CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY})
            continuous = (previous is not None and 0 < f.t-previous.t <= .76
                          and f.camera_scene_id == previous.camera_scene_id)
            if handling_start is not None and (not usable_handling or not continuous):
                if handling_confirmed:
                    spans.append((handling_start, f.t, "ball_handling_clip_boundary"))
                handling_start = None
                handling_confirmed = False
            if usable_handling:
                if handling_start is None:
                    handling_start = f.t
                if f.t-handling_start >= self.ball_stop.handling_confirmation_s-1e-9:
                    handling_confirmed = True
            previous = f
        if features:
            if replay_start is not None:
                spans.append((replay_start, features[-1].t, "replay_clip_boundary"))
            if foreign_start is not None:
                spans.append((foreign_start, features[-1].t, "foreign_match_clip_boundary"))
            if handling_start is not None and handling_confirmed:
                spans.append((handling_start, features[-1].t, "ball_handling_clip_boundary"))
        spans.extend(self._non_table_cutaway_spans(features))
        return sorted(spans)

    @staticmethod
    def _last_independent_object_motion(features, strike_t, end_t, times) -> float:
        """Require sustained measured object travel beyond the white's speed."""
        latest = 0.
        run_start = None
        previous = None
        for f in features[bisect_right(times, strike_t+.15):bisect_right(times, end_t)]:
            qualified = bool(
                f.observation_fps >= 10 and f.observation_valid and f.table_observable
                and f.ball_kinematics_valid and f.match_context_valid
                and not f.broadcast_replay and not f.table_handling
                and f.view_type not in {CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY}
                and f.scene_cut_score < .5 and f.cue_ball_detected and f.cue_ball_quality
                and f.cue_ball_track_confidence >= .65
                and f.cue_ball_stable_normalized_speed is not None
                and f.moving_ball_count >= 1
                and f.max_ball_normalized_speed >= max(
                    1.5, f.cue_ball_stable_normalized_speed + .75,
                ))
            continuous = bool(qualified and previous is not None and previous.camera_scene_id == f.camera_scene_id
                              and 0 < f.t-previous.t <= 2/min(f.observation_fps,previous.observation_fps)+.005)
            if not qualified:
                run_start = None
            elif run_start is None or not continuous:
                run_start = f.t
            elif f.t-run_start >= .16-1e-9:
                latest = f.t
            previous = f if qualified else None
        return latest

    @staticmethod
    def _replay_fade_start(features, index) -> float:
        """Bound the fade into an already paired detector-owned replay wipe."""
        opening = features[index]
        if (not opening.replay_stinger_annotated or opening.replay_stinger_original_marker
                or opening.observation_fps < 10):
            return opening.t
        start = opening.t
        previous = opening
        count = 0
        for f in reversed(features[:index]):
            if (opening.t-f.t > .60 or f.observation_fps < 10
                    or not f.match_context_valid or f.broadcast_replay
                    or not 0 < previous.t-f.t <= 2/min(f.observation_fps,previous.observation_fps)+.005
                    or f.scene_cut_score < .08):
                break
            start = f.t
            count += 1
            previous = f
        return start if count >= 3 and opening.t-start >= .16-1e-9 else opening.t

    def _non_table_cutaway_spans(self, features: list[FrameFeatures]) -> list[tuple[float, float, str]]:
        """Sustained measured non-table footage can bound an automatic edit.

        A partial table still supplies ball observations. Only a classified
        non-table view contributes, and holes longer than the recorded sampling
        cadence reset confirmation rather than becoming elapsed evidence.
        """
        confirmation = max(0.0, float(self.config.get(
            "ball_stop.non_table_cutaway_boundary_confirmation_seconds", 3.0)))
        spans = []
        start = None
        confirmed = False
        previous = None
        for f in features:
            non_table = bool(f.view_classified and not f.table_observable
                             and f.match_context_valid and not f.broadcast_replay
                             and f.view_type not in {
                                 CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY,
                             })
            cadence = min(f.observation_fps, previous.observation_fps) if previous is not None else 0
            # A 25fps analysis sampled from a 30fps proxy alternates one- and
            # two-frame gaps. Both belong to the measured cadence; treating the
            # 66.7ms step as a hole would prevent any sustained confirmation.
            max_gap = min(.76, 2.0 / cadence) if cadence > 0 else self.ball_stop.max_observation_gap_s
            continuous = previous is not None and 0 < f.t-previous.t <= max_gap + 1e-9
            if start is not None and (not non_table or not continuous):
                if confirmed:
                    spans.append((start, f.t, "non_table_cutaway_clip_boundary"))
                start = None
                confirmed = False
            if non_table:
                if start is None:
                    start = f.t
                if f.t-start >= confirmation-1e-9:
                    confirmed = True
            previous = f
        if features and start is not None and confirmed:
            spans.append((start, features[-1].t, "non_table_cutaway_clip_boundary"))
        return spans

    @staticmethod
    def _candidate_supported(candidate: StrikeCandidate) -> bool:
        evidence = candidate.evidence
        return bool(evidence.get("dense_transition_confirmed", 0) >= 0.5
                    or evidence.get("cue_ball_motion_confirmed", 0) >= 0.5
                    or (evidence.get("occlusion_inferred", 0) >= 0.5
                        and evidence.get("ball_onset_run", 0) >= 2))

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[StrikeCandidate],
        distance_s: float = 0.60,
    ) -> list[StrikeCandidate]:
        result: list[StrikeCandidate] = []
        for cand in sorted(candidates, key=lambda c: c.timestamp):
            if not result or cand.timestamp - result[-1].timestamp >= distance_s:
                result.append(cand)
                continue
            if cand.confidence > result[-1].confidence:
                result[-1] = cand
        return result

    @staticmethod
    def _views_between(
        features: list[FrameFeatures],
        start: float,
        end: float,
        cand: StrikeCandidate,
        times: list[float] | None = None,
    ) -> list[str]:
        values = {cand.camera_view.value}
        if times is None:
            times = [feature.t for feature in features]
        lo = bisect_left(times, start)
        hi = bisect_right(times, end)
        values.update(f.view_type.value for f in features[lo:hi])
        return sorted(values)

    def _independently_supported(self, shot: ShotRecord) -> bool:
        """Whether a record carries decisive, self-contained strike evidence.

        Two such records that merely collide in source time are treated as
        genuine fast-succession shots and reconciled by trimming.  A record
        that fails this bar while colliding with another shot's window is a
        preparation/collision artefact and enters conflict resolution instead.
        """
        evidence = shot.evidence
        confirmed_return = bool(
            shot.shot_confidence >= self.medium
            and evidence.get("reacquired_ball_roll", 0) >= .5
            and evidence.get("native_occlusion_confirmed", 0) >= .5
            and evidence.get("dense_transition_confirmed", 0) >= .5
            and evidence.get("cue_ball_motion_confirmed", 0) >= .5
            and evidence.get("ball_onset_run", 0) >= 8
            and evidence.get("cue_direction_consistency", 0) >= .90
            and evidence.get("cue_displacement_diameters", 0) >= .65)
        # A hidden contact has uncertain timing even when its returned roll is
        # independently measured. That timing confidence must not erase the
        # genuine next shot when the previous stop is still unresolved.
        if confirmed_return:
            return True
        if shot.shot_confidence < self.high:
            return False
        quiet = shot.evidence.get("pre_ball_quiet_ratio")
        if quiet is None:
            return True
        return float(quiet) >= self.support_quiet_ratio

    def _resolve_overlaps(
        self,
        shots: list[ShotRecord],
        *,
        strict: bool = True,
        source_duration: float = float("inf"),
        **_legacy_kwargs,
    ) -> list[ShotRecord]:
        """Resolve mutually impossible strikes without cutting shot footage.

        A genuine next cue strike cannot occur before every ball from the prior
        shot has stopped, and two genuine strikes cannot be closer than the
        configured minimum spacing (balls must stop and the player must
        re-address).  Such candidates are collisions, cushion impacts,
        feathering artefacts, or uncertain prior boundaries; when two
        confirmed-looking events conflict, the better-supported one is kept.

        Two independently supported strikes whose strict windows merely overlap
        (fast break play) are BOTH kept: the boundary between them is trimmed
        by ``_trim_adjacent_windows`` instead of deleting a real shot.
        """
        if not shots:
            return []
        ordered = sorted(shots, key=lambda s: s.cue_strike)
        resolved: list[ShotRecord] = []

        for shot in ordered:
            keep_current = True
            while resolved:
                prev = resolved[-1]
                prev_stop = float(
                    prev.evidence.get("uncapped_physical_stop_timestamp")
                    or getattr(prev, "physical_stop_timestamp", 0.0)
                    or prev.ball_motion_end
                )
                near_duplicate = abs(shot.cue_strike - prev.cue_strike) < 0.60
                # A max-duration cap means tracking never established the true
                # stop; it is not proof that balls were still moving at the
                # next independently verified strike.  Treat only a confirmed
                # stop boundary (or legacy records without this field) as
                # authoritative for mid-motion suppression.
                stop_confirmed = prev.evidence.get("stop_confirmed")
                reliable_stop = stop_confirmed is not False
                usable_end = float(prev.evidence.get("usable_source_end_timestamp", float("inf")))
                if prev_stop > usable_end+1e-6 and prev.evidence.get("usable_source_end_reason") in {
                    "non_table_cutaway_clip_boundary", "visual_hand_entry_clip_boundary", "replay_clip_boundary",
                }:
                    # A stop reacquired after unusable footage is an upper
                    # bound. It cannot erase the next independently proven
                    # contact as if its balls were observed moving throughout.
                    reliable_stop = False
                mid_motion = (
                    reliable_stop and shot.cue_strike <= prev_stop + 1e-6
                )
                too_close = (
                    shot.cue_strike - prev.cue_strike < self.min_shot_spacing
                )
                # Overlapping strict windows alone are no longer fatal: when
                # both records carry decisive strike evidence they are two real
                # shots in fast succession and the shared boundary is trimmed
                # later.  A colliding record without that support is an
                # artefact and must lose to its neighbour here.
                conflict_end = prev.clip_end
                if float(prev.evidence.get("minimum_clip_seconds", 0)) > 0:
                    # Optional mix handles must not create a new candidate
                    # conflict and discard a genuine fast-succession shot.
                    conflict_end = min(conflict_end, max(
                        prev.physical_stop_timestamp - float(prev.evidence.get("end_before_ball_stop_seconds", 0)),
                        prev.clip_start + float(prev.evidence["minimum_clip_seconds"]),
                        prev.cue_strike + float(prev.evidence.get("minimum_strike_visibility_seconds", 0)),
                    ))
                source_overlap = strict and shot.clip_start < conflict_end - 1e-6
                contested_overlap = source_overlap and not (
                    self._independently_supported(prev)
                    and self._independently_supported(shot)
                )

                if not (near_duplicate or mid_motion or too_close or contested_overlap):
                    break

                if self._prefer_later_conflicting_shot(prev, shot):
                    shot.evidence["replaced_conflicting_strike"] = prev.cue_strike
                    resolved.pop()
                    # Re-check the replacement against the shot before it. This
                    # matters for bursts such as valid -> false peak -> valid.
                    continue

                prev.manual_review_required = prev.manual_review_required or (
                    shot.shot_confidence >= self.high
                )
                evidence_key = (
                    "rejected_mid_motion_strike"
                    if mid_motion
                    else "rejected_overlapping_strike"
                )
                prev.evidence[evidence_key] = shot.cue_strike
                keep_current = False
                break

            if keep_current:
                resolved.append(shot)

        for i, shot in enumerate(resolved, start=1):
            shot.shot_id = i
            if strict:
                expected_start = self._strict_start(shot.cue_strike, float("inf"))
                expected_start = max(expected_start, float(shot.evidence.get("usable_source_start_timestamp", 0)))
                shot.clip_start = expected_start
                shot.preparation_start = expected_start
                physical = float(
                    getattr(shot, "physical_stop_timestamp", 0.0)
                    or shot.ball_motion_end
                )
                minimum_end = float(
                    shot.evidence.get("minimum_clip_end_timestamp") or physical
                )
                end_trim = float(shot.evidence.get("end_before_ball_stop_seconds", 0.0))
                shot.clip_end = min(
                    float(shot.evidence.get("usable_source_end_timestamp", source_duration)),
                    max(physical - end_trim, minimum_end),
                )
                shot.ball_motion_end = physical
                shot.cue_strike_timestamp = shot.cue_strike
                shot.clip_start_timestamp = shot.clip_start
                shot.clip_end_timestamp = shot.clip_end
        if strict:
            self._trim_adjacent_windows(resolved, source_duration=source_duration)
        return resolved

    def _trim_adjacent_windows(self, shots: list[ShotRecord], *, source_duration: float = float("inf")) -> None:
        """Share the boundary between fast consecutive shots without overlap.

        The previous shot gives up its minimum-hold padding (and any
        unresolved review tail) down to the next shot's pre-roll start, but a
        confirmed stop-minus-offset boundary is retained. Any residual overlap is then
        removed by shortening the next shot's pre-roll: in fast play the
        source simply does not contain two seconds of dead time before the
        next strike, and duplicating footage in a joined export would read as
        a glitch.
        """
        for prev, nxt in zip(shots, shots[1:]):
            if nxt.clip_start >= prev.clip_end - 1e-9:
                continue
            reliable_stop = prev.evidence.get("stop_confirmed") is not False
            physical = float(
                getattr(prev, "physical_stop_timestamp", 0.0) or prev.ball_motion_end
            )
            end_trim = float(prev.evidence.get("end_before_ball_stop_seconds", 0.0))
            floor = physical - end_trim if reliable_stop else nxt.clip_start
            if float(prev.evidence.get("minimum_clip_seconds", 0)) > 0:
                # Give up transition padding before clear viewing time.
                floor = max(
                    floor,
                    prev.clip_start + float(prev.evidence["minimum_clip_seconds"]),
                    prev.cue_strike + float(prev.evidence.get("minimum_strike_visibility_seconds", 0)),
                )
            new_end = min(prev.clip_end, max(nxt.clip_start, floor))
            new_end = max(new_end, prev.cue_strike)
            if new_end < prev.clip_end - 1e-9:
                prev.clip_end = new_end
                prev.clip_end_timestamp = new_end
                if not reliable_stop and prev.physical_stop_timestamp > new_end + 1e-9:
                    # Only an unconfirmed review cap can be trimmed here; the
                    # boundary stays reviewable, never silently authoritative.
                    prev.physical_stop_timestamp = new_end
                    prev.ball_motion_end = new_end
                    prev.stop_confirmation_timestamp = new_end
                    prev.last_ball_motion_timestamp = min(
                        prev.last_ball_motion_timestamp, new_end
                    )
                    prev.evidence["stop_reason"] = "trimmed_at_next_shot_start"
                    prev.manual_review_required = True
                prev.evidence["trimmed_for_next_shot"] = nxt.cue_strike
            if nxt.clip_start < prev.clip_end - 1e-9:
                new_start = min(prev.clip_end, nxt.cue_strike)
                nxt.evidence["pre_roll_trimmed_seconds"] = round(
                    new_start - nxt.clip_start, 6
                )
                nxt.clip_start = new_start
                nxt.clip_start_timestamp = new_start
                nxt.preparation_start = new_start
                minimum_clip = float(nxt.evidence.get("minimum_clip_seconds", 0))
                if minimum_clip > 0:
                    # A shortened pre-roll must not shorten the next shot's
                    # minimum viewing time. The following pair reconciles any
                    # new overlap; EOF remains an absolute limit.
                    minimum_end = min(source_duration, float(nxt.evidence.get("usable_source_end_timestamp", source_duration)), max(
                        new_start + minimum_clip + float(nxt.evidence.get("minimum_clip_transition_padding_seconds", 0)),
                        nxt.cue_strike + float(nxt.evidence.get("minimum_strike_visibility_seconds", 0)),
                    ))
                    nxt.evidence["minimum_clip_end_timestamp"] = minimum_end
                    nxt.clip_end = max(nxt.clip_end, minimum_end)
                    nxt.clip_end_timestamp = nxt.clip_end

    def _prefer_later_conflicting_shot(
        self,
        previous: ShotRecord,
        current: ShotRecord,
    ) -> bool:
        """Return whether a later incompatible candidate has stronger support.

        Compare launch trajectories, confidence, and pre-strike stillness.
        Commentary and collision sounds cannot change which event survives.
        """
        if previous.included != current.included:
            return current.included

        def launch_evidence(shot: ShotRecord) -> float:
            ev = shot.evidence
            speed = float(ev.get("post_peak_cue_speed", 0.0) or 0.0)
            disp = float(ev.get("cue_displacement_diameters", 0.0) or 0.0)
            onset_run = float(ev.get("ball_onset_run", 0.0) or 0.0)
            sustained = float(ev.get("sustained_run", 0.0) or 0.0)
            contact = float(ev.get("cue_contact_score", 0.0) or 0.0)
            return (
                1.0 * min(1.0, speed / 3.0)
                + 0.8 * min(1.0, disp / 1.0)
                + 0.5 * min(1.0, (onset_run + sustained) / 4.0)
                + 0.4 * min(1.0, contact / 0.50)
            )

        prev_launch = launch_evidence(previous)
        curr_launch = launch_evidence(current)
        if curr_launch - prev_launch >= 0.25:
            return True
        if prev_launch - curr_launch >= 0.25:
            return False

        confidence_delta = current.shot_confidence - previous.shot_confidence
        if abs(confidence_delta) > self.conflict_confidence_margin:
            return confidence_delta > 0.0

        previous_quiet = float(
            previous.evidence.get("pre_ball_quiet_ratio", 0.0) or 0.0
        )
        current_quiet = float(
            current.evidence.get("pre_ball_quiet_ratio", 0.0) or 0.0
        )
        if abs(current_quiet - previous_quiet) > 1e-6:
            return current_quiet > previous_quiet

        # Stable tie-break: preserve the earlier event. This is safer for an
        # actual shot followed by a cushion/collision peak with equal evidence.
        return False

    def recompute_durations(
        self,
        shots: list[ShotRecord],
        original: float,
    ) -> tuple[float, float]:
        edited = sum(s.duration() for s in shots if s.included)
        removed = max(0.0, original - edited)
        return edited, removed
