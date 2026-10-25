"""Apply explicit source-reviewed corrections to a copy of a saved timeline.

This is a human review import, not a detector or a score-to-contact inference.
New contacts retain the reviewer's uncertainty interval and editing boundaries.
"""

import math
from collections import Counter

from snooker_ai.types import AnalysisResult, PotStatus, ShotRecord, ShotSelection, TimelineEvent


def apply_source_review(result: AnalysisResult, manifest: dict) -> AnalysisResult:
    if manifest.get("source_reviewed") is not True:
        raise ValueError("Corrections require an explicitly source-reviewed manifest")
    duration = float(manifest["source_duration_seconds"])
    if not math.isfinite(duration) or abs(duration - result.original_duration) > 0.1:
        raise ValueError("Reviewed source duration does not match the saved analysis")
    reviewed = result.model_copy(deep=True)
    original_ids = {s.shot_id: s for s in reviewed.shots}
    added_ids = {}
    touched = set()
    for change in manifest.get("updates", []):
        identifier = change["shot_id"]
        if identifier in touched or identifier not in original_ids:
            raise ValueError("Duplicate or missing reviewed shot ID")
        touched.add(identifier)
        shot = original_ids[identifier]
        strike = float(change["strike"])
        if not math.isfinite(strike) or abs(shot.cue_strike - strike) > 0.05:
            raise ValueError("Reviewed contact no longer matches the saved shot")
        if not change.get("reason"):
            raise ValueError("Reviewed corrections require a source observation")
        possible_replay = change.get("possible_replay", shot.possible_replay)
        if not isinstance(possible_replay, bool):
            raise TypeError("Reviewed replay flag must be a boolean")
        if possible_replay and change.get("included", True):
            raise ValueError("A replay cannot be included as a live reviewed contact")
        shot.possible_replay = possible_replay
        shot.pot_status = PotStatus(change["status"])
        shot.included = bool(change.get("included", True))
        shot.evidence["source_review"] = dict(change)
        shot.evidence["pots_only_manual_override"] = True
        shot.evidence["pot_confirmation"] = {"reason": "source_review", **change}

    for change in manifest.get("additions", []):
        lower, upper = map(float, change["contact_interval"])
        start, end = map(float, change["clip"])
        if (
            not all(math.isfinite(v) for v in (lower, upper, start, end))
            or not 0 <= start <= lower <= upper < end <= reviewed.original_duration
            or upper - lower > 3
            or end - start < 4
            or not change.get("reason")
        ):
            raise ValueError("Invalid reviewed contact interval or clip")
        strike = (lower + upper) / 2
        if any(abs(s.cue_strike - strike) < 1.5 for s in reviewed.shots):
            raise ValueError("Reviewed addition duplicates an existing contact")
        status = PotStatus(change["status"])
        context = change.get("frame_bookend")
        if status not in {PotStatus.POT, PotStatus.FOUL, PotStatus.POINTS} and context not in {
            "first",
            "last",
        }:
            raise ValueError("Reviewed additions need a scoring outcome or explicit frame bookend")
        review_id = change.get("review_id")
        if review_id is not None and (
            not isinstance(review_id, str) or not review_id or review_id in added_ids
        ):
            raise ValueError("Reviewed addition IDs must be unique nonempty strings")
        shot = ShotRecord(
            shot_id=max((s.shot_id for s in reviewed.shots), default=0) + 1,
            cue_strike=strike,
            preparation_start=start,
            clip_start=start,
            clip_end=end,
            ball_motion_start=lower,
            ball_motion_end=end,
            shot_confidence=1,
            strike_confidence=0.8,
            start_confidence=1,
            end_confidence=0.5,
            stop_confidence=0.5,
            manual_review_required=True,
            user_modified=True,
            pot_status=status,
            included=True,
            evidence={
                "source_review": dict(change),
                "pots_only_manual_override": True,
                "pot_confirmation": {"reason": "source_review", **change},
                "stop_confirmed": False,
                "stop_reason": "reviewed_edit_boundary",
                "contact_time_lower_bound": lower,
                "contact_time_upper_bound": upper,
            },
        )
        reviewed.shots.append(shot)
        if review_id is not None:
            added_ids[review_id] = shot

    frames = manifest.get("frames", [])
    resolved_frames = []

    def bookend(frame, role):
        existing_key, added_key = f"{role}_shot_id", f"{role}_review_id"
        if (existing_key in frame) == (added_key in frame):
            raise ValueError("Each reviewed frame needs exactly one reference per bookend")
        try:
            if existing_key in frame:
                return original_ids[frame[existing_key]]
            return added_ids[frame[added_key]]
        except KeyError as exc:
            raise ValueError("Missing reviewed frame bookend") from exc

    for shot in reviewed.shots:
        shot.evidence.pop("frame_context", None)
    for index, frame in enumerate(frames, 1):
        first, last = bookend(frame, "first"), bookend(frame, "last")
        if first.cue_strike > last.cue_strike or first.possible_replay or last.possible_replay:
            raise ValueError("Invalid reviewed frame bookends")
        if resolved_frames and first.cue_strike <= resolved_frames[-1][1].cue_strike:
            raise ValueError("Reviewed frames must be ordered and nonoverlapping")
        resolved_frames.append((first, last))
        for shot in reviewed.shots:
            if first.cue_strike <= shot.cue_strike <= last.cue_strike:
                shot.evidence["frame_context"] = {
                    "frame": index,
                    "first": shot is first,
                    "last": shot is last,
                    "source_reviewed": True,
                }
        first.included = last.included = True
        first.evidence["pots_only_manual_override"] = True
        last.evidence["pots_only_manual_override"] = True

    for shot in reviewed.shots:
        change = shot.evidence.get("source_review", {})
        if (
            "contact_interval" in change
            and change.get("frame_bookend")
            and not shot.evidence.get("frame_context", {}).get(change["frame_bookend"])
        ):
            raise ValueError("A non-scoring addition must be a referenced frame bookend")

    reviewed.shots.sort(key=lambda s: s.cue_strike)
    prior_strikes = {identifier: shot.cue_strike for identifier, shot in original_ids.items()}
    strike_to_id = {}
    for index, shot in enumerate(reviewed.shots, 1):
        shot.shot_id = index
        strike_to_id[shot.cue_strike] = index
    for shot in reviewed.shots:
        if shot.linked_live_shot_id in prior_strikes:
            shot.linked_live_shot_id = strike_to_id[prior_strikes[shot.linked_live_shot_id]]
    for event in reviewed.events:
        identifier = event.metadata.get("shot_id")
        if identifier in prior_strikes:
            event.metadata["shot_id"] = strike_to_id[prior_strikes[identifier]]
    added = [s for s in reviewed.shots if "contact_interval" in s.evidence.get("source_review", {})]
    for shot in added:
        reviewed.events.append(
            TimelineEvent(
                event_type="cue_strike",
                timestamp=shot.cue_strike,
                confidence=shot.strike_confidence,
                metadata={
                    "shot_id": shot.shot_id,
                    "source_reviewed": True,
                    "contact_interval": shot.evidence["source_review"]["contact_interval"],
                },
            )
        )
    reviewed.events.sort(key=lambda event: event.timestamp)
    observed_counts = Counter(
        s.pot_status.value for s in reviewed.shots if s.pot_status is not None
    )
    counts = {status.value: observed_counts[status.value] for status in PotStatus}
    reviewed.selection = ShotSelection.POTS_AND_FOULS
    reviewed.pot_selection_summary = {
        **reviewed.pot_selection_summary,
        **counts,
        "included": sum(s.included and not s.possible_replay for s in reviewed.shots),
        "method": "stable_scores_and_source_review",
        "selection": "pots_and_fouls",
        "source_review": {
            "name": manifest["name"],
            "added_contacts": len(added),
            "updated_records": len(touched),
            "frames_reviewed": len(frames),
        },
        "frames": [
            {
                "frame": i,
                "first_strike": first.cue_strike,
                "last_strike": last.cue_strike,
                "first_shot_id": first.shot_id,
                "last_shot_id": last.shot_id,
            }
            for i, (first, last) in enumerate(resolved_frames, 1)
        ],
        "frame_context_shots": sum(
            s.included
            and bool(
                s.evidence.get("frame_context", {}).get("first")
                or s.evidence.get("frame_context", {}).get("last")
            )
            for s in reviewed.shots
        ),
        "non_scoring_context_shots": sum(
            s.included
            and s.pot_status not in {PotStatus.POT, PotStatus.FOUL, PotStatus.POINTS}
            and bool(
                s.evidence.get("frame_context", {}).get("first")
                or s.evidence.get("frame_context", {}).get("last")
            )
            for s in reviewed.shots
        ),
    }
    reviewed.edited_duration = sum(
        s.duration() for s in reviewed.shots if s.included and not s.possible_replay
    )
    reviewed.pause_removed_seconds = max(0, reviewed.original_duration - reviewed.edited_duration)
    return reviewed
