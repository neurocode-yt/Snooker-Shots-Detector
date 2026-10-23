"""Apply explicit source-reviewed corrections to a copy of a saved timeline.

This is a human review import, not a detector or a score-to-contact inference.
New contacts retain the reviewer's uncertainty interval and editing boundaries.
"""

from collections import Counter
import math

from snooker_ai.types import AnalysisResult, PotStatus, ShotRecord, ShotSelection, TimelineEvent


def apply_source_review(result: AnalysisResult, manifest: dict) -> AnalysisResult:
    if manifest.get("source_reviewed") is not True:
        raise ValueError("Corrections require an explicitly source-reviewed manifest")
    duration = float(manifest["source_duration_seconds"])
    if not math.isfinite(duration) or abs(duration - result.original_duration) > 0.1:
        raise ValueError("Reviewed source duration does not match the saved analysis")
    reviewed = result.model_copy(deep=True)
    original_ids = {s.shot_id: s for s in reviewed.shots}
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
        if shot.possible_replay and change.get("included", True):
            raise ValueError("A replay cannot be included as a live reviewed contact")
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
        if status not in {PotStatus.POT, PotStatus.FOUL, PotStatus.POINTS}:
            raise ValueError("Reviewed additions must have a confirmed scoring outcome")
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

    frames = manifest.get("frames", [])
    for shot in reviewed.shots:
        shot.evidence.pop("frame_context", None)
    for index, frame in enumerate(frames, 1):
        first, last = (original_ids[frame[key]] for key in ("first_shot_id", "last_shot_id"))
        if first.cue_strike > last.cue_strike or first.possible_replay or last.possible_replay:
            raise ValueError("Invalid reviewed frame bookends")
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
                "first_strike": original_ids[f["first_shot_id"]].cue_strike,
                "last_strike": original_ids[f["last_shot_id"]].cue_strike,
                "first_shot_id": original_ids[f["first_shot_id"]].shot_id,
                "last_shot_id": original_ids[f["last_shot_id"]].shot_id,
            }
            for i, f in enumerate(frames, 1)
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
