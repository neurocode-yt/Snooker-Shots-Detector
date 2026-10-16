"""Score changes can request a visual check, never manufacture a cue strike."""
from bisect import bisect_right

from snooker_ai.pot_detection.scoreboard import ScoreReading
from snooker_ai.pot_detection.selector import stable_scores
from snooker_ai.types import ShotRecord, StrikeCandidate
from snooker_ai.types import CameraViewType, FrameFeatures


def scoring_gap_proposals(readings: list[ScoreReading], shots: list[ShotRecord],
                          duration: float, *, lookback: float = 12.) -> list[StrikeCandidate]:
    runs = stable_scores(readings)
    live = sorted((shot for shot in shots if not shot.possible_replay),
                  key=lambda shot: shot.cue_strike_timestamp)
    times = [shot.cue_strike_timestamp for shot in live]
    assigned = set()
    proposals = []
    for before, after in zip(runs, runs[1:]):
        delta = (after.left-before.left, after.right-before.right)
        if (min(delta) < 0 or (delta[0] == 0) == (delta[1] == 0)
                or not 1 <= max(delta) <= 7 or after.start-before.end > lookback):
            continue
        index = bisect_right(times, after.start-.2)-1
        if index >= 0 and after.start-times[index] <= lookback and index not in assigned:
            assigned.add(index)
            continue
        lower = max(0., after.start-lookback)
        if index >= 0:
            lower = max(lower, times[index]+4.)
        upper = min(duration, after.start-.2)
        if upper-lower < .5:
            continue
        proposals.append(StrikeCandidate(timestamp=max(lower, after.start-3.), confidence=.5,
            uncertainty_start=lower, uncertainty_end=upper,
            evidence={'sparse_proposal':1., 'score_assisted_proposal':1.,
                      'score_update_time':after.start, 'score_delta':float(max(delta))}))
    return proposals


def scoring_object_contact(proposal: StrikeCandidate, features: list[FrameFeatures]):
    """Bound a hidden contact by independently tracked coloured-ball launch.

    The native tracker already requires pre-launch rest, contiguous positions,
    coherent travel and a large measured sphere. Score corroboration restricts
    this recovery to actual points changes; it remains a reviewable upper bound.
    """
    if proposal.evidence.get('score_assisted_proposal',0)<.5:
        return None
    run=[]
    for frame in features:
        valid=(proposal.uncertainty_start <= frame.t <= proposal.uncertainty_end
            and frame.observation_fps>=10 and frame.observation_valid
            and frame.match_context_valid and not frame.broadcast_replay and not frame.table_handling
            and not frame.rack_idle and frame.view_type==CameraViewType.BALL_CLOSEUP
            and frame.scene_cut_score<.42 and frame.camera_motion_magnitude<=1.5
            and frame.scoring_object_launch_count>=1)
        if not valid:
            run=[]
            continue
        if run and (frame.camera_scene_id!=run[-1].camera_scene_id
                    or frame.t-run[-1].t>2/min(frame.observation_fps,run[-1].observation_fps)):
            run=[]
        run.append(frame)
        if len(run)>=3 and run[-1].t-run[0].t>=.06:
            upper=run[0].t
            return StrikeCandidate(timestamp=max(proposal.uncertainty_start,upper-.30),confidence=.75,
                uncertainty_start=max(proposal.uncertainty_start,upper-.70),uncertainty_end=upper,
                camera_view=frame.view_type,evidence={**proposal.evidence,
                    'score_supported_object_launch':1.,'contact_time_upper_bound':1.,
                    'object_launch_confirmation_time':run[-1].t})
    return None
