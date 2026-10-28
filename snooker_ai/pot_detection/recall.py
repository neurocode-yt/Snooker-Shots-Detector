"""Score changes can request a visual check, never manufacture a cue strike."""
from bisect import bisect_right

import numpy as np

from snooker_ai.pot_detection.scoreboard import ScoreReading
from snooker_ai.pot_detection.selector import stable_scores
from snooker_ai.types import ShotRecord, StrikeCandidate
from snooker_ai.types import CameraViewType, FrameFeatures


def scoring_gap_proposals(readings: list[ScoreReading], shots: list[ShotRecord],
                          duration: float, *, lookback: float = 12.,
                          score_delay: float = 25.,
                          matched_updates: dict | None = None) -> list[StrikeCandidate]:
    runs = stable_scores(readings)
    live = sorted((shot for shot in shots if not shot.possible_replay),
                  key=lambda shot: shot.cue_strike_timestamp)
    times = [shot.cue_strike_timestamp for shot in live]
    assigned = set()
    proposals = []
    def assign(index, update, points):
        assigned.add(index)
        if matched_updates is not None:
            matched_updates[times[index]] = {'score_update_time': update,
                'score_delta': float(points), 'score_matched_existing_contact': 1.}

    baseline_starts = []
    for i, run in enumerate(runs):
        same = (i > 0 and (run.left, run.right) == (runs[i-1].left, runs[i-1].right)
                and run.start-runs[i-1].end <= score_delay)
        baseline_starts.append(baseline_starts[-1] if same else run.start)
    for run_index, (before, after) in enumerate(zip(runs, runs[1:])):
        delta = (after.left-before.left, after.right-before.right)
        if (min(delta) < 0 or (delta[0] == 0) == (delta[1] == 0)
                or not 1 <= max(delta) <= 7 or after.start-before.end > max(lookback, score_delay)):
            continue
        index = bisect_right(times, after.start-.2)-1
        # The scoreboard may be absent during a close-up or wait for a foul
        # decision. Associate that delayed update once with the latest observed
        # stroke, without widening the visual search or reusing an earlier pot.
        # A baseline after the stroke must still be the immediately preceding
        # score: intervening updates/corrections cannot be bridged.
        if (index >= 0 and index not in assigned
                and after.start-times[index] <= max(lookback, score_delay)
                and baseline_starts[run_index] <= times[index] <= before.end):
            assign(index, after.start, max(delta))
            continue
        if after.start-before.end > lookback:
            # A long overlay absence can corroborate an existing contact. It
            # cannot create an unobserved stroke or widen the visual search.
            continue
        if index >= 0 and after.start-times[index] <= lookback and index not in assigned:
            assign(index, after.start, max(delta))
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


def scoring_collision_contact(proposal: StrikeCandidate, features: list[FrameFeatures]):
    """Verify a short white departure followed immediately by an object collision.

    Ordinary recovery requires a longer straight roll. At close range the
    white has already changed direction by then. Here both legs must be measured
    at native cadence, with ball shape, a quiet address/occlusion and cue geometry.
    The score only requests this check; it supplies no motion or contact time.
    """
    if (proposal.evidence.get('score_assisted_proposal', 0) < .5
            or not 1 <= proposal.evidence.get('score_delta', 0) <= 7):
        return None
    rows = sorted((f for f in features if proposal.uncertainty_start <= f.t
                   <= proposal.uncertainty_end), key=lambda f: f.t)

    def valid(f):
        return (f.observation_fps >= 20 and f.observation_valid and f.match_context_valid
                and not f.broadcast_replay and not f.table_handling and not f.rack_idle
                and f.view_type == CameraViewType.MAIN_TABLE and f.scene_cut_score < .42
                and f.camera_motion_magnitude <= 1.5)

    def measured(f):
        return (f.cue_ball_detected and f.cue_ball_quality
                and f.cue_ball_track_confidence >= .75
                and f.cue_ball_x is not None and f.cue_ball_y is not None
                and np.isfinite([f.cue_ball_x, f.cue_ball_y, f.ball_diameter_px]).all())

    for i, onset in enumerate(rows):
        if i < 4 or not measured(onset) or measured(rows[i-1]):
            continue
        post = rows[i:i+8]
        pre = [f for f in rows[:i] if onset.t-.72 <= f.t]
        # Use only the new camera's quiet address after a cut. Never subtract
        # positions across cameras, including canonicalized merged scene IDs.
        while pre and (pre[0].view_type != onset.view_type
                       or pre[0].scene_cut_score >= .42 or not valid(pre[0])):
            pre.pop(0)
        if (len(pre) < 4 or len(post) < 8
                or any(not valid(f) or f.camera_scene_id != onset.camera_scene_id for f in pre+post)
                or any(not measured(f) for f in post)
                or any(not 0 < b.t-a.t <= 1.6/min(a.observation_fps, b.observation_fps)
                       for a, b in zip(pre+post, (pre+post)[1:]))):
            continue
        diameter = float(np.median([f.ball_diameter_px for f in post]))
        if (not np.isfinite(diameter) or diameter < 6 or max(f.ball_diameter_px for f in post)
                > 1.3*min(f.ball_diameter_px for f in post)):
            continue
        xy = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in post])
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        first, second = xy[3]-xy[0], xy[-1]-xy[3]
        travel = float(np.linalg.norm(first))/diameter
        exit_travel = float(np.linalg.norm(second))/diameter
        direction = float(np.linalg.norm(first))/max(float(np.sum(steps[:3])), 1e-6)
        exit_direction = float(np.linalg.norm(second))/max(float(np.sum(steps[3:])), 1e-6)
        turn = float(np.dot(first, second))/max(float(np.linalg.norm(first)*np.linalg.norm(second)), 1e-6)
        if (post[3].t-onset.t < .09 or post[-1].t-onset.t < .25
                or travel < .9 or exit_travel < .4 or direction < .94 or exit_direction < .90
                or turn > .25 or max(steps)/diameter > 1.25
                or min(f.cue_ball_stable_normalized_speed or 0 for f in post[1:4]) < 5):
            continue
        quiet = [f for f in pre if measured(f)]
        visible_rest = False
        if len(quiet) >= 4 and quiet[-1].t-quiet[0].t >= .12-1e-6:
            anchor = np.asarray([(f.cue_ball_x, f.cue_ball_y) for f in quiet])
            center = np.median(anchor, axis=0)
            visible_rest = bool(np.max(np.linalg.norm(anchor-center, axis=1)) <= .15*diameter
                and np.linalg.norm(xy[0]-center) <= 3*diameter
                and onset.t-quiet[-1].t <= .16+1e-6)
        hidden_rest = (pre[-1].t-pre[0].t >= .50 and not quiet
                       and np.median([f.max_ball_normalized_speed for f in pre]) <= 1.
                       and np.median([f.motion_raw for f in pre]) <= .20)
        addressed = max((f.cue_contact_score for f in pre+post[:4]
                         if f.cue_tip_visible and 0 < f.cue_tip_distance_to_ball <= 1.3*diameter), default=0.)
        if not (visible_rest and addressed >= .25 or hidden_rest and addressed >= .45):
            continue
        return StrikeCandidate(timestamp=onset.t, confidence=.76, camera_view=onset.view_type,
            uncertainty_start=quiet[-1].t if visible_rest else pre[0].t, uncertainty_end=onset.t,
            evidence={**proposal.evidence, 'score_supported_object_launch': 1.,
                'native_scoring_short_collision': 1., 'contact_time_upper_bound': 1.,
                'collision_launch_motion_start': onset.t,
                'collision_launch_travel_diameters': travel,
                'collision_exit_travel_diameters': exit_travel,
                'collision_launch_direction': direction, 'collision_exit_direction': exit_direction,
                'collision_turn_cosine': turn, 'collision_measured_frames': float(len(post)),
                'collision_address_score': addressed, 'collision_visible_rest': float(visible_rest),
                'collision_hidden_rest': float(hidden_rest),
                'object_launch_confirmation_time': post[-1].t})
    return None
