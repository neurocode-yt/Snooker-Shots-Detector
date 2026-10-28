"""Only source-verified contact can turn a score hint into an extra shot."""
from snooker_ai.pot_detection.recall import scoring_gap_proposals
from snooker_ai.pot_detection.recall import scoring_object_contact
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import ShotRecord
from snooker_ai.types import FrameFeatures, StrikeCandidate
import pytest
from tests.test_pot_selection import readings


def existing(t):
    return ShotRecord(shot_id=1,cue_strike=t,clip_start=t-2,clip_end=t+4)


def test_missing_colour_between_two_detected_contacts_requests_native_check():
    samples=(readings((20,4),start=8,end=11.5)+readings((21,4),start=12,end=24.5)
             +readings((28,4),start=25,end=30))
    proposals=scoring_gap_proposals(samples,[existing(10),existing(40)],60)
    assert len(proposals)==1
    assert proposals[0].uncertainty_start <= 22 <= proposals[0].uncertainty_end
    assert proposals[0].evidence['score_assisted_proposal']==1
    assert not SegmentBuilder._candidate_supported(proposals[0])


def test_one_detected_contact_cannot_cover_two_fast_scoring_updates():
    samples=(readings((20,4),start=8,end=11.5)+readings((21,4),start=12,end=17.5)
             +readings((28,4),start=18,end=23))
    proposals=scoring_gap_proposals(samples,[existing(10)],60)
    assert len(proposals)==1
    assert proposals[0].uncertainty_start >= 14


def test_already_detected_red_and_colour_do_not_open_duplicate_checks():
    samples=(readings((20,4),start=8,end=11.5)+readings((21,4),start=12,end=24.5)
             +readings((28,4),start=25,end=30))
    assert scoring_gap_proposals(samples,[existing(10),existing(22)],60)==[]


def test_resets_digit_glitches_and_unknown_score_gaps_never_request_shots():
    samples=(readings((70,30),start=8,end=11.5)+readings((0,0),start=12,end=24.5)
             +readings((7,0),start=40,end=45))
    assert scoring_gap_proposals(samples,[],60)==[]
    glitch=readings((20,4),start=8,end=11.5)+readings((50,4),start=12,end=17.5)
    assert scoring_gap_proposals(glitch,[],60)==[]


def test_replay_contact_cannot_claim_a_live_score_update():
    replay=existing(10);replay.possible_replay=True
    samples=readings((20,4),start=8,end=11.5)+readings((21,4),start=12,end=17.5)
    assert len(scoring_gap_proposals(samples,[replay],60))==1


@pytest.mark.parametrize('points,delay', [(1, 15), (4, 14), (4, 20)])
def test_delayed_score_is_attached_once_to_the_existing_stroke(points, delay):
    samples = (readings((20, 4), start=8, end=20)
               + readings((20+points, 4), start=10+delay, end=12+delay))
    assert scoring_gap_proposals(samples, [existing(10)], 60) == []


def test_delayed_update_cannot_hide_a_following_missing_pot_or_bridge_a_reset():
    samples = (readings((20, 4), start=8, end=20)
               + readings((21, 4), start=25, end=30)
               + readings((25, 4), start=31, end=35))
    proposals = scoring_gap_proposals(samples, [existing(10)], 60)
    assert len(proposals) == 1 and proposals[0].evidence['score_delta'] == 4
    samples = (readings((40, 4), start=8, end=20)
               + readings((0, 0), start=21, end=25)
               + readings((1, 0), start=26, end=30))
    assert len(scoring_gap_proposals(samples, [existing(10)], 60)) == 1


def test_identical_score_on_return_from_cutaway_retains_the_stroke_baseline():
    samples = (readings((0, 0), start=8, end=11)
               + readings((0, 0), start=17, end=21.5)
               + readings((4, 0), start=22, end=24))
    assert scoring_gap_proposals(samples, [existing(10)], 60) == []


def object_rows():
    return [FrameFeatures(t=10+i/30,observation_fps=30,view_type='ball_closeup',
        table_confidence=.9,scoring_object_launch_count=1) for i in range(6)]


def test_confirmed_score_and_measured_coloured_sphere_bound_hidden_contact():
    hint=StrikeCandidate(timestamp=10,confidence=.5,uncertainty_start=9,uncertainty_end=11,
        evidence={'score_assisted_proposal':1.,'score_delta':6.})
    contact=scoring_object_contact(hint,object_rows())
    assert contact is not None and contact.timestamp==pytest.approx(9.7)
    assert contact.evidence['contact_time_upper_bound']==1
    assert SegmentBuilder._candidate_supported(contact)
    assert scoring_object_contact(hint.model_copy(update={'evidence':{}}),object_rows()) is None


@pytest.mark.parametrize('defect',['replay','handling','foreign','sparse','cut','camera','ordinary_object','gap'])
def test_score_hint_cannot_promote_unmeasured_objects_or_invalid_context(defect):
    hint=StrikeCandidate(timestamp=10,confidence=.5,uncertainty_start=9,uncertainty_end=11,
        evidence={'score_assisted_proposal':1.,'score_delta':6.})
    rows=object_rows()
    for i,row in enumerate(rows):
        if defect=='replay':row.broadcast_replay=True
        elif defect=='handling':row.table_handling=True
        elif defect=='foreign':row.match_context_valid=False
        elif defect=='sparse':row.observation_fps=2
        elif defect=='cut':row.scene_cut_score=.8
        elif defect=='camera':row.camera_motion_magnitude=5
        elif defect=='ordinary_object':
            row.scoring_object_launch_count=0
            row.object_ball_launch_count=2
        else:row.camera_scene_id=i+1
    assert scoring_object_contact(hint,rows) is None


def test_scoring_recovery_preserves_previous_replay_boundaries(config,tmp_path,monkeypatch):
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.replay_detection.detector import ReplayDetector
    from snooker_ai.scene_detection.view_classifier import ViewClassifier
    from tests.test_replay_stinger import graphic
    signature=ViewClassifier.replay_stinger_signature(graphic())
    rows=[FrameFeatures(t=t,appearance_signature=signature if t in (10,16) else [])
          for t in (1,10,11,16,17,30)]
    prior=[StrikeCandidate(timestamp=t,confidence=.9) for t in (1,11,17)]
    owner=Analyzer(config,tmp_path/'score-replay')
    ReplayDetector(config).mark_candidates(prior,rows)
    assert rows[2].broadcast_replay
    monkeypatch.setattr('snooker_ai.pipeline.analyzer.PromotionDetector.detect',lambda *a:{})
    live=StrikeCandidate(timestamp=30,confidence=.9,evidence={'dense_transition_confirmed':1.})
    replay=StrikeCandidate(timestamp=11,confidence=.9,evidence={'dense_transition_confirmed':1.})
    recovered=owner._screen_scoring_contacts(tmp_path/'source',prior,[live,replay],rows,[],[])
    assert recovered==[live]
    assert rows[2].broadcast_replay  # No live shot can extend through this package.
    assert prior[1].possible_replay
