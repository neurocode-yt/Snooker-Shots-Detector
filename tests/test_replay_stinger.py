import cv2
import numpy as np
import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import FrameFeatures, StrikeCandidate


def graphic(title="SNOOKER"):
    frame = np.full((180, 320, 3), 25, np.uint8)
    cv2.circle(frame, (160, 90), 54, (255, 0, 255), 3)
    cv2.circle(frame, (160, 90), 64, (0, 255, 255), 3)
    cv2.putText(frame, title, (116, 86), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
    return frame


def test_concentric_graphic_requires_two_colours_and_full_rings():
    assert len(ViewClassifier.replay_stinger_signature(graphic())) == 192
    partial = np.full((180, 320, 3), 25, np.uint8)
    cv2.ellipse(partial, (160, 90), (54, 54), 0, 0, 140, (255, 0, 255), 3)
    cv2.ellipse(partial, (160, 90), (64, 64), 0, 0, 140, (0, 255, 255), 3)
    assert ViewClassifier.replay_stinger_signature(partial) == []
    table = np.full((180, 320, 3), (30, 150, 30), np.uint8)
    cv2.circle(table, (160, 90), 54, (255, 0, 255), 3)
    cv2.circle(table, (160, 90), 64, (0, 255, 255), 3)
    assert ViewClassifier.replay_stinger_signature(table) == []


@pytest.mark.parametrize("closing,prior,expected", [(True, True, True), (False, True, False), (True, False, False)])
def test_bracketed_stingers_need_close_and_preceding_live_play(config, closing, prior, expected):
    signature = ViewClassifier.replay_stinger_signature(graphic())
    features = [FrameFeatures(t=t, appearance_signature=signature if t == 10 or closing and t == 16 else [],
                              table_observable=10 < t < 16) for t in (1, 10, 11, 12, 14, 16, 17)]
    candidates = ([StrikeCandidate(timestamp=1, confidence=.9)] if prior else [])
    replay = StrikeCandidate(timestamp=11, confidence=.9)
    live = StrikeCandidate(timestamp=17, confidence=.9)
    candidates.extend((replay, live))
    detector = ReplayDetector(config)
    detector.mark_candidates(candidates, features)
    assert replay.possible_replay == expected
    assert bool(replay.evidence.get("replay_signature_confirmed")) == expected
    assert not live.possible_replay
    detector.mark_candidates(candidates, features)
    assert replay.possible_replay == expected


def test_closing_stinger_cannot_open_an_overlapping_replay(config):
    signature = ViewClassifier.replay_stinger_signature(graphic())
    features = [FrameFeatures(t=t, appearance_signature=signature if t in (10, 16, 19, 25) else [])
                for t in (1, 10, 11, 16, 17, 19, 20, 25, 26)]
    candidates = [StrikeCandidate(timestamp=t, confidence=.9) for t in (1, 11, 17, 20, 26)]
    ReplayDetector(config).mark_candidates(candidates, features)
    assert [c.timestamp for c in candidates if c.possible_replay] == [11, 20]


def test_missing_replay_contact_cannot_reassign_its_closing_wipe_to_live_play(config):
    signature = ViewClassifier.replay_stinger_signature(graphic())
    features = [FrameFeatures(t=t, appearance_signature=signature if t in (10,16,19,25) else [])
                for t in (1,10,11,16,17,19,20,25,26)]
    candidates = [StrikeCandidate(timestamp=t,confidence=.9) for t in (1,17,20,26)]
    ReplayDetector(config).mark_candidates(candidates,features)
    assert [c.timestamp for c in candidates if c.possible_replay] == [20]
    assert next(f for f in features if f.t==11).broadcast_replay
    assert not next(f for f in features if f.t==17).broadcast_replay


def test_recalculation_clears_owned_spans_and_preserves_independent_markers(config):
    signature = ViewClassifier.replay_stinger_signature(graphic())
    features = [FrameFeatures(t=t,appearance_signature=signature if t in (10,16) else [],
                              broadcast_replay=t==40) for t in (1,10,11,16,17,40)]
    candidates = [StrikeCandidate(timestamp=t,confidence=.9) for t in (1,11,17,40)]
    detector = ReplayDetector(config)
    detector.mark_candidates(candidates,features)
    assert candidates[1].possible_replay
    features[3].appearance_signature=[]
    detector.mark_candidates(candidates,features)
    assert not candidates[1].possible_replay
    assert candidates[-1].possible_replay


def test_original_source_live_black_survives_filtered_red_replay(config,tmp_path):
    import json
    from pathlib import Path
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.segmentation.builder import SegmentBuilder
    raw=json.loads((Path(__file__).parent/'fixtures/replay_closing_followup.json').read_text())
    features=[FrameFeatures.model_validate(f) for f in raw['features']]
    Analyzer._restore_legacy_replay_annotations(features,raw['features'])
    candidates=[StrikeCandidate.model_validate(c) for c in raw['candidates']]
    ReplayDetector(config).mark_candidates(candidates,features)
    assert all(not c.possible_replay for c in candidates)
    assert next(f for f in features if f.t==915.04).broadcast_replay
    assert not next(f for f in features if f.t==921.16).broadcast_replay
    shots=SegmentBuilder(config).build(candidates,features,1229.8)
    assert [s.cue_strike for s in shots if s.included]==[905.56,921.16]
