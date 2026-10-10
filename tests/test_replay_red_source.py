from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import FrameFeatures, StrikeCandidate


FIXTURES=Path(__file__).parent/'fixtures/classic_hard'


def signature(name):
    return ViewClassifier.replay_stinger_signature(cv2.imread(str(FIXTURES/f'{name}.png')))


def test_original_source_animation_phases_match_only_the_rendered_graphic():
    opening,closing=signature('red-opening'),signature('red-closing')
    assert len(opening)==len(closing)==194
    assert ReplayDetector._cosine(np.asarray(opening),np.asarray(closing))>.99
    photographed=np.full((180,320,3),(30,145,30),np.uint8)
    cv2.circle(photographed,(160,90),68,(10,10,210),-1)
    cv2.circle(photographed,(142,65),12,(245,245,245),-1)
    assert ViewClassifier.replay_stinger_signature(photographed)==[]
    assert signature('red-live-table')==[]


@pytest.mark.parametrize('closing,prior,expected',[(True,True,True),(False,True,False),(True,False,False)])
def test_red_replay_marker_pair_preserves_the_following_live_black(config,closing,prior,expected):
    features=[FrameFeatures(t=229.32,appearance_signature=signature('red-opening')),
              FrameFeatures(t=230.92),
              FrameFeatures(t=235.64,appearance_signature=signature('red-closing') if closing else []),
              FrameFeatures(t=254.28)]
    candidates=([StrikeCandidate(timestamp=220.16,confidence=.9)] if prior else [])
    replay=StrikeCandidate(timestamp=230.92,confidence=.9)
    black=StrikeCandidate(timestamp=254.28,confidence=.9)
    candidates.extend([replay,black])
    ReplayDetector(config).mark_candidates(candidates,features)
    assert replay.possible_replay==expected
    assert not black.possible_replay
