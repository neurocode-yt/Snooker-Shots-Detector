"""Source-verified title wipes require lettering and bounded replay ownership."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import FrameFeatures, StrikeCandidate


FIXTURES=Path(__file__).parent/'fixtures/classic_library'


def signature(name):
    return ViewClassifier.replay_stinger_signature(cv2.imread(str(FIXTURES/f'{name}.png')))


def test_original_source_blue_opening_and_closing_share_the_title_fingerprint():
    opening,closing=signature('blue-opening'),signature('blue-closing')
    assert len(opening)==len(closing)==193
    assert ReplayDetector._cosine(np.asarray(opening),np.asarray(closing))>=.92
    assert signature('blue-table-background')==[]
    assert signature('blue-other-replay')==[]


def test_blue_colour_or_unrelated_information_cannot_establish_the_graphic():
    blue=cv2.imread(str(FIXTURES/'blue-opening.png'))
    plain=np.full_like(blue,tuple(int(v) for v in blue[170,10]))
    assert ViewClassifier.replay_stinger_signature(plain)==[]
    cv2.putText(plain,'PLAYER STATISTICS',(180,180),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    assert ViewClassifier.replay_stinger_signature(plain)==[]
    blue[120:252,128:512]=0  # Remove the measured lettering.
    assert ViewClassifier.replay_stinger_signature(blue)==[]


@pytest.mark.parametrize('closing,prior,expected',[(True,True,True),(False,True,False),(True,False,False)])
def test_source_blue_replay_excludes_duplicate_and_retains_following_live_blue(config,closing,prior,expected):
    features=[FrameFeatures(t=37.56,appearance_signature=signature('blue-opening')),
              FrameFeatures(t=38.84),FrameFeatures(t=42.08,appearance_signature=signature('blue-closing') if closing else []),
              FrameFeatures(t=44.12)]
    candidates=([StrikeCandidate(timestamp=24.4,confidence=.9)] if prior else [])
    replay=StrikeCandidate(timestamp=38.84,confidence=.9)
    live=StrikeCandidate(timestamp=44.12,confidence=.9)
    candidates.extend([replay,live])
    ReplayDetector(config).mark_candidates(candidates,features)
    assert replay.possible_replay==expected
    assert not live.possible_replay


def test_different_graphic_families_cannot_be_paired(config):
    features=[FrameFeatures(t=10,appearance_signature=signature('blue-opening')),
              FrameFeatures(t=15,appearance_signature=[1.]*192)]
    candidates=[StrikeCandidate(timestamp=t,confidence=.9) for t in [1,12,17]]
    ReplayDetector(config).mark_candidates(candidates,features)
    assert all(not c.possible_replay for c in candidates)
