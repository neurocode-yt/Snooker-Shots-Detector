import json
from pathlib import Path

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures


def test_closeup_finger_swaps_are_rejected_without_losing_the_following_real_red(config):
    source=json.loads((Path(__file__).parent/'fixtures/classic_hard/finger-swaps-v37.json').read_text())
    features=[FrameFeatures.model_validate(f) for f in source['features']]
    detector=StrikeDetector(config)
    detector.score_frames(features)
    candidates=detector.detect_candidates(features)
    assert not any(270.<c.timestamp<272. for c in candidates)
    assert len([c for c in candidates if 273.5<c.timestamp<274.1])==1
