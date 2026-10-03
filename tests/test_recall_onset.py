"""Preserve measured launch timing when adopting a separately refined stop."""

import json
from pathlib import Path

import pytest

from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import FrameFeatures, StrikeCandidate


@pytest.mark.parametrize("case_name", ["recall_onset_before_native_stop", "recall_breakoff_before_native_stop"])
def test_refined_end_preserves_earlier_observed_closeup_launch(config, case_name):
    case = json.loads((Path(__file__).parent / f"fixtures/{case_name}.json").read_text())
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    candidate = StrikeCandidate.model_validate(case["candidate"])
    assert candidate.evidence["refined_ball_motion_start"] > candidate.timestamp + .75
    shots = SegmentBuilder(config).build([candidate], features, features[-1].t)
    assert len(shots) == 1
    shot = shots[0]
    assert case["source_contact_bounds"][0] <= shot.cue_strike <= case["source_contact_bounds"][1]
    assert shot.ball_motion_start < shot.cue_strike + .75
    assert shot.physical_stop_timestamp == pytest.approx(candidate.evidence["refined_stop_timestamp"])
    assert shot.clip_end-shot.clip_start >= 4
