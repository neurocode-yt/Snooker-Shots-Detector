"""Quiet ball anchors separate a crowded launch from moving cue outlines."""

import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.types import FrameFeatures


OBSERVATIONS = json.loads((Path(__file__).parent/'fixtures/multicamera_packed_cue_windows.json')
                          .read_text())
CASES = OBSERVATIONS["cases"]


@pytest.mark.parametrize("name", list(CASES))
def test_source_ball_launch_uses_anchor_and_rejects_feathering(config, name):
    case = CASES[name]
    features = [FrameFeatures.model_validate(f) for f in case["features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    if case["expected_contact"] is None:
        assert not candidates
    else:
        assert len(candidates) == 1
        lo, hi = case["expected_contact"]
        assert lo <= candidates[0].timestamp <= hi
        if name == "red_launch":
            assert candidates[0].evidence["pre_motion_raw_median"] > .5
            assert candidates[0].evidence["anchor_direction_consistency"] >= .9
            assert candidates[0].evidence["impact_occlusion_contact"] == 1
        if name == "green_launch":
            assert candidates[0].evidence["cue_contact_score"] >= .7
            assert candidates[0].evidence["pre_ball_quiet_ratio"] < .8


def test_fast_coarse_scan_proposes_both_crowded_camera_launches(config):
    features = [FrameFeatures.model_validate(f) for f in OBSERVATIONS["coarse_features"]]
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_sparse_candidates(features)
    for contact in (912.43, 924.96):
        assert any(c.uncertainty_start <= contact <= c.uncertainty_end for c in candidates)
