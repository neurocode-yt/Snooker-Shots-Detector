"""Sparse uncertainty must trigger bounded verification, not silent omission."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper


def partial_features():
    return [FrameFeatures(t=i*.5, observation_fps=2, view_type=CameraViewType.BALL_CLOSEUP,
                          table_observable=True, table_confidence=.7, table_mask_area_ratio=.5,
                          observation_valid=False, ball_count=2, ball_diameter_px=50)
            for i in range(20)]


@pytest.mark.parametrize("case", ["soft_colour", "occluded_closeup"])
def test_source_closeups_without_sparse_onset_receive_contact_verification(config, tmp_path, case):
    fixture = json.loads((Path(__file__).parent/'fixtures/recall_coarse_coverage.json').read_text())["cases"][case]
    features = [FrameFeatures.model_validate(f) for f in fixture["features"]]
    analyzer = Analyzer(config, tmp_path)
    proposals = analyzer._coverage_proposals(features, [])
    contact = fixture["reference_contact"]
    assert any(c.timestamp-2 <= contact <= c.timestamp+2 for c in proposals)
    assert all(c.evidence["coverage_proposal"] == 1 for c in proposals)
    assert all(not analyzer.segmenter._candidate_supported(c) for c in proposals)


@pytest.mark.parametrize("field,value", [
    ("match_context_valid", False), ("broadcast_replay", True),
    ("table_handling", True), ("rack_idle", True),
    ("table_observable", False), ("view_type", CameraViewType.AUDIENCE),
])
def test_confirmed_unusable_context_does_not_open_coverage_windows(config, tmp_path, field, value):
    features = partial_features()
    for frame in features:
        setattr(frame, field, value)
    assert Analyzer(config, tmp_path)._coverage_proposals(features, []) == []


def test_camera_cut_and_aggregate_motion_without_local_ball_evidence_do_not_seed(config, tmp_path):
    features = [FrameFeatures(t=i*.5, view_type=CameraViewType.MAIN_TABLE,
                             table_observable=True, table_confidence=.8,
                             ball_count=6, motion_raw=1, scene_cut_score=1 if i == 5 else 0)
                for i in range(12)]
    assert Analyzer(config, tmp_path)._coverage_proposals(features, []) == []
    for frame in features:
        frame.view_type = CameraViewType.BALL_CLOSEUP
        frame.ball_count = 0
    assert Analyzer(config, tmp_path)._coverage_proposals(features, []) == []


def test_continuous_uncertain_action_has_bounded_windows_and_reuses_proposal_coverage(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    features = partial_features()
    proposals = analyzer._coverage_proposals(features, [])
    assert 3 <= len(proposals) <= 6
    assert all(any(c.timestamp-2 <= f.t-.5 and f.t+.5 <= c.timestamp+2 for c in proposals)
               for f in features[1:])
    existing = [StrikeCandidate(timestamp=4.5, confidence=.6, uncertainty_start=0,
                                evidence={"sparse_proposal": 1})]
    additional = analyzer._coverage_proposals(features, existing)
    assert not any(3 <= c.timestamp <= 6 for c in additional)


def test_merged_sparse_uncertainty_is_actually_decoded_through_later_edge(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    proposals = [StrikeCandidate(timestamp=3, confidence=.7, uncertainty_start=1.5,
                                uncertainty_end=4.5, evidence={"sparse_proposal": 1}),
                 StrikeCandidate(timestamp=4, confidence=.5, uncertainty_start=2.5,
                                uncertainty_end=5.5,
                                evidence={"sparse_proposal": 1, "coverage_proposal": 1})]
    merged = analyzer._deduplicate_candidates(proposals)
    assert len(merged) == 1
    start, end = analyzer._contact_window(merged[0])
    assert start <= 1.5 and end >= 6


def test_coverage_proposal_still_requires_native_contact_and_records_rejection(config, tmp_path, monkeypatch):
    analyzer = Analyzer(config, tmp_path)
    features = partial_features()
    proposal = analyzer._coverage_proposals(features, [])[0]
    calls = []

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return features[:9], [], []

    monkeypatch.setattr(analyzer, "_extract_features", extract)
    candidates, _ = analyzer._refine_adaptive_windows(
        Path('proxy.mp4'), None, TimeMapper(10, source_fps=25), 10, [proposal], resume=False,
    )
    assert len(calls) == 1  # No stop scan for an unconfirmed proposal.
    assert not analyzer.segmenter._candidate_supported(candidates[0])
    analyzer._record_detection_stage("native_complete", [])
    saved = json.loads((tmp_path/'detection_diagnostics.json').read_text())
    assert saved["native_proposals"][0]["reason"] == "native_transition_unconfirmed"
    assert saved["native_proposals"][0]["accepted"] is False
    assert saved["stages"][0]["candidates"] == 0


def test_verified_target_return_reopens_only_bounded_episode_and_queues_zero_object_rows(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    features = partial_features()
    for f in features:
        f.match_context_valid = False
        f.ball_count = 0
    analyzer._coarse_context_reference = [f.model_copy() for f in features]
    analyzer._confirm_target_return(features, 2, 6)
    assert all(f.match_context_valid == (2 <= f.t <= 6) for f in features)
    assert all(not f.observation_valid for f in features)
    assert all(f.match_context_valid == (2 <= f.t <= 6) for f in analyzer._coarse_context_reference)
    proposals = analyzer._coverage_proposals(features, [])
    assert proposals and all(c.evidence["coverage_target_return"] for c in proposals)
    assert all(2 <= c.timestamp <= 6 for c in proposals)
    analyzer._save_coarse_cache("test", features, [], proposals)
    restored = Analyzer(config, tmp_path)
    assert restored._load_coarse_cache("test") is not None
    assert restored._confirmed_target_returns == [(2, 6)]


def cue_image(diameter, aligned=True, shaft=True):
    image = np.full((540, 960, 3), (30, 115, 30), np.uint8)
    center = (650, 300)
    cv2.circle(image, center, int(diameter/2), (240, 240, 240), -1)
    if shaft:
        y = center[1] if aligned else center[1]-diameter
        cv2.line(image, (center[0]-int(2*diameter), int(y)),
                 (center[0]-int(.52*diameter), int(y)), (170, 200, 215), 8)
    cue = SimpleNamespace(visible=True, positions=[(0, *center)])
    return image, cue


def test_large_closeup_uses_visible_shaft_without_impossible_hough_length(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    image, cue = cue_image(145)
    result = analyzer._cue_geometry(image, None, cue, 145, 0)
    assert result["visible"]
    assert .30*145 <= result["distance"] <= 1.25*145


@pytest.mark.parametrize("aligned,shaft", [(False, True), (True, False)])
def test_large_white_circle_or_misaligned_edge_is_not_cue_evidence(config, tmp_path, aligned, shaft):
    image, cue = cue_image(145, aligned=aligned, shaft=shaft)
    assert not Analyzer(config, tmp_path)._cue_geometry(image, None, cue, 145, 0)["visible"]
