"""Cross-camera coordinate jumps and foreign context must not create shots."""

import cv2
import json
import numpy as np
import pytest

from snooker_ai.config import load_config
from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.replay_detection.detector import ReplayDetector
from snooker_ai.table_detection.localizer import TableObservation
from snooker_ai.types import CameraViewType, FrameFeatures, SceneSegment, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper


def launch_features(scene_change=False):
    features = []
    for i in range(26):
        t = i * 0.1
        moving = i >= 10
        features.append(FrameFeatures(
            t=t, table_confidence=0.9, view_type=CameraViewType.MAIN_TABLE,
            ball_diameter_px=10, cue_ball_detected=True, cue_ball_track_confidence=0.9,
            camera_scene_id=1 if scene_change and moving else 0,
            cue_ball_x=100 + max(0, i-10)*4 + (200 if scene_change and moving else 0),
            cue_ball_y=100, cue_ball_normalized_speed=1.5 if moving else 0,
            cue_ball_acceleration=8 if i == 10 else 0,
            cue_contact_score=0.85 if i == 10 else 0,
            motion_raw=0.35 if moving else 0.02, motion_score=0.35 if moving else 0.02,
            max_ball_normalized_speed=1.5 if moving else 0,
            ball_residual_motion=0.5 if moving else 0,
        ))
    return features


def test_identical_observations_confirm_launch_only_when_camera_is_continuous(config):
    detector = StrikeDetector(config)
    continuous = launch_features()
    detector.score_frames(continuous)
    assert any(abs(c.timestamp-1) <= 0.1 for c in detector.detect_candidates(continuous))
    switched = launch_features(scene_change=True)
    detector.score_frames(switched)
    assert detector.detect_candidates(switched) == []
    metrics = detector._transition_metrics(switched, 10)
    assert metrics["pre_sample_count"] == 0
    assert metrics["stationary_ratio"] == 0


def test_sparse_white_coordinate_jump_between_cameras_does_not_propose_launch(config):
    features = [FrameFeatures(
        t=i*0.5, camera_scene_id=0 if i < 4 else 1,
        table_confidence=0.9, view_type=CameraViewType.MAIN_TABLE,
        cue_ball_detected=True, cue_ball_track_confidence=0.9, ball_diameter_px=10,
        cue_ball_x=100 if i < 4 else 350, cue_ball_y=100,
    ) for i in range(10)]
    assert StrikeDetector(config).detect_sparse_candidates(features) == []


def test_cue_address_before_closeup_cut_proposes_native_window_without_confirming_strike(config):
    features = [FrameFeatures(
        t=i*.5, camera_scene_id=0 if i < 4 else 1,
        scene_cut_score=1 if i == 4 else 0, table_confidence=.9,
        view_type=CameraViewType.MAIN_TABLE if i < 4 else CameraViewType.BALL_CLOSEUP,
        cue_ball_detected=i < 4, cue_ball_track_confidence=.9 if i < 4 else 0,
        cue_ball_x=100 if i < 4 else None, cue_ball_y=100 if i < 4 else None,
        ball_diameter_px=10, cue_tip_visible=i < 4, cue_tip_distance_to_ball=15,
    ) for i in range(10)]
    detector = StrikeDetector(config)
    proposals = detector.detect_sparse_candidates(features)
    covering = [c for c in proposals if c.evidence.get("camera_address_proposal")]
    assert covering
    assert covering[0].uncertainty_start <= 1.5
    assert covering[0].uncertainty_end >= 4.5
    assert not covering[0].evidence.get("dense_transition_confirmed")
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []
    for f in features:
        f.match_context_valid = False
    assert detector.detect_sparse_candidates(features) == []


def test_late_confirmed_contact_with_sparse_stop_rows_requires_forward_recovery(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    contact = StrikeCandidate(timestamp=1, confidence=.9, uncertainty_start=.9,
                              uncertainty_end=1.1, evidence={"dense_transition_confirmed": 1})
    features = launch_features()[:16]
    features.extend(FrameFeatures(
        t=t, observation_fps=2, view_classified=True, table_full_view=True,
        table_confidence=.9, ball_kinematics_valid=True,
    ) for t in (2, 2.5, 3, 3.5, 4))
    assert analyzer._unresolved_stop_candidates([contact], features, 10) == [contact]
    contact.evidence.update(refined_stop_timestamp=2, refined_stop_confidence=.85)
    assert analyzer._unresolved_stop_candidates([contact], features, 10) == []


def test_continuous_stop_coverage_does_not_trigger_additional_recovery(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    contact = StrikeCandidate(timestamp=1, confidence=.9, uncertainty_start=.9,
                              uncertainty_end=1.1, evidence={"dense_transition_confirmed": 1})
    features = launch_features()
    for f in features:
        f.ball_kinematics_valid = True
        f.table_full_view = True
        if f.t >= 1.6:
            f.max_ball_normalized_speed = f.cue_ball_normalized_speed = 0
            f.motion_score = f.motion_raw = f.ball_residual_motion = .01
    assert analyzer._unresolved_stop_candidates([contact], features, 10) == []


def test_large_cloth_geometry_change_invalidates_scale_even_when_green_histograms_match():
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], np.int32)
    before = TableObservation(confidence=.9, mask=None, contour=contour, bbox=(0, 88, 859, 452))
    overhead = TableObservation(confidence=.9, mask=None, contour=contour, bbox=(200, 156, 560, 294))
    assert Analyzer._table_view_changed(before, overhead)
    small_zoom = TableObservation(confidence=.9, mask=None, contour=contour, bbox=(8, 84, 845, 456))
    assert not Analyzer._table_view_changed(before, small_zoom)
    uncertain = TableObservation(confidence=.15, mask=None, contour=None, bbox=(200, 156, 560, 294))
    assert not Analyzer._table_view_changed(before, uncertain)


def test_cached_coarse_foreign_context_rejects_native_launch():
    dense = launch_features()
    reference = [FrameFeatures.model_validate_json(FrameFeatures(
        t=i*0.5, match_context_valid=False,
    ).model_dump_json()) for i in range(6)]
    Analyzer._apply_match_context(dense, reference)
    assert all(not f.match_context_valid and not f.observation_valid for f in dense)
    detector = StrikeDetector(load_config())
    detector.score_frames(dense)
    assert detector.detect_candidates(dense) == []


def test_context_propagation_does_not_contaminate_returned_target_camera():
    reference = [FrameFeatures(t=t, match_context_valid=valid) for t, valid in
                 [(9.5, True), (10, False), (10.5, False), (11, True)]]
    dense = [FrameFeatures(t=t) for t in (9.99, 10, 10.2, 10.7, 11, 11.2, 12)]
    Analyzer._apply_match_context(dense, reference)
    assert [f.match_context_valid for f in dense] == [True, False, False, False, True, True, True]


def test_scene_representative_label_cannot_overwrite_measured_closeup_geometry():
    feature = FrameFeatures(t=2, view_classified=True, table_full_view=False,
                            view_type=CameraViewType.BALL_CLOSEUP)
    Analyzer._annotate_scenes([feature], [SceneSegment(start=0, end=5,
                                                    view_type=CameraViewType.MAIN_TABLE)])
    assert feature.view_type == CameraViewType.BALL_CLOSEUP
    assert not feature.table_full_view


def scoreboard_frame(foreign=False):
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[80:299, 100:540] = (40, 140, 40)
    frame[306:325, 128:513] = (35, 35, 35)
    for x in (265, 348):
        cv2.rectangle(frame, (x, 306), (x+27, 323), (0, 255, 255), -1)
        cv2.putText(frame, "0", (x+7, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    if foreign:
        cv2.putText(frame, "Stan Moody", (190, 320), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
        cv2.putText(frame, "Liam Highfield", (404, 320), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255), 1)
    else:
        cv2.putText(frame, "Selby", (155, 320), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        cv2.putText(frame, "Lisowski", (405, 320), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    return frame


def test_native_seek_retains_coarse_foreign_decision_from_its_first_frame(tmp_path):
    config = load_config(overrides={"device": "cpu", "analysis": {
        "hwaccel_decode": False, "skip_racked_waits": False, "refine_fps": 10,
    }})
    analyzer = Analyzer(config, tmp_path / "job")
    for i in range(8):
        analyzer.broadcast_context.observe(scoreboard_frame(), i*0.5, CameraViewType.MAIN_TABLE)
    assert analyzer.broadcast_context.target_ready
    # A dense seek begins inside an already confirmed foreign interval; native
    # observations must not temporarily replace those cached coarse decisions.
    analyzer._coarse_context_reference = [FrameFeatures(t=i*0.5, match_context_valid=False)
                                          for i in range(5)]
    video = tmp_path / "foreign.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640,360))
    assert writer.isOpened()
    for i in range(20):
        frame = scoreboard_frame(foreign=True)
        cv2.circle(frame, (180+i*2, 180), 6, (240,240,240), -1)
        writer.write(frame)
    writer.release()
    features, _, _ = analyzer._extract_features(video, None, TimeMapper(2, source_fps=10),
                                               duration=2, sample_fps=10)
    assert len(features) >= 19
    assert all(not f.match_context_valid and not f.observation_valid for f in features)
    assert features[0].t == pytest.approx(0)


def hidden_impact_across_cut():
    """Aiming, a brief visible ball burst behind the bridge, then overhead roll.

    The old view supplies just one moving-ball observation; it cannot confirm
    an impact independently. The first overhead sample is a fresh tracker birth.
    """
    result = []
    for i in range(24):
        t = round(i * .1, 2)
        old = i < 14
        hidden = 10 <= i < 14
        cut = i == 14
        rolling = i >= 15
        burst = i == 10
        result.append(FrameFeatures(
            t=t, camera_scene_id=0 if old else 1, scene_cut_score=1 if cut else 0,
            observation_valid=not cut, table_confidence=.9,
            table_full_view=not old, view_classified=True,
            view_type=CameraViewType.BALL_CLOSEUP if old else CameraViewType.MAIN_TABLE,
            ball_diameter_px=10, cue_ball_detected=not hidden and not cut,
            cue_ball_track_confidence=.9 if not hidden and not cut else 0,
            cue_ball_x=None if hidden or cut else 100 if old else 400 + max(0, i-15)*4,
            cue_ball_y=None if hidden or cut else 100,
            cue_ball_normalized_speed=4 if i >= 16 else 0,
            max_ball_normalized_speed=2 if burst else 4 if i >= 16 else 0,
            moving_ball_count=1 if burst or i >= 16 else 0,
            motion_raw=.75 if burst else .35 if hidden or rolling else .02,
            motion_score=.75 if burst else .35 if hidden or rolling else .02,
            ball_residual_motion=.5 if burst or rolling else .02,
            cue_tip_visible=old and not hidden,
            cue_tip_distance_to_ball=15 if old and not hidden else 0,
        ))
    return result


def test_hidden_impact_followed_by_camera_cut_recovers_one_live_shot(config):
    features = hidden_impact_across_cut()
    detector = StrikeDetector(config)
    detector.score_frames(features)
    candidates = detector.detect_candidates(features)
    assert len(candidates) == 1
    shot = candidates[0]
    assert shot.timestamp == pytest.approx(1.0)
    assert shot.uncertainty_start <= shot.timestamp <= shot.uncertainty_end
    assert shot.evidence.get("camera_contact_inferred") == 1
    ReplayDetector(config).mark_candidates(candidates, features)
    assert not shot.possible_replay
    refined = detector.refine_boundaries([StrikeCandidate(
        timestamp=1.2, confidence=.55, uncertainty_start=.7, uncertainty_end=1.8,
        evidence={"sparse_proposal": 1},
    )], features)
    assert len(refined) == 1
    assert refined[0].timestamp == pytest.approx(1.0)
    assert refined[0].evidence["dense_transition_confirmed"] == 1


def test_camera_cut_and_cue_address_alone_do_not_create_hidden_impact(config):
    features = hidden_impact_across_cut()
    for f in features:
        f.max_ball_normalized_speed = 0
        f.cue_ball_normalized_speed = 0
        f.moving_ball_count = 0
        f.motion_raw = f.motion_score = f.ball_residual_motion = .02
        if f.cue_ball_x is not None and f.camera_scene_id == 1:
            f.cue_ball_x = 400
    detector = StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []


def test_colour_respot_with_stationary_white_across_cut_is_not_a_shot(config):
    features = hidden_impact_across_cut()
    for f in features:
        if f.camera_scene_id == 1:
            f.cue_ball_x = 400 if f.cue_ball_detected else None
            f.cue_ball_normalized_speed = 0
            # The colour is moving while the referee handles it; a quiet white
            # must not inherit that movement as cue impact after changing views.
            f.max_ball_normalized_speed = 2
            f.motion_raw = f.motion_score = .65
    detector = StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []


def test_cross_cut_contact_recovery_respects_explicit_foreign_native_context(config):
    features = hidden_impact_across_cut()
    reference = [FrameFeatures(t=i*.5, match_context_valid=False) for i in range(6)]
    Analyzer._apply_match_context(features, reference)
    detector = StrikeDetector(config)
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []


def test_native_foreign_rejection_revokes_inherited_sparse_contact_confirmation(config):
    features = hidden_impact_across_cut()
    for f in features:
        f.match_context_valid = False
        f.observation_valid = False
    candidate = StrikeCandidate(
        timestamp=1, confidence=.65, uncertainty_start=.7, uncertainty_end=1.8,
        evidence={"camera_contact_inferred": 1, "occlusion_inferred": 1,
                  "ball_onset_run": 2, "sparse_proposal": 1},
    )
    refined = StrikeDetector(config).refine_boundaries([candidate], features)[0]
    retained = bool(
        refined.evidence.get("dense_transition_confirmed", 0) >= .5
        or refined.evidence.get("sparse_dense_transition", 0) >= .5
        or (refined.evidence.get("occlusion_inferred", 0) >= .5
            and refined.evidence.get("ball_onset_run", 0) >= 2)
    )
    assert not retained


def test_native_unobservable_view_cannot_keep_sparse_inferred_contact(config):
    features = [FrameFeatures(t=i/10, view_classified=True, observation_valid=False,
                              table_observable=False, table_full_view=False) for i in range(20)]
    candidate = StrikeCandidate(timestamp=1, confidence=.65, uncertainty_start=.7,
                               uncertainty_end=1.8, evidence={"camera_contact_inferred": 1,
                               "occlusion_inferred": 1, "ball_onset_run": 2,
                               "dense_transition_confirmed": 1, "sparse_proposal": 1})
    StrikeDetector(config).refine_boundaries([candidate], features)
    assert candidate.evidence["dense_transition_confirmed"] == 0
    assert candidate.evidence["occlusion_inferred"] == 0


def test_merged_camera_ids_preserve_actual_cut_but_ignore_seek_local_ids():
    coarse = [FrameFeatures(t=t, camera_scene_id=100 if t < 1 else 200,
                            scene_cut_score=1 if t == 1 else 0)
              for t in (0, .5, 1, 1.5, 2)]
    dense = [FrameFeatures(t=t, camera_scene_id=0,
                           scene_cut_score=1 if t == 1 else 0)
             for t in (.8, .9, 1, 1.1, 1.2)]
    merged = Analyzer._merge_feature_layers(coarse, dense)
    before = [f.camera_scene_id for f in merged if f.t < 1]
    after = [f.camera_scene_id for f in merged if f.t >= 1]
    assert len(set(before)) == len(set(after)) == 1
    assert before[0] != after[0]


def test_contact_bridge_cannot_borrow_a_later_new_shot_after_quiet_reacquisition(config):
    features = hidden_impact_across_cut()
    for f in features:
        if f.camera_scene_id != 1:
            continue
        launched = f.t >= 2.0
        f.cue_ball_x = 400 + max(0, round(f.t*10)-20)*6 if f.cue_ball_detected else None
        f.cue_ball_normalized_speed = 6 if launched else 0
        f.cue_ball_acceleration = 8 if f.t == 2.0 else 0
        f.cue_contact_score = .85 if f.t == 2.0 else 0
        f.max_ball_normalized_speed = 6 if launched else 0
        f.motion_raw = f.motion_score = .35 if launched else .02
        f.ball_residual_motion = .5 if launched else .02
    detector = StrikeDetector(config)
    detector.score_frames(features)
    live = detector.detect_candidates(features)
    assert len(live) == 1 and live[0].timestamp == pytest.approx(2.0)
    # The earlier foreground/occlusion proposal has no native strike in its own
    # uncertainty window. A later, separate launch cannot confirm that proposal.
    proposed = StrikeCandidate(timestamp=1, confidence=.55,
                               uncertainty_start=.7, uncertainty_end=1.8,
                               evidence={"sparse_proposal": 1})
    refined = detector.refine_boundaries([proposed], features)[0]
    assert refined.evidence.get("dense_transition_confirmed", 0) == 0
    assert refined.evidence.get("camera_contact_inferred", 0) == 0


def test_native_window_beginning_at_cut_cannot_erase_coarse_camera_boundary():
    coarse = [FrameFeatures(t=.5, camera_scene_id=1, observation_fps=2),
              FrameFeatures(t=1, camera_scene_id=2, scene_cut_score=1, observation_fps=2),
              FrameFeatures(t=1.5, camera_scene_id=2, observation_fps=2)]
    # A seek begins at the first image from a new angle and has no previous
    # image to compare, so its first cut score is naturally zero.
    dense = [FrameFeatures(t=1, camera_scene_id=0, scene_cut_score=0, observation_fps=25),
             FrameFeatures(t=1.04, camera_scene_id=0, scene_cut_score=0, observation_fps=25)]
    merged = Analyzer._merge_feature_layers(coarse, dense)
    before = next(f for f in merged if f.t == .5)
    boundary = next(f for f in merged if f.t == 1)
    after = next(f for f in merged if f.t == 1.04)
    assert boundary.scene_cut_score >= .5
    assert before.camera_scene_id != boundary.camera_scene_id
    assert boundary.camera_scene_id == after.camera_scene_id


def test_native_cut_replaces_delayed_coarse_cut_without_mutating_contact():
    coarse = [FrameFeatures(t=t, observation_fps=2,
                            scene_cut_score=1 if t == 1.5 else 0)
              for t in (.5, 1, 1.5, 2)]
    dense = [FrameFeatures(t=round(.8+i*.04, 2), observation_fps=25,
                           camera_scene_id=7 if i < 10 else 8,
                           scene_cut_score=1 if i == 10 else 0)
             for i in range(26)]
    original_ids = [f.camera_scene_id for f in dense]
    merged = Analyzer._merge_feature_layers(coarse, dense)
    assert [f.t for f in merged if f.scene_cut_score >= .5] == [1.2]
    assert len({f.camera_scene_id for f in merged if 1.2 <= f.t <= 1.8}) == 1
    assert [f.camera_scene_id for f in dense] == original_ids
    assert coarse[2].scene_cut_score == 1


def test_native_measurement_wins_at_exact_coarse_cut_timestamp():
    coarse = [FrameFeatures(t=1, scene_cut_score=1)]
    dense = [FrameFeatures(t=t, camera_scene_id=3,
                           scene_cut_score=1 if t == .9 else 0)
             for t in (.8, .9, 1, 1.1)]
    merged = Analyzer._merge_feature_layers(coarse, dense)
    assert [f.t for f in merged if f.scene_cut_score >= .5] == [.9]
    assert dense[2].scene_cut_score == 0


def test_travel_layer_cannot_interleave_or_replace_native_contact_kinematics():
    native = [FrameFeatures(t=round(1+i*.04, 2), observation_fps=25,
                            cue_ball_normalized_speed=4) for i in range(11)]
    travel = [FrameFeatures(t=t, observation_fps=10,
                            cue_ball_normalized_speed=99) for t in (1.05, 1.15, 1.2, 1.25, 1.35)]
    merged = Analyzer._merge_feature_layers(native, travel)
    assert [f.t for f in merged] == [f.t for f in native]
    assert all(f.cue_ball_normalized_speed == 4 for f in merged)
    assert travel[0].cue_ball_normalized_speed == 99


def test_short_shot_stop_warmup_cannot_replace_its_verified_contact():
    contact = [FrameFeatures(t=round(1+i*.04, 2), observation_fps=25,
                             contact_window=True, cue_ball_normalized_speed=4)
               for i in range(16)]
    stopping = [FrameFeatures(t=f.t, observation_fps=25,
                              cue_ball_normalized_speed=0) for f in contact]
    merged = Analyzer._merge_feature_layers(contact, stopping)
    assert all(f.contact_window and f.cue_ball_normalized_speed == 4 for f in merged)


@pytest.mark.parametrize("source_fps", [24, 25, 30, 60])
def test_contact_refinement_never_upsamples_source_frames(tmp_path, monkeypatch, config, source_fps):
    analyzer = Analyzer(config, tmp_path / "job")
    calls = []
    def extract(*args, **kwargs):
        calls.append(kwargs["sample_fps"])
        return [], [], []
    monkeypatch.setattr(analyzer, "_extract_features", extract)
    analyzer._refine_candidate_windows(
        tmp_path / "video.mp4", None,
        TimeMapper(source_duration=10, source_fps=source_fps, analysis_fps=30),
        10, [StrikeCandidate(timestamp=5, confidence=.6)], [], resume=False)
    assert calls == [min(source_fps, 30)]


def test_cached_overlapping_seek_ids_cannot_create_an_unobserved_camera_cut(tmp_path, monkeypatch):
    config = load_config(overrides={"analysis": {"refine_fps": 10}})
    analyzer = Analyzer(config, tmp_path / "job")
    cached = launch_features()
    for f in cached:
        # Different independently merged windows used different local IDs;
        # neither measured a cut anywhere in this actual continuous shot.
        f.camera_scene_id = 101 if f.t < 1 else 202
        f.view_classified = True
        f.observation_fps = 10
    monkeypatch.setattr(analyzer, "_extract_features", lambda *args, **kwargs: ([], [], []))
    candidate = StrikeCandidate(timestamp=1, confidence=.6,
                                uncertainty_start=.8, uncertainty_end=1.3)
    candidates, _ = analyzer._refine_candidate_windows(
        tmp_path / "source.mp4", None, TimeMapper(source_duration=2.5, source_fps=10),
        2.5, [candidate], [], existing_dense=cached, resume=False)
    assert candidates[0].evidence.get("dense_transition_confirmed") == 1
    assert candidates[0].timestamp == pytest.approx(1)


def test_coarse_cache_binds_learned_target_to_its_source_signature(tmp_path):
    config = load_config(overrides={"device": "cpu"})
    original = Analyzer(config, tmp_path / "job")
    for i in range(8):
        original.broadcast_context.observe(scoreboard_frame(), i*.5, CameraViewType.MAIN_TABLE)
    expected = original.broadcast_context.export_target()
    assert expected
    original._save_coarse_cache("source-A", [FrameFeatures(t=0)], [], [])
    restored = Analyzer(config, tmp_path / "job")
    assert restored._load_coarse_cache("source-B") is None
    assert not restored.broadcast_context.target_ready
    assert restored._load_coarse_cache("source-A") is not None
    assert restored.broadcast_context.export_target() == pytest.approx(expected, abs=1e-7)


def test_old_unbound_target_file_cannot_contaminate_current_source_cache(tmp_path):
    config = load_config(overrides={"device": "cpu"})
    analyzer = Analyzer(config, tmp_path / "job")
    for i in range(8):
        analyzer.broadcast_context.observe(scoreboard_frame(), i*.5, CameraViewType.MAIN_TABLE)
    # Simulate the former two-file crash: source B's coarse cache was committed,
    # while the separate target file still belongs to source A.
    (analyzer.job_dir / "broadcast_target.json").write_text(
        json.dumps(analyzer.broadcast_context.export_target()), encoding="utf-8")
    analyzer.broadcast_context.reset_observations(preserve_target=False)
    analyzer._save_coarse_cache("source-B", [FrameFeatures(t=0)], [], [])
    payload = json.loads(analyzer._coarse_cache_path().read_text(encoding="utf-8"))
    payload.pop("broadcast_target")  # A legacy cache has no bundled identity.
    analyzer._coarse_cache_path().write_text(json.dumps(payload), encoding="utf-8")
    # Even a reused Analyzer must drop an earlier match's target on that load.
    for i in range(8):
        analyzer.broadcast_context.observe(scoreboard_frame(), i*.5, CameraViewType.MAIN_TABLE)
    assert analyzer.broadcast_context.target_ready
    assert analyzer._load_coarse_cache("source-B") is not None
    assert not analyzer.broadcast_context.target_ready


@pytest.mark.parametrize("source_fps", [24.0, 25.0])
def test_native_lower_frame_rate_receives_dense_detection_settings(tmp_path, source_fps):
    import cv2
    from snooker_ai.utils.timebase import TimeMapper
    config = load_config(overrides={"device": "cpu", "analysis": {"hwaccel_decode": False}})
    path = tmp_path / "native.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    assert writer.isOpened()
    frame = np.full((90, 160, 3), (40, 140, 40), np.uint8)
    for _ in range(30):
        writer.write(frame)
    writer.release()
    analyzer = Analyzer(config, tmp_path / "job")
    calls = []
    original = analyzer.objects.detect

    def detect(frame, *args, **kwargs):
        calls.append(kwargs["use_hough"])
        return original(frame, *args, **kwargs)

    analyzer.objects.detect = detect
    features, _, _ = analyzer._extract_features(
        path, None, TimeMapper(source_duration=1, proxy_duration=1, source_fps=source_fps),
        1, sample_fps=source_fps, end_time=1)
    assert features and calls
    # Native 24/25fps must use refine_hough_fps, rather than sparse 5fps circles
    # simply because it is below the configured 30fps upper refinement target.
    assert sum(calls) >= len(calls)/2
