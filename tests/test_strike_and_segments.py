import pytest
import numpy as np

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.audio.features import AudioFeatures
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import (
    CameraViewType,
    EditMode,
    FrameFeatures,
    ShotRecord,
    StrikeCandidate,
)


def _feat(t, motion=0.0, strike=0.0, onset=0.0, table=0.8):
    return FrameFeatures(
        t=t,
        motion_score=motion,
        motion_raw=motion,
        strike_score=strike,
        audio_onset=onset,
        table_confidence=table,
        view_type=CameraViewType.MAIN_TABLE,
        residual_motion_mean=motion * 2,
        residual_motion_max=motion * 5,
        motion_area_ratio=motion * 0.02,
    )


def test_audio_peaks_are_isolated_for_recovery_windows():
    times = np.arange(0.0, 5.0, 0.1, dtype=np.float32)
    onset = np.zeros_like(times)
    onset[[10, 11, 30, 31, 42]] = [0.35, 0.70, 0.55, 0.45, 0.80]
    audio = AudioFeatures(
        times=times,
        onset_env=onset,
        rms=np.zeros_like(times),
        highband=np.zeros_like(times),
        midband=np.zeros_like(times),
        sample_rate=16000,
    )

    peaks = audio.cue_peaks(min_score=0.30, min_distance=1.0)

    assert [round(item[0], 1) for item in peaks] == [1.1, 3.0, 4.2]
    assert peaks[0][1] == pytest.approx(0.70)


def test_audio_seed_adds_only_unmatched_transients(config, tmp_path, monkeypatch):
    analyzer = Analyzer(config, tmp_path / "job")
    times = np.arange(0.0, 8.0, 0.1, dtype=np.float32)
    onset = np.zeros_like(times)
    onset[[10, 30, 50]] = [0.8, 0.7, 0.9]
    audio = AudioFeatures(
        times=times,
        onset_env=onset,
        rms=np.zeros_like(times),
        highband=np.zeros_like(times),
        midband=np.zeros_like(times),
        sample_rate=16000,
    )
    monkeypatch.setattr(analyzer.audio_ext, "extract", lambda _path: audio)
    existing = [StrikeCandidate(timestamp=1.0, confidence=0.9)]

    seeded = analyzer._seed_audio_candidates(
        existing, tmp_path / "audio.wav", duration=8.0
    )

    assert [round(item.timestamp, 1) for item in seeded] == [1.0, 3.0, 5.0]
    assert seeded[-1].evidence["audio_seed"] == 1.0


def test_strike_detector_finds_peaks(config):
    feats = []
    for i in range(50):
        t = i * 0.1
        # motion onset around t=2.0 and t=5.0
        motion = 0.1
        if 2.0 <= t <= 3.0:
            motion = 0.1 + (t - 2.0) * 0.8
        if 5.0 <= t <= 6.0:
            motion = 0.1 + (t - 5.0) * 0.8
        if 3.0 < t < 3.5 or 6.0 < t < 6.5:
            motion = 0.7
        feats.append(_feat(t, motion=min(motion, 1.0), onset=0.5 if abs(t - 2.1) < 0.15 else 0.0))

    det = StrikeDetector(config)
    feats = det.score_frames(feats)
    cands = det.detect_candidates(feats)
    assert len(cands) >= 1
    assert all(c.confidence >= 0.0 for c in cands)


def test_sparse_detector_proposes_only_the_start_of_sustained_motion(config):
    features = []
    for index in range(20):
        t = index * 0.5
        moving = 2.0 <= t <= 6.0
        activity = 0.75 if moving else 0.05
        features.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                motion_raw=activity,
                motion_score=activity,
                ball_residual_motion=activity,
                max_ball_normalized_speed=2.0 if moving else 0.0,
                moving_ball_count=1 if moving else 0,
            )
        )

    candidates = StrikeDetector(config).detect_sparse_candidates(features)

    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(2.0)


def test_occluded_cue_ball_uses_sustained_visual_onset_for_review(config):
    """A hidden white ball may infer a strike, but audio/one-frame noise may not."""

    feats = []
    for i in range(31):
        t = i * 0.1
        moving = 1.1 <= t <= 1.5
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=12.0,
                ball_count=8,
                cue_ball_detected=t < 1.0,
                cue_ball_x=100.0 if t < 1.0 else None,
                cue_ball_y=100.0 if t < 1.0 else None,
                cue_ball_track_confidence=0.9 if t < 1.0 else 0.0,
                motion_raw=0.45 if moving else 0.02,
                motion_score=0.45 if moving else 0.02,
                max_ball_normalized_speed=2.0 if moving else 0.0,
                ball_residual_motion=0.5 if moving else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    candidates = detector.detect_candidates(feats)
    assert candidates
    assert any(c.evidence.get("occlusion_inferred", 0.0) >= 0.5 for c in candidates)


def test_colour_respot_is_not_a_cue_strike(config):
    """A moving colour with a reliably fixed white ball is table handling."""

    feats = []
    for i in range(31):
        t = i * 0.1
        colour_moving = 1.0 <= t <= 1.4
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=12.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=100.0,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=0.0,
                cue_ball_track_confidence=0.9,
                motion_raw=0.45 if colour_moving else 0.02,
                motion_score=0.45 if colour_moving else 0.02,
                max_ball_normalized_speed=3.0 if colour_moving else 0.0,
                ball_residual_motion=0.5 if colour_moving else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    assert detector.detect_candidates(feats) == []


def test_strong_cue_launch_survives_isolated_object_track_jitter(config):
    """Noisy object tracks must not hide a confirmed white-ball launch."""

    feats = []
    for i in range(26):
        t = i * 0.1
        launched = t >= 1.0
        step = max(0, i - 10)
        # Four noisy object observations leave only one quiet sample and also
        # raise the median object speed. Strong cue contact plus a stationary
        # white ball must still win over these heuristic identity jumps.
        noisy_object = t in (0.5, 0.6, 0.7, 0.8)
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=100.0 + step * 4.0,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=1.5 if launched else 0.0,
                cue_ball_acceleration=8.0 if t == 1.0 else 0.0,
                cue_ball_track_confidence=0.9,
                cue_contact_score=0.85 if t == 1.0 else 0.0,
                motion_raw=0.35 if launched else 0.02,
                motion_score=0.35 if launched else 0.02,
                max_ball_normalized_speed=(8.0 if noisy_object else (1.5 if launched else 0.0)),
                ball_residual_motion=0.5 if launched else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    candidates = detector.detect_candidates(feats)
    assert any(abs(c.timestamp - 1.0) <= 0.1 for c in candidates)


def test_strong_cue_launch_tolerates_one_player_motion_sample(config):
    """Starting the stroke must not hide the subsequent verified cue hit."""

    feats = []
    cue_x = 100.0
    for i in range(26):
        t = round(i * 0.1, 1)
        launched = t >= 1.0
        if launched:
            cue_x += 15.0
        # At 10 fps, one player/cue foreground sample in the four-sample
        # pre-impact window produces a 0.75 quiet ratio.
        player_motion = t == 0.7
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=cue_x,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=5.0 if launched else 0.0,
                cue_ball_acceleration=14.0 if t == 1.0 else 0.0,
                cue_ball_track_confidence=0.9,
                cue_contact_score=0.85 if t == 1.0 else 0.0,
                cue_approach_speed=4.0 if t == 1.0 else 0.0,
                motion_raw=0.35 if launched or player_motion else 0.02,
                motion_score=0.35 if launched or player_motion else 0.02,
                max_ball_normalized_speed=5.0 if launched else 0.0,
                moving_ball_count=1 if launched else 0,
                ball_residual_motion=0.5 if launched else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    candidates = detector.detect_candidates(feats)
    assert any(abs(c.timestamp - 1.0) <= 0.1 for c in candidates)


def test_player_motion_exception_requires_strong_cue_contact(config):
    """Foreground motion plus ball movement is not enough during a respot."""

    feats = []
    for i in range(26):
        t = round(i * 0.1, 1)
        launched = t >= 1.0
        player_motion = t == 0.7
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=100.0,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=0.0,
                cue_ball_acceleration=0.0,
                cue_ball_track_confidence=0.9,
                cue_contact_score=0.0,
                cue_approach_speed=0.0,
                motion_raw=0.35 if launched or player_motion else 0.02,
                motion_score=0.35 if launched or player_motion else 0.02,
                max_ball_normalized_speed=5.0 if launched else 0.0,
                moving_ball_count=1 if launched else 0,
                ball_residual_motion=0.5 if launched else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    assert detector.detect_candidates(feats) == []


def test_angle_shot_bridges_one_impact_frame_tracker_hole(config):
    """Strong cue contact may bridge one hidden white-ball observation."""

    feats = []
    cue_speeds = {1.0: 1.4, 1.1: 0.0, 1.2: 6.0, 1.3: 5.0}
    cue_x = 100.0
    for i in range(26):
        t = round(i * 0.1, 1)
        speed = cue_speeds.get(t, 0.0)
        if speed > 0:
            cue_x += speed * 2.0
        noisy_object = t in (0.6, 0.7, 0.8)
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=cue_x,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=speed,
                cue_ball_acceleration=14.0 if t == 1.0 else 0.0,
                cue_ball_track_confidence=0.9,
                cue_contact_score=0.88 if t == 1.0 else 0.0,
                cue_approach_speed=4.0 if t == 1.0 else 0.0,
                motion_raw=0.35 if t >= 1.0 else 0.02,
                motion_score=0.35 if t >= 1.0 else 0.02,
                max_ball_normalized_speed=(
                    9.0 if noisy_object else (max(speed, 3.0) if t >= 1.0 else 0.0)
                ),
                ball_residual_motion=0.5 if t >= 1.0 else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    candidates = detector.detect_candidates(feats)
    assert any(abs(c.timestamp - 1.0) <= 0.1 for c in candidates)


def test_tracker_hole_without_cue_contact_is_not_bridged(config):
    """A gapped speed track alone must not turn ball handling into a strike."""

    feats = []
    cue_speeds = {1.0: 1.4, 1.1: 0.0, 1.2: 6.0, 1.3: 5.0}
    for i in range(26):
        t = round(i * 0.1, 1)
        speed = cue_speeds.get(t, 0.0)
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=100.0,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=speed,
                cue_ball_track_confidence=0.9,
                cue_contact_score=0.0,
                cue_approach_speed=0.0,
                motion_raw=0.35 if t >= 1.0 else 0.02,
                motion_score=0.35 if t >= 1.0 else 0.02,
                max_ball_normalized_speed=max(speed, 3.0) if t >= 1.0 else 0.0,
                ball_residual_motion=0.5 if t >= 1.0 else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    assert detector.detect_candidates(feats) == []


def test_tracker_identity_break_can_propose_occluded_real_strike(config):
    """Keep the fallback when impact swaps the white-ball track identity."""

    feats = []
    for i in range(31):
        t = i * 0.1
        moving = 1.0 <= t <= 1.4
        jumped = t >= 1.1
        feats.append(
            FrameFeatures(
                t=t,
                table_confidence=0.9,
                view_type=CameraViewType.MAIN_TABLE,
                ball_diameter_px=10.0,
                ball_count=8,
                cue_ball_detected=True,
                cue_ball_x=140.0 if jumped else 100.0,
                cue_ball_y=100.0,
                cue_ball_normalized_speed=0.2 if moving else 0.0,
                cue_ball_track_confidence=0.55 if jumped else 0.9,
                motion_raw=0.45 if moving else 0.02,
                motion_score=0.45 if moving else 0.02,
                max_ball_normalized_speed=3.0 if moving else 0.0,
                ball_residual_motion=0.5 if moving else 0.0,
            )
        )

    detector = StrikeDetector(config)
    detector.score_frames(feats)
    candidates = detector.detect_candidates(feats)
    assert any(c.evidence.get("occlusion_inferred", 0.0) >= 0.5 for c in candidates)


def test_segment_builder_modes(config):
    cands = [
        StrikeCandidate(timestamp=10.0, confidence=0.8, camera_view=CameraViewType.MAIN_TABLE),
        StrikeCandidate(timestamp=30.0, confidence=0.55, camera_view=CameraViewType.MAIN_TABLE),
    ]
    feats = [
        _feat(t * 0.5, motion=0.6 if 10 <= t * 0.5 <= 14 or 30 <= t * 0.5 <= 34 else 0.05)
        for t in range(100)
    ]
    builder = SegmentBuilder(config)
    for mode in EditMode:
        shots = builder.build(cands, feats, duration=50.0, mode=mode)
        assert len(shots) >= 1
        for s in shots:
            assert s.clip_end > s.clip_start
            assert s.cue_strike >= s.clip_start - 0.01
            # HARD RULE: never cut before balls stop
            assert s.clip_end + 1e-6 >= s.ball_motion_end


def test_strict_mode_two_second_pre_roll(config):
    cands = [
        StrikeCandidate(timestamp=10.0, confidence=0.85, camera_view=CameraViewType.MAIN_TABLE),
    ]
    # Quiet then strong motion 10–14s then true quiet (not cloth noise)
    feats = []
    for i in range(200):
        t = i * 0.1
        if 10.0 <= t <= 14.0:
            motion = 0.75
            raw = 0.8
            mean, mx, area = 1.5, 10.0, 0.12
        else:
            motion = 0.04
            raw = 0.05
            mean, mx, area = 0.02, 0.06, 0.005
        f = _feat(t, motion=motion)
        f.motion_raw = raw
        f.residual_motion_mean = mean
        f.residual_motion_max = mx
        f.motion_area_ratio = area
        feats.append(f)
    shots = SegmentBuilder(config).build(cands, feats, 25.0, EditMode.STRICT)
    assert len(shots) == 1
    s = shots[0]
    assert abs(s.clip_start - 8.0) < 0.05  # 2s before strike
    assert s.clip_end >= s.ball_motion_end - 1e-6
    # Strict mode keeps at least four seconds after cue contact.
    assert s.clip_end == pytest.approx(14.0)
    assert s.ball_motion_end == pytest.approx(14.0)
    assert s.evidence["stop_reason"] == "max_seconds_after_strike_review_cap"


def test_mid_motion_false_peak_absorbed(config):
    """Collision peaks during ball travel must not create a second clip mid-shot."""
    cands = [
        StrikeCandidate(timestamp=5.0, confidence=0.9),
        StrikeCandidate(timestamp=7.0, confidence=0.55),  # mid-roll false peak
    ]
    feats = []
    for i in range(150):
        t = i * 0.1
        if 5.0 <= t <= 12.0:
            motion, raw, mean, mx, area = 0.7, 0.75, 1.5, 10.0, 0.12
        else:
            motion, raw, mean, mx, area = 0.04, 0.04, 0.02, 0.05, 0.004
        f = _feat(t, motion=motion)
        f.motion_raw = raw
        f.residual_motion_mean = mean
        f.residual_motion_max = mx
        f.motion_area_ratio = area
        feats.append(f)
    shots = SegmentBuilder(config).build(cands, feats, 20.0, EditMode.STRICT)
    assert len(shots) == 1
    assert shots[0].clip_end >= shots[0].ball_motion_end - 1e-6
    assert shots[0].ball_motion_end == pytest.approx(9.0)
    # False peak must not create a second shot or stretch the capped clip.
    assert shots[0].clip_end <= shots[0].cue_strike + 4.0 + 1e-6


def test_false_peak_does_not_force_ten_second_cap(config):
    """Regression: absorbing mid-shot peaks used to force clip_end = strike+10."""
    cands = [
        StrikeCandidate(timestamp=3.0, confidence=0.85),
        StrikeCandidate(timestamp=8.0, confidence=0.5),  # false peak after true stop ~7
    ]
    feats = []
    for i in range(200):
        t = i * 0.1
        if 3.0 <= t <= 6.5:
            raw, mean, mx, area = 0.9, 2.0, 12.0, 0.15
        else:
            raw, mean, mx, area = 0.04, 0.02, 0.05, 0.004
        f = _feat(t, motion=raw)
        f.motion_raw = raw
        f.residual_motion_mean = mean
        f.residual_motion_max = mx
        f.motion_area_ratio = area
        feats.append(f)
    shots = SegmentBuilder(config).build(cands, feats, 25.0, EditMode.STRICT)
    assert len(shots) >= 1
    s0 = shots[0]
    # True stop around 7s, not forced to 13s
    assert s0.ball_motion_end < 10.0, f"end stretched to {s0.ball_motion_end}"
    assert s0.clip_end - s0.cue_strike < 8.0


def test_overlap_resolution(config):
    cands = [
        StrikeCandidate(timestamp=5.0, confidence=0.9),
        StrikeCandidate(timestamp=5.5, confidence=0.6),
    ]
    feats = [_feat(i * 0.2, motion=0.5 if i * 0.2 < 10 else 0.05) for i in range(50)]
    shots = SegmentBuilder(config).build(cands, feats, 20.0, EditMode.STRICT)
    for s in shots:
        assert s.duration() < 15.0
        assert s.clip_end >= s.ball_motion_end - 1e-6
        assert s.clip_end <= s.cue_strike + 10.0 + 1e-6


def test_strict_overlap_prefers_real_audio_supported_strike(config):
    """Preparation motion must lose to the real strike in the same shot window."""
    cands = [
        StrikeCandidate(
            timestamp=1.266667,
            confidence=1.0,
            evidence={
                "audio_onset": 0.02,
                "pre_ball_quiet_ratio": 0.93,
                "dense_transition_confirmed": 1.0,
            },
        ),
        StrikeCandidate(
            timestamp=4.766671,
            confidence=1.0,
            evidence={
                "audio_onset": 0.58,
                "pre_ball_quiet_ratio": 1.0,
                "dense_transition_confirmed": 1.0,
            },
        ),
    ]
    feats = []
    for i in range(100):
        t = i * 0.1
        moving = 1.3 <= t <= 2.0 or 4.8 <= t <= 6.2
        f = _feat(t, motion=0.7 if moving else 0.04)
        f.motion_raw = 0.7 if moving else 0.04
        f.residual_motion_mean = 1.5 if moving else 0.02
        f.residual_motion_max = 8.0 if moving else 0.05
        f.motion_area_ratio = 0.10 if moving else 0.004
        feats.append(f)

    shots = SegmentBuilder(config).build(cands, feats, 10.0, EditMode.STRICT)

    assert len(shots) == 1
    assert shots[0].cue_strike == pytest.approx(4.766671)
    assert shots[0].evidence["replaced_conflicting_strike"] == pytest.approx(
        1.266667
    )


def test_strict_overlap_burst_keeps_valid_shots_around_false_peak(config):
    """A low-support peak between valid shots cannot create overlapping records."""
    builder = SegmentBuilder(config)

    def record(
        shot_id: int,
        strike: float,
        confidence: float,
        stop: float,
        *,
        audio: float,
        quiet: float,
    ) -> ShotRecord:
        minimum_end = strike + 4.0
        return ShotRecord(
            shot_id=shot_id,
            cue_strike=strike,
            cue_strike_timestamp=strike,
            clip_start=max(0.0, strike - 2.0),
            clip_end=max(stop, minimum_end),
            physical_stop_timestamp=stop,
            ball_motion_end=stop,
            shot_confidence=confidence,
            evidence={
                "uncapped_physical_stop_timestamp": stop,
                "minimum_clip_end_timestamp": minimum_end,
                "audio_onset": audio,
                "pre_ball_quiet_ratio": quiet,
            },
        )

    shots = [
        record(38, 575.567246, 1.0, 579.567246, audio=0.30, quiet=1.0),
        record(39, 580.167251, 0.78, 582.333919, audio=0.0, quiet=0.0),
        record(40, 583.700587, 1.0, 587.700587, audio=0.63, quiet=1.0),
    ]

    resolved = builder._resolve_overlaps(shots, strict=True)

    assert [shot.cue_strike for shot in resolved] == pytest.approx(
        [575.567246, 583.700587]
    )
    assert resolved[1].clip_start >= resolved[0].clip_end
    assert [shot.shot_id for shot in resolved] == [1, 2]


def test_unconfirmed_stop_cap_does_not_hide_next_verified_strike(config):
    """An unresolved prior tracker cannot suppress a later non-overlapping shot."""
    builder = SegmentBuilder(config)
    previous = ShotRecord(
        shot_id=1,
        cue_strike=10.0,
        clip_start=8.0,
        clip_end=14.0,
        physical_stop_timestamp=14.0,
        ball_motion_end=14.0,
        shot_confidence=1.0,
        evidence={
            "uncapped_physical_stop_timestamp": 20.0,
            "minimum_clip_end_timestamp": 14.0,
            "stop_confirmed": False,
            "stop_reason": "max_seconds_after_strike_review_cap",
        },
    )
    following = ShotRecord(
        shot_id=2,
        cue_strike=19.5,
        clip_start=17.5,
        clip_end=23.5,
        physical_stop_timestamp=23.5,
        ball_motion_end=23.5,
        shot_confidence=1.0,
        evidence={
            "uncapped_physical_stop_timestamp": 29.5,
            "minimum_clip_end_timestamp": 23.5,
            "stop_confirmed": False,
        },
    )

    resolved = builder._resolve_overlaps([previous, following], strict=True)

    assert [shot.cue_strike for shot in resolved] == [10.0, 19.5]


def test_unresolved_long_roll_is_not_cut_by_timeout(config):
    """An unresolved rolling track is capped at ten seconds and reviewed."""
    cands = [
        StrikeCandidate(timestamp=5.0, confidence=0.9, camera_view=CameraViewType.MAIN_TABLE),
    ]
    # Motion stays high for 25s — must still cap at strike+10
    feats = []
    for i in range(400):
        t = i * 0.1
        motion = 0.7 if 5.0 <= t <= 30.0 else 0.04
        feats.append(_feat(t, motion=motion))
    shots = SegmentBuilder(config).build(cands, feats, 40.0, EditMode.STRICT)
    assert len(shots) == 1
    s = shots[0]
    assert s.clip_end == pytest.approx(9.0)
    assert s.evidence["stop_reason"] == "max_seconds_after_strike_review_cap"
    assert s.ball_motion_end == s.clip_end
    assert s.manual_review_required is True
    # Still starts ~2s before strike
    assert abs(s.clip_start - 3.0) < 0.05


# ---- NEW TESTS for improved detection ----


def test_rapid_consecutive_shots_both_detected(config):
    """Two real cue strikes ~3s apart (rapid break) — both must be detected."""
    feats = []
    for i in range(150):
        t = i * 0.1
        # Shot 1: onset at t=3.0, travel 3.0–4.5
        if 3.0 <= t <= 4.0:
            motion = 0.1 + (t - 3.0) * 0.8
        elif 4.0 < t <= 4.5:
            motion = 0.7
        # Quiet gap 4.5–5.5
        elif 4.5 < t < 5.5:
            motion = 0.06
        # Shot 2: onset at t=6.0, travel 6.0–7.5
        elif 6.0 <= t <= 7.0:
            motion = 0.1 + (t - 6.0) * 0.8
        elif 7.0 < t <= 7.5:
            motion = 0.65
        else:
            motion = 0.05
        feats.append(_feat(t, motion=min(motion, 1.0)))

    det = StrikeDetector(config)
    feats = det.score_frames(feats)
    cands = det.detect_candidates(feats)
    # With min_dist reduced to 1.2s, should find both shots
    assert len(cands) >= 2, f"Expected >= 2 candidates, got {len(cands)}"


def test_soft_safety_shot_detected(config):
    """A soft safety shot with low motion must not be missed."""
    feats = []
    for i in range(80):
        t = i * 0.1
        # Very quiet, then a soft onset
        if 3.0 <= t <= 4.0:
            motion = 0.15 + (t - 3.0) * 0.15  # gentle rise to 0.30
        elif 4.0 < t < 5.0:
            motion = 0.30  # slow roll
        elif 5.0 <= t < 5.5:
            motion = 0.15  # decaying
        else:
            motion = 0.04
        feats.append(_feat(t, motion=min(motion, 1.0), onset=0.3 if abs(t - 3.1) < 0.15 else 0.0))

    det = StrikeDetector(config)
    feats = det.score_frames(feats)
    cands = det.detect_candidates(feats)
    # Should detect the soft safety shot
    assert len(cands) >= 1, "Soft safety shot was missed"


def test_cushion_bounce_not_separate_shot(config):
    """A cushion bounce during ball travel must NOT create a false second shot."""
    cands = [
        StrikeCandidate(timestamp=5.0, confidence=0.9),
        StrikeCandidate(timestamp=7.5, confidence=0.45),  # cushion bounce
    ]
    feats = []
    for i in range(200):
        t = i * 0.1
        if 5.0 <= t <= 12.0:
            motion = 0.65
            # Spike at 7.5 (cushion bounce)
            if 7.3 <= t <= 7.8:
                motion = 0.85
        else:
            motion = 0.04
        f = _feat(t, motion=motion)
        f.motion_raw = motion
        f.residual_motion_mean = motion * 2
        f.residual_motion_max = motion * 5
        f.motion_area_ratio = motion * 0.02
        feats.append(f)
    shots = SegmentBuilder(config).build(cands, feats, 25.0, EditMode.STRICT)
    # Should be just 1 shot — the bounce is absorbed
    assert len(shots) == 1, f"Expected 1 shot, got {len(shots)}"


def test_long_slow_roll_not_cut_early(config):
    """A long slow roll (e.g., snooker behind color) must not end mid-roll."""
    feats = []
    for i in range(150):
        t = i * 0.1
        if 3.0 <= t <= 4.0:
            raw = 0.9  # initial fast travel
            mean, mx, area = 2.0, 12.0, 0.15
        elif 4.0 < t <= 9.0:
            # Gradually slowing but still clearly moving
            raw = max(0.25, 0.9 - (t - 4.0) * 0.13)
            mean = raw * 1.5
            mx = raw * 5.0
            area = raw * 0.04
        elif 9.0 < t <= 10.0:
            # Final slowdown
            raw = max(0.08, 0.25 - (t - 9.0) * 0.17)
            mean, mx, area = 0.03, 0.1, 0.008
        else:
            raw = 0.04
            mean, mx, area = 0.02, 0.05, 0.004
        f = _feat(t, motion=raw)
        f.motion_raw = raw
        f.residual_motion_mean = mean
        f.residual_motion_max = mx
        f.motion_area_ratio = area
        feats.append(f)

    from snooker_ai.event_fusion.ball_stop import BallStopDetector

    det = BallStopDetector(config)
    cand = StrikeCandidate(timestamp=3.0, confidence=0.85)
    _, m1, _, _ = det.find_motion_window(cand, feats, duration=20.0)
    # Must not cut before 8.0s (ball is still clearly rolling)
    assert m1 >= 8.0, f"Ball stop too early at {m1} — ball was still rolling"
    # But must end before hard cap
    assert m1 <= 13.0 + 1e-6
