"""Commentary cannot change shot selection; regressions from the supplied match."""
import json
from pathlib import Path

import pytest

from snooker_ai.event_fusion.strike import StrikeDetector
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import FrameFeatures, ShotRecord, StrikeCandidate
from snooker_ai.segmentation.builder import SegmentBuilder


@pytest.fixture
def windows():
    return json.loads((Path(__file__).parent / 'fixtures/visual_strike_windows.json').read_text())


@pytest.mark.parametrize('name', ['walking', 'respot', 'preparation'])
@pytest.mark.parametrize('audio', [0.0, 1.0])
def test_nonshots_rejected_with_silence_or_loud_commentary(config, windows, name, audio):
    detector = StrikeDetector(config)
    features = [FrameFeatures.model_validate(f) for f in windows[name]['dense']]
    for f in features:
        f.audio_onset = f.audio_highband = f.audio_rms = audio
    detector.score_frames(features)
    assert detector.detect_candidates(features) == []
    seed = StrikeCandidate(timestamp=windows[name]['timestamp'], confidence=.9,
                           evidence={'audio_seed': 1, 'audio_visual_support': 1})
    detector.refine_boundaries([seed], features)
    assert not seed.evidence.get('dense_transition_confirmed')
    assert not seed.evidence.get('sparse_dense_transition')


@pytest.mark.parametrize('name', ['launch', 'jitter_launch', 'occluded_launch', 'fast_launch'])
def test_visual_scan_and_confirmation_recover_real_strikes_without_sound(config, windows, name):
    detector = StrikeDetector(config)
    coarse = [FrameFeatures.model_validate(f) for f in windows[name]['coarse']]
    dense = [FrameFeatures.model_validate(f) for f in windows[name]['dense']]
    for f in coarse + dense:
        f.audio_onset = f.audio_highband = f.audio_rms = 0
    t = windows[name]['timestamp']
    proposals = detector.detect_sparse_candidates(detector.score_frames(coarse))
    assert any(abs(c.timestamp-t) < .75 for c in proposals)
    confirmed = detector.detect_candidates(detector.score_frames(dense))
    assert any(abs(c.timestamp-t) < .4 for c in confirmed)


@pytest.mark.parametrize('name', ['launch', 'jitter_launch', 'occluded_launch', 'walking', 'respot'])
def test_scores_and_proposals_are_identical_with_changed_audio(config, tmp_path, windows, name):
    analyzer = Analyzer(config, tmp_path)
    original = [FrameFeatures.model_validate(f) for f in windows[name]['coarse']]
    changed = [f.model_copy(deep=True) for f in original]
    for i, f in enumerate(changed):
        f.audio_onset = (i % 3) / 2
        f.audio_highband = f.audio_rms = 1
        f.strike_score = 1  # stale score from an older checkpoint
    left, right = analyzer._visual_proposals(original), analyzer._visual_proposals(changed)
    assert [c.model_dump() for c in left] == [c.model_dump() for c in right]
    assert [f.strike_score for f in original] == [f.strike_score for f in changed]


def test_overlap_choice_ignores_commentary(config):
    builder = SegmentBuilder(config)
    earlier = ShotRecord(shot_id=1, cue_strike=2, clip_start=0, clip_end=6,
                         shot_confidence=.8, evidence={'pre_ball_quiet_ratio': .9, 'audio_onset': 0})
    later = ShotRecord(shot_id=2, cue_strike=4, clip_start=2, clip_end=8,
                       shot_confidence=.8, evidence={'pre_ball_quiet_ratio': .8, 'audio_onset': 1})
    assert not builder._prefer_later_conflicting_shot(earlier, later)



@pytest.mark.parametrize('name', ['launch', 'jitter_launch', 'occluded_launch', 'respot'])
def test_dense_confirmation_is_audio_invariant(config, windows, name):
    detector = StrikeDetector(config)
    quiet = [FrameFeatures.model_validate(f) for f in windows[name]['dense']]
    noisy = [f.model_copy(deep=True) for f in quiet]
    for f in quiet:
        f.audio_onset = f.audio_highband = f.audio_rms = 0
    for f in noisy:
        f.audio_onset = f.audio_highband = f.audio_rms = 1
    left = detector.detect_candidates(detector.score_frames(quiet))
    right = detector.detect_candidates(detector.score_frames(noisy))
    assert [c.model_dump() for c in left] == [c.model_dump() for c in right]


def test_feature_extraction_does_not_read_audio(config, synthetic_video, tmp_path, monkeypatch):
    from snooker_ai.audio.features import AudioFeatureExtractor
    from snooker_ai.utils.timebase import TimeMapper

    def forbidden(*args, **kwargs):
        pytest.fail('Shot analysis must not read commentary or cue audio')

    monkeypatch.setattr(AudioFeatureExtractor, 'extract', forbidden)
    analyzer = Analyzer(config, tmp_path / 'job')
    features, _, _ = analyzer._extract_features(
        synthetic_video, tmp_path / 'nonexistent-commentary.wav',
        TimeMapper(source_duration=7), 7, sample_fps=10, end_time=3,
    )
    assert features
    assert all(f.audio_onset == f.audio_highband == f.audio_rms == 0 for f in features)


def test_spatial_stillness_tolerates_jitter_but_not_sustained_drift(config):
    detector = StrikeDetector(config)
    jitter = [FrameFeatures(t=i/30, cue_ball_x=100+(.3 if i%2 else -.3),
                            cue_ball_y=100, cue_ball_normalized_speed=2,
                            ball_diameter_px=10) for i in range(15)]
    drift = [f.model_copy(update={'cue_ball_x': 100+2*i}) for i, f in enumerate(jitter)]
    assert detector._stationary_ratio(jitter) == 1
    assert detector._stationary_ratio(drift) == 0
