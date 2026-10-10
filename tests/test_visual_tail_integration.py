"""Referee footage limits the edit without changing measured ball-stop evidence."""

from types import SimpleNamespace

import pytest

from snooker_ai.event_fusion.ball_stop import StopDetection
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import AnalysisResult, EditMode, StrikeCandidate, VideoMetadata


def setup_shot(config, tmp_path, monkeypatch):
    analyzer = Analyzer(config, tmp_path)
    stop = StopDetection(
        motion_start=3., last_ball_motion_timestamp=7.9,
        physical_stop_timestamp=8., stop_confirmation_timestamp=8.6,
        end_confidence=.55, start_confidence=.9, confirmed=True,
        manual_review_required=True, reason="confirmed_after_unknown_gap",
    )
    monkeypatch.setattr(analyzer.segmenter.ball_stop, "detect_stop", lambda *a, **k: stop)
    candidate = StrikeCandidate(timestamp=3., confidence=.9,
                                evidence={"dense_transition_confirmed": 1.})
    shots = analyzer.segmenter.build([candidate], [], 12.)
    entry = SimpleNamespace(strike_timestamp=3., entry_timestamp=5.6,
                            confirmation_timestamp=5.8, source_entry_pts=5.6, confidence=.85)
    return analyzer, candidate, shots, entry


def test_verified_hand_entry_caps_export_and_survives_resegment(config, tmp_path, monkeypatch):
    analyzer, candidate, shots, entry = setup_shot(config, tmp_path, monkeypatch)
    assert analyzer._record_visual_tail_boundaries([candidate], shots, [entry], 12., 25.)
    revised = analyzer.segmenter.build([candidate], [], 12.)
    assert revised[0].clip_end == pytest.approx(5.56)
    assert revised[0].physical_stop_timestamp == 8.
    assert revised[0].stop_confirmation_timestamp == 8.6
    assert revised[0].evidence["end_before_ball_stop_seconds"] == 2.
    assert revised[0].evidence["usable_source_end_reason"] == "visual_hand_entry_clip_boundary"
    assert revised[0].duration() - .48 >= 4.
    assert revised[0].clip_end - revised[0].cue_strike - .24 >= 2.
    Exporter(config)._validate_strict_boundaries(revised, source_duration=12., source_fps=25.)
    result = AnalysisResult(job_id="test", source_path="source.mp4", original_duration=12.,
                            metadata=VideoMetadata(path="source.mp4", fps=25., duration=12.),
                            strike_candidates=[candidate], shots=revised)
    rebuilt = analyzer._rebuild_segments(result, EditMode.STRICT)
    assert rebuilt.shots[0].clip_end == pytest.approx(5.56)
    assert next(e for e in rebuilt.events if e.event_type == "ball_stop").timestamp == 8.


@pytest.mark.parametrize("case", ["early", "low_confidence", "different_shot", "nan",
                                  "unconfirmed_entry", "replay", "excluded", "user_modified"])
def test_unproven_or_too_early_entry_cannot_shorten_a_shot(config, tmp_path, monkeypatch, case):
    analyzer, candidate, shots, entry = setup_shot(config, tmp_path, monkeypatch)
    if case == "early":
        entry.entry_timestamp = entry.source_entry_pts = 4.9
    elif case == "low_confidence":
        entry.confidence = .5
    elif case == "different_shot":
        entry.strike_timestamp = 4.
    elif case == "nan":
        entry.entry_timestamp = float("nan")
    elif case == "unconfirmed_entry":
        entry.confirmation_timestamp = entry.entry_timestamp
    elif case == "replay":
        shots[0].possible_replay = True
    elif case == "user_modified":
        shots[0].user_modified = True
    else:
        shots[0].included = False
    assert not analyzer._record_visual_tail_boundaries([candidate], shots, [entry], 12., 25.)
    assert "visual_hand_entry_clip_cap_timestamp" not in candidate.evidence


def test_confident_stop_does_not_keep_referee_entry(config, tmp_path, monkeypatch):
    analyzer, candidate, shots, entry = setup_shot(config, tmp_path, monkeypatch)
    shots[0].end_confidence = .95
    assert shots[0].evidence["stop_confirmed"]
    assert analyzer._record_visual_tail_boundaries([candidate], shots, [entry], 12., 25.)
    revised = analyzer.segmenter.build([candidate], [], 12.)
    assert revised[0].clip_end == pytest.approx(5.56)
    assert revised[0].physical_stop_timestamp == 8.


def test_revalidated_native_stop_keeps_its_measured_boundary_visible(config,tmp_path,monkeypatch):
    analyzer,candidate,shots,entry=setup_shot(config,tmp_path,monkeypatch)
    config._data['modes']['strict']['end_before_ball_stop_seconds']=2.
    baseline=analyzer.segmenter.build([candidate],[],12.)
    assert baseline[0].clip_end < baseline[0].physical_stop_timestamp
    candidate.evidence['native_table_coverage_stop_revalidated']=1.
    revised=analyzer.segmenter.build([candidate],[],12.)
    assert revised[0].clip_end==revised[0].physical_stop_timestamp==8.


def test_verified_entry_reduces_dissolve_padding_instead_of_showing_referee(config, tmp_path, monkeypatch):
    analyzer, candidate, shots, entry = setup_shot(config, tmp_path, monkeypatch)
    entry.entry_timestamp = entry.source_entry_pts = 5.04
    entry.confirmation_timestamp = 5.24
    assert analyzer._record_visual_tail_boundaries([candidate], shots, [entry], 12., 25.)
    revised = analyzer.segmenter.build([candidate], [], 12.)
    shot = revised[0]
    assert shot.clip_end == pytest.approx(5.)
    assert shot.duration() == pytest.approx(4.)
    assert shot.clip_end-shot.cue_strike == pytest.approx(2.)
    Exporter(config)._validate_strict_boundaries(revised, source_duration=12., source_fps=25.)
    from snooker_ai.rendering.mix import plan_mix
    neighbours = [shot.model_copy(update={'clip_start': 0., 'clip_end': 6., 'cue_strike': 2.}),
                  shot,
                  shot.model_copy(update={'clip_start': 8., 'clip_end': 14., 'cue_strike': 10.})]
    plan = plan_mix(neighbours, 25., .24)
    assert plan.overlaps == [0., 0.]


def test_stop_beyond_verified_obstruction_cannot_erase_the_next_live_shot(config,tmp_path,monkeypatch):
    analyzer,candidate,shots,entry=setup_shot(config,tmp_path,monkeypatch)
    candidate.evidence['pre_ball_quiet_ratio']=.9
    candidate.evidence['uncapped_physical_stop_timestamp']=18.
    assert analyzer._record_visual_tail_boundaries([candidate],shots,[entry],20.,25.)
    later=StrikeCandidate(timestamp=12.,confidence=.95,
                          evidence={'dense_transition_confirmed':1.,'pre_ball_quiet_ratio':1.})
    def stop(c,*a,**k):
        return StopDetection(motion_start=c.timestamp,last_ball_motion_timestamp=17.9,
                             physical_stop_timestamp=18.,stop_confirmation_timestamp=18.6,
                             end_confidence=.8,start_confidence=.9,confirmed=True,
                             reason='confirmed_stationary_after_unseen_interval_upper_bound')
    monkeypatch.setattr(analyzer.segmenter.ball_stop,'detect_stop',stop)
    revised=analyzer.segmenter.build([candidate,later],[],20.)
    assert [s.cue_strike for s in revised if s.included]==[3.,12.]
    assert revised[0].clip_end==pytest.approx(5.56)


def test_visual_tail_setting_only_invalidates_final_results(config, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    analyzer = Analyzer(config, tmp_path / "job")
    features, result = analyzer._analysis_signature(source), analyzer._result_signature(source)
    config._data["visual_tail"] = {"enabled": False}
    assert analyzer._analysis_signature(source) == features
    assert analyzer._result_signature(source) != result
