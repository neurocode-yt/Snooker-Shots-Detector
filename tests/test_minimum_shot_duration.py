import pytest

from snooker_ai.rendering.exporter import Exporter
from snooker_ai.rendering.mix import plan_mix
from snooker_ai.segmentation.builder import SegmentBuilder
from snooker_ai.types import StrikeCandidate


def candidate(t, *, confidence=0.95):
    return StrikeCandidate(timestamp=t, confidence=confidence, evidence={
        "refined_stop_timestamp": t + 1,
        "refined_stop_confirmation_timestamp": t + 1.5,
        "refined_stop_confidence": 0.95,
    })


@pytest.mark.parametrize("fps", [25, 30, 24000 / 1001, 30000 / 1001])
def test_short_shots_have_four_clear_seconds_plus_mix_and_follow_through(config, fps):
    shots = SegmentBuilder(config).build([candidate(t) for t in [10, 20, 30]], [], 40)
    plan = plan_mix(shots, fps, .24)
    assert all(overlap > 0 for overlap in plan.overlaps)
    for i, shot in enumerate(shots):
        incoming = plan.overlaps[i - 1] if i else 0
        outgoing = plan.overlaps[i] if i < len(plan.overlaps) else 0
        assert plan.durations[i] - incoming - outgoing >= 4 - 1e-6
        assert shot.clip_start + plan.durations[i] - outgoing - shot.cue_strike >= 2 - 1e-6
        assert shot.physical_stop_timestamp == shot.cue_strike + 1
    Exporter(config)._validate_strict_boundaries(shots, source_duration=40, source_fps=fps)


def test_cut_export_has_four_seconds_without_transition_padding(config):
    config._data["export"]["transition"] = "cut"
    shot = SegmentBuilder(config).build([candidate(10)], [], 20)[0]
    assert shot.clip_start == 8
    assert shot.clip_end == 12
    assert shot.duration() == 4


def test_source_edge_never_fabricates_footage_to_reach_minimum(config):
    shot = SegmentBuilder(config).build([candidate(9)], [], 10.5)[0]
    assert shot.clip_start == 7
    assert shot.clip_end == 10.5
    Exporter(config)._validate_strict_boundaries([shot], source_duration=10.5, source_fps=30)


def test_fast_shots_give_up_mix_handles_before_clear_time_or_candidates(config):
    # Medium-confidence strikes previously conflicted only because of the extra
    # transition padding, even though their four-second bodies do not overlap.
    shots = SegmentBuilder(config).build(
        [candidate(t, confidence=.6) for t in [10, 14, 18]], [], 25,
    )
    assert len(shots) == 3
    for left, right in zip(shots, shots[1:]):
        assert left.clip_end <= right.clip_start
    plan = plan_mix(shots, 30, .24)
    for i, duration in enumerate(plan.durations):
        head = plan.overlaps[i - 1] if i else 0
        tail = plan.overlaps[i] if i < len(plan.overlaps) else 0
        assert duration - head - tail >= 4 - 1e-6
    Exporter(config)._validate_strict_boundaries(shots, source_duration=25, source_fps=30)


def test_export_rejects_accidental_undersized_automatic_clip(config):
    shot = SegmentBuilder(config).build([candidate(10)], [], 20)[0]
    shot.clip_end = shot.clip_end_timestamp = 10.1
    with pytest.raises(ValueError, match="minimum viewing time"):
        Exporter(config)._validate_strict_boundaries([shot], source_duration=20, source_fps=30)


def test_transition_context_changes_clips_without_invalidating_detection_cache(config, tmp_path):
    from snooker_ai.pipeline.analyzer import Analyzer

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fingerprint")
    analyzer = Analyzer(config, tmp_path / "job")
    features_signature = analyzer._analysis_signature(source)
    result_signature = analyzer._result_signature(source)
    config._data["export"]["transition_seconds"] = .4
    assert analyzer._analysis_signature(source) == features_signature
    assert analyzer._result_signature(source) != result_signature
