import math

import pytest

from snooker_ai.evaluation.parity import compare_shots


def shot(t):
    return {"cue_strike": t, "clip_start": max(0, t - 2), "clip_end": t + 5}


def test_equal_counts_cannot_hide_a_missed_and_an_extra_stroke():
    report = compare_shots([shot(10), shot(20)], [shot(10), shot(30)])
    assert not report["passed"]
    assert report["matched_count"] == 1
    assert report["missing"] == [shot(20)]
    assert report["extra"] == [shot(30)]


def test_matching_maximises_coverage_before_minimising_timing_error():
    report = compare_shots([shot(1), shot(2)], [shot(1.6), shot(2.8)])
    assert report["matched_count"] == 2
    assert not report["missing"] and not report["extra"]
    assert not report["passed"]


def test_same_contact_with_different_clip_end_is_a_failure():
    candidate = shot(10)
    candidate["clip_end"] += 1 / 30
    report = compare_shots([shot(10)], [candidate])
    assert report["matched_count"] == 1 and not report["passed"]
    assert report["maximum_error_seconds"]["clip_end"] == pytest.approx(1 / 30)


def test_exact_and_submillisecond_matches_pass():
    assert compare_shots([shot(10)], [shot(10)])["passed"]
    assert compare_shots([shot(10)], [shot(10.0005)])["passed"]
    assert compare_shots([], [])["passed"]
    assert not compare_shots([shot(10)], [])["passed"]
    assert not compare_shots([], [shot(10)])["passed"]


def test_nonfinite_boundaries_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        compare_shots([shot(math.nan)], [])
