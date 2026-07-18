"""Unit tests for baseline-relative ball-stop detection."""

from __future__ import annotations

from snooker_ai.event_fusion.ball_stop import BallStopDetector
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


def _f(t: float, raw: float, mean: float = 0.0, mx: float = 0.0, area: float = 0.0) -> FrameFeatures:
    return FrameFeatures(
        t=t,
        motion_raw=raw,
        motion_score=raw,
        residual_motion_mean=mean if mean else raw * 1.5,
        residual_motion_max=mx if mx else raw * 4.0,
        motion_area_ratio=area if area else raw * 0.05,
        table_confidence=0.8,
        view_type=CameraViewType.MAIN_TABLE,
    )


def test_ball_stop_after_decay_not_noise_tail(config):
    """
    Real-match pattern: quiet baseline → strike → strong travel → decay to noise.
    Must end near decay, not stretch to 10s on cloth noise.
    """
    feats: list[FrameFeatures] = []
    # baseline quiet 0–2s
    for i in range(20):
        t = i * 0.1
        feats.append(_f(t, 0.05, mean=0.03, mx=0.08, area=0.008))
    strike_t = 2.5
    # rise and travel 2.5–7.0
    for i in range(25, 70):
        t = i * 0.1
        # peak around 5–6.5
        if 5.0 <= t <= 6.8:
            raw = 0.9
            feats.append(_f(t, raw, mean=2.0, mx=15.0, area=0.15))
        elif 2.5 <= t < 5.0:
            raw = 0.35 + (t - 2.5) * 0.15
            feats.append(_f(t, min(raw, 0.8), mean=0.5, mx=4.0, area=0.06))
        else:
            feats.append(_f(t, 0.4, mean=0.4, mx=2.0, area=0.04))
    # decay to noise 7.0–12
    for i in range(70, 120):
        t = i * 0.1
        if t < 8.5:
            raw = max(0.08, 0.4 - (t - 7.0) * 0.2)
        else:
            raw = 0.06  # cloth noise only
        feats.append(_f(t, raw, mean=0.04, mx=0.1, area=0.009))

    det = BallStopDetector(config)
    cand = StrikeCandidate(timestamp=strike_t, confidence=0.8)
    m0, m1, end_c, start_c = det.find_motion_window(cand, feats, duration=15.0)

    assert m1 <= strike_t + 10.0 + 1e-6
    # Should end shortly after decay (~7–9s), NOT at 12.5
    assert m1 < 10.0, f"end too late: {m1}"
    assert m1 >= 7.0, f"end too early (mid-shot): {m1}"
    assert end_c >= 0.5


def test_ball_stop_keeps_open_when_motion_unresolved(config):
    feats = []
    for i in range(200):
        t = i * 0.1
        # perpetual high motion
        feats.append(_f(t, 0.85, mean=2.0, mx=12.0, area=0.2))
    det = BallStopDetector(config)
    cand = StrikeCandidate(timestamp=1.0, confidence=0.9)
    result = det.detect_stop(cand, feats, duration=30.0)
    # A runaway unresolved track is capped at ten seconds and flagged for review.
    assert result.physical_stop_timestamp == 11.0
    assert result.confirmed is False
    assert result.manual_review_required is True


def test_ball_stop_short_pot(config):
    """Quick pot: motion only ~1.2s then still."""
    feats = []
    for i in range(80):
        t = i * 0.1
        if 3.0 <= t <= 4.2:
            feats.append(_f(t, 0.8, mean=1.5, mx=8.0, area=0.12))
        else:
            feats.append(_f(t, 0.05, mean=0.02, mx=0.05, area=0.005))
    det = BallStopDetector(config)
    cand = StrikeCandidate(timestamp=3.0, confidence=0.85)
    _, m1, _, _ = det.find_motion_window(cand, feats, duration=10.0)
    assert 4.0 <= m1 <= 5.8, f"unexpected end {m1}"
