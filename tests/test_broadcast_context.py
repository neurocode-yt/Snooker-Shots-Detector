import cv2
import numpy as np
import pytest

from snooker_ai.config import Config
from snooker_ai.scene_detection.broadcast_context import BroadcastContextGuard
from snooker_ai.types import CameraViewType


def score_frame(left="Selby", right="Lisowski", score="0", shift=0):
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[80:299, 100:540] = (40, 140, 40)
    y = 306 + shift
    cv2.rectangle(frame, (128, y), (512, y + 18), (35, 35, 35), -1)
    for x in (265, 348):
        cv2.rectangle(frame, (x, y), (x + 27, y + 17), (0, 255, 255), -1)
        cv2.putText(frame, score, (x + 7, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(frame, left, (155, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, right, (405, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def learned_guard():
    guard = BroadcastContextGuard(Config({}))
    for i in range(8):
        guard.observe(score_frame(), i * 0.5, CameraViewType.MAIN_TABLE)
    assert guard.target_ready
    return guard


def foreign_score_frame():
    # The source's other-match cutaway uses a smaller, differently aligned
    # overlay.  This fixture intentionally supplies strong appearance evidence.
    frame = score_frame("", "")
    cv2.putText(frame, "Stan Moody", (190, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(frame, "Liam Highfield", (404, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return frame


def test_changing_score_and_overlay_vertical_position_preserve_identity():
    guard = learned_guard()
    obs = guard.observe(score_frame(score="8", shift=8), 4.0, CameraViewType.MAIN_TABLE)
    assert obs.scoreboard_visible and obs.identity_known
    assert obs.identity_similarity > 0.95
    assert not obs.foreign_match


def test_foreign_scoreboard_requires_sustained_evidence_and_reports_beginning():
    guard = learned_guard()
    foreign = foreign_score_frame()
    first = guard.observe(foreign, 4.0, CameraViewType.MAIN_TABLE)
    second = guard.observe(foreign, 4.5, CameraViewType.MAIN_TABLE)
    third = guard.observe(foreign, 5.0, CameraViewType.MAIN_TABLE)
    assert first.scoreboard_visible and first.identity_known
    assert not first.foreign_match and not second.foreign_match
    assert third.foreign_match
    assert third.foreign_interval_start == 4.0
    returned = guard.observe(score_frame(score="9"), 5.5, CameraViewType.MAIN_TABLE)
    assert not returned.foreign_match and returned.identity_similarity > 0.95


def test_missing_overlay_and_other_angles_are_unknown_not_foreign():
    guard = learned_guard()
    foreign = foreign_score_frame()
    for i in range(3):
        guard.observe(foreign, 4.0 + i * 0.5, CameraViewType.MAIN_TABLE)
    absent = guard.observe(np.zeros((360, 640, 3), np.uint8), 5.5, CameraViewType.MAIN_TABLE)
    angle = guard.observe(foreign, 6.0, CameraViewType.BALL_CLOSEUP)
    assert absent.target_ready and angle.target_ready
    assert not absent.scoreboard_visible and not absent.foreign_match
    assert not angle.identity_known and not angle.foreign_match


def test_seeking_keeps_learned_target_and_does_not_learn_foreign_window():
    guard = learned_guard()
    guard.reset_observations(preserve_target=True)
    foreign = foreign_score_frame()
    observations = [guard.observe(foreign, t) for t in (0.0, 0.5, 1.0)]
    assert observations[-1].foreign_match
    assert observations[-1].foreign_interval_start == 0.0
    guard.reset_observations(preserve_target=False)
    assert not guard.target_ready


def test_mismatch_after_large_gap_starts_new_confirmation():
    guard = learned_guard()
    foreign = foreign_score_frame()
    guard.observe(foreign, 4.0)
    guard.observe(foreign, 4.5)
    obs = guard.observe(foreign, 10.0)
    assert not obs.foreign_match
    with pytest.raises(ValueError, match="chronological"):
        guard.observe(score_frame(), 9.0)


def test_gloves_and_signage_without_paired_scoreboxes_are_not_scoreboards():
    frame = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(frame, "UNIBET", (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.circle(frame, (400, 300), 24, (255, 255, 255), -1)
    assert BroadcastContextGuard.scoreboard_signature(frame) is None


def test_similar_unfamiliar_names_are_unknown_instead_of_forced_rejection():
    guard = learned_guard()
    observations = [guard.observe(score_frame("Stan Moody", "Highfield"), t) for t in (4, 4.5, 5)]
    assert observations[-1].scoreboard_visible
    assert not observations[-1].identity_known
    assert not observations[-1].foreign_match


def test_target_roundtrip_restores_identity_for_cached_analysis():
    learned = learned_guard()
    restored = BroadcastContextGuard(Config({}))
    assert restored.restore_target(learned.export_target())
    assert restored.target_ready
    matched = restored.observe(score_frame(score="7"), 0)
    assert matched.identity_known and matched.identity_similarity > 0.95
    observations = [restored.observe(foreign_score_frame(), t) for t in (1.0, 1.5, 2.0)]
    assert observations[-1].foreign_match


@pytest.mark.parametrize("values", [[], [1.0], [0.0] * 3072, [float("nan")] * 3072, [float("inf")] * 3072, [-1.0] * 3072])
def test_corrupt_target_caches_are_rejected_without_replacing_learned_identity(values):
    guard = learned_guard()
    expected = guard.export_target()
    assert not guard.restore_target(values)
    assert guard.export_target() == expected
