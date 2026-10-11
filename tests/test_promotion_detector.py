"""A real countdown montage is excluded without rejecting its live match clock."""

from pathlib import Path

import cv2
import pytest

import snooker_ai.scene_detection.promotion as promotion
from snooker_ai.scene_detection.promotion import (
    PromotionDetector,
    _has_live_scorebar,
    _montage,
    title_card,
)
from snooker_ai.types import CameraViewType, FrameFeatures, StrikeCandidate


FIXTURES = Path(__file__).parent / "fixtures/classic_hard/promotion"


def image(name):
    frame = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    assert frame is not None
    return frame


class FixtureCapture:
    def __init__(self):
        self.requested = 0.0
        self.position = 0.0
        self.title_frames = [(608.0, image("608.00")), (608.5, image("608.50"))]
        self.timer_frames = [
            (614.2, image("614.20")),
            (616.2, image("616.20")),
            (618.2, image("618.20")),
        ]

    def isOpened(self):
        return True

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_MSEC
        self.requested = value / 1000
        self.position = self.requested
        return True

    def read(self):
        samples = self.title_frames if 607.5 <= self.requested <= 608.6 else self.timer_frames
        _, frame = min(samples, key=lambda pair: abs(pair[0] - self.requested))
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.position * 1000
        raise AssertionError(prop)

    def release(self):
        pass


def test_source_countdown_and_match_scorebar_are_distinct():
    card = image("608.00")
    assert title_card(card)
    assert title_card(image("608.50"))
    assert not title_card(image("614.20"))

    countdown = [
        (t, image(name)) for t, name in ((614.2, "614.20"), (616.2, "616.20"), (618.2, "618.20"))
    ]
    live_scorebar = image("590.20")
    assert _has_live_scorebar(live_scorebar)
    assert not _has_live_scorebar(countdown[1][1])
    assert _montage(countdown)
    assert not _montage(countdown + [(619.0, live_scorebar)])


def test_native_countdown_contact_is_filtered_only_after_title_and_timer_evidence(
    monkeypatch,
):
    monkeypatch.setattr(promotion, "open_capture", lambda _source: FixtureCapture())
    timestamp = 616.2
    features = [
        FrameFeatures(
            t=t,
            camera_scene_id=index,
            view_type=CameraViewType.MAIN_TABLE,
            view_classified=True,
            table_full_view=False,
        )
        for index, t in enumerate((timestamp - 2.5 + step * 0.5 for step in range(11)))
    ]
    candidate = StrikeCandidate(
        timestamp=timestamp,
        confidence=0.9,
        uncertainty_start=timestamp - 0.1,
        uncertainty_end=timestamp + 0.1,
    )
    assert PromotionDetector().detect("source.mp4", [candidate], features) == {timestamp}


def test_live_full_table_shot_never_enters_promo_filter(monkeypatch):
    monkeypatch.setattr(
        promotion,
        "open_capture",
        lambda _source: pytest.fail("normal full-table contacts must skip promo probing"),
    )
    candidate = StrikeCandidate(
        timestamp=590.2,
        confidence=0.9,
        uncertainty_start=590.0,
        uncertainty_end=590.4,
    )
    features = [
        FrameFeatures(t=590.0 + index * 0.1, camera_scene_id=index, table_full_view=True)
        for index in range(6)
    ]
    assert PromotionDetector().detect("source.mp4", [candidate], features) == set()
