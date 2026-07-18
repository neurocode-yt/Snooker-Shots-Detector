import numpy as np

from snooker_ai.motion.residual import ResidualMotionAnalyzer
from snooker_ai.table_detection.localizer import TableLocalizer


def test_table_localizer_finds_green(config, synthetic_green_frame):
    loc = TableLocalizer(config)
    obs = loc.detect(synthetic_green_frame)
    assert obs.mask is not None
    assert obs.area_ratio > 0.1
    assert obs.confidence > 0.2


def test_residual_motion_detects_change(config, synthetic_green_frame):
    import cv2

    analyzer = ResidualMotionAnalyzer(config)
    prev = cv2.cvtColor(synthetic_green_frame, cv2.COLOR_BGR2GRAY)
    moved = synthetic_green_frame.copy()
    # shift white ball area
    moved = np.roll(moved, 5, axis=1)
    gray = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
    loc = TableLocalizer(config)
    mask = loc.detect(synthetic_green_frame).mask
    sample = analyzer.analyze(prev, gray, mask)
    assert sample.residual_mean >= 0.0
    assert 0.0 <= sample.motion_score <= 1.0
