from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import CameraViewType


def test_main_table_view(config, synthetic_green_frame):
    clf = ViewClassifier(config)
    view, ratio, extra = clf.classify(synthetic_green_frame)
    assert ratio > 0.1
    assert view in (CameraViewType.MAIN_TABLE, CameraViewType.BALL_CLOSEUP, CameraViewType.WIDE_ARENA)
