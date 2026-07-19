import cv2

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.types import CameraViewType, FrameFeatures, SceneSegment


def test_main_table_view(config, synthetic_green_frame):
    clf = ViewClassifier(config)
    view, ratio, extra = clf.classify(synthetic_green_frame)
    assert ratio > 0.1
    assert view in (CameraViewType.MAIN_TABLE, CameraViewType.BALL_CLOSEUP, CameraViewType.WIDE_ARENA)


def test_saturated_tournament_banner_does_not_make_main_table_a_replay(
    config, synthetic_green_frame
):
    frame = synthetic_green_frame.copy()
    for x in range(0, frame.shape[1], 8):
        colour = (0, 0, 255) if (x // 8) % 2 else (255, 0, 0)
        cv2.rectangle(frame, (x, 0), (min(x + 7, frame.shape[1] - 1), 44), colour, -1)

    classifier = ViewClassifier(config)
    view, ratio, extra = classifier.classify(frame)

    assert ratio >= classifier.main_table_ratio
    assert extra["replay_graphic_score"] >= classifier.replay_score_thr
    assert extra["is_replay_candidate"] is False
    assert view not in (CameraViewType.REPLAY, CameraViewType.SLOW_MOTION_REPLAY)


def test_pathological_all_replay_checkpoint_is_repaired(config, tmp_job_dir):
    analyzer = Analyzer(config, tmp_job_dir)
    features = [
        FrameFeatures(t=float(index), green_ratio=0.34, view_type=CameraViewType.REPLAY)
        for index in range(20)
    ]
    scenes = [
        SceneSegment(
            start=0.0,
            end=20.0,
            view_type=CameraViewType.REPLAY,
            table_ratio=0.34,
            is_replay_candidate=True,
        )
    ]

    assert analyzer._repair_pathological_replay_labels(features, scenes) is True
    assert all(feature.view_type == CameraViewType.MAIN_TABLE for feature in features)
    assert scenes[0].view_type == CameraViewType.MAIN_TABLE
    assert scenes[0].is_replay_candidate is False
