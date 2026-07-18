"""Television camera-shot detection and view classification."""

from snooker_ai.scene_detection.detector import SceneDetector, detect_scenes
from snooker_ai.scene_detection.view_classifier import ViewClassifier

__all__ = ["SceneDetector", "detect_scenes", "ViewClassifier"]
