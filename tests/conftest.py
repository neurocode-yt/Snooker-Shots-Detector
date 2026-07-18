"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snooker_ai.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def tmp_job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "job"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_green_frame():
    """BGR frame with a green table rectangle."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Green cloth (HSV-friendly BGR approx)
    frame[60:300, 80:560] = (40, 140, 40)
    # White cue-ball-ish blob
    cv_center = (200, 180)
    import cv2

    cv2.circle(frame, cv_center, 6, (230, 230, 230), -1)
    # Red object ball
    cv2.circle(frame, (320, 200), 6, (30, 30, 200), -1)
    return frame


@pytest.fixture
def synthetic_video(tmp_path: Path):
    """
    Create a short synthetic 'snooker-like' video:
    stationary green table → motion burst → stationary.
    Two such cycles simulate two shots.
    """
    import cv2

    path = tmp_path / "synthetic.mp4"
    w, h, fps = 640, 360, 15
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened()

    def base_frame(offset: int = 0):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[50:310, 60:580] = (45, 150, 45)
        # scoreboard strip
        frame[0:30, :] = (30, 30, 30)
        cv2.circle(frame, (180 + offset, 180), 7, (240, 240, 240), -1)
        cv2.circle(frame, (300, 200), 7, (20, 20, 200), -1)
        return frame

    # Shot 1: idle 1s, motion 1s, idle 1s
    for i in range(fps):
        writer.write(base_frame(0))
    for i in range(fps):
        writer.write(base_frame(int(i * 4)))
    for i in range(fps):
        writer.write(base_frame(40))

    # Pause 1s
    for i in range(fps):
        writer.write(base_frame(40))

    # Shot 2
    for i in range(fps):
        writer.write(base_frame(40))
    for i in range(fps):
        writer.write(base_frame(40 + int(i * 3)))
    for i in range(fps):
        writer.write(base_frame(70))

    writer.release()
    return path
