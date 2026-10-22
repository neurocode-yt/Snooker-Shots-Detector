"""Exercise the Android decoder adapter using the host's real FFmpeg."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command

MODULE = Path(__file__).resolve().parents[1] / "apps/android/app/src/main/python/mobile/mobile_capture.py"
spec = importlib.util.spec_from_file_location("mobile_capture", MODULE)
mobile_capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mobile_capture)


@pytest.fixture
def video(tmp_path):
    output = tmp_path / "video with spaces.mp4"
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)])
    return output


def test_android_decoder_reads_seeks_and_releases(video):
    capture = mobile_capture.FFmpegCapture(video)
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == 10
    timestamps = []
    while capture.grab():
        ok, frame = capture.retrieve()
        assert ok and frame.shape == (96, 128, 3)
        timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    assert timestamps == pytest.approx([i / 10 for i in range(20)])
    assert capture.set(cv2.CAP_PROP_POS_MSEC, 1200)
    assert capture.read()[0]
    assert capture.get(cv2.CAP_PROP_POS_MSEC) == pytest.approx(1200)
    capture.release()
    assert not capture.isOpened()
    assert capture.process is None


def test_android_decoder_uses_vfr_presentation_times(tmp_path):
    output = tmp_path / "variable.mp4"
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=2",
                 "-vf", "select='not(mod(n,3))'", "-fps_mode", "vfr", "-c:v", "libx264", str(output)])
    capture = mobile_capture.FFmpegCapture(output)
    times = []
    try:
        while capture.grab():
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        capture.release()
    assert times == pytest.approx([0, .3, .6, .9, 1.2, 1.5, 1.8])


def test_android_decoder_missing_source(tmp_path):
    capture = mobile_capture.FFmpegCapture(tmp_path / "missing.mp4")
    assert not capture.isOpened()
    assert capture.read() == (False, None)


def test_android_decoder_matches_desktop_broadcast_colours(tmp_path):
    """HD colour tags must not change the detector's BGR input on Android."""
    output = tmp_path / "broadcast709.mp4"
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=1",
                 "-c:v", "libx264", "-colorspace", "bt709", "-color_primaries", "bt709",
                 "-color_trc", "bt709", str(output)])
    desktop = cv2.VideoCapture(str(output))
    android = mobile_capture.FFmpegCapture(output)
    try:
        while True:
            ok, expected = desktop.read()
            actual_ok, actual = android.read()
            assert actual_ok == ok
            if not ok:
                break
            np.testing.assert_array_equal(actual, expected)
    finally:
        desktop.release()
        android.release()


def test_android_decoder_keeps_full_precision_native_timestamps(tmp_path):
    output = tmp_path / "native30.mp4"
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30:duration=2",
                 "-c:v", "libx264", str(output)])
    capture = mobile_capture.FFmpegCapture(output)
    try:
        for index in range(60):
            assert capture.grab()
            assert abs(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - index / 30) < 1e-12
    finally:
        capture.release()


def test_android_decoder_normalizes_nonzero_source_start(tmp_path):
    output = tmp_path / "offset.mp4"
    run_command([find_ffmpeg(), "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=2",
                 "-c:v", "libx264", "-output_ts_offset", "100", str(output)])
    capture = mobile_capture.FFmpegCapture(output)
    try:
        assert capture.read()[0]
        assert capture.get(cv2.CAP_PROP_POS_MSEC) == pytest.approx(0)
        capture.set(cv2.CAP_PROP_POS_MSEC, 1200)
        assert capture.read()[0]
        assert capture.get(cv2.CAP_PROP_POS_MSEC) == pytest.approx(1200)
    finally:
        capture.release()
