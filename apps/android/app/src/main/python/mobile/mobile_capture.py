"""FFmpeg-backed OpenCV capture for the Android wheel (which has no video I/O).

Only one BGR frame and a bounded timestamp queue are retained. showinfo records
decoded presentation timestamps, including VFR input; they are never synthesized
from average FPS for the detector's native-time observations.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import math
import queue
import re
import subprocess
import threading

import cv2
import numpy as np

from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg

_PTS = re.compile(r"\bn:\s*\d+.*?\bpts:\s*(-?\d+)")
_TIME_BASE = re.compile(r"config in time_base:\s*(\d+/\d+)")


class FFmpegCapture:
    def __init__(self, filename, *args, **kwargs):
        self.path = str(filename)
        self.process = None
        self.reader = None
        self.frame = None
        self.position = 0
        self.timestamp = 0.0
        self.seek = 0.0
        self._discard_through = None
        self.opened = False
        try:
            data = ffprobe_json(self.path)
            stream = next(s for s in data["streams"] if s["codec_type"] == "video")
            self.width, self.height = int(stream["width"]), int(stream["height"])
            rotation = next((int(s.get("rotation", 0)) for s in stream.get("side_data_list", [])
                             if "rotation" in s), int(stream.get("tags", {}).get("rotate", 0)))
            if abs(rotation) % 180 == 90:
                self.width, self.height = self.height, self.width
            self.fps = float(Fraction(stream.get("avg_frame_rate") or "0/1"))
            if not self.fps:
                self.fps = float(Fraction(stream.get("r_frame_rate") or "0/1"))
            if not self.fps:
                raise ValueError("Video has no usable frame rate")
            self.duration = float(stream.get("duration") or data["format"]["duration"])
            self.count = int(stream.get("nb_frames") or math.ceil(self.duration * self.fps))
            self.opened = self.width > 0 and self.height > 0
        except (OSError, ValueError, ZeroDivisionError, KeyError, StopIteration, RuntimeError):
            self.opened = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: getattr(self, "width", 0),
            cv2.CAP_PROP_FRAME_HEIGHT: getattr(self, "height", 0),
            cv2.CAP_PROP_FPS: getattr(self, "fps", 0),
            cv2.CAP_PROP_FRAME_COUNT: getattr(self, "count", 0),
            cv2.CAP_PROP_POS_FRAMES: self.position,
            cv2.CAP_PROP_POS_MSEC: self.timestamp * 1000,
        }.get(prop, 0)

    def set(self, prop, value):
        if not self.opened or not math.isfinite(value) or value < 0:
            return False
        if prop not in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_POS_FRAMES):
            return False
        self._stop()
        self.seek = value / 1000 if prop == cv2.CAP_PROP_POS_MSEC else value / self.fps
        self.timestamp = self.seek
        self.position = round(self.seek * self.fps)
        self.frame = None
        self._discard_through = None
        return True

    def resume_after(self, timestamp, next_index):
        """Resume by original PTS, including when average FPS cannot locate a frame."""
        if not self.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
            return False
        self.position = next_index
        self._discard_through = timestamp
        return True

    def _start(self):
        self.timestamps = queue.Queue(maxsize=64)
        self.errors = deque(maxlen=30)
        self.stopping = threading.Event()
        command = [find_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "info",
                   "-threads", "2", "-copyts", "-start_at_zero", "-ss", str(self.seek), "-i", self.path,
                   # Desktop OpenCV's FFmpeg capture converts YUV using the
                   # BT.601 swscale matrix, even when an HD broadcast is tagged
                   # BT.709. Match those detector inputs rather than FFmpeg's
                   # metadata-aware default: colour changes alter white-ball
                   # identity and can suppress a real native contact.
                   "-map", "0:v:0", "-an", "-sn", "-vf",
                   "scale=in_color_matrix=bt601:out_color_matrix=bt601,showinfo",
                   "-fps_mode", "passthrough", "-threads", "1", "-pix_fmt", "bgr24",
                   "-f", "rawvideo", "pipe:1"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = self.process

        def collect():
            time_base = None
            try:
                for line in iter(process.stderr.readline, b""):
                    message = line.decode("utf-8", "replace")
                    base = _TIME_BASE.search(message)
                    if base:
                        time_base = Fraction(base[1])
                    match = _PTS.search(message)
                    if match and time_base is not None:
                        # pts_time is printed with only six significant digits.
                        # Its rounding changes dense flow/refresh decisions;
                        # use the integer PTS and exact stream time base instead.
                        timestamp = float(int(match[1]) * time_base)
                        while not self.stopping.is_set():
                            try:
                                self.timestamps.put(timestamp, timeout=.1)
                                break
                            except queue.Full:
                                pass
                    elif "error" in message.lower() or "failed" in message.lower():
                        self.errors.append(message)
            finally:
                while not self.stopping.is_set():
                    try:
                        self.timestamps.put(None, timeout=.1)
                        break
                    except queue.Full:
                        pass
        self.reader = threading.Thread(target=collect, name="android-video-pts", daemon=True)
        self.reader.start()

    def grab(self):
        while self._grab_decoded():
            if self._discard_through is None or self.timestamp > self._discard_through + 1e-9:
                self._discard_through = None
                return True
            self.position -= 1  # the saved frame is already in the detector state
        return False

    def _grab_decoded(self):
        if not self.opened:
            return False
        if self.process is None:
            self._start()
        size = self.width * self.height * 3
        raw = bytearray(size)
        view = memoryview(raw)
        offset = 0
        while offset < size:
            count = self.process.stdout.readinto(view[offset:])
            if not count:
                self.frame = None
                code = self.process.wait(timeout=15)
                if code:
                    raise RuntimeError("Android video decoder failed: " + "".join(self.errors)[-2000:])
                return False
            offset += count
        timestamp = self.timestamps.get(timeout=30)
        if timestamp is None:
            raise RuntimeError("Decoded frame has no presentation timestamp")
        self.frame = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)
        self.timestamp = timestamp
        self.position += 1
        return True

    def retrieve(self):
        return (self.frame is not None, self.frame)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def _stop(self):
        if self.process is None:
            return
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.process.stdout.close()
        self.reader.join(timeout=3)
        self.process.stderr.close()
        self.process = None
        self.reader = None

    def release(self):
        self._stop()
        self.opened = False
        self.frame = None

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass
