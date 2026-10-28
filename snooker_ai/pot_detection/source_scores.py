"""Reusable full-resolution score observations across the complete source."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2

from snooker_ai.pot_detection.scoreboard import ScoreboardReader, ScoreReading
from snooker_ai.scene_detection.view_classifier import ViewClassifier
from snooker_ai.utils.persistence import atomic_text
from snooker_ai.utils.video import open_capture, sampled_frames
from snooker_ai.utils.work_control import check_pause


def score_signature(source, duration, config):
    source = Path(source)
    stat = source.stat()
    return hashlib.sha256(
        json.dumps(
            [
                2,
                str(source.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                duration,
                config.get("pot_selection.sample_fps", 2),
                config.get("pot_selection.min_digit_confidence", 0.78),
                hashlib.sha256(
                    (Path(__file__).with_name("assets") / "score_digits.npz").read_bytes()
                ).hexdigest(),
            ]
        ).encode()
    ).hexdigest()


def _cached(source, duration, config, directory):
    signature = score_signature(source, duration, config)
    try:
        data = json.loads((Path(directory) / "source_scores.json").read_text("utf-8"))
        if data.get("signature") == signature:
            return signature, data, [ScoreReading(**r) for r in data["readings"]]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return signature, {}, []


def cached_source_scores(source, duration, config, directory):
    if not (Path(directory) / "source_scores.json").is_file():
        return None
    _, data, readings = _cached(source, duration, config, directory)
    return readings if data.get("complete") else None


def cached_source_markers(source, duration, config, directory):
    if not (Path(directory) / "source_scores.json").is_file():
        return []
    _, data, _ = _cached(source, duration, config, directory)
    return data.get("replay_markers", []) if data.get("complete") else []


def scan_source_scores(source, duration, config, directory, progress=None):
    signature, saved, readings = _cached(source, duration, config, directory)
    if saved.get("complete"):
        return readings
    period = 1 / max(0.5, float(config.get("pot_selection.sample_fps", 2)))
    next_sample = float(saved.get("next_sample", 0))
    markers = list(saved.get("replay_markers", []))
    complete = False
    cap = open_capture(source, prefer_hwaccel=bool(config.get("analysis.hwaccel_decode", True)))
    stream = None

    def checkpoint():
        atomic_text(
            Path(directory) / "source_scores.json",
            json.dumps(
                {
                    "signature": signature,
                    "complete": complete,
                    "next_sample": next_sample,
                    "readings": [asdict(r) for r in readings],
                    "replay_markers": markers,
                }
            ),
        )

    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open the source for full-match score recovery")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_MSEC, next_sample * 1000)
        reader = ScoreboardReader(float(config.get("pot_selection.min_digit_confidence", 0.78)))
        stream = sampled_frames(
            cap,
            proxy_fps=fps,
            to_source=lambda t: t,
            start_time=next_sample,
            end_time=duration,
            sample_period=period,
            next_sample_time=next_sample,
            presentation_timestamps=True,
        )
        for count, (_, t, image) in enumerate(stream, 1):
            check_pause()
            reading = reader.read(image, t)
            if reading is not None:
                readings.append(reading)
            graphic = ViewClassifier.replay_stinger_signature(image)
            if graphic:
                markers.append({"t": t, "appearance_signature": graphic})
            while next_sample <= t + 1e-6:
                next_sample += period
            if count % 120 == 0:
                checkpoint()
                if progress:
                    progress(
                        min(1.0, t / duration),
                        f"Checking source scores at {t:.0f}s / {duration:.0f}s",
                    )
        complete = True
    finally:
        if stream is not None:
            stream.close()
        cap.release()
        checkpoint()
    return readings
