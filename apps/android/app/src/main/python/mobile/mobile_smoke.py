"""Device smoke checks, called by the Android instrumentation test only."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request


def check_broadcast(source, expected_path, storage, native_dir):
    """Compare a real, user-supplied clip with its desktop analysis.

    The fixture is supplied through adb, never embedded in the released APK.
    A fresh output directory avoids accepting a previous decoder's checkpoints.
    """
    import hashlib
    import mobile_runtime
    import time
    from snooker_ai.pipeline.analyzer import Analyzer

    mobile_runtime.start(storage, native_dir)
    source = Path(source)
    expected = json.loads(Path(expected_path).read_text())
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected["source_sha256"]
    directory = Path(storage) / ("broadcast-parity-" + str(time.time_ns()))
    result = Analyzer(mobile_runtime._config, directory).analyze(source, "broadcast-parity")
    report = {
        "source_sha256": expected["source_sha256"],
        "shots": [{key: getattr(shot, key) for key in ("cue_strike", "clip_start", "clip_end")}
                  for shot in result.shots],
        "directory": str(directory),
    }
    Path(storage, "broadcast-parity.json").write_text(json.dumps(report, indent=2))
    assert len(report["shots"]) == len(expected["shots"]), report
    for actual, reference in zip(report["shots"], expected["shots"]):
        for key in reference:
            assert abs(actual[key] - reference[key]) < .001, (key, actual, reference)
    return json.dumps(report)


def check_scoreboards(directory):
    import cv2
    from snooker_ai.pot_detection.scoreboard import ScoreboardReader
    root = Path(directory) / "blue_26_31"
    case = json.loads((root / "source.json").read_text())
    count = 0
    for sample in case["heldout_frames"]:
        image = cv2.imread(str(root / (sample["name"] + ".png")))
        for width in (1280, 1920):
            reading = ScoreboardReader().read(cv2.resize(image, (width, round(width * 720 / 1280))),
                                             sample["source_time"])
            assert reading is not None, sample["name"]
            assert (reading.left, reading.right) == tuple(sample["score"]), sample["name"]
            assert reading.active_player == sample["active_player"]
            count += 1
    return count


def run(storage, native_dir):
    import importlib.util
    import mobile_runtime
    import time
    session = json.loads(mobile_runtime.start(storage, native_dir))
    for _ in range(600):
        if mobile_runtime.ready():
            break
        time.sleep(.1)
    assert mobile_runtime.ready(), "Local server did not start"
    base, token = session["url"], session["token"]

    def request(path, authenticated=True):
        headers = {"Cookie": "snooker_mobile=" + token} if authenticated else {}
        return urllib.request.urlopen(urllib.request.Request(base + path, headers=headers))

    try:
        request("/api/jobs", authenticated=False)
        raise AssertionError("Unauthenticated local access was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 403
    assert json.load(request("/health"))["status"] == "ok"
    assert [item["id"] for item in json.load(request("/api/algorithms"))["algorithms"]] == ["classic"]
    assert importlib.util.find_spec("snooker_ai.dl") is None, "DL code must not be in the mobile package"
    assert b"/mobile/mobile.js" in request("/").read()

    import cv2
    import numpy as np
    from snooker_ai.ingestion.probe import probe_video
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.rendering.exporter import Exporter
    from snooker_ai.rendering.shorts import ShortsSettings, render_short
    from snooker_ai.types import ShotRecord, ExportRequest
    from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command

    directory = Path(storage) / "device-smoke"
    directory.mkdir(exist_ok=True)
    raw, source = directory / "fixture.bgr", directory / "fixture.mp4"
    width, height, fps = 320, 180, 10
    with raw.open("wb") as stream:
        for index in range(100):
            image = np.full((height, width, 3), (30, 25, 20), np.uint8)
            cv2.rectangle(image, (25, 25), (295, 155), (40, 140, 40), -1)
            offset = min(70, max(0, index - 30) * 3)
            cv2.circle(image, (75 + offset, 95), 5, (245, 245, 245), -1)
            cv2.circle(image, (210, 90), 5, (20, 20, 210), -1)
            stream.write(image.tobytes())
    run_command([find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "bgr24", "-video_size", "320x180", "-framerate", str(fps),
                 "-i", str(raw), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                 "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source)])
    raw.unlink()
    metadata = probe_video(source)
    assert metadata.has_audio and metadata.width == width
    capture = cv2.VideoCapture(str(source))
    assert capture.isOpened()
    assert capture.set(cv2.CAP_PROP_POS_MSEC, 3200)
    assert capture.read()[0]
    assert abs(capture.get(cv2.CAP_PROP_POS_MSEC) - 3200) < 2
    capture.release()
    config = mobile_runtime._config
    checkpoint_report = check_checkpoint_resume(source, config, metadata.duration, directory)
    # The actual classic analyzer must finish; no mocks or server-side worker.
    result = Analyzer(config, directory / "analysis").analyze(source, "android-smoke")
    assert result.original_duration >= 9.9
    assert len(result.features) > 0
    # Explicit review edits also exercise the editor's export contract even
    # when this simple synthetic table produces no confirmed cue contact.
    shots = [ShotRecord(shot_id=1, cue_strike=2, ball_motion_end=6, clip_start=0, clip_end=4,
                        user_modified=True),
             ShotRecord(shot_id=2, cue_strike=7, ball_motion_end=10, clip_start=5, clip_end=9,
                        user_modified=True)]
    edited = result.model_copy(update={"shots": shots})
    exported = Exporter(config).export(edited, directory / "export", ExportRequest(export_clips=False))
    output = probe_video(exported.joined_path)
    assert output.has_audio and 7 < output.duration <= 8.1
    short = directory / "short.mp4"
    render_short(edited, shots[0], ShortsSettings(player_names="Android test"), short, config)
    portrait = probe_video(short)
    assert (portrait.width, portrait.height) == (1080, 1920) and portrait.has_audio
    return json.dumps({"server": True, "authentication": True, "decoder": True,
                       "analysis_features": len(result.features), "detected_shots": len(result.shots),
                       "highlight_seconds": output.duration, "shorts": True,
                       "checkpoint_recovery": checkpoint_report,
                       "opencv": cv2.__version__})


def check_checkpoint_resume(source, config, duration, directory):
    """Real Android decode/track/serialize/restore; no mocked detector output."""
    from snooker_ai.config import Config
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.pipeline.checkpoints import load_snapshot
    from snooker_ai.utils.timebase import TimeMapper
    from snooker_ai.utils.work_control import WorkPaused
    cfg = Config(config.as_dict())
    cfg._data["analysis"]["checkpoint_every_samples"] = 4
    mapper = TimeMapper(duration, duration, 10)
    options = dict(sample_fps=10, start_time=2, end_time=5,
                   checkpoint_signature="android-native-recovery", resume=False)
    jobs = Path(cfg.get("paths.jobs_dir"))
    whole = Analyzer(cfg, jobs / "checkpoint-smoke-whole")
    whole._native_source = source
    expected, _, _ = whole._extract_features(source, None, mapper, duration, **options)
    interrupted = Analyzer(cfg, jobs / "checkpoint-smoke-interrupted")
    interrupted._native_source = source
    original = interrupted._write_checkpoint

    def fail(data):
        original(data)
        if data["source_time"] >= 3:
            raise WorkPaused()

    interrupted._write_checkpoint = fail
    try:
        interrupted._extract_features(source, None, mapper, duration, **options)
        raise AssertionError("Interruption was not reached")
    except WorkPaused:
        pass
    saved = load_snapshot(interrupted.job_dir / "native_partial.npz")
    from mobile_backup import _store
    (interrupted.job_dir / "native_partial.npz").write_bytes(b"damaged primary checkpoint")
    repaired = _store.recover()
    assert any(path.endswith("checkpoint-smoke-interrupted/native_partial.npz") for path in repaired)
    assert load_snapshot(interrupted.job_dir / "native_partial.npz")["next_idx"] == saved["next_idx"]
    resumed = Analyzer(cfg, interrupted.job_dir)
    resumed._native_source = source
    messages = []
    options.update(resume=True, progress=lambda fraction, message: messages.append(message))
    actual, _, _ = resumed._extract_features(source, None, mapper, duration, **options)
    assert any("Continuing from saved checkpoint" in message for message in messages)
    assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]
    return {"saved_next_frame": saved["next_idx"], "identical_features": True,
            "corrupt_primary_repaired": True}
