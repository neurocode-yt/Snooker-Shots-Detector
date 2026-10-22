"""Compare desktop and on-device detection on exactly the same source bytes.

Build and install both APKs first (scripts/build_android.ps1 -Test). Run with
the desktop virtual environment, for example:
  .venv/Scripts/python.exe scripts/check_android_parity.py clip.mp4

No processing is delegated to the PC by the Android app. This developer check
runs two independent analyses and rejects differing counts or clip boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--device", help="adb serial; omit when only one device is connected")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = digest.hexdigest()
    directory = ROOT / "apps/android/vendor/parity" / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)

    from snooker_ai.config import load_config
    from snooker_ai.pipeline.analyzer import Analyzer

    config = load_config(overrides={"device": "cpu", "analysis": {
        "native_contact_workers": 1, "native_contact_backend": "thread"}})
    print("Analyzing desktop reference", flush=True)
    result = Analyzer(config, directory / "desktop").analyze(source, "android-parity")
    reference = directory / "desktop-reference.json"
    reference.write_text(json.dumps({"source_sha256": identity, "shots": [
        {key: getattr(shot, key) for key in ("cue_strike", "clip_start", "clip_end")}
        for shot in result.shots]}, indent=2))

    adb = ["adb", "-s", args.device] if args.device else ["adb"]
    package = "com.snookerai.android"

    def run(*command, **kwargs):
        return subprocess.run([*adb, *command], check=True, **kwargs)

    private = run("shell", "run-as", package, "pwd", capture_output=True, text=True).stdout.strip()
    device_source = private + "/cache/parity-source.mp4"
    device_reference = private + "/cache/parity-reference.json"
    for index, (local, remote) in enumerate(((source, device_source), (reference, device_reference))):
        temporary = f"/data/local/tmp/snooker-parity-{identity[:16]}-{index}"
        run("push", str(local), temporary)
        run("shell", "run-as", package, "cp", temporary, remote)
        run("shell", "rm", temporary)
    print("Analyzing independently on Android; comparing every shot and boundary", flush=True)
    completed = run("shell", "am", "instrument", "-w", "-e", "class",
                    package + ".OnDeviceSmokeTest#testBroadcastParity",
                    "-e", "parity_source", device_source, "-e", "parity_expected", device_reference,
                    package + ".test/android.test.InstrumentationTestRunner", capture_output=True, text=True)
    (directory / "instrumentation.txt").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    report = run("exec-out", "run-as", package, "cat",
                 private + "/files/snooker-parity/broadcast-parity.json", capture_output=True).stdout
    (directory / "android-reference.json").write_bytes(report)
    if "OK (1 test)" not in completed.stdout:
        raise SystemExit("Android parity FAILED. See " + str(directory))
    print("Desktop/Android parity passed: " + str(directory), flush=True)


if __name__ == "__main__":
    main()
