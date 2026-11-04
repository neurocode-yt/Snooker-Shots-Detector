"""Compare every exported clip with its source, then decode the complete video."""
# ruff: noqa: E402 -- repository bootstrap for direct CLI execution

import argparse
from fractions import Fraction
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snooker_ai.types import AnalysisResult
from snooker_ai.utils.ffmpeg import ffprobe_json, find_ffmpeg


def validate(analysis_path, export_dir, report_path):
    result = AnalysisResult.model_validate_json(Path(analysis_path).read_text("utf-8"))
    export_dir = Path(export_dir)
    metadata = json.loads((export_dir / "export_metadata.json").read_text("utf-8"))
    expected = [s.shot_id for s in result.shots if s.included and not s.possible_replay]
    if not expected:
        raise ValueError("The reviewed export has no selected clips")
    if [s["shot_id"] for s in metadata["shots"]] != expected:
        raise ValueError("Rendered clip IDs do not match the reviewed selection")
    output = export_dir / "highlights.mp4"
    probe = ffprobe_json(output)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(video["r_frame_rate"]))
    cv2.setNumThreads(2)
    rendered, source = cv2.VideoCapture(str(output)), cv2.VideoCapture(result.source_path)
    checks = []

    def pixels(capture, when):
        capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).ravel()

    try:
        for shot, start in zip(metadata["shots"], metadata["output_shot_starts"], strict=True):
            offset = min(max(shot["cue_strike"] - shot["clip_start"] + .5, 1),
                         shot["clip_end"] - shot["clip_start"] - 1)
            when = round((start + offset) * fps) / fps
            actual = pixels(rendered, when)
            if actual is None:
                raise RuntimeError(f"Cannot decode output shot {shot['shot_id']}")
            target = shot["clip_start"] + when - start
            matches = []
            for delta in (-2, -1, 0, 1, 2):
                reference = pixels(source, target + delta / fps)
                if reference is not None:
                    matches.append((float(np.corrcoef(actual, reference)[0, 1]), delta / fps))
            if not matches:
                raise RuntimeError(f"Cannot decode source for shot {shot['shot_id']}")
            correlation, shift = max(matches)
            checks.append({"shot_id": shot["shot_id"], "source_time": target,
                           "output_time": when, "luma_correlation": correlation,
                           "source_frame_offset": shift})
            if len(checks) % 40 == 0:
                print(f"Compared {len(checks)}/{len(expected)} clips", flush=True)
    finally:
        rendered.release()
        source.release()
    decoded = subprocess.run([find_ffmpeg(), "-v", "error", "-xerror", "-i", str(output),
                              "-f", "null", "-"], capture_output=True, text=True)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report = {"job_id": result.job_id, "selected_clips": len(expected),
              "output_duration": float(probe["format"]["duration"]),
              "expected_mix_duration": metadata["edited_duration"],
              "output_sha256": digest.hexdigest(), "full_decode_exit_code": decoded.returncode,
              "full_decode_errors": decoded.stderr,
              "minimum_luma_correlation": min(c["luma_correlation"] for c in checks),
              "frame_comparisons": checks}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    if (decoded.returncode or decoded.stderr.strip()
            or abs(report["output_duration"] - report["expected_mix_duration"]) > .15
            or not isfinite(report["minimum_luma_correlation"])
            or report["minimum_luma_correlation"] <= .995):
        raise RuntimeError(f"Export validation failed; see {report_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "frame_comparisons"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(args.analysis, args.export_dir, args.report)
