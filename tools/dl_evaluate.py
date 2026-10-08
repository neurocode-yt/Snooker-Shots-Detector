"""Compare DL and classic delivered clips on a frozen reviewed-source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snooker_ai.dl.dataset import evaluate_video, load_manifest  # noqa: E402


def prediction_clips(path: Path, expected_source: str) -> list[dict[str, float]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        source = payload.get("source_path")
        if source and Path(source).name != Path(expected_source).name:
            raise ValueError("Predictions describe a different source file than the annotation manifest")
        rows = payload.get("clips", payload.get("shots", []))
    else:
        rows = payload
    clips = []
    for row in rows:
        if row.get("included", True) is False or row.get("include", True) is False:
            continue
        if "contact" in row:
            clips.append({"contact": float(row["contact"]), "start": float(row["start"]),
                          "end": float(row["end"])})
        else:
            clips.append({"contact": float(row["cue_strike"]), "start": float(row["clip_start"]),
                          "end": float(row["clip_end"])})
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/dl/manifest.json"))
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--classic", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    video = next((video for video in manifest["videos"] if video["id"] == args.video_id), None)
    if video is None:
        raise ValueError(f"Unknown video id: {args.video_id}")
    dl_clips = prediction_clips(args.predictions, video["source_path"])
    report = {"video_id": video["id"], "group_id": video["group_id"], "split": video["split"],
              "independent_holdout": video["split"] == "holdout"
                  and manifest.get("independent_holdout_available", False),
              "dl": evaluate_video(dl_clips, video, tolerance_seconds=args.tolerance),
              "limitations": manifest.get("limitations", []) + video.get("limitations", [])}
    if args.classic:
        classic_clips = prediction_clips(args.classic, video["source_path"])
        report["classic"] = evaluate_video(classic_clips, video, tolerance_seconds=args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key in ("video_id", "split",
                       "independent_holdout")} | {"output": str(args.output)}))


if __name__ == "__main__":
    main()
