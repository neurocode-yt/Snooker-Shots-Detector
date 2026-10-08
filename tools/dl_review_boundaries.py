"""Create original-source montage evidence without consulting predictions.

This utility decodes the original Selby source, records native frame indexes and
presentation timestamps, and saves contact/outcome sheets for visual annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def make_sheets(source: Path, output: Path, requests: dict[str, list[float]]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode original source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    by_index: dict[int, list[tuple[str, int]]] = {}
    canvases: dict[str, Image.Image] = {}
    width, height, columns = 480, 270, 5
    rows: dict[str, list] = {name: [None] * len(times) for name, times in requests.items()}
    for name, times in requests.items():
        canvases[name] = Image.new("RGB", (width * columns, (height + 24) * ((len(times) + 4) // 5)))
        for slot, timestamp in enumerate(times):
            by_index.setdefault(round(timestamp * fps), []).append((name, slot))
    first, last = min(by_index), max(by_index)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    while True:
        index = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if index > last:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if index not in by_index:
            continue
        timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        thumbnail = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB))
        for name, slot in by_index[index]:
            x, y = (slot % columns) * width, (slot // columns) * (height + 24)
            canvas = canvases[name]
            canvas.paste(thumbnail, (x, y + 24))
            ImageDraw.Draw(canvas).text((x + 5, y + 4), f"PTS {timestamp:.3f}s | frame {index}", fill="white")
            rows[name][slot] = {"frame_index": index, "source_pts": timestamp,
                                "requested_time": requests[name][slot]}
    capture.release()
    if any(row is None for values in rows.values() for row in values):
        raise RuntimeError("Some requested original-source frames were not decoded")
    for name, canvas in canvases.items():
        canvas.save(output / f"{name}.jpg", quality=94)
    metadata = {"source_path": str(source), "opencv_average_fps": fps,
                "method": "Original-source decoding only; no detector, model output or classic endpoints read",
                "reviewed_frames": rows}
    (output / "frames.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=Path)
    args = parser.parse_args()
    if args.requests:
        requests = json.loads(args.requests.read_text(encoding="utf-8"))
    else:
        benchmark = json.loads(Path("benchmarks/selby_lisowski_recall.json").read_text(encoding="utf-8"))
        contacts = benchmark["windows"][0]["contacts"][:12]
        requests = {}
        for index, contact in enumerate(contacts, start=1):
            center = (contact["lower"] + contact["upper"]) / 2
            requests[f"{index:02d}_contact"] = np.arange(center - 1.2, center + 1.21, .12).tolist()
            requests[f"{index:02d}_outcome"] = np.arange(contact["lower"] - 3, contact["upper"] + 16, .5).tolist()
    metadata = make_sheets(args.source, args.output, requests)
    print(json.dumps({"sheets": len(requests), "frames": sum(len(v) for v in metadata["reviewed_frames"].values()),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
