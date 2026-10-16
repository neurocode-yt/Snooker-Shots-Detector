"""Compare classical evidence and timings against a Git baseline, without editing jobs.

Run from the repository root with a trusted local baseline revision. This checks
exact evidence parity on supplied inputs; it is not a ground-truth accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_baseline(revision: str):
    """Load only the trusted repository's prior implementation in separate modules."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    modules = {}
    for name, path in (
        ("objects", "snooker_ai/object_detection/detector.py"),
        ("motion", "snooker_ai/motion/residual.py"),
        ("scenes", "snooker_ai/scene_detection/detector.py"),
        ("pipeline", "snooker_ai/pipeline/analyzer.py"),
    ):
        source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
        module = types.ModuleType(f"_classic_baseline_{name}")
        module.__file__ = str(ROOT / path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(ROOT / path), "exec"), module.__dict__)
        modules[name] = module
    modules["pipeline"].ObjectDetector = modules["objects"].ObjectDetector
    modules["pipeline"].ResidualMotionAnalyzer = modules["motion"].ResidualMotionAnalyzer
    modules["pipeline"].SceneDetector = modules["scenes"].SceneDetector
    return revision, modules["objects"].ObjectDetector, modules["pipeline"].Analyzer


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_parallel_contacts(job, config, mapper, source, duration, starts, scratch, OldAnalyzer):
    """Compare ordered parent evidence to the bounded parallel contact path."""
    from snooker_ai.pipeline.analyzer import Analyzer, _NativeContactPrefetch
    from snooker_ai.types import FrameFeatures, StrikeCandidate

    cache = json.loads((job / "coarse_checkpoint.json").read_text())
    old, new = [cls(config, scratch / f"parallel-{i}") for i, cls in enumerate((OldAnalyzer, Analyzer))]
    for owner in (old, new):
        owner._native_source = source
        owner._coarse_context_reference = [FrameFeatures(t=f['t'], match_context_valid=f.get('match_context_valid', True))
                                          for f in cache['features']]
        owner.broadcast_context.restore_target(cache.get('broadcast_target', []))
    candidates = [StrikeCandidate(timestamp=t+2, confidence=.5) for t in starts]
    proxy = job / "proxy/proxy.mp4"
    with _NativeContactPrefetch(new, proxy, None, mapper, duration, candidates) as pool:
        eligible = sum(pool.eligible(key) for key in pool.plans)
        outputs, timings = [], []
        for use_pool, owner in ((False, old), (True, new)):
            rows = []
            began = time.perf_counter()
            for i, (start, end, fps) in enumerate(pool.plans):
                part = pool.pull(i, (start, end, fps), lambda _: True) if use_pool else None
                if part is None:
                    part, _, _ = owner._extract_features(proxy, None, mapper, duration,
                        sample_fps=fps, start_time=start, end_time=end)
                if not part:
                    raise ValueError("Refusing an empty parallel comparison")
                rows.append([f.model_dump(mode='json') for f in part])
            timings.append(time.perf_counter()-began)
            outputs.append(rows)
    return {"job": job.name, "frames": sum(map(len, outputs[0])), "eligible_contacts": eligible,
        "identical": outputs[0] == outputs[1], "baseline_seconds": timings[0], "optimized_seconds": timings[1],
        "baseline_hash": fingerprint(outputs[0]), "optimized_hash": fingerprint(outputs[1])}


def main():
    import cv2
    import numpy as np
    from snooker_ai.config import load_config
    from snooker_ai.ingestion.probe import probe_video
    from snooker_ai.object_detection.detector import ObjectDetector
    from snooker_ai.pipeline.analyzer import Analyzer
    from snooker_ai.utils.timebase import TimeMapper

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--job-dir", type=Path, action="append", default=[])
    parser.add_argument("--start", type=float, action="append")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--opencv-threads", type=int, help="Experiment with a CPU thread budget for the optimized pass")
    parser.add_argument("--parallel-contacts", action="store_true", help="Also compare serial/parallel contact evidence using the coarse checkpoint")
    parser.add_argument("--max-images", type=int, default=0, help="0 checks every hard-fixture image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.max_images < 0 or args.opencv_threads is not None and args.opencv_threads < 1:
        parser.error("image limit must be nonnegative and thread count must be positive")
    if any(not math.isfinite(start) or start < 0 for start in args.start or []):
        parser.error("--start must be nonnegative and finite")
    if any(args.output.resolve().is_relative_to(job.resolve()) for job in args.job_dir):
        parser.error("the report must be outside the inspected job directories")
    revision, OldObjects, OldAnalyzer = load_baseline(args.baseline_ref)
    config = load_config(overrides={"device": args.device})
    original_threads = cv2.getNumThreads()
    report = {"baseline": revision, "device": args.device, "images": [], "native_windows": [], "parallel_contacts": []}
    paths = sorted(p for p in (ROOT / "tests/fixtures/classic_hard").rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError("No reference images found; an empty comparison is not a parity check")
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable fixture: {path}")
        mask = np.full(frame.shape[:2], 255, np.uint8)
        outputs, timings = [], []
        for cls in (OldObjects, ObjectDetector):
            cv2.setNumThreads(original_threads if cls is OldObjects else args.opencv_threads or original_threads)
            detector = cls(config)
            cv2.setRNGSeed(0)
            started = time.perf_counter()
            detections = detector.detect(frame, mask, partial_view=True)
            timings.append(time.perf_counter() - started)
            outputs.append([asdict(item) for item in detections])
        report["images"].append({"image": str(path.relative_to(ROOT)), "identical": outputs[0] == outputs[1],
            "baseline_seconds": timings[0], "optimized_seconds": timings[1],
            "baseline_hash": fingerprint(outputs[0]), "optimized_hash": fingerprint(outputs[1])})
    with tempfile.TemporaryDirectory(prefix="classic-parity-") as scratch:
        for job in args.job_dir:
            source = Path(json.loads((job / "job.json").read_text())["source_path"])
            proxy = job / "proxy/proxy.mp4"
            meta, proxy_meta = probe_video(source), probe_video(proxy)
            mapper = TimeMapper(meta.duration, proxy_meta.duration, meta.fps)
            owners = [cls(config, Path(scratch) / str(i)) for i, cls in enumerate((OldAnalyzer, Analyzer))]
            for owner in owners:
                owner._native_source = source
            signature_unchanged = owners[0]._analysis_signature(source) == owners[1]._analysis_signature(source)
            for start in args.start or [73., 74.]:
                if start >= meta.duration:
                    raise ValueError(f"Window starts beyond the source duration: {start}")
                outputs, timings = [], []
                for owner in owners:
                    cv2.setNumThreads(original_threads if isinstance(owner, OldAnalyzer)
                                      else args.opencv_threads or original_threads)
                    cv2.setRNGSeed(0)
                    started = time.perf_counter()
                    frames, _, _ = owner._extract_features(proxy, None, mapper, meta.duration,
                        sample_fps=30, start_time=start, end_time=min(meta.duration, start + args.seconds))
                    timings.append(time.perf_counter() - started)
                    outputs.append([f.model_dump(mode="json") for f in frames])
                if not outputs[0] or not outputs[1]:
                    raise ValueError(f"No native evidence at {start}; refusing an empty comparison")
                row = {"job": job.name, "start": start, "frames": len(outputs[0]),
                    "identical": outputs[0] == outputs[1], "baseline_seconds": timings[0],
                    "signature_unchanged": signature_unchanged,
                    "optimized_seconds": timings[1], "baseline_hash": fingerprint(outputs[0]),
                    "optimized_hash": fingerprint(outputs[1])}
                report["native_windows"].append(row)
                print(json.dumps(row), flush=True)
            if args.parallel_contacts:
                cv2.setNumThreads(original_threads)
                row = compare_parallel_contacts(job, config, mapper, source, meta.duration,
                    args.start or [73., 74.], Path(scratch), OldAnalyzer)
                report["parallel_contacts"].append(row)
                print(json.dumps(row), flush=True)
    report["identical"] = all(row["identical"] for group in ("images", "native_windows", "parallel_contacts") for row in report[group])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"identical": report["identical"], "images": len(paths),
                      "native_windows": len(report["native_windows"])}), flush=True)
    if not report["identical"]:
        raise SystemExit("Evidence differs from baseline; do not claim unchanged accuracy")


if __name__ == "__main__":
    main()
