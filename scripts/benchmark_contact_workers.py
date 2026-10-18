"""Time unchanged native windows in a temporary directory, using saved match context."""

import argparse
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from snooker_ai.config import load_config
from snooker_ai.ingestion.probe import probe_video
from snooker_ai.pipeline.analyzer import Analyzer, _NativeContactPrefetch
from snooker_ai.types import FrameFeatures
from snooker_ai.utils.timebase import TimeMapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--starts", nargs="+", type=float, required=True)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--backend", choices=("thread", "process"), default="process")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or any(
        not math.isfinite(t) or t < 0 for t in args.starts
    ):
        parser.error("Use positive window lengths and nonnegative source times")
    job = args.job_dir.resolve()
    if args.output and args.output.resolve().is_relative_to(job):
        parser.error("Write benchmark reports outside the inspected job directory")
    meta = json.loads((job/"job.json").read_text("utf-8"))
    saved = json.loads((job/"coarse_checkpoint.json").read_text("utf-8"))
    source = probe_video(meta["source_path"])
    proxy = job/"proxy/proxy.mp4"
    proxy_info = probe_video(proxy)
    mapper = TimeMapper(source.duration, proxy_info.duration, source.fps, 2.)
    config = load_config(overrides={"analysis": {
        "native_contact_backend": args.backend, "native_contact_workers": args.workers,
    }})
    with tempfile.TemporaryDirectory(prefix="snooker-contact-bench-") as directory:
        analyzer = Analyzer(config, Path(directory))
        analyzer._native_source = Path(source.path)
        # Saved floats already describe the learned template. A second
        # normalization can perturb a score exactly at an identity threshold.
        target = np.asarray(saved.get("broadcast_target", []), dtype=np.float32)
        if target.size:
            if target.shape != (3072,) or not np.isfinite(target).all():
                raise ValueError("Invalid saved broadcast template")
            analyzer.broadcast_context._template = target
        analyzer._coarse_context_reference = [FrameFeatures.model_validate(f) for f in saved["features"]]
        analyzer._confirmed_target_returns = [tuple(v) for v in saved.get("confirmed_target_returns", [])]
        rows = []
        start = time.perf_counter()
        with _NativeContactPrefetch(analyzer, proxy, None, mapper, source.duration, []) as pool:
            fps = min(source.fps, config.get("analysis.refine_fps"), config.get("proxy.target_fps"))
            pool.plans = [(t, min(source.duration, t+args.seconds), fps) for t in args.starts]
            for number, key in enumerate(pool.plans):
                if not pool.eligible(key):
                    raise ValueError(f"Window at {key[0]}s requires chronological parent context")
                features = pool.pull(number, key, lambda _: True)
                if not features:
                    raise RuntimeError(f"Window at {key[0]}s needs parent fallback; choose another window")
                payload = json.dumps([f.model_dump(mode="json") for f in features], sort_keys=True)
                rows.append({"start": key[0], "frames": len(features),
                             "sha256": hashlib.sha256(payload.encode()).hexdigest()})
            workers = pool.workers
        elapsed = time.perf_counter()-start
    report = {"source": Path(source.path).name, "backend": args.backend, "workers": workers,
              "seconds": elapsed, "window_seconds": args.seconds, "windows": rows}
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
