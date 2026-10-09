"""Run one DL Algo extraction, training and decoded-clip benchmark command.

Run with the isolated DL interpreter. This is a single process, not a scheduler.
Feature extraction reuses compatible caches and resumes persisted chunks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def exclusive_run_lock(state_path: Path):
    """Use an OS-owned lock; abandoned processes release it automatically."""
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunState:
    def __init__(self, path: Path, arguments: dict[str, Any]):
        self.path = path
        self.data: dict[str, Any] = {
            "state_version": 1, "status": "running", "stage": "starting",
            "pid": os.getpid(), "started_at": utc_now(), "arguments": arguments,
            "deployment_quality_verified": False,
            "note": "One extraction/training/benchmark run; completion alone is not a quality guarantee.",
        }
        self.update()

    def update(self, **changes: Any) -> None:
        self.data.update(changes)
        self.data["updated_at"] = utc_now()
        atomic_json(self.path, self.data)

    def stage(self, name: str, message: str, **changes: Any) -> None:
        self.update(stage=name, message=message, **changes)
        print(json.dumps({"type": "stage", "stage": name, "message": message, **changes}), flush=True)


class TrainingOutput:
    """Forward training output while recording the latest completed epoch."""

    def __init__(self, state: RunState):
        self.state = state
        self.destination = sys.stdout
        self.pending = ""

    def write(self, value: str) -> int:
        self.destination.write(value)
        self.destination.flush()
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(entry, dict) and "epoch" in entry:
                self.state.update(training_epoch=entry, message=f"Completed temporal epoch {entry['epoch']}")
        return len(value)

    def flush(self) -> None:
        self.destination.flush()


def extract_features(args: argparse.Namespace, assets: Path, device: str, state: RunState) -> None:
    command = [sys.executable, "-u", "-m", "snooker_ai.dl.features",
               "--manifest", str(args.manifest), "--assets", str(assets), "--device", args.device]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    state.stage("extracting_features", "Extracting or resuming frozen RGB and learned motion features",
                feature_command=command, current_video=None, feature_percent=None)
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    state.update(feature_worker_pid=process.pid)
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            print(line, flush=True)
            changes: dict[str, Any] = {"last_feature_log": line[-2000:]}
            if line.startswith("FEATURE VIDEO "):
                changes.update(current_video=line.split(maxsplit=3)[2], feature_percent=0)
            progress = re.match(r"^(\d+)%\s+(.*)", line)
            if progress:
                changes.update(feature_percent=int(progress.group(1)), message=progress.group(2))
            if line.startswith("FEATURE READY "):
                changes.update(feature_percent=100, latest_feature_archive=line[len("FEATURE READY "):])
            state.update(**changes)
        code = process.wait()
        if code:
            raise RuntimeError(f"Neural feature worker exited with code {code}; see the preceding stage log.")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        state.update(feature_worker_pid=None)


def benchmark(args: argparse.Namespace, settings: dict, device: str, state: RunState) -> Path:
    import numpy as np

    from snooker_ai.dl.dataset import evaluate_video, load_feature_video, load_manifest
    from snooker_ai.dl.pipeline import validate_calibration
    from snooker_ai.dl.selection import select_highlights
    from snooker_ai.dl.settings import expected_feature_spec
    from snooker_ai.dl.temporal import HEAD_NAMES, load_temporal_checkpoint, predict_probabilities
    from tools.dl_evaluate import prediction_clips

    manifest = load_manifest(args.manifest)
    base = (args.manifest.parent / manifest.get("paths_relative_to", ".")).resolve()
    assets = Path(settings["assets"])
    identity = json.loads((assets / "assets.json").read_text(encoding="utf-8"))
    rgb_config = json.loads((assets / "rgb/config.json").read_text(encoding="utf-8"))
    specification = expected_feature_spec(settings, identity, rgb_config)
    model, checkpoint = load_temporal_checkpoint(args.output, device=device,
                                                expected_feature_spec=specification)
    calibration = checkpoint["calibration"]
    validate_calibration(calibration, settings["selection"])
    selection = dict(settings["selection"])
    selection.update(head_validation=calibration["frame_heads"])
    tolerance = (args.tolerance if args.tolerance is not None
                 else float(calibration["event_tolerance_seconds"]))
    index: dict[str, Any] = {
        "algorithm": "dl_algo", "scope": "decoded_selected_clips_from_cached_neural_features",
        "manifest": str(args.manifest), "manifest_sha256": file_hash(args.manifest),
        "checkpoint": str(args.output), "checkpoint_sha256": file_hash(args.output),
        "selection_settings": selection, "calibration": calibration,
        "contact_tolerance_seconds": tolerance, "videos": [],
        "independent_holdout_available": manifest.get("independent_holdout_available", False),
        "deployment_quality_verified": False,
        "limitations": manifest.get("limitations", []) + [
            "Benchmarks decode clip records; generated MP4s still require rendering and visual review.",
            "Partial-source cached features cover reviewed windows and context; unreviewed contact predictions are ignored.",
            "Training footage is in-sample; dev footage selects the model and thresholds.",
        ],
    }
    args.benchmarks.mkdir(parents=True, exist_ok=True)
    for video in manifest["videos"]:
        video_id = str(video["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError("Benchmark video identifiers must be simple filename-safe identifiers.")
        state.stage("benchmarking", f"Decoding actual selected clips for {video_id}", current_video=video_id)
        feature_path = Path(video["features_path"])
        if not feature_path.is_absolute():
            feature_path = base / feature_path
        timestamps, features, feature_spec = load_feature_video(feature_path)
        if feature_spec != specification:
            raise ValueError(f"Feature specification mismatch for {video_id}; refusing benchmark reuse.")
        probabilities = predict_probabilities(
            model, features, chunk_frames=int(settings.get("temporal_chunk_frames", 512)),
            device=device, temperatures=calibration["temperatures"], timestamps=timestamps)
        candidates, shots, diagnostics = select_highlights(
            timestamps, probabilities, float(video["duration"]), selection, calibration["thresholds"])
        clips = [{"contact": float(shot.cue_strike), "start": float(shot.clip_start),
                  "end": float(shot.clip_end)} for shot in shots if shot.included]
        report: dict[str, Any] = {
            "video_id": video_id, "group_id": video["group_id"], "split": video["split"],
            "source_path": video["source_path"], "features_path": str(feature_path),
            "feature_samples": len(timestamps), "selected_clip_count": len(clips),
            "selected_seconds": sum(clip["end"] - clip["start"] for clip in clips),
            "dl": evaluate_video(clips, video, tolerance_seconds=tolerance),
            "dl_strict": evaluate_video(clips, video, tolerance_seconds=0),
            "diagnostics": diagnostics,
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "shots": [shot.model_dump(mode="json") for shot in shots], "clips": clips,
            "independent_holdout": video["split"] == "holdout"
                and manifest.get("independent_holdout_available", False),
            "boundary_reference_kind": "Reviewed editorial preferences; no physical-stop observation claimed",
            "deployment_quality_verified": False, "limitations": video.get("limitations", []),
        }
        baseline_id = video.get("classic_baseline_job")
        if baseline_id and re.fullmatch(r"[A-Za-z0-9_-]+", str(baseline_id)):
            baseline = base / "data/jobs" / str(baseline_id) / "export/export_metadata.json"
            if baseline.exists():
                classic_clips = prediction_clips(baseline, video["source_path"])
                report.update(classic_baseline_path=str(baseline),
                              classic=evaluate_video(classic_clips, video, tolerance_seconds=tolerance))
        probability_path = args.benchmarks / f"{video_id}.probabilities.npz"
        temporary = probability_path.with_name(f"{video_id}.probabilities.partial.npz")
        np.savez_compressed(temporary, timestamps=timestamps, probabilities=probabilities,
                            head_names=np.asarray(HEAD_NAMES),
                            checkpoint_sha256=index["checkpoint_sha256"],
                            feature_spec=json.dumps(feature_spec, sort_keys=True))
        temporary.replace(probability_path)
        report_path = args.benchmarks / f"{video_id}.json"
        report["probabilities_path"] = str(probability_path)
        atomic_json(report_path, report)
        summary = {key: report[key] for key in ("video_id", "split", "selected_clip_count", "dl", "dl_strict")}
        summary["report_path"] = str(report_path)
        if "classic" in report:
            summary["classic"] = report["classic"]
        index["videos"].append(summary)
        state.update(benchmark_videos_completed=len(index["videos"]), latest_benchmark=summary)
        print(json.dumps({"type": "decoded_benchmark", "video_id": video_id,
                          "selected_clips": len(clips), "report_path": str(report_path),
                          "metrics": {key: report['dl'][key] for key in (
                              'true_positive', 'false_positive', 'false_negative', 'precision', 'recall',
                              'false_replay_clips', 'false_handling_clips')}}, allow_nan=False), flush=True)
        atomic_json(args.benchmarks / "summary.json", index)
    return args.benchmarks / "summary.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=ROOT / "data/evaluation/dl/manifest.json")
    result.add_argument("--output", type=Path, default=ROOT / "models/dl_algo/temporal-v1.pt")
    result.add_argument("--assets", type=Path)
    result.add_argument("--state", type=Path, default=ROOT / "data/dl/training/state.json")
    result.add_argument("--benchmarks", type=Path, default=ROOT / "data/dl/training/benchmarks")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--window-frames", type=int, default=512)
    result.add_argument("--hidden-dim", type=int, default=96)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--tolerance", type=float,
                        help="Decoded contact tolerance; defaults to the checkpoint calibration tolerance")
    return result


def run(args: argparse.Namespace) -> int:
    state = RunState(args.state, {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()})
    try:
        import torch

        from snooker_ai.dl.dataset import load_manifest
        from snooker_ai.dl.settings import dl_settings, resolved_path
        from snooker_ai.dl.training import TrainingConfig, train_temporal

        load_manifest(args.manifest)
        fingerprint = file_hash(args.manifest)
        settings = dl_settings()
        assets = args.assets or resolved_path(settings["assets"])
        settings.update(assets=str(assets), checkpoint=str(args.output))
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose --device auto or cpu.")
        if device == "cpu":
            torch.set_num_threads(4)
        selection = settings["selection"]
        configuration = TrainingConfig(
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            window_frames=args.window_frames, stride_frames=max(1, args.window_frames // 2),
            hidden_dim=args.hidden_dim, seed=args.seed,
            min_event_gap_seconds=float(selection.get("nms_seconds", 1)),
            event_uncertainty_ratio=float(selection.get("uncertainty_ratio", .5)),
            max_event_uncertainty_seconds=float(selection.get("max_uncertainty_seconds", 2)),
        )
        state.update(device=device, torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
                     manifest_sha256=fingerprint)
        extract_features(args, assets, device, state)
        if file_hash(args.manifest) != fingerprint:
            raise RuntimeError("Frozen annotation manifest changed during extraction; restart with a consistent manifest.")
        state.stage("training", "Fitting the new temporal network on frozen match-isolated labels",
                    current_video=None, feature_percent=None)
        with redirect_stdout(TrainingOutput(state)):
            report = train_temporal(args.manifest, args.output, config=configuration, device=device)
        state.update(checkpoint=str(args.output), checkpoint_sha256=file_hash(args.output),
                     training_report=str(args.output.with_suffix(args.output.suffix + ".report.json")),
                     best_epoch=report["best_epoch"])
        if file_hash(args.manifest) != fingerprint:
            raise RuntimeError("Frozen annotation manifest changed during training; benchmark stopped.")
        benchmark_path = benchmark(args, settings, device, state)
        state.stage("complete", "Extraction, temporal training and decoded-clip benchmarks completed",
                    status="complete", finished_at=utc_now(), benchmark_summary=str(benchmark_path),
                    current_video=None)
        return 0
    except BaseException as exc:
        state.stage("failed", f"{type(exc).__name__}: {exc}", status="failed", finished_at=utc_now(),
                    error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main() -> int:
    args = parser().parse_args()
    for key in ("manifest", "output", "state", "benchmarks", "assets"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    try:
        with exclusive_run_lock(args.state):
            return run(args)
    except OSError as exc:
        print(f"Cannot acquire the exclusive training state lock: {exc}. "
              "An existing run keeps its state and features.", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
