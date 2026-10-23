"""Create a separate highlights version from a fingerprint-bound source review."""
# ruff: noqa: E402 -- repository bootstrap for direct CLI execution

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snooker_ai.config import load_config
from snooker_ai.evaluation.source_review import apply_source_review
from snooker_ai.jobs.store import JobStore
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import AnalysisResult, ExportRequest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def rebuild(manifest_path, *, render=False):
    manifest = json.loads(Path(manifest_path).read_text("utf-8"))
    config = load_config()
    store = JobStore(config)
    parent_id = manifest["parent_job_id"]
    parent_path = store.job_dir(parent_id) / "analysis.json"
    if sha256(parent_path) != manifest["parent_analysis_sha256"]:
        raise ValueError("Parent analysis changed; review corrections before applying them")
    parent = AnalysisResult.model_validate_json(parent_path.read_text("utf-8"))
    source = Path(parent.source_path)
    if (
        source.stat().st_size != manifest["source_size_bytes"]
        or sha256(source) != manifest["source_sha256"]
    ):
        raise ValueError("Source fingerprint does not match the reviewed video")
    corrected = apply_source_review(parent, manifest)
    job_id = store.create(source, mode=parent.mode.value, selection=corrected.selection)
    corrected.job_id = job_id
    store.save_analysis(corrected)
    meta = store.get_meta(job_id)
    meta.update(
        derived_from_job_id=parent_id,
        source_review=manifest["name"],
        source_review_manifest=str(Path(manifest_path).resolve()),
        pot_selection_summary=corrected.pot_selection_summary,
        auto_export=True,
        auto_export_ready=False,
        export_needs_update=True,
    )
    store._write_meta(job_id, meta)
    (store.job_dir(job_id) / "source_review.json").write_text(
        json.dumps(manifest, indent=2), "utf-8"
    )
    print(
        f"Created reviewed job {job_id}: {len(corrected.shots)} records, "
        f"{sum(s.included for s in corrected.shots)} selected",
        flush=True,
    )
    if render:
        store.update_progress(job_id, 0.9, "exporting", "Rendering source-reviewed highlights")
        try:
            exported = Exporter(config).export(
                corrected,
                store.job_dir(job_id) / "export",
                ExportRequest(mode=corrected.mode, export_clips=False),
            )
            if exported.joined_path is None or not exported.joined_path.is_file():
                raise RuntimeError("Reviewed export did not produce a video")
            meta = store.get_meta(job_id)
            meta.update(auto_export_ready=True, export_needs_update=False)
            store._write_meta(job_id, meta)
            store.update_progress(
                job_id,
                1,
                "completed",
                f"Source-reviewed highlights ready: {sum(s.included for s in corrected.shots)} clips",
                shots_detected=len(corrected.shots),
            )
            print(exported.joined_path.resolve(), flush=True)
        except Exception as exc:
            store.update_progress(
                job_id, 0.9, "failed", "Source-reviewed export needs retry", error=str(exc)
            )
            raise
    else:
        store.update_progress(
            job_id,
            1,
            "ready_for_review",
            "Source-reviewed timeline ready",
            shots_detected=len(corrected.shots),
        )
    return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    rebuild(args.manifest, render=args.render)
