"""Freeze existing source reviews into leak-free neural training manifests.

Run from the repository root with ``python tools/dl_prepare_data.py``. The raw
videos and earlier audit media remain local and are never copied into the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snooker_ai.dl.dataset import HEADS, MANIFEST_VERSION, split_manifest, validate_manifest  # noqa: E402

ZHAO_GROUP = "zhao_trump_2026_quarter_final"
SELBY_GROUP = "lisowski_selby_2026_quarter_final"
SELBY_SOURCE = Path(
    "G:/Adobe Premiere Pro Auto-Save/Snooker yt shorts/nWindows/"
    "Jack Lisowski vs Mark Selby - 2026 Quarter-Final.mp4"
)


def read_json(root: Path, relative: str) -> tuple[dict[str, Any], str]:
    path = root / relative
    payload = path.read_bytes()
    provenance = f"{relative}#sha256={hashlib.sha256(payload).hexdigest()}"
    return json.loads(payload), provenance


def local_path(root: Path, path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def fingerprint(root: Path, path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    stat = resolved.stat()
    return {"basename": resolved.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "note": "Local file identity, not a cryptographic video-content hash"}


def classify_behavior(description: str) -> str | None:
    description = description.lower()
    if "replay" in description:
        return "replay"
    if any(word in description for word in ("referee", "respot", "replacement")):
        return "handling"
    return None


def live_contact_replay_negatives(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer non-replay only within a reviewed live-contact uncertainty bracket.

    This does not imply that adjacent footage is live or that the referee is
    absent. Preserve the original contact provenance and disclose the inference.
    """
    return [{"head": "replay", "start": contact["lower"], "end": contact["upper"],
             "value": 0, "weight": contact.get("weight", 1),
             "provenance": "derived_from_reviewed_live_contact:" + contact["provenance"],
             "description": "Reviewed live cue contact: " + contact.get("description", ""),
             "inference": "A reviewed live contact implies non-replay footage inside this "
                          "uncertainty bracket only; adjacent time and handling remain unknown"}
            for contact in contacts]


def preferred_clip(row: dict[str, Any], provenance: str,
                   contact: dict[str, Any] | None = None,
                   clean: bool = False) -> dict[str, Any]:
    result = {"start": float(row["clip_start"]), "end": float(row["clip_end"]),
              "weight": 0.35,
              "provenance": "classic_generated_source_output_reviewed:" + provenance,
              "confirmed_replay_free": clean, "confirmed_handling_free": clean}
    if contact:
        result.update(contact_lower=contact["lower"], contact_upper=contact["upper"])
    return result


def incorporate_visual_review(video: dict[str, Any], annotation: dict[str, Any],
                              provenance: str) -> None:
    """Apply a frozen DL-only source review; classic benchmark labels stay intact."""
    if annotation["source_group_id"] != video["group_id"] or (
        Path(video["source_path"]).name != annotation["source_basename"]
    ):
        raise ValueError("Visual review describes a different source or underlying match")
    for adjudication in annotation.get("contact_adjudications", []):
        original = adjudication["original_bounds"]
        matches = [contact for contact in video["contacts"]
                   if contact["lower"] == original["lower"] and contact["upper"] == original["upper"]
                   and contact["description"] == adjudication["description"]]
        if len(matches) != 1:
            raise ValueError("Source adjudication does not match exactly one frozen contact")
        contact = matches[0]
        contact["original_bounds"] = dict(original)
        contact.update(adjudication["new_bounds"])
        contact["timing_provenance"] = "ai_source_visual_adjudication:" + provenance
        contact["adjudication"] = dict(adjudication)
    for clip in annotation.get("clips", []):
        if not any(contact["lower"] == clip["contact_lower"]
                   and contact["upper"] == clip["contact_upper"] for contact in video["contacts"]):
            raise ValueError("Reviewed editorial clip is not tied to an existing source contact")
        video["clips"].append({**clip, "provenance": "ai_source_visual_editorial_review:" + provenance})
    video["clip_reviewed_intervals"].extend(
        {**row, "provenance": "ai_source_visual_editorial_review:" + provenance}
        for row in annotation.get("clip_reviewed_intervals", []))
    video["labels"].extend({**row, "provenance": "ai_source_visual_behavior_review:" + provenance}
                           for row in annotation.get("labels", []))
    video["additional_review"] = {"provenance": provenance, "human_approved": False,
                                  "reviewer_consulted_model_predictions": False,
                                  "independent_holdout": False,
                                  "reviewed_frame_count": annotation["reviewed_frame_count"]}
    video["limitations"] = [note for note in video.get("limitations", [])
                             if "No independently annotated editorial endpoints" not in note]
    video["limitations"].append("Twelve AI-reviewed editorial intervals; other boundaries remain unknown; not human approved")


def build_manifest(root: Path, selby_source: Path) -> dict[str, Any]:
    latest, latest_provenance = read_json(root,
        "data/editor_audit/full_frame_2026_10_07/reference.json")
    final, final_provenance = read_json(root,
        "data/editor_audit/full_frame_2026_10_07/final_verification.json")
    job, _ = read_json(root, f"data/jobs/{latest['source_job']}/job.json")
    duration = float(latest["source_duration"])
    latest_video: dict[str, Any] = {
        "id": "zhao_trump_full_frame_oct07", "group_id": ZHAO_GROUP, "split": "train",
        "source_path": local_path(root, job["source_path"]), "duration": duration,
        "features_path": "data/dl/features/zhao_trump_full_frame_oct07.npz",
        "source_fingerprint": fingerprint(root, job["source_path"]),
        "reviewed_intervals": [{"start": 0.0, "end": duration, "provenance": latest_provenance}],
        "clip_reviewed_intervals": [{"start": 0.0, "end": duration, "weight": 0.35,
                                     "provenance": final_provenance}],
        "contacts": [{**contact, "provenance": latest_provenance}
                     for contact in latest["contacts"]],
        "clips": [preferred_clip(row, final_provenance, row, clean=True)
                  for row in final["contacts"]],
        "labels": [], "annotation_method": latest["method"],
        "classic_baseline_job": final["summary"]["corrected_job"],
        "limitations": ["Development footage; failures and predictions informed timing revisions",
                        "Many contact brackets are broad overview bounds",
                        "Clip preferences came from the classic editor and subsequent visual review"],
    }
    for row in latest["negative_sequences"]:
        head = classify_behavior(row["description"])
        if head:
            latest_video["labels"].append({"head": head, "start": row["lower"],
                "end": row["upper"], "value": 1, "weight": 1,
                "provenance": latest_provenance, "description": row["description"]})

    earlier, earlier_provenance = read_json(root,
        "data/editor_audit/zhao_trump_2026_10_06/reference.json")
    earlier_clips, earlier_clip_provenance = read_json(root,
        "data/editor_audit/zhao_trump_2026_10_06/final_clip_qa.json")
    earlier_source = root / "data/uploads" / earlier["source_basename"]
    earlier_video = {
        "id": "zhao_trump_partial_frame_oct05", "group_id": ZHAO_GROUP, "split": "train",
        "source_path": local_path(root, earlier_source), "duration": 797.386,
        "features_path": "data/dl/features/zhao_trump_partial_frame_oct05.npz",
        "source_fingerprint": fingerprint(root, earlier_source),
        "reviewed_intervals": [{"start": row["start"], "end": row["end"],
                                 "provenance": earlier_provenance} for row in earlier["windows"]],
        "clip_reviewed_intervals": [{"start": row["start"], "end": row["end"],
                                      "weight": 0.35, "provenance": earlier_clip_provenance}
                                     for row in earlier["windows"]],
        "contacts": [{**contact, "provenance": earlier_provenance}
                     for window in earlier["windows"] for contact in window["contacts"]],
        "clips": [preferred_clip(row, earlier_clip_provenance, row["contact"])
                  for row in earlier_clips["contacts"]],
        "labels": [], "annotation_method": earlier["method"],
        "classic_baseline_job": "20261007-074546-37c59899",
        "limitations": ["Only six source intervals are exhaustively contact-labelled",
                        "Selected around known failures; unreviewed source remains masked",
                        "Clip preferences are lower-weight reviewed classic outputs"],
    }

    followup, followup_provenance = read_json(root,
        "data/editor_audit/referee_followup_2026_10_07/final_verification.json")
    followup_job, _ = read_json(root, f"data/jobs/{followup['job_id']}/job.json")
    handling = [{"start": left, "end": right, "provenance": followup_provenance}
                for left, right in followup["excluded_handling_intervals"]]
    followup_clips = [{"start": row["source_clip"][0], "end": row["source_clip"][1],
                       "weight": 0.35,
                       "provenance": "classic_generated_source_output_reviewed:" + followup_provenance,
                       "confirmed_replay_free": True, "confirmed_handling_free": True}
                      for row in followup["reported_contacts"]]
    followup_video = {
        "id": "zhao_trump_referee_oct07", "group_id": ZHAO_GROUP, "split": "train",
        "source_path": local_path(root, followup_job["source_path"]), "duration": 1229.8,
        "features_path": "data/dl/features/zhao_trump_referee_oct07.npz",
        "source_fingerprint": fingerprint(root, followup_job["source_path"]),
        # Behaviour review proves handling, not the absence of a simultaneous
        # stroke. Only exhaustively contact-reviewed intervals label event
        # negatives; these examples supervise the behaviour/edit heads only.
        "reviewed_intervals": [],
        "clip_reviewed_intervals": [],
        "contacts": [], "clips": followup_clips,
        "labels": [{**row, "head": "handling", "value": 1, "weight": 1} for row in handling],
        "classic_baseline_job": followup["job_id"],
        "limitations": ["No contact bags invented from algorithm-estimated strike timestamps",
                        "Four reviewed handling spans and three weak clips; no exhaustive contact review",
                        "All source contact labels are unknown; handling does not imply no stroke"],
    }

    benchmark, benchmark_provenance = read_json(root, "benchmarks/selby_lisowski_recall.json")
    # The existing benchmark carries original reviewed splits as provenance. All
    # intervals of this other match stay together for neural model selection.
    selby_video = {
        "id": "selby_lisowski_benchmark", "group_id": SELBY_GROUP, "split": "dev",
        "source_path": local_path(root, selby_source),
        "duration": max(float(row["end"]) for row in benchmark["windows"]),
        "features_path": "data/dl/features/selby_lisowski_benchmark.npz",
        "source_fingerprint": fingerprint(root, selby_source),
        "reviewed_intervals": [{"start": window["start"], "end": window["end"],
                                  "provenance": benchmark_provenance,
                                  "original_classic_split": window["split"]}
                                 for window in benchmark["windows"]],
        "contacts": [{**contact, "provenance": benchmark_provenance}
                     for window in benchmark["windows"] for contact in window["contacts"]],
        "clip_reviewed_intervals": [], "clips": [], "labels": [],
        "annotation_method": benchmark["method"],
        "limitations": ["Contact-reviewed intervals only; source outside them is unknown",
                        "Used as neural validation, so no longer a neural final holdout",
                        "No independently annotated editorial endpoints or physical stops"],
    }
    # Probe duration instead of shortening the source to the last annotated time.
    from snooker_ai.ingestion.probe import probe_video
    selby_video["duration"] = probe_video(selby_source).duration
    for window in benchmark["windows"]:
        for row in window.get("negative_examples", []):
            head = classify_behavior(row["description"])
            if head:
                selby_video["labels"].append({**row, "head": head, "value": 1, "weight": 1,
                                             "provenance": benchmark_provenance})
    visual_review, visual_provenance = read_json(root,
        "data/evaluation/dl/selby_opening_editorial_review.json")
    incorporate_visual_review(selby_video, visual_review, visual_provenance)
    for video in (latest_video, earlier_video, selby_video):
        video["labels"].extend(live_contact_replay_negatives(video["contacts"]))
    manifest = {
        "manifest_version": MANIFEST_VERSION, "heads": list(HEADS),
        "paths_relative_to": "../../..",
        "feature_contract": {"timestamps": "float64[T] source presentation seconds",
                             "features": "float32[T,D] frozen RGB+learned motion features",
                             "feature_spec": "JSON scalar; no classic detection features"},
        "label_policy": "Only explicitly reviewed intervals supervise negatives; contacts use uncertainty bags; "
                        "non-replay is inferred only inside reviewed live-contact brackets",
        "split_policy": "Underlying match group isolation, including different cleaned frames and replays",
        "independent_holdout_available": False,
        "limitations": ["Only two independently sourced underlying matches have reviewed contacts",
                        "Zhao–Trump trains; Selby–Lisowski selects hyperparameters",
                        "A third unseen match with frozen independent source labels is required for a final holdout",
                        "Neural superiority over classic cannot be claimed from this dataset"],
        "videos": [latest_video, earlier_video, followup_video, selby_video],
    }
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/dl"))
    parser.add_argument("--selby-source", type=Path, default=SELBY_SOURCE)
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.selby_source)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split, content in split_manifest(manifest).items():
        (output / f"{split}.json").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "videos": len(manifest["videos"]),
                      "contacts": sum(len(video["contacts"]) for video in manifest["videos"]),
                      "independent_holdout_available": False}))


if __name__ == "__main__":
    main()
