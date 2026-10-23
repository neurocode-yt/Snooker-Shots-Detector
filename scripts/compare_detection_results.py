"""Compare saved PC and Android analysis JSON, retaining every stroke and boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snooker_ai.evaluation.parity import compare_shots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=.001)
    args = parser.parse_args()
    reference, candidate = (json.loads(path.read_text(encoding="utf-8-sig"))
                            for path in (args.reference, args.candidate))
    hashes = [data.get("source_sha256") for data in (reference, candidate)]
    if all(hashes) and hashes[0] != hashes[1]:
        raise SystemExit("Cannot compare different source bytes")
    report = compare_shots(reference["shots"], candidate["shots"], tolerance=args.tolerance)
    report.update(reference_file=str(args.reference.resolve()), candidate_file=str(args.candidate.resolve()),
                  source_sha256=hashes[0] if all(hashes) else None,
                  source_identity_verified=all(hashes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "reference_count", "candidate_count", "matched_count", "maximum_error_seconds")}, indent=2))
    print(f"Unmatched reference strokes: {len(report['missing'])}; unmatched candidate strokes: {len(report['extra'])}")
    print(f"Full comparison: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
