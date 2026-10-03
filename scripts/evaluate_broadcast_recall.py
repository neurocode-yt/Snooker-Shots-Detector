"""Compare an analysis/export JSON with complete bounded source references."""

import argparse
import json
from pathlib import Path

from snooker_ai.evaluation.recall import evaluate_broadcast_recall
from snooker_ai.types import ExportRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=.5)
    parser.add_argument("--delivered", action="store_true", help="Count every listed clip as delivered")
    parser.add_argument("--include-replays", action="store_true")
    parser.add_argument("--all-shots", action="store_true", help="Export non-included live records too")
    parser.add_argument("--min-confidence", type=float, default=0.)
    parser.add_argument("--min-importance", type=float, default=0.)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    report = evaluate_broadcast_recall(
        analysis["shots"], reference, args.tolerance,
        delivered=args.delivered or "export_accurate" in analysis,
        export_request=ExportRequest(include_replays=args.include_replays,
                                     only_included=not args.all_shots,
                                     min_confidence=args.min_confidence,
                                     min_importance=args.min_importance),
    )
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
