#!/usr/bin/env python3
"""
Training entrypoint (Phase 2/3 scaffolding).

Phase 1 uses a rule-based temporal state machine. This script validates a
dataset config and prints the training plan for object detection / temporal models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snooker AI training scaffold")
    parser.add_argument("config", type=Path, help="Dataset YAML config")
    parser.add_argument("--task", choices=["detect", "temporal", "view"], default="temporal")
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print("=== Snooker AI Training (scaffold) ===")
    print(f"Task: {args.task}")
    print(f"Config: {args.config}")
    print(f"Root: {cfg.get('root', '.')}")
    print(f"Train split: {cfg.get('splits', {}).get('train')}")
    print(f"Val split:   {cfg.get('splits', {}).get('val')}")
    print(f"Test split:  {cfg.get('splits', {}).get('test')}")
    print()
    print("Phase 1 note: production detector is rule-based (no weights to train).")
    print("Next steps for Phase 2:")
    print("  1. Export corrections from the review UI as training_labels.json")
    print("  2. Annotate table polygons and balls (see annotation/)")
    print("  3. Fine-tune a YOLO-style detector on ball/cue/player classes")
    print("  4. Train a TCN/BiLSTM temporal model on FrameFeatures sequences")
    print("  5. Evaluate with: snooker-ai evaluate ./test-dataset")
    print()
    print("No weights were written (scaffold only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
