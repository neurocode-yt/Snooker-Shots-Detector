"""Separate detection algorithms; editing mode remains an independent setting."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from snooker_ai.config import Config


class Algorithm(str, Enum):
    CLASSIC = "classic"
    DL_ALGO = "dl_algo"

    @classmethod
    def parse(cls, value: str | Algorithm | None) -> Algorithm:
        return cls.CLASSIC if value is None else cls(value)


def algorithm_capabilities(config: Config) -> list[dict[str, Any]]:
    """Inspect readiness without loading weights or substituting another algorithm."""
    classic = {"id": Algorithm.CLASSIC.value, "label": "Current algorithm", "available": True,
               "reason": "Ready"}
    try:
        from snooker_ai.dl.pipeline import DLAnalyzer

        dl = dict(DLAnalyzer.capabilities(config))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        dl = {"available": False, "reason": f"DL Algo is not ready: {exc}"}
    dl.update(id=Algorithm.DL_ALGO.value, label="DL Algo")
    return [classic, dl]


def create_analyzer(config: Config, job_dir: Path, algorithm: str | Algorithm = Algorithm.CLASSIC):
    """Construct only the selected backend. Never silently fall back to classic."""
    selected = Algorithm.parse(algorithm)
    if selected is Algorithm.DL_ALGO:
        from snooker_ai.dl.pipeline import DLAnalyzer

        return DLAnalyzer(config, job_dir)
    from snooker_ai.pipeline.analyzer import Analyzer

    return Analyzer(config, job_dir)
