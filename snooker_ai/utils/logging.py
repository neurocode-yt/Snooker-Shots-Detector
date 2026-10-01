"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False


def setup_logging(level: str = "INFO", name: str = "snooker_ai") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        # PowerShell/redirection can expose a cp1252 console. Unicode arrows
        # and international filenames must never generate logging tracebacks.
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("snooker_ai")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name and not name.startswith("snooker_ai"):
        name = f"snooker_ai.{name}"
    return logging.getLogger(name or "snooker_ai")
