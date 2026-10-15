"""Structured logging setup."""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional


_CONFIGURED = False


class StatusAccessFilter(logging.Filter):
    """Keep errors and actions visible without printing successful UI polls."""

    _status_path = re.compile(
        r"^/api/(?:work|jobs|queue|capcut|jobs/[^/]+(?:/progress|/shorts(?:/[^/]+)?)?)$"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access records use (client, method, path, HTTP version, status).
        # Other record formats and warning/error records must pass through.
        args = record.args
        if record.levelno >= logging.WARNING or not isinstance(args, tuple) or len(args) != 5:
            return True
        _, method, path, _, status = args
        return not (
            method == "GET" and status == 200 and isinstance(path, str)
            and self._status_path.fullmatch(path.split("?", 1)[0])
        )


def setup_logging(level: str = "INFO", name: str = "snooker_ai") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, StatusAccessFilter) for item in access_logger.filters):
        access_logger.addFilter(StatusAccessFilter())
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
