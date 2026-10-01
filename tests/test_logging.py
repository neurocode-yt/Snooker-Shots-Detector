"""Windows legacy console encodings must not turn status messages into errors."""

import io
import logging

import snooker_ai.utils.logging as logging_module


def test_unicode_status_and_filenames_work_on_cp1252_console(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    root = logging.getLogger("snooker_ai")
    handlers, level, propagate = root.handlers[:], root.level, root.propagate
    monkeypatch.setattr(logging_module, "_CONFIGURED", False)
    monkeypatch.setattr(logging_module.sys, "stdout", stream)
    try:
        logger = logging_module.setup_logging()
        logger.info("Export → 試合.mp4")
        stream.flush()
        assert b"Export" in buffer.getvalue()
        assert b"\\u2192" in buffer.getvalue()
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        root.propagate = propagate
