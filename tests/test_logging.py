"""Windows legacy console encodings must not turn status messages into errors."""

import io
import logging

import pytest

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


def access_record(path, method="GET", status=200, level=logging.INFO):
    return logging.LogRecord(
        "uvicorn.access", level, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:12345", method, path, "1.1", status), None,
    )


@pytest.mark.parametrize("path", [
    "/api/work", "/api/jobs", "/api/queue", "/api/capcut",
    "/api/jobs/match-1", "/api/jobs/match-1/progress",
    "/api/jobs/match-1/shorts", "/api/jobs/match-1/shorts/batch-1",
    "/api/jobs?order=recent",
])
def test_successful_status_polls_are_quiet(path):
    assert not logging_module.StatusAccessFilter().filter(access_record(path))


@pytest.mark.parametrize("record", [
    access_record("/api/work", status=500),
    access_record("/api/jobs/missing", status=404),
    access_record("/api/work", level=logging.ERROR),
    access_record("/api/jobs", method="POST"),
    access_record("/api/queue", method="PUT"),
    access_record("/api/jobs/match-1/pause", method="POST"),
    access_record("/api/jobs/match-1/shorts/batch-1/zip"),
    access_record("/api/jobs/match-1/shorts/batch-1/clips/1"),
    access_record("/api/jobs/match-1/video", status=206),
    access_record("/api/jobs/match-1/shots"),
    access_record("/"),
    logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "ordinary message", (), None),
])
def test_errors_actions_media_and_other_logs_remain_visible(record):
    assert logging_module.StatusAccessFilter().filter(record)


def test_status_filter_installed_once_even_when_logging_already_configured(monkeypatch):
    access = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(access, "filters", [])
    monkeypatch.setattr(logging_module, "_CONFIGURED", True)
    logging_module.setup_logging()
    logging_module.setup_logging()
    assert len(access.filters) == 1
    assert not access.filters[0].filter(access_record("/api/work"))
