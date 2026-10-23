"""Three checksum-verified local recovery generations for progress and edits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import threading
import time
import uuid

from snooker_ai.utils.persistence import RecoverySaveError, set_publish_guard

_store = None
_DATA = {".json", ".npz", ".csv", ".edl", ".yaml", ".yml"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value):
    if not isinstance(value, str):
        raise ValueError("Invalid recovery file path")
    path = PurePosixPath(value)
    if ("\\" in value or path.is_absolute()
            or any(part in {".", "..", ""} or part.startswith(".") for part in path.parts)
            or not path.parts or path.parts[0] not in {"jobs", "uploads"}):
        raise ValueError("Invalid backup file path")
    return path


class RecoveryStore:
    def __init__(self, root, directory):
        self.root = Path(root).resolve()
        self.directory = Path(directory).resolve()
        self.objects = self.directory / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory / "index.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, versions TEXT)")
        self.db.commit()
        self.last_saved = 0.
        self.error = ""
        # Startup happens before processing starts. Discard incomplete objects
        # and recovery history belonging to deliberately removed projects.
        for relative, _ in self.rows():
            if not (self.root / relative).parent.exists():
                self.db.execute("DELETE FROM files WHERE path=?", (relative,))
        self.db.commit()
        retained = {item["sha256"] for _, versions in self.rows() for item in versions}
        for path in self.objects.iterdir():
            if path.is_file() and path.name not in retained:
                path.unlink()

    def relative(self, path):
        try:
            relative = Path(path).resolve().relative_to(self.root)
            _safe_relative(relative.as_posix())
            return relative.as_posix()
        except ValueError:
            return None

    def guarded(self, path):
        relative = self.relative(path)
        return (relative is not None and Path(path).suffix.lower() in _DATA
                and Path(path).name not in {"queued.json"})

    def publish(self, path, temporary):
        if not self.guarded(path):
            return
        try:
            self.capture(self.relative(path), temporary)
        except Exception as error:
            self.error = "Local recovery save failed: " + str(error)
            raise RecoverySaveError(self.error) from error

    def capture(self, relative, source):
        source = Path(source)
        before = source.stat()
        temporary = self.objects / ("." + uuid.uuid4().hex)
        digest = hashlib.sha256()
        try:
            with source.open("rb") as stream, temporary.open("wb") as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("File changed during backup")
            identity = digest.hexdigest()
            record = {"sha256": identity, "size": before.st_size, "mtime_ns": before.st_mtime_ns}
            with self.lock:
                blob = self.objects / identity
                if not blob.exists():
                    temporary.replace(blob)
                elif digest_file(blob) != identity:
                    temporary.replace(blob)
                _sync_directory(self.objects)
                row = self.db.execute("SELECT versions FROM files WHERE path=?", (relative,)).fetchone()
                previous = json.loads(row[0]) if row else []
                versions = [record] + [item for item in previous if item["sha256"] != identity][:2]
                self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?)", (relative, json.dumps(versions)))
                self.db.commit()
                # Only three generations per file. Delete superseded objects
                # under the same lock so concurrent saves cannot race cleanup.
                retained = {v["sha256"] for _, items in self.rows() for v in items}
                for old in previous:
                    if old["sha256"] not in retained:
                        (self.objects / old["sha256"]).unlink(missing_ok=True)
                self.last_saved = time.time()
                self.error = ""
            return record
        finally:
            temporary.unlink(missing_ok=True)

    def rows(self):
        with self.lock:
            return [(path, json.loads(versions)) for path, versions in self.db.execute(
                "SELECT path, versions FROM files ORDER BY path").fetchall()]

    def recover(self):
        repaired = []
        for relative, versions in self.rows():
            path = self.root / relative
            # Explicitly deleted projects/caches stay deleted. Recovery repairs
            # files inside existing project directories, never resurrects jobs.
            if not path.parent.exists():
                continue
            if path.name == "coarse_partial.npz" and (path.parent / "coarse_checkpoint.json").exists():
                continue
            for item in versions:
                blob = self.objects / item["sha256"]
                if not blob.exists() or digest_file(blob) != item["sha256"]:
                    continue
                if not path.is_file() or digest_file(path) != item["sha256"]:
                    _write(path, blob.read_bytes())
                    os.utime(path, ns=(item["mtime_ns"], item["mtime_ns"]))
                    repaired.append(relative)
                break
            else:
                self.error = "No valid recovery copy for " + relative
        return repaired

    def seed(self):
        known = {path: versions[0] for path, versions in self.rows()}
        for path in (self.root / "jobs").rglob("*"):
            if not path.is_file() or not self.guarded(path):
                continue
            relative = self.relative(path)
            prior = known.get(relative)
            stat = path.stat()
            if prior and (stat.st_size, stat.st_mtime_ns) == (prior["size"], prior["mtime_ns"]):
                continue
            if path.suffix == ".json":
                json.loads(path.read_text("utf-8"))
            self.capture(relative, path)

    def status(self):
        return {"protected_files": len(self.rows()), "last_saved": self.last_saved, "error": self.error}


def configure(root):
    global _store
    root = Path(root).resolve()
    if _store is not None and _store.root == root:
        return _store
    _store = RecoveryStore(root, root.with_name(root.name + "-recovery"))
    _store.recover()
    _store.seed()
    set_publish_guard(_store.publish)
    return _store


def status():
    return json.dumps(_store.status() if _store else {"protected_files": 0, "error": "Starting protection"})

