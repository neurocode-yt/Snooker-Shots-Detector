"""OS-owned job locks are released automatically when a process stops."""

from __future__ import annotations

import os
from pathlib import Path


class JobLease:
    def __init__(self, job_dir: Path):
        self.path = Path(job_dir) / ".processing.lock"
        self.stream = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("This job is already processing")
        return self

    def __exit__(self, *args):
        self.release()

    @classmethod
    def is_running(cls, job_dir: Path) -> bool:
        lease = cls(job_dir)
        acquired = lease.acquire()
        lease.release()
        return not acquired
