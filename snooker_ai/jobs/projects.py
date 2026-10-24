"""Names and resource ownership for source families in the video library."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from snooker_ai.jobs.lease import JobLease
from snooker_ai.utils.persistence import atomic_text


class ProjectConflict(ValueError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _path(value):
    text = str(value).replace("\\", "/")
    return text.lower() if re.match(r"^[a-z]:/", text, re.I) or text.startswith("//") else text


def families(jobs):
    """Match library.js, including legacy jobs and replaced source files."""
    revisions = {}
    records = []
    for job in jobs:
        fp = job.get("source_fingerprint") or {}
        source = f"{job.get('algorithm', 'classic')}:{_path(fp.get('path') or job.get('source_path') or 'job:' + job['job_id'])}"
        revision = (
            f"{fp['size']}:{fp['mtime_ns']}"
            if fp.get("size") is not None and fp.get("mtime_ns") is not None
            else None
        )
        revisions.setdefault(source, set())
        if revision:
            revisions[source].add(revision)
        records.append((job, source, revision))
    groups = {}
    for job, source, revision in records:
        known = revisions[source]
        revision = revision or (
            next(iter(known))
            if len(known) == 1
            else f"unverified:{job['job_id']}"
            if known
            else "legacy"
        )
        key = hashlib.sha256(json.dumps([source, revision]).encode()).hexdigest()
        groups.setdefault(key, []).append(job)
    return groups


def _contained(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    if (
        path.resolve() == root
        or not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or getattr(path, "is_junction", lambda: False)()
    ):
        raise ProjectConflict(f"Resource is outside its managed directory: {path.name}")
    return path


def capcut_running():
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq CapCut.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=True,
    )
    return '"capcut.exe"' in result.stdout.lower()


def _restore_index_entries(originals, selected_ids):
    """Undo our removals without overwriting another project's new entries."""
    for index, text in originals.items():
        previous = json.loads(text)
        current = _read(index)
        present = {entry.get("draft_id") for entry in current["all_draft_store"]}
        restored = [
            entry
            for entry in previous["all_draft_store"]
            if entry.get("draft_id") in selected_ids.get(index, [])
            and entry.get("draft_id") not in present
        ]
        if restored:
            current["all_draft_store"] = restored + current["all_draft_store"]
            current["draft_ids"] = len(current["all_draft_store"])
            atomic_text(Path(index), json.dumps(current, ensure_ascii=False, indent=2))


class ProjectLibrary:
    def __init__(self, store):
        self.store = store
        self.names_path = store.root / "project-names.json"

    def names(self):
        return _read(self.names_path) if self.names_path.exists() else {}

    def annotate(self, jobs):
        names = self.names()
        return [
            {**job, "project_id": key, "display_name": self._name(names, key, group)}
            for key, group in families(jobs).items()
            for job in group
        ]

    @staticmethod
    def _name(names, key, group):
        return names.get(key) or next(
            (names.get("job:" + j["job_id"]) for j in group if names.get("job:" + j["job_id"])), ""
        )

    def info(self, job_id):
        return next(
            (job for job in self.annotate(self.store.list_jobs()) if job["job_id"] == job_id), None
        )

    def select(self, job_ids):
        jobs = self.store.list_jobs()
        groups = families(jobs)
        requested = set(job_ids)
        if not requested or len(requested) > 500:
            raise ValueError("Select between 1 and 500 projects")
        if requested - {job["job_id"] for job in jobs}:
            raise ProjectConflict("A selected project no longer exists. Refresh the library.")
        selected = {
            key: group
            for key, group in groups.items()
            if any(j["job_id"] in requested for j in group)
        }
        return jobs, selected

    def rename(self, job_id, name):
        name = name.strip()
        if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ValueError(
                "Enter a project name between 1 and 120 characters, without control characters"
            )
        _, selected = self.select([job_id])
        names = self.names()
        key = next(iter(selected))
        names[key] = name
        for job in selected[key]:
            names["job:" + job["job_id"]] = name
        atomic_text(self.names_path, json.dumps(names, ensure_ascii=False, indent=2))
        return {"project_id": key, "display_name": name}

    def capcut_roots(self, jobs):
        cfg = self.store.config
        roots = set()
        configured = cfg.get("capcut.projects_dir")
        if configured:
            roots.add(Path(configured).expanduser().absolute())
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.add(Path(local) / "CapCut/User Data/Projects/com.lveditor.draft")
        cache = Path(cfg.get("paths.data_dir", "data")) / "capcut_installation.json"
        if cache.exists():
            roots.add(Path(_read(cache)["projects_dir"]))
        for job in jobs:
            receipt = self.store.job_dir(job["job_id"]) / "export/capcut_project.json"
            if receipt.exists():
                roots.add(Path(_read(receipt)["project_path"]).parent)
        # Resolve aliases once; never scan the same moved AppData twice.
        return sorted({p.resolve() for p in roots if p.is_dir()}, key=str)

    def plan(self, job_ids):
        jobs, groups = self.select(job_ids)
        names = self.names()
        selected = [job for group in groups.values() for job in group]
        ids = {j["job_id"] for j in selected}
        directories = [_contained(self.store.job_dir(j), self.store.root) for j in sorted(ids)]
        sources = {Path(j["source_path"]).resolve() for j in selected if j.get("source_path")}
        remaining = [j for j in jobs if j["job_id"] not in ids]
        shared = {Path(j["source_path"]).resolve() for j in remaining if j.get("source_path")}
        # Saved preprocessing receipts bind a cleaned copy to its uploaded input.
        for source in list(sources):
            receipt = source.with_suffix(source.suffix + ".source.json")
            if receipt.is_file():
                sources.add(Path(_read(receipt)["source_path"]).resolve())
        for job in remaining:
            source = Path(job["source_path"]).resolve()
            receipt = source.with_suffix(source.suffix + ".source.json")
            if receipt.is_file():
                shared.add(Path(_read(receipt)["source_path"]).resolve())
            timeline = self.store.job_dir(job["job_id"]) / "timeline.json"
            proxy = _read(timeline).get("proxy_path") if timeline.is_file() else None
            if proxy and any(
                Path(proxy).resolve().is_relative_to(d.resolve()) for d in directories
            ):
                raise ProjectConflict(
                    "Another project uses this project's proxy. Select both projects to delete them."
                )
        capcut = []
        indexes = {}
        for root in self.capcut_roots(jobs):
            index = root / "root_meta_info.json"
            data = _read(index) if index.exists() else {"all_draft_store": [], "draft_ids": 0}
            if not isinstance(data.get("all_draft_store"), list):
                raise ProjectConflict(
                    "CapCut's project list is unreadable. Repair it before deleting projects."
                )
            indexes[str(index)] = data
            for directory in root.iterdir():
                if not directory.is_dir():
                    continue
                marker = directory / "snooker_shots.json"
                own = _read(marker) if marker.is_file() else {}
                meta_path = directory / "draft_meta_info.json"
                meta = _read(meta_path) if meta_path.is_file() else {}
                if own.get("job_id") in ids:
                    _contained(directory, root)
                    if (
                        not meta.get("draft_id")
                        or Path(own.get("source", "")).resolve() not in sources
                    ):
                        raise ProjectConflict(
                            "A linked CapCut project's ownership could not be verified."
                        )
                    capcut.append(
                        {"path": str(directory), "id": meta["draft_id"], "index": str(index)}
                    )
                else:
                    if own.get("source"):
                        shared.add(Path(own["source"]).resolve())
                    for material in meta.get("draft_materials", []):
                        for item in material.get("value", []):
                            if item.get("file_Path"):
                                shared.add(Path(item["file_Path"]).resolve())
        # Remove only managed uploads. External originals remain user-owned.
        uploads = []
        kept = []
        for source in sorted(sources, key=str):
            if not source.exists():
                continue
            if source in shared or not source.is_relative_to(self.store.uploads.resolve()):
                kept.append(str(source))
            else:
                uploads.append(str(_contained(source, self.store.uploads)))
                receipt = source.with_suffix(source.suffix + ".source.json")
                if receipt.exists():
                    uploads.append(str(_contained(receipt, self.store.uploads)))
        issues = []
        if capcut and capcut_running():
            issues.append(
                "Close CapCut, then check deletion again so it cannot restore its project list."
            )
        for directory in directories:
            if any(
                JobLease.is_running(directory, n)
                for n in (".processing.lock", ".queue.lock", ".capcut.lock")
            ) or JobLease.is_running(directory / "shorts", ".shorts.lock"):
                issues.append("Pause active processing and Shorts, then check deletion again.")
                break
        titles = []
        for key, group in groups.items():
            stem = str(group[0]["source_path"]).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
            title = self._name(names, key, group) or re.sub(
                r"^\d{8}-\d{6}_[a-f0-9]{8,32}_", "", stem, flags=re.I
            ).replace("_", " ")
            titles.append(title)
        plan = {
            "project_ids": sorted(groups),
            "job_ids": sorted(ids),
            "project_names": titles,
            "job_directories": [str(d) for d in directories],
            "uploads": uploads,
            "capcut": capcut,
            "kept_sources": kept,
            "issues": issues,
        }
        # Review binds membership, index contents and saved edits to confirmation.
        identities = {
            str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for d in directories
            for p in (d / "job.json", d / "timeline.json")
            if p.exists()
        }
        plan["token"] = hashlib.sha256(
            json.dumps([plan, indexes, identities], sort_keys=True).encode()
        ).hexdigest()
        plan["indexes"] = indexes
        return plan

    def delete(self, plan, confirmed_token):
        if plan["token"] != confirmed_token:
            raise ProjectConflict(
                "The selected projects changed. Check the deletion details again."
            )
        if plan["issues"]:
            raise ProjectConflict(plan["issues"][0])
        if plan["capcut"] and capcut_running():
            raise ProjectConflict("Close CapCut before deleting its linked projects.")
        ticket = uuid.uuid4().hex
        journal = self.store.root / ".deletions" / f"{ticket}.json"
        moved = []
        paths = plan["uploads"] + [p["path"] for p in plan["capcut"]] + plan["job_directories"]
        moves = [
            [str(Path(p)), str(Path(p).with_name(f".snooker-delete-{ticket}-{Path(p).name}"))]
            for p in paths
        ]
        originals = {
            index: Path(index).read_text(encoding="utf-8-sig")
            for index in plan["indexes"]
            if any(p["index"] == index for p in plan["capcut"])
        }
        names_before = self.names()
        recovery = {
            "moves": moves,
            "indexes": originals,
            "names": names_before,
            "job_ids": plan["job_ids"],
            "restore_ids": {
                index: [p["id"] for p in plan["capcut"] if p["index"] == index]
                for index in originals
            },
            "state": "preparing",
            "created": time.time(),
        }
        atomic_text(journal, json.dumps(recovery))
        try:
            for original, staged in moves:
                path, target = Path(original), Path(staged)
                _contained(path, path.parent)
                path.rename(target)
                moved.append([str(path), str(target)])
            for index, data in plan["indexes"].items():
                removed = {p["id"] for p in plan["capcut"] if p["index"] == index}
                if not removed:
                    continue
                path = Path(index)
                if _read(path) != data:
                    raise ProjectConflict(
                        "CapCut's project list changed. Check the deletion details again."
                    )
                data = dict(data)
                data["all_draft_store"] = [
                    e for e in data["all_draft_store"] if e.get("draft_id") not in removed
                ]
                data["draft_ids"] = len(data["all_draft_store"])
                atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2))
            names = dict(names_before)
            for key in plan["project_ids"]:
                names.pop(key, None)
            for job_id in plan["job_ids"]:
                names.pop("job:" + job_id, None)
            atomic_text(self.names_path, json.dumps(names, ensure_ascii=False, indent=2))
            atomic_text(journal, json.dumps({**recovery, "state": "committed"}))
        except BaseException:
            _restore_index_entries(originals, recovery["restore_ids"])
            for original, staged in reversed(moved):
                Path(staged).rename(original)
            atomic_text(self.names_path, json.dumps(names_before, ensure_ascii=False))
            journal.unlink(missing_ok=True)
            raise
        pending = self.cleanup(journal)
        return {
            "deleted_projects": len(plan["project_ids"]),
            "deleted_versions": len(plan["job_ids"]),
            "capcut_projects": len(plan["capcut"]),
            "kept_sources": plan["kept_sources"],
            "cleanup_pending": pending,
        }

    def cleanup(self, journal_only=None):
        pending = 0
        journals = (
            [journal_only] if journal_only else (self.store.root / ".deletions").glob("*.json")
        )
        for journal in journals:
            data = _read(journal)
            if data["state"] == "preparing":
                if data["indexes"] and capcut_running():
                    raise ProjectConflict(
                        "Close CapCut before recovering the interrupted project deletion."
                    )
                for original, staged in reversed(data["moves"]):
                    if Path(staged).exists():
                        Path(staged).rename(original)
                from snooker_ai.rendering.capcut import capcut_index_lease

                with ExitStack() as stack:
                    for index in sorted(data["indexes"]):
                        lease = capcut_index_lease(Path(index).parent)
                        if not lease.acquire():
                            raise ProjectConflict(
                                "CapCut's project list is being updated. Retry cleanup shortly."
                            )
                        stack.callback(lease.release)
                    _restore_index_entries(data["indexes"], data.get("restore_ids", {}))
                atomic_text(self.names_path, json.dumps(data["names"], ensure_ascii=False))
                for job_id in data["job_ids"]:
                    (self.store.root / ".deleting" / job_id).unlink(missing_ok=True)
                journal.unlink()
                continue
            remaining = []
            for original, value in data["moves"]:
                path = Path(value)
                if (
                    not path.name.startswith(".snooker-delete-")
                    or path.parent != Path(original).parent
                ):
                    raise ProjectConflict("Invalid cleanup journal")
                try:
                    _contained(path, Path(original).parent)
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    remaining.append(value)
            if remaining:
                atomic_text(
                    journal,
                    json.dumps({**data, "moves": [m for m in data["moves"] if m[1] in remaining]}),
                )
                pending += len(remaining)
            else:
                for job_id in data["job_ids"]:
                    (self.store.root / ".deleting" / job_id).unlink(missing_ok=True)
                journal.unlink()
        for marker in (self.store.root / ".deleting").glob("*"):
            marker.unlink(missing_ok=True)
        return pending
