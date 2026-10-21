"""Editable, local CapCut Desktop projects built from the retained shot list.

CapCut's private draft schema is version-specific. The bundled neutral schema
matches Windows 9.5 and opens in Windows 9.6. No user's draft, effects, or
account identifiers are copied.
Media stays at its original path so trims can be extended without re-encoding.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snooker_ai.config import Config
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import AnalysisResult, ExportRequest, ShotSelection
from snooker_ai.utils.persistence import atomic_text

_store_lock = threading.Lock()
_SCHEMA = Path(__file__).with_name("assets") / "capcut_9_5.json"
_SUPPORTED_VERSIONS = ("9.5.", "9.6.")
_VERSION_PATTERN = r"\d+(?:\.\d+){2,3}"


class CapCutUnavailable(RuntimeError):
    """The local installation cannot accept the supported draft format."""


@dataclass(frozen=True)
class CapCutInstallation:
    executable: Path
    projects_dir: Path
    version: str


def _registered_local_appdata() -> Path | None:
    """Read the user's Windows folder independently of a terminal's environment."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            saved = winreg.QueryValueEx(key, "Local AppData")[0]
        expanded = re.sub(r"%([^%]+)%", lambda match: os.environ.get(match[1], match[0]), str(saved))
        return Path(expanded)
    except (ImportError, OSError, TypeError, ValueError):
        return None


def _registry_executable(value: str) -> Path | None:
    """Read a registry command/icon path; never execute the registry command."""
    match = re.match(r'^\s*(?:"([^"]+\.exe)"|(.+?\.exe))(?:\s|,|$)',
                     str(value), flags=re.IGNORECASE)
    if match is None:
        return None
    path = match[1] or match[2]
    return Path(re.sub(r"%([^%]+)%", lambda m: os.environ.get(m[1], m[0]), path))


def _capcut_root(path: Path) -> Path:
    if path.suffix.lower() == ".exe":
        path = path.parent
    if re.fullmatch(_VERSION_PATTERN, path.name) and path.parent.name.lower() == "apps":
        path = path.parent
    return path.parent if path.name.lower() == "apps" else path


def _registered_capcut_roots() -> list[Path]:
    """Find moved/custom installs from Windows registration, in both registry views."""
    try:
        import winreg
    except ImportError:
        return []
    roots = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\Uninstall",
                                    0, winreg.KEY_READ | view) as uninstall:
                    for index in range(winreg.QueryInfoKey(uninstall)[0]):
                        try:
                            name = winreg.EnumKey(uninstall, index)
                            with winreg.OpenKey(uninstall, name) as entry:
                                display = str(winreg.QueryValueEx(entry, "DisplayName")[0])
                                if display.strip().casefold() != "capcut":
                                    continue
                                for field in ("InstallLocation", "DisplayIcon", "UninstallString"):
                                    try:
                                        value = str(winreg.QueryValueEx(entry, field)[0]).strip()
                                    except OSError:
                                        continue
                                    if field == "InstallLocation":
                                        expanded = re.sub(r"%([^%]+)%", lambda m: os.environ.get(m[1], m[0]), value)
                                        path = Path(expanded.strip('"')) if value else None
                                    else:
                                        path = _registry_executable(value)
                                    if path is not None:
                                        root = _capcut_root(path)
                                        if root not in roots:
                                            roots.append(root)
                        except OSError:
                            continue
            except OSError:
                continue
    return roots


def _app_path_executables() -> list[Path]:
    """Read Windows' registered application path without launching a command."""
    try:
        import winreg
    except ImportError:
        return []
    paths = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive,
                        r"Software\Microsoft\Windows\CurrentVersion\App Paths\CapCut.exe",
                        0, winreg.KEY_READ | view) as key:
                    path = _registry_executable(str(winreg.QueryValueEx(key, "")[0]))
                if path is not None and path not in paths:
                    paths.append(path)
            except (OSError, KeyError, TypeError, ValueError):
                continue
    return paths


def _installation_cache_path(config: Config) -> Path:
    return Path(config.get("capcut.discovery_cache",
                           str(Path(config.get("paths.data_dir", "data")) / "capcut_installation.json")))


def _saved_executable(config: Config) -> Path | None:
    try:
        data = json.loads(_installation_cache_path(config).read_text("utf-8"))
        path = Path(data["executable"])
        if path.is_absolute() and path.name.casefold() == "capcut.exe":
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def find_capcut(config: Config) -> CapCutInstallation:
    if os.name != "nt":
        raise CapCutUnavailable("Open in CapCut requires the Windows desktop app on this PC.")
    explicit_exe = config.get("capcut.executable_path")
    version_pattern = _VERSION_PATTERN
    candidates = []
    if explicit_exe:
        executable = Path(explicit_exe).expanduser().resolve()
        apps = executable.parent.parent if re.fullmatch(version_pattern, executable.parent.name) else executable.parent
        candidates.append((apps.parent, executable))
    # A stale configured path must not disable all the automatic fallbacks.
    local = os.environ.get("LOCALAPPDATA")
    registered = _registered_local_appdata()
    for base in (Path(local) if local else None, registered, Path.home() / "AppData" / "Local"):
        if base is not None:
            candidates.append(((base / "CapCut").resolve(), None))
    for root in _registered_capcut_roots():
        candidates.append((root.resolve(), None))
    for executable in _app_path_executables():
        candidates.append((_capcut_root(executable).resolve(), executable))
    saved = _saved_executable(config)
    if saved is not None:
        # Re-enumerate its root first so an installed update wins over the old
        # saved version. The actual saved file remains a final direct fallback.
        candidates.extend([(_capcut_root(saved).resolve(), None), (_capcut_root(saved).resolve(), saved)])
    found = None
    for root, configured in candidates:
        apps = root / "Apps"
        if configured is not None and not configured.is_file():
            continue
        if configured is not None and re.fullmatch(version_pattern, configured.parent.name):
            found = root, configured, configured.parent.name
            break
        try:
            versions = [p for p in apps.glob("*") if p.is_dir() and re.fullmatch(version_pattern, p.name)]
        except OSError:
            versions = []
        versions.sort(key=lambda p: tuple(int(n) for n in p.name.split(".")), reverse=True)
        if not versions:
            continue
        # Prefer the real version executable: launchers can be stale or report
        # version 1.0, and incomplete update folders are not installed versions.
        installed = next((p for p in versions if (p / "CapCut.exe").is_file()), None)
        executable = installed / "CapCut.exe" if installed else (configured or apps / "CapCut.exe")
        if executable.is_file():
            version = executable.parent.name if re.fullmatch(version_pattern, executable.parent.name) else versions[0].name
            found = root, executable, version
            break
    if found is None:
        raise CapCutUnavailable(
            "CapCut could not be verified at its saved or registered locations. "
            "Check that its installation drive is available, or set capcut.executable_path to CapCut.exe."
        )
    root, executable, version = found
    if not version.startswith(_SUPPORTED_VERSIONS):
        raise CapCutUnavailable(
            f"CapCut {version} uses an unverified project format. This integration supports Windows 9.5 and 9.6."
        )
    # Respect CapCut's configured location when its local index records one.
    projects = root / "User Data" / "Projects" / "com.lveditor.draft"
    explicit = config.get("capcut.projects_dir")
    if explicit:
        projects = Path(explicit).expanduser()
    else:
        index_path = projects / "root_meta_info.json"
        if index_path.is_file():
            try:
                saved = json.loads(index_path.read_text(encoding="utf-8-sig")).get("root_path")
                if saved and Path(saved).is_dir():
                    projects = Path(saved)
            except (OSError, ValueError, AttributeError):
                pass
    # Keep the path spelling saved by CapCut. Resolving a moved AppData junction
    # makes the app register the same draft again under its original C: alias.
    installation = CapCutInstallation(executable.resolve(), projects.absolute(), version)
    try:
        atomic_text(_installation_cache_path(config), json.dumps({
            "executable": str(installation.executable), "projects_dir": str(installation.projects_dir),
            "version": installation.version,
        }, indent=2))
    except OSError:
        # Discovery is usable even when preferences cannot be saved. The next
        # request still validates the executable; a cache never makes it true.
        pass
    return installation


def _id() -> str:
    return str(uuid.uuid4()).upper()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _register(project: Path, meta: dict[str, Any]) -> None:
    """Append a new draft while preserving and backing up the existing index."""
    index_path = project.parent / "root_meta_info.json"
    entry = {k: v for k, v in meta.items() if k not in ("draft_materials", "draft_materials_copied_info")}
    entry.update(
        draft_json_file=(project / "draft_content.json").as_posix(),
        draft_cover=(project / "draft_cover.jpg").as_posix(),
        draft_timeline_materials_size=meta["draft_timeline_materials_size_"],
        streaming_edit_draft_ready=True,
    )
    entry.pop("draft_timeline_materials_size_", None)
    original = index_path.read_bytes() if index_path.is_file() else None
    try:
        index = json.loads(original.decode("utf-8-sig")) if original else {
            "all_draft_store": [], "draft_ids": 0, "root_path": project.parent.as_posix(),
        }
        if not isinstance(index, dict) or not isinstance(index.get("all_draft_store"), list):
            raise ValueError("unexpected index structure")
        if not isinstance(index.get("draft_ids", 0), int):
            raise ValueError("unexpected draft counter")
    except (ValueError, UnicodeError) as exc:
        raise CapCutUnavailable("CapCut's project index is unreadable; it was left untouched.") from exc
    index["all_draft_store"].insert(0, entry)
    index["draft_ids"] = index.get("draft_ids", 0) + 1
    temporary = index_path.with_name(f".snooker-index-{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, index)
        current = index_path.read_bytes() if index_path.is_file() else None
        if current != original:
            raise CapCutUnavailable("CapCut's project list changed. Try Open in CapCut again.")
        if original is not None:
            backup = index_path.with_name(f"root_meta_info.snooker-{time.time_ns()}.bak")
            backup.write_bytes(original)
        os.replace(temporary, index_path)
    finally:
        temporary.unlink(missing_ok=True)


def create_capcut_project(
    result: AnalysisResult, config: Config, installation: CapCutInstallation,
) -> dict[str, Any]:
    """Create a fresh registered draft; never overwrite an existing CapCut edit."""
    source = Path(result.source_path).resolve()
    if not source.is_file():
        raise ValueError("The original source video is missing. Restore it before opening in CapCut.")
    metadata = result.metadata
    if not all(math.isfinite(v) and v > 0 for v in (metadata.duration, metadata.fps)):
        raise ValueError("The source video needs a valid duration and frame rate.")
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError("The source video needs a valid resolution.")
    selected = Exporter(config)._filter_shots(result.shots, ExportRequest())
    selected.sort(key=lambda shot: (shot.clip_start, shot.shot_id))
    if not selected:
        raise ValueError("No included live shots are selected for CapCut.")
    # CapCut stores microseconds; snap source boundaries to source frames and
    # accumulate integer durations to avoid gaps at fractional frame rates.
    duration_us = round(metadata.duration * 1_000_000)
    ranges = []
    for shot in selected:
        start, end = shot.clip_start, shot.clip_end
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= metadata.duration):
            raise ValueError(f"Shot {shot.shot_id} has invalid source boundaries.")
        start_us = round(round(start * metadata.fps) / metadata.fps * 1_000_000)
        end_us = min(duration_us, round(round(end * metadata.fps) / metadata.fps * 1_000_000))
        if end_us <= start_us:
            raise ValueError(f"Shot {shot.shot_id} is shorter than one source frame.")
        ranges.append((shot, start_us, end_us - start_us))
    template = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    content = copy.deepcopy(template["content"])
    project_id, timeline_id, media_id, local_id = (_id() for _ in range(4))
    display_name = re.sub(r"^\d{8}-\d{6}_[0-9a-f]{8}_", "", source.stem)
    display_name = re.sub(r"_cleaned_[0-9a-f]{8}$", "", display_name).replace("_", " ")
    name_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", display_name).strip(" .")[:65] or "match"
    selection_label = " - Pots + fouls" if result.selection is ShotSelection.POTS_AND_FOULS else ""
    project_name = f"Snooker - {name_stem}{selection_label} - {uuid.uuid4().hex[:8]}"
    project = installation.projects_dir / project_name
    timestamp = time.time_ns() // 1000
    platform = {
        "os": "windows", "os_version": "", "app_id": 359289,
        "app_version": ".".join(installation.version.split(".")[:3]), "app_source": "cc",
        "device_id": "", "hard_disk_id": "", "mac_address": "",
    }
    content.update(
        id=timeline_id, name=project_name, duration=sum(length for _, _, length in ranges),
        create_time=timestamp, update_time=timestamp, fps=metadata.fps, path="",
        canvas_config={"width": metadata.width, "height": metadata.height, "ratio": "original", "background": None},
        platform=platform, last_modified_platform=dict(platform),
    )
    video = copy.deepcopy(template["video"])
    video.update(
        id=media_id, local_material_id=local_id, path=source.as_posix(),
        material_name=source.name, duration=duration_us, width=metadata.width,
        height=metadata.height, has_audio=metadata.has_audio,
    )
    content["materials"]["videos"] = [video]
    segments, manifest_shots, cursor = [], [], 0
    for shot, start_us, length in ranges:
        segment = copy.deepcopy(template["segment"])
        refs = []
        for category, neutral in template["auxiliary"].items():
            auxiliary = copy.deepcopy(neutral)
            auxiliary["id"] = _id()
            content["materials"][category].append(auxiliary)
            refs.append(auxiliary["id"])
        segment.update(
            id=_id(), material_id=media_id, extra_material_refs=refs,
            source_timerange={"start": start_us, "duration": length},
            target_timerange={"start": cursor, "duration": length},
            desc=f"Shot {shot.shot_id}", volume=1.0 if metadata.has_audio else 0.0,
        )
        segments.append(segment)
        manifest_shots.append({
            "shot_id": shot.shot_id, "source_start_us": start_us,
            "duration_us": length, "timeline_start_us": cursor,
        })
        cursor += length
    content["tracks"] = [{
        "id": _id(), "type": "video", "flag": 0, "attribute": 0,
        "name": "Snooker shots", "is_default_name": False, "segments": segments,
    }]
    media_size = source.stat().st_size
    meta = {
        "draft_id": project_id, "draft_name": project_name,
        "draft_fold_path": project.as_posix(), "draft_root_path": project.parent.as_posix(),
        "draft_cover": "draft_cover.jpg", "draft_new_version": "", "draft_type": "",
        "draft_is_invisible": False, "draft_is_cloud_temp_draft": False,
        "draft_is_from_deeplink": "false", "draft_need_rename_folder": False,
        "cloud_draft_sync": False, "cloud_draft_cover": False,
        "draft_timeline_materials_size_": media_size,
        "tm_draft_create": timestamp, "tm_draft_modified": timestamp,
        "tm_draft_removed": 0, "tm_duration": cursor,
        "tm_draft_cloud_entry_id": -1, "tm_draft_cloud_parent_entry_id": -1,
        "tm_draft_cloud_space_id": -1, "tm_draft_cloud_user_id": -1,
        "tm_draft_cloud_modified": 0, "tm_draft_cloud_completed": "",
        "draft_materials_copied_info": [],
        "draft_materials": [{"type": 0, "value": [{
            "id": local_id, "file_Path": source.as_posix(), "extra_info": source.name,
            "duration": duration_us, "width": metadata.width, "height": metadata.height,
            "metetype": "video", "type": 0, "item_source": 1, "md5": "",
            "create_time": timestamp // 1_000_000, "import_time": timestamp // 1_000_000,
            "import_time_ms": timestamp, "roughcut_time_range": {"start": 0, "duration": duration_us},
            "sub_time_range": {"start": -1, "duration": -1},
        }]}] + [{"type": kind, "value": []} for kind in (1, 2, 3, 6, 7, 8, 18)],
    }
    timeline_index = {
        "id": project_id, "version": 0, "main_timeline_id": timeline_id,
        "create_time": timestamp, "update_time": timestamp,
        "config": {"color_space": -1, "hdr_vivid": False, "mixed_track_mode_on": False,
                   "render_index_track_mode_on": True, "use_float_render": False},
        "timelines": [{"id": timeline_id, "name": "Timeline 01", "is_marked_delete": False,
                       "create_time": timestamp, "update_time": timestamp}],
    }
    # Serialize registration across all jobs. Each click gets a unique folder.
    with _store_lock:
        project.mkdir(parents=True, exist_ok=False)
        registered = False
        try:
            timeline = project / "Timelines" / timeline_id
            timeline.mkdir(parents=True)
            for parent in (project, timeline):
                for filename in ("draft_content.json", "draft_content.json.bak", "template-2.tmp"):
                    _write_json(parent / filename, content)
                _write_json(parent / "attachment_editing.json", {})
            _write_json(project / "Timelines" / "project.json", timeline_index)
            _write_json(project / "Timelines" / "project.json.bak", timeline_index)
            _write_json(project / "timeline_layout.json", {
                "dockItems": [{"dockIndex": 0, "ratio": 1, "timelineIds": [timeline_id],
                               "timelineNames": ["Timeline 01"]}], "layoutOrientation": 1,
            })
            _write_json(project / "draft_meta_info.json", meta)
            _write_json(project / "draft_agency_config.json", {
                "is_auto_agency_enabled": False, "is_auto_agency_popup": False,
                "is_single_agency_mode": False, "marterials": None,
                "use_converter": False, "video_resolution": 720,
            })
            for filename in ("draft_biz_config.json", "draft_virtual_store.json", "key_value.json"):
                _write_json(project / filename, {})
            (project / "draft_settings").write_text(
                f"[General]\ndraft_create_time={timestamp // 1_000_000}\n"
                f"draft_last_edit_time={timestamp // 1_000_000}\nreal_edit_seconds=0\n"
                "real_edit_keys=0\ncloud_last_modify_platform=windows\n", encoding="utf-8",
            )
            _write_json(project / "snooker_shots.json", {
                "job_id": result.job_id, "source": source.as_posix(), "shots": manifest_shots,
            })
            _register(project, meta)
            registered = True
        finally:
            if not registered:
                # Only this newly created folder is eligible for rollback.
                shutil.rmtree(project)
    return {
        "project_name": project_name, "project_path": str(project), "project_id": project_id,
        "shot_count": len(segments), "duration": cursor / 1_000_000,
        "source_path": str(source), "capcut_version": installation.version,
    }


def launch_capcut(installation: CapCutInstallation) -> None:
    """Start/focus CapCut; the user selects the named draft on the Home screen."""
    subprocess.Popen([str(installation.executable)], cwd=str(installation.executable.parent))
