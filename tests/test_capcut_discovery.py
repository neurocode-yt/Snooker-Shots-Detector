"""Moved installations, registry values and incomplete version directories."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from snooker_ai.rendering.capcut import (
    CapCutUnavailable, _capcut_root, _registered_capcut_roots, _registry_executable, find_capcut,
)


@pytest.fixture
def isolated_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path/"local"))
    monkeypatch.setattr("snooker_ai.rendering.capcut.os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr("snooker_ai.rendering.capcut._registered_local_appdata", lambda: tmp_path/"local")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path/"home"))
    monkeypatch.setattr("snooker_ai.rendering.capcut._registered_capcut_roots", lambda: [])


@pytest.mark.parametrize("value,expected", [
    ('"D:/Moved apps/CapCut/Apps/uninst.exe" /quiet', "D:/Moved apps/CapCut/Apps/uninst.exe"),
    ('"D:/Moved apps/CapCut/Apps/CapCut.exe",0', "D:/Moved apps/CapCut/Apps/CapCut.exe"),
    ('D:/Moved apps/CapCut/Apps/CapCut.exe,0', "D:/Moved apps/CapCut/Apps/CapCut.exe"),
    ('not an executable', None),
])
def test_registry_commands_are_parsed_without_execution(value, expected):
    actual = _registry_executable(value)
    assert actual == (Path(expected) if expected else None)


@pytest.mark.parametrize("suffix", ["", "/Apps", "/Apps/CapCut.exe", "/Apps/9.6.0.4097/CapCut.exe"])
def test_registry_locations_resolve_to_the_install_root(tmp_path, suffix):
    root = tmp_path/"moved CapCut"
    assert _capcut_root(Path(str(root)+suffix)) == root


def test_registry_install_is_found_without_an_appdata_link(config, tmp_path, monkeypatch, isolated_windows):
    root = tmp_path/"another drive"/"CapCut"
    apps = root/"Apps"
    (apps/"9.6.0.4097").mkdir(parents=True)
    executable = apps/"CapCut.exe"
    executable.touch()
    monkeypatch.setattr("snooker_ai.rendering.capcut._registered_capcut_roots", lambda: [root])
    result = find_capcut(config)
    assert result.executable == executable.resolve()
    assert result.version == "9.6.0.4097"
    assert result.projects_dir == (root/"User Data/Projects/com.lveditor.draft").absolute()


def test_incomplete_update_does_not_hide_an_installed_version(config, tmp_path, monkeypatch, isolated_windows):
    root = tmp_path/"local/CapCut"
    (root/"Apps/9.6.0.4097").mkdir(parents=True)
    (root/"Apps/9.7.0.1000").mkdir()
    executable = root/"Apps/9.6.0.4097/CapCut.exe"
    executable.touch()
    result = find_capcut(config)
    assert result.executable == executable.resolve()
    assert result.version == "9.6.0.4097"


def test_missing_app_does_not_tell_an_installed_user_to_reinstall(config, isolated_windows):
    with pytest.raises(CapCutUnavailable, match="restart Snooker AI or set capcut.executable_path"):
        find_capcut(config)


def test_both_user_machine_and_registry_views_are_checked(tmp_path, monkeypatch):
    root = tmp_path/"installed elsewhere/CapCut"
    calls = []

    class Key:
        def __init__(self, hive, view, entry=False):
            self.hive, self.view, self.entry = hive, view, entry
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass

    def open_key(hive, name, reserved=0, access=0):
        if isinstance(hive, Key):
            return Key(hive.hive, hive.view, True)
        calls.append((hive, access))
        return Key(hive, access)

    def read(key, field):
        values = {"DisplayName": "CapCut", "InstallLocation": "",
                  "DisplayIcon": str(root/"Apps/uninst.exe"),
                  "UninstallString": '"'+str(root/"Apps/uninst.exe")+'" /quiet'}
        return values[field], 1

    monkeypatch.setitem(sys.modules, "winreg", SimpleNamespace(
        HKEY_CURRENT_USER=1, HKEY_LOCAL_MACHINE=2, KEY_WOW64_64KEY=16,
        KEY_WOW64_32KEY=32, KEY_READ=4, OpenKey=open_key, QueryValueEx=read,
        QueryInfoKey=lambda _: (1, 0, 0), EnumKey=lambda _, i: "CapCut",
    ))
    assert _registered_capcut_roots() == [root]
    assert set(calls) == {(1, 20), (1, 36), (2, 20), (2, 36)}
