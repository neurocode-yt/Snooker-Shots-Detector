"""Configuration loading and access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

from snooker_ai.types import EditMode

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Nested configuration with attribute and dict access."""

    def __init__(self, data: Optional[dict[str, Any]] = None):
        self._data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, key: str) -> "Config":
        value = self.get(key, {})
        if not isinstance(value, dict):
            return Config({})
        return Config(value)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def mode_settings(self, mode: EditMode | str) -> dict[str, Any]:
        if isinstance(mode, EditMode):
            mode_key = mode.value
        else:
            mode_key = EditMode.from_string(mode).value
        modes = self.get("modes", {})
        if mode_key not in modes:
            raise KeyError(f"Unknown edit mode: {mode_key}")
        return copy.deepcopy(modes[mode_key])

    def resolve_device(self) -> str:
        device = str(self.get("device", "auto")).lower()
        if device != "auto":
            return device
        # The Windows OpenCV wheel used by this project exposes the NVIDIA
        # device through OpenCL even when PyTorch is not installed. Prefer that
        # runtime before falling back to the optional torch probe.
        try:
            from snooker_ai.utils.acceleration import configure_acceleration

            if configure_acceleration(self).enabled:
                return "cuda"
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def ensure_dirs(self, root: Optional[Path] = None) -> None:
        root = root or Path.cwd()
        paths = self.get("paths", {})
        for key in ("data_dir", "uploads_dir", "jobs_dir", "outputs_dir", "models_dir"):
            rel = paths.get(key)
            if rel:
                (root / rel).mkdir(parents=True, exist_ok=True)


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data)


def default_config() -> Config:
    return load_config()
