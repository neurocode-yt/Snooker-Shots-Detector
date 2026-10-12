"""Safe snapshots of the original detector's streaming state (no pickle)."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from snooker_ai.motion.residual import MotionSample
from snooker_ai.object_detection.detector import Detection
from snooker_ai.scene_detection.broadcast_context import BroadcastContextObservation
from snooker_ai.scene_detection.detector import SceneObservation
from snooker_ai.scene_detection.table_context import ViewGeometry
from snooker_ai.table_detection.localizer import TableObservation
from snooker_ai.tracking.tracker import Track
from snooker_ai.types import CameraViewType
from snooker_ai.utils.persistence import atomic_output

_TYPES = {cls.__name__: cls for cls in (
    MotionSample, Detection, BroadcastContextObservation, SceneObservation,
    ViewGeometry, TableObservation, Track, CameraViewType,
)}


def save_snapshot(path: Path, payload: dict) -> None:
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Enum):
            return {"enum": type(value).__name__, "value": value.value}
        if is_dataclass(value):
            return {"type": type(value).__name__, "fields": {
                field.name: encode(getattr(value, field.name)) for field in fields(value)
            }}
        if isinstance(value, tuple):
            return {"tuple": [encode(item) for item in value]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

    encoded = json.dumps(encode(payload), separators=(",", ":")).encode("utf-8")
    arrays["metadata"] = np.frombuffer(encoded, dtype=np.uint8)
    with atomic_output(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)


def load_snapshot(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        def decode(value):
            if isinstance(value, list):
                return [decode(item) for item in value]
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return archive[value["array"]].copy()
            if "tuple" in value:
                return tuple(decode(item) for item in value["tuple"])
            if "dict" in value:
                return {decode(key): decode(item) for key, item in value["dict"]}
            if "enum" in value:
                return _TYPES[value["enum"]](value["value"])
            cls = _TYPES[value["type"]]
            return cls(**{key: decode(item) for key, item in value["fields"].items()})

        return decode(json.loads(archive["metadata"].tobytes().decode("utf-8")))
