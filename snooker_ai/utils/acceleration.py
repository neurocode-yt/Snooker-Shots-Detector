"""Runtime selection for the local NVIDIA/OpenCL acceleration path.

The standard ``opencv-python-headless`` wheel does not expose CUDA kernels, but
it does expose OpenCL through the NVIDIA D3D11 bridge on Windows.  Keeping the
selection here makes the rest of the pipeline backend-agnostic and guarantees a
CPU fallback when a driver, CI runner, or Linux host has no compatible device.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from snooker_ai.config import Config
from snooker_ai.utils.logging import get_logger

logger = get_logger("acceleration")


@dataclass(frozen=True)
class AccelerationInfo:
    requested: str
    backend: str
    device_name: str = "CPU"
    vendor: str = ""

    @property
    def enabled(self) -> bool:
        return self.backend == "nvidia_opencl"


_active: AccelerationInfo | None = None


def configure_acceleration(config: Config) -> AccelerationInfo:
    """Enable the NVIDIA OpenCL device when ``device`` is ``auto``/``cuda``.

    The result is cached because OpenCV's OpenCL switch is process-global.  An
    explicit ``device: cpu`` always disables this path.
    """

    global _active
    requested = str(config.get("device", "auto")).strip().lower()
    if _active is not None and _active.requested == requested:
        return _active

    if requested in {"cpu", "none", "off", "false", "0"}:
        cv2.ocl.setUseOpenCL(False)
        _active = AccelerationInfo(requested=requested, backend="cpu")
        return _active

    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            device = cv2.ocl.Device_getDefault()
            name = str(device.name())
            vendor = str(device.vendorName())
            is_nvidia = "nvidia" in f"{name} {vendor}".lower()
            if device.available() and cv2.ocl.useOpenCL() and is_nvidia:
                _active = AccelerationInfo(
                    requested=requested,
                    backend="nvidia_opencl",
                    device_name=name,
                    vendor=vendor,
                )
                logger.info("GPU acceleration enabled: %s (%s)", name, vendor)
                return _active
    except Exception as exc:  # pragma: no cover - driver-specific failures
        logger.warning("Could not enable NVIDIA OpenCL acceleration: %s", exc)

    cv2.ocl.setUseOpenCL(False)
    _active = AccelerationInfo(requested=requested, backend="cpu")
    logger.info("GPU acceleration unavailable; using CPU/OpenCV")
    return _active


def acceleration_enabled(config: Config) -> bool:
    return configure_acceleration(config).enabled


def active_acceleration() -> AccelerationInfo | None:
    return _active
