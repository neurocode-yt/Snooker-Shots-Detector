import numpy as np

import snooker_ai.table_detection.localizer as localizer_module
from snooker_ai.motion.residual import ResidualMotionAnalyzer
from snooker_ai.table_detection.localizer import TableLocalizer


def test_table_localizer_finds_green(config, synthetic_green_frame):
    loc = TableLocalizer(config)
    obs = loc.detect(synthetic_green_frame)
    assert obs.mask is not None
    assert obs.area_ratio > 0.1
    assert obs.confidence > 0.2


def test_table_localizer_retries_on_cpu_after_opencl_allocation_failure(
    config, synthetic_green_frame, monkeypatch
):
    import cv2

    loc = TableLocalizer(config)
    loc.use_opencl = True
    disabled: list[str] = []
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: True)
    monkeypatch.setattr(
        cv2,
        "UMat",
        lambda _value: (_ for _ in ()).throw(
            cv2.error("OpenCL error CL_MEM_OBJECT_ALLOCATION_FAILURE")
        ),
    )
    monkeypatch.setattr(
        localizer_module,
        "disable_acceleration",
        lambda reason: disabled.append(reason),
    )

    obs = loc.detect(synthetic_green_frame)

    assert obs.mask is not None
    assert obs.area_ratio > 0.1
    assert loc.use_opencl is False
    assert disabled and "CL_MEM_OBJECT_ALLOCATION_FAILURE" in disabled[0]


def test_residual_motion_detects_change(config, synthetic_green_frame):
    import cv2

    analyzer = ResidualMotionAnalyzer(config)
    prev = cv2.cvtColor(synthetic_green_frame, cv2.COLOR_BGR2GRAY)
    moved = synthetic_green_frame.copy()
    # shift white ball area
    moved = np.roll(moved, 5, axis=1)
    gray = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
    loc = TableLocalizer(config)
    mask = loc.detect(synthetic_green_frame).mask
    sample = analyzer.analyze(prev, gray, mask)
    assert sample.residual_mean >= 0.0
    assert 0.0 <= sample.motion_score <= 1.0
