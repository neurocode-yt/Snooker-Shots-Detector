"""Real spawn/codec/IPC regressions for CPU parallel native extraction."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pytest

from snooker_ai.config import load_config
from snooker_ai.pipeline.analyzer import Analyzer, _NativeContactPrefetch
from snooker_ai.pipeline import contact_workers
from snooker_ai.types import FrameFeatures, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper
from snooker_ai.utils.work_control import WorkControl, WorkPaused, current_control


@pytest.fixture(autouse=True)
def stable_worker_budget(monkeypatch):
    monkeypatch.setattr(contact_workers.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(contact_workers, "available_memory", lambda: 8 * 1024**3)


@pytest.mark.parametrize("logical,free,requested,backend,expected", [
    (8, 8, "auto", "process", 4), (16, 16, "auto", "process", 4),
    (2, 8, "auto", "process", 1), (8, 2, "auto", "process", 1),
    (8, 3, "auto", "process", 2), (8, None, "auto", "process", 2),
    (8, 8, 1, "process", 1), (8, 8, 99, "process", 4),
    (8, 8, "auto", "thread", 2),
])
def test_worker_budget_accounts_for_cpu_memory_and_backend(monkeypatch, logical, free,
                                                          requested, backend, expected):
    monkeypatch.setattr(contact_workers.os, "cpu_count", lambda: logical)
    monkeypatch.setattr(contact_workers, "available_memory",
                        lambda: free * 1024**3 if free is not None else None)
    assert contact_workers.contact_worker_count(requested, backend) == expected


@pytest.mark.parametrize("known_target", [False, True])
def test_real_spawn_preserves_all_native_pixels_and_feature_fields(tmp_path, synthetic_video,
                                                                  known_target):
    config = load_config(overrides={"analysis": {"native_contact_workers": 2}})
    analyzer = Analyzer(config, tmp_path/"job")
    analyzer._native_source = synthetic_video
    if known_target:
        assert analyzer.broadcast_context.restore_target([1.] * 3072)
    analyzer._coarse_context_reference = [FrameFeatures(t=i*.5) for i in range(15)]
    mapper = TimeMapper(7, 7, 15, 2)
    candidates = [StrikeCandidate(timestamp=2, confidence=.5)]
    with _NativeContactPrefetch(analyzer, synthetic_video, None, mapper, 7, candidates) as pool:
        assert pool.backend == "process"
        # Direct thread implementation is the pre-existing isolated-window path.
        expected = pool._run(pool.plans[0])
        actual = pool.pull(0, pool.plans[0], lambda _: True)
        assert isinstance(pool.pool, ProcessPoolExecutor)
        assert [f.model_dump() for f in actual] == [f.model_dump() for f in expected]
        assert len(actual) == 61  # every 15fps source image in [0,4]


def test_pause_file_reaches_real_child_before_decode(tmp_path):
    control = WorkControl(tmp_path)
    control.request()
    token = current_control.set(control)
    try:
        analyzer = Analyzer(load_config(), tmp_path/"job")
        analyzer._coarse_context_reference = [FrameFeatures(t=i*.5) for i in range(15)]
        with _NativeContactPrefetch(analyzer, Path("absent.mp4"), None, TimeMapper(7), 7,
                                    [StrikeCandidate(timestamp=2, confidence=.5)]) as pool:
            with pytest.raises(WorkPaused):
                pool.pull(0, pool.plans[0], lambda _: True)
    finally:
        current_control.reset(token)


def test_process_scheduling_preserves_saved_detection_signatures(tmp_path):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"source identity")
    old = Analyzer(load_config(overrides={"analysis": {
        "native_contact_workers": 2, "native_contact_backend": "thread",
    }}), tmp_path/"old")
    new = Analyzer(load_config(), tmp_path/"new")
    assert old._analysis_signature(source) == new._analysis_signature(source)
    assert old._result_signature(source) == new._result_signature(source)


def test_failed_pool_creation_retries_on_parent_without_repeated_launches(tmp_path, monkeypatch):
    analyzer = Analyzer(load_config(), tmp_path/"job")
    analyzer._coarse_context_reference = [FrameFeatures(t=i*.5) for i in range(15)]
    calls = []
    with _NativeContactPrefetch(analyzer, Path("unused.mp4"), None, TimeMapper(7), 7,
                                [StrikeCandidate(timestamp=2, confidence=.5)]) as pool:
        def fail():
            calls.append(1)
            raise OSError("process launch unavailable")
        monkeypatch.setattr(pool, "_start_pool", fail)
        assert pool.pull(0, pool.plans[0], lambda _: True) is None
        assert pool.pull(0, pool.plans[0], lambda _: True) is None
    assert calls == [1]
