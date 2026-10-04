"""Parallel contact extraction preserves evidence, ordering and bounded work."""

from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from snooker_ai.config import load_config
from snooker_ai.pipeline.analyzer import Analyzer, _NativeContactPrefetch
from snooker_ai.types import FrameFeatures, StrikeCandidate
from snooker_ai.utils.timebase import TimeMapper


def ready_analyzer(tmp_path, monkeypatch, *, workers=2):
    monkeypatch.setattr("snooker_ai.pipeline.analyzer.os.cpu_count", lambda: 8)
    analyzer = Analyzer(load_config(overrides={"analysis": {"native_contact_workers": workers}}), tmp_path)
    assert analyzer.broadcast_context.restore_target([1.] * 3072)
    analyzer._coarse_context_reference = [FrameFeatures(t=i*.5, match_context_valid=True)
                                          for i in range(61)]
    return analyzer


def proposals():
    return [StrikeCandidate(timestamp=t, confidence=.5, uncertainty_start=t-.5,
                            uncertainty_end=t+.5) for t in (3., 7., 11.)]


def prefetch(analyzer, candidates=None):
    return _NativeContactPrefetch(analyzer, Path("proxy.mp4"), None,
                                  TimeMapper(30, source_fps=25), 30,
                                  candidates or proposals())


def still_features(key):
    start, end, fps = key
    return [FrameFeatures(t=start+i/fps, observation_fps=fps)
            for i in range(round((end-start)*fps)+1)]


def test_two_outstanding_exact_requests_and_order(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    lock, following_started = Lock(), Event()
    calls, active, peak = [], 0, 0

    def run(key):
        nonlocal active, peak
        with lock:
            calls.append(key)
            active += 1
            peak = max(peak, active)
        if key[0] == 1:
            assert following_started.wait(3)
        else:
            following_started.set()
        with lock:
            active -= 1
        return still_features(key)

    with prefetch(analyzer) as pool:
        monkeypatch.setattr(pool, "_run", run)
        assert pool.plans == [(1., 5., 25.), (5., 9., 25.), (9., 13., 25.)]
        results = [pool.pull(i, key, lambda _: True) for i, key in enumerate(pool.plans)]
        assert [result[0].t for result in results] == [1., 5., 9.]
        assert len(pool.pending) <= 2
    assert sorted(calls) == pool.plans
    assert peak == 2
    assert not pool.pending


@pytest.mark.parametrize("reason", ["serial", "small_cpu", "unknown", "foreign", "return", "missing_start", "gap"])
def test_unsafe_or_serial_context_stays_on_parent(tmp_path, monkeypatch, reason):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    if reason == "serial":
        analyzer.config = load_config(overrides={"analysis": {"native_contact_workers": 1}})
    elif reason == "small_cpu":
        monkeypatch.setattr("snooker_ai.pipeline.analyzer.os.cpu_count", lambda: 2)
    elif reason == "unknown":
        analyzer.broadcast_context.reset_observations(preserve_target=False)
    elif reason == "foreign":
        analyzer._coarse_context_reference[5].match_context_valid = False
    elif reason == "return":
        analyzer._confirmed_target_returns = [(5.5, 7.)]  # Include the .76sec safety margin.
    elif reason == "missing_start":
        analyzer._coarse_context_reference = analyzer._coarse_context_reference[3:]
    elif reason == "gap":
        del analyzer._coarse_context_reference[5:8]
    with prefetch(analyzer) as pool:
        assert pool.pull(0, pool.plans[0], lambda _: True) is None
        assert pool.pool is None


def test_complete_native_coverage_skips_extraction(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    candidate = proposals()[0]
    known = still_features((1., 5., 25.))

    def forbidden(*args, **kwargs):
        pytest.fail("Complete native evidence must not be decoded again")

    monkeypatch.setattr(analyzer, "_extract_features", forbidden)
    monkeypatch.setattr(_NativeContactPrefetch, "_run", forbidden)
    _, features = analyzer._refine_adaptive_windows(
        Path("proxy.mp4"), None, TimeMapper(30, source_fps=25), 30,
        [candidate], existing_dense=known, resume=False,
    )
    assert len(features) == len(known)
    assert not analyzer._detection_diagnostics["native_proposals"][0]["accepted"]


def test_current_dense_cache_is_checked_before_prefetch(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    cached = still_features((1., 5., 25.))
    monkeypatch.setattr(analyzer, "_load_dense_window", lambda *args: cached)

    def forbidden(*args, **kwargs):
        pytest.fail("A cached contact must not be extracted")

    monkeypatch.setattr(analyzer, "_extract_features", forbidden)
    monkeypatch.setattr(_NativeContactPrefetch, "_run", forbidden)
    _, features = analyzer._refine_adaptive_windows(
        Path("proxy.mp4"), None, TimeMapper(30, source_fps=25), 30,
        proposals()[:1], signature="cached", resume=True,
    )
    assert len(features) == len(cached)


def test_known_following_window_does_not_enter_worker_queue(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    calls = []
    with prefetch(analyzer) as pool:
        monkeypatch.setattr(pool, "_run", lambda key: calls.append(key) or still_features(key))
        first = pool.pull(0, pool.plans[0], lambda _: False)
        assert first
    assert calls == [pool.plans[0]]


def test_worker_failure_retries_same_request_synchronously(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    calls = []

    def fail(*args):
        raise OSError("decoder failure")

    def extract(*args, **kwargs):
        calls.append(kwargs)
        return still_features((kwargs["start_time"], kwargs["end_time"], kwargs["sample_fps"])), [], []

    monkeypatch.setattr(_NativeContactPrefetch, "_run", fail)
    monkeypatch.setattr(analyzer, "_extract_features", extract)
    _, features = analyzer._refine_adaptive_windows(
        Path("proxy.mp4"), None, TimeMapper(30, source_fps=25), 30,
        proposals()[:1], resume=False,
    )
    assert len(calls) == 1
    assert (calls[0]["start_time"], calls[0]["end_time"], calls[0]["sample_fps"]) == (1., 5., 25.)
    assert features
    assert not analyzer._detection_diagnostics["native_proposals"][0]["accepted"]


@pytest.mark.parametrize("context_return", [False, True])
def test_worker_has_isolated_exact_target_and_context(tmp_path, monkeypatch, context_return):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    target = analyzer.broadcast_context.export_target()
    calls = []

    def extract(worker, proxy, audio, mapper, duration, **kwargs):
        assert worker is not analyzer
        assert worker.job_dir.parent == analyzer.job_dir/"contact_workers"
        assert worker.broadcast_context.export_target() == target
        assert worker._coarse_context_reference is not analyzer._coarse_context_reference
        assert worker._coarse_context_reference[0].t == .5
        calls.append(kwargs)
        if context_return:
            worker._confirm_target_return([], 2, 4)
        return still_features((1., 5., 25.)), [], []

    monkeypatch.setattr(Analyzer, "_extract_features", extract)
    with prefetch(analyzer, proposals()[:1]) as pool:
        result = pool.pull(0, pool.plans[0], lambda _: True)
    assert (result is None) == context_return
    assert len(calls) == 1
    assert calls[0] == dict(sample_fps=25., start_time=1., end_time=5.,
                           collect_scene_observations=False, checkpoint_stage="contact_prefetch")
    assert analyzer.broadcast_context.export_target() == target
    assert analyzer._confirmed_target_returns == []


def test_discarded_running_work_keeps_two_task_bound(tmp_path, monkeypatch):
    analyzer = ready_analyzer(tmp_path, monkeypatch)
    started, release = Event(), Event()
    calls = []

    def run(key):
        calls.append(key)
        if key[0] == 5:
            started.set()
            assert release.wait(3)
        return still_features(key)

    with prefetch(analyzer) as pool:
        monkeypatch.setattr(pool, "_run", run)
        assert pool.pull(0, pool.plans[0], lambda _: True)
        assert started.wait(3)
        pool.discard(pool.plans[1])
        try:
            assert pool.plans[1] in pool.pending
            assert pool.pull(2, pool.plans[2], lambda _: True)
            assert len(pool.pending) <= 2
        finally:
            release.set()
    assert len(calls) == 3
    assert not pool.pending


def test_scheduling_option_preserves_feature_and_result_signatures(tmp_path):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"signature fixture")
    serial = Analyzer(load_config(overrides={"analysis": {"native_contact_workers": 1}}), tmp_path/"serial")
    parallel = Analyzer(load_config(overrides={"analysis": {"native_contact_workers": 2}}), tmp_path/"parallel")
    assert serial._analysis_signature(source) == parallel._analysis_signature(source)
    assert serial._result_signature(source) == parallel._result_signature(source)


def test_parent_confirmation_merge_and_early_stop_order_match_serial(tmp_path, monkeypatch):
    results, stop_requests, confirmation_order = [], [], []
    monkeypatch.setattr(_NativeContactPrefetch, "_run", lambda self, key: still_features(key))
    for workers in (1, 2):
        analyzer = ready_analyzer(tmp_path/str(workers), monkeypatch, workers=workers)
        detector = analyzer.segmenter.ball_stop
        monkeypatch.setattr(detector, "max_after_strike", 5.)
        calls, confirmed = [], []

        def stop(candidate, frames, duration):
            timestamp = candidate.timestamp
            return SimpleNamespace(
                confirmed=frames[-1].t >= timestamp+2, reason="stationary_confirmed",
                physical_stop_timestamp=timestamp+1,
                stop_confirmation_timestamp=timestamp+2,
                last_ball_motion_timestamp=timestamp+.9, end_confidence=.9,
                manual_review_required=False, motion_start=timestamp,
            )

        def refine(candidates, frames):
            confirmed.append(candidates[0].timestamp)
            candidates[0].evidence["dense_transition_confirmed"] = 1.

        def extract(*args, **kwargs):
            frames = []
            for frame in still_features((kwargs["start_time"], kwargs["end_time"], kwargs["sample_fps"])):
                frames.append(frame)
                if kwargs["stop_when"] is not None and kwargs["stop_when"](frames):
                    break
            if kwargs["stop_when"] is not None:
                calls.append((kwargs["start_time"], kwargs["end_time"],
                              kwargs["sample_fps"], frames[-1].t))
            return frames, [], []

        monkeypatch.setattr(detector, "detect_stop", stop)
        monkeypatch.setattr(analyzer.strike_det, "score_frames", lambda frames: frames)
        monkeypatch.setattr(analyzer.strike_det, "refine_boundaries", refine)
        monkeypatch.setattr(analyzer, "_extract_features", extract)
        candidates, frames = analyzer._refine_adaptive_windows(
            Path("proxy.mp4"), None, TimeMapper(30, source_fps=25), 30,
            proposals()[:2], resume=False,
        )
        results.append(([c.model_dump() for c in candidates], [f.model_dump() for f in frames]))
        stop_requests.append(calls)
        confirmation_order.append(confirmed)
    assert results[0] == results[1]
    assert stop_requests[0] == stop_requests[1]
    assert confirmation_order == [[3., 7.], [3., 7.]]
    assert len(stop_requests[0]) == 4  # Cheap travel and native stop for both shots.
    assert all(actual_end < planned_end for _, planned_end, _, actual_end in stop_requests[0])
