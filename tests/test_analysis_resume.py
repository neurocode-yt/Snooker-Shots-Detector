"""Saved analysis must be invalidated when its source or detection contract changes."""

import pytest
from types import SimpleNamespace

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import AnalysisResult, FrameFeatures, StrikeCandidate, VideoMetadata
from snooker_ai.utils.timebase import TimeMapper


def _saved_analysis(config, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    analyzer = Analyzer(config, tmp_path / "job")
    result = AnalysisResult(
        job_id="job", source_path=str(source), metadata=VideoMetadata(path=str(source)),
        analysis_signature=analyzer._result_signature(source),
    )
    analyzer._save_result(result)
    return source, analyzer, result


def test_completed_analysis_resumes_without_rescanning(config, tmp_path, monkeypatch):
    source, analyzer, _ = _saved_analysis(config, tmp_path)
    monkeypatch.setattr(
        "snooker_ai.pipeline.analyzer.validate_video",
        lambda *args, **kwargs: pytest.fail("unchanged analysis should resume"),
    )
    assert analyzer.analyze(source, "job").job_id == "job"


@pytest.mark.parametrize("change", ["source", "detection", "segmentation", "legacy"])
def test_stale_completed_analysis_is_rebuilt(config, tmp_path, monkeypatch, change):
    source, analyzer, result = _saved_analysis(config, tmp_path)
    if change == "source":
        source.write_bytes(b"replacement video")
    elif change == "detection":
        config._data["strike_fusion"]["min_confidence"] = 0.99
    elif change == "segmentation":
        config._data["modes"]["strict"]["pre_roll"] = 3.0
    else:
        result.analysis_signature = ""
        analyzer._save_result(result)

    def needs_rebuild(*args, **kwargs):
        raise RuntimeError("fresh validation reached")

    monkeypatch.setattr("snooker_ai.pipeline.analyzer.validate_video", needs_rebuild)
    with pytest.raises(RuntimeError, match="fresh validation reached"):
        analyzer.analyze(source, "job")


def test_feature_caches_invalidate_detection_but_preserve_export_changes(config, tmp_path):
    source, analyzer, _ = _saved_analysis(config, tmp_path)
    signature = analyzer._analysis_signature(source)
    config._data["export"]["crf"] = 30
    assert analyzer._analysis_signature(source) == signature
    config._data["analysis"]["refine_fps"] = 20.0
    assert analyzer._analysis_signature(source) != signature


@pytest.mark.parametrize("change", ["none", "source", "detection", "legacy"])
def test_prior_dense_observations_require_matching_source_and_detection_signature(
    config, tmp_path, monkeypatch, change,
):
    source, analyzer, result = _saved_analysis(config, tmp_path)
    result.analysis_signature = "old editing policy"
    result.analysis_feature_signature = (
        analyzer._analysis_signature(source) if change != "legacy" else "")
    result.features = [FrameFeatures(t=1, observation_fps=25)]
    analyzer._save_result(result)
    if change == "source":
        source.write_bytes(b"another video")
    elif change == "detection":
        config._data["ball_stop"]["motion_stop_normalized_speed"] = .20
    monkeypatch.setattr("snooker_ai.pipeline.analyzer.validate_video",
                        lambda *args, **kwargs: VideoMetadata(path=str(source), duration=10, fps=25))
    monkeypatch.setattr("snooker_ai.pipeline.analyzer.generate_proxy",
                        lambda *args, **kwargs: SimpleNamespace(
                            proxy_path=source, audio_path=None, mapper=TimeMapper(source_duration=10)))
    monkeypatch.setattr(analyzer, "_load_coarse_cache", lambda signature: ([FrameFeatures(t=0)], [], []))
    monkeypatch.setattr(analyzer, "_visual_proposals",
                        lambda features: [StrikeCandidate(timestamp=1, confidence=.6)])
    def refinement(*args, existing_dense=None, **kwargs):
        assert (existing_dense is not None) == (change == "none")
        if existing_dense is not None:
            assert existing_dense[0].t == 1
        raise RuntimeError("refinement checked")
    monkeypatch.setattr(analyzer, "_refine_candidate_windows", refinement)
    with pytest.raises(RuntimeError, match="refinement checked"):
        analyzer.analyze(source, "job")
