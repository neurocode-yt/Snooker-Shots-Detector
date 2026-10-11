"""The optional neural runtime cannot overwrite or fall back to classic jobs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from snooker_ai.dl.pipeline import DLAnalyzer, dl_result_signature
from snooker_ai.dl.settings import dl_settings, expected_feature_spec
from snooker_ai.types import AnalysisResult, EditMode, VideoMetadata


@pytest.fixture
def dl_runtime(config, tmp_path):
    assets = tmp_path/'backbones'
    (assets/'rgb').mkdir(parents=True)
    (assets/'raft-small.safetensors').write_bytes(b'flow test weights')
    (assets/'rgb/model.safetensors').write_bytes(b'rgb test weights')
    identity = {'rgb_model': 'facebook/dinov2-small', 'rgb_revision': 'pinned-revision',
                'flow_model': 'torchvision/raft_small/C_T_V2',
                'flow_sha256': hashlib.sha256(b'flow test weights').hexdigest()}
    rgb_config = {'model_type': 'dinov2', 'hidden_size': 384, 'patch_size': 14}
    (assets/'assets.json').write_text(json.dumps(identity), encoding='utf-8')
    (assets/'rgb/config.json').write_text(json.dumps(rgb_config), encoding='utf-8')
    python = tmp_path/'isolated python.exe'
    python.write_bytes(b'fixture interpreter')
    checkpoint = tmp_path/'temporal.pt'
    checkpoint.write_bytes(b'fixture checkpoint')
    config._data['dl_algo'] = {'python': str(python), 'checkpoint': str(checkpoint), 'assets': str(assets)}
    settings = dl_settings(config)
    spec = expected_feature_spec(settings, identity, rgb_config)
    calibration = {
        'thresholds': dict.fromkeys(('event', 'keep', 'end', 'replay', 'handling'), .5),
        'temperatures': [1., 1., 1., 1., 1.], 'event': {'recall': .8},
        'frame_heads': {}, 'scope': 'group_disjoint_validation',
        'event_candidate_settings': {'nms_seconds': 1., 'uncertainty_ratio': .5,
                                     'max_uncertainty_seconds': 2.},
    }
    report = {'feature_spec': spec, 'calibration': calibration,
              'independent_test_available': False, 'deployment_quality_verified': False}
    report_path = checkpoint.with_suffix('.pt.report.json')
    report_path.write_text(json.dumps(report), encoding='utf-8')
    source = tmp_path/'source with spaces.mp4'
    source.write_bytes(b'fixture video')
    return config, source, tmp_path/'job', settings, report, report_path, assets


def _result(source, job_id, signature, spec):
    return AnalysisResult(job_id=job_id, source_path=str(source),
                          metadata=VideoMetadata(path=str(source), duration=12),
                          mode=EditMode.STRICT, analysis_signature=signature,
                          analysis_feature_signature=json.dumps(spec))


def _worker(monkeypatch, job, result, *, code=0, lines=()):
    calls = []

    class Process:
        stdout = iter(lines)
        terminated = False
        killed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def wait(self, timeout=None):
            return code

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        if result is not None:
            (job/'analysis.json').write_text(result.model_dump_json(), encoding='utf-8')
        return Process()

    monkeypatch.setattr('snooker_ai.dl.pipeline.subprocess.Popen', popen)
    return calls, Process


def test_capability_is_optional_experimental_and_preserves_quality_flags(dl_runtime):
    config, *_ = dl_runtime
    capability = DLAnalyzer.capabilities(config)
    assert capability['available'] is True
    assert capability['experimental'] is True
    assert capability['uses_classic_detector'] is False
    assert capability['independent_test_available'] is False
    assert capability['deployment_quality_verified'] is False


@pytest.mark.parametrize('field,value', [
    ('flow_size', [128, 256]), ('flow_grid', [4, 7]), ('flow_updates', 12),
    ('flow_units', 'pixels_per_frame'), ('source_time', 'proxy'), ('dimension', 20),
    ('rgb_views', 'only_global'), ('version', 99), ('flow_channels', 'magnitude'),
])
def test_capability_requires_complete_feature_semantics(dl_runtime, field, value):
    config, _, _, _, report, report_path, _ = dl_runtime
    report['feature_spec'][field] = value
    report_path.write_text(json.dumps(report), encoding='utf-8')
    assert DLAnalyzer.capabilities(config)['available'] is False


@pytest.mark.parametrize('invalid', [None, [], 'report', {'feature_spec': []}])
def test_malformed_report_does_not_break_classic_capability_endpoint(dl_runtime, invalid):
    config, _, _, _, _, report_path, _ = dl_runtime
    report_path.write_text(json.dumps(invalid), encoding='utf-8')
    capability = DLAnalyzer.capabilities(config)
    assert capability['available'] is False
    assert capability['uses_classic_detector'] is False


@pytest.mark.parametrize('malformed', ['threshold', 'temperatures', 'decoder', 'quality'])
def test_capability_rejects_invalid_or_incompatible_calibration(dl_runtime, malformed):
    config, _, _, _, report, report_path, _ = dl_runtime
    if malformed == 'threshold':
        report['calibration']['thresholds']['event'] = float('nan')
    elif malformed == 'temperatures':
        report['calibration']['temperatures'] = [0]*5
    elif malformed == 'decoder':
        report['calibration']['event_candidate_settings']['nms_seconds'] = 3.
    else:
        report['independent_test_available'] = 'false'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    assert DLAnalyzer.capabilities(config)['available'] is False


@pytest.mark.parametrize('missing', ['python', 'checkpoint', 'report', 'rgb_weights', 'flow_weights'])
def test_missing_assets_fail_without_creating_job(dl_runtime, missing):
    config, source, job, settings, _, report_path, assets = dl_runtime
    paths = {'python': Path(settings['python']), 'checkpoint': Path(settings['checkpoint']),
             'report': report_path, 'rgb_weights': assets/'rgb/model.safetensors',
             'flow_weights': assets/'raft-small.safetensors'}
    paths[missing].unlink()
    with pytest.raises(RuntimeError, match='DL Algo'):
        DLAnalyzer(config, job).analyze(source, 'test')
    assert not job.exists()


def test_modified_flow_weights_disable_ready_mode(dl_runtime):
    config, _, _, _, _, _, assets = dl_runtime
    (assets/'raft-small.safetensors').write_bytes(b'other model')
    assert DLAnalyzer.capabilities(config)['available'] is False


@pytest.mark.parametrize('override', ['bad', {'python': None}, {'assets': 1}])
def test_invalid_runtime_configuration_returns_unavailable(dl_runtime, override):
    config, *_ = dl_runtime
    config._data['dl_algo'] = override
    assert DLAnalyzer.capabilities(config)['available'] is False


@pytest.mark.parametrize('metadata', [None, {}, {'algorithm': 'classic'}, {'algorithm': 'dl_algo'}])
def test_legacy_analysis_is_protected_even_if_new_metadata_says_dl(dl_runtime, metadata):
    config, source, job, *_ = dl_runtime
    job.mkdir()
    if metadata is not None:
        (job/'job.json').write_text(json.dumps(metadata), encoding='utf-8')
    old_analysis = {'analysis_feature_signature': 'classic-feature-hash'}
    (job/'analysis.json').write_text(json.dumps(old_analysis), encoding='utf-8')
    (job/'highlights.mp4').write_bytes(b'old highlight')
    before = {path.name: path.read_bytes() for path in job.iterdir()}
    with pytest.raises(ValueError, match='protect|separate'):
        DLAnalyzer(config, job).analyze(source, 'test', resume=False)
    assert {path.name: path.read_bytes() for path in job.iterdir()} == before


def test_saved_dl_result_resumes_only_matching_identity(dl_runtime, monkeypatch):
    config, source, job, settings, report, *_ = dl_runtime
    job.mkdir()
    signature = dl_result_signature(source, settings)
    result = _result(source, 'test', signature, report['feature_spec'])
    (job/'analysis.json').write_text(result.model_dump_json(), encoding='utf-8')
    calls, _ = _worker(monkeypatch, job, None)
    progress = []
    resumed = DLAnalyzer(config, job).analyze(source, 'test', progress=lambda *args: progress.append(args))
    assert resumed == result
    assert calls == []
    assert progress[-1][0] == 1.


def test_worker_command_is_isolated_offline_and_handles_non_progress_output(dl_runtime, monkeypatch):
    config, source, job, settings, report, *_ = dl_runtime
    result = _result(source, 'test', dl_result_signature(source, settings), report['feature_spec'])
    lines = ['model warning\n', '[]\n', 'null\n', '{"type":"progress"}\n',
             '{"type":"progress","fraction":"NaN","stage":"x","message":"x"}\n',
             '{"type":"progress","fraction":0.6,"stage":"analyzing","message":"Neural"}\n']
    calls, _ = _worker(monkeypatch, job, result, lines=lines)
    progress = []
    produced = DLAnalyzer(config, job).analyze(source, 'test', progress=lambda *args: progress.append(args))
    assert produced == result
    command, kwargs = calls[0]
    assert command == [settings['python'], '-m', 'snooker_ai.dl.worker', '--request', str(job/'dl_request.json')]
    assert 'shell' not in kwargs
    assert kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
    if os.name == 'nt':
        assert kwargs['creationflags'] == subprocess.CREATE_NO_WINDOW
    assert progress == [(0.6, 'analyzing', 'Neural')]
    assert json.loads((job/'dl_request.json').read_text())['algorithm'] == 'dl_algo'


@pytest.mark.parametrize('wrong', ['job_id', 'source', 'signature'])
def test_worker_output_must_match_requested_source_job_and_signature(dl_runtime, monkeypatch, wrong):
    config, source, job, settings, report, *_ = dl_runtime
    result = _result(source, 'test', dl_result_signature(source, settings), report['feature_spec'])
    if wrong == 'job_id':
        result.job_id = 'another'
    elif wrong == 'source':
        result.source_path = str(source.parent/'another.mp4')
    else:
        result.analysis_signature = 'wrong'
    _worker(monkeypatch, job, result)
    with pytest.raises(RuntimeError, match='mismatched'):
        DLAnalyzer(config, job).analyze(source, 'test')


def test_worker_failure_is_reported_without_classic_fallback(dl_runtime, monkeypatch):
    config, source, job, *_ = dl_runtime
    calls, _ = _worker(monkeypatch, job, None, code=1, lines=['CUDA unavailable\n'])
    with pytest.raises(RuntimeError, match='isolated runtime.*CUDA unavailable'):
        DLAnalyzer(config, job).analyze(source, 'test')
    assert len(calls) == 1
    assert not (job/'analysis.json').exists()


def test_worker_success_without_result_is_clear_failure(dl_runtime, monkeypatch):
    config, source, job, *_ = dl_runtime
    _worker(monkeypatch, job, None)
    with pytest.raises(RuntimeError, match='without producing'):
        DLAnalyzer(config, job).analyze(source, 'test')


def test_worker_passes_calibration_temperatures_and_has_no_physical_stop_claims(dl_runtime, monkeypatch):
    from snooker_ai.dl import worker

    config, source, job, settings, report, *_ = dl_runtime
    calibration = copy.deepcopy(report['calibration'])
    calibration['temperatures'] = [2., 1.5, 1., .75, .5]
    captured = {}
    times = np.arange(0., 12., .125)
    features = np.zeros((len(times), 2), dtype=np.float32)
    output = np.full((len(times), 5), .1, dtype=np.float32)
    output[40, 0] = .9

    class Encoder:
        device = 'cpu'
        spec = report['feature_spec']

        def __init__(self, *_):
            pass

        def extract(self, *_args, **_kwargs):
            return times, features

    def predict(_model, _features, **kwargs):
        captured.update(kwargs)
        return output

    monkeypatch.setattr(worker, 'NeuralFeatures', Encoder)
    monkeypatch.setattr(worker, 'validate_video', lambda *_args, **_kwargs: VideoMetadata(path=str(source), duration=12.))
    monkeypatch.setattr(worker, 'load_temporal_checkpoint', lambda *_args, **_kwargs: (
        object(), {'calibration': calibration, 'training': {'independent_test_available': False}}))
    monkeypatch.setattr(worker, 'predict_probabilities', predict)
    result = worker.run_request({'algorithm': 'dl_algo', 'source': str(source), 'job_dir': str(job),
                                 'job_id': 'test', 'signature': 'test-signature', 'settings': settings,
                                 'config': config.as_dict(), 'mode': 'strict'})
    assert captured['temperatures'] == calibration['temperatures']
    assert all(not frame.ball_kinematics_valid and not frame.table_observable for frame in result.features)
    assert len(result.shots) == 1
    assert result.shots[0].evidence['physical_stop_observed'] is False
    assert result.shots[0].evidence['boundary_kind'] == 'neural_edit_estimate'
    assert (job/'analysis.json').is_file()
    assert (job/'timeline.json').is_file()
    assert not list(job.glob('*.tmp'))


def test_worker_rejects_classic_directory_before_reading_video(dl_runtime, monkeypatch):
    from snooker_ai.dl import worker

    config, source, job, settings, *_ = dl_runtime
    job.mkdir()
    (job/'job.json').write_text('{"algorithm":"classic"}', encoding='utf-8')
    called = []
    monkeypatch.setattr(worker, 'validate_video', lambda *_: called.append(True))
    with pytest.raises(ValueError, match='protected'):
        worker.run_request({'source': str(source), 'job_dir': str(job),
                            'settings': settings, 'config': config.as_dict()})
    assert called == []
    assert set(path.name for path in job.iterdir()) == {'job.json'}
