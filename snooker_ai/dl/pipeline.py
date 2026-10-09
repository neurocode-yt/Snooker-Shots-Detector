"""Isolated DL Algo adapter. The current algorithm/environment is never replaced."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Callable

from snooker_ai.config import Config
from snooker_ai.dl.settings import ROOT, dl_settings, expected_feature_spec, resolved_path
from snooker_ai.types import AnalysisResult, EditMode

PIPELINE_VERSION = 5
HEAD_NAMES = ('event', 'keep', 'end', 'replay', 'handling')


def validate_calibration(calibration: dict, selection: dict) -> None:
    """Reject incomplete calibration before exposing or using a trained mode."""
    if not isinstance(calibration, dict):
        raise ValueError('DL calibration must be a mapping.')
    thresholds = calibration['thresholds']
    for head in HEAD_NAMES:
        value = float(thresholds[head])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('DL thresholds must be finite probabilities.')
    temperatures = calibration['temperatures']
    if len(temperatures) != len(HEAD_NAMES) or any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in temperatures):
        raise ValueError('DL calibration requires five positive finite temperatures.')
    if not isinstance(calibration['event'], dict) or not isinstance(calibration['frame_heads'], dict):
        raise ValueError('DL calibration metrics must be mappings.')
    candidate_settings = calibration['event_candidate_settings']
    for key, default in (('nms_seconds', 1.), ('uncertainty_ratio', .5), ('max_uncertainty_seconds', 2.)):
        if float(candidate_settings[key]) != float(selection.get(key, default)):
            raise ValueError('DL event decoder differs from its calibrated configuration.')


def protect_job_directory(job: Path) -> None:
    """Preserve old results even for direct adapters or metadata-free exports."""
    metadata_path = job/'job.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('algorithm', 'classic') != 'dl_algo':
            raise ValueError('DL Algo requires a separate job; current-mode results are protected.')
    analysis_path = job/'analysis.json'
    if analysis_path.exists():
        prior = json.loads(analysis_path.read_text(encoding='utf-8'))
        try:
            spec = json.loads(prior.get('analysis_feature_signature') or '{}')
        except (TypeError, ValueError):
            spec = {}
        if not isinstance(spec, dict) or 'rgb_views' not in spec or 'flow_model' not in spec:
            raise ValueError('Choose a separate DL Algo job directory; existing analysis is protected.')


def atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(contents, encoding='utf-8')
    temporary.replace(path)


def _matches_result(result: AnalysisResult, source: Path, job_id: str, signature: str) -> bool:
    return (result.analysis_signature == signature and result.job_id == job_id
            and Path(result.source_path).resolve() == source.resolve())


def training_progress() -> dict:
    """Expose only user-facing progress from the single local training run."""
    try:
        state = json.loads((ROOT/'data/dl/training/state.json').read_text(encoding='utf-8'))
        result = {key: state[key] for key in ('status', 'stage', 'updated_at')
                  if isinstance(state.get(key), str)}
        percent = state.get('feature_percent')
        if isinstance(percent, (int, float)) and math.isfinite(percent) and 0 <= percent <= 100:
            result['video_percent'] = percent
        return result
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def dl_result_signature(source: Path, settings: dict) -> str:
    checkpoint = resolved_path(settings['checkpoint'])
    stat = source.stat()
    payload = {
        'pipeline_version': PIPELINE_VERSION, 'source': str(source.resolve()),
        'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'settings': settings,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DLAnalyzer:
    """Run learned RGB/flow/temporal inference in its own optional Python runtime."""
    def __init__(self, config: Config, job_dir: Path):
        self.config, self.job_dir = config, Path(job_dir)
        self.settings = dl_settings(config)

    @classmethod
    def capabilities(cls, config: Config) -> dict:
        status = {'available': False, 'experimental': True,
                  'algorithm': 'dl_algo', 'uses_classic_detector': False}
        try:
            settings = dl_settings(config)
            interpreter = resolved_path(settings['python'])
            checkpoint = resolved_path(settings['checkpoint'])
            report_path = checkpoint.with_suffix(checkpoint.suffix+'.report.json')
            assets = resolved_path(settings['assets'])
        except (KeyError, TypeError, ValueError, OSError) as error:
            return {**status, 'reason': f'DL Algo configuration needs repair: {type(error).__name__}.'}
        if not interpreter.is_file():
            return {**status, 'reason': 'DL Algo needs its separate ML runtime.'}
        if not (assets/'assets.json').is_file() or not (assets/'rgb/config.json').is_file() or not (
                assets/'raft-small.safetensors').is_file():
            return {**status, 'reason': 'DL Algo pretrained models are being prepared.'}
        if not checkpoint.is_file() or not report_path.is_file():
            training = training_progress()
            reason = 'DL Algo is waiting for a trained snooker model.'
            if training.get('status') == 'running':
                stage = training.get('stage')
                if stage == 'extracting_features':
                    reason = 'DL Algo is preparing the reviewed videos for learning.'
                    if 'video_percent' in training:
                        reason += f" Current video: {training['video_percent']:.0f}%."
                elif stage == 'training':
                    reason = 'DL Algo is training its snooker model.'
                elif stage == 'benchmarking':
                    reason = 'DL Algo is checking its highlight predictions.'
            elif training.get('status') == 'failed':
                reason = 'DL Algo training stopped before the model was ready. Review the training log.'
            return {**status, 'reason': reason, 'training': training}
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            identity = json.loads((assets/'assets.json').read_text(encoding='utf-8'))
            rgb_config = json.loads((assets/'rgb/config.json').read_text(encoding='utf-8'))
            weights = assets/'rgb/model.safetensors'
            index = assets/'rgb/model.safetensors.index.json'
            if not weights.is_file():
                weight_map = json.loads(index.read_text(encoding='utf-8'))['weight_map']
                if not weight_map or not all((assets/'rgb'/name).is_file() for name in set(weight_map.values())):
                    raise ValueError('RGB backbone weights are missing.')
            if hashlib.sha256((assets/'raft-small.safetensors').read_bytes()).hexdigest() != identity['flow_sha256']:
                raise ValueError('Cached flow weights differ from their recorded identity.')
            spec = report['feature_spec']
            if spec != expected_feature_spec(settings, identity, rgb_config):
                return {**status, 'reason': 'DL Algo model and settings differ; matching training is required.'}
            calibration = report['calibration']
            validate_calibration(calibration, settings['selection'])
            for key in ('independent_test_available', 'deployment_quality_verified'):
                if key in report and not isinstance(report[key], bool):
                    raise ValueError('DL quality flags must be boolean.')
            return {
                **status, 'available': True, 'reason': 'Experimental trained model ready for comparison.',
                'model': spec['rgb_model'], 'flow_model': spec['flow_model'],
                'trained_model': checkpoint.name,
                'validation': calibration.get('event', {}) if calibration.get('scope') == 'group_disjoint_validation' else {},
                'calibration_metrics': calibration.get('event', {}),
                'calibration_scope': calibration.get('scope'),
                'metric_scope': report.get('metric_scope', calibration.get('scope')),
                'independent_test_available': bool(report.get('independent_test_available', False)),
                'deployment_quality_verified': bool(report.get('deployment_quality_verified', False)),
            }
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError, OSError) as error:
            return {**status, 'reason': f'DL Algo training report needs repair: {type(error).__name__}.'}

    def analyze(self, source: str | Path, job_id: str, mode: EditMode = EditMode.STRICT,
                progress: Callable | None = None, resume: bool = True,
                force_reanalyze: bool = False) -> AnalysisResult:
        available = self.capabilities(self.config)
        if not available['available']:
            raise RuntimeError(available['reason'])
        source = Path(source).resolve()
        protect_job_directory(self.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = self.job_dir/'analysis.json'
        signature = dl_result_signature(source, self.settings)
        if resume and not force_reanalyze and analysis_path.exists():
            result = AnalysisResult.model_validate_json(analysis_path.read_text(encoding='utf-8'))
            if _matches_result(result, source, job_id, signature):
                if progress:
                    progress(1., 'ready_for_review', 'Resumed saved DL Algo highlights')
                return result
        request = {
            'algorithm': 'dl_algo',
            'source': str(source), 'job_id': job_id, 'job_dir': str(self.job_dir.resolve()),
            'settings': self.settings, 'config': self.config.as_dict(),
            'signature': signature, 'mode': mode.value,
        }
        request_path = self.job_dir/'dl_request.json'
        atomic_text(request_path, json.dumps(request, indent=2))
        environment = dict(os.environ)
        environment.update(PYTHONUNBUFFERED='1', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
        command = [str(resolved_path(self.settings['python'])), '-m', 'snooker_ai.dl.worker',
                   '--request', str(request_path.resolve())]
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        tail: deque[str] = deque(maxlen=20)
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                              **options) as process:
            try:
                for line in process.stdout:
                    tail.append(line.strip())
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get('type') == 'progress' and progress:
                        try:
                            fraction = float(event['fraction'])
                            stage, message = event['stage'], event['message']
                            if not math.isfinite(fraction) or not isinstance(stage, str) or not isinstance(message, str):
                                continue
                        except (KeyError, TypeError, ValueError):
                            continue
                        progress(max(0., min(1., fraction)), stage, message)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                raise
        if code:
            raise RuntimeError('DL Algo inference failed in its isolated runtime: '+' | '.join(tail)[-1600:])
        if not analysis_path.is_file():
            raise RuntimeError('DL Algo worker finished without producing an analysis.')
        result = AnalysisResult.model_validate_json(analysis_path.read_text(encoding='utf-8'))
        if not _matches_result(result, source, job_id, signature):
            raise RuntimeError('DL Algo worker produced a mismatched result.')
        return result
