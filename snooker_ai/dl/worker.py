"""Learned-only highlight processing inside the dedicated DL runtime."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from snooker_ai.config import Config
from snooker_ai.dl.features import NeuralFeatures
from snooker_ai.dl.pipeline import atomic_text, protect_job_directory, validate_calibration
from snooker_ai.dl.selection import select_highlights
from snooker_ai.dl.settings import resolved_path
from snooker_ai.dl.temporal import load_temporal_checkpoint, predict_probabilities
from snooker_ai.ingestion.probe import validate_video
from snooker_ai.types import AnalysisResult, EditMode, FrameFeatures, ShotState, TimelineEvent


def emit(fraction: float, stage: str, message: str) -> None:
    print(json.dumps({'type': 'progress', 'fraction': fraction,
                      'stage': stage, 'message': message}), flush=True)


def run_request(request: dict) -> AnalysisResult:
    if request.get('algorithm', 'dl_algo') != 'dl_algo':
        raise ValueError('A DL worker cannot process another algorithm job.')
    source, job = Path(request['source']).resolve(), Path(request['job_dir']).resolve()
    settings = request['settings']
    config = Config(request['config'])
    protect_job_directory(job)
    job.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    emit(.02, 'validating', 'Checking source for DL Algo')
    metadata = validate_video(source, max_hours=float(config.get('analysis.max_video_hours', 12)))
    encoder = NeuralFeatures(settings, resolved_path(settings['assets']))
    model, checkpoint = load_temporal_checkpoint(
        resolved_path(settings['checkpoint']), device=encoder.device, expected_feature_spec=encoder.spec)
    calibration = checkpoint['calibration']
    validate_calibration(calibration, settings['selection'])
    emit(.08, 'analyzing', 'Reading learned visual and motion features')
    timestamps, features = encoder.extract(source, job/'dl_features.npz', metadata.duration,
        progress=lambda frac, message: emit(.08+.70*frac, 'analyzing', message))
    emit(.80, 'detecting', 'Predicting shot events and editing boundaries')
    probabilities = predict_probabilities(model, features,
        chunk_frames=int(settings.get('temporal_chunk_frames', 512)), device=encoder.device,
        temperatures=calibration['temperatures'], timestamps=timestamps)
    selection_settings = dict(settings['selection'])
    selection_settings.update(head_validation=calibration.get('frame_heads', {}))
    emit(.90, 'segmenting', 'Selecting neural highlights')
    candidates, shots, diagnostics = select_highlights(
        timestamps, probabilities, metadata.duration, selection_settings, calibration['thresholds'])
    np.savez_compressed(job/'dl_predictions.npz', timestamps=timestamps, probabilities=probabilities,
                        head_names=np.asarray(['event','keep','end','replay','handling']))
    records = [FrameFeatures(
        t=float(t), observation_fps=float(encoder.spec['sample_fps']),
        observation_valid=True, table_observable=False, table_full_view=False,
        ball_kinematics_valid=False, motion_score=float(p[1]), strike_score=float(p[0]),
        state=ShotState.STRIKE_CANDIDATE if p[0]>=calibration['thresholds']['event'] else (
            ShotState.BALLS_MOVING if p[1]>=calibration['thresholds']['keep'] else ShotState.WAITING),
    ) for t,p in zip(timestamps, probabilities)]
    events = [TimelineEvent(event_type='cue_strike', timestamp=s.cue_strike,
        confidence=s.strike_confidence, metadata={'shot_id': s.shot_id, 'algorithm':'dl_algo',
                                                'estimated': True}) for s in shots]
    total = sum(s.duration() for s in shots if s.included)
    result = AnalysisResult(
        job_id=request['job_id'], source_path=str(source), proxy_path=str(source),
        metadata=metadata, features=records, strike_candidates=candidates, shots=shots,
        events=events, mode=EditMode(request['mode']), original_duration=metadata.duration,
        edited_duration=total, pause_removed_seconds=max(0., metadata.duration-total),
        analysis_signature=request['signature'],
        analysis_feature_signature=json.dumps(encoder.spec, sort_keys=True),
    )
    diagnostics.update(algorithm='dl_algo', feature_spec=encoder.spec,
                       elapsed_seconds=time.monotonic()-started,
                       head_validation_meaning='Availability of positive and negative calibration labels; not a deployment accuracy guarantee.',
                       calibrated_on=calibration.get('scope'),
                       independent_test_available=checkpoint['training'].get('independent_test_available',False),
                       uses_classic_detector=False, thresholds=calibration['thresholds'])
    atomic_text(job/'dl_diagnostics.json', json.dumps(diagnostics, indent=2))
    atomic_text(job/'analysis.json', result.model_dump_json(indent=2))
    timeline = result.model_dump(mode='json', exclude={'features'})
    atomic_text(job/'timeline.json', json.dumps(timeline, indent=2))
    emit(1., 'ready_for_review', f'DL Algo selected {sum(s.included for s in shots)} shots')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    run_request(json.loads(args.request.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
