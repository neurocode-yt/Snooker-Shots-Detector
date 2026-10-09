"""Explicit final development refit; retain a declared independent holdout."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--all-reviewed', action='store_true',
                        help='Refit train+dev groups; resulting calibration is in-sample')
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/evaluation/dl/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'models/dl_algo/temporal-v1.pt')
    parser.add_argument('--work-dir', type=Path, default=ROOT/'data/dl/training')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if not args.all_reviewed:
        parser.error('--all-reviewed is required: refit calibration is not validation')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    sys.path.insert(0, str(ROOT))
    import torch
    from snooker_ai.dl.dataset import load_manifest
    from snooker_ai.dl.temporal import (
        TemporalHighlightEnsemble, checkpoint_payload, load_temporal_checkpoint, predict_probabilities,
    )
    from snooker_ai.dl.training import TrainingConfig, train_temporal, _prepare_videos, _calibration
    from snooker_ai.dl.settings import dl_settings, resolved_path
    from tools.dl_train_pipeline import RunState, benchmark, atomic_json

    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    folder = args.work_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(load_manifest(args.manifest.resolve()))
    base = (args.manifest.resolve().parent/manifest.get('paths_relative_to', '.')).resolve()
    manifest['paths_relative_to'] = os.path.relpath(base, folder)
    for video in manifest['videos']:
        if video.get('split') != 'holdout':
            video['split'] = 'train'
    manifest['limitations'].append('Final refit uses train+dev groups; its calibration metrics are in-sample.')
    path = folder/'final-refit-manifest.json'
    atomic_json(path, manifest)
    state = RunState(folder/'final-fit-state.json', {'manifest': str(path), 'scope': 'development refit'})
    models, reports = [], []
    for spatial in (False, True):
        name = 'spatial' if spatial else 'flat'
        state.stage('training', f'Fitting {name} model on available training and development labels')
        output = folder/f'final-{name}.pt'
        configuration = TrainingConfig(epochs=args.epochs, patience=args.epochs, dropout=.1,
                                       spatial_flow=spatial)
        reports.append(train_temporal(path, output, config=configuration, device=device,
                                      allow_development_only=True))
        model, _ = load_temporal_checkpoint(output, device=device)
        models.append(model)
    videos, specification, fingerprint = _prepare_videos(path)
    calibration_videos = [video for video in videos if video.entry.get('split') != 'holdout']
    predictions = [[predict_probabilities(model, video.features, timestamps=video.timestamps,
                                         device=device) for video in calibration_videos] for model in models]
    best = None
    for weight in (0., .25, .5, .75, 1.):
        probabilities = [weight*a+(1-weight)*b for a, b in zip(*predictions)]
        calibration = _calibration(calibration_videos, probabilities, TrainingConfig())
        metric = calibration['event']
        rank = (metric['f2'], metric['recall'], metric['precision'])
        if best is None or rank > best[0]:
            best = (rank, weight, calibration)
    _, weight, calibration = best
    calibration.update(scope='development_training_only', used_for_model_selection=True)
    model = TemporalHighlightEnsemble(models, [weight, 1-weight]).eval()
    report = {
        'architecture': 'Neural RGB/flat-motion and RGB/spatial-motion candidate ensemble',
        'feature_spec': specification, 'calibration': calibration, 'dataset_fingerprint': fingerprint,
        'ensemble_weights': [weight, 1-weight], 'member_reports': reports,
        'independent_test_available': any(video.entry.get('split') == 'holdout' for video in videos),
        'deployment_quality_verified': False,
        'metric_scope': 'Train+dev groups participate in refitting; calibration metrics are in-sample.',
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, feature_spec=specification, dataset_fingerprint=fingerprint,
                                training={k: v for k, v in report.items() if k not in ('feature_spec', 'calibration')},
                                calibration=calibration)
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    atomic_json(output.with_suffix(output.suffix+'.report.json'), report)
    settings = dl_settings()
    settings['assets'] = str(resolved_path(settings['assets']))
    benchmark_args = Namespace(manifest=path, output=output, benchmarks=folder/'final-benchmarks', tolerance=.25)
    summary = benchmark(benchmark_args, settings, device, state)
    state.stage('complete', 'Development refit and source-window benchmarks complete', status='complete',
                benchmark_summary=str(summary), checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    print(json.dumps({'model': str(output), 'ensemble_weights': [weight, 1-weight],
                      'calibration_metrics': calibration['event'], 'scope': calibration['scope']}), flush=True)


if __name__ == '__main__':
    main()
