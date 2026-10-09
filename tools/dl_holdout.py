"""Evaluate a frozen DL checkpoint on match-disjoint, source-reviewed holdouts.

Inference only: thresholds, weights and normalization are never fitted here.
Run with the isolated DL interpreter after compatible source features exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snooker_ai.dl.dataset import evaluate_video, load_feature_video, load_manifest  # noqa: E402
from snooker_ai.dl.settings import dl_settings  # noqa: E402
from tools.dl_train_pipeline import atomic_json, file_hash, utc_now  # noqa: E402


def used_groups(training: dict[str, Any]) -> set[str]:
    """Include model-selection groups and every ensemble member's groups."""
    groups = set()
    splits = training.get('split_groups', {})
    for split in ('train', 'validation', 'dev'):
        groups.update(splits.get(split, []))
    for member in training.get('member_reports', []):
        groups.update(used_groups(member))
    return groups


def validate_holdout(manifest: dict, training: dict) -> set[str]:
    groups = used_groups(training)
    if not groups:
        raise ValueError('Checkpoint has no training/model-selection group provenance.')
    if not manifest.get('independent_holdout_available') or not manifest.get('videos'):
        raise ValueError('An explicitly declared nonempty independent holdout is required.')
    for video in manifest['videos']:
        for name in ('event', 'keep', 'end', 'replay', 'replays', 'handling'):
            if video.get(name):
                raise ValueError(f'{name} annotations must use the evaluated labels schema, not an ignored key.')
        if video.get('split') != 'holdout' or video['group_id'] in groups:
            raise ValueError(f'{video["id"]} overlaps fitting/model selection or is not a holdout.')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(video['id'])):
            raise ValueError('Video IDs must be safe filenames.')
    return groups


def checked_features(video: dict, base: Path):
    """Reject stale, wrong-source or incomplete whole-source feature caches."""
    import numpy as np

    source = (base/Path(video['source_path'])).resolve()
    path = (base/Path(video['features_path'])).resolve()
    times, features, spec = load_feature_video(path)
    with np.load(path, allow_pickle=False) as archive:
        identity = json.loads(str(archive['source_identity'].item()))
        fingerprint = str(archive['fingerprint'].item())
    stat = source.stat()
    expected = {'path': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'intervals': [[0., float(video['duration'])]]}
    declared = video.get('source_fingerprint', {})
    if (declared.get('size_bytes') != stat.st_size or declared.get('mtime_ns') != stat.st_mtime_ns
            or identity != expected):
        raise ValueError(f'Source identity/whole-source coverage mismatch for {video["id"]}.')
    key = hashlib.sha256(json.dumps({'source': expected, 'spec': spec}, sort_keys=True).encode()).hexdigest()
    cadence = 1/float(spec['sample_fps'])
    if (key != fingerprint or times[0] > cadence or
            times[-1] < float(video['duration'])-2*cadence or
            np.any(np.diff(times) > 2*cadence+1e-6)):
        raise ValueError(f'Incomplete or incompatible whole-source cache for {video["id"]}.')
    return times, features, spec, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--video-id', help='Evaluate one entry; other frozen holdouts remain untouched.')
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    args = parser.parse_args()
    import numpy as np
    import torch
    from snooker_ai.dl.pipeline import validate_calibration
    from snooker_ai.dl.selection import select_highlights
    from snooker_ai.dl.temporal import load_temporal_checkpoint, predict_probabilities

    manifest = load_manifest(args.manifest)
    manifest_hash = file_hash(args.manifest)
    checkpoint_hash = file_hash(args.checkpoint)
    if checkpoint_hash != manifest.get('expected_checkpoint_sha256'):
        raise ValueError('Checkpoint differs from the source-review plan. Freeze a separate test plan.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cpu':
        torch.set_num_threads(4)
    model, metadata = load_temporal_checkpoint(args.checkpoint, device=device)
    groups = validate_holdout(manifest, metadata['training'])
    settings = dl_settings()
    calibration = metadata['calibration']
    validate_calibration(calibration, settings['selection'])
    selection = dict(settings['selection'], head_validation=calibration['frame_heads'])
    base = (args.manifest.resolve().parent/manifest.get('paths_relative_to', '.')).resolve()
    videos = [v for v in manifest['videos'] if not args.video_id or v['id'] == args.video_id]
    if not videos:
        raise ValueError('Video ID is absent from the frozen manifest.')
    for video in videos:
        ident = video['id']
        times, features, spec, path = checked_features(video, base)
        if spec != metadata['feature_spec']:
            raise ValueError(f'Checkpoint feature specification differs for {ident}.')
        report_path = args.output_dir/f'{ident}.json'
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding='utf-8'))
            if (previous['checkpoint_sha256'] != checkpoint_hash or previous['manifest_sha256'] != manifest_hash):
                raise ValueError('Output already belongs to another test. Choose a new output directory.')
        probabilities = predict_probabilities(model, features, timestamps=times, device=device,
            chunk_frames=int(settings['temporal_chunk_frames']), temperatures=calibration['temperatures'])
        candidates, shots, diagnostics = select_highlights(times, probabilities, video['duration'],
                                                           selection, calibration['thresholds'])
        clips = [{'contact':s.cue_strike, 'start':s.clip_start, 'end':s.clip_end} for s in shots if s.included]
        metrics = evaluate_video(clips, video, tolerance_seconds=.25)
        strict = evaluate_video(clips, video, tolerance_seconds=0)
        report = {'video_id':ident, 'group_id':video['group_id'], 'split':'holdout',
            'independent_holdout': True, 'completed_at': utc_now(),
            'checkpoint_sha256':checkpoint_hash, 'manifest_sha256':manifest_hash,
            'fitting_and_model_selection_groups': sorted(groups),
            'source_path':video['source_path'], 'feature_path':str(path), 'feature_samples':len(times),
            'selected_clip_count':len(clips), 'selected_seconds':sum(c['end']-c['start'] for c in clips),
            'dl':metrics, 'dl_strict':strict, 'selection_settings':selection,
            'thresholds':calibration['thresholds'], 'temperatures':calibration['temperatures'],
            'clips':clips, 'shots':[s.model_dump(mode='json') for s in shots],
            'candidate_count':len(candidates), 'diagnostics':diagnostics,
            'labels_human_approved':manifest.get('labels_human_approved', False),
            'deployment_quality_verified':False,
            'limitations':manifest.get('limitations', [])+video.get('limitations', [])+
                ['Clip records evaluated; complete outcomes, rendered MP4s and frame-exact boundaries require separate review.']}
        if file_hash(args.manifest) != manifest_hash or file_hash(args.checkpoint) != checkpoint_hash:
            raise RuntimeError('Frozen inputs changed during inference.')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir/f'{ident}.probabilities.npz', timestamps=times,
                            probabilities=probabilities, checkpoint_sha256=checkpoint_hash,
                            manifest_sha256=manifest_hash)
        atomic_json(report_path, report)
        print(json.dumps({'video_id':ident, 'clips':len(clips),
            'metrics':{key:metrics[key] for key in ['true_positive','false_positive','false_negative',
                                                   'precision','recall','false_handling_clips']},
            'coverage':metrics['source_contact_coverage'], 'report_path':str(report_path)}), flush=True)


if __name__ == '__main__':
    main()
