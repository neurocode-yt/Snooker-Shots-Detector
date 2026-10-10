"""Fit rule thresholds to reviewed native-frame contact cases, without DL.

This fits the contact detector only. A candidate still needs independent
whole-pipeline and rendered-highlight checks before changing app defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_dataset(manifest: dict) -> None:
    tolerance = float(manifest.get('contact_tolerance_seconds', .25))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Contact tolerance must be finite and nonnegative.')
    ownership = {}
    ids = set()
    for case in manifest['cases']:
        if case['id'] in ids:
            raise ValueError('Contact-case IDs must be unique.')
        ids.add(case['id'])
        split = case['split']
        if split not in {'development', 'selection', 'test'}:
            raise ValueError('Cases require an explicit development, selection or test split.')
        group = case['source_group']
        if not group:
            raise ValueError('Every case requires its underlying match group.')
        if ownership.setdefault(group, split) != split:
            raise ValueError('An underlying match cannot appear in multiple splits.')
        if not case.get('source_reviewed'):
            raise ValueError('Automatic predictions cannot supply contact ground truth.')
        if case.get('label_predictions_consulted') and split != 'development':
            raise ValueError('Prediction-assisted label adjudication belongs in development.')
        if not case.get('features_sha256'):
            raise ValueError('Pin extracted feature files before parameter selection.')
        lo, hi = case['window']['start'], case['window']['end']
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError('Each case requires a finite, ordered source interval.')
        for contact in case['contacts']:
            if not lo <= contact['lower'] <= contact['upper'] < hi:
                raise ValueError('Contact labels must lie inside their reviewed interval.')
    if 'development' not in ownership.values():
        raise ValueError('At least one development match is required.')


def candidate_overrides(grid: dict, config) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    for key in keys:
        default = config.get(key)
        if (not key.startswith('strike_fusion.') or not isinstance(default, (int, float))
                or isinstance(default, bool)):
            raise ValueError('Only existing numeric contact-detector settings can be fitted.')
        if not grid[key] or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                or not math.isfinite(v) or v < 0 for v in grid[key]):
            raise ValueError('Grid values must be finite and nonnegative.')
    if math.prod(len(grid[key]) for key in keys) > 256:
        raise ValueError('Keep calibration grids to at most 256 candidates.')
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def nested(overrides: dict) -> dict:
    result = {}
    for key, value in overrides.items():
        node = result
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def evaluate_cases(cases: list[dict], config, root: Path, *, tolerance: float,
                   apply_replay_filter: bool = False) -> dict:
    from snooker_ai.evaluation.recall import evaluate_broadcast_recall
    from snooker_ai.event_fusion.strike import StrikeDetector
    from snooker_ai.replay_detection.detector import ReplayDetector
    from snooker_ai.types import FrameFeatures

    details = []
    for case in cases:
        path = root / case['features_path']
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != case['features_sha256']:
            raise ValueError(f"Feature evidence changed: {case['id']}")
        data = json.loads(content)
        features = [FrameFeatures.model_validate(f) for f in data['features']]
        start, end = case['window']['start'], case['window']['end']
        native_fps = float(case.get('native_fps', 25.))
        max_gap = 2/native_fps if math.isfinite(native_fps) and native_fps > 0 else 0.
        observed = [f for f in features if start <= f.t <= end]
        if (max_gap <= 0 or len(observed) < 2 or observed[0].t-start > max_gap
                or end-observed[-1].t > max_gap
                or any(b.t <= a.t or b.t-a.t > max_gap+1e-6
                       for a, b in zip(observed, observed[1:]))
                or any(f.observation_fps+1e-3 < native_fps for f in observed)):
            raise ValueError(f"Incomplete native-frame evidence: {case['id']}")
        detector = StrikeDetector(config)
        detector.score_frames(features)
        candidates = detector.detect_candidates(features)
        if apply_replay_filter:
            ReplayDetector(config).mark_candidates(candidates, features)
        window = {'name': case['id'], **case['window'], 'contacts': case['contacts']}
        result = evaluate_broadcast_recall(
            [{'cue_strike': c.timestamp, 'possible_replay': c.possible_replay} for c in candidates],
            {'windows': [window]}, tolerance=tolerance,
        )
        result['prediction_scope'] = ('native_contact_candidates_after_replay_filter'
                                      if apply_replay_filter else 'native_contact_candidates')
        details.append({'id': case['id'], 'source_group': case['source_group'], **result})
    tp = sum(c['matched'] for c in details)
    fp = sum(c['false_positive'] for c in details)
    fn = sum(c['missed'] for c in details)
    denominator = 5*tp + 4*fn + fp
    return {'matched': tp, 'false_positive': fp, 'missed': fn,
            'f2': 5*tp/denominator if denominator else 1., 'cases': details}


def fit(manifest: dict, grid: dict, config, root: Path, *, progress=None) -> dict:
    from snooker_ai.config import Config, deep_merge

    validate_dataset(manifest)
    # Reserved test features are never opened by fitting or candidate selection.
    development = [c for c in manifest['cases'] if c['split'] == 'development']
    selection = [c for c in manifest['cases'] if c['split'] == 'selection']
    tolerance = float(manifest.get('contact_tolerance_seconds', .25))
    candidates = [{}] + [c for c in candidate_overrides(grid, config) if c]
    reports = []
    for overrides in candidates:
        cfg = Config(deep_merge(config.as_dict(), nested(overrides)))
        dev = evaluate_cases(development, cfg, root, tolerance=tolerance)
        selected = evaluate_cases(selection, cfg, root, tolerance=tolerance) if selection else dev
        reports.append({'overrides': overrides, 'development': dev,
                        'selection': selected if selection else None,
                        'selection_score': selected['f2'], 'selection_false_positive': selected['false_positive'],
                        'selection_missed': selected['missed']})
        if progress:
            progress(len(reports), len(candidates), reports[-1])
    baseline = reports[0]['development']
    eligible = [r for r in reports if r['development']['matched'] >= baseline['matched']
                and r['development']['false_positive'] <= baseline['false_positive']]
    winner = max(eligible, key=lambda r: (r['selection_score'], -r['selection_false_positive'],
                                       -r['selection_missed'], -len(r['overrides'])))
    return {'algorithm': 'classic', 'uses_deep_learning': False,
            'scope': 'Reviewed native-feature contact cases only; whole highlights still require validation.',
            'test_data_consulted': False, 'deployment_quality_verified': False,
            'selection_policy': 'Preserve development recall and false-positive count, then maximize '
                                'selection F2; retain defaults on equal scores.',
            'contact_tolerance_seconds': tolerance, 'selected_overrides': winner['overrides'],
            'selected_candidate': reports.index(winner), 'candidates': reports,
            'groups': {split: sorted({c['source_group'] for c in manifest['cases'] if c['split'] == split})
                       for split in ['development', 'selection', 'test']}}


def main() -> None:
    from snooker_ai.config import load_config
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-config', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    def progress(done, total, result):
        print(f"Candidate {done}/{total}: development "
              f"{result['development']['matched']} contacts, "
              f"{result['development']['false_positive']} false; selection "
              f"F2={result['selection_score']:.4f}", flush=True)
    report = fit(json.loads(raw), json.loads(args.grid.read_text(encoding='utf-8')), load_config(), ROOT,
                 progress=progress)
    report['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    args.candidate_config.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_config.write_text(yaml.safe_dump(nested(report['selected_overrides'])), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['selected_overrides', 'groups', 'test_data_consulted']}))


if __name__ == '__main__':
    main()
