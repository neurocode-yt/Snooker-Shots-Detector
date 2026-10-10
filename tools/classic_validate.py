"""Validate pinned native contact evidence with the original replay filter.

This is a candidate-stage benchmark. Full pipeline selection, clip endpoints,
and rendered output require separate checks against source footage.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    from snooker_ai.config import load_config
    from snooker_ai.pipeline.analyzer import _CACHE_VERSION
    from tools.classic_calibrate import evaluate_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    for case in manifest['cases']:
        data = json.loads((ROOT / case['features_path']).read_text(encoding='utf-8'))
        identity = data.get('identity', {})
        if (identity.get('feature_cache_version') != _CACHE_VERSION
                or identity.get('source_sha256') != case['source_sha256']
                or identity.get('window') != case['window']):
            raise ValueError(f"Re-extract current source evidence: {case['id']}")
    report = evaluate_cases(manifest['cases'], load_config(), ROOT,
        tolerance=float(manifest.get('contact_tolerance_seconds', .25)), apply_replay_filter=True)
    report.update(algorithm='classic', uses_deep_learning=False,
        labels_human_approved=all(c.get('labels_human_approved', False) for c in manifest['cases']),
        feature_cache_version=_CACHE_VERSION,
        scope='Native contact candidates after original replay filtering; full pipeline and render assessed separately.',
        annotations_sha256=manifest.get('annotations_sha256'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print({k:report[k] for k in ('matched','missed','false_positive')})


if __name__ == '__main__':
    main()
