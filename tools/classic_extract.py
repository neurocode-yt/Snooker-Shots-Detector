"""Extract original-rule native-frame evidence from source-reviewed sections."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(annotations: Path, output: Path, config, *, source_root: Path | None = None) -> None:
    from snooker_ai.ingestion.probe import probe_video
    from snooker_ai.pipeline.analyzer import Analyzer, _CACHE_VERSION
    from snooker_ai.utils.ffmpeg import find_ffmpeg, run_command
    from snooker_ai.utils.timebase import TimeMapper

    raw = annotations.read_bytes()
    labels = json.loads(raw)
    if not labels.get('sections'):
        raise ValueError('Extraction requires source-reviewed section labels.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'algorithm': 'classic', 'uses_deep_learning': False,
                'annotations_sha256': hashlib.sha256(raw).hexdigest(),
                'scope': 'Native contact detector; whole-pipeline and rendered quality are separate checks.',
                'contact_tolerance_seconds': .25, 'cases': []}
    signature = hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest()
    for section in labels['sections']:
        if not section.get('source_reviewed'):
            raise ValueError('Every extraction interval must be explicitly source-reviewed.')
        consulted = section.get('predictions_consulted', labels.get('predictions_consulted', False))
        if consulted and section['split'] != 'development':
            raise ValueError('Prediction-assisted label adjudication belongs in development, not reserved matches.')
        source = (source_root/section['source_relative_path'] if source_root is not None
                  else Path(section['source_path']))
        if sha256(source) != section['source_sha256']:
            raise ValueError(f'Source footage changed: {source.name}')
        metadata = probe_video(source)
        identifier = f"source-{section['library_index']:02d}"
        folder = output/identifier
        folder.mkdir(parents=True, exist_ok=True)
        evidence = folder/'features.json'
        start, end = section['window']['start'], section['window']['end']
        if not 0 <= start < end <= metadata.duration:
            raise ValueError('The reviewed interval must lie inside its source video.')
        identity = {'source_sha256': section['source_sha256'], 'window': section['window'],
                    'context_start': max(0., start-1.),
                    'config_sha256': signature, 'feature_cache_version': _CACHE_VERSION}
        existing = json.loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else None
        if existing is None or existing.get('identity') != identity:
            # Only the geometry/cadence of this tiny proxy is used. All actual
            # contact images are decoded from the raw source at its native PTS.
            proxy = folder/'geometry-proxy.mp4'
            width = int(config.get('proxy.max_width', 960))
            height = int(config.get('proxy.max_height', 540))
            run_command([find_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                         '-ss', str(start), '-i', str(source), '-t', '1', '-an',
                         '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,setsar=1',
                         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-threads', '2',
                         str(proxy)], timeout=120)
            analyzer = Analyzer(config, folder/'extraction')
            analyzer._native_source = source
            mapper = TimeMapper(source_duration=metadata.duration, source_fps=metadata.fps,
                                analysis_fps=metadata.fps)
            print(f'{identifier}: extracting [{start}, {end}) at {metadata.fps:.3f} fps', flush=True)
            features, _, _ = analyzer._extract_features(
                proxy, None, mapper, metadata.duration, sample_fps=metadata.fps,
                start_time=identity['context_start'], end_time=end,
            )
            evidence.write_text(json.dumps({'identity': identity,
                                           'features': [f.model_dump() for f in features]}), encoding='utf-8')
            print(f'{identifier}: saved {len(features)} native observations', flush=True)
        try:
            evidence_path = str(evidence.resolve().relative_to(ROOT))
        except ValueError:
            evidence_path = str(evidence.resolve())
        manifest['cases'].append({
            'id': identifier, 'source_group': section['source_group'], 'split': section['split'],
            'source_reviewed': True, 'labels_human_approved': labels.get('labels_human_approved', False),
            'label_predictions_consulted': consulted,
            'source_sha256': section['source_sha256'], 'source_path': str(source),
            'features_path': evidence_path,
            'features_sha256': sha256(evidence), 'native_fps': metadata.fps,
            'window': section['window'], 'contacts': section['contacts'],
        })
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def main() -> None:
    from snooker_ai.config import load_config
    from snooker_ai.utils.logging import setup_logging
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path,
                        help='Resolve portable benchmark source filenames under this directory.')
    args = parser.parse_args()
    setup_logging('INFO')
    extract(args.annotations, args.output, load_config(), source_root=args.source_root)


if __name__ == '__main__':
    main()
