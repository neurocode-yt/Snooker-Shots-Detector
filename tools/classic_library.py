"""Catalogue source footage for rule-based detector calibration and review.

Metadata is an inventory, not a shot annotation or accuracy measurement.
Obvious filename variants share a proposed split; visual review must confirm
match identity before any independently held-out accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}


def proposed_group(path: Path) -> str:
    """Group numbered copies/parts conservatively without calling them holdouts."""
    name = path.stem.casefold().strip()
    name = re.sub(r'\(\d+\)$', '', name).strip()
    name = re.sub(r'(?:\s+|_)p(?:art)?\s*\d+$', '', name).strip()
    name = re.sub(r'-[a-f0-9]{8}$', '', name)
    return re.sub(r'[^\w]+', '_', name).strip('_')


def proposed_split(group: str) -> str:
    # A stable group-level assignment avoids moving copies between iterations.
    value = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
    return 'reserved_test_candidate' if value % 5 == 0 else 'development_candidate'


def classify(relative: Path, duration: float) -> str:
    if relative.parts[0].casefold() == 'ad':
        return 'advertisement_or_promo_candidate'
    if len(relative.parts) > 1:
        return 'edited_source_candidate'
    return 'match_candidate' if duration >= 300 else 'short_source_candidate'


def inventory(directory: Path, *, workers: int = 2) -> dict:
    from snooker_ai.utils.ffmpeg import find_ffprobe

    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('The source library must be a directory.')
    ffprobe = find_ffprobe()
    paths = sorted((p for p in directory.rglob('*')
                    if p.is_file() and p.suffix.casefold() in EXTENSIONS),
                   key=lambda p: str(p.relative_to(directory)).casefold())

    def probe(path: Path) -> dict:
        relative = path.relative_to(directory)
        stat = path.stat()
        group = proposed_group(path)
        row = {'relative_path': relative.as_posix(), 'source_path': str(path),
               'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
               'proposed_group': group, 'proposed_split': proposed_split(group),
               'visually_reviewed': False, 'contacts_annotated': False}
        try:
            result = subprocess.run(
                [ffprobe, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            if result.returncode:
                raise ValueError(result.stderr.strip()[:500])
            metadata = json.loads(result.stdout)
            stream = next(s for s in metadata['streams'] if s.get('codec_type') == 'video')
            duration = float(metadata.get('format', {}).get('duration') or stream.get('duration') or 0)
            numerator, denominator = stream.get('avg_frame_rate', '0/1').split('/')
            fps = float(numerator) / float(denominator) if float(denominator) else 0.
            row.update(duration_seconds=duration, width=stream.get('width'),
                       height=stream.get('height'), fps=fps, codec=stream.get('codec_name'),
                       role=classify(relative, duration), probe_ok=True)
            if row['role'] == 'advertisement_or_promo_candidate':
                row['proposed_split'] = 'unassigned_negative_candidate'
        except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as error:
            row.update(probe_ok=False, probe_error=str(error), role='unreadable_source')
        return row

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        videos = list(executor.map(probe, paths))
    roles = sorted({v['role'] for v in videos})
    return {'inventory_version': 1, 'algorithm': 'classic', 'source_root': str(directory),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Recursive file inventory and metadata probing only; no accuracy claim.',
            'split_policy': 'Proposed filename-group split. Confirm underlying match identity and '
                            'prior development use before freezing any independent test set. '
                            'No automatic detections are treated as ground truth.',
            'summary': {'video_files': len(videos),
                        'readable_videos': sum(v['probe_ok'] for v in videos),
                        'duration_hours': sum(v.get('duration_seconds', 0) for v in videos) / 3600,
                        'roles': {role: sum(v['role'] == role for v in videos) for role in roles}},
            'videos': videos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    report = inventory(args.directory, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
