"""Render a saved DL selection and check decoded source-to-output contacts.

This checks delivery and timestamp mapping. It cannot verify missing strokes,
complete outcomes, referee exclusion or model accuracy by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dl_train_pipeline import atomic_json, file_hash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from snooker_ai.config import load_config
    from snooker_ai.dl.features import source_frames
    from snooker_ai.ingestion.probe import probe_video
    from snooker_ai.rendering.exporter import Exporter
    from snooker_ai.types import AnalysisResult, EditMode, ExportRequest, ShotRecord
    from snooker_ai.utils.ffmpeg import find_ffmpeg

    report_hash = file_hash(args.report)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    source = (ROOT/Path(report['source_path'])).resolve()
    shots = [ShotRecord.model_validate(row) for row in report['shots']]
    included = [shot for shot in shots if shot.included]
    if not included:
        raise ValueError('Report has no included shots to render.')
    metadata = probe_video(source)
    result = AnalysisResult(job_id=f'dl-audit-{report["video_id"]}', source_path=str(source),
        proxy_path=str(source), metadata=metadata, shots=shots, mode=EditMode.STRICT,
        original_duration=metadata.duration, edited_duration=report['selected_seconds'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = Exporter(load_config()).export(result, args.output_dir,
        ExportRequest(mode=EditMode.STRICT, export_clips=False, export_joined=True))
    decoded = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(export.joined_path),
                              '-f', 'null', '-'], capture_output=True, text=True)
    output_metadata = probe_video(export.joined_path)
    mapping = json.loads(export.metadata_path.read_text(encoding='utf-8'))
    starts = mapping['output_shot_starts'] or np.cumsum(
        [0.]+[shot.duration() for shot in included[:-1]]).tolist()
    if len(starts) != len(included):
        raise ValueError('Export timestamp mapping does not cover each delivered clip.')
    similarities, records, pairs = [], [], []
    for index, shot in enumerate(included):
        for offset in (0., .4):
            source_t = shot.cue_strike+offset
            if source_t >= shot.clip_end:
                continue
            output_t = starts[index]+source_t-shot.clip_start
            source_obs = next(source_frames(source, [(source_t, source_t+.2)], metadata.fps))
            output_obs = next(source_frames(export.joined_path, [(output_t, output_t+.2)], output_metadata.fps))
            left = np.asarray(Image.fromarray(source_obs[1]).resize((640, 360)))
            right = np.asarray(Image.fromarray(output_obs[1]).resize((640, 360)))
            similarity = float(cv2.PSNR(left, right))
            similarities.append(similarity)
            records.append({'prediction_index': index, 'source_pts': source_obs[0],
                            'output_pts': output_obs[0], 'psnr_db': similarity})
            pairs.extend([(f'Source {source_obs[0]:.3f}s', left),
                          (f'Output {output_obs[0]:.3f}s', right)])
        print(f'Compared delivered contact {index+1}/{len(included)}', flush=True)
    for page, start in enumerate(range(0, len(pairs), 16)):
        items = pairs[start:start+16]
        canvas = Image.new('RGB', (1280, 384*((len(items)+1)//2)))
        draw = ImageDraw.Draw(canvas)
        for index, (label, pixels) in enumerate(items):
            x, y = index % 2*640, index//2*384
            draw.text((x+8, y+4), label, fill='white')
            canvas.paste(Image.fromarray(pixels), (x, y+24))
        canvas.save(args.output_dir/f'contacts_{page:03d}.jpg', quality=94)
    if file_hash(args.report) != report_hash:
        raise RuntimeError('Saved selection changed during export audit.')
    audit = {'video_id': report['video_id'], 'selection_report_sha256': report_hash,
        'checkpoint_sha256': report['checkpoint_sha256'], 'output_path': str(export.joined_path),
        'ffmpeg_decode_exit_code': decoded.returncode, 'decode_errors': decoded.stderr,
        'output_duration': output_metadata.duration, 'output_fps': output_metadata.fps,
        'compared_contact_frames': len(similarities), 'minimum_psnr_db': min(similarities),
        'contact_frame_comparisons': records, 'complete_shot_outcomes_verified': False,
        'deployment_quality_verified': False}
    atomic_json(args.output_dir/'render_audit.json', audit)
    print(json.dumps({key: value for key, value in audit.items()
                     if key != 'contact_frame_comparisons'}), flush=True)


if __name__ == '__main__':
    main()
