"""Frozen neural RGB and optical-flow features from original video frames.

No classic detector, table mask, cue tracking or hand-coded strike score is used.
The learned visual backbone sees the full image and four higher-resolution tiles.
RAFT supplies spatial motion. Timestamp gaps reset motion pairs and temporal
context. Both backbone identities and preprocessing belong to the cache spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from snooker_ai.dl.dataset import load_manifest
from snooker_ai.dl.settings import dl_settings, expected_feature_spec, resolved_path

FEATURE_VERSION = 1


def annotation_intervals(video: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract every supervised head's coverage, including isolated weak clips."""
    rows = [row for name in ('reviewed_intervals', 'clip_reviewed_intervals', 'labels', 'clips')
            for row in video.get(name, [])]
    rows.extend({'start': contact['lower'], 'end': contact['upper']}
                for contact in video.get('contacts', []))
    return rows


def merge_intervals(intervals: list[dict[str, Any]], duration: float,
                    context_seconds: float = 17.) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for row in sorted(intervals, key=lambda r: float(r['start'])):
        lo = max(0., float(row['start']) - context_seconds)
        hi = min(duration, float(row['end']) + context_seconds)
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def source_frames(path: Path, intervals: list[tuple[float, float]],
                  sample_fps: float, resume_after: float = -math.inf) -> Iterator[tuple[float, np.ndarray]]:
    """Sample decoded source PTS, never an upsampled/downsampled analysis proxy."""
    import av

    if not 0 < sample_fps <= 60:
        raise ValueError('DL sampling FPS must be in (0, 60].')
    last_emitted = resume_after
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        origin = float((stream.start_time or 0) * stream.time_base)
        for start, end in intervals:
            if end <= resume_after:
                continue
            next_sample = start
            if resume_after >= start:
                next_sample = start+(math.floor((resume_after-start)*sample_fps+1e-8)+1)/sample_fps
            container.seek(max(0, int((next_sample + origin) / stream.time_base)),
                           stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise RuntimeError('Source frame has no presentation timestamp.')
                timestamp = float(frame.pts * frame.time_base) - origin
                if timestamp < start - 1e-6:
                    continue
                if timestamp > end + 1e-6:
                    break
                if timestamp + 1e-6 < next_sample or timestamp <= last_emitted:
                    continue
                next_sample += max(1, int((timestamp-next_sample)*sample_fps)+1)/sample_fps
                last_emitted = timestamp
                yield timestamp, frame.to_ndarray(format='rgb24')


class NeuralFeatures:
    def __init__(self, settings: dict[str, Any] | None = None,
                 assets: Path | None = None, device: str | None = None):
        import torch
        from safetensors.torch import load_file
        from torchvision.models.optical_flow import raft_small
        from transformers import Dinov2Model

        self.torch = torch
        self.settings = settings or dl_settings()
        self.assets = assets or resolved_path(self.settings['assets'])
        requested = device or self.settings.get('device', 'auto')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
        if self.device == 'cpu':
            torch.set_num_threads(int(self.settings.get('cpu_threads', 4)))
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('The selected DL CUDA runtime is unavailable.')
        identity = json.loads((self.assets / 'assets.json').read_text())
        if identity['rgb_model'] != self.settings['rgb_model']:
            raise ValueError('Cached RGB backbone differs from DL configuration; prepare matching assets.')
        self.rgb = Dinov2Model.from_pretrained(self.assets/'rgb', local_files_only=True,
                                             use_safetensors=True).eval().to(self.device)
        self.flow = raft_small(weights=None, progress=False).eval().to(self.device)
        self.flow.load_state_dict(load_file(str(self.assets/'raft-small.safetensors')))
        self.embedding_dim = int(self.rgb.config.hidden_size)
        self.spec = expected_feature_spec(self.settings, identity, self.rgb.config.to_dict())
        self.previous: tuple[float, Any] | None = None

    def _tensor(self, image: np.ndarray, size: tuple[int, int]):
        import cv2
        image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return self.torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255.

    def encode_batch(self, frames: list[tuple[float, np.ndarray]]) -> np.ndarray:
        torch = self.torch
        vectors = []
        views = []
        global_size = int(self.settings['global_size'])
        crop_size = int(self.settings['crop_size'])
        if global_size != crop_size:
            raise ValueError('This feature version requires equal global/crop dimensions.')
        if global_size % self.rgb.config.patch_size:
            raise ValueError('DINO image size must be divisible by its patch size.')
        for _, image in frames:
            height, width = image.shape[:2]
            hy, hx = height//2, width//2
            tiles = [image, image[:hy, :hx], image[:hy, hx:], image[hy:, :hx], image[hy:, hx:]]
            views.extend(self._tensor(tile, (global_size, global_size)) for tile in tiles)
        images = torch.stack(views).to(self.device)
        mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                                    enabled=self.device == 'cuda'):
            encoded = self.rgb(pixel_values=(images-mean)/std).last_hidden_state.float()
            cls = encoded[:, 0].view(len(frames), 5, self.embedding_dim)
            patch = encoded[::5, 1:].mean(dim=1)
            rgb_vectors = torch.cat([cls[:, 0], patch, cls[:, 1:].reshape(len(frames), -1)], dim=1).cpu().numpy()
        del images, encoded, cls, patch
        flow_h, flow_w = self.spec['flow_size']
        grid_h, grid_w = self.spec['flow_grid']
        for index, (timestamp, image) in enumerate(frames):
            current = self._tensor(image, (flow_h, flow_w)).to(self.device)
            motion = torch.zeros(6, grid_h, grid_w, device=self.device)
            if self.previous is not None:
                previous_t, previous_image = self.previous
                dt = timestamp-previous_t
                if 0 < dt <= 3/self.spec['sample_fps']:
                    with torch.inference_mode():
                        flow = self.flow(previous_image[None]*2-1, current[None]*2-1,
                                         num_flow_updates=self.spec['flow_updates'])[-1][0]
                        velocity = flow / torch.tensor([flow_w, flow_h], device=self.device)[:, None, None] / dt
                        global_motion = velocity.flatten(1).median(dim=1).values[:, None, None]
                        magnitude = torch.linalg.vector_norm(velocity, dim=0, keepdim=True)
                        residual = torch.linalg.vector_norm(velocity-global_motion, dim=0, keepdim=True)
                        maps = torch.cat([velocity, magnitude, residual])
                        mean_motion = torch.nn.functional.adaptive_avg_pool2d(maps, (grid_h, grid_w))
                        # Tiny moving balls must not disappear in average pooling.
                        peaks = torch.nn.functional.adaptive_max_pool2d(
                            torch.cat([magnitude, residual]), (grid_h, grid_w))
                        motion = torch.cat([mean_motion, peaks])
            self.previous = (timestamp, current)
            vectors.append(np.concatenate([rgb_vectors[index], motion.float().cpu().numpy().reshape(-1)]))
        result = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError('Neural feature extraction produced non-finite values.')
        return result

    def extract(self, source: Path, output: Path, duration: float,
                intervals: list[tuple[float, float]] | None = None,
                progress=None) -> tuple[np.ndarray, np.ndarray]:
        intervals = intervals or [(0., duration)]
        identity = {'path': str(source.resolve()), 'size': source.stat().st_size,
                    'mtime_ns': source.stat().st_mtime_ns, 'intervals': intervals}
        key = hashlib.sha256(json.dumps({'source': identity, 'spec': self.spec}, sort_keys=True).encode()).hexdigest()
        if output.exists():
            with np.load(output, allow_pickle=False) as stored:
                if 'fingerprint' in stored and str(stored['fingerprint'].item()) == key:
                    times = np.asarray(stored['timestamps'], dtype=np.float64)
                    values = np.asarray(stored['features'], dtype=np.float32)
                    self._validate_part(times, values)
                    return times, values
        output.parent.mkdir(parents=True, exist_ok=True)
        # A fully extracted training source can also be analyzed as a new job.
        # Reuse only an archive with exactly the same source identity, coverage,
        # model weights and preprocessing; partial reviewed sources cannot match.
        cache_setting = self.settings.get('feature_cache')
        if cache_setting:
            cache_dir = resolved_path(cache_setting)
            for cached in sorted(cache_dir.glob('*.npz')):
                if cached.resolve() == output.resolve() or cached.name.endswith('.partial.npz'):
                    continue
                with np.load(cached, allow_pickle=False) as stored:
                    if 'fingerprint' not in stored or str(stored['fingerprint'].item()) != key:
                        continue
                    times = np.asarray(stored['timestamps'], dtype=np.float64)
                    values = np.asarray(stored['features'], dtype=np.float32)
                    self._validate_part(times, values)
                temporary = output.with_name(output.stem+'.partial.npz')
                shutil.copyfile(cached, temporary)
                temporary.replace(output)
                if progress:
                    progress(1., f'Reused {len(times)} compatible learned observations')
                return times, values
        checkpoint_dir = output.parent / (output.stem+'.chunks') / key
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamps, parts, batch = [], [], []
        restored_state = None
        committed = sorted(path for path in checkpoint_dir.glob('chunk_*.npz')
                           if re.fullmatch(r'chunk_\d{5}\.npz', path.name))
        for index, path in enumerate(committed):
            if path.name != f'chunk_{index:05d}.npz':
                raise RuntimeError('DL feature checkpoint is missing a committed chunk.')
            with np.load(path, allow_pickle=False) as stored:
                chunk_times = np.asarray(stored['timestamps'], dtype=np.float64)
                chunk_features = np.asarray(stored['features'], dtype=np.float32)
                self._validate_part(chunk_times, chunk_features)
                if timestamps and chunk_times[0] <= timestamps[-1]:
                    raise RuntimeError('DL feature checkpoint is corrupt or out of order.')
                timestamps.extend(chunk_times.tolist())
                parts.append(chunk_features)
                restored_state = np.asarray(stored['previous_image'], dtype=np.float32)
                if (restored_state.shape != (3, *self.spec['flow_size'])
                        or not np.isfinite(restored_state).all()):
                    raise RuntimeError('DL feature checkpoint has an invalid motion state.')
        self.previous = None
        if restored_state is not None:
            self.previous = (timestamps[-1], self.torch.from_numpy(restored_state.copy()).to(self.device))
        persisted = len(timestamps)
        checkpoint_index = len(parts)
        def save_checkpoint():
            nonlocal persisted, checkpoint_index
            if len(timestamps)<=persisted or self.previous is None:
                return
            complete = np.concatenate(parts)
            temporary = checkpoint_dir/f'chunk_{checkpoint_index:05d}.partial.npz'
            state = self.previous[1].detach().float().cpu().numpy()
            np.savez_compressed(temporary, timestamps=np.asarray(timestamps[persisted:]),
                                features=complete[persisted:], previous_image=state)
            temporary.replace(checkpoint_dir/f'chunk_{checkpoint_index:05d}.npz')
            persisted = len(timestamps)
            checkpoint_index += 1
        total = sum(b-a for a, b in intervals)
        seen = len(timestamps)/self.spec['sample_fps']
        started = time.monotonic()
        if timestamps and progress:
            progress(min(1., seen/max(total,1)), f'Resumed {len(timestamps)} learned observations')
        for timestamp, frame in source_frames(source, intervals, self.spec['sample_fps'],
                                              timestamps[-1] if timestamps else -math.inf):
            batch.append((timestamp, frame))
            if len(batch) >= int(self.settings['batch_frames']):
                parts.append(self.encode_batch(batch))
                timestamps.extend(t for t, _ in batch)
                seen += len(batch)/self.spec['sample_fps']
                batch.clear()
                if len(timestamps)-persisted >= 256:
                    save_checkpoint()
                if progress:
                    progress(min(1., seen/max(total, 1)), f'Learned RGB/motion observations: {len(timestamps)}')
        if batch:
            parts.append(self.encode_batch(batch))
            timestamps.extend(t for t, _ in batch)
        save_checkpoint()
        if not parts:
            raise RuntimeError('No original source frames were decoded for DL Algo.')
        times, features = np.asarray(timestamps, dtype=np.float64), np.concatenate(parts)
        temporary = output.with_name(output.stem+'.partial.npz')
        np.savez_compressed(temporary, timestamps=times, features=features,
                            feature_spec=json.dumps(self.spec, sort_keys=True), fingerprint=key,
                            source_identity=json.dumps(identity, sort_keys=True),
                            elapsed_seconds=time.monotonic()-started)
        temporary.replace(output)
        return times, features

    def _validate_part(self, times: np.ndarray, values: np.ndarray) -> None:
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or values.ndim != 2
                or values.shape != (len(times), self.spec['dimension'])
                or not np.isfinite(values).all()):
            raise RuntimeError('DL feature cache is corrupt or out of order.')


def prepare_assets(destination: Path, rgb_model: str = 'facebook/dinov2-small') -> dict[str, Any]:
    """Fetch official public pretrained weights; inference stays local afterwards."""
    import torch
    from huggingface_hub import model_info
    from safetensors.torch import save_file
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    from transformers import Dinov2Model

    destination.mkdir(parents=True, exist_ok=True)
    revision = model_info(rgb_model).sha
    rgb = Dinov2Model.from_pretrained(rgb_model, revision=revision, use_safetensors=True)
    rgb.save_pretrained(destination/'rgb', safe_serialization=True)
    torch.hub.set_dir(str(destination/'torch-cache'))
    flow = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=True)
    path = destination/'raft-small.safetensors'
    save_file({k: v.contiguous() for k, v in flow.state_dict().items()}, str(path))
    identity = {'rgb_model': rgb_model, 'rgb_revision': revision,
                'flow_model': 'torchvision/raft_small/C_T_V2',
                'flow_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination/'assets.json').write_text(json.dumps(identity, indent=2))
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare neural RGB/flow features for DL Algo.')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--prepare-assets', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--only', help='One video id, otherwise all manifest videos')
    args = parser.parse_args()
    settings = dl_settings()
    assets = args.assets or resolved_path(settings['assets'])
    if args.prepare_assets:
        print(json.dumps(prepare_assets(assets, settings['rgb_model']), indent=2), flush=True)
    if args.manifest:
        manifest = load_manifest(args.manifest)
        base = (args.manifest.parent / manifest.get('paths_relative_to', '.')).resolve()
        encoder = NeuralFeatures(settings, assets, args.device)
        for video in manifest['videos']:
            if args.only and video['id'] != args.only:
                continue
            source = Path(video['source_path'])
            output = Path(video['features_path'])
            source = source if source.is_absolute() else base/source
            output = output if output.is_absolute() else base/output
            annotations = annotation_intervals(video)
            intervals = merge_intervals(annotations, video['duration'])
            print('FEATURE VIDEO', video['id'], intervals, flush=True)
            previous = [-1]
            def report(frac, message):
                percent = int(frac*100)
                if percent >= previous[0]+2:
                    print(f'{percent}% {message}', flush=True)
                    previous[0] = percent
            encoder.extract(source, output, video['duration'], intervals, progress=report)
            print('FEATURE READY', output, flush=True)


if __name__ == '__main__':
    main()
