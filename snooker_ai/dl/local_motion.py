"""Experimental dense neural motion features, separate from deployed DL caches.

This extractor reuses the frozen RAFT-small backbone at higher image resolution.
It preserves a local motion grid and the direction of each cell's strongest
residual. It does not decide whether a shot occurred, detect a table, or call
the classic pipeline. Higher-resolution motion alone is not validated accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalMotionConfig:
    height: int = 576
    width: int = 1024
    grid_height: int = 36
    grid_width: int = 64
    updates: int = 6
    tile_rows: int = 1
    tile_columns: int = 1

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError('Local motion dimensions and updates must be positive integers.')
        if (min(self.height, self.width) < 128 or self.height % 8 or self.width % 8
                or self.grid_height > self.height or self.grid_width > self.width
                or self.height % self.tile_rows or self.width % self.tile_columns
                or self.height//self.tile_rows < 128 or self.width//self.tile_columns < 128
                or (self.height//self.tile_rows) % 8 or (self.width//self.tile_columns) % 8):
            raise ValueError('RAFT dimensions must be multiples of eight and contain the grid.')

    @property
    def dimension(self) -> int:
        return 8 * self.grid_height * self.grid_width


def pool_local_motion(velocity: Any, grid: tuple[int, int]) -> Any:
    """Pool normalized learned velocity [2,H,W] without discarding peak direction.

    The first six channels match the existing flow feature semantics; channels
    seven and eight contain x/y velocity at the maximum residual pixel in each
    cell. All channels have units of fraction of frame per second. A single
    peak location supplies both directions, avoiding incompatible componentwise
    maxima. Ties deterministically select PyTorch's first maximum.
    """
    import torch
    from torch.nn import functional as functional

    if velocity.ndim != 3 or velocity.shape[0] != 2:
        raise ValueError('Expected normalized velocity [2,height,width].')
    height, width = velocity.shape[-2:]
    if (len(grid) != 2 or min(grid) < 1 or grid[0] > height or grid[1] > width
            or not bool(torch.isfinite(velocity).all())):
        raise ValueError('Local motion grid must fit finite velocity dimensions.')
    global_motion = velocity.flatten(1).median(dim=1).values[:, None, None]
    magnitude = torch.linalg.vector_norm(velocity, dim=0, keepdim=True)
    residual = torch.linalg.vector_norm(velocity-global_motion, dim=0, keepdim=True)
    means = functional.adaptive_avg_pool2d(torch.cat((velocity, magnitude, residual)), grid)
    peaks = functional.adaptive_max_pool2d(torch.cat((magnitude, residual)), grid)
    _, indices = functional.adaptive_max_pool2d(residual, grid, return_indices=True)
    directions = velocity.flatten(1).gather(1, indices.flatten().expand(2, -1))
    return torch.cat((means, peaks, directions.reshape(2, *grid)))


class LocalMotionEncoder:
    """Frozen high-resolution optical flow; optional torch imports stay local."""

    def __init__(self, assets: Path, config: LocalMotionConfig | None = None,
                 device: str = 'auto') -> None:
        import torch
        from safetensors.torch import load_file
        from torchvision.models.optical_flow import raft_small

        self.torch = torch
        self.config = config or LocalMotionConfig()
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Local motion device must be auto, cpu or cuda.')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable for the experimental motion extractor.')
        path = Path(assets) / 'raft-small.safetensors'
        identity = json.loads((Path(assets)/'assets.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (identity.get('flow_model') != 'torchvision/raft_small/C_T_V2'
                or identity.get('flow_sha256') != digest):
            raise ValueError('Experimental RAFT weights differ from the prepared backbone identity.')
        self.model = raft_small(weights=None, progress=False).eval().to(self.device)
        self.model.load_state_dict(load_file(str(path)))
        self.spec = {
            'version': 1, 'experimental': True, 'backbone': 'torchvision/raft_small/C_T_V2',
            'weights_sha256': digest,
            **asdict(self.config), 'dimension': self.config.dimension,
            'channels': ['mean_velocity_x', 'mean_velocity_y', 'mean_magnitude',
                         'mean_global_residual', 'max_magnitude', 'max_global_residual',
                         'residual_peak_velocity_x', 'residual_peak_velocity_y'],
            'units': 'fraction_of_frame_per_second',
            'source_time': 'native_video_pts',
        }

    def _tensor(self, image: np.ndarray) -> Any:
        import cv2

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected an RGB uint8 image.')
        resized = cv2.resize(image, (self.config.width, self.config.height),
                             interpolation=cv2.INTER_AREA)
        return self.torch.from_numpy(resized.copy()).permute(2, 0, 1).float().to(self.device)/127.5-1

    def velocity(self, previous: np.ndarray, current: np.ndarray, dt: float) -> Any:
        """Return learned normalized flow for one pair; caller controls continuity."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Source frame time difference must be finite and positive.')
        if previous.shape != current.shape:
            raise ValueError('Motion pairs must have identical source dimensions.')
        with self.torch.inference_mode():
            before, after = self._tensor(previous), self._tensor(current)
            rows, columns = self.config.tile_rows, self.config.tile_columns
            if rows*columns == 1:
                flow = self.model(before[None], after[None],
                                  num_flow_updates=self.config.updates)[-1][0]
            else:
                # Fixed tiles reduce RAFT's quadratic all-pairs correlation cost.
                # Motion cannot cross tile boundaries in this experimental mode;
                # retain the full-frame branch or overlapping crops if required.
                tile_h, tile_w = self.config.height//rows, self.config.width//columns
                def tiles(value):
                    return value.reshape(3, rows, tile_h, columns, tile_w).permute(
                        1, 3, 0, 2, 4).reshape(rows*columns, 3, tile_h, tile_w)
                tile_flow = self.model(tiles(before), tiles(after),
                                       num_flow_updates=self.config.updates)[-1]
                flow = tile_flow.reshape(rows, columns, 2, tile_h, tile_w).permute(
                    2, 0, 3, 1, 4).reshape(2, self.config.height, self.config.width)
            scale = self.torch.tensor([self.config.width, self.config.height], device=self.device)
            result = flow / scale[:, None, None] / dt
        if not bool(self.torch.isfinite(result).all()):
            raise RuntimeError('Learned optical flow produced non-finite values.')
        return result

    def encode_pair(self, previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
        velocity = self.velocity(previous, current, dt)
        pooled = pool_local_motion(velocity, (self.config.grid_height, self.config.grid_width))
        return pooled.detach().float().cpu().numpy().reshape(-1)
