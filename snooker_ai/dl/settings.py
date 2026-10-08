"""Independent configuration and runtime locations for DL Algo."""
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import yaml

from snooker_ai.config import Config, deep_merge

ROOT = Path(__file__).resolve().parents[2]


def dl_settings(config: Config | None = None) -> dict[str, Any]:
    path = ROOT / 'configs' / 'dl_algo.yaml'
    base = yaml.safe_load(path.read_text(encoding='utf-8'))['dl_algo']
    override = config.get('dl_algo', {}) if config is not None else {}
    if not isinstance(override, dict):
        raise ValueError('DL Algo configuration must be a mapping.')
    result = deep_merge(base, override)
    if os.name != 'nt' and result.get('python') == '.venv-dl/Scripts/python.exe':
        result['python'] = '.venv-dl/bin/python'
    return result


def expected_feature_spec(settings: dict, identity: dict, rgb_config: dict) -> dict:
    """Describe version-1 features without importing an optional ML runtime."""
    if identity['rgb_model'] != settings['rgb_model'] or rgb_config['model_type'] != 'dinov2':
        raise ValueError('Cached RGB backbone differs from DL configuration.')
    if identity['flow_model'] != 'torchvision/raft_small/C_T_V2':
        raise ValueError('Unsupported cached flow backbone.')

    def positive_integer(value):
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number < 1 or int(number) != number:
            raise ValueError('DL feature dimensions and update counts must be positive integers.')
        return int(number)

    size = positive_integer(settings['global_size'])
    crop = positive_integer(settings['crop_size'])
    patch = positive_integer(rgb_config['patch_size'])
    height, width = positive_integer(settings['flow_height']), positive_integer(settings['flow_width'])
    grid_h = positive_integer(settings['flow_grid_height'])
    grid_w = positive_integer(settings['flow_grid_width'])
    fps = float(settings['sample_fps'])
    if (size != crop or size % patch or height % 8 or width % 8 or min(height, width) < 128
            or not math.isfinite(fps) or not 0 < fps <= 60):
        raise ValueError('Unsupported DL feature preprocessing dimensions or sampling rate.')
    return {
        'version': 1, 'rgb_model': identity['rgb_model'], 'rgb_revision': identity['rgb_revision'],
        'flow_model': identity['flow_model'], 'flow_sha256': identity['flow_sha256'],
        'global_size': size, 'crop_size': crop,
        'rgb_views': 'global_cls,global_patch_mean,tl,tr,bl,br', 'sample_fps': fps,
        'flow_size': [height, width], 'flow_updates': positive_integer(settings['flow_updates']),
        'flow_grid': [grid_h, grid_w],
        'flow_channels': 'mean_velocity_x,mean_velocity_y,mean_magnitude,mean_global_residual,max_magnitude,max_global_residual',
        'flow_units': 'fraction_of_frame_per_second', 'source_time': 'video_pts_zero_origin',
        'dimension': 6*positive_integer(rgb_config['hidden_size']) + 6*grid_h*grid_w,
    }


def resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
