"""Learned temporal decisions for the isolated DL Algo backend.

The RGB/flow backbone is deliberately outside this module. This network accepts
only learned visual embeddings; it never calls the classic shot detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Classic installations do not need the optional DL runtime.
    torch = None
    nn = None


HEAD_NAMES = ("event", "keep", "end", "replay", "handling")
CHECKPOINT_VERSION = 1


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "DL Algo needs its optional PyTorch runtime. "
            "Install the project DL dependencies or use the configured DL interpreter."
        )
    return torch


@dataclass(frozen=True)
class TemporalModelConfig:
    input_dim: int
    hidden_dim: int = 96
    local_dilations: tuple[int, ...] = (1, 2, 4)
    context_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.15
    flow_grid: tuple[int, ...] = ()
    flow_channels: int = 6
    use_rgb: bool = True
    rgb_dropout: float = 0.0
    rgb_clip: float = 0.0
    motion_translate_cells: int = 0
    motion_scale_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 4:
            raise ValueError("Temporal feature and hidden dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not self.local_dilations or not self.context_dilations:
            raise ValueError("Both temporal branches need at least one dilation.")
        if any(d < 1 for d in self.local_dilations + self.context_dilations):
            raise ValueError("Temporal dilations must be positive.")
        if self.flow_grid and (len(self.flow_grid) != 2 or min(self.flow_grid) < 3
                               or self.flow_channels < 1 or self.rgb_dim < 1):
            raise ValueError('Spatial motion geometry must leave a positive RGB feature dimension.')
        if (not np.isfinite([self.rgb_dropout, self.rgb_clip, self.motion_scale_jitter]).all()
                or not 0 <= self.rgb_dropout < 1 or self.rgb_clip < 0
                or not 0 <= self.motion_scale_jitter <= 2
                or not isinstance(self.use_rgb, bool)
                or isinstance(self.motion_translate_cells, bool)
                or not isinstance(self.motion_translate_cells, int) or self.motion_translate_cells < 0):
            raise ValueError('Invalid neural modality/augmentation settings.')
        if not self.flow_grid and (not self.use_rgb or self.rgb_dropout or self.rgb_clip
                                   or self.motion_translate_cells or self.motion_scale_jitter):
            raise ValueError('Modality and motion augmentation require spatial motion geometry.')

    @property
    def rgb_dim(self) -> int:
        return self.input_dim - int(np.prod(self.flow_grid))*self.flow_channels if self.flow_grid else self.input_dim

    @property
    def receptive_field(self) -> int:
        # Two kernel-3 convolutions in each residual block.
        return 1 + 4 * max(sum(self.local_dilations), sum(self.context_dilations))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TemporalModelConfig":
        values = dict(values)
        for key in ("local_dilations", "context_dilations", "flow_grid"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


if nn is not None:

    class _TemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(channels)
            self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.norm2 = nn.LayerNorm(channels)
            self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, value: Any) -> Any:
            residual = value
            value = self.norm1(value.transpose(1, 2)).transpose(1, 2)
            value = self.dropout(self.activation(self.conv1(value)))
            value = self.norm2(value.transpose(1, 2)).transpose(1, 2)
            value = self.dropout(self.activation(self.conv2(value)))
            return residual + value


class TemporalHighlightNet(nn.Module if nn is not None else object):
    """A multi-scale residual TCN emitting five learned logits per source sample."""

    def __init__(self, config: TemporalModelConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.input_dim))
        self.register_buffer("feature_std", torch.ones(config.input_dim))
        if config.flow_grid:
            # Shared spatial filters recognize the same local motion pattern
            # anywhere on the table, rather than memorizing flattened cells.
            self.rgb_projection = nn.Sequential(nn.Linear(config.rgb_dim, config.hidden_dim),
                                                nn.LayerNorm(config.hidden_dim), nn.GELU())
            self.motion_encoder = nn.Sequential(nn.Conv2d(config.flow_channels, 32, 3, padding=1),
                                                nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
            self.motion_projection = nn.Sequential(nn.Linear(64, config.hidden_dim),
                                                   nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim*2 if config.flow_grid else config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            *[_TemporalBlock(config.hidden_dim, d, config.dropout) for d in config.local_dilations]
        )
        self.context = nn.Sequential(
            *[_TemporalBlock(config.hidden_dim, d, config.dropout) for d in config.context_dilations]
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(HEAD_NAMES)),
        )

    def set_feature_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (self.config.input_dim,) or std.shape != mean.shape:
            raise ValueError("Feature normalization does not match model input dimension.")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Feature normalization must be finite with positive scales.")
        self.feature_mean.copy_(torch.as_tensor(mean, device=self.feature_mean.device))
        self.feature_std.copy_(torch.as_tensor(std, device=self.feature_std.device))

    def forward(self, features: Any) -> Any:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("Expected features [batch, samples, input_dim].")
        normalized = (features - self.feature_mean) / self.feature_std
        if self.config.flow_grid:
            batch, frames = features.shape[:2]
            appearance = normalized[..., :self.config.rgb_dim]
            if self.config.rgb_clip:
                appearance = appearance.clamp(-self.config.rgb_clip, self.config.rgb_clip)
            if not self.config.use_rgb:
                appearance = torch.zeros_like(appearance)
            elif self.training and self.config.rgb_dropout:
                # One decision for the entire sequence: avoid inventing frame cuts.
                retained = (torch.rand((batch,1,1), device=features.device) >= self.config.rgb_dropout)
                appearance = appearance * retained / (1-self.config.rgb_dropout)
            rgb = self.rgb_projection(appearance)
            motion_features = features[..., self.config.rgb_dim:]
            if self.training and self.config.motion_scale_jitter:
                gain = torch.exp(torch.empty((batch,1,1), device=features.device).uniform_(
                    -self.config.motion_scale_jitter, self.config.motion_scale_jitter))
                motion_features = motion_features * gain
            spatial = ((motion_features-self.feature_mean[self.config.rgb_dim:]) /
                       self.feature_std[self.config.rgb_dim:]).reshape(
                batch, frames, self.config.flow_channels, *self.config.flow_grid)
            if self.training and self.config.motion_translate_cells:
                # Apply the same translation throughout each source sequence.
                # This preserves event timing and local velocity semantics.
                radius = self.config.motion_translate_cells
                shifted=[]
                for sequence in spatial:
                    shifts=torch.randint(-radius,radius+1,(2,),device=features.device).tolist()
                    shifted.append(torch.roll(sequence, shifts=shifts, dims=(-2,-1)))
                spatial=torch.stack(shifted)
            spatial=spatial.reshape(batch*frames,self.config.flow_channels,*self.config.flow_grid)
            spatial = self.motion_encoder(spatial)
            pooled = torch.cat((spatial.mean(dim=(2, 3)), spatial.amax(dim=(2, 3))), dim=1)
            motion = self.motion_projection(pooled).reshape(batch, frames, -1)
            normalized = torch.cat((rgb, motion), dim=-1)
        value = self.projection(normalized)
        value = value.transpose(1, 2)
        local = self.local(value).transpose(1, 2)
        context = self.context(value).transpose(1, 2)
        return self.fusion(torch.cat((local, context), dim=-1))


class TemporalHighlightEnsemble(nn.Module if nn is not None else object):
    """Combine separately fitted neural models without another detector."""
    def __init__(self, members: list[TemporalHighlightNet], weights: list[float],
                 aggregation: str = 'mean'):
        require_torch()
        super().__init__()
        values = np.asarray(weights, dtype=np.float32)
        if (not members or values.shape != (len(members),) or not np.isfinite(values).all()
                or (values < 0).any() or values.sum() <= 0
                or len({member.config.input_dim for member in members}) != 1):
            raise ValueError('Neural ensemble members and weights are invalid.')
        if aggregation not in ('mean', 'max_event'):
            raise ValueError('Unknown neural ensemble aggregation.')
        self.aggregation = aggregation
        self.members = nn.ModuleList(members)
        self.config = max((member.config for member in members), key=lambda item: item.receptive_field)
        self.register_buffer('ensemble_weights', torch.from_numpy(values/values.sum()))

    @property
    def feature_std(self):
        return torch.stack([member.feature_std for member in self.members])

    def forward(self, features):
        outputs = [torch.sigmoid(member(features)) for member in self.members]
        probability = sum(weight*output for weight, output in zip(self.ensemble_weights, outputs))
        if self.aggregation == 'max_event':
            # An enabled neural expert can recover a contact missed by another.
            # Other heads retain their probability-weighted ensemble behavior.
            event = torch.stack([output[..., :1] for weight, output in zip(self.ensemble_weights, outputs)
                                 if weight > 0]).amax(dim=0)
            probability = torch.cat((event, probability[..., 1:]), dim=-1)
        return torch.logit(probability.clamp(1e-6, 1-1e-6))


def contiguous_feature_spans(timestamps: np.ndarray) -> list[tuple[int, int]]:
    """Separate extracted source windows; a missing minute is not one frame."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Feature timestamps must be finite and strictly increasing.")
    if not len(times):
        return []
    if len(times) == 1:
        return [(0, 1)]
    cadence = float(np.median(np.diff(times)))
    breaks = np.flatnonzero(np.diff(times) > cadence * 1.75 + 1e-6) + 1
    boundaries = [0, *breaks.tolist(), len(times)]
    return list(zip(boundaries[:-1], boundaries[1:]))


def predict_probabilities(
    model: TemporalHighlightNet,
    features: np.ndarray,
    *,
    chunk_frames: int = 512,
    device: str | None = None,
    temperatures: list[float] | np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Return [samples, five heads] probabilities with complete context at seams.

    Halo windows include the network's entire receptive field. Normalization is
    fitted on training videos and stored in the checkpoint; inference never
    derives statistics from the test video.
    """
    require_torch()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != model.config.input_dim:
        raise ValueError("Feature shape does not match the temporal checkpoint.")
    if not np.isfinite(features).all():
        raise ValueError("Temporal features contain non-finite values.")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive.")
    if len(features) == 0:
        return np.empty((0, len(HEAD_NAMES)), dtype=np.float32)
    spans = [(0, len(features))]
    if timestamps is not None:
        if len(timestamps) != len(features):
            raise ValueError("Feature timestamps and embeddings have different lengths.")
        spans = contiguous_feature_spans(timestamps)
    chosen_device = device or str(next(model.parameters()).device)
    model = model.to(chosen_device)
    model.eval()
    scales = np.ones(len(HEAD_NAMES), dtype=np.float32)
    if temperatures is not None:
        scales = np.asarray(temperatures, dtype=np.float32)
        if scales.shape != (len(HEAD_NAMES),) or not np.isfinite(scales).all():
            raise ValueError("Expected one finite calibration temperature for each head.")
        if np.any(scales <= 0):
            raise ValueError("Calibration temperatures must be positive.")
    temperature_tensor = torch.as_tensor(scales, device=chosen_device)
    output = np.empty((len(features), len(HEAD_NAMES)), dtype=np.float32)
    halo = (model.config.receptive_field - 1) // 2
    with torch.inference_mode():
        for span_start, span_stop in spans:
            for start in range(span_start, span_stop, chunk_frames):
                stop = min(span_stop, start + chunk_frames)
                left, right = max(span_start, start - halo), min(span_stop, stop + halo)
                sample = torch.from_numpy(features[left:right]).unsqueeze(0).to(chosen_device)
                logits = model(sample)[0, start - left : stop - left]
                output[start:stop] = torch.sigmoid(logits / temperature_tensor).cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("DL Algo checkpoint produced non-finite temporal predictions.")
    return output


def load_temporal_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_feature_spec: dict[str, Any] | None = None,
) -> tuple[TemporalHighlightNet, dict[str, Any]]:
    """Load a tensor-only checkpoint and reject incompatible feature semantics."""
    require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported or incomplete DL Algo temporal checkpoint.")
    if tuple(payload.get("head_names", ())) != HEAD_NAMES:
        raise ValueError("Temporal checkpoint has incompatible output heads.")
    if expected_feature_spec is not None and payload.get("feature_spec") != expected_feature_spec:
        raise ValueError("Feature extraction specification differs from the trained checkpoint.")
    if payload.get('model_type') == 'probability_ensemble':
        members = [TemporalHighlightNet(TemporalModelConfig.from_dict(config))
                   for config in payload['ensemble_configs']]
        model = TemporalHighlightEnsemble(members, payload['ensemble_weights'],
                                          payload.get('ensemble_aggregation', 'mean'))
    else:
        model = TemporalHighlightNet(TemporalModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Temporal checkpoint contains non-finite model parameters.")
    if not bool((model.feature_std > 0).all()):
        raise ValueError("Temporal checkpoint contains invalid feature normalization.")
    if isinstance(model, TemporalHighlightEnsemble) and (
            bool((model.ensemble_weights < 0).any()) or
            not torch.isclose(model.ensemble_weights.sum(), torch.tensor(1.)).item()):
        raise ValueError('Temporal checkpoint contains invalid ensemble weights.')
    model.to(device).eval()
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    return model, metadata


def checkpoint_payload(
    model: TemporalHighlightNet,
    *,
    feature_spec: dict[str, Any],
    dataset_fingerprint: str,
    training: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    require_torch()
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "head_names": list(HEAD_NAMES),
        "model_config": asdict(model.config),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_spec": feature_spec,
        "dataset_fingerprint": dataset_fingerprint,
        "training": training,
        "calibration": calibration,
    }
    if isinstance(model, TemporalHighlightEnsemble):
        payload.update(model_type='probability_ensemble',
                       ensemble_configs=[asdict(member.config) for member in model.members],
                       ensemble_aggregation=model.aggregation,
                       ensemble_weights=model.ensemble_weights.detach().cpu().tolist())
    return payload
