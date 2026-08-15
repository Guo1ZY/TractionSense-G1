"""Causal forward-speed estimator for magnetic-foot locomotion.

The estimator consumes the *same* 1864-D deployable observation used by the
Hall policy: IMU/gravity, command, joint/action history, two-foot Bx/By/Bz
history, packet timing, and L/R health.  It intentionally has no simulator
root velocity, contact force, or friction coefficient input.  Those quantities
are allowed only as offline training labels and evaluation diagnostics.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


class ForwardVelocityEstimator(nn.Module):
    """Small MLP that predicts body-frame forward speed in m/s."""

    def __init__(
        self,
        input_dim: int = 1864,
        hidden_dims: Sequence[int] = (384, 192, 64),
        output_clip: float = 2.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        layers: list[nn.Module] = []
        width = int(input_dim)
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, int(hidden)), nn.ELU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.output_clip = float(output_clip)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError(
                f"expected [batch, features], got {tuple(observation.shape)}"
            )
        speed = self.network(observation).squeeze(-1)
        # A bounded estimate prevents a rare corrupt Hall packet from turning
        # a downstream safety monitor into a command spike.
        return torch.clamp(speed, -self.output_clip, self.output_clip)


class NormalizedForwardVelocityEstimator(nn.Module):
    """Frozen normalization and feature projection around the core network."""

    def __init__(
        self,
        estimator: ForwardVelocityEstimator,
        mean: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        feature_indices: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).flatten()
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).flatten()
        if mean_tensor.shape != scale_tensor.shape:
            raise ValueError("mean and scale must have identical shape")
        if torch.any(scale_tensor <= 0):
            raise ValueError("normalization scale must be positive")
        if feature_indices is None:
            feature_tensor = torch.arange(mean_tensor.numel(), dtype=torch.long)
        else:
            feature_tensor = torch.as_tensor(feature_indices, dtype=torch.long).flatten()
        if feature_tensor.numel() != mean_tensor.numel():
            raise ValueError("feature_indices and normalization dimensions differ")
        if estimator.network[0].in_features != mean_tensor.numel():
            raise ValueError("estimator input_dim and normalization dimensions differ")
        self.estimator = estimator
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("scale", scale_tensor)
        self.register_buffer("feature_indices", feature_tensor)

    @property
    def input_dim(self) -> int:
        return int(self.feature_indices.numel())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError(
                f"expected [batch, features], got {tuple(observation.shape)}"
            )
        feature = observation.index_select(1, self.feature_indices)
        normalized = (feature - self.mean) / self.scale
        return self.estimator(normalized)


def build_forward_velocity_estimator(
    payload: Mapping[str, object],
) -> NormalizedForwardVelocityEstimator:
    """Restore a normalized estimator from a training checkpoint payload."""
    input_dim = int(payload["input_dim"])
    hidden_dims = tuple(int(value) for value in payload["hidden_dims"])
    output_clip = float(payload.get("output_clip", 2.0))
    core = ForwardVelocityEstimator(input_dim, hidden_dims, output_clip)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state dictionary")
    core.load_state_dict(state, strict=True)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    feature_indices = np.asarray(
        payload.get("feature_indices", np.arange(input_dim)), dtype=np.int64
    )
    return NormalizedForwardVelocityEstimator(
        core, mean, scale, feature_indices
    )
