"""Frozen model6149 LOW-recovery expert used only by the PPO algorithm.

The expert is deliberately a plain deterministic RSL actor reconstruction.
It is never attached to the deployable FastBase actor, optimizer, checkpoint,
TorchScript or ONNX graph.  Its only job is producing one cached rollout
target from a counterfactual low-speed command.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .frozen_speedboost_teacher import INPUT_DIM, OUTPUT_DIM


TERM_MAJOR_COMMAND_SLICE = slice(30, 45)
TERM_MAJOR_COMMAND_HISTORY = 5
LOW_EXPERT_COMMAND = (0.16, 0.0, 0.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_term_major_velocity_command(
    observation: torch.Tensor,
    command: Sequence[float] = LOW_EXPERT_COMMAND,
) -> torch.Tensor:
    """Return a copy with all five command frames rewritten at ``[30:45)``."""

    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(
            f"expert observation must be [batch,{INPUT_DIM}], got "
            f"{tuple(observation.shape)}"
        )
    if len(command) != 3 or any(not math.isfinite(float(value)) for value in command):
        raise ValueError("expert command must contain three finite values")
    rewritten = observation.clone()
    command_tensor = torch.as_tensor(
        tuple(float(value) for value in command),
        device=rewritten.device,
        dtype=rewritten.dtype,
    )
    rewritten[:, TERM_MAJOR_COMMAND_SLICE] = (
        command_tensor.view(1, 1, 3)
        .expand(observation.shape[0], TERM_MAJOR_COMMAND_HISTORY, 3)
        .reshape(observation.shape[0], -1)
    )
    return rewritten


class FrozenLowRecoveryExpert(nn.Module):
    """Strict frozen reconstruction of the 1864→512→256→128→29 RSL actor."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, OUTPUT_DIM),
        )
        self.freeze()

    def freeze(self) -> "FrozenLowRecoveryExpert":
        super().train(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> "FrozenLowRecoveryExpert":
        del mode
        return self.freeze()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(
                f"expert observation must be [batch,{INPUT_DIM}], got "
                f"{tuple(observation.shape)}"
            )
        return self.mlp(observation)


def load_frozen_low_recovery_expert(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
) -> FrozenLowRecoveryExpert:
    """Strictly load only model6149's deterministic actor mean tensors."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LOW recovery expert checkpoint not found: {path}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None:
        normalized = expected_sha256.strip().lower()
        if len(normalized) != 64:
            raise ValueError("LOW recovery expert expected_sha256 must have 64 hex digits")
        int(normalized, 16)
        if actual_sha256 != normalized:
            raise ValueError(
                "LOW recovery expert SHA256 mismatch: "
                f"expected {normalized}, got {actual_sha256}"
            )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("LOW recovery expert checkpoint must contain a dictionary")
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise KeyError("LOW recovery expert checkpoint has no actor_state_dict")
    allowed_non_mean = {"distribution.std_param"}
    unexpected = sorted(
        name
        for name in actor_state
        if not name.startswith("mlp.") and name not in allowed_non_mean
    )
    if unexpected:
        raise ValueError(
            "LOW recovery expert contains unsupported actor tensors: "
            f"{unexpected}"
        )
    mean_state = {
        name.removeprefix("mlp."): value
        for name, value in actor_state.items()
        if name.startswith("mlp.")
    }
    expert = FrozenLowRecoveryExpert()
    expert.mlp.load_state_dict(mean_state, strict=True)
    return expert.to(device=device).freeze()


__all__ = [
    "FrozenLowRecoveryExpert",
    "LOW_EXPERT_COMMAND",
    "TERM_MAJOR_COMMAND_HISTORY",
    "TERM_MAJOR_COMMAND_SLICE",
    "load_frozen_low_recovery_expert",
    "rewrite_term_major_velocity_command",
]
