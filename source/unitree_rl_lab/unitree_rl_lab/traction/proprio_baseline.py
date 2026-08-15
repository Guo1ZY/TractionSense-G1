"""Exact 480-D Unitree proprioceptive baseline behind the Hall interface.

The original ``model_49999.pt`` actor was trained without empirical
observation normalization and consumes the first 480 elements of the current
1864-D Hall policy group.  This module deliberately loads only its deterministic
actor mean.  It does not create an RSL runner with a mismatched input layer and
it has no computational path from Hall channels to actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .networks import LegacyLocomotionActor
from .schema import ACTION_DIM


HALL_POLICY_DIM = 1864
LEGACY_PROPRIO_DIM = 480
HIGH_SPEED_POLICY_DIM = 482


class ProprioBaseline1864(nn.Module):
    """Adapt the audited legacy actor to an 1864-D observation mapping."""

    consumed_observation_dimension = LEGACY_PROPRIO_DIM
    environment_observation_dimension = HALL_POLICY_DIM

    def __init__(self, actor: LegacyLocomotionActor) -> None:
        super().__init__()
        self.actor = actor

    @staticmethod
    def policy_tensor(
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(observation, Mapping):
            try:
                tensor = observation["policy"]
            except KeyError as exc:
                raise RuntimeError(
                    "proprio baseline requires the environment 'policy' group"
                ) from exc
        elif isinstance(observation, torch.Tensor):
            tensor = observation
        else:
            raise TypeError(
                "observation must be a tensor or a mapping containing 'policy'"
            )
        if tensor.ndim != 2 or tensor.shape[1] != HALL_POLICY_DIM:
            raise RuntimeError(
                "proprio baseline requires the same 1864-D Hall environment "
                f"observation, got {tuple(tensor.shape)}"
            )
        prefix = tensor[:, :LEGACY_PROPRIO_DIM]
        if not torch.isfinite(prefix).all():
            raise FloatingPointError(
                "non-finite value in the consumed 480-D proprioceptive prefix"
            )
        return prefix

    def forward(
        self, observation: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.actor(self.policy_tensor(observation))


class HallBackbone1864(nn.Module):
    """Strict deterministic mean of a plain 1864-D RSL MLP actor."""

    consumed_observation_dimension = HALL_POLICY_DIM
    environment_observation_dimension = HALL_POLICY_DIM

    def __init__(self, actor: LegacyLocomotionActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(
        self, observation: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(observation, Mapping):
            try:
                tensor = observation["policy"]
            except KeyError as exc:
                raise RuntimeError(
                    "Hall backbone requires the environment 'policy' group"
                ) from exc
        elif isinstance(observation, torch.Tensor):
            tensor = observation
        else:
            raise TypeError(
                "observation must be a tensor or a mapping containing 'policy'"
            )
        if tensor.ndim != 2 or tensor.shape[1] != HALL_POLICY_DIM:
            raise RuntimeError(
                f"Hall backbone requires [N,{HALL_POLICY_DIM}], got "
                f"{tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Hall backbone input contains NaN/Inf")
        return self.actor(tensor)


class HighSpeedBackbone482(nn.Module):
    """Strict high-speed branch using proprio history plus lateral feedback.

    The environment continues to publish the complete 1864-D Hall observation
    for the traction-adaptation branch.  This isolated high-grip controller
    consumes the separately audited ``high_speed_policy`` group:
    ``policy[0:480] + [body_vy, relative_heading]``.  It never consumes force,
    contact, friction, slip or privileged course state.
    """

    consumed_observation_dimension = HIGH_SPEED_POLICY_DIM
    environment_observation_dimension = HALL_POLICY_DIM

    def __init__(self, actor: LegacyLocomotionActor) -> None:
        super().__init__()
        self.actor = actor

    @staticmethod
    def policy_tensor(
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(observation, Mapping):
            try:
                tensor = observation["high_speed_policy"]
            except KeyError as exc:
                raise RuntimeError(
                    "482-D backbone requires the 'high_speed_policy' group"
                ) from exc
        elif isinstance(observation, torch.Tensor):
            tensor = observation
        else:
            raise TypeError(
                "observation must be a tensor or contain 'high_speed_policy'"
            )
        if tensor.ndim != 2 or tensor.shape[1] != HIGH_SPEED_POLICY_DIM:
            raise RuntimeError(
                f"high-speed backbone requires [N,{HIGH_SPEED_POLICY_DIM}], "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("high-speed backbone input contains NaN/Inf")
        return tensor

    def forward(
        self, observation: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.actor(self.policy_tensor(observation))


def _strict_actor_mean_state(
    checkpoint_path: str | Path,
    *,
    input_dim: int,
    actor_label: str = "actor mean",
) -> tuple[Path, dict[str, torch.Tensor]]:
    """Read and validate one plain 512-256-128 ELU RSL actor mean."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint root must be a mapping")
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path}: missing actor_state_dict")

    actor = LegacyLocomotionActor(input_dim)
    expected = set(actor.state_dict())
    mlp_state = {
        key: value for key, value in state.items() if key.startswith("mlp.")
    }
    missing = sorted(expected - set(mlp_state))
    unexpected = sorted(set(mlp_state) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path}: incompatible {actor_label} keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
    extra_actor_keys = sorted(
        set(state) - expected - {"distribution.std_param"}
    )
    if extra_actor_keys:
        raise ValueError(f"{path}: unexpected actor keys {extra_actor_keys}")
    std = state.get("distribution.std_param")
    if std is not None and (
        not isinstance(std, torch.Tensor) or tuple(std.shape) != (ACTION_DIM,)
    ):
        shape = tuple(std.shape) if isinstance(std, torch.Tensor) else type(std)
        raise ValueError(f"{path}: invalid action distribution shape {shape}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"{path}: actor parameter {key!r} is not finite")
    return path, mlp_state


def load_proprio_baseline(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ProprioBaseline1864:
    """Strictly load the deterministic actor mean from an RSL checkpoint.

    ``weights_only=True`` avoids executing arbitrary pickle payloads.  The
    actor layout, output size and optional stochastic distribution parameter
    are checked before the policy is returned.
    """

    _, mlp_state = _strict_actor_mean_state(
        checkpoint_path,
        input_dim=LEGACY_PROPRIO_DIM,
        actor_label="legacy actor",
    )
    actor = LegacyLocomotionActor(LEGACY_PROPRIO_DIM)
    actor.load_state_dict(mlp_state, strict=True)
    policy = ProprioBaseline1864(actor.eval()).to(device=device)
    return policy.eval()


def load_hall_backbone(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> HallBackbone1864:
    """Strictly load a standard 1864-D RSL deterministic actor mean."""

    _, mlp_state = _strict_actor_mean_state(
        checkpoint_path,
        input_dim=HALL_POLICY_DIM,
        actor_label="Hall actor",
    )
    actor = LegacyLocomotionActor(HALL_POLICY_DIM)
    actor.load_state_dict(mlp_state, strict=True)
    return HallBackbone1864(actor.eval()).to(device=device).eval()


def load_high_speed_backbone(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> HighSpeedBackbone482:
    """Strictly load a standard 482-D RSL deterministic actor mean."""

    _, mlp_state = _strict_actor_mean_state(
        checkpoint_path,
        input_dim=HIGH_SPEED_POLICY_DIM,
        actor_label="high-speed 482-D actor",
    )
    actor = LegacyLocomotionActor(HIGH_SPEED_POLICY_DIM)
    actor.load_state_dict(mlp_state, strict=True)
    return HighSpeedBackbone482(actor.eval()).to(device=device).eval()


__all__ = [
    "HALL_POLICY_DIM",
    "HIGH_SPEED_POLICY_DIM",
    "LEGACY_PROPRIO_DIM",
    "HallBackbone1864",
    "HighSpeedBackbone482",
    "ProprioBaseline1864",
    "load_hall_backbone",
    "load_high_speed_backbone",
    "load_proprio_baseline",
]
