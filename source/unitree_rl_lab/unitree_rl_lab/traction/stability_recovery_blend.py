"""Causal, opt-in action handoff to a frozen low-speed recovery actor.

This module is deliberately independent from Isaac Sim.  It consumes only the
deployable 1864-D actor observation, the baseline action and the state emitted
by :mod:`high_speed_stability_envelope`.  Material identity, friction,
contacts, forces and course-stage labels are not inputs.

The frozen Stage7 recovery actor was trained at a 0.16 m/s crawl command.  Its
private observation copy therefore receives that command in the correct
term-major columns 30:45.  The newest last-action history sample is replaced by
the action that this blender actually returned on the preceding control step;
the recovery actor never observes an imaginary all-baseline or all-expert
handoff history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from .high_speed_stability_envelope import (
    ACTION_DIM,
    EMERGENCY,
    HISTORY_FRAMES,
    LAST_ACTION_HISTORY_SLICE,
    POLICY_OBSERVATION_DIM,
)


RECOVERY_COMMAND_VX_INDICES = (30, 33, 36, 39, 42)
RECOVERY_COMMAND_VY_INDICES = (31, 34, 37, 40, 43)
RECOVERY_COMMAND_YAW_INDICES = (32, 35, 38, 41, 44)


@dataclass(frozen=True)
class StabilityRecoveryBlendCfg:
    """Configuration for a smooth EMERGENCY-to-recovery handoff."""

    recovery_forward_command: float = 0.16
    blend_in_time_s: float = 0.20
    blend_out_time_s: float = 0.30
    trigger_state: int = EMERGENCY

    def validate(self) -> None:
        scalars = (
            self.recovery_forward_command,
            self.blend_in_time_s,
            self.blend_out_time_s,
        )
        if not all(torch.isfinite(torch.tensor(float(value))).item() for value in scalars):
            raise ValueError("stability-recovery configuration must be finite")
        if self.recovery_forward_command < 0.0:
            raise ValueError("recovery_forward_command must be non-negative")
        if not 0.15 <= self.blend_in_time_s <= 0.30:
            raise ValueError("blend_in_time_s must be in the reviewed [0.15, 0.30] s range")
        if not 0.15 <= self.blend_out_time_s <= 0.30:
            raise ValueError("blend_out_time_s must be in the reviewed [0.15, 0.30] s range")
        if not 0 <= int(self.trigger_state) <= EMERGENCY:
            raise ValueError("trigger_state is outside the stability-state range")


@dataclass(frozen=True)
class StabilityRecoveryBlendOutput:
    """One batched causal blend update."""

    action: torch.Tensor
    baseline_action: torch.Tensor
    recovery_action: torch.Tensor
    gate: torch.Tensor
    active: torch.Tensor


class FrozenStage7RecoveryActor(nn.Module):
    """Exact deterministic mean network from the legacy Stage7 RSL checkpoint.

    Stage7 used ``MLPModel([512, 256, 128], activation='elu')``.  Its checkpoint
    stores the Gaussian distribution separately; deterministic evaluation is
    exactly the actor MLP mean and does not sample ``distribution.std_param``.
    """

    EXPECTED_DIMS = (POLICY_OBSERVATION_DIM, 512, 256, 128, ACTION_DIM)

    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_path: str | None = None
        dims = self.EXPECTED_DIMS
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.requires_grad_(False)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FrozenStage7RecoveryActor":
        """Strictly load the frozen actor mean from a Stage7 checkpoint."""

        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RuntimeError("recovery checkpoint must contain a mapping")
        state = payload.get("actor_state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("recovery checkpoint is missing actor_state_dict")

        expected = cls()
        expected_state = expected.state_dict()
        source_state: dict[str, torch.Tensor] = {}
        for key in expected_state:
            source_key = key.removeprefix("mlp.")
            checkpoint_key = f"mlp.{source_key}"
            value = state.get(checkpoint_key)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"recovery checkpoint is missing {checkpoint_key}")
            if tuple(value.shape) != tuple(expected_state[key].shape):
                raise RuntimeError(
                    f"recovery checkpoint {checkpoint_key} shape {tuple(value.shape)} "
                    f"!= expected {tuple(expected_state[key].shape)}"
                )
            source_state[key] = value
        expected.load_state_dict(source_state, strict=True)
        expected.to(device=torch.device(device))
        expected.requires_grad_(False)
        expected.eval()
        expected.checkpoint_path = str(path)
        return expected


def rewrite_recovery_observation(
    policy_observation: torch.Tensor,
    *,
    forward_command: float,
    actual_previous_action: torch.Tensor,
) -> torch.Tensor:
    """Build the expert-private causal observation without changing other terms."""

    observation = torch.as_tensor(policy_observation)
    if observation.ndim != 2 or observation.shape[1] != POLICY_OBSERVATION_DIM:
        raise ValueError(
            "recovery observation must have shape [N,1864], got "
            f"{tuple(observation.shape)}"
        )
    previous = torch.as_tensor(
        actual_previous_action, device=observation.device, dtype=observation.dtype
    )
    if previous.shape != (observation.shape[0], ACTION_DIM):
        raise ValueError(
            "actual_previous_action must have shape "
            f"{(observation.shape[0], ACTION_DIM)}, got {tuple(previous.shape)}"
        )
    if not torch.isfinite(observation).all() or not torch.isfinite(previous).all():
        raise FloatingPointError("recovery observation inputs must be finite")
    command = float(forward_command)
    if not torch.isfinite(torch.tensor(command)).item():
        raise ValueError("recovery forward command must be finite")

    rewritten = observation.clone()
    rewritten[:, list(RECOVERY_COMMAND_VX_INDICES)] = command
    rewritten[:, list(RECOVERY_COMMAND_VY_INDICES)] = 0.0
    rewritten[:, list(RECOVERY_COMMAND_YAW_INDICES)] = 0.0
    action_history = rewritten[:, LAST_ACTION_HISTORY_SLICE].reshape(
        observation.shape[0], HISTORY_FRAMES, ACTION_DIM
    )
    action_history[:, -1] = previous
    return rewritten


class StabilityRecoveryBlend:
    """Smoothly hand off from a baseline action to a frozen recovery actor."""

    def __init__(
        self,
        recovery_actor: nn.Module,
        *,
        num_envs: int,
        dt: float,
        device: str | torch.device,
        cfg: StabilityRecoveryBlendCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not torch.isfinite(torch.tensor(float(dt))).item() or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.cfg = cfg or StabilityRecoveryBlendCfg()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.recovery_actor = recovery_actor.to(self.device).eval()
        self.recovery_actor.requires_grad_(False)
        self.gate = torch.zeros(self.num_envs, device=self.device)
        self._last_applied_action = torch.zeros(
            (self.num_envs, ACTION_DIM), device=self.device
        )
        self._has_last_action = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
                raise IndexError("env_ids contains an out-of-range index")
        self.gate[ids] = 0.0
        self._last_applied_action[ids] = 0.0
        self._has_last_action[ids] = False

    def update(
        self,
        policy_observation: torch.Tensor,
        baseline_action: torch.Tensor,
        stability_state: torch.Tensor,
    ) -> StabilityRecoveryBlendOutput:
        observation = torch.as_tensor(policy_observation, device=self.device)
        if observation.shape != (self.num_envs, POLICY_OBSERVATION_DIM):
            raise ValueError(
                "policy_observation must have shape "
                f"{(self.num_envs, POLICY_OBSERVATION_DIM)}, got {tuple(observation.shape)}"
            )
        if not observation.is_floating_point():
            observation = observation.float()
        baseline = torch.as_tensor(
            baseline_action, device=self.device, dtype=observation.dtype
        )
        if baseline.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"baseline_action must have shape {(self.num_envs, ACTION_DIM)}, "
                f"got {tuple(baseline.shape)}"
            )
        state = torch.as_tensor(
            stability_state, device=self.device, dtype=torch.long
        )
        if state.shape != (self.num_envs,):
            raise ValueError(
                f"stability_state must have shape {(self.num_envs,)}, got {tuple(state.shape)}"
            )
        if not torch.isfinite(observation).all() or not torch.isfinite(baseline).all():
            raise FloatingPointError("stability-recovery inputs must be finite")

        target_active = state >= int(self.cfg.trigger_state)
        rise = self.dt / self.cfg.blend_in_time_s
        fall = self.dt / self.cfg.blend_out_time_s
        # Preserve the ordinary tensor allocated in ``__init__``.  The Isaac
        # evaluator invokes policy/recovery inference inside
        # ``torch.inference_mode()``; assigning the result object to
        # ``self.gate`` would turn persistent state into an inference tensor,
        # which then rejects the out-of-context in-place reset performed after
        # a managed environment termination.
        next_gate = torch.where(
            target_active,
            torch.clamp(self.gate + rise, max=1.0),
            torch.clamp(self.gate - fall, min=0.0),
        )
        self.gate.copy_(next_gate)

        observation_last_action = observation[:, LAST_ACTION_HISTORY_SLICE].reshape(
            self.num_envs, HISTORY_FRAMES, ACTION_DIM
        )[:, -1]
        actual_previous_action = torch.where(
            self._has_last_action[:, None],
            self._last_applied_action,
            observation_last_action,
        )
        recovery_observation = rewrite_recovery_observation(
            observation,
            forward_command=self.cfg.recovery_forward_command,
            actual_previous_action=actual_previous_action,
        )
        recovery = baseline.clone()
        needs_recovery = self.gate > 0.0
        if bool(needs_recovery.any().item()):
            with torch.inference_mode():
                recovery_subset = self.recovery_actor(
                    recovery_observation[needs_recovery]
                )
            if recovery_subset.shape != (int(needs_recovery.sum().item()), ACTION_DIM):
                raise RuntimeError(
                    "recovery actor returned wrong shape: "
                    f"{tuple(recovery_subset.shape)}"
                )
            if not torch.isfinite(recovery_subset).all():
                raise FloatingPointError("recovery actor returned non-finite actions")
            recovery[needs_recovery] = recovery_subset.to(
                device=self.device, dtype=baseline.dtype
            )

        action = torch.lerp(baseline, recovery, self.gate[:, None])
        self._last_applied_action.copy_(action.detach())
        self._has_last_action.fill_(True)
        return StabilityRecoveryBlendOutput(
            action=action,
            baseline_action=baseline.clone(),
            recovery_action=recovery,
            gate=self.gate.clone(),
            active=target_active,
        )


__all__ = [
    "FrozenStage7RecoveryActor",
    "RECOVERY_COMMAND_VX_INDICES",
    "RECOVERY_COMMAND_VY_INDICES",
    "RECOVERY_COMMAND_YAW_INDICES",
    "StabilityRecoveryBlend",
    "StabilityRecoveryBlendCfg",
    "StabilityRecoveryBlendOutput",
    "rewrite_recovery_observation",
]
