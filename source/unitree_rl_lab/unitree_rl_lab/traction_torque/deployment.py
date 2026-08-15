"""Pure-PyTorch deployment runtime; no simulator ground truth is accepted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from .governor import TorqueGovernorOutput, TorqueTractionCommandGovernor
from .networks import TorqueTractionStudentOutput, TorqueTractionStudentPolicy
from .schema import TORQUE_TRACTION_FRAME_SCHEMA


@dataclass(frozen=True)
class TorqueTractionSafetyCfg:
    action_limit: float = 4.0
    action_scale_rad: float = 0.25
    slip_probability_threshold: float = 0.60
    nonfinite_fallback_action: float = 0.0


class TorqueTractionRuntimeOutput(NamedTuple):
    policy: TorqueTractionStudentOutput
    governor: TorqueGovernorOutput
    action: torch.Tensor
    joint_position_target: torch.Tensor
    safety_flags: torch.Tensor


class TorqueTractionPolicyRuntime:
    """Two-pass fixed-policy governor runtime for one or more robots."""

    def __init__(self, policy: TorqueTractionStudentPolicy, default_joint_position: torch.Tensor, *, governor_enabled: bool = True, safety_cfg: TorqueTractionSafetyCfg = TorqueTractionSafetyCfg()) -> None:
        self.policy = policy.eval()
        self.safety_cfg = safety_cfg
        if default_joint_position.shape[-1] != 29:
            raise ValueError("default_joint_position must contain 29 joints")
        self.registered_default = default_joint_position.detach().clone()
        self.governor_enabled = governor_enabled
        self.governor: TorqueTractionCommandGovernor | None = None
        self.slip_duration: torch.Tensor | None = None

    def reset(self) -> None:
        if self.governor is not None:
            self.governor.reset()
        if self.slip_duration is not None:
            self.slip_duration.zero_()

    @torch.inference_mode()
    def step(self, history: torch.Tensor, current_velocity: torch.Tensor) -> TorqueTractionRuntimeOutput:
        if history.ndim != 3 or history.shape[1:] != (15, 125):
            raise ValueError("runtime history must be [batch,15,125]")
        batch = history.shape[0]
        if current_velocity.shape != (batch, 3):
            raise ValueError("current_velocity must be [batch,3]")
        if self.governor is None or self.governor.num_envs != batch:
            self.governor = TorqueTractionCommandGovernor(batch, device=history.device, enabled=self.governor_enabled)
            self.slip_duration = torch.zeros(batch, 2, device=history.device)
        first = self.policy(history)
        contact = first.contact_probability
        slipping = (first.slip_probability >= self.safety_cfg.slip_probability_threshold) & (contact >= 0.5)
        self.slip_duration = torch.where(slipping, self.slip_duration + 0.02, torch.zeros_like(self.slip_duration))
        current = history[:, -1]
        raw_command = current[:, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")]
        foot_velocity = current[:, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("foot_planar_velocity")].reshape(batch, 2, 2)
        governor = self.governor.update(
            raw_command=raw_command,
            slip_probability=first.slip_probability,
            traction_utilization=first.traction_utilization,
            traction_margin=first.traction_margin,
            contact_probability=contact,
            estimator_confidence=first.estimator_confidence,
            foot_relative_velocity=foot_velocity,
            slip_duration=self.slip_duration,
            current_velocity=current_velocity,
        )
        # The preserved locomotion actor consumes the command embedded in its
        # five-frame proprioceptive history.  Replacing only the newest command
        # makes the governor effective without rewriting past observations.
        governed_history = history.clone()
        governed_history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")] = governor.adjusted_command
        second = self.policy(governed_history)
        finite = torch.isfinite(second.action).all(dim=1, keepdim=True)
        action = torch.nan_to_num(second.action).clamp(-self.safety_cfg.action_limit, self.safety_cfg.action_limit)
        fallback = torch.full_like(action, self.safety_cfg.nonfinite_fallback_action)
        action = torch.where(finite, action, fallback)
        default = self.registered_default.to(history.device).reshape(1, 29)
        target = default + self.safety_cfg.action_scale_rad * action
        flags = governor.safety_flags | ((~finite[:, 0]).long() << 2)
        return TorqueTractionRuntimeOutput(second, governor, action, target, flags)
