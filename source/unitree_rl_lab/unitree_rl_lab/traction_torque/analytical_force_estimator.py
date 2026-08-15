"""Regularized analytical per-leg force estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class AnalyticalForceEstimatorCfg:
    temporal_regularization: float = 0.04
    norm_regularization: float = 2.0e-4
    singular_value_floor: float = 1.0e-5
    maximum_force_n: float = 900.0
    maximum_force_rate_n_s: float = 8000.0
    inactive_decay_time_constant_s: float = 0.06
    contact_force_constraint_probability: float = 0.50
    residual_confidence_scale_nm: float = 12.0
    condition_confidence_floor: float = 0.02
    dt: float = 0.02

    def __post_init__(self) -> None:
        if self.temporal_regularization < 0.0 or self.norm_regularization <= 0.0:
            raise ValueError("force regularization is invalid")
        if self.maximum_force_n <= 0.0 or self.maximum_force_rate_n_s <= 0.0:
            raise ValueError("force bounds must be positive")
        if self.dt <= 0.0 or self.inactive_decay_time_constant_s <= 0.0:
            raise ValueError("force estimator timing is invalid")


class AnalyticalForceEstimatorInput(NamedTuple):
    leg_jacobian_linear_local: torch.Tensor
    contact_generalized_leg_force: torch.Tensor
    contact_probability: torch.Tensor
    joint_weight: torch.Tensor | None = None


class AnalyticalForceEstimatorOutput(NamedTuple):
    force_local_n: torch.Tensor
    confidence: torch.Tensor
    residual_norm_nm: torch.Tensor
    condition_score: torch.Tensor
    unconstrained_force_local_n: torch.Tensor


class AnalyticalDualFootForceEstimator:
    """Solve two batched weighted regularized least-squares systems.

    The input Jacobian is linear foot velocity in the desired local force
    frame and has shape ``[N,2,3,6]``. Its transpose therefore maps a local
    three-axis contact force into six leg-joint generalized forces.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: AnalyticalForceEstimatorCfg = AnalyticalForceEstimatorCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.previous_force = torch.zeros((num_envs, 2, 3), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.previous_force.zero_()
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
            self.previous_force[ids] = 0.0

    def update(
        self,
        value: AnalyticalForceEstimatorInput,
    ) -> AnalyticalForceEstimatorOutput:
        jacobian = torch.nan_to_num(value.leg_jacobian_linear_local.to(self.device))
        generalized_force = torch.nan_to_num(
            value.contact_generalized_leg_force.to(self.device)
        )
        contact = torch.nan_to_num(value.contact_probability.to(self.device)).clamp(0, 1)
        if jacobian.shape != (self.num_envs, 2, 3, 6):
            raise ValueError("leg Jacobian must have shape [N,2,3,6]")
        if generalized_force.shape != (self.num_envs, 2, 6):
            raise ValueError("leg generalized force must have shape [N,2,6]")
        if contact.shape != (self.num_envs, 2):
            raise ValueError("contact probability must have shape [N,2]")
        weight = value.joint_weight
        if weight is None:
            weight_tensor = torch.ones_like(generalized_force)
        else:
            weight_tensor = torch.nan_to_num(weight.to(self.device)).clamp_min(0.0)
            if weight_tensor.shape not in ((6,), (2, 6), (self.num_envs, 2, 6)):
                raise ValueError("joint_weight must broadcast to [N,2,6]")
            weight_tensor = torch.broadcast_to(weight_tensor, generalized_force.shape)

        # A maps local force to leg-joint generalized force.
        a = jacobian.transpose(-1, -2)
        aw = a * weight_tensor[..., :, None]
        rw = generalized_force * weight_tensor
        normal = torch.matmul(aw.transpose(-1, -2), aw)
        identity = torch.eye(3, device=self.device).view(1, 1, 3, 3)
        normal = normal + (
            self.cfg.temporal_regularization + self.cfg.norm_regularization
        ) * identity
        rhs = torch.matmul(aw.transpose(-1, -2), rw.unsqueeze(-1)).squeeze(-1)
        rhs = rhs + self.cfg.temporal_regularization * self.previous_force
        unconstrained = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
        unconstrained = torch.nan_to_num(unconstrained)

        force = unconstrained.clone()
        active = contact >= self.cfg.contact_force_constraint_probability
        force[..., 2] = torch.where(
            active,
            force[..., 2].clamp_min(0.0),
            force[..., 2],
        )
        decay = self.cfg.dt / self.cfg.inactive_decay_time_constant_s
        inactive_force = self.previous_force * max(0.0, 1.0 - decay)
        force = torch.where(active[..., None], force, inactive_force)
        magnitude = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
        force = force * (
            self.cfg.maximum_force_n / magnitude.clamp_min(self.cfg.maximum_force_n)
        )
        maximum_delta = self.cfg.maximum_force_rate_n_s * self.cfg.dt
        force = self.previous_force + (force - self.previous_force).clamp(
            -maximum_delta, maximum_delta
        )

        reconstructed = torch.matmul(a, force.unsqueeze(-1)).squeeze(-1)
        residual = torch.linalg.vector_norm(
            reconstructed - generalized_force, dim=-1
        ) / (6.0**0.5)
        singular_values = torch.linalg.svdvals(a)
        condition_score = (
            singular_values[..., -1]
            / singular_values[..., 0].clamp_min(self.cfg.singular_value_floor)
        ).clamp(0.0, 1.0)
        condition_confidence = (
            condition_score / self.cfg.condition_confidence_floor
        ).clamp(0.0, 1.0)
        residual_confidence = torch.exp(
            -residual / self.cfg.residual_confidence_scale_nm
        )
        confidence = (
            contact * condition_confidence * residual_confidence
        ).clamp(0.0, 1.0)
        self.previous_force.copy_(force)
        return AnalyticalForceEstimatorOutput(
            force.reshape(self.num_envs, 6),
            confidence,
            residual,
            condition_score,
            unconstrained.reshape(self.num_envs, 6),
        )
