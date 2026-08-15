"""Engine-independent full-body inverse-dynamics residual calculation."""

from __future__ import annotations

from typing import NamedTuple

import torch


class InverseDynamicsResult(NamedTuple):
    model_generalized_force: torch.Tensor
    model_joint_torque: torch.Tensor
    tau_est_minus_model: torch.Tensor
    contact_generalized_joint_force: torch.Tensor


def inverse_dynamics_torque_residual(
    mass_matrix: torch.Tensor,
    bias_force: torch.Tensor,
    generalized_acceleration: torch.Tensor,
    tau_est: torch.Tensor,
    *,
    root_dofs: int = 6,
) -> InverseDynamicsResult:
    """Compute the no-contact model torque and contact-induced residual.

    The user-facing diagnostic ``tau_est_minus_model`` follows the requested
    definition. For a ground-reaction force acting on the robot, the dynamics
    equation is ``M qdd + h = S^T tau_est + J^T F``; therefore the physical
    contact generalized force is its negative, ``tau_model - tau_est``.
    Keeping both values explicit prevents a hidden sign convention.
    """

    if mass_matrix.ndim != 3 or mass_matrix.shape[-1] != mass_matrix.shape[-2]:
        raise ValueError("mass_matrix must be [batch,nv,nv]")
    batch, nv, _ = mass_matrix.shape
    if bias_force.shape != (batch, nv):
        raise ValueError("bias_force shape mismatch")
    if generalized_acceleration.shape != (batch, nv):
        raise ValueError("generalized_acceleration shape mismatch")
    if tau_est.shape != (batch, nv - root_dofs):
        raise ValueError("tau_est shape mismatch")
    finite_inputs = (
        torch.isfinite(mass_matrix).all(dim=(-2, -1))
        & torch.isfinite(bias_force).all(dim=-1)
        & torch.isfinite(generalized_acceleration).all(dim=-1)
        & torch.isfinite(tau_est).all(dim=-1)
    )
    mass = torch.nan_to_num(mass_matrix)
    bias = torch.nan_to_num(bias_force)
    acceleration = torch.nan_to_num(generalized_acceleration)
    model_force = torch.bmm(mass, acceleration.unsqueeze(-1)).squeeze(-1) + bias
    model_joint = model_force[:, root_dofs:]
    requested_residual = torch.nan_to_num(tau_est) - model_joint
    contact_force = -requested_residual
    mask = finite_inputs[:, None]
    return InverseDynamicsResult(
        torch.where(mask, model_force, torch.zeros_like(model_force)),
        torch.where(mask, model_joint, torch.zeros_like(model_joint)),
        torch.where(mask, requested_residual, torch.zeros_like(requested_residual)),
        torch.where(mask, contact_force, torch.zeros_like(contact_force)),
    )
