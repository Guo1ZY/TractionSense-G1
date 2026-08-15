"""Numerical tests for full-body residual and analytical foot-force solve."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.analytical_force_estimator import (
    AnalyticalDualFootForceEstimator,
    AnalyticalForceEstimatorCfg,
    AnalyticalForceEstimatorInput,
)
from unitree_rl_lab.traction_torque.dynamics import inverse_dynamics_torque_residual


def test_inverse_dynamics_exposes_physical_contact_sign() -> None:
    mass = torch.eye(35).unsqueeze(0)
    bias = torch.zeros(1, 35)
    acceleration = torch.zeros(1, 35)
    tau_est = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    result = inverse_dynamics_torque_residual(mass, bias, acceleration, tau_est)
    torch.testing.assert_close(result.tau_est_minus_model, tau_est)
    torch.testing.assert_close(result.contact_generalized_joint_force, -tau_est)


def test_analytical_estimator_recovers_force_in_well_conditioned_system() -> None:
    cfg = AnalyticalForceEstimatorCfg(
        temporal_regularization=0.0,
        norm_regularization=1.0e-8,
        maximum_force_rate_n_s=1.0e9,
    )
    estimator = AnalyticalDualFootForceEstimator(1, cfg=cfg)
    jacobian = torch.zeros(1, 2, 3, 6)
    jacobian[:, :, :, :3] = torch.eye(3)
    expected = torch.tensor([[[10.0, -20.0, 300.0], [-7.0, 4.0, 120.0]]])
    generalized = torch.einsum("blij,bli->blj", jacobian, expected)
    output = estimator.update(
        AnalyticalForceEstimatorInput(jacobian, generalized, torch.ones(1, 2))
    )
    torch.testing.assert_close(output.force_local_n.view(1, 2, 3), expected, atol=1e-4, rtol=1e-5)
    assert torch.isfinite(output.force_local_n).all()


def test_singular_jacobian_lowers_confidence_without_nonfinite_values() -> None:
    estimator = AnalyticalDualFootForceEstimator(2)
    jacobian = torch.zeros(2, 2, 3, 6)
    generalized = torch.randn(2, 2, 6)
    output = estimator.update(
        AnalyticalForceEstimatorInput(jacobian, generalized, torch.ones(2, 2))
    )
    assert output.condition_score.max().item() == 0.0
    assert output.confidence.max().item() == 0.0
    assert torch.isfinite(output.force_local_n).all()


def test_inactive_foot_decays_and_reset_clears_state() -> None:
    estimator = AnalyticalDualFootForceEstimator(1)
    jacobian = torch.zeros(1, 2, 3, 6)
    jacobian[:, :, :, :3] = torch.eye(3)
    generalized = torch.zeros(1, 2, 6)
    generalized[:, :, 2] = 200.0
    estimator.update(AnalyticalForceEstimatorInput(jacobian, generalized, torch.ones(1, 2)))
    decayed = estimator.update(
        AnalyticalForceEstimatorInput(jacobian, generalized, torch.zeros(1, 2))
    ).force_local_n
    assert 0.0 < decayed[0, 2].item() < 200.0
    estimator.reset()
    assert estimator.previous_force.count_nonzero().item() == 0

