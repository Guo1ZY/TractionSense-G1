"""Coordinate, support-load, and history-reset regressions."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.analytical_force_estimator import (
    AnalyticalDualFootForceEstimator,
    AnalyticalForceEstimatorCfg,
    AnalyticalForceEstimatorInput,
)
from unitree_rl_lab.traction_torque.history import TorqueTractionHistory


def _identity_leg_jacobian(batch: int = 1) -> torch.Tensor:
    jacobian = torch.zeros(batch, 2, 3, 6)
    jacobian[:, :, :, :3] = torch.eye(3)
    return jacobian


def _estimator(batch: int = 1) -> AnalyticalDualFootForceEstimator:
    return AnalyticalDualFootForceEstimator(
        batch,
        cfg=AnalyticalForceEstimatorCfg(
            temporal_regularization=0.0,
            norm_regularization=1.0e-8,
            maximum_force_rate_n_s=1.0e9,
            maximum_force_n=2000.0,
        ),
    )


def test_left_right_order_and_horizontal_force_sign_are_preserved() -> None:
    expected = torch.tensor([[[40.0, -12.0, 180.0], [-25.0, 7.0, 240.0]]])
    jacobian = _identity_leg_jacobian()
    generalized = torch.einsum("blij,bli->blj", jacobian, expected)
    result = _estimator().update(AnalyticalForceEstimatorInput(jacobian, generalized, torch.ones(1, 2)))
    torch.testing.assert_close(result.force_local_n.reshape(1, 2, 3), expected, atol=1e-4, rtol=1e-5)


def test_static_double_support_and_single_support_load_distribution() -> None:
    jacobian = _identity_leg_jacobian()
    double = torch.tensor([[[0.0, 0.0, 350.0], [0.0, 0.0, 350.0]]])
    double_output = _estimator().update(
        AnalyticalForceEstimatorInput(jacobian, torch.einsum("blij,bli->blj", jacobian, double), torch.ones(1, 2))
    ).force_local_n.reshape(1, 2, 3)
    assert abs(double_output[..., 2].sum().item() - 700.0) < 1e-3
    single = torch.tensor([[[0.0, 0.0, 700.0], [0.0, 0.0, 0.0]]])
    single_output = _estimator().update(
        AnalyticalForceEstimatorInput(jacobian, torch.einsum("blij,bli->blj", jacobian, single), torch.tensor([[1.0, 0.0]]))
    ).force_local_n.reshape(1, 2, 3)
    assert single_output[0, 0, 2] > 699.0
    assert single_output[0, 1].norm() < 1e-5


def test_history_reset_cannot_leak_previous_episode() -> None:
    history = TorqueTractionHistory(2)
    history.append(torch.ones(2, 125))
    history.reset(torch.tensor([1]))
    output = history.sequence()
    assert output[0].count_nonzero() == 125
    assert output[1].count_nonzero() == 0
