"""History, randomization, and traction-state regression tests."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.history import TorqueTractionHistory
from unitree_rl_lab.traction_torque.randomization import (
    TorqueDynamicsObservationModel,
    TorqueDynamicsRandomizationCfg,
)
from unitree_rl_lab.traction_torque.traction_estimator import (
    SlipEventState,
    TractionEstimatorInput,
    TractionStateEstimator,
    TractionStateEstimatorCfg,
)


def test_history_is_time_major_and_reset_isolated() -> None:
    history = TorqueTractionHistory(2)
    for value in range(17):
        history.append(torch.full((2, 125), float(value)))
    sequence = history.sequence()
    assert sequence.shape == (2, 15, 125)
    assert sequence[0, 0, 0].item() == 2.0
    assert sequence[0, -1, 0].item() == 16.0
    history.reset(torch.tensor([1]))
    assert history.sequence()[1].count_nonzero().item() == 0
    assert history.sequence()[0].count_nonzero().item() > 0


def test_stage_zero_randomization_is_identity_and_finite() -> None:
    model = TorqueDynamicsObservationModel(2, 12, 18)
    q = torch.randn(2, 12)
    kwargs = dict(
        joint_position=q,
        joint_velocity=q + 1,
        joint_acceleration=q + 2,
        tau_est=q + 3,
        imu_linear_acceleration=torch.randn(2, 3),
        mass_matrix=torch.eye(18).repeat(2, 1, 1),
        bias_force=torch.randn(2, 18),
        leg_jacobian=torch.randn(2, 2, 3, 6),
    )
    output = model.update(**kwargs)
    torch.testing.assert_close(output.joint_position, kwargs["joint_position"])
    torch.testing.assert_close(output.tau_est, kwargs["tau_est"])
    torch.testing.assert_close(output.mass_matrix, kwargs["mass_matrix"])
    assert output.valid.all()


def test_full_randomization_is_seed_reproducible() -> None:
    cfg = TorqueDynamicsRandomizationCfg(curriculum_stage=5)
    first = TorqueDynamicsObservationModel(2, 12, 18, cfg=cfg, seed=7)
    second = TorqueDynamicsObservationModel(2, 12, 18, cfg=cfg, seed=7)
    kwargs = dict(
        joint_position=torch.zeros(2, 12), joint_velocity=torch.zeros(2, 12),
        joint_acceleration=torch.zeros(2, 12), tau_est=torch.ones(2, 12),
        imu_linear_acceleration=torch.zeros(2, 3), mass_matrix=torch.eye(18).repeat(2, 1, 1),
        bias_force=torch.ones(2, 18), leg_jacobian=torch.ones(2, 2, 3, 6),
    )
    out1, out2 = first.update(**kwargs), second.update(**kwargs)
    torch.testing.assert_close(out1.tau_est, out2.tau_est)
    torch.testing.assert_close(out1.leg_jacobian, out2.leg_jacobian)


def test_slip_state_requires_multiple_frames_and_event_mu_is_only_event_value() -> None:
    cfg = TractionStateEstimatorCfg(
        lowpass_tau_s=0.0, candidate_threshold=0.25, confirm_duration_s=0.06
    )
    estimator = TractionStateEstimator(1, cfg=cfg)
    value = TractionEstimatorInput(
        force_local_n=torch.tensor([[150.0, 0.0, 200.0, 150.0, 0.0, 200.0]]),
        contact_probability=torch.ones(1, 2),
        foot_planar_velocity_m_s=torch.full((1, 2, 2), 0.3),
        foot_planar_acceleration_m_s2=torch.full((1, 2, 2), 3.0),
        force_growth_n_s=torch.full((1, 2), 1200.0),
        torque_residual_norm_nm=torch.full((1, 2), 20.0),
        imu_linear_acceleration_m_s2=torch.zeros(1, 3),
        estimator_confidence=torch.ones(1, 2),
    )
    first = estimator.update(value)
    assert (first.state == int(SlipEventState.SLIP_CANDIDATE)).all()
    assert torch.isnan(first.slip_event_mu_estimate).all()
    estimator.update(value)
    third = estimator.update(value)
    assert (third.state == int(SlipEventState.CONFIRMED_SLIP)).all()
    assert torch.isfinite(third.slip_event_mu_estimate).all()
    assert torch.all(third.traction_utilization > 0.0)
