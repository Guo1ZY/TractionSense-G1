"""Contact hysteresis and reset tests using deployment-only signals."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.contact_estimator import (
    HybridContactEstimator,
    HybridContactEstimatorCfg,
    HybridContactInput,
)


def _input(*, stance: bool) -> HybridContactInput:
    height = 0.0 if stance else 0.30
    speed = 0.0 if stance else 2.0
    torque = 30.0 if stance else 0.0
    fz = 200.0 if stance else 0.0
    return HybridContactInput(
        foot_height_m=torch.full((1, 2), height),
        foot_vertical_velocity_m_s=torch.full((1, 2), speed),
        foot_planar_velocity_m_s=torch.full((1, 2, 2), speed),
        leg_torque_nm=torch.full((1, 2, 6), torque),
        estimated_fz_n=torch.full((1, 2), fz),
        joint_configuration=torch.zeros(1, 2, 6),
        imu_linear_acceleration_m_s2=torch.zeros(1, 3),
    )


def test_contact_debounce_hold_hysteresis_and_reset() -> None:
    cfg = HybridContactEstimatorCfg(
        probability_lowpass_tau_s=0.0,
        on_threshold=0.55,
        off_threshold=0.35,
        debounce_s=0.04,
        minimum_hold_s=0.08,
    )
    estimator = HybridContactEstimator(1, cfg=cfg)
    first = estimator.update(_input(stance=True))
    assert not first.state.any()
    second = estimator.update(_input(stance=True))
    assert second.state.all()
    early_swing = estimator.update(_input(stance=False))
    assert early_swing.state.all(), "minimum hold must prevent a one-frame release"
    for _ in range(8):
        released = estimator.update(_input(stance=False))
    assert not released.state.any()
    estimator.reset()
    assert estimator.probability.count_nonzero().item() == 0
    assert not estimator.state.any()

