from __future__ import annotations

import torch

from unitree_rl_lab.traction.hall_recovery_policy import HallRecoveryPolicy
from unitree_rl_lab.traction.hall_risk_estimator import (
    BaselineInvariantHallTractionRiskEstimator,
    HallTractionRiskEstimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (
    AGE_SLICE,
    INPUT_DIM,
    VALID_SLICE,
    LayoutMagneticStudent,
)


def make_policy() -> HallRecoveryPolicy:
    torch.manual_seed(4)
    base = LayoutMagneticStudent(residual_limit=0.35)
    risk = HallTractionRiskEstimator()
    policy = HallRecoveryPolicy(base, risk, correction_limit=0.25)
    with torch.no_grad():
        policy.recovery_head[-1].weight.fill_(0.1)
        policy.risk_estimator.risk_head.weight.zero_()
        policy.risk_estimator.risk_head.bias.fill_(10.0)
    return policy.eval()


def healthy_observation(batch: int = 3) -> torch.Tensor:
    observation = torch.randn(batch, INPUT_DIM) * 0.1
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = 0.0
    return observation


def test_recovery_shape_bound_and_frozen_upstream() -> None:
    policy = make_policy()
    observation = healthy_observation()
    action, base, risk, correction = policy.recovery_outputs(observation)
    assert action.shape == base.shape == correction.shape == (3, 29)
    assert risk.shape == (3, 1)
    assert torch.max(torch.abs(correction)) <= 0.25 + 1.0e-6
    assert not any(item.requires_grad for item in policy.base_policy.parameters())
    assert not any(item.requires_grad for item in policy.risk_estimator.parameters())


def test_missing_hall_has_exact_action_fallback() -> None:
    policy = make_policy()
    observation = healthy_observation()
    observation[:, VALID_SLICE] = 0.0
    observation[:, AGE_SLICE] = 1.0
    action, base, risk, correction = policy.recovery_outputs(observation)
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=0.0, rtol=0.0)


def test_low_risk_has_exact_action_fallback() -> None:
    policy = make_policy()
    with torch.no_grad():
        policy.risk_estimator.risk_head.bias.fill_(-10.0)
    observation = healthy_observation()
    action, base, risk, correction = policy.recovery_outputs(observation)
    assert torch.all(risk < 0.35)
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)


def test_confidence_gated_terrain_residual_keeps_sensor_loss_fallback() -> None:
    base = LayoutMagneticStudent(residual_limit=0.35)
    risk = HallTractionRiskEstimator()
    policy = HallRecoveryPolicy(
        base,
        risk,
        correction_limit=0.20,
        risk_gate_start=0.0,
        risk_gate_full=1.0e-3,
    ).eval()
    with torch.no_grad():
        policy.recovery_head[-1].weight.fill_(0.1)
        policy.risk_estimator.risk_head.weight.zero_()
        policy.risk_estimator.risk_head.bias.fill_(-3.0)
    healthy = healthy_observation(2)
    _, _, _, healthy_correction = policy.recovery_outputs(healthy)
    assert torch.max(torch.abs(healthy_correction)) > 0.0

    failed = healthy.clone()
    failed[:, VALID_SLICE] = 0.0
    failed[:, AGE_SLICE] = 1.0
    action, base_action, risk_value, correction = policy.recovery_outputs(failed)
    torch.testing.assert_close(action, base_action, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)
    torch.testing.assert_close(risk_value, torch.ones_like(risk_value), atol=0.0, rtol=0.0)


def test_nonfinite_input_remains_finite_and_conservative() -> None:
    policy = make_policy()
    observation = healthy_observation(2)
    observation[:] = float("nan")
    action, base, risk, correction = policy.recovery_outputs(observation)
    assert torch.isfinite(action).all()
    assert torch.isfinite(base).all()
    assert torch.isfinite(correction).all()
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=0.0, rtol=0.0)
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)


def test_recovery_accepts_baseline_invariant_risk_features() -> None:
    base = LayoutMagneticStudent(residual_limit=0.35)
    risk = BaselineInvariantHallTractionRiskEstimator()
    policy = HallRecoveryPolicy(base, risk, correction_limit=0.20).eval()
    observation = healthy_observation(2)
    with torch.inference_mode():
        action, base_action, risk_value, correction = policy.recovery_outputs(observation)
    assert policy.recovery_head[0].in_features == risk.feature_dim
    assert action.shape == base_action.shape == correction.shape == (2, 29)
    assert risk_value.shape == (2, 1)
    assert torch.isfinite(action).all()


def test_recovery_rejects_shape_equal_but_semantically_different_tail() -> None:
    base = LayoutMagneticStudent(
        residual_limit=0.35, trailing_feature_mode="motion_feedback"
    )
    risk = HallTractionRiskEstimator(trailing_feature_mode="sensor_age")
    try:
        HallRecoveryPolicy(base, risk, correction_limit=0.20)
    except ValueError as error:
        assert "channels 1862:1864" in str(error)
    else:
        raise AssertionError("mismatched trailing-feature semantics must fail closed")
