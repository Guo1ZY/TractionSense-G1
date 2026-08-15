from __future__ import annotations

from pathlib import Path
import sys

import torch

from unitree_rl_lab.traction.anchored_hall_recovery import AnchoredHallRecoveryPolicy
from unitree_rl_lab.traction.hall_risk_estimator import (
    BaselineInvariantHallTractionRiskEstimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import AGE_SLICE, INPUT_DIM, VALID_SLICE
from unitree_rl_lab.traction.networks import LegacyLocomotionActor


ROOT = Path(__file__).resolve().parents[2]
TRACTION_SCRIPTS = ROOT / "scripts" / "traction"
if str(TRACTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACTION_SCRIPTS))
from train_anchored_hall_recovery import offline_supervision_gate


def healthy_observation(batch: int = 3) -> torch.Tensor:
    value = torch.randn(batch, INPUT_DIM) * 0.05
    value[:, VALID_SLICE] = 1.0
    value[:, AGE_SLICE] = 0.0
    return value


def make_policy(risk_bias: float) -> AnchoredHallRecoveryPolicy:
    torch.manual_seed(18)
    actor = LegacyLocomotionActor(480)
    risk = BaselineInvariantHallTractionRiskEstimator()
    policy = AnchoredHallRecoveryPolicy(
        actor, risk, correction_limit=0.15, risk_gate_start=0.40, risk_gate_full=0.70
    ).eval()
    with torch.no_grad():
        policy.risk_estimator.network[-1].weight.zero_()
        policy.risk_estimator.network[-1].bias.fill_(risk_bias)
        policy.recovery_head[-1].weight.fill_(0.05)
    return policy


def test_high_risk_correction_is_bounded_and_baseline_is_frozen() -> None:
    policy = make_policy(10.0)
    action, base, risk, correction = policy.recovery_outputs(healthy_observation())
    assert action.shape == base.shape == correction.shape == (3, 29)
    assert risk.shape == (3, 1)
    assert torch.max(torch.abs(correction)) <= 0.15 + 1.0e-6
    assert not any(item.requires_grad for item in policy.baseline_actor.parameters())
    assert not any(item.requires_grad for item in policy.risk_estimator.parameters())


def test_low_risk_and_sensor_loss_are_exact_original_actor_fallbacks() -> None:
    policy = make_policy(-10.0)
    observation = healthy_observation()
    action, base, risk, correction = policy.recovery_outputs(observation)
    assert torch.all(risk < 0.40)
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)

    failed = observation.clone()
    failed[:, VALID_SLICE] = 0.0
    failed[:, AGE_SLICE] = 1.0
    action, base, risk, correction = policy.recovery_outputs(failed)
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=0.0, rtol=0.0)


def test_nonfinite_hall_keeps_recovery_finite_and_has_no_residual_authority() -> None:
    policy = make_policy(10.0)
    observation = healthy_observation(2)
    observation[:, 480:] = float("nan")
    action, base, risk, correction = policy.recovery_outputs(observation)
    assert torch.isfinite(action).all()
    assert torch.isfinite(base).all()
    assert torch.isfinite(correction).all()
    torch.testing.assert_close(action, base, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=0.0, rtol=0.0)


def test_low_mu_supervision_floor_is_training_only_and_bounded() -> None:
    deploy_gate = torch.tensor([0.0, 0.30, 0.80])
    mu = torch.tensor([0.08, 0.80, 0.08])
    supervised = offline_supervision_gate(deploy_gate, mu, 0.75)
    torch.testing.assert_close(supervised, torch.tensor([0.75, 0.30, 0.80]))
    torch.testing.assert_close(
        offline_supervision_gate(deploy_gate, mu, 0.0), deploy_gate
    )
