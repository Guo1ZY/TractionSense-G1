"""Shape, initialization, and baseline-preservation tests for torque networks."""

from __future__ import annotations

import torch

from unitree_rl_lab.traction_torque.networks import (
    TemporalForceCorrector,
    TorqueTractionStudentCfg,
    TorqueTractionStudentPolicy,
    TorqueTractionTeacherCfg,
    TorqueTractionTeacherPolicy,
    torque_history_to_legacy_proprio,
)


def test_history_conversion_is_term_major_newest_five() -> None:
    history = torch.arange(15 * 125, dtype=torch.float32).reshape(1, 15, 125)
    converted = torque_history_to_legacy_proprio(history)
    expected = torch.cat([history[:, -5:, item].reshape(1, -1) for item in (
        slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 38), slice(38, 67), slice(67, 96)
    )], dim=-1)
    torch.testing.assert_close(converted, expected)
    assert converted.shape == (1, 480)


def test_student_starts_as_exact_baseline_actor_and_has_all_heads() -> None:
    torch.manual_seed(3)
    student = TorqueTractionStudentPolicy(TorqueTractionStudentCfg(freeze_baseline=False))
    history = torch.randn(4, 15, 125)
    output = student(history)
    baseline = student.baseline_actor(torque_history_to_legacy_proprio(history))
    torch.testing.assert_close(output.action, baseline, atol=0.0, rtol=0.0)
    assert output.action.shape == (4, 29)
    assert output.estimated_force.shape == (4, 6)
    assert output.contact_probability.shape == (4, 2)
    assert output.slip_probability.shape == (4, 2)
    assert output.traction_utilization.shape == (4, 2)
    assert output.traction_margin.shape == (4, 2)
    assert output.estimator_confidence.shape == (4, 2)
    assert output.traction_latent.shape == (4, 16)
    assert torch.isfinite(output.action).all()


def test_force_corrector_starts_as_analytical_identity() -> None:
    corrector = TemporalForceCorrector()
    history = torch.randn(2, 15, 125)
    analytical = torch.randn(2, 6)
    result = corrector(history, analytical)
    torch.testing.assert_close(result.corrected_force_n, analytical, atol=0.0, rtol=0.0)
    assert result.confidence.shape == (2, 2)


def test_student_residual_is_explicitly_bounded() -> None:
    student = TorqueTractionStudentPolicy(TorqueTractionStudentCfg(freeze_baseline=False, residual_action_limit=0.7))
    student.residual_gate_logit.data.fill_(100.0)
    student.residual_actor[-1].bias.data.fill_(100.0)
    history = torch.randn(4, 15, 125)
    output = student(history)
    baseline = student.baseline_actor(torque_history_to_legacy_proprio(history))
    assert float((output.action - baseline).abs().max()) <= 0.700001


def test_teacher_auxiliary_heads_and_action_dimension() -> None:
    teacher = TorqueTractionTeacherPolicy(TorqueTractionTeacherCfg(privileged_input_dim=42))
    output = teacher(torch.randn(3, 480), torch.randn(3, 3), torch.randn(3, 42))
    assert output.action.shape == (3, 29)
    assert output.traction_latent.shape == (3, 16)
    assert output.force_correction_target.shape == (3, 6)
