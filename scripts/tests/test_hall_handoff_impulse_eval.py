#!/usr/bin/env python3
"""Unit and source-boundary tests for the Stage7 handoff evaluator."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from unitree_rl_lab.traction.networks import LegacyLocomotionActor
from unitree_rl_lab.traction.proprio_baseline import (
    HALL_POLICY_DIM,
    LEGACY_PROPRIO_DIM,
    load_proprio_baseline,
)


ROOT = Path(__file__).resolve().parents[2]
HELPER = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/traction/handoff_metrics.py"
)
EVALUATOR = ROOT / "scripts/rsl_rl/eval_hall_handoff_impulse.py"


spec = importlib.util.spec_from_file_location("handoff_metrics", HELPER)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_body_forward_impulse_respects_robot_yaw() -> None:
    half = math.sqrt(0.5)
    quaternion = torch.tensor([[half, 0.0, 0.0, half]])
    forward = MODULE.body_forward_axis_world(quaternion)
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0.0)


def test_roll_pitch_conversion_uses_wxyz() -> None:
    angle = 0.30
    roll_quaternion = torch.tensor(
        [[math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]]
    )
    roll, pitch = MODULE.roll_pitch_from_wxyz(roll_quaternion)
    torch.testing.assert_close(roll, torch.tensor([angle]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(pitch, torch.zeros(1), atol=1e-6, rtol=0.0)


def test_one_second_deceleration_excludes_fallen_rows() -> None:
    result = MODULE.one_second_deceleration(
        torch.tensor([0.80, 0.75, 0.70]),
        torch.tensor([0.20, 0.80, float("nan")]),
        torch.tensor([True, False, True]),
        speed_limit=0.25,
    )
    assert result["valid_count"] == 1.0
    assert result["initial_mean_m_s"] == pytest.approx(0.80)
    assert result["after_mean_m_s"] == pytest.approx(0.20)
    assert result["reduction_mean_m_s"] == pytest.approx(0.60)
    assert result["pass_fraction"] == 1.0


def test_evaluator_enforces_exact_hall_only_actor_boundary() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "tensor.shape[1] != 1864" in source
    assert '"foot_magnetic_array"' in source
    assert '"foot_contact_force"' not in source
    assert '"ground_friction_mu"' not in source.split("def _force_mu", 1)[0]
    assert "FORBIDDEN_POLICY_TOKENS" in source
    assert '"contact",' in source
    assert "policy_terms != EXPECTED_POLICY_TERMS" not in source
    assert "terms != EXPECTED_POLICY_TERMS" in source


def test_original_baseline_strict_loader_uses_only_480_prefix(tmp_path: Path) -> None:
    torch.manual_seed(27)
    actor = LegacyLocomotionActor(LEGACY_PROPRIO_DIM).eval()
    checkpoint = tmp_path / "model_49999_layout.pt"
    torch.save(
        {
            "actor_state_dict": {
                **actor.state_dict(),
                "distribution.std_param": torch.ones(29),
            }
        },
        checkpoint,
    )
    baseline = load_proprio_baseline(checkpoint)
    observation = torch.randn(4, HALL_POLICY_DIM)
    changed_hall = observation.clone()
    changed_hall[:, LEGACY_PROPRIO_DIM:] = float("nan")

    with torch.inference_mode():
        expected = actor(observation[:, :LEGACY_PROPRIO_DIM])
        actual = baseline({"policy": observation})
        changed = baseline({"policy": changed_hall})

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed, expected, atol=0.0, rtol=0.0)
    assert actual.shape == (4, 29)


def test_original_baseline_rejects_non_hall_environment_dimension(
    tmp_path: Path,
) -> None:
    actor = LegacyLocomotionActor(LEGACY_PROPRIO_DIM).eval()
    checkpoint = tmp_path / "baseline.pt"
    torch.save(
        {"actor_state_dict": actor.state_dict()},
        checkpoint,
    )
    baseline = load_proprio_baseline(checkpoint)

    with pytest.raises(RuntimeError, match="same 1864-D Hall environment"):
        baseline({"policy": torch.zeros(2, LEGACY_PROPRIO_DIM)})


def test_original_baseline_loader_rejects_incompatible_actor(tmp_path: Path) -> None:
    actor = LegacyLocomotionActor(LEGACY_PROPRIO_DIM).eval()
    state = actor.state_dict()
    state.pop("mlp.6.bias")
    checkpoint = tmp_path / "bad.pt"
    torch.save({"actor_state_dict": state}, checkpoint)

    with pytest.raises(ValueError, match="incompatible legacy actor keys"):
        load_proprio_baseline(checkpoint)


def test_evaluator_baseline_branch_preserves_fair_runtime_contracts() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert '"--proprio_baseline_checkpoint"' in source
    assert "select exactly one policy" in source
    assert "tensor.shape[1] != 1864" in source
    assert "runner = OnPolicyRunner(" in source
    assert "intentionally never loaded into this runner" in source
    assert "load_proprio_baseline(" in source
    assert "empirical_normalization=False" in source
    assert "JointPositionAction scale=0.25" in source
    assert '"consumed_observation_dimension"' in source


def test_warmup_failed_rows_never_contaminate_handoff_metrics() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert "warmup_failed |= warmup_step_falls" in source
    assert "eligible = ~warmup_failed" in source
    assert "alive = eligible & ~fallen" in source
    assert "one_second_survivor = eligible & ~fallen" in source
    assert "valid_mask=one_second_survivor" in source
    assert "post_reset_fall_events" in source


def test_evaluator_records_required_safety_outputs() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    for field in (
        "first_fall_s",
        "vx_1s_mean_m_s",
        "decel_1s_mean_m_s",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "nan_detected",
    ):
        assert field in source
    assert "body_forward_axis_world" in source
    assert "robot.write_root_velocity_to_sim" in source


def test_json_manifest_replaces_non_finite_values() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "def _strict_json" in source
    assert "allow_nan=False" in source


# Keep pytest imported after the module-by-path setup so this test file never
# imports the Isaac package just to use approx().
import pytest  # noqa: E402
