"""CPU and source-contract tests for the optional Hall health envelope."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from unitree_rl_lab.traction.health_envelope import (
    DEFAULT_COMMAND_X_INDICES,
    DEFAULT_MAX_PACKET_AGE_S,
    FAIL_STOP,
    HEALTHY,
    SINGLE_FOOT,
    HealthEnvelope,
    HealthEnvelopeCfg,
    rewrite_command_history,
    summarize_health_envelope_trace,
)


ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"
TEACHER_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "traction"
    / "frozen_speedboost_teacher.py"
)
MODULE_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "traction"
    / "health_envelope.py"
)


def _healthy_inputs(num_envs: int):
    return (
        torch.ones((num_envs, 2), dtype=torch.bool),
        torch.zeros((num_envs, 2)),
        torch.ones((num_envs, 2), dtype=torch.bool),
    )


def _assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    )
    return ast.literal_eval(assignment.value)


def test_first_healthy_update_is_exact_transparent_passthrough() -> None:
    envelope = HealthEnvelope(2, 0.02, "cpu")
    requested = torch.tensor([[0.8, 0.1, -0.2], [-0.4, -0.1, 0.3]])
    valid, age, finite = _healthy_inputs(2)
    output = envelope.update(requested, valid, age, finite)

    assert output.state.tolist() == [HEALTHY, HEALTHY]
    assert torch.equal(output.foot_healthy, torch.ones((2, 2), dtype=torch.bool))
    assert torch.allclose(output.target_command, requested)
    assert torch.allclose(output.effective_command, requested)
    assert not output.intervened.any()


def test_default_stale_threshold_matches_robot_bridge_and_boundary_is_inclusive() -> None:
    cfg = HealthEnvelopeCfg()
    assert DEFAULT_MAX_PACKET_AGE_S == pytest.approx(0.25)
    assert cfg.max_packet_age_s == pytest.approx(DEFAULT_MAX_PACKET_AGE_S)
    envelope = HealthEnvelope(2, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.0, 0.0], [0.8, 0.0, 0.0]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    finite = torch.ones((2, 2), dtype=torch.bool)
    age = torch.tensor([[0.25, 0.25], [0.2501, 0.0]])
    output = envelope.update(requested, valid, age, finite)
    assert output.state.tolist() == [HEALTHY, SINGLE_FOOT]


def test_single_foot_fallback_is_side_symmetric_and_caps_forward_speed() -> None:
    envelope = HealthEnvelope(2, 0.02, "cpu")
    requested = torch.tensor([[0.8, 0.2, 0.4], [-0.8, -0.2, -0.4]])
    valid = torch.tensor([[True, False], [False, True]])
    output = envelope.update(
        requested,
        valid,
        torch.zeros((2, 2)),
        torch.ones((2, 2), dtype=torch.bool),
    )

    assert output.state.tolist() == [SINGLE_FOOT, SINGLE_FOOT]
    assert torch.allclose(
        output.effective_command,
        torch.tensor([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]]),
    )
    assert output.intervened.tolist() == [True, True]


def test_both_feet_failed_targets_stop_with_bounded_deceleration() -> None:
    cfg = HealthEnvelopeCfg(linear_decel_rate=2.0, recovery_hold_s=0.2)
    envelope = HealthEnvelope(1, 0.1, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.0, 0.0]])
    valid, age, finite = _healthy_inputs(1)
    initial = envelope.update(requested, valid, age, finite)
    assert initial.effective_command[0, 0] == pytest.approx(0.8)

    failed = torch.zeros((1, 2), dtype=torch.bool)
    output = envelope.update(requested, failed, age, finite)
    assert output.state.item() == FAIL_STOP
    assert torch.equal(output.target_command, torch.zeros_like(requested))
    assert output.effective_command[0, 0].item() == pytest.approx(0.6)
    for _ in range(3):
        output = envelope.update(requested, failed, age, finite)
    assert torch.allclose(output.effective_command, torch.zeros_like(requested))


def test_initial_autozero_invalid_window_does_not_add_recovery_delay() -> None:
    cfg = HealthEnvelopeCfg(recovery_hold_s=0.5)
    envelope = HealthEnvelope(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.0, 0.0]])
    invalid = torch.zeros((1, 2), dtype=torch.bool)
    age = torch.zeros((1, 2))
    finite = torch.ones((1, 2), dtype=torch.bool)
    for _ in range(7):
        waiting = envelope.update(requested, invalid, age, finite)
        assert waiting.state.item() == FAIL_STOP
        assert torch.equal(waiting.effective_command, torch.zeros_like(requested))

    healthy = torch.ones((1, 2), dtype=torch.bool)
    acquired = envelope.update(requested, healthy, age, finite)
    assert acquired.state.item() == HEALTHY
    assert torch.allclose(acquired.effective_command, requested)
    assert acquired.recovery_timer_s.item() == pytest.approx(0.0)


def test_stale_age_nonfinite_packet_and_negative_age_are_unhealthy() -> None:
    cfg = HealthEnvelopeCfg(max_packet_age_s=0.1)
    envelope = HealthEnvelope(3, 0.02, "cpu", cfg)
    requested = torch.full((3, 3), 0.8)
    valid = torch.ones((3, 2), dtype=torch.bool)
    age = torch.tensor([[0.11, 0.0], [0.0, 0.0], [-0.01, 0.0]])
    finite = torch.tensor([[True, True], [True, False], [True, True]])
    output = envelope.update(requested, valid, age, finite)

    assert output.foot_healthy.tolist() == [[False, True], [True, False], [False, True]]
    assert output.state.tolist() == [SINGLE_FOOT, SINGLE_FOOT, SINGLE_FOOT]
    assert torch.allclose(output.effective_command[:, 0], torch.full((3,), 0.25))


def test_nonfinite_numeric_health_flag_is_never_cast_to_true() -> None:
    envelope = HealthEnvelope(1, 0.02, "cpu")
    output = envelope.update(
        torch.tensor([[0.8, 0.0, 0.0]]),
        torch.tensor([[1.0, float("nan")]]),
        torch.zeros((1, 2)),
        torch.ones((1, 2)),
    )
    assert output.valid.tolist() == [[True, False]]
    assert output.state.item() == SINGLE_FOOT
    assert output.effective_command[0, 0].item() == pytest.approx(0.25)


def test_recovery_requires_stable_hold_then_uses_acceleration_rate() -> None:
    cfg = HealthEnvelopeCfg(
        linear_accel_rate=0.3,
        linear_decel_rate=10.0,
        recovery_hold_s=0.3,
    )
    envelope = HealthEnvelope(1, 0.1, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.0, 0.0]])
    valid, age, finite = _healthy_inputs(1)
    envelope.update(requested, valid, age, finite)
    single = torch.tensor([[True, False]])
    degraded = envelope.update(requested, single, age, finite)
    assert degraded.state.item() == SINGLE_FOOT
    assert degraded.effective_command[0, 0].item() == pytest.approx(0.25)

    recovery_1 = envelope.update(requested, valid, age, finite)
    recovery_2 = envelope.update(requested, valid, age, finite)
    assert recovery_1.state.item() == SINGLE_FOOT
    assert recovery_2.state.item() == SINGLE_FOOT
    assert recovery_2.recovery_timer_s.item() == pytest.approx(0.2)
    recovered = envelope.update(requested, valid, age, finite)
    assert recovered.state.item() == HEALTHY
    assert recovered.recovery_timer_s.item() == pytest.approx(0.0)
    assert recovered.effective_command[0, 0].item() == pytest.approx(0.28)


def test_recovery_hysteresis_resets_when_health_flaps() -> None:
    cfg = HealthEnvelopeCfg(recovery_hold_s=0.2)
    envelope = HealthEnvelope(1, 0.1, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.0, 0.0]])
    valid, age, finite = _healthy_inputs(1)
    envelope.update(requested, valid, age, finite)
    single = torch.tensor([[True, False]])
    envelope.update(requested, single, age, finite)
    assert envelope.update(requested, valid, age, finite).state.item() == SINGLE_FOOT
    envelope.update(requested, single, age, finite)
    assert envelope.recovery_timer_s.item() == pytest.approx(0.0)
    assert envelope.update(requested, valid, age, finite).state.item() == SINGLE_FOOT


def test_per_environment_reset_does_not_change_other_state() -> None:
    envelope = HealthEnvelope(2, 0.02, "cpu")
    request = torch.full((2, 3), 0.8)
    valid, age, finite = _healthy_inputs(2)
    envelope.update(request, valid, age, finite)
    envelope.reset(torch.tensor([1]))
    assert envelope.state.tolist() == [HEALTHY, FAIL_STOP]
    assert torch.allclose(envelope.effective_command[0], request[0])
    assert torch.equal(envelope.effective_command[1], torch.zeros(3))


def test_command_history_rewrite_covers_all_frames_without_mutating_input() -> None:
    observation = torch.randn(2, 1864)
    original = observation.clone()
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rewritten = rewrite_command_history(observation, command)

    assert torch.equal(observation, original)
    for index in DEFAULT_COMMAND_X_INDICES:
        assert torch.equal(rewritten[:, index : index + 3], command)
    untouched = torch.ones(1864, dtype=torch.bool)
    for index in DEFAULT_COMMAND_X_INDICES:
        untouched[index : index + 3] = False
    assert torch.equal(rewritten[:, untouched], original[:, untouched])


def test_health_trace_summary_counts_states_and_interventions() -> None:
    report = summarize_health_envelope_trace(
        requested_command=torch.tensor([[0.8, 0.0, 0.0]] * 3),
        effective_command=torch.tensor(
            [[0.8, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        state=torch.tensor([HEALTHY, SINGLE_FOOT, FAIL_STOP]),
        valid=torch.tensor([[True, True], [True, False], [False, False]]),
        age_s=torch.zeros((3, 2)),
        finite=torch.ones((3, 2), dtype=torch.bool),
        foot_healthy=torch.tensor([[True, True], [True, False], [False, False]]),
        intervened=torch.tensor([False, True, True]),
    )
    assert report["sample_count"] == 3
    assert report["intervention_fraction"] == pytest.approx(2.0 / 3.0)
    assert report["by_state"]["HEALTHY"]["samples"] == 1
    assert report["by_state"]["SINGLE_FOOT"]["samples"] == 1
    assert report["by_state"]["FAIL_STOP"]["samples"] == 1
    assert report["health_patterns"] == {
        "both_healthy_samples": 1,
        "single_healthy_samples": 1,
        "neither_healthy_samples": 1,
        "invalid_flag_foot_samples": 3,
        "nonfinite_foot_samples": 0,
    }


def test_health_module_is_pure_torch_and_command_schema_matches_actor() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported_roots & {"isaaclab", "isaacsim", "omni", "pxr"}
    assert DEFAULT_COMMAND_X_INDICES == _assignment_literal(
        TEACHER_PATH, "COMMAND_VX_INDICES"
    )


def test_spatial_evaluator_opt_in_boundary_and_dataset_contract() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert '"--hall_health_envelope"' in source
    assert 'action="store_true"' in source
    assert 'default=0.25' in source
    assert "health_envelope: HealthEnvelope | None = None" in source
    assert "if health_envelope is None:" in source
    assert "_force_command(base_env, args_cli.command)" in source
    assert "observation = _rewrite_actor_command_history(" in source
    assert source.index("observation = _rewrite_actor_command_history(") < source.index(
        "actions = policy(observation)"
    )

    health_reader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_read_health_envelope_inputs"
    )
    loaded_names = {
        node.id
        for node in ast.walk(health_reader)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for forbidden in (
        "local_x",
        "low",
        "contact_patch",
        "contact_force",
        "friction",
        "ground_mu",
    ):
        assert forbidden not in loaded_names
    reader_source = ast.unparse(health_reader)
    assert "policy_delay_steps" in reader_source
    assert "reported_sample_period" in reader_source
    assert "delivered_age_s" in reader_source

    for field in (
        "health_requested_command",
        "health_effective_command",
        "health_state",
        "health_valid",
        "health_age_s",
        "health_finite",
        "health_foot_healthy",
        'report["health_envelope"]',
        '"max_packet_age_s"',
    ):
        assert field in source
    # Legacy datasets still begin with exactly the historical base payload;
    # optional diagnostic arrays are added only when present in ``trace``.
    assert '"observation": trace["dataset_observation"]' in source
    assert '"action": trace["dataset_action"]' in source
    assert '"low": trace["dataset_low"]' in source
    assert "if name in trace:" in source
