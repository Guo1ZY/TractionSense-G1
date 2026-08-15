"""Pure-CPU tests for the optional deployable stability envelope."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from unitree_rl_lab.traction.high_speed_stability_envelope import (
    COMMAND_X_INDICES,
    COMMAND_YAW_INDICES,
    EMERGENCY,
    LIMIT,
    NORMAL,
    REASON_ACTION_NORM,
    REASON_ACTION_SATURATION,
    REASON_HEADING_LIMIT_048,
    REASON_HEADING_OMEGA,
    REASON_HEADING_WARN,
    REASON_TILT,
    REASON_TURNING_OMEGA,
    WARN,
    HighSpeedStabilityEnvelope,
    HighSpeedStabilityEnvelopeCfg,
    summarize_high_speed_stability_trace,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "traction"
    / "high_speed_stability_envelope.py"
)
EVAL_PATH = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"


def _observation(
    num_envs: int = 1,
    *,
    vx: float = 0.8,
    yaw: float = 0.0,
    heading: float = 0.0,
    omega_x: float = 0.0,
    omega_y: float = 0.0,
    tilt: float = 0.0,
    previous_action: torch.Tensor | None = None,
    current_action: torch.Tensor | None = None,
) -> torch.Tensor:
    observation = torch.zeros((num_envs, 1864))
    observation[:, list(COMMAND_X_INDICES)] = vx
    observation[:, list(COMMAND_YAW_INDICES)] = yaw
    observation[:, 1863] = heading
    # Actor base-angular-velocity observations are scaled by 0.2.
    observation[:, 12] = omega_x * 0.2
    observation[:, 13] = omega_y * 0.2
    observation[:, 27] = tilt
    observation[:, 29] = -1.0
    history = observation[:, 335:480].reshape(num_envs, 5, 29)
    if previous_action is not None:
        history[:, -2] = previous_action
    if current_action is not None:
        history[:, -1] = current_action
    return observation


def _command(num_envs: int = 1, vx: float = 0.8) -> torch.Tensor:
    command = torch.zeros((num_envs, 3))
    command[:, 0] = vx
    return command


def test_nominal_update_is_exact_only_attenuating_passthrough() -> None:
    envelope = HighSpeedStabilityEnvelope(2, "cpu")
    observation = _observation(2)
    command = torch.tensor([[0.8, 0.2, -0.03], [-0.4, -0.2, 0.03]])
    output = envelope.update(observation, command)

    assert output.state.tolist() == [NORMAL, NORMAL]
    assert torch.equal(output.effective_command, command)
    assert not output.intervened.any()
    assert output.heading_enabled.tolist() == [True, True]


def test_warn_requires_five_frames_and_caps_forward_only() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    observation = _observation(heading=0.41)
    command = torch.tensor([[0.8, 0.2, -0.03]])
    for _ in range(4):
        output = envelope.update(observation, command)
        assert output.state.item() == NORMAL
    output = envelope.update(observation, command)

    assert output.state.item() == WARN
    assert output.reason_mask.item() & REASON_HEADING_WARN
    assert torch.allclose(
        output.effective_command, torch.tensor([[0.55, 0.2, -0.03]])
    )
    assert not output.heading_correction_active.item()
    assert output.heading_correction_yaw.item() == pytest.approx(0.0)


def test_default_off_preserves_v1_yaw_exactly_in_all_non_normal_states() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    command = torch.tensor([[0.8, 0.0, 0.03]])
    observation = _observation(heading=0.49)
    for _ in range(5):
        output = envelope.update(observation, command)
    assert output.state.item() == LIMIT
    assert output.effective_command[0, 2].item() == command[0, 2].item()
    assert not output.heading_correction_active.item()


@pytest.mark.parametrize(
    ("heading", "expected_yaw"),
    [(0.41, -0.328), (-0.41, 0.328)],
)
def test_opt_in_heading_correction_has_negative_feedback_sign(
    heading: float, expected_yaw: float
) -> None:
    envelope = HighSpeedStabilityEnvelope(
        1,
        "cpu",
        HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True),
    )
    observation = _observation(heading=heading)
    for _ in range(5):
        output = envelope.update(observation, _command())
    assert output.state.item() >= WARN
    assert output.heading_correction_active.item()
    assert output.effective_command[0, 2].item() == pytest.approx(expected_yaw)
    assert output.heading_correction_yaw.item() == pytest.approx(expected_yaw)


def test_opt_in_heading_correction_is_bounded_and_normal_is_transparent() -> None:
    cfg = HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True)
    normal = HighSpeedStabilityEnvelope(1, "cpu", cfg).update(
        _observation(heading=0.20), torch.tensor([[0.8, 0.0, 0.03]])
    )
    assert normal.state.item() == NORMAL
    assert normal.effective_command[0, 2].item() == pytest.approx(0.03)
    assert not normal.heading_correction_active.item()

    envelope = HighSpeedStabilityEnvelope(1, "cpu", cfg)
    observation = _observation(heading=0.80)
    for _ in range(3):
        bounded = envelope.update(observation, _command())
    assert bounded.state.item() == LIMIT
    assert bounded.effective_command[0, 2].item() == pytest.approx(-0.40)
    assert bounded.heading_correction_yaw.item() == pytest.approx(-0.40)


def test_correction_does_not_misclassify_its_rewritten_history_as_a_turn() -> None:
    envelope = HighSpeedStabilityEnvelope(
        1,
        "cpu",
        HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True),
    )
    observation = _observation(heading=0.49)
    raw_yaw_history = torch.zeros(5)
    outputs = []
    for _ in range(10):
        output = envelope.update(observation, _command())
        outputs.append(output)
        # Mirror the environment observation manager, which advances one raw
        # command-history sample per control step.  The evaluator rewrites a
        # policy-view clone, not the manager's persistent five-frame buffer.
        raw_yaw_history = torch.roll(raw_yaw_history, shifts=-1)
        raw_yaw_history[-1] = output.effective_command[0, 2]
        observation[:, list(COMMAND_YAW_INDICES)] = raw_yaw_history

    # The correction must remain continuously enabled.  In the old
    # last-sample subtraction implementation it pulsed once every six steps,
    # resetting both limit counters and leaving the faster WARN cap active.
    assert all(item.heading_enabled.item() for item in outputs)
    assert all(item.heading_correction_active.item() for item in outputs[2:])
    assert output.heading_command_mean.item() == pytest.approx(0.0, abs=1.0e-6)
    assert output.state.item() == LIMIT
    assert output.limit_count.item() >= 5
    assert output.hard_limit_count.item() >= 3
    assert output.effective_command[0, 0].item() == pytest.approx(0.40)
    assert output.effective_command[0, 2].item() == pytest.approx(-0.392)


def test_correction_fifo_is_per_environment_and_selected_reset_clears_it() -> None:
    envelope = HighSpeedStabilityEnvelope(
        2,
        "cpu",
        HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True),
    )
    observation = _observation(2, heading=0.49)
    raw_yaw_history = torch.zeros((2, 5))
    for _ in range(4):
        output = envelope.update(observation, _command(2))
        raw_yaw_history = torch.roll(raw_yaw_history, shifts=-1, dims=1)
        raw_yaw_history[:, -1] = output.effective_command[:, 2]
        observation[:, list(COMMAND_YAW_INDICES)] = raw_yaw_history
    assert torch.all(torch.any(envelope._heading_correction_history != 0.0, dim=1))

    envelope.reset(torch.tensor([1]))
    assert torch.all(envelope._heading_correction_history[1] == 0.0)
    assert torch.any(envelope._heading_correction_history[0] != 0.0)


def test_commanded_turn_does_not_count_as_recovery_from_heading_limit() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    hazard = _observation(heading=0.49)
    for _ in range(3):
        output = envelope.update(hazard, _command())
    assert output.state.item() == LIMIT

    # Reset-relative heading is invalid for new heading triggers during a
    # commanded turn, but disabling it must not be treated as a safe sample.
    turning = _observation(yaw=0.06, heading=0.49)
    for _ in range(20):
        output = envelope.update(turning, torch.tensor([[0.8, 0.0, 0.06]]))
    assert not output.heading_enabled.item()
    assert output.state.item() == LIMIT
    assert output.recovery_count.item() == 0


def test_new_upstream_turn_disables_existing_correction_immediately() -> None:
    envelope = HighSpeedStabilityEnvelope(
        1,
        "cpu",
        HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True),
    )
    observation = _observation(heading=0.41)
    for _ in range(5):
        output = envelope.update(observation, _command())
    for index in COMMAND_X_INDICES:
        observation[:, index : index + 3] = output.effective_command
    turning = envelope.update(observation, torch.tensor([[0.8, 0.0, 0.06]]))
    assert not turning.heading_enabled.item()
    assert not turning.heading_correction_active.item()
    assert turning.effective_command[0, 2].item() == pytest.approx(0.06)


def test_opt_in_heading_correction_is_transparent_during_commanded_turn() -> None:
    envelope = HighSpeedStabilityEnvelope(
        1,
        "cpu",
        HighSpeedStabilityEnvelopeCfg(enable_heading_correction=True),
    )
    command = torch.tensor([[0.8, 0.0, 0.06]])
    # Turning omega emergency enters EMERGENCY, but reset-relative heading must
    # still have zero authority over the yaw command.
    output = envelope.update(
        _observation(yaw=0.06, heading=0.80, omega_x=1.21), command
    )
    assert output.state.item() == EMERGENCY
    assert not output.heading_enabled.item()
    assert not output.heading_correction_active.item()
    assert output.effective_command[0, 2].item() == pytest.approx(0.06)
    assert output.heading_correction_yaw.item() == pytest.approx(0.0)


def test_both_limit_persistence_paths_take_precedence_over_warn() -> None:
    five_frame = HighSpeedStabilityEnvelope(1, "cpu")
    observation = _observation(heading=0.46)
    for _ in range(5):
        output = five_frame.update(observation, _command())
    assert output.state.item() == LIMIT
    assert output.effective_command[0, 0].item() == pytest.approx(0.40)

    three_frame = HighSpeedStabilityEnvelope(1, "cpu")
    observation = _observation(heading=0.49)
    for _ in range(3):
        output = three_frame.update(observation, _command())
    assert output.state.item() == LIMIT
    assert output.reason_mask.item() & REASON_HEADING_LIMIT_048
    assert output.effective_command[0, 0].item() == pytest.approx(0.40)


def test_upstream_high_request_keeps_limit_detection_alive_after_warn_rewrite() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    observation = _observation(heading=0.49)
    # Mirror evaluator synchronization after an earlier WARN: actor history is
    # capped, while the upstream operator request remains high.
    observation[:, list(COMMAND_X_INDICES)] = 0.55
    for _ in range(3):
        output = envelope.update(observation, _command(vx=0.8))
    assert output.state.item() == LIMIT
    assert output.reason_mask.item() & REASON_HEADING_LIMIT_048


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (_observation(heading=0.49, omega_x=1.21), REASON_HEADING_OMEGA),
        (_observation(tilt=0.181), REASON_TILT),
        (
            _observation(current_action=torch.full((29,), 0.8)),
            REASON_ACTION_NORM,
        ),
        (
            _observation(
                current_action=torch.tensor([2.5, -2.5] + [0.0] * 27)
            ),
            REASON_ACTION_SATURATION,
        ),
    ],
)
def test_each_emergency_path_is_immediate(
    observation: torch.Tensor, reason: int
) -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    output = envelope.update(observation, _command())
    assert output.state.item() == EMERGENCY
    assert output.reason_mask.item() & reason
    assert output.effective_command[0, 0].item() == pytest.approx(0.25)


def test_action_history_norm_and_slew_are_actor_visible_diagnostics() -> None:
    previous = torch.full((29,), 0.1)
    current = torch.full((29,), 0.3)
    output = HighSpeedStabilityEnvelope(1, "cpu").update(
        _observation(previous_action=previous, current_action=current), _command()
    )
    assert output.previous_action_norm.item() == pytest.approx(
        torch.linalg.vector_norm(previous).item()
    )
    assert output.current_action_norm.item() == pytest.approx(
        torch.linalg.vector_norm(current).item()
    )
    assert output.action_slew_norm.item() == pytest.approx(
        torch.linalg.vector_norm(current - previous).item()
    )


def test_recovery_requires_ten_safe_frames_and_never_auto_downgrades() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    hazard = _observation(heading=0.41)
    for _ in range(5):
        output = envelope.update(hazard, _command())
    assert output.state.item() == WARN

    safe = _observation(heading=0.29, omega_x=0.79, tilt=0.09)
    for step in range(9):
        output = envelope.update(safe, _command())
        assert output.state.item() == WARN, step
        assert output.effective_command[0, 0].item() == pytest.approx(0.55)
    output = envelope.update(safe, _command())
    assert output.state.item() == NORMAL
    assert output.effective_command[0, 0].item() == pytest.approx(0.8)


def test_recovery_counter_resets_when_a_safe_signal_flaps() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    emergency = _observation(tilt=0.19)
    assert envelope.update(emergency, _command()).state.item() == EMERGENCY
    safe = _observation(heading=0.0, tilt=0.0)
    for _ in range(5):
        envelope.update(safe, _command())
    unsafe = _observation(heading=0.0, tilt=0.11)
    output = envelope.update(unsafe, _command())
    assert output.state.item() == EMERGENCY
    assert output.recovery_count.item() == 0


def test_turning_disables_heading_only_checks_but_retains_omega_emergency() -> None:
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    intentional_turn = _observation(yaw=0.06, heading=0.80)
    for _ in range(8):
        output = envelope.update(intentional_turn, _command())
    assert output.state.item() == NORMAL
    assert not output.heading_enabled.item()
    assert output.warn_count.item() == 0
    assert output.hard_limit_count.item() == 0

    turning_instability = _observation(
        yaw=0.06, heading=0.80, omega_y=1.21
    )
    output = envelope.update(turning_instability, _command())
    assert output.state.item() == EMERGENCY
    assert output.reason_mask.item() & REASON_TURNING_OMEGA
    assert not (output.reason_mask.item() & REASON_HEADING_OMEGA)


def test_cap_never_increases_small_or_negative_forward_command() -> None:
    envelope = HighSpeedStabilityEnvelope(2, "cpu")
    observation = _observation(2, heading=0.49, omega_x=1.21)
    upstream = torch.tensor([[0.20, 0.3, 0.1], [-0.80, -0.3, -0.1]])
    output = envelope.update(observation, upstream)
    assert torch.allclose(
        output.effective_command,
        torch.tensor([[0.20, 0.3, 0.1], [-0.25, -0.3, -0.1]]),
    )
    assert (
        output.effective_command[:, 0].abs() <= upstream[:, 0].abs()
    ).all()


def test_per_environment_reset_preserves_other_state_and_counters() -> None:
    envelope = HighSpeedStabilityEnvelope(2, "cpu")
    observation = _observation(2, heading=0.41)
    for _ in range(5):
        envelope.update(observation, _command(2))
    envelope.reset(torch.tensor([1]))
    assert envelope.state.tolist() == [WARN, NORMAL]
    assert envelope.warn_count.tolist() == [5, 0]


def test_summary_counts_states_reasons_and_reduction() -> None:
    report = summarize_high_speed_stability_trace(
        upstream_command=torch.tensor([[0.8, 0.0, 0.0]] * 4),
        effective_command=torch.tensor(
            [
                [0.8, 0.0, 0.0],
                [0.55, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        ),
        state=torch.tensor([NORMAL, WARN, LIMIT, EMERGENCY]),
        reason_mask=torch.tensor(
            [0, REASON_HEADING_WARN, REASON_HEADING_LIMIT_048, REASON_TILT]
        ),
        intervened=torch.tensor([False, True, True, True]),
        heading_enabled=torch.tensor([True, True, True, False]),
        heading_command_mean=torch.tensor([0.0, 0.0, 0.0, 0.06]),
        heading_error=torch.tensor([0.0, 0.41, 0.49, 0.8]),
        heading_error_abs=torch.tensor([0.0, 0.41, 0.49, 0.8]),
        heading_correction_active=torch.tensor([False, True, True, False]),
        heading_correction_yaw=torch.tensor([0.0, -0.328, -0.4, 0.0]),
        omega_xy=torch.tensor([0.0, 0.2, 0.3, 1.3]),
        tilt=torch.tensor([0.0, 0.02, 0.03, 0.19]),
    )
    assert report["sample_count"] == 4
    assert report["intervention_fraction"] == pytest.approx(0.75)
    assert report["by_state"]["EMERGENCY"]["samples"] == 1
    assert report["reason_sample_counts"]["projected_gravity_tilt_emergency"] == 1
    assert report["heading_correction_active_fraction"] == pytest.approx(0.5)


def test_module_is_pure_torch_and_has_no_forbidden_runtime_inputs() -> None:
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
    assert not imported_roots & {"isaaclab", "isaacsim", "omni", "pxr", "numpy"}
    update = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    loaded_names = {
        node.id
        for node in ast.walk(update)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not loaded_names & {
        "mu",
        "friction",
        "contact",
        "contact_force",
        "force",
        "course_stage",
        "stage",
    }


def test_spatial_evaluator_is_default_off_and_records_aligned_diagnostics() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    assert '"--high_speed_stability_envelope"' in source
    assert '"--stability_heading_correction"' in source
    assert '"--stability_conservative_preset"' in source
    assert '"--stability_early_heading_preset"' in source
    assert "early_heading_or_conservative = conservative or early_heading" in source
    assert "emergency_speed_cap=0.10 if conservative else 0.25" in source
    assert (
        "--stability_conservative_preset requires --high_speed_stability_envelope"
        in source
    )
    assert 'action="store_true"' in source
    assert "stability_envelope: HighSpeedStabilityEnvelope | None = None" in source
    assert "upstream_command = health_output.effective_command" in source
    assert "effective_command = stability_output.effective_command" in source
    assert source.index("observation = _rewrite_actor_command_history(") < source.index(
        "actions = policy(observation)"
    )
    for field in (
        "stability_upstream_command",
        "stability_effective_command",
        "stability_state",
        "stability_reason_mask",
        "stability_heading_command_mean",
        "stability_heading_error_abs",
        "stability_heading_correction_active",
        "stability_heading_correction_yaw",
        "stability_action_slew_norm",
        "stability_rollout_step",
        'report["high_speed_stability_envelope"]',
    ):
        assert field in source


def test_invalid_configuration_and_shapes_fail_closed() -> None:
    with pytest.raises(ValueError):
        HighSpeedStabilityEnvelopeCfg(
            recovery_heading_threshold=0.45
        ).validate()
    with pytest.raises(ValueError):
        HighSpeedStabilityEnvelopeCfg(heading_correction_abs_cap=0.0).validate()
    envelope = HighSpeedStabilityEnvelope(1, "cpu")
    with pytest.raises(ValueError):
        envelope.update(torch.zeros(1, 100), _command())
    malformed = _observation()
    malformed[0, 1863] = float("nan")
    with pytest.raises(FloatingPointError):
        envelope.update(malformed, _command())
