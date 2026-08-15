from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/rsl_rl/eval_uniform_high_friction_long.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_evaluator_has_exact_policy_boundaries_and_mutually_exclusive_modes() -> None:
    source = _source()
    assert "POLICY_DIM = 1864" in source
    assert "LEGACY_DIM = 480" in source
    assert "HIGH_SPEED_DIM = 482" in source
    assert "args_cli.high_speed_backbone_checkpoint" in source
    assert "load_proprio_baseline" in source
    assert "load_hall_backbone" in source
    assert "load_high_speed_backbone" in source
    # FastBase gate/residual checkpoints cannot be loaded by the plain
    # ``load_hall_backbone`` path.  The runner-based path is opt-in behind
    # ``--rsl_rl_cfg_entry_point`` and must never replace the standalone
    # legacy loaders for ordinary checkpoints.
    assert "OnPolicyRunner" in source
    assert '"--rsl_rl_cfg_entry_point"' in source
    assert "native_fastbase_rsl_runner_candidate" in source
    assert '"--disable_fabric"' in source
    assert '"--hall_contact_distribution"' in source
    assert 'choices=("aggregate", "detailed")' in source
    assert '"--print_progress"' in source


def test_actor_terms_exclude_privileged_contact_force_mu_slip_stage() -> None:
    source = _source()
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXPECTED_POLICY_TERMS" for target in node.targets)
    )
    terms = ast.literal_eval(assignment.value)
    # The field name is a deliberately retained checkpoint/config ABI alias;
    # its callable and values must be audited as lateral motion feedback.
    assert terms[-1] == "foot_sensor_age_lr"
    assert "foot_magnetic_array" in terms
    lowered = " ".join(terms).lower()
    for token in ("contact", "force", "friction", "slip", "stage"):
        assert token not in lowered
    assert "term_cfgs[-1].func is not mdp.lateral_motion_feedback" in source
    assert '"lateral_motion_feedback"' in source
    assert '"trailing_feature_mode": "motion_feedback"' in source
    assert "_audit_policy_runtime_values(base, observation, actor_schema)" in source
    assert "motion feedback [body_vy,relative_heading]" in source


def test_constant_command_history_is_audited_at_the_canonical_slice() -> None:
    source = _source()
    assert "COMMAND_SLICE = slice(30, 45)" in source
    assert "reshape(-1, 5, 3)" in source
    assert "_audit_command_history(policy_obs, args_cli.command)" in source
    assert "term.vel_command_b[:, 0] = float(command)" in source


def test_first_failure_censoring_is_persistent() -> None:
    source = _source()
    assert "pre_active = active.clone()" in source
    assert "active &= ~done.bool()" in source
    assert "metric_mask = pre_active" in source
    assert "first episode only" in source
    assert "managed-reset" in source


def test_long_horizon_gate_does_not_reward_slow_survival() -> None:
    source = _source()
    assert 'default=1500' in source
    assert 'default=0.69' in source
    assert 'default=0.65' in source
    assert '"zero_falls"' in source
    assert '"heading_rms"' in source
    assert '"body_vy_rms"' in source
    assert '"action_saturation"' in source


def test_trace_contains_failure_precursors_and_hall_health() -> None:
    source = _source()
    for field in (
        '"heading"',
        '"vy"',
        '"omega_x"',
        '"omega_y"',
        '"omega_z"',
        '"action_norm"',
        '"action_slew"',
        '"hall_valid_left"',
        '"hall_valid_right"',
    ):
        assert field in source
    assert "Exactly one device-to-host transfer per step" in source


def test_script_parses() -> None:
    ast.parse(_source())


def test_strict_plain_hall_actor_loader_round_trip(tmp_path: Path) -> None:
    import torch

    from unitree_rl_lab.traction.networks import LegacyLocomotionActor
    from unitree_rl_lab.traction.proprio_baseline import load_hall_backbone

    actor = LegacyLocomotionActor(1864).eval()
    checkpoint = tmp_path / "hall_actor.pt"
    state = {key: value.clone() for key, value in actor.state_dict().items()}
    state["distribution.std_param"] = torch.full((29,), 0.12)
    torch.save({"actor_state_dict": state}, checkpoint)
    loaded = load_hall_backbone(checkpoint)
    observation = {"policy": torch.randn(3, 1864)}
    with torch.inference_mode():
        expected = actor(observation["policy"])
        actual = loaded(observation)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_strict_482_actor_loader_round_trip(tmp_path: Path) -> None:
    import torch

    from unitree_rl_lab.traction.networks import LegacyLocomotionActor
    from unitree_rl_lab.traction.proprio_baseline import load_high_speed_backbone

    actor = LegacyLocomotionActor(482).eval()
    checkpoint = tmp_path / "high_speed_actor.pt"
    state = {key: value.clone() for key, value in actor.state_dict().items()}
    state["distribution.std_param"] = torch.full((29,), 0.05)
    torch.save({"actor_state_dict": state}, checkpoint)
    loaded = load_high_speed_backbone(checkpoint)
    observation = {"high_speed_policy": torch.randn(3, 482)}
    with torch.inference_mode():
        expected = actor(observation["high_speed_policy"])
        actual = loaded(observation)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_evaluator_audits_482_mapping_bit_for_bit() -> None:
    source = _source()
    assert 'observation["high_speed_policy"]' in source
    assert '"policy[0:480]"' in source
    assert '"policy[1862:1864]"' in source
    assert '"high-speed 482-D proprio prefix"' in source
    assert '"high-speed 482-D motion tail"' in source
    assert "torch.cat(" not in source[source.index("def _audit_high_speed_runtime_values"):source.index("def _audit_policy_runtime_values")]
    assert '"body_vy_m_s", "relative_heading_rad"' in source
