from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _class_block(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_all_three_floor_segments_are_long_opaque_high_grip() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    )
    block = _class_block(path, "HallUniformHighFrictionLongSceneCfg")
    assert block.count("friction=0.90") == 3
    assert block.count("color=(0.05, 0.30, 0.78)") == 3
    assert "size_x=8.0" in block
    assert "size_x=12.0" in block
    assert "size_x=20.0" in block


def test_high_grip_task_has_no_material_transition_or_stage_reward() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    )
    block = _class_block(
        path,
        "RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg",
    )
    assert "update_uniform_high_friction_buffer" in block
    assert "self.events.spatial_friction_update = None" in block
    assert "self.terminations.course_success = None" in block
    assert 'cruise_speed_range = (0.80, 0.80)' in block
    assert 'high_speed_range = (0.80, 0.80)' in block
    assert "episode_length_s = 30.0" in block
    assert "push_by_setting_velocity" in block


def test_runner_uses_exact_actor_and_critic_groups() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
    )
    block = _class_block(
        path, "FootTractionHallUniformHighFrictionLongBackbonePPORunnerCfg"
    )
    assert 'obs_groups = {"actor": ["policy"], "critic": ["critic"]}' in block
    assert "hidden_dims=[512, 256, 128]" in block
    assert "learning_rate=8.0e-6" in block
    assert "num_steps_per_env = 64" in block


def test_h0_warmup_removes_stressors_but_keeps_long_task() -> None:
    env_path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    )
    block = _class_block(
        env_path,
        "RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneWarmupEnvCfg",
    )
    assert "self.events.push_robot = None" in block
    for name in (
        '"add_base_mass"',
        '"base_com"',
        '"actuator_gains"',
        '"motor_strength"',
        '"joint_dynamics"',
    ):
        assert name in block
    assert "max_delay = 0" in block
    assert "enable_domain_randomization = False" in block
    assert '"x": (0.10, 0.20)' in block

    runner_path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
    )
    runner = _class_block(
        runner_path,
        "FootTractionHallUniformHighFrictionLongBackboneWarmupPPORunnerCfg",
    )
    assert "init_std=0.05" in runner
    assert "learning_rate=3.0e-6" in runner
    assert "max_iterations = 300" in runner


def test_nominal_play_really_disables_dynamics_and_latency_dr() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    )
    block = _class_block(
        path,
        "RobotFootTractionMagneticMotionUniformHighFrictionLongBackbonePlayEnvCfg",
    )
    assert "max_delay = 0" in block
    assert "delay_probabilities = (1.0,)" in block
    for name in (
        '"add_base_mass"',
        '"base_com"',
        '"actuator_gains"',
        '"motor_strength"',
        '"joint_dynamics"',
    ):
        assert name in block


def test_model49999_motion_hall_expansion_is_explicit() -> None:
    from unitree_rl_lab.utils.partial_checkpoint import _known_observation_mapping

    actor = _known_observation_mapping(480, 1864)
    high_speed_actor = _known_observation_mapping(480, 482)
    critic = _known_observation_mapping(495, 570)
    assert actor == list(range(480))
    assert high_speed_actor == list(range(480))
    assert critic == list(range(495))
    assert _known_observation_mapping(481, 1864) is None


def test_482_observation_removes_hall_but_keeps_motion_feedback() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    )
    block = _class_block(path, "FootTractionHighSpeedBackbone482ObservationsCfg")
    assert "foot_magnetic_array = None" in block
    assert "foot_sample_period_lr = None" in block
    assert "foot_sensor_valid_lr = None" in block
    assert "high_speed_policy" in block
    assert "foot_sensor_age_lr = None" not in block


def test_482_runner_uses_only_high_speed_actor_group() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
    )
    block = _class_block(path, "FootTractionHighSpeedBackbone482PPORunnerCfg")
    assert 'obs_groups = {"actor": ["high_speed_policy"], "critic": ["critic"]}' in block
    assert "max_iterations = 400" in block


def test_uniform_high_mu_buffer_uses_material_scale_without_actor_data() -> None:
    path = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/spatial_friction.py"
    )
    source = path.read_text()
    block = source.split("def update_uniform_high_friction_buffer", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert 'getattr(env, "friction_material_scale_buf", None)' in block
    assert "effective = selected * float(ground_patch_mu)" in block
    assert "ground_friction_mu_buf" in block
    assert "observation" not in block


def test_registry_wires_isolated_task() -> None:
    registry = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py"
    ).read_text()
    assert registry.count('"UniformHighFrictionLongBackbone"\n    ),') == 1
    assert (
        "RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg"
        in registry
    )
    assert "FootTractionHallUniformHighFrictionLongBackbonePPORunnerCfg" in registry
    assert "UniformHighFrictionLongBackboneWarmup" in registry
    assert (
        "FootTractionHallUniformHighFrictionLongBackboneWarmupPPORunnerCfg"
        in registry
    )
    assert "UniformHighFrictionLongBackbone482" in registry
    assert "FootTractionHighSpeedBackbone482PPORunnerCfg" in registry
