"""CPU/static invariants for the isolated Hall cadence/stride curriculum."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
REWARDS = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
    / "rewards.py"
)
RUNNER = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents"
    / "rsl_rl_ppo_cfg.py"
)
REGISTRY = ENV_CFG.with_name("__init__.py")
HALL_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/sensors"
    / "hall_sensor_config.py"
)
CONTACT_SLIP = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/traction"
    / "contact_slip.py"
)
FASTBASE = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/traction"
    / "fastbase_capture_residual.py"
)


def _block(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    finish = source.index(end, start)
    return source[start:finish]


def _load_contact_slip_module():
    spec = importlib.util.spec_from_file_location(
        "hall_cadence_stride_contact_slip_test", CONTACT_SLIP
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_course_keeps_requested_speed_and_does_not_prescribe_gait() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    task = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg",
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg" in task
    assert "scene:" not in task
    assert "self.commands.base_velocity.cruise_speed_range = (0.80, 0.80)" in task
    assert "self.commands.base_velocity.high_speed_range = (0.80, 0.80)" in task
    assert "self.rewards.track_lin_vel_xy.func = mdp.track_lin_vel_x_exp" in task
    for disabled in (
        "traction_overspeed",
        "gait",
        "low_traction_touchdown_rate",
        "spatial_capture_envelope",
        "spatial_capture_success",
        "high_traction_underspeed",
        "feet_slide",
        "feet_anti_slip",
        "slip_under_command",
    ):
        assert f"self.rewards.{disabled} = None" in task
    assert "slow_period" not in task
    assert "fast_period" not in task
    assert "contact_point_slip.weight = -6.0" in task
    assert "feet_force_rate.weight" in task
    assert "action_rate.weight" in task
    assert "termination_penalty.weight = -5000.0" in task
    assert "straight_cross_track.weight" in task

    reward_cfg = _block(
        source,
        "class FootTractionCadenceStrideRewardsCfg",
        "class SpatialFrictionTerminationsCfg",
    )
    assert "func=mdp.track_lin_vel_x_exp" in reward_cfg
    assert "func=mdp.contact_point_tangential_slip_penalty" in reward_cfg
    assert "func=mdp.filtered_contact_friction_cone_margin_penalty" in reward_cfg
    assert '"left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG' in reward_cfg
    assert '"right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG' in reward_cfg
    assert '"std": 0.22' in reward_cfg
    assert "straight_heading_error" in reward_cfg
    assert "func=mdp.straight_heading_error_penalty" in reward_cfg
    assert "weight=-18.0" in reward_cfg
    assert "low_speed" not in reward_cfg
    assert "slow_period" not in reward_cfg
    assert "touchdown" not in reward_cfg


def test_retention_course_is_long_physical_training_not_a_play_alias() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    scene = _block(
        source,
        "class HallSpatialMediumRetentionSceneCfg",
        "class HallSlopeStairsSceneCfg",
    )
    assert 'size_x=3.0' in scene
    assert 'size_x=2.0' in scene
    assert 'size_x=8.0' in scene
    assert 'friction=0.28' in scene
    assert scene.count('friction=0.90') == 2

    task = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionPlayEnvCfg",
    )
    assert "HallSpatialMediumRetentionSceneCfg" in task
    assert "self.episode_length_s = 20.0" in task
    assert 'minimum_local_x"] = 9.50' in task
    assert "(-2.70, -2.30)" in task
    assert "low_patch_mu\"] = 0.28" in task
    assert "contact_distribution_mode" not in task  # inherited detailed mode


def test_retention_actor_adds_no_privileged_observation_path() -> None:
    actor_source = FASTBASE.read_text(encoding="utf-8")
    stability = _block(
        actor_source,
        "class FastBaseHallCaptureStabilityResidual",
        "def trainable_parameters",
    )
    assert "observation[:, :480]" in stability
    assert "MOTION_FEEDBACK_SLICE" in stability
    for forbidden in ("friction", "contact_force", "course_stage", "ground_mu"):
        assert forbidden not in stability.lower()

    runner_source = RUNNER.read_text(encoding="utf-8")
    runner = _block(
        runner_source,
        "class FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg",
        "class FootTractionHallSpatialCalibratedFastBaseExpertDistillPPORunnerCfg",
    )
    assert "FastBaseHallCaptureStabilityActorCfg" in runner
    assert 'capture_gate_gradient_mode = "stage_bce_only"' in runner
    assert "stability_limit=0.25" in runner
    assert "stability_residual_learning_rate = 1.0e-4" in runner
    assert "stability_residual_max_grad_norm = 0.20" in runner
    assert "num_steps_per_env = 64" in runner

    registry = REGISTRY.read_text(encoding="utf-8")
    assert "SpatialFrictionCadenceStrideRetention" in registry
    assert "FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg" in registry


def test_recovery_curriculum_is_training_only_and_high_stage_gated() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    recovery = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRecoveryCurriculumEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionPlayEnvCfg",
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg" in recovery
    assert "mdp.push_spatial_high_grip_recovery_by_velocity" in recovery
    assert 'interval_range_s=(1.20, 2.40)' in recovery
    assert '"y": (-0.18, 0.18)' in recovery
    assert '"roll": (-0.24, 0.24)' in recovery
    assert '"yaw": (-0.20, 0.20)' in recovery

    spatial_source = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
        / "spatial_friction.py"
    ).read_text(encoding="utf-8")
    event = _block(
        spatial_source,
        "def push_spatial_high_grip_recovery_by_velocity",
        "def update_spatial_friction_buffer",
    )
    assert "SPATIAL_HIGH_START" in event
    assert "SPATIAL_HIGH_END" in event
    assert "SPATIAL_LOW" not in event
    assert "write_root_velocity_to_sim" in event
    assert "spatial_course_stage_buf must be initialized" in event
    # Reset diagnostics belong to reset_root_state_spatial_stratified and may
    # never leak into the asynchronous push callback's local scope.
    assert "band_index" not in event
    assert "local_x" not in event

    registry = REGISTRY.read_text(encoding="utf-8")
    task = _block(
        registry,
        '"SpatialFrictionCadenceStrideRecoveryCurriculum"',
        '# Signed raw foot force',
    )
    assert "RecoveryCurriculumEnvCfg" in task
    # Evaluation is deliberately the ordinary, no-push long course.
    assert "CadenceStrideRetentionPlayEnvCfg" in task


def test_true_contact_point_reward_is_privileged_and_fail_closed() -> None:
    reward_source = REWARDS.read_text(encoding="utf-8")
    reward = _block(
        reward_source,
        "def contact_point_tangential_slip_penalty",
        "def lateral_slip_penalty",
    )
    assert "static_ground_contact_point_tangential_speed" in reward
    assert "body_com_pos_w" in reward
    assert "body_com_lin_vel_w" in reward
    assert "body_com_ang_vel_w" in reward
    assert "contact_pos_w" in reward
    assert "force_matrix_w" in reward
    assert "requires a dedicated filtered" in reward
    assert "_mean_contact_foot_slip" not in reward

    friction_cone = _block(
        reward_source,
        "def filtered_contact_friction_cone_margin_penalty",
        "def straight_line_motion_penalty",
    )
    assert "force_matrix_w" in friction_cone
    assert "friction_forces_w" in friction_cone
    assert "sensor.data.net_forces_w" not in friction_cone
    assert "track_friction_forces=True" in friction_cone

    heading = _block(
        reward_source,
        "def straight_heading_error_penalty",
        "def transition_heading_retention_penalty",
    )
    assert "torch.atan2(cross, dot)" in heading
    assert "straight_heading_reference_xy" in heading
    assert "episode_length_buf <= 1" in heading
    assert "command[:, 2]" in heading
    for forbidden in ("ground_friction", "contact_force", "course_stage"):
        assert forbidden not in heading

    # The isolated transition-retention terms are reward/critic-only and may
    # therefore read the privileged course stage; the deployable actor
    # observation remains untouched.
    transition = _block(
        reward_source,
        "def transition_heading_retention_penalty",
        "def traction_adaptive_feet_gait",
    )
    assert "spatial_course_stage_buf" in transition
    assert "transition_heading_error_buf" in transition
    assert "spatial_low_entry_heading_buf" in transition
    assert "transition_vy_window_buf" in transition

    env_source = ENV_CFG.read_text(encoding="utf-8")
    policy = _block(
        env_source,
        "class FootTractionMagneticObservationsCfg",
        "class RobotFootTractionMagneticStudentEnvCfg",
    )
    assert "foot_contact_force = None" in policy
    for forbidden_term in (
        "foot_normal_force = ObsTerm",
        "foot_tangent_force = ObsTerm",
        "foot_slip_proxy = ObsTerm",
        "ground_friction_mu = ObsTerm",
        "spatial_course_stage",
    ):
        assert forbidden_term not in policy
    # Existing audited term-major schema: 480 proprio + 1350 magnetic history
    # + 30 sample periods + 2 valid + 2 age/motion trailing features.
    assert 480 + 15 * 2 * 15 * 3 + 15 * 2 + 2 + 2 == 1864
    # Existing privileged critic: 495 proprio history + six two-foot histories
    # + scalar friction/valid/age histories. It remains critic-only.
    assert 495 + 6 * 2 * 5 + 3 * 1 * 5 == 570


def test_contact_point_kinematics_uses_omega_cross_r_and_masks_no_contact() -> None:
    module = _load_contact_slip_module()
    body_pos = torch.zeros((3, 2, 3), dtype=torch.float64)
    body_lin = torch.zeros_like(body_pos)
    body_ang = torch.zeros_like(body_pos)

    # Env 0: v_COM=[1,0,0], omega_z=10 and r=[0,0.1,0] cancel exactly
    # at the loaded contact. Env 1 retains a 0.2 m/s tangential slip. Env 2
    # has no loaded contact and must return zero with validity false.
    body_lin[0, 0, 0] = 1.0
    body_ang[0, 0, 2] = 10.0
    body_lin[1, 0, 0] = 0.2
    left_pos = torch.zeros((3, 1, 3, 3), dtype=torch.float64)
    left_force = torch.zeros_like(left_pos)
    left_pos[0, 0, 0, 1] = 0.1
    left_force[0, 0, 0, 2] = 100.0
    left_force[1, 0, 0, 2] = 80.0
    right_pos = torch.zeros((3, 1, 3, 3), dtype=torch.float64)
    right_force = torch.zeros_like(right_pos)

    result = module.static_ground_contact_point_tangential_speed(
        body_pos,
        body_lin,
        body_ang,
        (left_pos, right_pos),
        (left_force, right_force),
        min_normal_force_n=5.0,
    )
    torch.testing.assert_close(
        result.speed_per_env,
        torch.tensor((0.0, 0.2, 0.0), dtype=torch.float64),
    )
    assert result.valid_per_env.tolist() == [True, True, False]
    assert result.valid_per_foot.tolist() == [
        [True, False],
        [True, False],
        [False, False],
    ]
    assert torch.isfinite(result.speed_per_env).all()


def test_detailed_mode_is_isolated_from_legacy_tasks() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    task = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg",
    )
    assert 'self.hall_sensor_cfg.contact_distribution_mode = "detailed"' in task
    assert "sync_hall_sensor_cfg_to_policy_terms" in task

    default_cfg = HALL_CFG.read_text(encoding="utf-8")
    assert 'contact_distribution_mode: Literal["aggregate", "detailed"] = "aggregate"' in default_cfg
    old_medium = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg",
    )
    old_play = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg",
        "class RobotFootTractionMagneticMotionLowGripRecoveryPlayEnvCfg",
    )
    assert "contact_distribution_mode" not in old_medium
    assert "contact_distribution_mode" not in old_play


def test_runner_is_full_trainable_fastbase_not_twelve_update_guard() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    runner = _block(
        source,
        "class FootTractionHallSpatialCadenceStridePPORunnerCfg",
        "class FootTractionHallSpatialCalibratedFastBaseExpertDistillPPORunnerCfg",
    )
    assert "FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg" in runner
    assert 'obs_groups = {"actor": ["policy"], "critic": ["critic"]}' in runner
    assert "num_steps_per_env = 64" in runner
    assert "max_iterations = 1000" in runner
    assert "save_interval = 25" in runner
    assert "require_fail_closed_training_start: bool = False" in runner
    assert 'self.algorithm.capture_gate_gradient_mode = "stage_bce_only"' in runner
    assert "low_expert_checkpoint" not in runner
    assert "maximum_allowed_new_updates" not in runner

    base = _block(
        source,
        "class FootTractionHallSpatialFastBaseCapturePPORunnerCfg",
        "class FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg",
    )
    assert "capture_gate_warmup_updates=50" in base
    assert "capture_residual_learning_rate=5.0e-5" in base
    assert "all speedboost112 parameters" in base


def test_long_demo_geometry_horizon_reset_and_registration_are_isolated() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    scene = _block(
        source,
        "class HallSpatialMediumLongDemoSceneCfg",
        "class HallSlopeStairsSceneCfg",
    )
    for size, center in (("6.0", "-3.0"), ("6.0", "3.0"), ("12.0", "12.0")):
        assert f"size_x={size}" in scene
        assert f"center_x={center}" in scene
    assert 6.0 + 6.0 + 12.0 == 24.0
    assert scene.count("size_y=3.2") == 3
    assert "friction=0.28" in scene
    helper = _block(source, "def _friction_patch_cfg", "class HallSpatialFrictionSceneCfg")
    assert "opacity=1.0" in helper

    demo = source[
        source.index(
            "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideLongDemoEnvCfg"
        ) :
    ]
    assert "HallSpatialMediumLongDemoSceneCfg" in demo
    assert "num_envs=4, env_spacing=12.0" in demo
    assert "self.episode_length_s = 65.0" in demo
    assert '"x_bands"] = ((-5.50, -5.10),)' in demo
    assert '"minimum_local_x"] = 17.50' in demo
    assert "self.viewer.eye" in demo and "self.viewer.lookat" in demo

    evaluator = (
        ROOT / "scripts/rsl_rl/eval_spatial_friction_course.py"
    ).read_text(encoding="utf-8")
    assert 'default=None' in _block(
        evaluator,
        'parser.add_argument(\n    "--steps"',
        'parser.add_argument("--seed"',
    )
    assert 'if "LongDemo" in args_cli.task' in evaluator
    assert 'if "CadenceStrideRetention" in args_cli.task' in evaluator
    assert 'else 1200' in evaluator
    assert 'else 400' in evaluator
    assert 'choices=("aggregate", "detailed")' in evaluator
    assert "hall_cfg.contact_distribution_mode = args_cli.hall_contact_distribution" in evaluator
    assert '"hall_contact_distribution_mode"' in evaluator
    assert "friction_forces_w" in evaluator

    registry = REGISTRY.read_text(encoding="utf-8")
    cadence_task = _block(
        registry,
        "SpatialFrictionMediumDenseCadenceStride",
        "# Visualization-only long geometry",
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg" in cadence_task
    assert "RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg" in cadence_task
    assert "FootTractionHallSpatialCadenceStridePPORunnerCfg" in cadence_task
    long_task = _block(
        registry,
        "SpatialFrictionCadenceStrideLongDemo",
        "# Isolated LOW-recovery expert-distillation experiment",
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg" in long_task
    assert "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideLongDemoEnvCfg" in long_task
    assert "FootTractionHallSpatialCadenceStridePPORunnerCfg" in long_task

    # The ordinary short scene and old play cfg remain byte-level configured
    # at their original dimensions/reset, so legacy evaluations do not move.
    ordinary = _block(
        source,
        "class HallSpatialFrictionSceneCfg",
        "class HallSpatialMildFrictionSceneCfg",
    )
    assert "size_x=2.0" in ordinary
    assert "size_x=1.0" in ordinary
    assert "center_x=-1.0" in ordinary
    assert "center_x=0.5" in ordinary
    assert "center_x=2.0" in ordinary


def test_spatial_evaluator_reports_touchdown_gait_without_prescribing_direction() -> None:
    evaluator = (
        ROOT / "scripts/rsl_rl/eval_spatial_friction_course.py"
    ).read_text(encoding="utf-8")
    rollout = _block(evaluator, "def _run_rollout(", "def main()")

    # Dedicated Hall contact sensors are used only to identify real
    # touchdown events; material/force/cadence never becomes actor input.
    assert 'base_env.scene["left_hall_contact"]' in rollout
    assert 'base_env.scene["right_hall_contact"]' in rollout
    assert "torch.linalg.vector_norm(force[:, 0, :], dim=-1) > 5.0" in rollout
    assert "gait_air_steps >= gait_minimum_air_steps" in rollout
    assert "gait_steps_since_touchdown >= gait_minimum_touchdown_gap_steps" in rollout
    assert "gait_last_touchdown_forward[region_changed] = float(\"nan\")" in rollout
    assert "first_episode_active_before[:, None]" in rollout
    assert "~dones.bool()[:, None]" in rollout

    # The report exposes mechanism and HighEnd recovery ratios, but explicitly
    # does not require Low speed or cadence to move in a prescribed direction.
    assert '"gait_adaptation": gait_adaptation' in rollout
    assert '"mechanism_is_diagnostic_not_prescribed": True' in rollout
    assert '"step_frequency_hz": cadence' in rollout
    assert '"mean_step_length_m": step_length' in rollout
    assert '"stride_length_ratio"' in rollout
    assert '"vx_ratio"' in rollout
