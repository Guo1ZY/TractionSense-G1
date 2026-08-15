"""Fast source-level invariants for Stage7 Hall recovery and spatial friction."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
EVENTS = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
    / "spatial_friction.py"
)
RUNNER = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents"
    / "rsl_rl_ppo_cfg.py"
)
REGISTRY = ENV_CFG.parent / "__init__.py"
TRAIN = ROOT / "scripts" / "rsl_rl" / "train.py"


def _block(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_handoff_curriculum_keeps_hall_actor_contract() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    event = _block(
        source,
        "class TractionMagneticLowGripHandoffEventCfg",
        "class TractionMagneticSpatialFrictionEventCfg",
    )
    assert "func=mdp.push_by_setting_velocity" in event
    assert 'is_global_time=False' in event
    assert 'interval_range_s=(1.5, 3.0)' in event
    assert '"x": (0.35, 0.65)' in event

    task = _block(
        source,
        "class RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg",
        "class RobotFootTractionMagneticMotionLowGripHandoffMildEnvCfg",
    )
    assert "self.events.friction_switch = None" in task
    assert '"x": (0.25, 0.55)' in task
    assert "self.rewards.gait.weight = 1.25" in task
    assert "self.rewards.low_traction_touchdown_rate.weight = -3.0" in task
    assert "self.rewards.termination_penalty.weight = -5000.0" in task
    assert "foot_contact_force" not in task
    assert "ground_friction_mu" not in task

    mild = _block(
        source,
        "class RobotFootTractionMagneticMotionLowGripHandoffMildEnvCfg",
        "class RobotFootTractionMagneticMotionLowGripHandoffExtremeEnvCfg",
    )
    assert 'event.params["low_friction_range"] = (0.18, 0.26)' in mild
    assert '"x": (0.20, 0.40)' in mild

    extreme = _block(
        source,
        "class RobotFootTractionMagneticMotionLowGripHandoffExtremeEnvCfg",
        "class RobotFootTractionMagneticMotionLowGripHandoffRecoveryPlayEnvCfg",
    )
    assert 'event.params["low_friction_range"] = (0.12, 0.20)' in extreme
    assert '"x": (0.45, 0.75)' in extreme


def test_spatial_course_uses_real_colored_colliders_and_no_plane() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    scene = _block(
        source,
        "class HallSpatialFrictionSceneCfg",
        "class HallSlopeStairsSceneCfg",
    )
    for name in ("FrictionHighStart", "FrictionLow", "FrictionHighEnd"):
        assert name in scene
        assert f'{{ENV_REGEX_NS}}/{name}/geometry/mesh' in source
    assert "friction=0.16" in scene
    assert "opacity=1.0" in source

    task = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg",
    )
    assert "commands: CommandsCfg = HallZeroFallEnvelopeCommandsCfg()" in task
    assert "self.scene.terrain = None" in task
    assert "self.scene.height_scanner = None" in task
    assert "self.events.push_robot = None" in task
    assert "self.commands.base_velocity.stop_fraction = 0.0" in task
    assert "self.commands.base_velocity.crawl_fraction = 0.0" in task
    assert "foot_contact_force" not in task


def test_spatial_labels_are_explicitly_privileged() -> None:
    source = EVENTS.read_text(encoding="utf-8")
    assert "never create actor observations" in source
    assert "def randomize_coherent_material_scale" in source
    assert "def update_spatial_friction_buffer" in source
    assert "force_matrix_w" in source
    assert "foot_low_contact = patch_contact[:, :, 1]" in source
    assert "advance_spatial_course_stage" in source
    assert "spatial_friction_course_success" in source
    assert '"spatial_foot_friction_mu_buf"' in source
    assert '"ground_friction_mu_buf"' in source
    for buffer_name in (
        "spatial_low_entry_step_buf",
        "spatial_low_elapsed_s_buf",
        "spatial_low_stable_count_buf",
        "spatial_low_capture_success_buf",
    ):
        assert buffer_name in source

    env_source = ENV_CFG.read_text(encoding="utf-8")
    policy = _block(
        env_source,
        "class FootTractionMagneticObservationsCfg",
        "class RobotFootTractionMagneticStudentEnvCfg",
    )
    assert "func=mdp.hall_magnetic_array" in policy
    assert "foot_contact_force = None" in policy
    assert "ground_friction_mu" not in policy
    assert "spatial_low_" not in policy


def test_spatial_capture_reward_is_privileged_and_never_short_circuits_course() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    reward_cfg = _block(
        source,
        "class FootTractionSpatialCaptureRewardsCfg",
        "class SpatialFrictionTerminationsCfg",
    )
    assert "spatial_low_capture_envelope_penalty" in reward_cfg
    assert "spatial_low_capture_reward" in reward_cfg
    assert '"deadline_s": 0.90' in reward_cfg
    assert '"timely_completion_bonus": 12.0' in reward_cfg

    terminations = _block(
        source,
        "class SpatialFrictionTerminationsCfg",
        "class RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg",
    )
    assert "spatial_friction_course_success" in terminations
    assert "capture" not in terminations
    assert 'time_out=True' in terminations

    final = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
    )
    assert "FootTractionSpatialCaptureRewardsCfg" in final
    assert "target_speed=0.24" in final
    assert "deadline_s=0.90" in final
    assert "traverse FrictionHighEnd" in final

    mild = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
    )
    medium = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg",
    )
    assert "target_speed=0.45" in mild and "deadline_s=1.00" in mild
    assert "target_speed=0.32" in medium and "deadline_s=0.95" in medium


def test_stage7_runners_and_tasks_are_registered() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    handoff = _block(
        runner,
        "class FootTractionHallHandoffRecoveryPPORunnerCfg",
        "class FootTractionHallSpatialTransitionPPORunnerCfg",
    )
    assert "max_iterations = 1500" in handoff
    assert "clip_param=0.05" in handoff
    assert "learning_rate=5.0e-6" in handoff
    assert "desired_kl=0.001" in handoff

    registry = REGISTRY.read_text(encoding="utf-8")
    for suffix in (
        "LowGripHandoffMild",
        "LowGripHandoffRecovery",
        "LowGripHandoffExtreme",
    ):
        assert f'"{suffix}"' in registry
    assert "TractionMagneticMotionStudent-SpatialFriction" in registry
    assert "FootTractionHallHandoffRecoveryPPORunnerCfg" in registry
    assert "FootTractionHallSpatialTransitionPPORunnerCfg" in registry


def test_spatial_speed_curriculum_is_physical_and_hall_only() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    for scene, friction in (
        ("HallSpatialMildFrictionSceneCfg", "friction=0.45"),
        ("HallSpatialMediumFrictionSceneCfg", "friction=0.28"),
    ):
        block = _block(source, f"class {scene}", "@configclass")
        assert '"FrictionLow"' in block
        assert friction in block

    mild = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
    )
    medium = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg",
    )
    assert 'event.params["low_patch_mu"] = 0.45' in mild
    assert 'event.params["low_patch_mu"] = 0.28' in medium
    for block in (mild, medium):
        assert "termination_penalty.weight = -5000.0" in block
        assert "foot_contact_force" not in block
        assert "ground_friction_mu" not in block

    registry = REGISTRY.read_text(encoding="utf-8")
    assert '"SpatialFrictionMild"' in registry
    assert '"SpatialFrictionMedium"' in registry
    assert "FootTractionHallSpatialRetentionPPORunnerCfg" in registry
    assert "FootTractionHallSpatialCapturePPORunnerCfg" in registry

    runner = RUNNER.read_text(encoding="utf-8")
    retention = _block(
        runner,
        "class FootTractionHallSpatialRetentionPPORunnerCfg",
        "class FootTractionHallSpatialCapturePPORunnerCfg",
    )
    capture = _block(
        runner,
        "class FootTractionHallSpatialCapturePPORunnerCfg",
        "class FootTractionSlopeStairsTeacherPPORunnerCfg",
    )
    assert "num_steps_per_env = 48" in retention
    assert "learning_rate=2.0e-6" in retention
    assert "num_steps_per_env = 64" in capture
    assert "learning_rate=5.0e-6" in capture

    train = TRAIN.read_text(encoding="utf-8")
    assert 'sequential_spatial_course = "SpatialFriction" in args_cli.task' in train
    assert "init_at_random_ep_len=not sequential_spatial_course" in train
