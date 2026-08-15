"""Fast invariants for the safe Hall-only friction-switch PPO task."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
RUNNER_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents"
    / "rsl_rl_ppo_cfg.py"
)
REGISTRY = ENV_CFG.parent / "__init__.py"


def _block(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_switch_ppo_actor_stays_hall_only_and_trains_asynchronously() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    event = _block(
        source,
        "class TractionMagneticSwitchTrainingEventCfg",
        "class FootTractionMagneticObservationsCfg",
    )
    assert "is_global_time=False" in event
    assert "interval_range_s=(2.5, 5.5)" in event
    assert "low_friction_range\": (0.08, 0.20)" in event

    task = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchStudentEnvCfg",
        "class RobotFootTractionMagneticMotionSwitchTrainEnvCfg",
    )
    assert "rewards: RewardsCfg = FootTractionMotionSwitchRewardsCfg()" in task
    assert "termination_penalty.weight = -800.0" in task
    assert "foot_contact_force" not in task
    assert "ground_friction_mu" not in task

    warmup = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchWarmupEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert 'event.params["low_friction_range"] = (0.22, 0.35)' in warmup
    assert "interval_range_s = (3.0, 5.5)" in warmup

    bridge = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchBridgeEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert 'event.params["low_friction_range"] = (0.14, 0.26)' in bridge
    assert "interval_range_s = (2.8, 5.0)" in bridge

    hardening = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert "foot_dropout_probability = 0.10" in hardening
    assert "dead_channel_probability = 0.08" in hardening
    assert "maximum_packet_delay_steps = 5" in hardening
    assert "termination_penalty.weight = -1400.0" in hardening
    assert "foot_contact_force" not in hardening
    assert "ground_friction_mu" not in hardening

    envelope = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchCommandEnvelopeEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert "commands: CommandsCfg = HallSafetyEnvelopeCommandsCfg()" in envelope
    assert "resampling_time_range = (2.5, 5.0)" in envelope
    assert "stop_fraction = 0.18" in envelope
    assert "crawl_speed_range = (0.20, 0.35)" in envelope
    assert "foot_contact_force" not in envelope
    assert "ground_friction_mu" not in envelope

    command = _block(
        source,
        "class HallSafetyEnvelopeCommandsCfg",
        "class TractionTeacherEventCfg",
    )
    assert "HallSafetyEnvelopeVelocityCommandCfg" in command
    assert "cruise_speed_range=(0.45, 0.90)" in command

    recovery = _block(
        source,
        "class RobotFootTractionMagneticMotionSwitchZeroFallRecoveryEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert "commands: CommandsCfg = HallZeroFallEnvelopeCommandsCfg()" in recovery
    assert "termination_penalty.weight = -3000.0" in recovery
    assert "cruise_speed_range = (0.70, 0.85)" in recovery
    assert "interval_range_s = (1.75, 3.50)" in recovery
    assert "foot_contact_force" not in recovery
    assert "ground_friction_mu" not in recovery

    zero_fall_command = _block(
        source,
        "class HallZeroFallEnvelopeCommandsCfg",
        "class TractionTeacherEventCfg",
    )
    assert "HallSafetyEnvelopeVelocityCommandCfg" in zero_fall_command
    assert "cruise_speed_range=(0.70, 0.85)" in zero_fall_command


def test_switch_runner_is_a_long_conservative_hall_policy_run() -> None:
    source = RUNNER_CFG.read_text(encoding="utf-8")
    block = _block(
        source,
        "class FootTractionHallSwitchStudentPPORunnerCfg",
        "class FootTractionSlopeStairsTeacherPPORunnerCfg",
    )
    assert "max_iterations = 5000" in block
    assert "clip_param=0.08" in block
    assert "learning_rate=1.5e-5" in block
    assert "max_grad_norm=0.12" in block


def test_switch_train_registry_uses_the_hall_runner() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    block = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchTrain"',
        "# Frozen/noisy Teacher rollouts",
    )
    assert "RobotFootTractionMagneticMotionSwitchTrainEnvCfg" in block
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in block

    deterministic = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-Switch"',
        "# Frozen/noisy Teacher rollouts",
    )
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in deterministic

    warmup = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchWarmup"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionSwitchWarmupEnvCfg" in warmup

    bridge = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchBridge"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionSwitchBridgeEnvCfg" in bridge

    hardening = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchFaultHardening"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg" in hardening
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in hardening

    envelope = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchCommandEnvelope"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionSwitchCommandEnvelopeEnvCfg" in envelope
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in envelope

    recovery = _block(
        source,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchZeroFallRecovery"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionSwitchZeroFallRecoveryEnvCfg" in recovery
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in recovery


def test_fastbase_switch_collection_task_uses_flat_switch_physics_and_exact_actor() -> None:
    """Matched-policy risk data must not inherit the fixed spatial course clock."""

    source = REGISTRY.read_text(encoding="utf-8")
    block = _block(
        source,
        '"SwitchEvalFastBaseCaptureCalibrated"',
        'id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral"',
    )
    assert "RobotFootTractionMagneticMotionSwitchStudentEnvCfg" in block
    assert "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg" in block
    assert "FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg" in block
    assert "SpatialFrictionMedium" not in block


def test_low_grip_recovery_is_separate_and_keeps_the_hall_only_contract() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    event = _block(
        source,
        "class TractionMagneticLowGripRecoveryEventCfg",
        "class RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg",
    )
    assert 'event.params["low_friction_range"] = (0.14, 0.20)' in event
    assert 'event.params["initial_high_probability"] = 0.0' in event

    recovery = _block(
        source,
        "class RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg",
        "class RobotFootTractionMagneticMotionStudentPlayEnvCfg",
    )
    assert "events: EventCfg = TractionMagneticLowGripRecoveryEventCfg()" in recovery
    assert "self.events.friction_switch = None" in recovery
    assert "cruise_speed_range = (0.70, 0.85)" in recovery
    assert "termination_penalty.weight = -4000.0" in recovery
    assert "foot_contact_force" not in recovery
    assert "ground_friction_mu" not in recovery

    registry = REGISTRY.read_text(encoding="utf-8")
    block = _block(
        registry,
        'id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-LowGripRecovery"',
        "gym.register(",
    )
    assert "RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg" in block
    assert "FootTractionHallSwitchStudentPPORunnerCfg" in block
