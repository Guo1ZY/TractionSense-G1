import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Training-only recoverable-disturbance curriculum.  Play deliberately maps
# back to the disturbance-free Retention config so every checkpoint is judged
# on the same physical H--L--H course used by the rejected r1/r2/r3 runs.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideRecoveryCurriculum"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRecoveryCurriculumEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg"
        ),
    },
)

# Isolated stable high-speed controller.  Training consumes only the audited
# 482-D group (legacy proprio history + body_vy/relative_heading), while the
# environment simultaneously retains policy[1864] for later Hall composition.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "UniformHighFrictionLongBackbone482"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackbone482EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackbone482PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHighSpeedBackbone482PPORunnerCfg"
        ),
    },
)

# Signed raw foot force appended to the unchanged 29-DoF baseline interface.
gym.register(
    id="Unitree-G1-29dof-Velocity-RawFoot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_raw_foot_env_cfg:RobotRawFootEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_raw_foot_env_cfg:RobotRawFootPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Same class with its raw-force switch disabled: exact 480/495 A/B control.
gym.register(
    id="Unitree-G1-29dof-Velocity-RawFoot-Off",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_raw_foot_env_cfg:RobotRawFootBaselineCompatEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_raw_foot_env_cfg:RobotRawFootBaselineCompatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Foot-sensor + friction-adaptive fine-tune task (resume from model_49999).
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootPPORunnerCfg"
        ),
    },
)

# Explicit degrade path (all foot extras off) — same physics as baseline.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Off",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootBaselineCompatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootBaselineCompatEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Yaw-only enlarge (vx/vy same as 4000; wz ±0.6). Resume from model_4000.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Turn",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootTurnEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootTurnPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootTurnPPORunnerCfg"
        ),
    },
)

# Combined: faster walk + in-place turn + friction adaptive (high μ fast / low μ slow-stable).
# Warm-start model_foot_4000 via --partial_checkpoint (critic obs grows with ρ/slip).
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Adaptive",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptiveEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptivePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootAdaptivePPORunnerCfg"
        ),
    },
)

# Same env as Adaptive but yaw-focused PPO (lower LR). Resume model_5400 after NaN.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Adaptive-Yaw",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptiveEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptivePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootAdaptiveYawPPORunnerCfg"
        ),
    },
)

# Fix idle stomping + low-μ slow-down. Resume adaptive_yaw model_6600.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Adaptive-Stable",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootStableAdaptiveEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootStableAdaptivePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootStableAdaptivePPORunnerCfg"
        ),
    },
)

# Clean multi-objective from model_49999 (partial). Prefer over stacking on 6600.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Full",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootFullEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootFullPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootFullPPORunnerCfg"
        ),
    },
)

# Adaptive-V2: outcome rewards + deployable actor obs (no overwrite of Foot-Full).
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-Adaptive-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptiveV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootAdaptiveV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootAdaptiveV2PPORunnerCfg"
        ),
    },
)

# MuAdapt: same 510 actor as Foot-Full (Fn+Ft) + outcome rewards (no slip-aware floor).
# Recommended for MuJoCo/zorn which both expose normal + tangent forces.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-MuAdapt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootMuAdaptEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootMuAdaptPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootMuAdaptPPORunnerCfg"
        ),
    },
)

# Straight-Mu: clean from 49999 — high-μ fast straight / low-μ slow-stable; NO turn.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-StraightMu",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootStraightMuEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootStraightMuPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootStraightMuPPORunnerCfg"
        ),
    },
)

# Traction-Adaptive: normal commands <=1.0, rare 1.0--1.5 stress probes;
# privileged μ teaches high-grip fast / low-grip slow without exposing μ to actor.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionAdaptive",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootTractionAdaptiveEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionAdaptivePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionAdaptivePPORunnerCfg"
        ),
    },
)

# Flat-ground privileged-μ teacher.  This is an isolated training task and
# never replaces the deployable TractionAdaptive registration above.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootTractionTeacherEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionTeacherPPORunnerCfg"
        ),
    },
)

# Same 641-D Oracle interface as TractionTeacher, with action latency,
# actuator/body/contact randomization and structured sensor corruption.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionRobustTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionRobustTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustTeacherPPORunnerCfg"
        ),
    },
)

# Targeted continuation of the robust Teacher.  It keeps the exact same
# 641-D actor / 570-D critic interfaces and dynamics randomization, while
# adding the standard terminal-failure cost and balancing both stress tails.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Stability",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustStabilityTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustStabilityTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustStabilityTeacherPPORunnerCfg"
        ),
    },
)

# Low-LR continuation from the frozen stable Teacher.  This task increases the
# mu=0.70--0.95 high-command density without weakening the low/high safety tails.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Shoulder",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustShoulderTeacherPPORunnerCfg"
        ),
    },
)

# Targeted recovery from the first shoulder checkpoint.  High-friction abrupt
# command starts are oversampled without changing the 641-D Oracle interface.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Shoulder-Recovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderRecoveryTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderRecoveryTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustShoulderRecoveryTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-Shoulder-Guard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderGuardTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustShoulderGuardTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustShoulderGuardTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Robust-LowMu-Recovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustLowMuRecoveryTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionRobustLowMuRecoveryTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionRobustLowMuRecoveryTeacherPPORunnerCfg"
        ),
    },
)

# Stage-2 deployable student.  It shares the old 640-D deploy schema but uses
# balanced teacher commands and structured, temporally correlated sensor DR.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionStudent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_foot_env_cfg:RobotFootTractionStudentEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionStudentPPORunnerCfg"
        ),
    },
)

# Straight-path continuation of the safe capped Teacher. This remains a
# separate task/checkpoint and never overwrites the selected 7989 artifacts.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-LateralGuard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionLateralGuardTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionLateralGuardTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionLateralGuardTeacherPPORunnerCfg"
        ),
    },
)

# End-to-end Hall-only PPO uses asynchronous material transitions.  The
# original ``...Student-Switch`` ID remains the deterministic phase-evaluation
# task, so recorded high→low→high comparisons stay reproducible.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchTrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchTrainEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchWarmup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchWarmupEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchBridge",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchBridgeEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Safety continuation after the bridge curriculum.  The actor interface stays
# 1864-D Hall/proprioception; only the simulated Hall/BLE fault distribution
# is made intentionally harsher during PPO.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchFaultHardening",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Command-envelope continuation.  It keeps Hall/proprioception as the only
# actor input while explicitly practicing the stop/crawl commands that the
# deploy-time Hall-risk governor may request.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchCommandEnvelope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchCommandEnvelopeEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Measured-tail recovery after the first command-envelope campaign.  This task
# preserves the Hall-only actor contract and focuses sampling around the fixed
# 0.8 m/s high→low→high acceptance trajectory plus safe crawl/stop requests.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SwitchZeroFallRecovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchZeroFallRecoveryEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Separate low-grip recovery actor used only after the causal Hall/proprio
# future-slip guard enters LOW.  It keeps the exact 1864-D deployment schema;
# fast high-grip tracking remains in the independently audited baseline actor.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-LowGripRecovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionLowGripRecoveryPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Stage7A/B/C preserve the exact Hall-only actor schema while progressively
# expanding high-momentum takeover states on fixed low friction.
for task_suffix, env_cfg_name in (
    (
        "LowGripHandoffMild",
        "RobotFootTractionMagneticMotionLowGripHandoffMildEnvCfg",
    ),
    (
        "LowGripHandoffRecovery",
        "RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg",
    ),
    (
        "LowGripHandoffExtreme",
        "RobotFootTractionMagneticMotionLowGripHandoffExtremeEnvCfg",
    ),
    (
        "LowGripHandoffHighCommand",
        "RobotFootTractionMagneticMotionLowGripHandoffHighCommandEnvCfg",
    ),
):
    gym.register(
        id=(
            "Unitree-G1-29dof-Velocity-Foot-"
            f"TractionMagneticMotionStudent-{task_suffix}"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:{env_cfg_name}"
            ),
            "play_env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:"
                "RobotFootTractionMagneticMotionLowGripHandoffRecoveryPlayEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                "FootTractionHallHandoffRecoveryPPORunnerCfg"
            ),
        },
    )

# Causal physical high--low--high course.  The orange patch owns a real PhysX
# material; no global friction clock or material label is exposed to the actor.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFriction",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialTransitionPPORunnerCfg"
        ),
    },
)

# Staged causal courses used to retain the original high-speed gait before the
# final mu=0.16 transition.  They keep the identical 1864-D Hall actor schema.
for task_suffix, env_cfg_name, play_cfg_name, runner_cfg_name in (
    (
        "SpatialFrictionMild",
        "RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
        "RobotFootTractionMagneticMotionSpatialFrictionMildPlayEnvCfg",
        "FootTractionHallSpatialRetentionPPORunnerCfg",
    ),
    (
        "SpatialFrictionMedium",
        "RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
        "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg",
        "FootTractionHallSpatialCapturePPORunnerCfg",
    ),
):
    gym.register(
        id=(
            "Unitree-G1-29dof-Velocity-Foot-"
            f"TractionMagneticMotionStudent-{task_suffix}"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:{env_cfg_name}"
            ),
            "play_env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:{play_cfg_name}"
            ),
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                f"{runner_cfg_name}"
            ),
        },
    )

# Explicit frozen-Teacher anchored variants.  These reuse the same physical
# scenes and the same 1864-D deployable Actor schema; only the project-local
# PPO training algorithm changes.  Keeping separate task IDs prevents an
# accidental anchored/unanchored comparison from sharing a registry entry.
for task_suffix, env_cfg_name, play_cfg_name, runner_cfg_name in (
    (
        "SpatialFrictionMildAnchored",
        "RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
        "RobotFootTractionMagneticMotionSpatialFrictionMildPlayEnvCfg",
        "FootTractionHallSpatialAnchoredRetentionPPORunnerCfg",
    ),
    (
        "SpatialFrictionMediumAnchored",
        "RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg",
        "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg",
        "FootTractionHallSpatialAnchoredCapturePPORunnerCfg",
    ),
):
    gym.register(
        id=(
            "Unitree-G1-29dof-Velocity-Foot-"
            f"TractionMagneticMotionStudent-{task_suffix}"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:{env_cfg_name}"
            ),
            "play_env_cfg_entry_point": (
                f"{__name__}.velocity_foot_env_cfg:{play_cfg_name}"
            ),
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                f"{runner_cfg_name}"
            ),
        },
    )

# Structurally protected fast-base variant.  It shares the exact Medium
# physical course and 1864-D deployable observation with the ordinary Actor,
# but PPO can modify only a bounded Hall-conditioned capture correction.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-SpatialFrictionMediumFastBaseCapture"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialFastBaseCapturePPORunnerCfg"
        ),
    },
)

# Transition-dense training task for the protected fast-base Actor.  Only the
# training reset distribution differs: every reset remains on HighStart, but
# far/mid/near distances are mixed across environments.  The play entry point
# is intentionally the ordinary Medium cfg, preserving the original H-L-H
# geometry and far reset for fair evaluation.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-SpatialFrictionMediumDenseFastBaseCapture"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialFastBaseCapturePPORunnerCfg"
        ),
    },
)

# Explicit calibrated deployment/evaluation task.  It keeps the same dense
# training distribution, ordinary H-L-H play course, actor schema and learned
# checkpoint tensors as the raw FastBase task.  Only the config-owned monotone
# gate authority calibration differs, so legacy task behavior never changes
# silently.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionMediumDenseFastBaseCaptureCalibrated"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg"
        ),
    },
)

# Non-prescriptive cadence/stride experiment.  The physical training course is
# still the short transition-dense Medium scene, but its Hall driver is the
# isolated detailed-contact mode and its objective retains the same requested
# velocity across H--L--H.  No legacy task or checkpoint behavior is changed.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionMediumDenseCadenceStride"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStridePPORunnerCfg"
        ),
    },
)

# Visualization-only long geometry for the same actor/checkpoint.  Training
# through this ID still resolves to the short MediumDense env above; play uses
# a wide 24 m, 65 s H[-6,0]--L[0,6]--H[6,18] scene so every material segment
# reaches a visible steady state without altering ordinary play/eval geometry.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideLongDemo"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideLongDemoEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStridePPORunnerCfg"
        ),
    },
)

# Sustained HighEnd consolidation.  Unlike LongDemo, this ID uses the long
# physical course for training as well as play and selects the isolated Actor
# with a zero-initialized proprioceptive stability residual.  The policy ABI
# remains exactly 1864 Hall/proprio values and the floor material never enters
# the Actor observation.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideRetention"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetention"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR2"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR2EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionR2PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR3"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR3EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionR3PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR4a"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR4aEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionR4PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR4b"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR4bEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionR4PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR5"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionR5PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideHighEndRecoveryExpert"
    ),
    entry_point=(
        "unitree_rl_lab.tasks.locomotion.high_end_recovery_env:"
        "HighEndRecoveryRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideHighEndRecoveryExpertPPORunnerCfg"
        ),
    },
)

# Independent long-horizon high-grip backbone diagnostic.  All three physical
# floor meshes are mu=0.90; no material transition or privileged stage enters
# the 1864-D Hall/proprio actor.  The intended warm-start is model_49999 via
# the explicit 480->1864 partial-checkpoint mapping.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "UniformHighFrictionLongBackbone"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackbonePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallUniformHighFrictionLongBackbonePPORunnerCfg"
        ),
    },
)

# H0 nominal curriculum for the same 1864-D long-horizon backbone.  Play maps
# to the common nominal evaluator cfg, so H0 does not receive an easier gate.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "UniformHighFrictionLongBackboneWarmup"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneWarmupEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionUniformHighFrictionLongBackbonePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallUniformHighFrictionLongBackboneWarmupPPORunnerCfg"
        ),
    },
)

# Isolated LOW-recovery expert-distillation experiment.  Physics, deployable
# 1864-D actor schema and calibrated gate are unchanged; model6149 exists only
# inside AnchoredPPO to create cached LOW targets during training.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertDistill"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCalibratedFastBaseExpertDistillPPORunnerCfg"
        ),
    },
)

# Strong-direction A/B for the same LOW-only model6149 supervision.  This is a
# separate task/experiment so the completed conservative expert r1 remains an
# immutable comparison point.  Actor inputs, physics and play course are
# intentionally identical to the calibrated expert task above.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertStrongDirection"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCalibratedFastBaseExpertStrongDirectionPPORunnerCfg"
        ),
    },
)

# Fail-closed follow-up after the strong run exposed a HIGH_END gate leak.  It
# uses the same actor schema/physics/expert, but PPO is denied gradients into
# both private action branches: raw stage BCE owns the gate, while HIGH anchor
# plus LOW expert own the residual.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SpatialFrictionMediumDenseFastBaseCaptureCalibratedExpertGateBceOnly"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCalibratedFastBaseExpertGateBceOnlyPPORunnerCfg"
        ),
    },
)

# Evaluation/data-collection-only flat switch task for the calibrated FastBase
# actor.  The actor/checkpoint schema is identical to the spatial task, while
# the physical environment is the ordinary synchronized material-switch scene.
# This breaks the fixed H-L-H position/time correlation when collecting a
# Hall-risk dataset.  AnchoredPPO deliberately requires a private spatial stage
# in its training ``act`` path, so accidentally invoking PPO training with this
# task fails closed instead of silently using a fabricated label.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-"
        "SwitchEvalFastBaseCaptureCalibrated"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionSpeedLateralTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralV2TeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralV2TeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionSpeedLateralV2TeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral-Symmetry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralV2TeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSpeedLateralV2TeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionSpeedLateralSymmetryTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionMotionTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-Stress",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionStressTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionStressTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionMotionTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-StrongSymmetry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionMotionStrongSymmetryTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-BalancedSymmetry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionMotionBalancedSymmetryTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-MotionFeedback-Switch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionSwitchTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMotionSwitchTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionMotionSwitchTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticStudentEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionLateralGuardTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SlopeStairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSlopeStairsTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionSlopeStairsTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionSlopeStairsTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent-SlopeStairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticSlopeStairsEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticSlopeStairsPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionLateralGuardTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionSlopeStairsV1"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSlopeStairsEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSlopeStairsPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSpatialCadenceStrideTransitionRetentionSlopeStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent-Deformable",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticStudentDeformableEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticStudentDeformablePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionLateralGuardTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionStudentEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionLateralGuardTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-Switch",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:"
            "RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionHallSwitchStudentPPORunnerCfg"
        ),
    },
)

# Frozen/noisy Teacher rollouts provide labeled (sensor history, true mu)
# pairs for the deployable estimator.  This task is for collection/evaluation.
gym.register(
    id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-Noisy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionNoisyTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_foot_env_cfg:RobotFootTractionNoisyTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "FootTractionTeacherPPORunnerCfg"
        ),
    },
)

# Pure-480-D proprio actor ablation on the exact R5 course (2026-08-14
# proprio480 experiment).  New isolated files only; R5 classes are untouched.
gym.register(
    id=(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionProprio480MagneticMotionStudent-"
        "SpatialFrictionCadenceStrideTransitionRetentionR5"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_proprio480_env_cfg:"
            "RobotFootTractionProprio480MagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_proprio480_env_cfg:"
            "RobotFootTractionProprio480MagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_proprio480_cfg:"
            "FootTractionProprio480SpatialCadenceStrideTransitionRetentionR5PPORunnerCfg"
        ),
    },
)
