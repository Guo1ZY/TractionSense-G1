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
            "FootTractionLateralGuardTeacherPPORunnerCfg"
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
