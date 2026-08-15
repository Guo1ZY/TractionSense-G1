"""Gym registrations isolated from the legacy G1 task registry."""

import gymnasium as gym


_ENV_MODULE = (
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof."
    "velocity_canonical_traction_env_cfg"
)
_AGENT_MODULE = "unitree_rl_lab.tasks.locomotion.agents.traction_rsl_cfg"

gym.register(
    id="Unitree-G1-29dof-Velocity-TractionCanonicalTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{_ENV_MODULE}:CanonicalTractionTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{_ENV_MODULE}:CanonicalTractionTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{_AGENT_MODULE}:CanonicalTractionTeacherPPORunnerCfg"
        ),
    },
)

for _task_suffix, _env_name, _play_name in (
    (
        "",
        "CanonicalTractionStudentEnvCfg",
        "CanonicalTractionStudentPlayEnvCfg",
    ),
    (
        "-Ideal",
        "CanonicalTractionStudentIdealEnvCfg",
        "CanonicalTractionStudentIdealPlayEnvCfg",
    ),
    (
        "-Proprio",
        "CanonicalTractionStudentProprioEnvCfg",
        "CanonicalTractionStudentProprioPlayEnvCfg",
    ),
):
    gym.register(
        id=f"Unitree-G1-29dof-Velocity-TractionCanonicalStudent{_task_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_ENV_MODULE}:{_env_name}",
            "play_env_cfg_entry_point": f"{_ENV_MODULE}:{_play_name}",
            "rsl_rl_cfg_entry_point": (
                f"{_AGENT_MODULE}:CanonicalTractionStudentPPORunnerCfg"
            ),
            "rsl_rl_distillation_cfg_entry_point": (
                f"{_AGENT_MODULE}:CanonicalTractionDistillationRunnerCfg"
            ),
        },
    )
