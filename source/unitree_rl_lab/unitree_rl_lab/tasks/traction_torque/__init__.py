"""Independent Gym registrations for motor-torque traction locomotion."""

import gymnasium as gym


_ENV_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof"
_AGENT_MODULE = "unitree_rl_lab.tasks.locomotion.agents.torque_traction_rsl_cfg"

gym.register(
    id="Unitree-G1-29dof-Velocity-TorqueTractionTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_MODULE}.velocity_torque_traction_teacher_env_cfg:RobotTorqueTractionTeacherEnvCfg",
        "play_env_cfg_entry_point": f"{_ENV_MODULE}.velocity_torque_traction_teacher_env_cfg:RobotTorqueTractionTeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:TorqueTractionTeacherRunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-TorqueTractionStudent",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_MODULE}.velocity_torque_traction_student_env_cfg:RobotTorqueTractionStudentEnvCfg",
        "play_env_cfg_entry_point": f"{_ENV_MODULE}.velocity_torque_traction_student_env_cfg:RobotTorqueTractionStudentPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:TorqueTractionStudentRunnerCfg",
    },
)
