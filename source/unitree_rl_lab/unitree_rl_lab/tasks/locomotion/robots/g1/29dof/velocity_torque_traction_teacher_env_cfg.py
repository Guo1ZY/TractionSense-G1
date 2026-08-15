"""Privileged torque-traction Teacher task; never exported for deployment."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from unitree_rl_lab.traction_torque.isaac_teacher import torque_teacher_observation

from .velocity_torque_traction_student_env_cfg import (
    TORQUE_FOOT_FORCE_ASSET_CFG,
    TORQUE_LEFT_FOOT_SENSOR_CFG,
    TORQUE_RIGHT_FOOT_SENSOR_CFG,
    RobotTorqueTractionStudentEnvCfg,
)


def _teacher_term() -> ObsTerm:
    return ObsTerm(
        func=torque_teacher_observation,
        params={
            "left_sensor_cfg": TORQUE_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": TORQUE_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": TORQUE_FOOT_FORCE_ASSET_CFG.copy(),
        },
        history_length=5,
        flatten_history_dim=True,
    )


@configclass
class TorqueTractionTeacherObservationsCfg:
    @configclass
    class TeacherGroupCfg(ObsGroup):
        torque_teacher_frame = _teacher_term()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: TeacherGroupCfg = TeacherGroupCfg()
    critic: TeacherGroupCfg = TeacherGroupCfg()


@configclass
class RobotTorqueTractionTeacherEnvCfg(RobotTorqueTractionStudentEnvCfg):
    observations: TorqueTractionTeacherObservationsCfg = TorqueTractionTeacherObservationsCfg()


@configclass
class RobotTorqueTractionTeacherPlayEnvCfg(RobotTorqueTractionTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
