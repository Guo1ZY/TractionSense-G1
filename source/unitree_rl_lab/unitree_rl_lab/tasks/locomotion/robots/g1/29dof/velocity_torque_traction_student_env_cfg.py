"""Independent motor-torque traction Student task for the 29-DoF G1."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.traction_torque.isaac_observations import IsaacTorqueTractionFrame
from unitree_rl_lab.traction_torque import rewards as torque_rewards
from unitree_rl_lab.traction.isaac_observations import (
    canonical_slip_penalty,
    high_traction_unnecessary_slowdown,
)

from .velocity_env_cfg import ObservationsCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg
from .velocity_canonical_traction_env_cfg import CanonicalTractionEventCfg


TORQUE_FOOT_GROUND_FILTER = ["/World/ground/terrain/mesh"]
TORQUE_LEFT_FOOT_SENSOR_CFG = SceneEntityCfg("left_torque_truth_contact")
TORQUE_RIGHT_FOOT_SENSOR_CFG = SceneEntityCfg("right_torque_truth_contact")
TORQUE_FOOT_FORCE_ASSET_CFG = SceneEntityCfg(
    "robot",
    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    preserve_order=True,
)


@configclass
class TorqueTractionSceneCfg(RobotSceneCfg):
    """Baseline scene plus truth-only ground-filtered foot sensors.

    These sensors provide Teacher targets, rewards, and evaluation labels. The
    deployment Student observation term never receives their tensors.
    """

    left_torque_truth_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=list(TORQUE_FOOT_GROUND_FILTER),
    )
    right_torque_truth_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=list(TORQUE_FOOT_GROUND_FILTER),
    )


def _truth_params() -> dict[str, SceneEntityCfg]:
    return {
        "left_sensor_cfg": TORQUE_LEFT_FOOT_SENSOR_CFG.copy(),
        "right_sensor_cfg": TORQUE_RIGHT_FOOT_SENSOR_CFG.copy(),
        "asset_cfg": TORQUE_FOOT_FORCE_ASSET_CFG.copy(),
    }


@configclass
class TorqueTractionStudentObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        torque_traction_frame = ObsTerm(
            func=IsaacTorqueTractionFrame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "randomization_stage": 5,
                "seed": 20260803,
            },
            history_length=15,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # The value function retains the exact audited 495-D baseline critic.
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class TorqueTractionRewardsCfg(RewardsCfg):
    force_estimation = RewTerm(func=torque_rewards.force_estimation_error, weight=-0.10, params=_truth_params())
    contact_estimation = RewTerm(func=torque_rewards.contact_estimation_error, weight=-0.08, params=_truth_params())
    traction_utilization = RewTerm(func=torque_rewards.estimated_traction_utilization_penalty, weight=-0.04)
    force_temporal_consistency = RewTerm(func=torque_rewards.force_estimator_temporal_consistency, weight=-0.01)
    tangential_push = RewTerm(func=torque_rewards.ground_truth_tangential_push, weight=-0.05, params=_truth_params())
    ground_truth_slip = RewTerm(func=canonical_slip_penalty, weight=-0.40, params=_truth_params())
    high_traction_slowdown = RewTerm(func=high_traction_unnecessary_slowdown, weight=-0.20)


@configclass
class RobotTorqueTractionStudentEnvCfg(RobotEnvCfg):
    scene: TorqueTractionSceneCfg = TorqueTractionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TorqueTractionStudentObservationsCfg = TorqueTractionStudentObservationsCfg()
    rewards: TorqueTractionRewardsCfg = TorqueTractionRewardsCfg()
    events: CanonicalTractionEventCfg = CanonicalTractionEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.left_torque_truth_contact.update_period = self.sim.dt
        self.scene.right_torque_truth_contact.update_period = self.sim.dt
        if hasattr(self.observations.policy, "torque_traction_frame"):
            self.observations.policy.torque_traction_frame.params["randomization_stage"] = 5


@configclass
class RobotTorqueTractionStudentPlayEnvCfg(RobotTorqueTractionStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.observations.policy.torque_traction_frame.params["randomization_stage"] = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
