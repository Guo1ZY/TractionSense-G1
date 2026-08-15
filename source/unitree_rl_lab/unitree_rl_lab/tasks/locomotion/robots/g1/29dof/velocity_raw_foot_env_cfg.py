"""Minimal G1-29DoF velocity task with signed raw foot-force observations.

This task deliberately keeps the original robot, action space, rewards,
commands and baseline observation prefix from :mod:`velocity_env_cfg`.
One six-dimensional term is appended after ``last_action`` and is stacked by
the existing five-frame observation history.
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.foot_sensor import FOOT_BODY_NAMES

from .velocity_env_cfg import ObservationsCfg, RobotEnvCfg, RobotSceneCfg


RAW_FOOT_GROUND_FILTER = ["/World/ground/terrain/mesh"]
RAW_LEFT_FOOT_SENSOR_CFG = SceneEntityCfg("left_raw_foot_contact")
RAW_RIGHT_FOOT_SENSOR_CFG = SceneEntityCfg("right_raw_foot_contact")
RAW_FOOT_FORCE_ASSET_CFG = SceneEntityCfg(
    "robot",
    body_names=list(FOOT_BODY_NAMES),
    preserve_order=True,
)


@configclass
class RawFootForceSceneCfg(RobotSceneCfg):
    """Baseline scene plus one friction-tracking ContactSensor per foot."""

    left_raw_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=list(RAW_FOOT_GROUND_FILTER),
    )
    right_raw_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=list(RAW_FOOT_GROUND_FILTER),
    )


def _raw_foot_force_term() -> ObsTerm:
    """Build the shared policy/critic force term without sharing mutable cfg."""
    return ObsTerm(
        func=mdp.normalized_raw_foot_force_local,
        params={
            "left_sensor_cfg": RAW_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": RAW_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": RAW_FOOT_FORCE_ASSET_CFG.copy(),
            "robot_mass_kg": None,
            "gravity_m_s2": 9.81,
        },
        clip=(-2.0, 2.0),
    )


@configclass
class RawFootForceObservationsCfg(ObservationsCfg):
    """Baseline observations plus signed local L/R force components."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Appended after all inherited baseline terms, including last_action.
        raw_foot_force = _raw_foot_force_term()

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        # Same clean force signal is available to the critic.
        raw_foot_force = _raw_foot_force_term()

    critic: CriticCfg = CriticCfg()


@configclass
class RobotRawFootEnvCfg(RobotEnvCfg):
    """Raw-force task; toggle off for an exact 480/495 baseline interface."""

    enable_raw_foot_force_obs: bool = True
    raw_foot_force_clip: tuple[float, float] = (-2.0, 2.0)
    raw_foot_force_robot_mass_kg: float | None = None

    scene: RawFootForceSceneCfg = RawFootForceSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: RawFootForceObservationsCfg = RawFootForceObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.left_raw_foot_contact.update_period = self.sim.dt
        self.scene.right_raw_foot_contact.update_period = self.sim.dt
        for group in (self.observations.policy, self.observations.critic):
            if not self.enable_raw_foot_force_obs:
                group.raw_foot_force = None
                continue
            group.raw_foot_force.clip = self.raw_foot_force_clip
            group.raw_foot_force.params["robot_mass_kg"] = self.raw_foot_force_robot_mass_kg


@configclass
class RobotRawFootPlayEnvCfg(RobotRawFootEnvCfg):
    """Short inference/debug configuration for the raw-force task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.observations.policy.enable_corruption = False


@configclass
class RobotRawFootBaselineCompatEnvCfg(RobotRawFootEnvCfg):
    """Configuration-switch OFF path for direct 480/495 baseline comparison."""

    enable_raw_foot_force_obs: bool = False
