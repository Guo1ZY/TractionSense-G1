"""Canonical 29-DOF traction Teacher and temporal Student Isaac tasks."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.traction.diagnostics import TractionDiagnosticsCfg
from unitree_rl_lab.traction.isaac_observations import (
    canonical_slip_penalty,
    canonical_student_frame,
    canonical_tangential_push_penalty,
    canonical_teacher_observation,
    high_traction_unnecessary_slowdown,
)
from unitree_rl_lab.traction.isaac_events import (
    CoherentFootFrictionWithBuffer,
)
from unitree_rl_lab.traction.schema import FORCE_FRAME, FORCE_ORDER
from unitree_rl_lab.traction.tactile import TactileDomainRandomizationCfg

from .velocity_env_cfg import EventCfg, RewardsCfg, RobotEnvCfg, RobotSceneCfg


RAW_FOOT_GROUND_FILTER = ["/World/ground/terrain/mesh"]
RAW_LEFT_FOOT_SENSOR_CFG = SceneEntityCfg("left_raw_foot_contact")
RAW_RIGHT_FOOT_SENSOR_CFG = SceneEntityCfg("right_raw_foot_contact")
RAW_FOOT_FORCE_ASSET_CFG = SceneEntityCfg(
    "robot",
    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    preserve_order=True,
)


@configclass
class CanonicalTractionSceneCfg(RobotSceneCfg):
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


def _teacher_term() -> ObsTerm:
    return ObsTerm(
        func=canonical_teacher_observation,
        params={
            "left_sensor_cfg": RAW_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": RAW_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": RAW_FOOT_FORCE_ASSET_CFG.copy(),
            "diagnostics_cfg": TractionDiagnosticsCfg(),
        },
    )


def _student_term(
    mode: str,
    *,
    tactile_curriculum_stage: int,
) -> ObsTerm:
    return ObsTerm(
        func=canonical_student_frame,
        params={
            "left_sensor_cfg": RAW_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": RAW_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": RAW_FOOT_FORCE_ASSET_CFG.copy(),
            "mode": mode,
            "tactile_cfg": TactileDomainRandomizationCfg(),
            "tactile_seed": 20260731,
            "tactile_curriculum_stage": tactile_curriculum_stage,
        },
    )


@configclass
class CanonicalTeacherObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        teacher = _teacher_term()

        def __post_init__(self):
            # Five time-major Teacher frames are converted by the RSL adapter
            # into the exact 480-D term-major pretrained actor observation.
            self.history_length = 5
            self.flatten_history_dim = True
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        teacher = _teacher_term()

        def __post_init__(self):
            # The critic adapter reconstructs 495-D legacy history, including
            # the privileged base-linear-velocity samples.
            self.history_length = 5
            self.flatten_history_dim = True
            self.concatenate_terms = True
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class CanonicalRandomizedStudentObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        student_frame = _student_term(
            "randomized_tactile_force",
            tactile_curriculum_stage=5,
        )

        def __post_init__(self):
            self.history_length = 15
            self.flatten_history_dim = True
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        teacher = _teacher_term()

        def __post_init__(self):
            # Keep the same five-frame privileged history used by the Teacher
            # critic and offline Teacher queries during Student/DAgger rollout.
            self.history_length = 5
            self.flatten_history_dim = True
            self.concatenate_terms = True
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class CanonicalIdealStudentObservationsCfg(CanonicalRandomizedStudentObservationsCfg):
    @configclass
    class PolicyCfg(CanonicalRandomizedStudentObservationsCfg.PolicyCfg):
        student_frame = _student_term(
            "ideal_raw_force",
            tactile_curriculum_stage=0,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CanonicalProprioStudentObservationsCfg(CanonicalRandomizedStudentObservationsCfg):
    @configclass
    class PolicyCfg(CanonicalRandomizedStudentObservationsCfg.PolicyCfg):
        student_frame = _student_term(
            "proprio_only",
            tactile_curriculum_stage=0,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CanonicalTractionRewardsCfg(RewardsCfg):
    slip = RewTerm(
        func=canonical_slip_penalty,
        weight=-0.5,
        params={
            "left_sensor_cfg": RAW_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": RAW_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": RAW_FOOT_FORCE_ASSET_CFG.copy(),
        },
    )
    tangential_push = RewTerm(
        func=canonical_tangential_push_penalty,
        weight=-0.08,
        params={
            "left_sensor_cfg": RAW_LEFT_FOOT_SENSOR_CFG.copy(),
            "right_sensor_cfg": RAW_RIGHT_FOOT_SENSOR_CFG.copy(),
            "asset_cfg": RAW_FOOT_FORCE_ASSET_CFG.copy(),
        },
    )
    high_traction_slowdown = RewTerm(
        func=high_traction_unnecessary_slowdown,
        weight=-0.2,
    )


@configclass
class CanonicalTractionEventCfg(EventCfg):
    """Exact per-foot μ at startup plus abrupt interval transitions."""

    physics_material = EventTerm(
        func=CoherentFootFrictionWithBuffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "left_body_name": "left_ankle_roll_link",
            "right_body_name": "right_ankle_roll_link",
            "asymmetric_probability": 0.5,
        },
    )
    friction_transition = EventTerm(
        func=CoherentFootFrictionWithBuffer,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "left_body_name": "left_ankle_roll_link",
            "right_body_name": "right_ankle_roll_link",
            "asymmetric_probability": 0.5,
        },
    )


@configclass
class CanonicalTractionBaseEnvCfg(RobotEnvCfg):
    """Original G1 asset/actions/PD with dedicated ground-filtered foot sensors."""

    action_dimension: int = 29
    foot_force_enabled: bool = True
    foot_force_scale: str = "inverse_current_robot_weight"
    foot_force_clip: tuple[float, float] = (-2.0, 2.0)
    foot_force_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    foot_force_frame: str = FORCE_FRAME
    foot_force_order: tuple[str, ...] = FORCE_ORDER
    scene: CanonicalTractionSceneCfg = CanonicalTractionSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    events: CanonicalTractionEventCfg = CanonicalTractionEventCfg()
    rewards: CanonicalTractionRewardsCfg = CanonicalTractionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.left_raw_foot_contact.update_period = self.sim.dt
        self.scene.right_raw_foot_contact.update_period = self.sim.dt


@configclass
class CanonicalTractionTeacherEnvCfg(CanonicalTractionBaseEnvCfg):
    observations: CanonicalTeacherObservationsCfg = CanonicalTeacherObservationsCfg()


@configclass
class CanonicalTractionStudentEnvCfg(CanonicalTractionBaseEnvCfg):
    observations: CanonicalRandomizedStudentObservationsCfg = (
        CanonicalRandomizedStudentObservationsCfg()
    )


@configclass
class CanonicalTractionStudentIdealEnvCfg(CanonicalTractionBaseEnvCfg):
    observations: CanonicalIdealStudentObservationsCfg = (
        CanonicalIdealStudentObservationsCfg()
    )


@configclass
class CanonicalTractionStudentProprioEnvCfg(CanonicalTractionBaseEnvCfg):
    observations: CanonicalProprioStudentObservationsCfg = (
        CanonicalProprioStudentObservationsCfg()
    )
    foot_force_enabled: bool = False


def _configure_play(cfg: CanonicalTractionBaseEnvCfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class CanonicalTractionTeacherPlayEnvCfg(CanonicalTractionTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class CanonicalTractionStudentPlayEnvCfg(CanonicalTractionStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class CanonicalTractionStudentIdealPlayEnvCfg(CanonicalTractionStudentIdealEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class CanonicalTractionStudentProprioPlayEnvCfg(
    CanonicalTractionStudentProprioEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)
