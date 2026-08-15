# Copyright (c) 2022-2025, The Isaac Lab Project Developers / local foot-sensor extension.
# SPDX-License-Identifier: BSD-3-Clause
"""G1-29DoF velocity env with optional foot-sensor observations + friction DR.

Design goals (user pipeline):
  1. Align zorn foot_sensor semantics (L/R, force, contact, units, Hz).
  2. Observation interface is toggleable; default OFF degrades to baseline 49999.
  3. Fine-tune from model_49999 with foot obs + sensor noise DR.
  4. Random μ + soft anti-slip rewards (policy learns; no hard if-μ rules).

Toggle via class flags on :class:`RobotFootEnvCfg`:
  - ``enable_foot_policy_obs``  (default True for this task)
  - ``enable_foot_critic_obs``  (default True)
  - ``enable_friction_dr``      (default True, wider μ)
  - ``enable_anti_slip``        (default True)

Baseline task ``Unitree-G1-29dof-Velocity`` is unchanged (always 49999-compatible).
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.sensors import HallFootSensorCfg, sync_hall_sensor_cfg_to_policy_terms
from unitree_rl_lab.sensors.hall_deformable_sole import HallSoleAttachmentActionCfg
from unitree_rl_lab.assets.deformable_usd import DeformableUsdFileCfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.foot_sensor import FOOT_BODY_NAMES

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    TerminationsCfg,
)


# --- Command envelopes for staged fine-tunes from model_4000 ---
# Baseline 4000 limits: vx∈[-0.5,1.0], vy∈[-0.3,0.3], wz∈[-0.2,0.2]


@configclass
class WideCommandsCfg(CommandsCfg):
    """Large strafe + turn (heavier; only if you need big lateral)."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.2), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-0.8, 0.8)
        ),
    )


@configclass
class TurnCommandsCfg(CommandsCfg):
    """In-place / walking turn fine-tune: enlarge yaw only; vx/vy same as model_4000.

    Key for right-stick spin:
      - ``rel_spin_envs=0.30`` → ~30% pure yaw (vx=vy=0, |wz|≥min_spin)
      - stock uniform rarely samples this; standing zeros *all* cmds including yaw
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,  # still practice true standstill
        rel_spin_envs=0.30,  # pure in-place turn (right stick only)
        min_spin_ang_vel=0.18,  # spin slot is meaningful, not near-zero
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Start near model_4000 envelope so resume is stable, then curriculum widens yaw.
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),  # same as model_4000
            lin_vel_y=(-0.3, 0.3),  # same as model_4000
            ang_vel_z=(-0.6, 0.6),  # ~3× turn rate for right stick
        ),
    )


@configclass
class TurnCurriculumCfg(CurriculumCfg):
    """Terrain + lin (capped at model_4000) + ang curriculum → wz ±0.6.

    Baseline foot task only expanded lin_vel; without ang_vel_cmd_levels the
    turn task would stay stuck at the start ranges forever.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


# ---------------------------------------------------------------------------
# Defaults for the foot-aware task (finetune). Baseline stays on velocity_env_cfg.
# ---------------------------------------------------------------------------

FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES))
FOOT_ASSET_CFG = SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES))
HALL_FOOT_ASSET_CFG = SceneEntityCfg(
    "robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True
)
HALL_LEFT_CONTACT_CFG = SceneEntityCfg("left_hall_contact")
HALL_RIGHT_CONTACT_CFG = SceneEntityCfg("right_hall_contact")
# This magnetic Student task is forced to TerrainImporter ``terrain_type=plane``
# by its teacher base class.  Isaac Lab 2.3.2 exposes that collider at this
# current path; generator terrain would instead use /World/ground/terrain/mesh.
HALL_GROUND_FILTER = ["/World/ground/terrain/GroundPlane/CollisionPlane"]
HALL_GENERATOR_GROUND_FILTER = ["/World/ground/terrain/mesh"]
HALL_SPATIAL_PATCH_FILTER = [
    "{ENV_REGEX_NS}/FrictionHighStart/geometry/mesh",
    "{ENV_REGEX_NS}/FrictionLow/geometry/mesh",
    "{ENV_REGEX_NS}/FrictionHighEnd/geometry/mesh",
]

# Dedicated Hall-foot terrain curriculum.  Difficulty is intentionally capped
# below Isaac Lab's generic rough-terrain maximum: a 29-DoF G1 with a 10 mm
# compliant sole first learns 0--11 degree ramps and 2--10 cm stairs.  Flat,
# ascent and descent are equally represented so Hall temporal features cannot
# memorize one terrain direction or one static magnetic baseline.
HALL_SLOPE_STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=12.0,
    num_rows=6,
    num_cols=5,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.20),
            platform_width=2.5,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.20),
            platform_width=2.5,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.10),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.10),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
    },
)
MAGNETIZED_TPU_USD = str(
    Path(__file__).resolve().parents[5]
    / "assets"
    / "meshes"
    / "tpu_sole_a40_grid35.usd"
)


@configclass
class HallFootSceneCfg(RobotSceneCfg):
    """Original scene plus current-API filtered contact data for Scheme A."""

    left_hall_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        track_friction_forces=True,
        # Stair/ramp edges can generate more than 16 filtered contact points
        # for one sole.  Keep enough headroom for the current Isaac Sim GPU
        # contact backend; this is a buffer-size safeguard, not an extra
        # observation channel.
        max_contact_data_count_per_prim=64,
        filter_prim_paths_expr=list(HALL_GROUND_FILTER),
    )
    right_hall_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        filter_prim_paths_expr=list(HALL_GROUND_FILTER),
    )


def _friction_patch_cfg(
    name: str,
    *,
    size_x: float,
    size_y: float = 2.0,
    center_x: float,
    friction: float,
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """Create one opaque, static, per-environment course collider."""
    thickness = 0.08
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(size_x, size_y, thickness),
            # CollisionAPI without RigidBodyAPI is a static collider in
            # Isaac Sim 5.1.  This avoids thousands of dynamic ground bodies.
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.88,
                opacity=1.0,
            ),
        ),
        # Every top face is exactly z=0.  The inherited dummy terrain is
        # removed at the end of the spatial-task post-init, so no plane is
        # hidden underneath these patches.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(center_x, 0.0, -0.5 * thickness)
        ),
        collision_group=0,
    )


@configclass
class HallSpatialFrictionSceneCfg(HallFootSceneCfg):
    """Three physical high--low--high floor patches in every cloned env."""

    friction_high_start = _friction_patch_cfg(
        "FrictionHighStart",
        size_x=2.0,
        center_x=-1.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=1.0,
        center_x=0.5,
        friction=0.16,
        color=(0.95, 0.22, 0.04),
    )
    friction_high_end = _friction_patch_cfg(
        "FrictionHighEnd",
        size_x=2.0,
        center_x=2.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )

    left_hall_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        filter_prim_paths_expr=list(HALL_SPATIAL_PATCH_FILTER),
    )
    right_hall_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=64,
        filter_prim_paths_expr=list(HALL_SPATIAL_PATCH_FILTER),
    )


@configclass
class HallSpatialMildFrictionSceneCfg(HallSpatialFrictionSceneCfg):
    """Stage-S1 course: real but recoverable first low-grip transition."""

    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=1.0,
        center_x=0.5,
        friction=0.45,
        color=(0.95, 0.48, 0.04),
    )


@configclass
class HallSpatialMediumFrictionSceneCfg(HallSpatialFrictionSceneCfg):
    """Stage-S2 course between mild grip and the final mu=0.16 patch."""

    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=1.0,
        center_x=0.5,
        friction=0.28,
        color=(0.95, 0.34, 0.04),
    )


@configclass
class HallSpatialMediumLongDemoSceneCfg(HallSpatialMediumFrictionSceneCfg):
    """Long, opaque H--L--H course for steady-state visual comparison only.

    Bounds in each environment's local X frame are exactly HighStart
    ``[-6,0]``, Low ``[0,6]`` and HighEnd ``[6,18]``.  All three patches are
    3.2 m wide, giving the robot and the camera much more lateral margin than
    the 2.0 m-wide training course.  The ordinary short Medium/MediumDense
    training scenes are intentionally untouched.
    """

    friction_high_start = _friction_patch_cfg(
        "FrictionHighStart",
        size_x=6.0,
        size_y=3.2,
        center_x=-3.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=6.0,
        size_y=3.2,
        center_x=3.0,
        friction=0.28,
        color=(0.95, 0.72, 0.05),
    )
    friction_high_end = _friction_patch_cfg(
        "FrictionHighEnd",
        size_x=12.0,
        size_y=3.2,
        center_x=12.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )


@configclass
class HallSpatialMediumRetentionSceneCfg(HallSpatialMediumFrictionSceneCfg):
    """Compact-clone long course used for high-speed retention training.

    Every environment still owns an isolated collision group, so 2.5 m clone
    spacing is valid even though the static floor meshes overlap in world
    space.  This avoids large world coordinates degrading the small Hall-field
    differences while providing roughly nine seconds of final-high exposure.
    """

    friction_high_start = _friction_patch_cfg(
        "FrictionHighStart",
        size_x=3.0,
        center_x=-1.5,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=2.0,
        center_x=1.0,
        friction=0.28,
        color=(0.95, 0.34, 0.04),
    )
    friction_high_end = _friction_patch_cfg(
        "FrictionHighEnd",
        size_x=8.0,
        center_x=6.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )


@configclass
class HallSlopeStairsSceneCfg(HallFootSceneCfg):
    """Scheme-A Hall scene on a five-family ramp/stair generator."""

    terrain = RobotSceneCfg(num_envs=1, env_spacing=2.5).terrain.replace(
        terrain_type="generator",
        terrain_generator=HALL_SLOPE_STAIRS_TERRAINS_CFG,
        max_init_terrain_level=1,
    )


@configclass
class HallUniformHighFrictionLongSceneCfg(HallSpatialFrictionSceneCfg):
    """Forty-metre opaque high-friction course for backbone training.

    The legacy filter prim names are retained so detailed Hall mechanics and
    contact-point rewards use the same code path as H--L--H.  All three
    colliders are physically and visually identical ``mu=0.90`` blue ground;
    there is no hidden material transition in this diagnostic task.
    """

    friction_high_start = _friction_patch_cfg(
        "FrictionHighStart",
        size_x=8.0,
        size_y=3.2,
        center_x=-4.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=12.0,
        size_y=3.2,
        center_x=6.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_high_end = _friction_patch_cfg(
        "FrictionHighEnd",
        size_x=20.0,
        size_y=3.2,
        center_x=22.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )


def _magnetized_tpu_asset_cfg(side: str, hall_cfg: HallFootSensorCfg) -> DeformableObjectCfg:
    """Build the sole's only deformable layer; the PCB enclosure stays rigid."""
    return DeformableObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{side}_magnetized_tpu",
        spawn=DeformableUsdFileCfg(
            usd_path=MAGNETIZED_TPU_USD,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                self_collision=hall_cfg.tpu_self_collision,
                solver_position_iteration_count=hall_cfg.tpu_solver_position_iteration_count,
                simulation_hexahedral_resolution=hall_cfg.tpu_simulation_hexahedral_resolution,
                collision_simplification=True,
                collision_simplification_force_conforming=True,
                contact_offset=hall_cfg.tpu_contact_offset,
                rest_offset=hall_cfg.tpu_rest_offset,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=hall_cfg.tpu_density,
                dynamic_friction=hall_cfg.tpu_dynamic_friction,
                youngs_modulus=hall_cfg.tpu_youngs_modulus,
                poissons_ratio=hall_cfg.tpu_poisson_ratio,
                elasticity_damping=hall_cfg.tpu_damping,
                damping_scale=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.32, 0.72) if side == "left" else (0.18, 0.58, 0.36),
                roughness=0.75,
                opacity=1.0,
            ),
        ),
        # The attachment action relocates every node from this harmless staging
        # pose to the current foot pose before the first physics step.
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=False,
    )


_DEFORMABLE_HALL_CFG = HallFootSensorCfg(implementation_mode="deformable")


@configclass
class HallFootDeformableSceneCfg(HallFootSceneCfg):
    """Scheme-B scene: rigid foot/PCB assembly plus one deformable TPU per foot."""

    left_magnetized_tpu = _magnetized_tpu_asset_cfg("left", _DEFORMABLE_HALL_CFG)
    right_magnetized_tpu = _magnetized_tpu_asset_cfg("right", _DEFORMABLE_HALL_CFG)


@configclass
class HallDeformableActionsCfg(ActionsCfg):
    """Normal 29-DoF policy action plus a zero-dimensional attachment hook."""

    hall_tpu_attachment = HallSoleAttachmentActionCfg(
        asset_name="robot",
        hall_cfg=_DEFORMABLE_HALL_CFG,
    )


def strip_foot_obs_terms(group: ObsGroup) -> None:
    """Remove optional foot observation terms (module-level so Hydra won't pickle it as a cfg field)."""
    for name in (
        "foot_contact",
        "foot_normal_force",
        "foot_tangent_force",
        "foot_force_history",
        "foot_force_vector",
        "foot_friction_ratio",
        "foot_slip_proxy",
        "foot_load_ratio",
        "foot_planar_vel",
        "foot_sensor_valid",
        "foot_sensor_age",
        "ground_friction_mu",
    ):
        if hasattr(group, name):
            setattr(group, name, None)


@configclass
class FootEventCfg(EventCfg):
    """Wider friction domain randomization for adaptive locomotion."""

    # Override startup material ranges (baseline is 0.3–1.0).
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.2),
            "dynamic_friction_range": (0.1, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Re-sample friction at episode reset so a single env sees many μ values.
    physics_material_reset = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.2),
            "dynamic_friction_range": (0.1, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class FootObservationsCfg(ObservationsCfg):
    """Policy / critic observations with optional foot terms.

    Foot terms are appended after the baseline terms so that a partial weight
    transfer from model_49999 can expand the first linear layer cleanly
    (old dims stay in the front).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        # --- foot sensor (aligned with zorn frame semantics) ---
        # Sensor noise DR mimics real force/contact quantization / drift.
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 1.0),
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 5.0),
        )
        foot_tangent_force = ObsTerm(
            func=mdp.foot_tangent_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 5.0),
        )
        # Short history: PolicyCfg.history_length=5 stacks contact/normal/tangent
        # (no extra foot_force_history term — avoids double temporal inflation).

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # Privileged / cleaner foot channels (no noise)
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
        )
        foot_tangent_force = ObsTerm(
            func=mdp.foot_tangent_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
        )
        # Optional privileged richer force history (sensor T=3 → 18 dims).
        # Group history stacks this as well; keep noise-free for critic only.
        foot_force_history = ObsTerm(
            func=mdp.foot_force_history,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01, "history_steps": 3},
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class FootRewardsCfg(RewardsCfg):
    """Baseline rewards + soft anti-slip (no hard if-μ branching)."""

    # Strengthen classic slide slightly under low-μ DR
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.35,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )

    # Continuous anti-slip (force-weighted velocity + soft cone stress)
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.25,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.15,
        },
    )

    # Mild force smoothness (sensor realism). Clamped inside the reward fn so
    # hard impacts cannot produce 1e10+ raw terms (was poisoning value loss).
    feet_force_rate = RewTerm(
        func=mdp.feet_force_rate,
        weight=-0.01,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_delta_clip": 200.0,
            "force_scale": 100.0,
        },
    )


@configclass
class FootTurnRewardsCfg(FootRewardsCfg):
    """Emphasize yaw tracking for turn fine-tune (keep lin track at 1.0)."""

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,  # was 0.5 on baseline
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class RobotFootEnvCfg(RobotEnvCfg):
    """Foot-sensor + friction-adaptive velocity environment.

    Class flags (set before gym.make / override in scripts):
      enable_foot_policy_obs, enable_foot_critic_obs,
      enable_friction_dr, enable_anti_slip
    """

    # --- toggles (True = foot finetune path; False degrades toward 49999) ---
    enable_foot_policy_obs: bool = True
    enable_foot_critic_obs: bool = True
    enable_friction_dr: bool = True
    enable_anti_slip: bool = True

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = FootObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # Default foot: same lin limits as model_4000 (not wide strafe)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = FootRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = FootEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Contact sensor already history_length=3 on the scene; keep update = sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        # Slightly longer history for foot_force_history term (still cheap)
        if self.scene.contact_forces.history_length < 3:
            self.scene.contact_forces.history_length = 3

        # ---- degrade path: strip foot terms / rewards / wider DR if disabled ----
        if not self.enable_foot_policy_obs:
            strip_foot_obs_terms(self.observations.policy)
        if not self.enable_foot_critic_obs:
            strip_foot_obs_terms(self.observations.critic)

        if not self.enable_anti_slip:
            # remove extra anti-slip terms; keep baseline feet_slide weight from parent
            if hasattr(self.rewards, "feet_anti_slip"):
                self.rewards.feet_anti_slip = None  # type: ignore
            if hasattr(self.rewards, "feet_force_rate"):
                self.rewards.feet_force_rate = None  # type: ignore
            # restore baseline slide weight
            if hasattr(self.rewards, "feet_slide") and self.rewards.feet_slide is not None:
                self.rewards.feet_slide.weight = -0.2

        if not self.enable_friction_dr:
            # fall back to baseline EventCfg friction (0.3–1.0, startup only)
            self.events = EventCfg()


@configclass
class RobotFootPlayEnvCfg(RobotFootEnvCfg):
    """Play / eval config for foot-aware policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Eval: keep corruption off so we can see clean sensor channels
        self.observations.policy.enable_corruption = False


@configclass
class RobotFootBaselineCompatEnvCfg(RobotFootEnvCfg):
    """Same scene as foot task but **all foot extras OFF** → 49999-compatible obs.

    Use this to verify degradation / A-B against baseline without switching task id.
    """

    enable_foot_policy_obs: bool = False
    enable_foot_critic_obs: bool = False
    enable_friction_dr: bool = False
    enable_anti_slip: bool = False


@configclass
class RobotFootTurnEnvCfg(RobotFootEnvCfg):
    """Foot env + in-place/walking yaw enlarge (vx/vy same as model_4000).

    Resume from ``model_foot_4000.pt``. Deploy right-stick already maps to ``ang_vel_z``.
    """

    commands: CommandsCfg = TurnCommandsCfg()
    rewards: RewardsCfg = FootTurnRewardsCfg()
    curriculum: CurriculumCfg = TurnCurriculumCfg()


@configclass
class RobotFootTurnPlayEnvCfg(RobotFootTurnEnvCfg):
    """Play: fixed pure in-place spin (vx=vy=0) to visually check right-stick turn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # Default fixed cmd = pure spin (matches right-stick only, left stick center).
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.4, 0.55),  # continuous in-place turn
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0  # ranges already pure yaw
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Adaptive: fast walk / light jog on high μ + slow-stable on low μ + turn
# ---------------------------------------------------------------------------


@configclass
class AdaptiveCommandsCfg(CommandsCfg):
    """Speed envelope up + yaw/spin for right-stick turn; vy kept moderate.

    Targets:
      * high μ: follow large vx (fast walk / light jog style)
      * low μ: policy may lag cmd (slip-aware track) → slow but upright
      * right stick: pure yaw via ``rel_spin_envs``

    Yaw start is forced open (±0.4) so resume from model_5400 immediately
    practices mid-rate turns; curriculum (relaxed threshold) pushes to ±0.6.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_spin_envs=0.32,  # more pure in-place turn samples
        min_spin_ang_vel=0.25,  # spin slot uses meaningful |wz|
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Lin already learned at 5400 → start wide. Yaw forced above old ±0.2.
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.9),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.4, 0.4),  # was ±0.2; force mid yaw immediately
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2),  # faster forward than 4000 (1.0)
            lin_vel_y=(-0.3, 0.3),  # same as 4000 — no huge strafe
            ang_vel_z=(-0.6, 0.6),  # right-stick turn (~3×)
        ),
    )


@configclass
class AdaptiveCurriculumCfg(CurriculumCfg):
    """Open linear (if needed) + yaw; yaw threshold relaxed so track_ang~0.6 can expand."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    # Default 0.8*weight blocked expansion when track_ang≈0.63; use 0.5.
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={
            "reward_term_name": "track_ang_vel_z",
            "reward_threshold_frac": 0.5,
            "delta_ang": 0.1,
        },
    )


@configclass
class AdaptiveEventCfg(FootEventCfg):
    """Slightly wider μ so low-friction episodes appear often."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.2),
            "dynamic_friction_range": (0.08, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.2),
            "dynamic_friction_range": (0.08, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class FootAdaptiveObservationsCfg(FootObservationsCfg):
    """Policy same as foot_4000; critic adds ρ + slip (asymmetric / privileged)."""

    @configclass
    class CriticCfg(FootObservationsCfg.CriticCfg):
        foot_friction_ratio = ObsTerm(
            func=mdp.foot_friction_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 1.0, "clip_max": 5.0},
        )
        foot_slip_proxy = ObsTerm(
            func=mdp.foot_slip_proxy,
            params={
                "sensor_cfg": FOOT_SENSOR_CFG,
                "asset_cfg": FOOT_ASSET_CFG,
                "force_threshold": 5.0,
                "soft_scale": 0.5,
                "vel_scale": 1.0,
            },
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class FootAdaptiveRewardsCfg(FootRewardsCfg):
    """Slip-aware tracking + stronger anti-slip + turn emphasis.

    Design:
      * track_*_slip_aware: full reward when planted; soft when sliding
        → high μ uses full cmd (fast); low μ not forced to 1.2 m/s
      * stronger feet_slide / feet_anti_slip: prefer slow-stable over skid
      * higher yaw track: in-place / walk-turn for right stick
      * slightly quicker gait period: light-jog style cadence
    """

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_slip_aware,
        weight=1.15,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.45,
            "min_track_scale": 0.35,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_slip_aware,
        weight=1.15,  # emphasize yaw for right-stick / spin after model_5400
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.50,
            "min_track_scale": 0.45,  # less aggressive soft-down so curriculum can fire
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.45,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.40,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.20,
        },
    )

    # Slightly faster cadence for light-jog style (still bipedal walk, not flight).
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.72,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootAdaptiveEnvCfg(RobotFootEnvCfg):
    """Foot + speed + turn + friction-adaptive (recommended combined finetune).

    Resume/warm-start from ``model_foot_4000.pt``:
      * policy obs layout matches foot_4000 (contact/normal/tangent) → actor can copy
      * critic gains ρ + slip_proxy → use ``--partial_checkpoint`` to expand critic
    """

    observations: ObservationsCfg = FootAdaptiveObservationsCfg()
    commands: CommandsCfg = AdaptiveCommandsCfg()
    rewards: RewardsCfg = FootAdaptiveRewardsCfg()
    events: EventCfg = AdaptiveEventCfg()
    curriculum: CurriculumCfg = AdaptiveCurriculumCfg()


@configclass
class RobotFootAdaptivePlayEnvCfg(RobotFootAdaptiveEnvCfg):
    """Play default: brisk forward walk (check high-speed tracking)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.9, 1.1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Stable fix: stop idle stomping + stronger low-μ slow-down (from model_6600)
# ---------------------------------------------------------------------------


@configclass
class StableAdaptiveCommandsCfg(CommandsCfg):
    """Keep speed/yaw limits; much more pure standing practice."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 12.0),
        rel_standing_envs=0.18,  # was 0.05 — learn zero-cmd stand still
        rel_spin_envs=0.22,
        min_spin_ang_vel=0.22,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Already know speed/yaw; start mid-open so standing + slip get sample mass
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.8),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.45, 0.45),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
        ),
    )


@configclass
class FootStableAdaptiveRewardsCfg(FootAdaptiveRewardsCfg):
    """Fix idle marching + force slow-down when feet slip (low μ).

    User feedback on model_6600:
      * zero stick → keeps L/R stomping (want stand still like 4000)
      * ICE vs GRIP → |v| almost same (want low μ slower / safer)
    """

    # Stronger soft-down when sliding so low-μ is not forced to track 1.0 m/s
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_slip_aware,
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.30,  # scale drops faster as slip grows
            "min_track_scale": 0.12,  # allow almost not tracking when skating
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_slip_aware,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.35,
            "min_track_scale": 0.20,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.55,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.55,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.28,
        },
    )

    # --- idle stand still (kill in-place stomp) ---
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-1.2,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.12,
        },
    )
    base_still_when_idle = RewTerm(
        func=mdp.base_still_when_idle,
        weight=-0.8,
        params={"command_name": "base_velocity", "cmd_threshold": 0.12},
    )
    # Prefer both feet planted when idle
    feet_contact_idle = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.35,
        params={"sensor_cfg": FOOT_SENSOR_CFG, "command_name": "base_velocity"},
    )

    # Gait only when moving (already gated); slightly weaker so idle wins
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.35,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootStableAdaptiveEnvCfg(RobotFootAdaptiveEnvCfg):
    """Resume from adaptive_yaw model_6600: stand-still + low-μ slow-down."""

    commands: CommandsCfg = StableAdaptiveCommandsCfg()
    rewards: RewardsCfg = FootStableAdaptiveRewardsCfg()
    # Keep adaptive curriculum (ang threshold 0.5) + friction events


@configclass
class RobotFootStableAdaptivePlayEnvCfg(RobotFootStableAdaptiveEnvCfg):
    """Play: zero command stand (check no stomping)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Clean rebuild from model_49999 (partial). Priority: μ adapt + multi-speed.
# Foot channels are sensors for friction (not "fix foot"); 49999 walk stays base.
# ---------------------------------------------------------------------------


@configclass
class FullCommandsCfg(CommandsCfg):
    """Priority: multi-speed tracking under random μ; yaw moderate bonus.

    Start near 49999 band (stable partial warm-start), curriculum opens:
      vx → 1.2 (slow / mid / fast), wz → ±0.6, spin for right stick.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),  # more speed changes per episode
        rel_standing_envs=0.06,  # light stand (49999 already stands; not main goal)
        rel_spin_envs=0.18,  # some pure yaw; secondary to speed+μ
        min_spin_ang_vel=0.18,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.15),  # start mild; curriculum grows multi-speed
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.15, 0.15),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2),  # multi-speed envelope
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
        ),
    )


@configclass
class FullCurriculumCfg(CurriculumCfg):
    """Main: open lin speed. Yaw secondary (relaxed threshold)."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={
            "reward_term_name": "track_ang_vel_z",
            "reward_threshold_frac": 0.5,
            "delta_ang": 0.1,
        },
    )


@configclass
class FootFullRewardsCfg(FootRewardsCfg):
    """μ adapt + multi-speed first; yaw/stand light (from 49999 walk base).

    Core idea (no hard if-μ rules):
      * high μ / no slip → full track_lin → can follow large stick
      * slip (low μ) → track softens + anti_slip → prefer slow stable over crash
      * foot obs = friction cues (Fn/Ft/contact), not a separate "foot task"
    """

    # --- PRIMARY: multi-speed tracking, slip-aware ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_slip_aware,
        weight=1.35,  # main task
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.28,  # slip rises → track drops fast
            "min_track_scale": 0.10,  # low μ: almost stop forcing high speed
        },
    )
    # --- PRIMARY: friction / anti-slip (weights moderate — term already clamps) ---
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.45,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.45,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.25,
        },
    )
    # Inherited feet_force_rate already ΔF-clamped (avoids 1e10 value targets).

    # --- SECONDARY: yaw (keep usable, not the focus) ---
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_slip_aware,
        weight=0.85,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.35,
            "min_track_scale": 0.25,
        },
    )

    # --- light idle (keep 49999 stand; avoid 6600-style stomp if any) ---
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-0.45,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.10,
        },
    )

    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.45,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootFullEnvCfg(RobotFootEnvCfg):
    """From model_49999 (partial). Focus: μ adaptation + multi-speed; yaw secondary.

    Foot terms = sensors for friction (49999 walk already stable).
    Do **not** resume from 6600 stack.
    """

    observations: ObservationsCfg = FootAdaptiveObservationsCfg()
    commands: CommandsCfg = FullCommandsCfg()
    rewards: RewardsCfg = FootFullRewardsCfg()
    events: EventCfg = AdaptiveEventCfg()  # μ ∈ [0.08, 1.2] each episode
    curriculum: CurriculumCfg = FullCurriculumCfg()


@configclass
class RobotFootFullPlayEnvCfg(RobotFootFullEnvCfg):
    """Play: forward walk at mid speed."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Foot-Adaptive-V2: fix μ-invariant mid-speed (outcome rewards, actor deployable)
# ---------------------------------------------------------------------------


@configclass
class V2CommandsCfg(CommandsCfg):
    """Multi-speed in training distribution (must cover desired high speed).

    limit vx → 1.3 so high-μ fast is *inside* command curriculum, not OOD.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.08,
        rel_spin_envs=0.15,
        min_spin_ang_vel=0.18,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.2),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.15, 0.15),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.3),  # high-μ target inside distribution
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
        ),
    )


@configclass
class V2CurriculumCfg(CurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={
            "reward_term_name": "track_ang_vel_z",
            "reward_threshold_frac": 0.5,
            "delta_ang": 0.1,
        },
    )


@configclass
class V2EventCfg(EventCfg):
    """Friction DR + privileged μ buffer + sensor dropout."""

    physics_material = EventTerm(
        func=mdp.randomize_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.25),
            "dynamic_friction_range": (0.06, 1.15),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.25),
            "dynamic_friction_range": (0.06, 1.15),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    foot_sensor_reset = EventTerm(
        func=mdp.reset_foot_sensor_valid,
        mode="reset",
        params={},
    )
    foot_sensor_dropout = EventTerm(
        func=mdp.randomize_foot_sensor_dropout,
        mode="reset",
        params={"dropout_prob": 0.06, "stale_age": 0.3},
    )


@configclass
class FootV2ObservationsCfg(ObservationsCfg):
    """Actor: deployable only (no true Ft, no true μ). Critic: privileged.

    Single-step actor ≈ 96 + contact2 + Fn2 + load2 + valid1 + age1 = 104
    history 5 → 520. Remove foot_tangent from actor (HW may lack shear; Ft≠μ).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 1.0),
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 5.0),
        )
        foot_load_ratio = ObsTerm(
            func=mdp.foot_load_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 1.0},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(0.0, 1.0),
        )
        foot_sensor_valid = ObsTerm(
            func=mdp.foot_sensor_valid,
            params={"default_valid": 1.0},
            clip=(0.0, 1.0),
        )
        foot_sensor_age = ObsTerm(
            func=mdp.foot_sensor_age,
            params={"age_scale": 0.25, "clip_max": 1.0},
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
        )
        foot_load_ratio = ObsTerm(
            func=mdp.foot_load_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 1.0},
        )
        foot_friction_ratio = ObsTerm(
            func=mdp.foot_friction_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 1.0, "clip_max": 5.0},
        )
        foot_slip_proxy = ObsTerm(
            func=mdp.foot_slip_proxy,
            params={
                "sensor_cfg": FOOT_SENSOR_CFG,
                "asset_cfg": FOOT_ASSET_CFG,
                "force_threshold": 5.0,
                "soft_scale": 0.5,
                "vel_scale": 1.0,
            },
        )
        ground_friction_mu = ObsTerm(
            func=mdp.ground_friction_mu,
            params={"default_mu": 0.8, "clip_max": 2.0},
        )
        foot_sensor_valid = ObsTerm(func=mdp.foot_sensor_valid, params={"default_valid": 1.0})
        foot_sensor_age = ObsTerm(func=mdp.foot_sensor_age, params={"age_scale": 0.25})

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class FootV2RewardsCfg(RewardsCfg):
    """Outcome multi-objective: full track + stable_speed_bonus + slip costs.

    Design intent (fix μ-invariant mid-speed):
      * track always full → high cmd still pulls on high μ
      * stable_speed_bonus only when track AND low slip → high μ gets extra reward for going fast
      * slip_under_command + feet_slide/anti_slip → low μ cannot skate for free
      * NO slip_aware min_track_scale floor (that taught give-up / uniform mid-speed)
    """

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_full,
        weight=1.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_full,
        weight=0.7,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    stable_speed_bonus = RewTerm(
        func=mdp.stable_speed_bonus,
        weight=0.55,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.30,
            "cmd_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.50,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.40,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.22,
        },
    )
    slip_under_command = RewTerm(
        func=mdp.slip_under_command,
        weight=-0.35,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "cmd_scale": 1.0,
        },
    )
    feet_force_rate = RewTerm(
        func=mdp.feet_force_rate,
        weight=-0.01,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_delta_clip": 200.0,
            "force_scale": 100.0,
        },
    )
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-0.50,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.10,
        },
    )
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.45,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootAdaptiveV2EnvCfg(RobotFootEnvCfg):
    """Foot-Adaptive-V2: μ adapt via outcome rewards; actor deployable obs only.

    Warm-start: ``--partial_checkpoint model/rl/model_49999.pt`` (obs dim grows).
    Do not overwrite existing Foot-Full / foot ONNX.
    """

    observations: ObservationsCfg = FootV2ObservationsCfg()
    commands: CommandsCfg = V2CommandsCfg()
    rewards: RewardsCfg = FootV2RewardsCfg()
    events: EventCfg = V2EventCfg()
    curriculum: CurriculumCfg = V2CurriculumCfg()


@configclass
class RobotFootAdaptiveV2PlayEnvCfg(RobotFootAdaptiveV2EnvCfg):
    """Play: fixed forward cmd for qualitative μ tests."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.9, 1.1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Foot-MuAdapt: 510-dim actor (Fn+Ft, same as Foot-Full) + outcome rewards
# Fix: μ-invariant mid-speed / ICE side-skid with high |v_xy|.
# ---------------------------------------------------------------------------


@configclass
class MuAdaptCommandsCfg(CommandsCfg):
    """Forward-biased multi-speed; keep vy moderate so lateral_slip_penalty is meaningful."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.08,
        rel_spin_envs=0.12,
        min_spin_ang_vel=0.18,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.2),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.15, 0.15),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2),  # match deploy foot; high speed in distribution
            lin_vel_y=(-0.25, 0.25),  # slightly tighter than ±0.3 — less pure strafe
            ang_vel_z=(-0.6, 0.6),
        ),
    )


@configclass
class FootMuAdaptRewardsCfg(FootRewardsCfg):
    """510-dim compatible rewards: full track + stable bonus + foot slip + lateral skid.

    No slip_aware min_track_scale (that taught mid-speed give-up).
    Actor still uses contact/Fn/Ft (same as Foot-Full ONNX 510).
    Weights kept moderate to avoid value/std explosion on resume from Full.
    """

    # Full tracking — always pull toward cmd
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_full,
        weight=1.05,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Extra weight on forward axis (MuJoCo tests: ICE inflated |v| via vy)
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=0.35,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_full,
        weight=0.6,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # High μ: track + planted → bonus; low μ: slip kills bonus
    stable_speed_bonus = RewTerm(
        func=mdp.stable_speed_bonus,
        weight=0.40,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.28,
            "cmd_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.40,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.35,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.22,
        },
    )
    slip_under_command = RewTerm(
        func=mdp.slip_under_command,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "cmd_scale": 1.0,
        },
    )
    # Body lateral skid when cmd is forward (ICE side-skid fix)
    lateral_slip = RewTerm(
        func=mdp.lateral_slip_penalty,
        weight=-0.40,
        params={
            "command_name": "base_velocity",
            "cmd_x_threshold": 0.15,
            "vy_clip": 1.2,
        },
    )
    feet_force_rate = RewTerm(
        func=mdp.feet_force_rate,
        weight=-0.005,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_delta_clip": 150.0,
            "force_scale": 100.0,
        },
    )
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-0.40,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.10,
        },
    )
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.40,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootMuAdaptEnvCfg(RobotFootEnvCfg):
    """510 actor (Fn+Ft) + outcome rewards. Partial/strict from foot checkpoints OK.

    Prefer warm-start:
      * model_foot_4000 / Foot-Full same 510 → can strict resume weights
      * or partial from model_49999
    Deploy dir: config/policy/velocity/foot_mu (do not overwrite foot/).
    """

    # Policy layout = Foot (contact/Fn/Ft); critic = Adaptive (ρ+slip)
    observations: ObservationsCfg = FootAdaptiveObservationsCfg()
    commands: CommandsCfg = MuAdaptCommandsCfg()
    rewards: RewardsCfg = FootMuAdaptRewardsCfg()
    events: EventCfg = AdaptiveEventCfg()  # μ [0.08, 1.2]
    curriculum: CurriculumCfg = FullCurriculumCfg()


@configclass
class RobotFootMuAdaptPlayEnvCfg(RobotFootMuAdaptEnvCfg):
    """Play: fixed forward mid-high speed (straight-line test)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Straight-Mu (2026-07-17 clean line): high-μ fast straight / low-μ slow-stable
# From model_49999 partial. NO turn stack / NO spin / narrow yaw.
# ---------------------------------------------------------------------------


@configclass
class StraightMuCommandsCfg(CommandsCfg):
    """Forward multi-speed under μ-DR; yaw stays 49999-narrow; no pure-spin slots.

    Deploy: full stick → vx_cmd = 1.5 (G1_CMD_GAIN_LIN × stick, clamp to deploy max).
    Training MUST sample up to 1.5 so high-μ tracking is in-distribution.

    Goals:
      * high μ: curriculum opens vx → **1.5**, track + planted bonus → catch full-stick
      * low μ: slip kills stable_speed_bonus + slip_under_command → lag cmd, stay upright
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.10,  # stand still practice
        rel_spin_envs=0.0,  # NO in-place turn sampling
        min_spin_ang_vel=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # Start mild near 49999; lin curriculum grows vx toward limit 1.5
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.15),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-0.1, 0.1),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),  # match full-stick deploy (high-μ target in dist.)
            lin_vel_y=(-0.2, 0.2),  # tight strafe — prefer straight
            ang_vel_z=(-0.2, 0.2),  # same as 49999 — NO yaw expand
        ),
    )


@configclass
class StraightMuCurriculumCfg(CurriculumCfg):
    """Open linear speed only. Do NOT expand yaw (keeps walk straight)."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    # no ang_vel_cmd_levels — wz stays within start/limit ±0.2


@configclass
class FootStraightMuRewardsCfg(FootRewardsCfg):
    """Outcome rewards: full-stick 1.5 on high μ; lag & stable on low μ.

    No if-μ rules. High cmd always present in training (limit 1.5).
      * high μ: low slip → track + stable_speed_bonus → chase 1.5
      * low μ: slip_under_command + anti_slip + lost bonus → slower than cmd OK
    """

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_full,
        weight=1.15,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=0.55,  # push forward catch-up to large cmd (1.5)
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Keep yaw mild (49999 band only) — not a training focus
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_full,
        weight=0.35,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # High μ: track + planted → bonus; low μ: slip kills bonus → prefer slower
    stable_speed_bonus = RewTerm(
        func=mdp.stable_speed_bonus,
        weight=0.55,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_vel_scale": 0.24,
            "cmd_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.45,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.45,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.26,
        },
    )
    # Cost of skating while commanding speed (full-stick 1.5 on ICE is expensive)
    slip_under_command = RewTerm(
        func=mdp.slip_under_command,
        weight=-0.38,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "cmd_scale": 1.0,  # slip × (1 + |cmd|); larger when stick full
        },
    )
    # Kill ICE side-skid when cmd is mostly forward
    lateral_slip = RewTerm(
        func=mdp.lateral_slip_penalty,
        weight=-0.60,
        params={
            "command_name": "base_velocity",
            "cmd_x_threshold": 0.12,
            "vy_clip": 1.2,
        },
    )
    feet_force_rate = RewTerm(
        func=mdp.feet_force_rate,
        weight=-0.005,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_delta_clip": 150.0,
            "force_scale": 100.0,
        },
    )
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-0.45,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.10,
        },
    )
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.40,
        params={
            "period": 0.75,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
        },
    )


@configclass
class RobotFootStraightMuEnvCfg(RobotFootEnvCfg):
    """Clean rebuild from model_49999: foot Fn+Ft + μ outcome; no turn.

    Deploy later to config/policy/velocity/foot/ (or foot_mu/).
    """

    observations: ObservationsCfg = FootAdaptiveObservationsCfg()  # policy 510 Fn+Ft
    commands: CommandsCfg = StraightMuCommandsCfg()
    rewards: RewardsCfg = FootStraightMuRewardsCfg()
    events: EventCfg = AdaptiveEventCfg()  # μ ∈ [0.08, 1.2]
    curriculum: CurriculumCfg = StraightMuCurriculumCfg()


@configclass
class RobotFootStraightMuPlayEnvCfg(RobotFootStraightMuEnvCfg):
    """Play: pure forward full-stick band (check high-μ catch-up to ~1.5)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.3, 1.5),  # full-stick stress
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Traction-Adaptive: default <= 1.0 m/s, high-grip fast / low-grip slow
# ---------------------------------------------------------------------------


@configclass
class TractionAdaptiveCommandsCfg(CommandsCfg):
    """Straight velocity commands with a safe default and rare stress probes.

    The normal distribution never exceeds 1.0 m/s.  Fifteen percent of moving
    episodes deliberately request 1.0--1.5 m/s so a full-stick or oversized
    command is still in-distribution.  Those probes are sampled independently
    of friction: high-grip episodes may run fast, while low-grip episodes are
    explicitly taught to saturate below the requested speed.
    """

    base_velocity = mdp.TractionAdaptiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.12,
        rel_spin_envs=0.0,
        min_spin_ang_vel=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.TractionAdaptiveVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        # No velocity curriculum uses this envelope.  It documents the stress
        # support and keeps deployment/export metadata honest.
        limit_ranges=mdp.TractionAdaptiveVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        high_speed_fraction=0.15,
        high_speed_range=(1.0, 1.5),
    )


@configclass
class TractionAdaptiveCurriculumCfg(CurriculumCfg):
    """Terrain curriculum only; do not silently expand the normal speed range."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None


@configclass
class TractionAdaptiveEventCfg(EventCfg):
    """Coherent per-environment friction plus deploy-style sensor dropout."""

    physics_material = EventTerm(
        func=mdp.randomize_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.04, 1.10),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.04, 1.10),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    foot_sensor_reset = EventTerm(func=mdp.reset_foot_sensor_valid, mode="reset", params={})
    foot_sensor_dropout = EventTerm(
        func=mdp.randomize_foot_sensor_dropout,
        mode="reset",
        params={"dropout_prob": 0.05, "stale_age": 0.30},
    )


@configclass
class FootTractionAdaptiveObservationsCfg(ObservationsCfg):
    """Baseline-compatible proprioception plus long foot-force context.

    Actor layout keeps the original 49999 columns first (480 dims), then adds
    0.3 s of contact/Fn/Ft/utilization/load history and sensor health.  The
    actor never observes true friction or the simulator slip proxy.  The critic
    gets both as privileged teacher information.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Keep exact baseline order/history for partial warm-start from 49999.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=5,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=5,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=5,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=5)

        # 15 policy steps at 50 Hz = 0.30 s of traction evidence.
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={
                "sensor_cfg": FOOT_SENSOR_CFG,
                "threshold": 5.0,
                "soft": True,
                "respect_sensor_valid": True,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 1.0),
            history_length=15,
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01, "respect_sensor_valid": True},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 5.0),
            history_length=15,
        )
        foot_tangent_force = ObsTerm(
            func=mdp.foot_tangent_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01, "respect_sensor_valid": True},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 5.0),
            history_length=15,
        )
        foot_friction_ratio = ObsTerm(
            func=mdp.foot_friction_ratio,
            params={
                "sensor_cfg": FOOT_SENSOR_CFG,
                "eps": 5.0,
                "clip_max": 2.0,
                "respect_sensor_valid": True,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(0.0, 2.0),
            history_length=15,
        )
        foot_load_ratio = ObsTerm(
            func=mdp.foot_load_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 5.0, "respect_sensor_valid": True},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(0.0, 1.0),
            history_length=15,
        )
        foot_sensor_valid = ObsTerm(
            func=mdp.foot_sensor_valid,
            params={"default_valid": 1.0},
            clip=(0.0, 1.0),
            history_length=5,
        )
        foot_sensor_age = ObsTerm(
            func=mdp.foot_sensor_age,
            params={"age_scale": 0.25, "clip_max": 1.0},
            clip=(0.0, 1.0),
            history_length=5,
        )

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Exact baseline critic prefix = 99 dims/step * 5 = 495.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5)
        last_action = ObsTerm(func=mdp.last_action, history_length=5)

        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
            history_length=5,
        )
        foot_normal_force = ObsTerm(
            func=mdp.foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            history_length=5,
        )
        foot_tangent_force = ObsTerm(
            func=mdp.foot_tangent_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            history_length=5,
        )
        foot_friction_ratio = ObsTerm(
            func=mdp.foot_friction_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 5.0, "clip_max": 2.0},
            history_length=5,
        )
        foot_slip_proxy = ObsTerm(
            func=mdp.foot_slip_proxy,
            params={
                "sensor_cfg": FOOT_SENSOR_CFG,
                "asset_cfg": FOOT_ASSET_CFG,
                "force_threshold": 5.0,
                "soft_scale": 0.5,
                "vel_scale": 1.0,
            },
            history_length=5,
        )
        foot_load_ratio = ObsTerm(
            func=mdp.foot_load_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 5.0},
            history_length=5,
        )
        ground_friction_mu = ObsTerm(
            func=mdp.ground_friction_mu,
            params={"default_mu": 0.8, "clip_max": 2.0},
            history_length=5,
        )
        foot_sensor_valid = ObsTerm(
            func=mdp.foot_sensor_valid,
            params={"default_valid": 1.0},
            history_length=5,
        )
        foot_sensor_age = ObsTerm(
            func=mdp.foot_sensor_age,
            params={"age_scale": 0.25, "clip_max": 1.0},
            history_length=5,
        )

        def __post_init__(self):
            self.history_length = None
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class FootTractionAdaptiveRewardsCfg(FootRewardsCfg):
    """Teach a smooth traction-dependent speed cap and straight stable gait."""

    # This replaces raw command tracking: on ice the target itself is safely
    # capped, while high μ retains the full 1.5-m/s stress target.
    track_lin_vel_xy = RewTerm(
        func=mdp.traction_limited_track_lin_vel_x_exp,
        weight=1.80,
        params={
            "command_name": "base_velocity",
            "std": 0.40,
            "low_speed": 0.20,
            "high_speed": 1.50,
            "mu_midpoint": 0.55,
            "mu_width": 0.14,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_full,
        weight=0.60,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    traction_overspeed = RewTerm(
        func=mdp.traction_overspeed_penalty,
        weight=-1.20,
        params={
            "command_name": "base_velocity",
            "low_speed": 0.20,
            "high_speed": 1.50,
            "mu_midpoint": 0.55,
            "mu_width": 0.14,
        },
    )
    friction_cone_margin = RewTerm(
        func=mdp.friction_cone_margin_penalty,
        weight=-0.45,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "safe_utilization": 0.75,
            "force_threshold": 5.0,
            "force_eps": 5.0,
        },
    )
    straight_line_motion = RewTerm(
        func=mdp.straight_line_motion_penalty,
        weight=-1.00,
        params={"command_name": "base_velocity", "cmd_x_threshold": 0.10, "yaw_rate_scale": 0.75},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.70,
        params={"asset_cfg": FOOT_ASSET_CFG, "sensor_cfg": FOOT_SENSOR_CFG},
    )
    feet_anti_slip = RewTerm(
        func=mdp.feet_anti_slip,
        weight=-0.55,
        params={
            "asset_cfg": FOOT_ASSET_CFG,
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "slip_ratio_coef": 0.25,
        },
    )
    slip_under_command = RewTerm(
        func=mdp.slip_under_command,
        weight=-0.60,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "force_threshold": 5.0,
            "soft_scale": 0.5,
            "cmd_scale": 1.0,
        },
    )
    feet_force_rate = RewTerm(
        func=mdp.feet_force_rate,
        weight=-0.005,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "force_delta_clip": 150.0,
            "force_scale": 100.0,
        },
    )
    feet_motion_when_idle = RewTerm(
        func=mdp.feet_motion_when_idle,
        weight=-0.40,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "asset_cfg": FOOT_ASSET_CFG,
            "command_name": "base_velocity",
            "cmd_threshold": 0.10,
        },
    )
    gait = RewTerm(
        func=mdp.traction_adaptive_feet_gait,
        weight=0.20,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "slow_period": 0.85,
            "fast_period": 0.50,
            "stance_threshold": 0.55,
            "low_speed": 0.20,
            "high_speed": 1.50,
            "mu_midpoint": 0.55,
            "mu_width": 0.14,
        },
    )


@configclass
class RobotFootTractionAdaptiveEnvCfg(RobotFootEnvCfg):
    """Production candidate for traction-conditioned straight locomotion."""

    observations: ObservationsCfg = FootTractionAdaptiveObservationsCfg()
    commands: CommandsCfg = TractionAdaptiveCommandsCfg()
    rewards: RewardsCfg = FootTractionAdaptiveRewardsCfg()
    events: EventCfg = TractionAdaptiveEventCfg()
    curriculum: CurriculumCfg = TractionAdaptiveCurriculumCfg()


@configclass
class RobotFootTractionAdaptivePlayEnvCfg(RobotFootTractionAdaptiveEnvCfg):
    """Fixed high command for visual high-μ/low-μ saturation checks."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = mdp.TractionAdaptiveVelocityCommandCfg.Ranges(
            lin_vel_x=(1.3, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.commands.base_velocity.high_speed_fraction = 1.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Privileged traction teacher: flat ground, coherent μ and balanced commands
# ---------------------------------------------------------------------------


@configclass
class TractionTeacherCommandsCfg(CommandsCfg):
    """25% low/high high-speed, 30% medium normal, 20% special commands."""

    base_velocity = mdp.TractionTeacherVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.0,
        rel_spin_envs=0.0,
        min_spin_ang_vel=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.TractionTeacherVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.30, 1.50),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.TractionTeacherVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.30, 1.50),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        high_speed_range=(1.0, 1.5),
        normal_speed_range=(0.30, 1.0),
        low_speed_range=(0.05, 0.30),
        reverse_speed_range=(-0.30, -0.05),
        mid_normal_fraction=0.60,
        special_stop_fraction=0.40,
    )


@configclass
class HallSafetyEnvelopeCommandsCfg(CommandsCfg):
    """Deploy-safe stop/crawl/cruise commands for Hall-only PPO hardening.

    Command samples are deliberately independent of the hidden ground
    material.  The 1864-D actor must therefore derive friction adaptation from
    Hall Bx/By/Bz histories, packet health and proprioception, rather than
    learning a command-to-friction shortcut.  Contact forces, true ``mu`` and
    slip remain unavailable to the actor.
    """

    base_velocity = mdp.HallSafetyEnvelopeVelocityCommandCfg(
        asset_name="robot",
        # Re-sample inside one rollout so deceleration, crawl and re-accel are
        # learned transitions rather than reset-only behaviours.
        resampling_time_range=(2.5, 5.0),
        rel_standing_envs=0.0,
        rel_spin_envs=0.0,
        min_spin_ang_vel=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.HallSafetyEnvelopeVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.90),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.HallSafetyEnvelopeVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.90),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        stop_fraction=0.18,
        crawl_fraction=0.32,
        crawl_speed_range=(0.20, 0.35),
        cruise_speed_range=(0.45, 0.90),
        # Compatibility fields for inherited switch configuration.  The
        # HallSafetyEnvelope command does not read them for sampling.
        high_speed_range=(0.45, 0.90),
        high_speed_regimes=(0, 2),
    )


@configclass
class HallZeroFallEnvelopeCommandsCfg(CommandsCfg):
    """Safety-recovery command distribution centred on the acceptance speed.

    The first command-envelope pass proved that crawl and stop must be part of
    the actor's training support, but its broad 0.45--0.90 m/s cruise range
    under-sampled the fixed 0.8 m/s acceptance request.  This recovery stage
    keeps those safety commands while concentrating normal cruise samples near
    the requested real-robot walking band.  Sampling remains independent of
    hidden friction; the actor still has to use Hall/proprioceptive evidence
    to choose a stable response to material changes.
    """

    base_velocity = mdp.HallSafetyEnvelopeVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.75, 3.50),
        rel_standing_envs=0.0,
        rel_spin_envs=0.0,
        min_spin_ang_vel=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.HallSafetyEnvelopeVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.86),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.HallSafetyEnvelopeVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.86),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        stop_fraction=0.12,
        crawl_fraction=0.26,
        crawl_speed_range=(0.20, 0.35),
        cruise_speed_range=(0.70, 0.85),
        # Compatibility only; HallSafetyEnvelopeVelocityCommand samples from
        # cruise_speed_range, not from a hidden friction regime.
        high_speed_range=(0.70, 0.85),
        high_speed_regimes=(0, 2),
    )


@configclass
class TractionTeacherEventCfg(EventCfg):
    """Only coherent teacher friction is randomized in the first stage."""

    physics_material = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.05, 0.25), (0.25, 0.75), (0.75, 1.20)),
            "regime_probabilities": (0.25, 0.50, 0.25),
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.05, 0.25), (0.25, 0.75), (0.75, 1.20)),
            "regime_probabilities": (0.25, 0.50, 0.25),
        },
    )
    foot_sensor_reset = EventTerm(func=mdp.reset_foot_sensor_valid, mode="reset", params={})
    foot_sensor_dropout = None
    add_base_mass = None
    push_robot = None


@configclass
class FootTractionTeacherObservationsCfg(FootTractionAdaptiveObservationsCfg):
    """Append one privileged effective-μ scalar after the old 640-D actor prefix."""

    @configclass
    class PolicyCfg(FootTractionAdaptiveObservationsCfg.PolicyCfg):
        effective_friction_mu = ObsTerm(
            func=mdp.ground_friction_mu,
            params={"default_mu": 0.5, "clip_max": 1.20},
            clip=(0.0, 1.20),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class FootTractionMotionTeacherObservationsCfg(
    FootTractionTeacherObservationsCfg
):
    """Keep 641 dimensions while adding closed-loop lateral motion feedback."""

    @configclass
    class PolicyCfg(FootTractionTeacherObservationsCfg.PolicyCfg):
        # Replace the Teacher's constant 5-D validity + 5-D age histories with
        # five frames of [body vy, relative heading error].  The field override
        # deliberately preserves the original column position (630:640).
        foot_sensor_valid = ObsTerm(
            func=mdp.lateral_motion_feedback,
            params={
                "asset_name": "robot",
                "lateral_velocity_clip": 1.5,
                "heading_error_clip": 1.0,
            },
            clip=(-1.5, 1.5),
            history_length=5,
        )
        foot_sensor_age = None

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobotFootTractionTeacherEnvCfg(RobotFootTractionAdaptiveEnvCfg):
    """Stage-1 teacher that isolates the requested μ-conditioned speed map."""

    observations: ObservationsCfg = FootTractionTeacherObservationsCfg()
    commands: CommandsCfg = TractionTeacherCommandsCfg()
    events: EventCfg = TractionTeacherEventCfg()
    curriculum: CurriculumCfg = TractionAdaptiveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False


@configclass
class RobotFootTractionTeacherPlayEnvCfg(RobotFootTractionTeacherEnvCfg):
    """Small deterministic flat-ground teacher evaluation environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Deployable traction student: no true mu, structured sensor-domain noise
# ---------------------------------------------------------------------------


@configclass
class TractionStudentEventCfg(TractionTeacherEventCfg):
    """Balanced coherent friction plus episode/time-correlated sensor errors."""

    structured_foot_sensor = EventTerm(
        func=mdp.randomize_structured_foot_sensor,
        mode="reset",
        params={
            "gain_range": (0.80, 1.20),
            "normal_bias_range": (-0.05, 0.05),
            "tangent_bias_range": (-0.03, 0.03),
            "lowpass_alpha_range": (0.25, 1.00),
            "delay_steps_range": (0, 3),
            "sample_dropout_prob_range": (0.0, 0.05),
            "burst_dropout_prob_range": (0.0, 0.02),
            "burst_length_range": (2, 10),
        },
    )


@configclass
class FootTractionStudentObservationsCfg(FootTractionAdaptiveObservationsCfg):
    """640-D deploy schema with stateful foot-sensor domain randomization."""

    @configclass
    class PolicyCfg(FootTractionAdaptiveObservationsCfg.PolicyCfg):
        # Attribute names intentionally remain deploy-compatible.  Only their
        # Isaac-side functions change to the structured sensor model.
        foot_contact = ObsTerm(
            func=mdp.structured_foot_contact,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 5.0, "soft": True},
            clip=(0.0, 1.0),
            history_length=15,
        )
        foot_normal_force = ObsTerm(
            func=mdp.structured_foot_normal_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            clip=(0.0, 5.0),
            history_length=15,
        )
        foot_tangent_force = ObsTerm(
            func=mdp.structured_foot_tangent_force,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
            clip=(0.0, 5.0),
            history_length=15,
        )
        foot_friction_ratio = ObsTerm(
            func=mdp.structured_foot_friction_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 5.0, "clip_max": 2.0},
            clip=(0.0, 2.0),
            history_length=15,
        )
        foot_load_ratio = ObsTerm(
            func=mdp.structured_foot_load_ratio,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "eps": 5.0},
            clip=(0.0, 1.0),
            history_length=15,
        )
        foot_sensor_valid = ObsTerm(
            func=mdp.structured_foot_sensor_valid,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "default_valid": 1.0},
            clip=(0.0, 1.0),
            history_length=5,
        )
        foot_sensor_age = ObsTerm(
            func=mdp.structured_foot_sensor_age,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "age_scale": 0.25, "clip_max": 1.0},
            clip=(0.0, 1.0),
            history_length=5,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobotFootTractionStudentEnvCfg(RobotFootTractionAdaptiveEnvCfg):
    """Flat stage-2 student; true mu stays critic-only and never enters actor."""

    observations: ObservationsCfg = FootTractionStudentObservationsCfg()
    commands: CommandsCfg = TractionTeacherCommandsCfg()
    events: EventCfg = TractionStudentEventCfg()
    curriculum: CurriculumCfg = TractionAdaptiveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class RobotFootTractionStudentPlayEnvCfg(RobotFootTractionStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class FootTractionNoisyTeacherObservationsCfg(FootTractionStudentObservationsCfg):
    """Teacher-compatible 641-D input with noisy 640-D deployable prefix."""

    @configclass
    class PolicyCfg(FootTractionStudentObservationsCfg.PolicyCfg):
        effective_friction_mu = ObsTerm(
            func=mdp.ground_friction_mu,
            params={"default_mu": 0.5, "clip_max": 1.20},
            clip=(0.0, 1.20),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobotFootTractionNoisyTeacherEnvCfg(RobotFootTractionTeacherEnvCfg):
    """Data-collection task: frozen Teacher actions under structured sensor DR."""

    observations: ObservationsCfg = FootTractionNoisyTeacherObservationsCfg()
    events: EventCfg = TractionStudentEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Base IMU/joint corruption plus the stateful foot model.  True mu is
        # kept exact so it remains a clean supervised label/Teacher input.
        self.observations.policy.enable_corruption = True


@configclass
class RobotFootTractionNoisyTeacherPlayEnvCfg(RobotFootTractionNoisyTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Sim2Sim-robust privileged teacher: same 641-D interface, broader dynamics
# ---------------------------------------------------------------------------


@configclass
class TractionRobustActionsCfg(ActionsCfg):
    """Apply episode-randomized 0--2 control-step latency to policy actions."""

    JointPositionAction = mdp.RandomDelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        min_delay=0,
        max_delay=2,
    )


@configclass
class TractionRobustStabilityActionsCfg(TractionRobustActionsCfg):
    """Oversample the diagnosed 40-ms action-latency failure tail."""

    JointPositionAction = mdp.RandomDelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        min_delay=0,
        max_delay=2,
        delay_probabilities=(0.15, 0.15, 0.70),
    )


@configclass
class TractionRobustTeacherEventCfg(TractionStudentEventCfg):
    """Conservative dynamics and sensor DR for Teacher Sim2Sim transfer.

    Friction remains coherent (static == dynamic == effective_mu), so the
    final actor observation is still the exact physical label.  The low-grip
    stratum is increased from 25% to 30% to focus the rare low-mu/high-command
    fall seen in the final clean-Teacher matrix.
    """

    physics_material = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.08),
            "num_buckets": 64,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.05, 0.25), (0.25, 0.75), (0.75, 1.20)),
            "regime_probabilities": (0.30, 0.45, 0.25),
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.05, 1.20),
            "dynamic_friction_range": (0.05, 1.20),
            "restitution_range": (0.0, 0.08),
            "num_buckets": 64,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.05, 0.25), (0.25, 0.75), (0.75, 1.20)),
            "regime_probabilities": (0.30, 0.45, 0.25),
        },
    )

    # Mass scaling recomputes each link's inertia tensor from its default.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.015, 0.015),
                "y": (-0.015, 0.015),
                "z": (-0.010, 0.010),
            },
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
        },
    )
    motor_strength = EventTerm(
        func=mdp.randomize_motor_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.85, 1.15),
        },
    )
    joint_dynamics = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.80, 1.20),
            "armature_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )


@configclass
class FootTractionRobustStabilityRewardsCfg(FootTractionAdaptiveRewardsCfg):
    """Second-pass shaping for the rare randomized-domain fall tail.

    The first robust pass inherited the local baseline reward, which has an
    alive bonus but no explicit non-timeout termination cost.  Isaac Lab's G1
    locomotion configuration uses a -200 termination term; retaining that
    convention makes a rare fall materially more expensive than a small
    velocity-tracking gain.  The mild roll/pitch terms provide a dense warning
    before the discrete termination signal without changing the traction speed
    target.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-12.0, params={"target_height": 0.78})


@configclass
class RobotFootTractionRobustTeacherEnvCfg(RobotFootTractionNoisyTeacherEnvCfg):
    """641-D Oracle Teacher for flat-ground Sim2Sim robustness fine-tuning."""

    actions: ActionsCfg = TractionRobustActionsCfg()
    events: EventCfg = TractionRobustTeacherEventCfg()


@configclass
class RobotFootTractionRobustStabilityTeacherEnvCfg(RobotFootTractionRobustTeacherEnvCfg):
    """Targeted tail-risk pass with balanced low/high-friction stress cases."""

    actions: ActionsCfg = TractionRobustStabilityActionsCfg()
    rewards: RewardsCfg = FootTractionRobustStabilityRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Low- and high-friction regimes always receive high-speed requests.
        # Raising both tails from 55% to 60% keeps low-mu braking practice while
        # addressing the newly exposed high-mu/high-speed fall tail.
        probabilities = (0.30, 0.40, 0.30)
        self.events.physics_material.params["regime_probabilities"] = probabilities
        self.events.physics_material_reset.params["regime_probabilities"] = probabilities
        # The matrix exposed a narrow 1.0-m/s transient that the previous
        # 1.0--1.5 high-speed sampler almost never visited.  Include the
        # neighbourhood below 1.0 during this tail-risk pass without reducing
        # the number of low/high-friction stress samples.
        self.commands.base_velocity.high_speed_range = (0.85, 1.50)
        # Preserve the μ≈0.8 high-speed gait needed for MuJoCo transfer, while
        # selecting a stable fast walk at the μ=1.2 upper boundary where the
        # full-speed policy's rare failures are forward-pitch events.
        safe_high_speed = 1.50
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["high_speed"] = safe_high_speed
            term.params["very_high_speed"] = 1.10
            term.params["very_high_mu_midpoint"] = 1.00
            term.params["very_high_mu_width"] = 0.06


@configclass
class RobotFootTractionRobustTeacherPlayEnvCfg(RobotFootTractionRobustTeacherEnvCfg):
    """Small robust-domain matrix-evaluation environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionRobustStabilityTeacherPlayEnvCfg(
    RobotFootTractionRobustStabilityTeacherEnvCfg
):
    """Evaluation variant of the targeted stability pass."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Stable-baseline continuation: strengthen the mu=0.70--0.95 speed shoulder
# ---------------------------------------------------------------------------


@configclass
class RobotFootTractionRobustShoulderTeacherEnvCfg(
    RobotFootTractionRobustStabilityTeacherEnvCfg
):
    """Short low-LR pass that improves the conservative mid/high-grip shoulder.

    The selected model_7750 remains untouched.  This branch increases the
    frequency of mu=0.70--0.95 + high-command episodes while retaining low-mu
    braking, ordinary commands, the very-high-mu tail and all Sim2Sim DR.
    """

    def __post_init__(self):
        super().__post_init__()
        ranges = (
            (0.05, 0.25),  # low grip: keep active braking under large commands
            (0.25, 0.70),  # ordinary walking / stop / reverse retention
            (0.70, 0.95),  # target shoulder
            (0.95, 1.20),  # extreme high-grip stability tail
        )
        probabilities = (0.20, 0.25, 0.35, 0.20)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["teacher_friction_ranges"] = ranges
            event.params["regime_probabilities"] = probabilities
        self.commands.base_velocity.high_speed_regimes = (0, 2, 3)
        self.commands.base_velocity.high_speed_range = (1.00, 1.50)

        # A continuous target: fast around mu=0.8, smoothly settling to the
        # already-validated 1.1-m/s safe gait at the mu=1.2 endpoint.
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.20
            term.params["high_speed"] = 1.50
            term.params["mu_midpoint"] = 0.55
            term.params["mu_width"] = 0.14
            term.params["very_high_speed"] = 1.10
            term.params["very_high_mu_midpoint"] = 1.00
            term.params["very_high_mu_width"] = 0.06
        self.rewards.track_lin_vel_xy.weight = 2.20
        self.rewards.termination_penalty.weight = -250.0
        self.rewards.flat_orientation_l2.weight = -12.0


@configclass
class RobotFootTractionRobustShoulderTeacherPlayEnvCfg(
    RobotFootTractionRobustShoulderTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Shoulder candidate recovery: retain mu=0.8 speed, harden high-mu transients
# ---------------------------------------------------------------------------


@configclass
class RobotFootTractionRobustShoulderRecoveryTeacherEnvCfg(
    RobotFootTractionRobustShoulderTeacherEnvCfg
):
    """Very-low-LR recovery pass for high-grip, high-command pitch failures.

    Independent 128-environment tests showed that the first shoulder candidate
    retained the desired mu=0.8 speed but could pitch over shortly after an
    abrupt 1.5-m/s request at mu=1.2.  This configuration keeps the shoulder
    samples and the complete Sim2Sim randomization, while deliberately making
    the extreme-high-friction transient the largest training stratum.
    """

    def __post_init__(self):
        super().__post_init__()
        probabilities = (0.15, 0.15, 0.25, 0.45)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities

        # Do not remove abrupt high commands: those are the actual failing
        # cases.  A 1.0-m/s high-mu target still clears the >=0.8-m/s transfer
        # gate, while leaving the mu=0.70--0.95 target at 1.5 m/s.
        self.commands.base_velocity.high_speed_range = (0.85, 1.50)
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["very_high_speed"] = 1.00

        # Dense pitch/roll warnings must dominate the small extra tracking
        # return before the discrete bad-orientation termination is reached.
        self.rewards.termination_penalty.weight = -350.0
        self.rewards.flat_orientation_l2.weight = -18.0
        self.rewards.base_angular_velocity.weight = -0.25
        self.rewards.base_height.weight = -15.0
        self.rewards.action_rate.weight = -0.07


@configclass
class RobotFootTractionRobustShoulderRecoveryTeacherPlayEnvCfg(
    RobotFootTractionRobustShoulderRecoveryTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionRobustShoulderGuardTeacherEnvCfg(
    RobotFootTractionRobustShoulderRecoveryTeacherEnvCfg
):
    """Final high-mu guard for the capped 1.0-m/s deployment envelope."""

    def __post_init__(self):
        super().__post_init__()
        probabilities = (0.15, 0.10, 0.20, 0.55)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities

        # The real-robot command envelope is capped at 1.0 m/s.  Concentrate
        # high-friction transient practice around that boundary instead of
        # spending the majority of this short guard pass above deploy speed.
        self.commands.base_velocity.high_speed_range = (0.85, 1.05)
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["very_high_speed"] = 0.90
            term.params["very_high_mu_midpoint"] = 0.96
            term.params["very_high_mu_width"] = 0.04
        # The failing rollouts overshot to roughly 2 m/s during the command
        # ramp.  Make that high-mu overshoot more expensive than a marginal
        # tracking improvement; the mu=0.8 cap remains 1.5 m/s.
        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.traction_overspeed.weight = -5.0
        self.rewards.termination_penalty.weight = -400.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.base_angular_velocity.weight = -0.30


@configclass
class RobotFootTractionRobustShoulderGuardTeacherPlayEnvCfg(
    RobotFootTractionRobustShoulderGuardTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionRobustLowMuRecoveryTeacherEnvCfg(
    RobotFootTractionRobustShoulderRecoveryTeacherEnvCfg
):
    """Harden the delayed low-friction/high-command tail without dropping the shoulder."""

    def __post_init__(self):
        super().__post_init__()
        # The strict ramped matrix exposed a single low-mu failure with both
        # action and foot-sensor delay at their maximum.  Restore low grip as
        # the largest stratum, but keep enough shoulder/high-grip samples to
        # protect the speed behavior already demonstrated in MuJoCo.
        probabilities = (0.40, 0.10, 0.25, 0.25)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities

        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.18
        self.rewards.traction_overspeed.weight = -3.0
        self.rewards.termination_penalty.weight = -400.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.base_angular_velocity.weight = -0.30
        self.rewards.action_rate.weight = -0.08


@configclass
class RobotFootTractionRobustLowMuRecoveryTeacherPlayEnvCfg(
    RobotFootTractionRobustLowMuRecoveryTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Straight-path continuation: remove inherited lateral drift at <=1.0 m/s
# ---------------------------------------------------------------------------


@configclass
class FootTractionLateralGuardRewardsCfg(FootTractionRobustStabilityRewardsCfg):
    """Preserve traction behavior while making straight-path error expensive."""

    straight_line_motion = RewTerm(
        func=mdp.straight_line_motion_penalty,
        weight=-6.0,
        params={
            "command_name": "base_velocity",
            "cmd_x_threshold": 0.08,
            "yaw_rate_scale": 1.25,
            "lateral_clip": 1.0,
            "yaw_clip": 1.5,
        },
    )
    straight_cross_track = RewTerm(
        func=mdp.straight_cross_track_error,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "cmd_x_threshold": 0.08,
            "error_clip": 0.75,
        },
    )


@configclass
class RobotFootTractionLateralGuardTeacherEnvCfg(
    RobotFootTractionRobustShoulderGuardTeacherEnvCfg
):
    """Low-LR continuation of the safe 8030 Teacher for straight tracking."""

    rewards: RewardsCfg = FootTractionLateralGuardRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Retain all traction regimes rather than optimizing only high grip.
        probabilities = (0.25, 0.20, 0.25, 0.30)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities
        self.commands.base_velocity.high_speed_range = (0.85, 1.05)
        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.traction_overspeed.weight = -5.0
        self.rewards.straight_line_motion.weight = -6.0
        self.rewards.termination_penalty.weight = -400.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.base_angular_velocity.weight = -0.30
        self.rewards.action_rate.weight = -0.08


@configclass
class RobotFootTractionLateralGuardTeacherPlayEnvCfg(
    RobotFootTractionLateralGuardTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Privileged terrain Teacher: ramps/stairs are simulation-only supervision
# ---------------------------------------------------------------------------


@configclass
class RobotFootTractionSlopeStairsTeacherEnvCfg(
    RobotFootTractionLateralGuardTeacherEnvCfg
):
    """Curriculum Teacher for the five-family Hall ramp/stair terrain.

    The 641-D actor keeps the existing privileged contact/friction stream and
    is used only to generate actions for distillation.  The deployed magnetic
    Student remains 1864-D and never receives terrain identity, contact force,
    terrain height or friction.
    """

    curriculum: CurriculumCfg = TractionAdaptiveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The inherited Teacher deliberately forces a plane.  Restore the
        # dedicated terrain generator here and begin at gentle rows; the
        # existing distance curriculum promotes successful environments.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = (
            HALL_SLOPE_STAIRS_TERRAINS_CFG.replace()
        )
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.scene.height_scanner.mesh_prim_paths = [
            "/World/ground/terrain/mesh"
        ]
        # Avoid an abrupt 1.05 m/s restart while the old flat policy is first
        # learning slope posture and stair clearance.  The final Student
        # governor still owns friction-adaptive speed at deployment time.
        self.commands.base_velocity.high_speed_range = (0.45, 0.85)


@configclass
class RobotFootTractionSlopeStairsTeacherPlayEnvCfg(
    RobotFootTractionSlopeStairsTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 25
        self.scene.terrain.max_init_terrain_level = (
            self.scene.terrain.terrain_generator.num_rows - 1
        )
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.34, 0.27, 0.16),
            roughness=0.88,
            opacity=1.0,
        )
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Joint speed/path continuation: recover 1.0-m/s high-grip tracking
# ---------------------------------------------------------------------------


@configclass
class FootTractionSpeedLateralRewardsCfg(FootTractionLateralGuardRewardsCfg):
    """Jointly penalize accumulated side drift and high-grip underspeed."""

    high_traction_underspeed = RewTerm(
        func=mdp.high_traction_underspeed_penalty,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "target_speed": 1.0,
            "mu_midpoint": 0.78,
            "mu_width": 0.08,
            "command_midpoint": 0.70,
            "command_width": 0.08,
            "tolerance": 0.03,
            "error_clip": 1.0,
        },
    )


@configclass
class RobotFootTractionSpeedLateralTeacherEnvCfg(
    RobotFootTractionLateralGuardTeacherEnvCfg
):
    """Low-LR Oracle continuation for straight, accurate 1.0-m/s tracking."""

    rewards: RewardsCfg = FootTractionSpeedLateralRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the low-grip braking task while showing high-grip/high-command
        # cases often enough to overcome the inherited conservative gait.
        probabilities = (0.25, 0.15, 0.25, 0.35)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities
        self.commands.base_velocity.high_speed_range = (0.75, 1.05)
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["high_speed"] = 1.05
            term.params["very_high_speed"] = 1.00
            term.params["very_high_mu_midpoint"] = 0.95
            term.params["very_high_mu_width"] = 0.05
        self.rewards.track_lin_vel_xy.weight = 2.8
        self.rewards.track_lin_vel_xy.params["std"] = 0.30
        self.rewards.traction_overspeed.weight = -3.0
        self.rewards.straight_line_motion.weight = -5.0
        self.rewards.straight_cross_track.weight = -4.0
        self.rewards.termination_penalty.weight = -450.0
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.base_angular_velocity.weight = -0.30
        self.rewards.action_rate.weight = -0.08


@configclass
class RobotFootTractionSpeedLateralTeacherPlayEnvCfg(
    RobotFootTractionSpeedLateralTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Speed/path V2: stronger learned correction for persistent Sim2Sim side drift
# ---------------------------------------------------------------------------


@configclass
class RobotFootTractionSpeedLateralV2TeacherEnvCfg(
    RobotFootTractionSpeedLateralTeacherEnvCfg
):
    """Oracle continuation emphasizing straight high-grip trajectories."""

    def __post_init__(self):
        super().__post_init__()
        probabilities = (0.25, 0.15, 0.20, 0.40)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["regime_probabilities"] = probabilities
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.high_traction_underspeed.weight = -5.0
        self.rewards.straight_line_motion.weight = -8.0
        self.rewards.straight_line_motion.params["yaw_rate_scale"] = 1.5
        self.rewards.straight_cross_track.weight = -8.0
        self.rewards.straight_cross_track.params["error_clip"] = 1.0
        self.rewards.base_angular_velocity.weight = -0.45
        self.rewards.flat_orientation_l2.weight = -22.0
        self.rewards.termination_penalty.weight = -500.0


@configclass
class RobotFootTractionSpeedLateralV2TeacherPlayEnvCfg(
    RobotFootTractionSpeedLateralV2TeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class TractionTwoSurfaceSwitchEventCfg(TractionRobustTeacherEventCfg):
    """Real2Sim-oriented low/high traction alternation under one command.

    Half of the environments start on each surface.  Every four seconds all
    environments flip to the opposite regime, so every batch contains both
    high->low braking and low->high recovery without a CPU material update on
    every policy step.
    """

    physics_material = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": False,
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": False,
        },
    )
    friction_switch = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": True,
        },
    )


@configclass
class RobotFootTractionMotionTeacherEnvCfg(
    RobotFootTractionSpeedLateralV2TeacherEnvCfg
):
    """641-D Oracle Teacher with deployable straight-path feedback channels."""

    observations: ObservationsCfg = FootTractionMotionTeacherObservationsCfg()


@configclass
class FootTractionMotionSwitchRewardsCfg(FootTractionSpeedLateralRewardsCfg):
    """Transition shaping with an explicit low-traction cadence cost."""

    low_traction_touchdown_rate = RewTerm(
        func=mdp.low_traction_touchdown_rate_penalty,
        weight=-12.0,
        params={
            "sensor_cfg": FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "mu_midpoint": 0.30,
            "mu_width": 0.06,
            "minimum_air_time": 0.08,
            "command_threshold": 0.10,
        },
    )


@configclass
class RobotFootTractionMotionSwitchTeacherEnvCfg(
    RobotFootTractionMotionTeacherEnvCfg
):
    """Oracle transition Teacher: fixed command, alternating traction."""

    events: EventCfg = TractionTwoSurfaceSwitchEventCfg()
    rewards: RewardsCfg = FootTractionMotionSwitchRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The command is sampled once at reset and remains identical across
        # every material transition.  Both low and high regimes are assigned
        # the same forward-command distribution by TractionTeacherVelocityCommand.
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.high_speed_regimes = (0, 2)
        self.commands.base_velocity.high_speed_range = (0.75, 1.00)
        # Make the requested slow, short low-traction gait explicit.  The
        # original weak cue produced short but faster contact cycling, so this
        # continuation gives cadence enough weight to be visible while safety
        # and speed shaping remain dominant.
        self.rewards.gait.weight = 4.00
        self.rewards.gait.params["slow_period"] = 1.20
        self.rewards.gait.params["fast_period"] = 0.60
        self.rewards.gait.params["low_speed"] = 0.15
        self.rewards.gait.params["high_speed"] = 1.00
        self.rewards.track_lin_vel_xy.weight = 3.50
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_lin_vel_xy.params["low_speed"] = 0.15
        self.rewards.track_lin_vel_xy.params["high_speed"] = 1.00
        self.rewards.traction_overspeed.weight = -8.0
        self.rewards.traction_overspeed.params["low_speed"] = 0.15
        self.rewards.traction_overspeed.params["high_speed"] = 1.00
        self.rewards.feet_slide.weight = -1.00
        self.rewards.feet_anti_slip.weight = -0.80
        self.rewards.termination_penalty.weight = -800.0
        self.rewards.high_traction_underspeed.weight = -7.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 1.00


@configclass
class RobotFootTractionMotionTeacherPlayEnvCfg(
    RobotFootTractionMotionTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMotionSwitchTeacherPlayEnvCfg(
    RobotFootTractionMotionSwitchTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class RobotFootTractionMotionStressTeacherEnvCfg(
    RobotFootTractionMotionTeacherEnvCfg
):
    """Short continuation that makes 1.0--1.5 m/s requests in-distribution."""

    def __post_init__(self):
        super().__post_init__()
        # Preserve the safe traction-dependent target cap (about 1.0 m/s at
        # extreme high grip), but expose the policy to the full joystick stress
        # range so a 1.5-m/s request is handled by saturation rather than OOD
        # extrapolation.
        self.commands.base_velocity.high_speed_range = (1.0, 1.5)
        self.rewards.termination_penalty.weight = -550.0
        self.rewards.flat_orientation_l2.weight = -24.0


@configclass
class RobotFootTractionMotionStressTeacherPlayEnvCfg(
    RobotFootTractionMotionStressTeacherEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Final deploy observation: shared dual-foot 15xXYZ magnetic history
# ---------------------------------------------------------------------------


@configclass
class TractionMagneticStudentEventCfg(TractionRobustTeacherEventCfg):
    hall_foot_sensor_reset = EventTerm(
        func=mdp.reset_hall_foot_sensor,
        mode="reset",
    )
    magnetic_array_proxy = EventTerm(
        func=mdp.randomize_magnetic_array_proxy,
        mode="reset",
        params={
            "sensor_gain_range": (0.72, 1.28),
            "axis_gain_range": (0.75, 1.25),
            "zero_residual_std": 0.06,
            "dead_channel_prob": 0.015,
            "foot_dropout_prob": 0.02,
            "period_range_s": (0.018, 0.048),
        },
    )


@configclass
class TractionMagneticSwitchEventCfg(TractionMagneticStudentEventCfg):
    """Magnetic Student sensor DR plus the same two-surface transition."""

    physics_material = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": False,
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": False,
        },
    )
    friction_switch = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": True,
        },
    )


@configclass
class TractionMagneticSwitchTrainingEventCfg(TractionMagneticSwitchEventCfg):
    """Asynchronous two-surface curriculum for Hall-only PPO training.

    The evaluation task intentionally switches every environment at the same
    instant so phase plots are directly comparable.  Training with that
    global clock would leave a spurious time correlation.  This variant
    samples each environment's switch time independently; the deployable
    actor must therefore respond to its own Hall/proprioceptive history rather
    than episode time or a shared simulator event.
    """

    friction_switch = EventTerm(
        func=mdp.two_surface_friction_with_buffer,
        mode="interval",
        interval_range_s=(2.5, 5.5),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.08, 1.20),
            "dynamic_friction_range": (0.08, 1.20),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
            "make_consistent": True,
            "low_friction_range": (0.08, 0.20),
            "high_friction_range": (0.80, 1.20),
            "initial_high_probability": 0.50,
            "flip_existing": True,
        },
    )


@configclass
class FootTractionMagneticObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=5,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=5,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=5,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=5)
        foot_magnetic_array = ObsTerm(
            func=mdp.hall_magnetic_array,
            params={
                "hall_cfg": HallFootSensorCfg(),
                "asset_cfg": HALL_FOOT_ASSET_CFG,
                "contact_sensor_cfg": FOOT_SENSOR_CFG,
                "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
                "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
            },
            clip=(-6.0, 6.0),
            history_length=15,
        )
        foot_sample_period_lr = ObsTerm(
            func=mdp.hall_sample_period_lr,
            params={
                "hall_cfg": HallFootSensorCfg(),
                "asset_cfg": HALL_FOOT_ASSET_CFG,
                "contact_sensor_cfg": FOOT_SENSOR_CFG,
                "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
                "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
            },
            clip=(0.001, 0.25),
            history_length=15,
        )
        foot_sensor_valid_lr = ObsTerm(
            func=mdp.hall_sensor_valid_lr,
            params={
                "hall_cfg": HallFootSensorCfg(),
                "asset_cfg": HALL_FOOT_ASSET_CFG,
                "contact_sensor_cfg": FOOT_SENSOR_CFG,
                "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
                "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
            },
            clip=(0.0, 1.0),
        )
        foot_sensor_age_lr = ObsTerm(
            func=mdp.hall_sensor_age_lr,
            params={
                "hall_cfg": HallFootSensorCfg(),
                "asset_cfg": HALL_FOOT_ASSET_CFG,
                "contact_sensor_cfg": FOOT_SENSOR_CFG,
                "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
                "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
                "age_scale": 0.25,
            },
            clip=(0.0, 1.0),
        )
        # Optional comparison channel configured by the environment.  It is
        # absent by default so the audited 1864-D magnetic actor is unchanged.
        foot_contact_force = None

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: FootTractionAdaptiveObservationsCfg.CriticCfg = (
        FootTractionAdaptiveObservationsCfg.CriticCfg()
    )
    # Evaluation-only DAgger target stream.  It is never consumed or exported
    # by the magnetic actor, but lets a frozen 641-D Teacher label the exact
    # on-policy magnetic trajectories instead of a separately randomized proxy.
    teacher: FootTractionTeacherObservationsCfg.PolicyCfg = (
        FootTractionTeacherObservationsCfg.PolicyCfg()
    )


@configclass
class RobotFootTractionMagneticStudentEnvCfg(
    RobotFootTractionLateralGuardTeacherEnvCfg
):
    """Magnetic walking task with contact/hall/both output selection.

    ``sensor_output_mode='hall'`` preserves the deployed 1864-D actor schema.
    ``contact`` and ``both`` intentionally change policy dimensions and are for
    ablation/calibration runs, not direct loading of an 1864-D checkpoint.
    """

    sensor_output_mode: str = "hall"
    use_legacy_magnetic_proxy: bool = False
    hall_sensor_cfg: HallFootSensorCfg = HallFootSensorCfg(
        enable_domain_randomization=True
    )

    scene: RobotSceneCfg = HallFootSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = FootTractionMagneticObservationsCfg()
    events: EventCfg = TractionMagneticStudentEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.sensor_output_mode not in ("contact", "hall", "both"):
            raise ValueError("sensor_output_mode must be 'contact', 'hall', or 'both'")
        if self.hall_sensor_cfg.num_hall_sensors != 15:
            raise ValueError(
                "this deployed magnetic task requires 15 Hall sites; other counts are supported by HallFootSensor "
                "but need a matching observation/network schema"
            )
        self.scene.left_hall_contact.update_period = self.sim.dt
        self.scene.right_hall_contact.update_period = self.sim.dt

        if self.use_legacy_magnetic_proxy:
            self.events.hall_foot_sensor_reset = None
            self.scene.left_hall_contact = None
            self.scene.right_hall_contact = None
            self.observations.policy.foot_magnetic_array.func = mdp.magnetic_array_proxy
            self.observations.policy.foot_magnetic_array.params = {"sensor_cfg": FOOT_SENSOR_CFG}
            self.observations.policy.foot_sample_period_lr.func = mdp.magnetic_sample_period_lr
            self.observations.policy.foot_sample_period_lr.params = {"sensor_cfg": FOOT_SENSOR_CFG}
            self.observations.policy.foot_sensor_valid_lr.func = mdp.magnetic_sensor_valid_lr
            self.observations.policy.foot_sensor_valid_lr.params = {"sensor_cfg": FOOT_SENSOR_CFG}
            # The Motion variant intentionally reuses these last two actor
            # slots for proprioceptive lateral-motion feedback.  Replace only
            # the ordinary Hall age term, not that schema-preserving override.
            if self.observations.policy.foot_sensor_age_lr.func is mdp.hall_sensor_age_lr:
                self.observations.policy.foot_sensor_age_lr.func = mdp.magnetic_sensor_age_lr
                self.observations.policy.foot_sensor_age_lr.params = {
                    "sensor_cfg": FOOT_SENSOR_CFG,
                    "age_scale": 0.25,
                }
        else:
            # The legacy proxy remains available, but its episode-level DR
            # buffers are unnecessary for the physical Hall path.
            self.events.magnetic_array_proxy = None

        if self.sensor_output_mode in ("contact", "both"):
            self.observations.policy.foot_contact_force = ObsTerm(
                func=mdp.foot_force_vector,
                params={"sensor_cfg": FOOT_SENSOR_CFG, "scale": 0.01},
                clip=(-6.0, 6.0),
                history_length=15,
            )
        if self.sensor_output_mode == "contact":
            self.events.hall_foot_sensor_reset = None
            self.scene.left_hall_contact = None
            self.scene.right_hall_contact = None
            self.observations.policy.foot_magnetic_array = None
            self.observations.policy.foot_sample_period_lr = None
            self.observations.policy.foot_sensor_valid_lr = None
            self.observations.policy.foot_sensor_age_lr = None
        self.observations.policy.enable_corruption = True
        if not self.use_legacy_magnetic_proxy and self.sensor_output_mode in ("hall", "both"):
            sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)


@configclass
class RobotFootTractionMagneticStudentPlayEnvCfg(
    RobotFootTractionMagneticStudentEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        # Deterministic nominal Hall mechanics for reproducible selection.
        # Robustness matrices explicitly re-enable DR/fault profiles instead
        # of silently changing every evaluation rollout.
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMagneticSlopeStairsEnvCfg(
    RobotFootTractionMagneticStudentEnvCfg
):
    """Hall-only magnetic-foot training task on ramps and stairs.

    The actor schema remains exactly 1864 -> 29.  Terrain identity, height
    scans, contact force and ground friction remain critic/mechanics data and
    never enter the deployable policy observation.
    """

    scene: RobotSceneCfg = HallSlopeStairsSceneCfg(num_envs=2048, env_spacing=2.5)
    curriculum: CurriculumCfg = TractionAdaptiveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The inherited traction task deliberately selects a flat plane.  This
        # dedicated subclass restores generator terrain after that safety
        # default and updates the filtered Hall contact paths atomically.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = HALL_SLOPE_STAIRS_TERRAINS_CFG.replace()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.scene.left_hall_contact.filter_prim_paths_expr = list(
            HALL_GENERATOR_GROUND_FILTER
        )
        self.scene.right_hall_contact.filter_prim_paths_expr = list(
            HALL_GENERATOR_GROUND_FILTER
        )
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain/mesh"]


@configclass
class RobotFootTractionMagneticSlopeStairsPlayEnvCfg(
    RobotFootTractionMagneticSlopeStairsEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        # Five environments map one-to-one to flat, uphill, downhill,
        # ascending stairs and descending stairs for visual comparison.
        self.scene.num_envs = 5
        self.scene.terrain.max_init_terrain_level = (
            self.scene.terrain.terrain_generator.num_rows - 1
        )
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        # Opaque earth tone keeps stair edges readable and avoids the
        # transparent-layer ambiguity seen in the earlier sole demo.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.34, 0.27, 0.16),
            roughness=0.88,
            opacity=1.0,
        )
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 5
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class TractionMagneticSlopeStairsFrictionEventCfg(
    TractionMagneticStudentEventCfg
):
    """Coherent friction randomization mu in [0.2, 1.0] for ramps/stairs.

    The high-friction stratum (0.75, 1.0], which contains the 0.8 reference
    surface, is slightly over-weighted so the policy keeps a solid nominal
    walking baseline while still seeing 0.2-level slip on slopes and steps.
    """

    physics_material = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 48,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.2, 0.45), (0.45, 0.75), (0.75, 1.0)),
            "regime_probabilities": (0.25, 0.40, 0.35),
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_teacher_friction_with_buffer,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 48,
            "make_consistent": True,
            "teacher_friction_ranges": ((0.2, 0.45), (0.45, 0.75), (0.75, 1.0)),
            "regime_probabilities": (0.25, 0.40, 0.35),
        },
    )
    # AnchoredPPO needs the privileged H/L stage even without spatial patches.
    # These terms run after the friction terms within startup/reset mode, so
    # the regime buffer they read is already the current episode's mu.  The
    # top friction stratum (mu in (0.75, 1.0], containing the 0.8 reference)
    # keeps the frozen-Teacher anchor; lower-mu rows stay free to adapt.
    spatial_stage_sync = EventTerm(
        func=mdp.sync_uniform_friction_course_stage,
        mode="startup",
        params={"high_regime_indices": (2,)},
    )
    spatial_stage_reset = EventTerm(
        func=mdp.sync_uniform_friction_course_stage,
        mode="reset",
        params={"high_regime_indices": (2,)},
    )


@configclass
class RobotFootTractionMagneticMotionSlopeStairsEnvCfg(
    RobotFootTractionMagneticSlopeStairsEnvCfg
):
    """Ramps/stairs training task with the deployable 1864-D motion ABI.

    This is a separate model family from ``transition_retention_r5``: the
    actor schema is identical (1864 -> 29, foot Hall + [body_vy,
    relative_heading]) but the terrain is generator ramps/stairs and the
    training/checkpoint identity is its own.
    """

    # TEMP-FIX proprio480 experiment (2026-08-14): the referenced class is
    # defined below this class in the file, so a module-level default raises
    # NameError at import.  Resolve it in __post_init__ instead; the final
    # configuration value is identical.  Revert when the class ordering is
    # fixed by the owning workflow.
    observations: ObservationsCfg = FootTractionMagneticObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = FootTractionMagneticMotionObservationsCfg()
        self.events = TractionMagneticSlopeStairsFrictionEventCfg()


@configclass
class RobotFootTractionMagneticMotionSlopeStairsPlayEnvCfg(
    RobotFootTractionMagneticMotionSlopeStairsEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        # Five environments map one-to-one to flat, uphill, downhill,
        # ascending stairs and descending stairs for visual comparison.
        self.scene.num_envs = 5
        self.scene.terrain.max_init_terrain_level = (
            self.scene.terrain.terrain_generator.num_rows - 1
        )
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.34, 0.27, 0.16),
            roughness=0.88,
            opacity=1.0,
        )
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 5
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMagneticStudentDeformableEnvCfg(
    RobotFootTractionMagneticStudentEnvCfg
):
    """Scheme-B geometry-gated audit task with a deforming magnetized TPU layer.

    The rigid robot foot and rigid PCB enclosure are represented by the ankle
    link attachment boundary.  There is no connector layer.  Only
    ``left/right_magnetized_tpu`` are PhysX deformable bodies.  This task is
    intentionally small because Isaac Sim 5.1 / Isaac Lab 2.3.2 deformables
    require ``replicate_physics=False``.  The strict cooked-thickness check is
    intentionally enabled: use Scheme A for PPO until an explicit 10 mm volume
    mesh passes the Scheme-B platen validation, then enable residual fine-tuning.
    """

    hall_sensor_cfg: HallFootSensorCfg = HallFootSensorCfg(
        implementation_mode="deformable",
        enable_domain_randomization=True,
    )
    scene: RobotSceneCfg = HallFootDeformableSceneCfg(
        num_envs=16,
        env_spacing=2.5,
        replicate_physics=False,
    )
    actions: ActionsCfg = HallDeformableActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.hall_sensor_cfg.implementation_mode != "deformable":
            raise ValueError("Scheme-B task requires implementation_mode='deformable'")
        if not Path(MAGNETIZED_TPU_USD).is_file():
            raise FileNotFoundError(
                f"magnetized TPU USD is missing: {MAGNETIZED_TPU_USD}; run scripts/assets/export_tpu_sole_mesh.py "
                "and Isaac Lab scripts/tools/convert_mesh.py"
            )
        self.scene.replicate_physics = False
        self.actions.hall_tpu_attachment.hall_cfg = self.hall_sensor_cfg

        # A smaller physics step is used for the 10 mm, 1.7 MPa layer while
        # preserving the original 50 Hz policy period (0.0025 * 8 = 0.02 s).
        self.sim.dt = 0.0025
        self.decimation = 8
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.left_hall_contact.update_period = self.sim.dt
        self.scene.right_hall_contact.update_period = self.sim.dt

        for asset_cfg in (
            self.scene.left_magnetized_tpu,
            self.scene.right_magnetized_tpu,
        ):
            props = asset_cfg.spawn.deformable_props
            material = asset_cfg.spawn.physics_material
            props.self_collision = self.hall_sensor_cfg.tpu_self_collision
            props.solver_position_iteration_count = (
                self.hall_sensor_cfg.tpu_solver_position_iteration_count
            )
            props.simulation_hexahedral_resolution = (
                self.hall_sensor_cfg.tpu_simulation_hexahedral_resolution
            )
            props.contact_offset = self.hall_sensor_cfg.tpu_contact_offset
            props.rest_offset = self.hall_sensor_cfg.tpu_rest_offset
            material.density = self.hall_sensor_cfg.tpu_density
            material.dynamic_friction = self.hall_sensor_cfg.tpu_dynamic_friction
            material.youngs_modulus = self.hall_sensor_cfg.tpu_youngs_modulus
            material.poissons_ratio = self.hall_sensor_cfg.tpu_poisson_ratio
            material.elasticity_damping = self.hall_sensor_cfg.tpu_damping


@configclass
class RobotFootTractionMagneticStudentDeformablePlayEnvCfg(
    RobotFootTractionMagneticStudentDeformableEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.hall_sensor_cfg.enable_domain_randomization = False
        # Viewing only: Isaac Sim 5.1's automatic hexahedral cooker currently
        # inflates the thin CAD volume.  The adapter emits a warning and all
        # resulting deformation values must be treated as diagnostic.
        self.hall_sensor_cfg.deformable_strict_geometry_check = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 2
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class FootTractionMagneticMotionObservationsCfg(
    FootTractionMagneticObservationsCfg
):
    """1864-D deploy schema with current motion feedback in the final two slots.

    The final layout remains 480 + 1350 + 30 + 4.  The four trailing channels
    are now [valid_left, valid_right, body_vy, relative_heading], so the
    1864->shared-foot-encoder->548 actor contract is unchanged.
    """

    @configclass
    class PolicyCfg(FootTractionMagneticObservationsCfg.PolicyCfg):
        foot_sensor_age_lr = ObsTerm(
            func=mdp.lateral_motion_feedback,
            params={
                "asset_name": "robot",
                "lateral_velocity_clip": 1.5,
                "heading_error_clip": 1.0,
            },
            clip=(-1.5, 1.5),
        )

    policy: PolicyCfg = PolicyCfg()
    teacher: FootTractionMotionTeacherObservationsCfg.PolicyCfg = (
        FootTractionMotionTeacherObservationsCfg.PolicyCfg()
    )


@configclass
class FootTractionHighSpeedBackbone482ObservationsCfg(
    FootTractionMagneticMotionObservationsCfg
):
    """Keep the full Hall ABI and add an isolated 482-D high-speed group.

    ``high_speed_policy`` is exactly the legacy 480-D proprioceptive history
    followed by current ``[body_vy, relative_heading]``.  The complete
    ``policy[1864]`` group remains available for the Hall traction branch and
    for paired evaluator diagnostics; no force/contact/mu/slip/stage term is
    present in the 482-D actor group.
    """

    @configclass
    class HighSpeedPolicyCfg(FootTractionMagneticMotionObservationsCfg.PolicyCfg):
        foot_magnetic_array = None
        foot_sample_period_lr = None
        foot_sensor_valid_lr = None

    high_speed_policy: HighSpeedPolicyCfg = HighSpeedPolicyCfg()


@configclass
class RobotFootTractionMagneticMotionStudentEnvCfg(
    RobotFootTractionMagneticStudentEnvCfg
):
    observations: ObservationsCfg = FootTractionMagneticMotionObservationsCfg()


@configclass
class RobotFootTractionMagneticMotionSwitchStudentEnvCfg(
    RobotFootTractionMagneticMotionStudentEnvCfg
):
    """Deployable observation task for switch DAgger collection/evaluation."""

    events: EventCfg = TractionMagneticSwitchEventCfg()
    # Contact/material values below are reward-only simulator signals.  The
    # PolicyCfg remains the Hall/proprioception-only 1864-D deploy interface.
    rewards: RewardsCfg = FootTractionMotionSwitchRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.high_speed_regimes = (0, 2)
        # Keep the requested nominal speed inside the proven 0.8 m/s
        # evaluation envelope.  High traction still asks for a substantially
        # faster gait than low traction, while avoiding a speed-only solution
        # that looks good in high-mu plots but is unsafe at the next switch.
        self.commands.base_velocity.high_speed_range = (0.70, 0.90)
        self.rewards.gait.weight = 4.00
        self.rewards.gait.params["slow_period"] = 1.20
        self.rewards.gait.params["fast_period"] = 0.60
        self.rewards.gait.params["low_speed"] = 0.15
        self.rewards.gait.params["high_speed"] = 0.90
        self.rewards.track_lin_vel_xy.weight = 3.50
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_lin_vel_xy.params["low_speed"] = 0.15
        self.rewards.track_lin_vel_xy.params["high_speed"] = 0.90
        self.rewards.traction_overspeed.weight = -8.0
        self.rewards.traction_overspeed.params["low_speed"] = 0.15
        self.rewards.traction_overspeed.params["high_speed"] = 0.90
        self.rewards.feet_slide.weight = -1.00
        self.rewards.feet_anti_slip.weight = -0.80
        self.rewards.termination_penalty.weight = -800.0
        self.rewards.high_traction_underspeed.weight = -7.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 0.90


@configclass
class RobotFootTractionMagneticMotionSwitchTrainEnvCfg(
    RobotFootTractionMagneticMotionSwitchStudentEnvCfg
):
    """PPO-only variant with per-environment asynchronous friction changes."""

    events: EventCfg = TractionMagneticSwitchTrainingEventCfg()


@configclass
class RobotFootTractionMagneticMotionSwitchWarmupEnvCfg(
    RobotFootTractionMagneticMotionSwitchTrainEnvCfg
):
    """Safety curriculum before exposing the Hall actor to μ=0.08.

    This is not an easier evaluation: it is a short learning stage that lets
    the warm-started gait associate measured magnetic temporal changes with a
    slower, longer-period gait before the extreme low-friction interval is
    introduced.  The final candidate is always retrained/evaluated on the
    full 0.08--0.20 regime in ``...SwitchTrain``.
    """

    def __post_init__(self):
        super().__post_init__()
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
            self.events.friction_switch,
        ):
            event.params["low_friction_range"] = (0.22, 0.35)
        self.events.friction_switch.interval_range_s = (3.0, 5.5)


@configclass
class RobotFootTractionMagneticMotionSwitchBridgeEnvCfg(
    RobotFootTractionMagneticMotionSwitchTrainEnvCfg
):
    """Intermediate low-traction stage between warmup and μ=0.08 training."""

    def __post_init__(self):
        super().__post_init__()
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
            self.events.friction_switch,
        ):
            event.params["low_friction_range"] = (0.14, 0.26)
        self.events.friction_switch.interval_range_s = (2.8, 5.0)


@configclass
class RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg(
    RobotFootTractionMagneticMotionSwitchBridgeEnvCfg
):
    """Safety-only continuation for intermittent Hall/BLE faults.

    This stage deliberately remains inside the already demonstrated
    ``mu >= 0.14`` bridge curriculum.  Its purpose is not to claim that a
    disconnected sole can walk on arbitrarily poor ground: it makes the
    deployed 1864-D Hall/proprio actor retain the conservative bridge gait
    when a complete foot stream, individual channels, or delayed packets are
    missing.  The actor still receives only Bx/By/Bz histories, packet health
    and proprioception.  Contact force, friction and slip remain critic/reward
    signals in Isaac only.

    The raised fault rates are intentionally harsher than the normal
    evaluation profile.  A candidate must subsequently pass the unchanged
    nominal and default-fault zero-fall gates; training under a harsher
    profile is not itself evidence of safety.
    """

    def __post_init__(self):
        super().__post_init__()
        # Whole-foot loss represents the independent left/right BLE streams.
        # The remaining variation covers a failed Hall IC and a delayed packet
        # without inventing a Hall-to-force conversion.
        self.hall_sensor_cfg.foot_dropout_probability = 0.10
        self.hall_sensor_cfg.dead_channel_probability = 0.08
        self.hall_sensor_cfg.maximum_packet_delay_steps = 5
        self.hall_sensor_cfg.observation_zero_residual_std = 0.10
        self.hall_sensor_cfg.observation_cross_axis_std = 0.10
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)

        # During safety hardening it is preferable to retain a slower but
        # stable gait rather than trade a terminal event for a marginal
        # high-traction tracking gain.  The later selection gate still checks
        # high/low speed separation, so this does not silently become a
        # stand-still solution.
        self.rewards.termination_penalty.weight = -1400.0
        self.rewards.traction_overspeed.weight = -10.0
        self.rewards.high_traction_underspeed.weight = -4.0


@configclass
class RobotFootTractionMagneticMotionSwitchCommandEnvelopeEnvCfg(
    RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg
):
    """Final PPO stage before enabling a Hall-only speed safety governor.

    The previous fault-hardening actor had only a fixed 0.7--0.9 m/s request
    during switch training.  A runtime governor that correctly asked it to
    crawl or stop therefore produced an out-of-distribution command and was
    unsafe.  This stage keeps the same ``mu >= 0.14`` fault-hardening physics,
    but learns the full stop/crawl/cruise envelope using commands independent
    of material friction.  It preserves the exact 1864-D deploy actor schema.
    """

    commands: CommandsCfg = HallSafetyEnvelopeCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # RobotFootTractionMagneticMotionSwitchStudentEnvCfg writes these
        # compatibility values during its post-init.  Keep the actual safety
        # envelope explicit after that inherited configuration step.
        self.commands.base_velocity.resampling_time_range = (2.5, 5.0)
        self.commands.base_velocity.stop_fraction = 0.18
        self.commands.base_velocity.crawl_fraction = 0.32
        self.commands.base_velocity.crawl_speed_range = (0.20, 0.35)
        self.commands.base_velocity.cruise_speed_range = (0.45, 0.90)

        # Make falls dominate any transient tracking reward.  This does not
        # use contact/force/mu as an actor input; it only changes the Isaac
        # training objective and selection pressure.
        self.rewards.termination_penalty.weight = -1800.0
        self.rewards.flat_orientation_l2.weight = -24.0
        self.rewards.base_angular_velocity.weight = -0.35


@configclass
class RobotFootTractionMagneticMotionSwitchZeroFallRecoveryEnvCfg(
    RobotFootTractionMagneticMotionSwitchCommandEnvelopeEnvCfg
):
    """Targeted recovery for the measured high/low switch fall tail.

    This is a safety continuation, not a claim that the training log is a
    release test.  It replays short, asynchronous friction changes and
    frequent 0.8 m/s-neighbourhood commands while retaining true stops/crawls
    for the future Hall-risk speed governor.  All contact/friction/slip values
    used by the reward stay critic/simulator-only; the actor interface remains
    the exact 1864-D Hall/proprioception schema.
    """

    commands: CommandsCfg = HallZeroFallEnvelopeCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.75, 3.50)
        self.commands.base_velocity.stop_fraction = 0.12
        self.commands.base_velocity.crawl_fraction = 0.26
        self.commands.base_velocity.crawl_speed_range = (0.20, 0.35)
        self.commands.base_velocity.cruise_speed_range = (0.70, 0.85)

        # Explicitly bias PPO against the observed rare pitch/roll tail.  The
        # terminal term dominates a marginal command-tracking improvement,
        # while the high-traction tracking term remains active so the policy
        # cannot trivially solve the objective by always standing still.
        self.rewards.termination_penalty.weight = -3000.0
        self.rewards.flat_orientation_l2.weight = -30.0
        self.rewards.base_height.weight = -20.0
        self.rewards.base_angular_velocity.weight = -0.50
        self.rewards.action_rate.weight = -0.10
        self.rewards.high_traction_underspeed.weight = -5.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 0.80

        # In the failed campaign the phase transitions were 4 s long.  More
        # frequent, independently timed transitions prevent the actor from
        # relying on a fixed gait clock and create many more recovery samples.
        self.events.friction_switch.interval_range_s = (1.75, 3.50)


@configclass
class TractionMagneticLowGripRecoveryEventCfg(TractionMagneticSwitchTrainingEventCfg):
    """Low-grip-only dynamics for the runtime Hall recovery action policy.

    This is deliberately a separate policy-training distribution.  The
    resulting actor is never responsible for high-traction speed tracking: a
    Hall-only risk state machine selects it only after a causal prospective
    slip signal has entered LOW.  Keeping the high-speed gait out of this
    objective prevents the zero-fall recovery loss from degrading the audited
    nominal fast-walk actor.

    The low/high parameters remain present because the shared material event
    validates both ranges, but startup/reset always choose the low interval
    and there is no in-episode material switch.  Ground friction stays an
    Isaac-only physics/reward quantity and is not added to the 1864-D actor
    observation.
    """

    def __post_init__(self):
        super().__post_init__()
        for event in (self.physics_material, self.physics_material_reset):
            event.params["low_friction_range"] = (0.14, 0.20)
            event.params["high_friction_range"] = (0.80, 1.20)
            event.params["initial_high_probability"] = 0.0
            event.params["flip_existing"] = False


@configclass
class TractionMagneticLowGripHandoffEventCfg(
    TractionMagneticLowGripRecoveryEventCfg
):
    """Low-grip recovery plus asynchronous high-momentum handoff states."""

    # Isaac Lab 2.3.2's current event adds this velocity to the live root
    # state.  Because it fires during a gait, Hall/proprio/action histories are
    # retained, unlike a reset-only impulse.  The perturbation is training
    # physics and never becomes an actor observation.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (0.35, 0.65),
                "y": (-0.12, 0.12),
                "roll": (-0.15, 0.15),
                "pitch": (-0.30, 0.30),
                "yaw": (-0.15, 0.15),
            },
        },
    )


@configclass
class TractionMagneticSpatialFrictionEventCfg(TractionMagneticStudentEventCfg):
    """Material-scale DR and privileged labels for physical floor patches."""

    physics_material = EventTerm(
        func=mdp.randomize_coherent_material_scale,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 0.16 * [0.875, 1.25] = the Stage7 low-mu range [0.14, 0.20].
            # The same multiplier gives high-mu [0.7875, 1.125].
            "scale_range": (0.875, 1.25),
            "restitution_range": (0.0, 0.03),
        },
    )
    physics_material_reset = EventTerm(
        func=mdp.randomize_coherent_material_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scale_range": (0.875, 1.25),
            "restitution_range": (0.0, 0.03),
        },
    )
    spatial_friction_reset = EventTerm(
        func=mdp.update_spatial_friction_buffer,
        mode="reset",
        params={
            "low_patch_mu": 0.16,
            "high_patch_mu": 0.90,
            "contact_force_threshold": 5.0,
            "control_dt": 0.02,
            "capture_target_speed": 0.24,
            "capture_speed_tolerance": 0.05,
            "capture_stable_time_s": 0.12,
            "capture_deadline_s": 0.90,
            "asset_cfg": SceneEntityCfg("robot"),
            "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
            "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
        },
    )
    spatial_friction_update = EventTerm(
        func=mdp.update_spatial_friction_buffer,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "low_patch_mu": 0.16,
            "high_patch_mu": 0.90,
            "contact_force_threshold": 5.0,
            "control_dt": 0.02,
            "capture_target_speed": 0.24,
            "capture_speed_tolerance": 0.05,
            "capture_stable_time_s": 0.12,
            "capture_deadline_s": 0.90,
            "asset_cfg": SceneEntityCfg("robot"),
            "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
            "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
        },
    )


@configclass
class FootTractionSpatialCaptureRewardsCfg(FootTractionMotionSwitchRewardsCfg):
    """Reward-only first-contact capture objective for the physical H-L-H course."""

    spatial_capture_envelope = RewTerm(
        func=mdp.spatial_low_capture_envelope_penalty,
        weight=-12.0,
        params={
            "target_speed": 0.24,
            "deadline_s": 0.90,
            "tolerance": 0.03,
            "decay_power": 1.0,
            "excess_clip": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    spatial_capture_success = RewTerm(
        func=mdp.spatial_low_capture_reward,
        weight=3.0,
        params={
            "target_speed": 0.24,
            "speed_tolerance": 0.05,
            "deadline_s": 0.90,
            "progress_bonus": 0.50,
            "hold_bonus": 1.00,
            "timely_completion_bonus": 12.0,
            "late_completion_bonus": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FootTractionCadenceStrideRewardsCfg(FootTractionSpatialCaptureRewardsCfg):
    """Requested-speed objective with cadence and step length left unconstrained.

    For walking, average forward speed is approximately cadence times step
    length.  This reward intentionally specifies neither factor and contains no
    LOW-stage speed cap.  PPO may therefore increase cadence while shortening
    steps on the medium-friction patch, provided true contact-point slip,
    impact, posture, lateral drift and action-slew costs remain acceptable.
    """

    # Symmetric command tracking: exceeding the requested velocity is not
    # rewarded.  The identical command remains active in HIGH_START, LOW and
    # HIGH_END, so stage identity cannot change the target.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=8.0,
        params={
            "std": 0.22,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Contact-point velocity includes foot angular velocity and the lever arm
    # to the actual filtered patch.  This replaces ankle-origin slip proxies.
    contact_point_slip = RewTerm(
        func=mdp.contact_point_tangential_slip_penalty,
        weight=-6.0,
        params={
            "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
            "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
            "asset_cfg": HALL_FOOT_ASSET_CFG,
            "min_normal_force_n": 5.0,
            "speed_deadband_m_s": 0.025,
            "speed_clip_m_s": 1.5,
            "squared": True,
        },
    )

    # Isaac Lab exposes normal and tangential contact forces in separate
    # buffers.  Use the dedicated, floor-filtered Hall contact sensors instead
    # of the legacy normal-only net_forces_w approximation.
    friction_cone_margin = RewTerm(
        func=mdp.filtered_contact_friction_cone_margin_penalty,
        weight=-0.55,
        params={
            "left_contact_sensor_cfg": HALL_LEFT_CONTACT_CFG,
            "right_contact_sensor_cfg": HALL_RIGHT_CONTACT_CFG,
            "safe_utilization": 0.75,
            "force_threshold": 5.0,
            "force_eps": 5.0,
        },
    )

    # Directly close the long-horizon failure mode that yaw-rate and
    # cross-track costs cannot identify: a persistent accumulated heading
    # error after the LOW-to-HIGH handoff.  This is gait-agnostic and uses no
    # force, contact, friction, or course-stage input.
    straight_heading_error = RewTerm(
        func=mdp.straight_heading_error_penalty,
        weight=-18.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_x_threshold": 0.10,
            "yaw_command_threshold": 0.05,
            "error_clip": 1.0,
        },
    )

    # Transition-retention terms are inert at zero weight in every legacy
    # task.  Only the isolated transition-retention environment raises them,
    # and their stage weighting keeps HIGH_START untouched.
    transition_heading_retention = RewTerm(
        func=mdp.transition_heading_retention_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "cmd_x_threshold": 0.10,
            "yaw_command_threshold": 0.05,
            "low_weight": 1.0,
            "high_start_weight": 0.0,
            "high_end_peak_weight": 1.0,
            "high_end_decay_s": 3.0,
        },
    )
    transition_vy_retention = RewTerm(
        func=mdp.transition_vy_retention_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_x_threshold": 0.10,
            "low_weight": 1.0,
            "high_start_weight": 0.0,
            "high_end_peak_weight": 1.0,
            "high_end_decay_s": 3.0,
            "lateral_clip": 1.5,
        },
    )
    low_stage_yaw_rate = RewTerm(
        func=mdp.low_stage_yaw_rate_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_rate_clip": 2.0,
        },
    )
    low_stage_leg_symmetry = RewTerm(
        func=mdp.low_stage_leg_symmetry_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    low_entry_heading_change = RewTerm(
        func=mdp.low_entry_heading_change_penalty,
        weight=0.0,
        params={"error_clip": 0.5},
    )
    windowed_vy = RewTerm(
        func=mdp.windowed_vy_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_x_threshold": 0.10,
            "window_steps": 50,
            "lateral_clip": 1.5,
        },
    )

@configclass
class SpatialFrictionTerminationsCfg(TerminationsCfg):
    """Ordinary fall terms plus successful finite-course truncation."""

    course_success = DoneTerm(
        func=mdp.spatial_friction_course_success,
        params={
            "minimum_local_x": 2.60,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        # Curriculum completion is not a physical failure.  Isaac Lab keeps
        # timeout terms out of mdp.is_terminated and its -3500 penalty.
        time_out=True,
    )


@configclass
class RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg(
    RobotFootTractionMagneticMotionSwitchZeroFallRecoveryEnvCfg
):
    """Hall/proprio-only low-traction recovery policy for a two-policy guard.

    The policy sees the same 1864-D deployment schema as the fast actor and
    is trained on stop/crawl/cruise commands so a runtime governor may enter
    it without an out-of-distribution command history.  It is intentionally
    optimized only for ``mu in [0.14, 0.20]``; high-grip tracking remains the
    job of the fast baseline.  This separation is a safety design choice, not
    a hidden friction input: action selection at deployment is based only on
    the independently validated Hall/proprio future-slip risk score.
    """

    events: EventCfg = TractionMagneticLowGripRecoveryEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Be explicit after inherited switch-task post-init methods.  A
        # recovery trajectory must not receive a surprise high-grip flip;
        # high/low switching is tested at the controller integration layer.
        self.events.friction_switch = None
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["low_friction_range"] = (0.14, 0.20)
            event.params["high_friction_range"] = (0.80, 1.20)
            event.params["initial_high_probability"] = 0.0
            event.params["flip_existing"] = False

        # Retain command-envelope support for the governor's crawl/stop
        # output, while keeping enough 0.8-m/s requests for the recovery
        # actor to learn a controlled deceleration when it takes over from
        # the fast actor.
        self.commands.base_velocity.resampling_time_range = (1.50, 3.00)
        self.commands.base_velocity.stop_fraction = 0.10
        self.commands.base_velocity.crawl_fraction = 0.30
        self.commands.base_velocity.crawl_speed_range = (0.10, 0.30)
        self.commands.base_velocity.cruise_speed_range = (0.70, 0.85)

        # Low grip has a deliberately small forward target.  These are
        # simulator-only rewards; the actor continues to receive only Hall
        # Bx/By/Bz histories, packet validity/timing and proprioception.
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.16
            term.params["high_speed"] = 0.16
        self.rewards.track_lin_vel_xy.weight = 4.50
        self.rewards.track_lin_vel_xy.params["std"] = 0.18
        self.rewards.traction_overspeed.weight = -18.0
        self.rewards.gait.weight = 4.50
        self.rewards.gait.params["slow_period"] = 1.25
        self.rewards.gait.params["fast_period"] = 1.25
        self.rewards.feet_slide.weight = -1.40
        self.rewards.feet_anti_slip.weight = -1.20
        self.rewards.slip_under_command.weight = -1.20
        self.rewards.low_traction_touchdown_rate.weight = -16.0
        self.rewards.termination_penalty.weight = -4000.0
        self.rewards.flat_orientation_l2.weight = -35.0
        self.rewards.base_height.weight = -25.0
        self.rewards.base_angular_velocity.weight = -0.60
        self.rewards.action_rate.weight = -0.12


@configclass
class RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg(
    RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg
):
    """Stage7: capture-step recovery from a live high-speed handoff.

    The low-friction material remains fixed for the whole episode.  Random
    root-velocity increments create the missing 0.6--0.8 m/s takeover states
    while retaining Hall, last-action and proprioceptive histories.  No
    impulse magnitude, contact force, slip or friction label enters PolicyCfg.
    """

    events: EventCfg = TractionMagneticLowGripHandoffEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.friction_switch = None

        # Align world +X with body-forward so the reset momentum has a clear
        # physical meaning.  These values are root linear/angular velocities,
        # not actor observations or synthetic force channels.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "yaw": (-0.10, 0.10),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.25, 0.55),
            "y": (-0.08, 0.08),
            "z": (0.0, 0.0),
            "roll": (-0.12, 0.12),
            "pitch": (-0.20, 0.20),
            "yaw": (-0.10, 0.10),
        }

        # A fixed 1.25-s cadence and a large touchdown penalty prevented the
        # emergency step observed in failed handoffs.  Keep weak gait shaping
        # but let PPO discover an asynchronous capture step.
        self.rewards.gait.weight = 1.25
        self.rewards.low_traction_touchdown_rate.weight = -3.0
        self.rewards.track_lin_vel_xy.weight = 4.50
        self.rewards.traction_overspeed.weight = -24.0
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_anti_slip.weight = -1.5
        self.rewards.slip_under_command.weight = -1.5
        self.rewards.termination_penalty.weight = -5000.0
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.base_height.weight = -28.0
        self.rewards.base_angular_velocity.weight = -0.80
        self.rewards.action_rate.weight = -0.09


@configclass
class RobotFootTractionMagneticMotionLowGripHandoffMildEnvCfg(
    RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg
):
    """Stage7A curriculum: moderate grip and moderate takeover momentum."""

    def __post_init__(self):
        super().__post_init__()
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["low_friction_range"] = (0.18, 0.26)
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.10, 0.30),
                "y": (-0.05, 0.05),
                "roll": (-0.08, 0.08),
                "pitch": (-0.12, 0.12),
                "yaw": (-0.08, 0.08),
            }
        )
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.20, 0.40),
            "y": (-0.08, 0.08),
            "roll": (-0.10, 0.10),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.10, 0.10),
        }


@configclass
class RobotFootTractionMagneticMotionLowGripHandoffExtremeEnvCfg(
    RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg
):
    """Stage7C curriculum: lower-mu tail and 0.75-m/s momentum increments."""

    def __post_init__(self):
        super().__post_init__()
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["low_friction_range"] = (0.12, 0.20)
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (0.35, 0.65),
                "y": (-0.10, 0.10),
                "roll": (-0.16, 0.16),
                "pitch": (-0.28, 0.28),
                "yaw": (-0.14, 0.14),
            }
        )
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.45, 0.75),
            "y": (-0.15, 0.15),
            "roll": (-0.20, 0.20),
            "pitch": (-0.40, 0.40),
            "yaw": (-0.20, 0.20),
        }


@configclass
class RobotFootTractionMagneticMotionLowGripHandoffHighCommandEnvCfg(
    RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg
):
    """Stage7 high-command expert for a real fast-to-low-grip handoff.

    Unlike the steady low-grip expert, the requested command remains 0.80 m/s
    while the privileged low-grip reward target is 0.24 m/s.  This trains the
    actor with the same command/history distribution it receives when the
    frozen fast actor hands over on a real walk; friction remains physics-only.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.stop_fraction = 0.0
        self.commands.base_velocity.crawl_fraction = 0.0
        self.commands.base_velocity.cruise_speed_range = (0.80, 0.80)
        self.commands.base_velocity.high_speed_range = (0.80, 0.80)
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.24
            term.params["high_speed"] = 0.80
        self.rewards.track_lin_vel_xy.weight = 4.5
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.traction_overspeed.weight = -24.0
        self.rewards.gait.weight = 1.25
        self.rewards.low_traction_touchdown_rate.weight = -3.0
        self.rewards.feet_slide.weight = -2.0
        self.rewards.feet_anti_slip.weight = -1.5
        self.rewards.slip_under_command.weight = -1.5
        self.rewards.termination_penalty.weight = -5000.0
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.base_angular_velocity.weight = -0.80
        self.rewards.action_rate.weight = -0.09


@configclass
class RobotFootTractionMagneticMotionLowGripHandoffRecoveryPlayEnvCfg(
    RobotFootTractionMagneticMotionLowGripHandoffRecoveryEnvCfg
):
    """Small deterministic Hall inspection task retaining handoff impulses."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionEnvCfg(
    RobotFootTractionMagneticMotionSwitchStudentEnvCfg
):
    """Hall-only PPO on physical blue--orange--blue floor patches.

    The external command remains independent of patch identity.  A leading
    foot therefore has to contact the orange patch before Hall/proprioception
    can causally support slowing the gait.  True material and contact values
    are confined to physics, rewards, critic labels and evaluation.
    """

    scene: RobotSceneCfg = HallSpatialFrictionSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    commands: CommandsCfg = HallZeroFallEnvelopeCommandsCfg()
    events: EventCfg = TractionMagneticSpatialFrictionEventCfg()
    rewards: RewardsCfg = FootTractionSpatialCaptureRewardsCfg()
    terminations: TerminationsCfg = SpatialFrictionTerminationsCfg()

    def __post_init__(self):
        # Parent tasks need a temporary terrain config during post-init.  The
        # scene has not been constructed yet, so removing it here prevents the
        # old plane from ever being spawned beneath the Cuboids.
        super().__post_init__()
        self.scene.terrain = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            # Give the frozen/original fast gait 2--3 s to settle before the
            # first causal low-patch contact.  The high-start patch spans
            # x=[-2, 0], so this remains safely on its physical collider.
            "x": (-1.70, -1.40),
            "y": (-0.06, 0.06),
            "yaw": (-0.06, 0.06),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.05, 0.20),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.0),
            "roll": (-0.04, 0.04),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.04, 0.04),
        }

        # A fixed high request exposes adaptation without a hidden
        # command-to-material shortcut.  Low-speed behavior is selected by
        # privileged training rewards and must be inferred from Hall history.
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.stop_fraction = 0.0
        self.commands.base_velocity.crawl_fraction = 0.0
        self.commands.base_velocity.cruise_speed_range = (0.75, 0.85)
        self.commands.base_velocity.high_speed_range = (0.75, 0.85)

        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.16
            term.params["high_speed"] = 0.80
        # The first spatial run converged to a safe 0.2 m/s crawl because the
        # survival/underspeed trade-off was too conservative.  Keep the large
        # fall penalty, but make the requested high-friction speed materially
        # valuable so the Hall actor cannot solve the course by simply walking
        # slowly everywhere.
        # Speed-recovery continuation: 6350 already passed the hardened
        # zero-fall screen, so restore pressure toward the previously proven
        # fast Hall gait only on HIGH patches.  LOW overspeed and termination
        # penalties remain unchanged; the actor still has to slow safely.
        self.rewards.track_lin_vel_xy.func = mdp.spatial_stage_track_lin_vel_x_exp
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_lin_vel_xy.params = {
            "std": 0.24,
            "command_name": "base_velocity",
            "low_speed": 0.24,
            "high_speed": 0.80,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.traction_overspeed.func = mdp.spatial_stage_overspeed_penalty
        self.rewards.traction_overspeed.weight = -20.0
        self.rewards.traction_overspeed.params = {
            "command_name": "base_velocity",
            "low_speed": 0.24,
            "high_speed": 0.80,
            "tolerance": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.gait.weight = 1.0
        self.rewards.low_traction_touchdown_rate.weight = -5.0
        self.rewards.feet_slide.weight = -1.5
        self.rewards.feet_anti_slip.weight = -1.2
        self.rewards.slip_under_command.weight = -1.2
        self.rewards.termination_penalty.weight = -3500.0
        self.rewards.flat_orientation_l2.weight = -32.0
        self.rewards.base_angular_velocity.weight = -0.60
        # Continuation tuning: the previous -40 coefficient caused PPO to
        # trade away posture safety for an abrupt speed increase.  A bounded
        # 0.60-m/s high-patch target supplies useful speed pressure while the
        # frozen 6350 gait remains the dominant safety anchor.
        self.rewards.high_traction_underspeed.weight = -12.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 0.60

        # Capture completion never terminates the episode. The robot must still
        # traverse FrictionHighEnd before the existing course-success timeout.
        self._configure_spatial_capture(
            target_speed=0.24,
            deadline_s=0.90,
            envelope_weight=-12.0,
            success_weight=3.0,
        )

    def _configure_spatial_capture(
        self,
        target_speed: float,
        deadline_s: float,
        envelope_weight: float,
        success_weight: float,
    ) -> None:
        """Keep event-state and reward targets identical across curriculum stages."""

        for event in (
            self.events.spatial_friction_reset,
            self.events.spatial_friction_update,
        ):
            event.params["capture_target_speed"] = float(target_speed)
            event.params["capture_deadline_s"] = float(deadline_s)
        self.rewards.spatial_capture_envelope.weight = float(envelope_weight)
        self.rewards.spatial_capture_envelope.params["target_speed"] = float(target_speed)
        self.rewards.spatial_capture_envelope.params["deadline_s"] = float(deadline_s)
        self.rewards.spatial_capture_success.weight = float(success_weight)
        self.rewards.spatial_capture_success.params["target_speed"] = float(target_speed)
        self.rewards.spatial_capture_success.params["deadline_s"] = float(deadline_s)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionEnvCfg
):
    """Stage-S1: preserve the 49999 fast gait across a causal mu=0.45 patch."""

    scene: RobotSceneCfg = HallSpatialMildFrictionSceneCfg(
        num_envs=4096, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.hall_sensor_cfg.enable_domain_randomization = True
        self.hall_sensor_cfg.foot_dropout_probability = 0.02
        self.hall_sensor_cfg.dead_channel_probability = 0.02
        self.hall_sensor_cfg.maximum_packet_delay_steps = 2
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )
        for event in (
            self.events.spatial_friction_reset,
            self.events.spatial_friction_update,
        ):
            event.params["low_patch_mu"] = 0.45
            event.params["high_patch_mu"] = 0.90
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.45
            term.params["high_speed"] = 0.80
        self.rewards.termination_penalty.weight = -5000.0
        self.rewards.high_traction_underspeed.weight = -10.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 0.70
        self._configure_spatial_capture(
            target_speed=0.45,
            deadline_s=1.00,
            envelope_weight=-8.0,
            success_weight=2.0,
        )


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionEnvCfg
):
    """Stage-S2: continue the fast gait across a causal mu=0.28 patch."""

    scene: RobotSceneCfg = HallSpatialMediumFrictionSceneCfg(
        num_envs=4096, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.hall_sensor_cfg.enable_domain_randomization = True
        self.hall_sensor_cfg.foot_dropout_probability = 0.05
        self.hall_sensor_cfg.dead_channel_probability = 0.04
        self.hall_sensor_cfg.maximum_packet_delay_steps = 4
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )
        for event in (
            self.events.spatial_friction_reset,
            self.events.spatial_friction_update,
        ):
            event.params["low_patch_mu"] = 0.28
            event.params["high_patch_mu"] = 0.90
        for term in (
            self.rewards.track_lin_vel_xy,
            self.rewards.traction_overspeed,
            self.rewards.gait,
        ):
            term.params["low_speed"] = 0.32
            term.params["high_speed"] = 0.80
        self.rewards.termination_penalty.weight = -5000.0
        self.rewards.high_traction_underspeed.weight = -10.0
        self.rewards.high_traction_underspeed.params["target_speed"] = 0.65
        self._configure_spatial_capture(
            target_speed=0.32,
            deadline_s=0.95,
            envelope_weight=-10.0,
            success_weight=2.5,
        )


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg
):
    """Training-only Medium course with de-synchronized causal transitions.

    All environments still start on the physical HighStart collider and LOW
    rewards still latch only after filtered LOW contact.  The three reset
    distance bands prevent a 4096-environment rollout from sharing one course
    phase.  Root x and the sampled band are privileged diagnostics, never
    policy observations.  Fair evaluation deliberately uses the ordinary
    Medium play cfg with its original far reset range.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.func = mdp.reset_root_state_spatial_stratified
        self.events.reset_base.params = {
            # Far/mid/near bands are all on HighStart (which ends at x=0).
            # The nearest reset retains >=0.45 m geometric clearance so a
            # foot cannot begin by straddling the material boundary.
            "x_bands": (
                (-1.70, -1.35),
                (-1.20, -0.85),
                (-0.75, -0.45),
            ),
            "band_probabilities": (0.25, 0.40, 0.35),
            "low_boundary_x": 0.0,
            "minimum_high_margin": 0.30,
            "pose_range": {
                "y": (-0.06, 0.06),
                "yaw": (-0.06, 0.06),
            },
            "velocity_range": {
                "x": (0.05, 0.20),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "roll": (-0.04, 0.04),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.04, 0.04),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionEnvCfg
):
    """Four-course GUI/evaluation scene with deterministic Hall mechanics."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 4
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.cruise_speed_range = (0.80, 0.80)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["scale_range"] = (1.0, 1.0)
            event.params["restitution_range"] = (0.0, 0.0)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionMildPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["scale_range"] = (1.0, 1.0)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        for event in (self.events.physics_material, self.events.physics_material_reset):
            event.params["scale_range"] = (1.0, 1.0)


@configclass
class RobotFootTractionMagneticMotionLowGripRecoveryPlayEnvCfg(
    RobotFootTractionMagneticMotionLowGripRecoveryEnvCfg
):
    """Small deterministic inspection variant for the recovery policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMagneticMotionStudentPlayEnvCfg(
    RobotFootTractionMagneticMotionStudentEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


@configclass
class RobotFootTractionMagneticMotionSwitchStudentPlayEnvCfg(
    RobotFootTractionMagneticMotionSwitchStudentEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg
):
    """Isolated Medium course that leaves cadence and stride to PPO.

    The requested forward velocity stays at 0.80 m/s across the complete
    High--Low--High traversal.  There is no material-dependent target speed,
    fixed gait period, touchdown-count penalty, or reward for exceeding the
    request.  Instead, PPO must trade cadence against step length while
    respecting true contact-point slip, impact, posture, lateral-path and
    action-slew constraints.  The 1864-D Hall/proprio actor schema is inherited
    unchanged; contact points and material state remain privileged training
    quantities only.
    """

    rewards: RewardsCfg = FootTractionCadenceStrideRewardsCfg()

    def __post_init__(self):
        # Keep the existing MediumDense physics, Hall randomization and
        # de-synchronized reset bands, then remove only the old objective's
        # prescriptive gait/speed terms.  Doing this after super() is required:
        # the legacy spatial config initializes those terms during post-init.
        super().__post_init__()

        # This isolated task consumes the raw per-patch/point contact buffers
        # and distributes them over the 15 Hall sites.  Legacy tasks retain the
        # HallFootSensorCfg default (aggregate), so the A/B is explicit.
        self.hall_sensor_cfg.contact_distribution_mode = "detailed"
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )

        # A single material-independent request prevents command or stage from
        # disclosing the floor.  The symmetric error below also prevents the
        # actor from earning extra reward by running faster than requested.
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.stop_fraction = 0.0
        self.commands.base_velocity.crawl_fraction = 0.0
        self.commands.base_velocity.cruise_speed_range = (0.80, 0.80)
        self.commands.base_velocity.high_speed_range = (0.80, 0.80)
        self.rewards.track_lin_vel_xy.func = mdp.track_lin_vel_x_exp
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_lin_vel_xy.params = {
            "std": 0.22,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        }

        # Remove every inherited term that imposes a LOW speed, cadence,
        # touchdown count, or inaccurate ankle-origin slip proxy.  In
        # particular, no 0.24/0.32 m/s target survives in this task.
        self.rewards.traction_overspeed = None
        self.rewards.gait = None
        self.rewards.low_traction_touchdown_rate = None
        self.rewards.spatial_capture_envelope = None
        self.rewards.spatial_capture_success = None
        self.rewards.high_traction_underspeed = None
        self.rewards.feet_slide = None
        self.rewards.feet_anti_slip = None
        self.rewards.slip_under_command = None

        # Safety constraints do not specify step timing.  Contact-point slip
        # includes omega x r at the actual patch, while force-rate limits hard
        # impacts without penalizing cadence itself.
        self.rewards.contact_point_slip.weight = -6.0
        self.rewards.feet_force_rate.weight = -0.012
        self.rewards.action_rate.weight = -0.09
        self.rewards.termination_penalty.weight = -5000.0
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.base_angular_velocity.weight = -0.75
        self.rewards.base_height.weight = -25.0
        self.rewards.straight_line_motion.weight = -6.0
        self.rewards.straight_line_motion.params["yaw_rate_scale"] = 1.25
        self.rewards.straight_cross_track.weight = -6.0
        self.rewards.friction_cone_margin.weight = -0.55

        # Capture buffers are retained only as privileged diagnostics and for
        # the runner's LOW/HIGH auxiliary label.  Their target now matches the
        # requested-speed retention objective and has no reward attached.
        for event in (
            self.events.spatial_friction_reset,
            self.events.spatial_friction_update,
        ):
            event.params["capture_target_speed"] = 0.80


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg
):
    """Deterministic short-course inspection for the isolated curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 6.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 4
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )
        self.events.reset_base.func = mdp.reset_root_state_spatial_stratified
        self.events.reset_base.params["x_bands"] = ((-1.70, -1.40),)
        self.events.reset_base.params["band_probabilities"] = (1.0,)
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
        ):
            event.params["scale_range"] = (1.0, 1.0)
            event.params["restitution_range"] = (0.0, 0.0)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideLongDemoEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStridePlayEnvCfg
):
    """Wide 24 m H[-6,0]--L[0,6]--H[6,18] visualization course."""

    scene: RobotSceneCfg = HallSpatialMediumLongDemoSceneCfg(
        num_envs=4, env_spacing=12.0
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 12.0
        # From x~-5.3 to the x=17.5 success line is about 22.8 m.  A 65 s
        # horizon also lets a conservative 0.4 m/s gait finish, rather than
        # turning slow-but-safe behavior into an artificial timeout.
        self.episode_length_s = 65.0
        self.events.reset_base.params["x_bands"] = ((-5.50, -5.10),)
        self.events.reset_base.params["band_probabilities"] = (1.0,)
        self.events.reset_base.params["low_boundary_x"] = 0.0
        self.events.reset_base.params["minimum_high_margin"] = 0.30
        self.terminations.course_success.params["minimum_local_x"] = 17.50
        # Frame the full 24 m course by default; command-line viewer overrides
        # remain available for recordings.
        self.viewer.eye = (11.0, -28.0, 15.0)
        self.viewer.lookat = (6.0, 0.0, 0.4)


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg
):
    """Long-tail H--L--H training with a sustained final high-grip segment."""

    scene: RobotSceneCfg = HallSpatialMediumRetentionSceneCfg(
        num_envs=512, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        self.events.reset_base.func = mdp.reset_root_state_spatial_stratified
        self.events.reset_base.params = {
            "x_bands": (
                (-2.70, -2.30),
                (-1.85, -1.45),
                (-1.00, -0.60),
            ),
            "band_probabilities": (0.35, 0.35, 0.30),
            "low_boundary_x": 0.0,
            "minimum_high_margin": 0.30,
            "pose_range": {
                "y": (-0.06, 0.06),
                "yaw": (-0.06, 0.06),
            },
            "velocity_range": {
                "x": (0.05, 0.20),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "roll": (-0.04, 0.04),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.04, 0.04),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.course_success.params["minimum_local_x"] = 9.50
        for event in (
            self.events.spatial_friction_reset,
            self.events.spatial_friction_update,
        ):
            event.params["low_patch_mu"] = 0.28
            event.params["high_patch_mu"] = 0.90


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRecoveryCurriculumEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg
):
    """Retention course with dense, recoverable HIGH-stage perturbations.

    The ordinary Retention play/evaluation config remains disturbance-free.
    This isolated training distribution oversamples the heading/lateral
    precursor states that were seen only once per long episode and therefore
    produced no useful stability-residual gradient in the first three runs.
    """

    def __post_init__(self):
        super().__post_init__()
        # Base spatial post-init intentionally clears legacy push events.  Add
        # this curriculum last so it exists only on the isolated training ID.
        self.events.push_robot = EventTerm(
            func=mdp.push_spatial_high_grip_recovery_by_velocity,
            mode="interval",
            interval_range_s=(1.20, 2.40),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    # Preserve the requested forward gait while sampling the
                    # observed failure axes.  All values are velocity deltas.
                    "x": (-0.06, 0.06),
                    "y": (-0.18, 0.18),
                    "z": (0.0, 0.0),
                    "roll": (-0.24, 0.24),
                    "pitch": (-0.12, 0.12),
                    "yaw": (-0.20, 0.20),
                },
            },
        )


@configclass
class HallSpatialTransitionRetentionSceneCfg(HallSpatialMediumRetentionSceneCfg):
    """Wide retention course with a long final-high runout.

    The legacy training patches are only 2.0 m wide; removing course-success
    truncation let slow drifting robots walk off the side and forward edges,
    which showed up as ~94% bad-orientation terminations.  An 8 m width gives
    the 26 s maneuver enough lateral margin and the 18 m high-end runout keeps
    the 5-15 s post-transition window inside the floor.
    """

    friction_high_start = _friction_patch_cfg(
        "FrictionHighStart",
        size_x=3.0,
        size_y=8.0,
        center_x=-1.5,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )
    friction_low = _friction_patch_cfg(
        "FrictionLow",
        size_x=2.0,
        size_y=8.0,
        center_x=1.0,
        friction=0.28,
        color=(0.95, 0.34, 0.04),
    )
    friction_high_end = _friction_patch_cfg(
        "FrictionHighEnd",
        size_x=18.0,
        size_y=8.0,
        center_x=11.0,
        friction=0.90,
        color=(0.05, 0.30, 0.78),
    )


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg
):
    """Transition-retention course: heading injection + post-L→H convergence.

    Only this isolated training distribution raises the transition heading/vy
    terms and injects heading disturbances during LOW and early HIGH_END.  The
    LOW cadence/stride adaptation, Hall gate and frozen fast base are all
    retained; PPO only fits the bounded stability residual.
    """

    scene: RobotSceneCfg = HallSpatialTransitionRetentionSceneCfg(
        num_envs=512, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5
        self.episode_length_s = 22.0
        # No course-success truncation: the full 16 m final-high runout is the
        # post-transition retention window (5-15 s at the requested speed).
        self.terminations.course_success.params["minimum_local_x"] = 100.0
        # Concentrate reset density on the final 1-2 s before LOW entry.
        self.events.reset_base.params["x_bands"] = (
            (-1.90, -1.35),
            (-1.15, -0.75),
            (-0.65, -0.35),
        )
        self.events.reset_base.params["band_probabilities"] = (0.30, 0.40, 0.30)
        # Discrete heading steps plus small vy/yaw-rate increments, only inside
        # LOW and the first five seconds after HIGH_END contact.
        self.events.push_robot = EventTerm(
            func=mdp.push_spatial_transition_heading_recovery,
            mode="interval",
            interval_range_s=(1.10, 2.20),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "vy_range": (-0.10, 0.10),
                "yaw_rate_range": (-0.45, 0.45),
                "high_end_window_s": 5.0,
            },
        )
        self.rewards.transition_heading_retention.weight = -24.0
        self.rewards.transition_vy_retention.weight = -8.0
        self.rewards.low_entry_heading_change.weight = -16.0
        self.rewards.windowed_vy.weight = -6.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionEnvCfg
):
    """Nominal transition-retention rollout without the disturbance curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 4.0
        self.events.push_robot = None
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = False
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR2EnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionEnvCfg
):
    """Round 2: survivable disturbances, longer convergence pressure.

    R1 made 96% of training episodes end in bad orientation because the
    discrete heading steps were too strong; the policy learned survival, not
    post-L→H convergence.  R2 uses smaller heading/velocity increments, keeps
    the same concentrated reset distribution, and lengthens the HIGH_END
    penalty window while raising the Δψ and windowed-vy weights.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.params["vy_range"] = (-0.08, 0.08)
        self.events.push_robot.params["yaw_rate_range"] = (-0.55, 0.55)
        self.events.push_robot.interval_range_s = (1.50, 2.50)
        self.rewards.transition_heading_retention.weight = -30.0
        self.rewards.transition_heading_retention.params["high_end_decay_s"] = 6.0
        self.rewards.transition_vy_retention.weight = -12.0
        self.rewards.low_entry_heading_change.weight = -30.0
        self.rewards.windowed_vy.weight = -10.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR3EnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR2EnvCfg
):
    """Round 3: attack the remaining LOW heading injection directly."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.low_entry_heading_change.weight = -45.0
        self.rewards.low_stage_yaw_rate.weight = -8.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR4aEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR3EnvCfg
):
    """R4 ablation A: low-mu curriculum only, no symmetry penalty.

    35% of environments get mu in [0.10, 0.16] and the rest [0.16, 0.28], so
    the maneuver residual sees the full fault distribution instead of only
    the 0.28 training point.  Every other R3 term stays unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.low_mu_curriculum = EventTerm(
            func=mdp.randomize_spatial_low_patch_mu,
            mode="startup",
            params={
                "extreme_mu_range": (0.10, 0.16),
                "mild_mu_range": (0.16, 0.28),
                "extreme_fraction": 0.35,
            },
        )
        self.rewards.low_stage_leg_symmetry.weight = 0.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR4bEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR4aEnvCfg
):
    """R4 full: low-mu curriculum plus a small LOW-only leg symmetry cost."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.low_stage_leg_symmetry.weight = -3.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR3EnvCfg
):
    """R5: rebalanced curriculum with an explicit mu=0.28 anchor.

    35% of environments stay at exactly 0.28 (nominal retention), 45% sample
    the 0.14-0.28 fault band and only 20% sample the extreme 0.10-0.14 range.
    The ineffective leg-symmetry penalty from R4b is removed; attitude-side
    terms stay at their R3 values.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.low_mu_curriculum = EventTerm(
            func=mdp.randomize_spatial_low_patch_mu,
            mode="startup",
            params={
                "extreme_mu_range": (0.10, 0.14),
                "mild_mu_range": (0.14, 0.28),
                "extreme_fraction": 0.20,
                "anchor_mu": 0.28,
                "anchor_fraction": 0.35,
            },
        )


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg
):
    """Train a fresh high-speed recovery expert on sustained HighEnd states."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.terminations.course_success.params["minimum_local_x"] = 100.0
        self.events.reset_base.func = mdp.reset_root_state_high_end_perturbed
        self.events.reset_base.params = {
            "x_range": (3.0, 7.5),
            "pose_range": {"y": (-0.05, 0.05), "yaw": (-0.12, 0.12)},
            # V2 is built only from training seeds 510--513.  Locked seed500
            # is rejected by the loader even if this path is changed by hand.
            "state_bank_path": (
                "/home/mosense/guo/unitree_rl_lab/artifacts/hall_cadence_stride/"
                "high_end_state_bank_train_510_513_v2.npz"
            ),
            "state_bank_required_role": "training_high_end_state_bank",
            "velocity_range": {
                "x": (0.10, 0.25),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (-0.08, 0.08),
                "pitch": (-0.06, 0.06),
                "yaw": (-0.08, 0.08),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }
        # The V2 bank reset writes root *and* joint state atomically.  The
        # inherited reset_robot_joints term runs later in EventManager order
        # and would otherwise overwrite the restored walking pose with the
        # default noisy pose while leaving all 1864-D histories unchanged.
        self.events.reset_robot_joints = None
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.straight_line_motion.weight = -18.0
        self.rewards.straight_line_motion.params["yaw_rate_scale"] = 2.0
        self.rewards.straight_line_motion.params["lateral_clip"] = 1.5
        self.rewards.straight_cross_track.weight = -14.0
        self.rewards.straight_cross_track.params["error_clip"] = 1.0
        self.rewards.straight_heading_error.weight = -60.0
        self.rewards.straight_heading_error.params["error_clip"] = 1.0
        self.rewards.contact_point_slip.weight = -8.0
        self.rewards.flat_orientation_l2.weight = -60.0
        self.rewards.base_angular_velocity.weight = -1.20
        self.rewards.base_height.weight = -35.0
        self.rewards.action_rate.weight = -0.12
        self.rewards.termination_penalty.weight = -5000.0


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 6.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = False
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )


@configclass
class RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideEnvCfg
):
    """Long-horizon 0.8 m/s backbone task with no friction transition.

    This is an isolated diagnostic/retraining route for the weak original
    high-speed gait.  The actor ABI remains Hall/proprio ``policy[1864]``;
    friction/contact/force remain mechanics, critic and reward quantities.
    """

    scene: RobotSceneCfg = HallUniformHighFrictionLongSceneCfg(
        num_envs=512, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0

        # There is no LOW state in this task.  Replace the spatial state event
        # with a critic-only buffer that mirrors the real multiply-combined
        # floor/robot material, and remove the interval state machine.
        self.events.spatial_friction_reset.func = mdp.update_uniform_high_friction_buffer
        self.events.spatial_friction_reset.params = {"ground_patch_mu": 0.90}
        self.events.spatial_friction_update = None
        self.terminations.course_success = None

        # High grip remains randomized but never crosses into the low-grip
        # domain.  This tests robust high-speed stabilization, not adaptation.
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
        ):
            event.params["scale_range"] = (0.95, 1.10)
            event.params["restitution_range"] = (0.0, 0.02)

        self.events.reset_base.func = mdp.reset_root_state_uniform
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-7.20, -6.80),
                "y": (-0.08, 0.08),
                "yaw": (-0.10, 0.10),
            },
            "velocity_range": {
                "x": (0.08, 0.22),
                "y": (-0.08, 0.08),
                "z": (0.0, 0.0),
                "roll": (-0.08, 0.08),
                "pitch": (-0.06, 0.06),
                "yaw": (-0.10, 0.10),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }
        # Target the observed late-failure axes without injecting an
        # unobservable material switch.  This is a velocity increment applied
        # to the live gait, not a command or actor input.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            is_global_time=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.04, 0.04),
                    "y": (-0.12, 0.12),
                    "z": (0.0, 0.0),
                    "roll": (-0.14, 0.14),
                    "pitch": (-0.08, 0.08),
                    "yaw": (-0.14, 0.14),
                },
            },
        )

        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.stop_fraction = 0.0
        self.commands.base_velocity.crawl_fraction = 0.0
        self.commands.base_velocity.cruise_speed_range = (0.80, 0.80)
        self.commands.base_velocity.high_speed_range = (0.80, 0.80)

        # Velocity reward saturates at the requested speed.  Persistent
        # heading/lateral error, not instantaneous speed alone, dominates the
        # late-fall objective.
        self.rewards.track_lin_vel_xy.weight = 7.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.straight_heading_error.weight = -35.0
        self.rewards.straight_heading_error.params["error_clip"] = 1.0
        self.rewards.straight_line_motion.weight = -12.0
        self.rewards.straight_line_motion.params["yaw_rate_scale"] = 2.0
        self.rewards.straight_cross_track.weight = -10.0
        self.rewards.straight_cross_track.params["error_clip"] = 1.0
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.base_angular_velocity.weight = -0.90
        self.rewards.action_rate.weight = -0.10
        self.rewards.contact_point_slip.weight = -4.0
        self.rewards.friction_cone_margin.weight = -0.35
        self.rewards.termination_penalty.weight = -5000.0


@configclass
class RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneWarmupEnvCfg(
    RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg
):
    """H0 curriculum: learn a stable 30-s high-speed attractor first.

    The full backbone task intentionally contains latency, dynamics/Hall DR
    and targeted pushes.  Applying every stressor from the first PPO update
    produced a policy that merely delayed late falls.  H0 removes those
    stressors while preserving the exact 1864-D actor, 570-D critic, 0.8-m/s
    request, 40-m floor and long-horizon stability rewards.  A checkpoint is
    promoted to the full task only after the independent nominal evaluator
    passes; this is a curriculum stage, never an easier acceptance test.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        for name in (
            "add_base_mass",
            "base_com",
            "actuator_gains",
            "motor_strength",
            "joint_dynamics",
        ):
            setattr(self.events, name, None)
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
        ):
            event.params["scale_range"] = (1.0, 1.0)
            event.params["restitution_range"] = (0.0, 0.0)
        self.actions.JointPositionAction.min_delay = 0
        self.actions.JointPositionAction.max_delay = 0
        self.actions.JointPositionAction.delay_probabilities = (1.0,)
        self.hall_sensor_cfg.enable_domain_randomization = False
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )
        self.events.reset_base.params["pose_range"] = {
            "x": (-7.10, -6.90),
            "y": (-0.03, 0.03),
            "yaw": (-0.03, 0.03),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.10, 0.20),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.0),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.03, 0.03),
        }


@configclass
class RobotFootTractionMagneticMotionUniformHighFrictionLongBackbonePlayEnvCfg(
    RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 4.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = False
        sync_hall_sensor_cfg_to_policy_terms(self.observations, self.hall_sensor_cfg)
        self.events.push_robot = None
        for name in (
            "add_base_mass",
            "base_com",
            "actuator_gains",
            "motor_strength",
            "joint_dynamics",
        ):
            setattr(self.events, name, None)
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
        ):
            event.params["scale_range"] = (1.0, 1.0)
            event.params["restitution_range"] = (0.0, 0.0)
        self.actions.JointPositionAction.min_delay = 0
        self.actions.JointPositionAction.max_delay = 0
        self.actions.JointPositionAction.delay_probabilities = (1.0,)
        self.viewer.eye = (7.0, -24.0, 11.0)
        self.viewer.lookat = (7.0, 0.0, 0.5)


@configclass
class RobotFootTractionMagneticMotionUniformHighFrictionLongBackbone482EnvCfg(
    RobotFootTractionMagneticMotionUniformHighFrictionLongBackboneWarmupEnvCfg
):
    """Nominal long-horizon retraining with the isolated 482-D actor group."""

    observations: ObservationsCfg = FootTractionHighSpeedBackbone482ObservationsCfg()


@configclass
class RobotFootTractionMagneticMotionUniformHighFrictionLongBackbone482PlayEnvCfg(
    RobotFootTractionMagneticMotionUniformHighFrictionLongBackbonePlayEnvCfg
):
    """Common nominal gate with both 1864-D Hall and 482-D high-speed groups."""

    observations: ObservationsCfg = FootTractionHighSpeedBackbone482ObservationsCfg()


@configclass
class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideRetentionEnvCfg
):
    """Deterministic four-environment inspection of the retention course."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 12.0
        self.hall_sensor_cfg.enable_domain_randomization = False
        self.hall_sensor_cfg.enable_debug_vis = True
        self.hall_sensor_cfg.debug_vis_max_envs = 4
        sync_hall_sensor_cfg_to_policy_terms(
            self.observations, self.hall_sensor_cfg
        )
        self.events.reset_base.params["x_bands"] = ((-2.70, -2.30),)
        self.events.reset_base.params["band_probabilities"] = (1.0,)
        for event in (
            self.events.physics_material,
            self.events.physics_material_reset,
        ):
            event.params["scale_range"] = (1.0, 1.0)
            event.params["restitution_range"] = (0.0, 0.0)
        self.viewer.eye = (10.5, -15.0, 8.5)
        self.viewer.lookat = (3.2, 0.0, 0.35)
