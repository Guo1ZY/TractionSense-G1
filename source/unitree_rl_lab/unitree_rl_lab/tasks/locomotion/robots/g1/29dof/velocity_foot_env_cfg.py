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

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
            func=mdp.magnetic_array_proxy,
            params={"sensor_cfg": FOOT_SENSOR_CFG},
            clip=(-6.0, 6.0),
            history_length=15,
        )
        foot_sample_period_lr = ObsTerm(
            func=mdp.magnetic_sample_period_lr,
            params={"sensor_cfg": FOOT_SENSOR_CFG},
            clip=(0.001, 0.25),
            history_length=15,
        )
        foot_sensor_valid_lr = ObsTerm(
            func=mdp.magnetic_sensor_valid_lr,
            params={"sensor_cfg": FOOT_SENSOR_CFG},
            clip=(0.0, 1.0),
        )
        foot_sensor_age_lr = ObsTerm(
            func=mdp.magnetic_sensor_age_lr,
            params={"sensor_cfg": FOOT_SENSOR_CFG, "age_scale": 0.25},
            clip=(0.0, 1.0),
        )

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
    observations: ObservationsCfg = FootTractionMagneticObservationsCfg()
    events: EventCfg = TractionMagneticStudentEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True


@configclass
class RobotFootTractionMagneticStudentPlayEnvCfg(
    RobotFootTractionMagneticStudentEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
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

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.high_speed_regimes = (0, 2)
        self.commands.base_velocity.high_speed_range = (0.75, 1.00)
        self.rewards.gait.weight = 0.40
        self.rewards.gait.params["slow_period"] = 0.90
        self.rewards.gait.params["fast_period"] = 0.55
        self.rewards.gait.params["low_speed"] = 0.20
        self.rewards.gait.params["high_speed"] = 1.00
        self.rewards.track_lin_vel_xy.params["low_speed"] = 0.20
        self.rewards.track_lin_vel_xy.params["high_speed"] = 1.00
        self.rewards.traction_overspeed.params["low_speed"] = 0.20
        self.rewards.traction_overspeed.params["high_speed"] = 1.00


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
