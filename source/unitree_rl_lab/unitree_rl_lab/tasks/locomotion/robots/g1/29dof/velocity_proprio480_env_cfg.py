# Copyright (c) 2026. Pure-480-D proprioceptive friction-adaptation experiment.
# SPDX-License-Identifier: BSD-3-Clause
"""Isolated 480-D proprio actor ABI on the exact R5 transition-retention course.

This module deliberately adds new classes instead of editing the shared
``velocity_foot_env_cfg`` module: the main agent's R5 deployment work keeps
those files hot.  Everything below is an *addition*:

* the environment keeps the complete R5 physics, friction curriculum, rewards
  and the 1864-D Hall policy group (so Hall-domain-randomization and the
  audited course state machine stay bit-identical to R5);
* a new ``proprio480_policy`` observation group contains exactly the legacy
  480-D proprioceptive history (base_ang_vel 15 + projected_gravity 15 +
  velocity_commands 15 + joint_pos_rel 145 + joint_vel_rel 145 +
  last_action 145).  No Hall, contact, force, friction, mu or slip term ever
  enters this group;
* the privileged 570-D critic group is unchanged from R5.

The actor consumes only ``proprio480_policy``; the 1864-D group is retained
for Hall-sensor configuration sync and paired-evaluator diagnostics only.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from .velocity_env_cfg import ObservationsCfg
from .velocity_foot_env_cfg import (
    FootTractionMagneticMotionObservationsCfg,
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg,
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg,
)


@configclass
class FootTractionProprio480ObservationsCfg(
    FootTractionMagneticMotionObservationsCfg
):
    """R5 observation layout plus an isolated pure-480-D actor group.

    Term order, per-term history length, scaling and noise are copied verbatim
    from the first six terms of the audited 1864-D magnetic policy group, so
    the concatenated 480 columns are bit-compatible with ``policy[:, :480]``
    and therefore with ``model_49999`` warm starts.
    """

    @configclass
    class Proprio480PolicyCfg(ObsGroup):
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

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = True
            self.concatenate_terms = True

    proprio480_policy: Proprio480PolicyCfg = Proprio480PolicyCfg()


@configclass
class RobotFootTractionProprio480MagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionR5EnvCfg
):
    """R5 course with the pure-480-D proprio actor group.

    Physics, scene, commands (0.8 m/s), rewards, terminations, disturbance and
    low-mu curriculum are inherited unchanged from R5.  Only the observation
    dictionary gains the 480-D group consumed by the actor.
    """

    observations: ObservationsCfg = FootTractionProprio480ObservationsCfg()


@configclass
class RobotFootTractionProprio480MagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg(
    RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideTransitionRetentionPlayEnvCfg
):
    """Disturbance-free play variant with the same 480-D actor group."""

    observations: ObservationsCfg = FootTractionProprio480ObservationsCfg()
