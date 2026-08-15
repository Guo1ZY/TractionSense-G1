# Copyright (c) 2026. Pure-480-D proprioceptive friction-adaptation experiment.
# SPDX-License-Identifier: BSD-3-Clause
"""PPO runner for the isolated 480-D proprio actor on the R5 course.

Structure choice (documented as an ablation in the experiment report):
the actor is the plain ``model_49999`` architecture (512-256-128 ELU MLP),
not the R5 FastBase + Hall-gate + capture/stability composite.  Reusing the
audited AnchoredPPO would require its 1864-D Hall-teacher anchor and stage
machinery to be duplicated for a 480-D anchor, which is out of scope for the
main line of this experiment.  The env-side recipe (course, low-mu curriculum,
rewards, terminations, 570-D privileged critic) is inherited unchanged.

PPO hyperparameters follow the repository's full-MLP continuation recipe
(the audited UniformHighFrictionLongBackbone run).  R5's per-branch learning
rates and clip parameter are bound to its frozen-branch composite and do not
transfer to a plain trainable MLP trunk; this is recorded in the report.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FootTractionProprio480SpatialCadenceStrideTransitionRetentionR5PPORunnerCfg(
    RslRlOnPolicyRunnerCfg
):
    """480-D actor + 570-D privileged critic continuation on the R5 course."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_proprio480_spatial_"
        "cadence_stride_transition_retention_r5"
    )
    obs_groups = {"actor": ["proprio480_policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.08,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.06,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=8.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=0.10,
    )
    num_steps_per_env = 64
    max_iterations = 1200
    save_interval = 25
    clip_actions = 100.0
    save_exported_policy = True
