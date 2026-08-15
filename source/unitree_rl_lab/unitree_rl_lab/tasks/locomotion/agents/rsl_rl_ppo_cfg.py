# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from unitree_rl_lab.tasks.locomotion.mdp import (
    compute_g1_29dof_motion_symmetric_states,
    compute_g1_29dof_symmetric_states,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    KNOWN_SPEEDBOOST112_SHA256,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_SPEEDBOOST112_FROZEN_TEACHER = str(
    _PROJECT_ROOT
    / "artifacts"
    / "hall_speed_demo"
    / "speedboost112_frozen_teacher.pt"
)
_LOW_RECOVERY_6149_EXPERT = str(
    _PROJECT_ROOT
    / "logs"
    / "rsl_rl"
    / "unitree_g1_29dof_velocity_foot_traction_hall_handoff_recovery"
    / "2026-08-10_13-31-15_stage7a_handoff_mild_mu018_026"
    / "model_6149.pt"
)
_LOW_RECOVERY_6149_SHA256 = (
    "2cef06ae9d189a4cb3ef22de4ce24be0d780f49007b0ee9ce2c897ff8a66f1ec"
)
_FASTBASE_GATE_MODEL49_SHA256 = (
    "beb4574037f5ab342cca01d67cfdca9b802d15bcdc85fa6d0d019d57d43f4955"
)


@configclass
class RslRlAnchoredPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Serializable Isaac Lab config for the project-local AnchoredPPO."""

    class_name: str = "unitree_rl_lab.traction.anchored_ppo:AnchoredPPO"
    anchor_teacher_checkpoint: str = ""
    anchor_loss_coef: float = 1.0
    anchor_delta_cap: float = 0.25
    anchor_teacher_action_clamp: float = 3.0
    anchor_policy_observation_group: str = "policy"
    anchor_sensor_age_scale: float = 0.25
    anchor_expected_teacher_source_sha256: str = KNOWN_SPEEDBOOST112_SHA256
    anchor_min_learning_rate: float = 1.0e-7
    anchor_max_learning_rate: float = 1.0e-5
    # Training-only LOW/HIGH classifier on the actor's shared MLP latent.  Its
    # privileged label is cached outside the policy observation and its head is
    # omitted from actor export; gradients still teach the 1864-D Hall trunk to
    # represent the physical transition sooner than sparse locomotion reward.
    stage_aux_loss_coef: float = 0.10
    stage_aux_hidden_dim: int = 64
    stage_aux_reset_mask_steps: int = 1
    stage_aux_high_end_weight: float = 1.0
    # FastBase-only optimization phase: keep its residual action identically
    # zero while the Hall/proprio gate first learns LOW versus both HIGH
    # regions.  The algorithm checkpoints the completed-update counter.
    capture_gate_warmup_updates: int = 0
    capture_gate_warmup_learning_rate: float = 1.0e-4
    capture_gate_learning_rate: float = 1.0e-5
    capture_gate_max_grad_norm: float = 1.0
    capture_gate_gradient_mode: str = "joint"
    # Once gate-only warm-up completes, the residual receives its own Adam
    # role/LR/clip instead of inheriting the deliberately tiny PPO trust-region
    # rate.  During warm-up its group remains present at lr=0 and is frozen.
    capture_residual_learning_rate: float = 5.0e-5
    capture_residual_max_grad_norm: float = 0.5
    stability_residual_learning_rate: float = 2.0e-5
    stability_residual_max_grad_norm: float = 0.20
    # Disabled by default.  The explicit expert-distillation runner below is
    # the only task that loads model6149 inside the algorithm.
    low_expert_checkpoint: str = ""
    low_expert_expected_sha256: str = ""
    low_expert_distillation_loss_coef: float = 0.0
    low_expert_target_cap: float = 0.20
    low_expert_smooth_l1_beta: float = 0.05
    low_expert_command: tuple[float, float, float] = (0.16, 0.0, 0.0)
    low_expert_residual_gradient_mode: str = "joint"


@configclass
class FastBaseHallCaptureActorCfg(RslRlMLPModelCfg):
    """Native RSL Actor: immutable speedboost112 plus Hall capture residual."""

    class_name: str = (
        "unitree_rl_lab.traction.fastbase_capture_residual:"
        "FastBaseHallCaptureRslModel"
    )
    teacher_checkpoint: str = _SPEEDBOOST112_FROZEN_TEACHER
    residual_limit: float = 0.55
    gate_power: float = 1.0
    # Deployment calibration is material actor state: the FastBase module
    # persists both scalars as buffers.  Identity defaults preserve historical
    # behavior; audited legacy checkpoints migrate from these explicit values.
    gate_logit_scale: float = 1.0
    gate_logit_bias: float = 0.0
    teacher_trailing_mode: str = "assume_fresh"
    structured_features: bool = True


@configclass
class FastBaseHallCaptureStabilityActorCfg(FastBaseHallCaptureActorCfg):
    """Fast Hall adapter plus a small deployable long-horizon safety residual."""

    class_name: str = (
        "unitree_rl_lab.traction.fastbase_capture_residual:"
        "FastBaseHallCaptureStabilityRslModel"
    )
    # Actor units are converted by JointPositionAction scale=0.25 rad.  The
    # maximum additional correction is therefore 0.0625 rad per joint.
    stability_limit: float = 0.25
    stability_heading_start: float = 0.25
    stability_heading_full: float = 0.55
    stability_tilt_start: float = 0.08
    stability_tilt_full: float = 0.25
    stability_omega_start: float = 0.60
    stability_omega_full: float = 1.80
    stability_turning_yaw_threshold: float = 0.05
    # Optional transition-retention mode: keep the audited Hall gate and
    # capture residual bit-frozen while only the stability branch trains.
    freeze_capture_branches: bool = False


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FootPPORunnerCfg(BasePPORunnerCfg):
    """PPO config for foot-sensor fine-tune (resume from model_49999).

    Conservative LR / KL / grad clip — foot force spikes previously drove
    value loss → inf/NaN and scalar action-std below 0 around ~900 iter.
    """

    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot"
    # Keep scalar std so we can strict-resume foot checkpoints (same layout as 49999 path).
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class FootTurnPPORunnerCfg(FootPPORunnerCfg):
    """Yaw fine-tune from model_4000: lower LR, keep walking skills."""

    max_iterations = 6000
    experiment_name = "unitree_g1_29dof_velocity_foot_turn"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # a bit more exploration for new yaw cmds
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class FootAdaptivePPORunnerCfg(FootPPORunnerCfg):
    """Speed + turn + friction adaptive from model_foot_4000 (partial load)."""

    max_iterations = 8000
    experiment_name = "unitree_g1_29dof_velocity_foot_adaptive"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # explore higher speed / yaw / low-μ behaviors
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class FootAdaptiveYawPPORunnerCfg(FootAdaptivePPORunnerCfg):
    """Resume after NaN (e.g. model_5400): lower LR, more yaw exploration."""

    max_iterations = 9000
    experiment_name = "unitree_g1_29dof_velocity_foot_adaptive_yaw"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,  # safer after value NaN
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.007,
        max_grad_norm=0.4,
    )


@configclass
class FootStableAdaptivePPORunnerCfg(FootAdaptivePPORunnerCfg):
    """From model_6600: idle stand + low-μ slow (conservative LR)."""

    max_iterations = 5000
    experiment_name = "unitree_g1_29dof_velocity_foot_adaptive_stable"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.4,
    )


@configclass
class FootFullPPORunnerCfg(FootPPORunnerCfg):
    """Clean rebuild from model_49999 (partial). Tuned against mid-run value NaN.

    Past failures: high LR + foot force spikes → value loss → inf.
    Keep: clipped value loss, tight grad clip, mild KL, modest LR + adaptive schedule.
    """

    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_full"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,  # was 0.65 — less early exploration chaos
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,  # tighter PPO ratio (was 0.2)
        entropy_coef=0.008,
        num_learning_epochs=4,  # fewer passes per batch
        num_mini_batches=4,
        learning_rate=4.0e-5,  # safer than 8e-5 / 1e-4 (foot NaN history)
        schedule="adaptive",  # shrinks LR if KL too large
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,  # stricter than 0.008
        max_grad_norm=0.35,  # hard clip gradients
    )


@configclass
class FootAdaptiveV2PPORunnerCfg(FootPPORunnerCfg):
    """Adaptive-V2: outcome rewards + asymmetric critic μ. Partial from 49999."""

    max_iterations = 15000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_adaptive_v2"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.01,  # explore high-μ speed + low-μ slow
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.007,
        max_grad_norm=0.4,
    )


@configclass
class FootMuAdaptPPORunnerCfg(FootPPORunnerCfg):
    """510-dim MuAdapt: same actor layout as Foot-Full; outcome + lateral slip rewards.

    Tuned against RuntimeError: normal expects std >= 0 (scalar std can go negative
    under large grads). Lower LR / entropy / tighter grad clip than first MuAdapt try.
    """

    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_mu_adapt"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.45,  # was 0.55; resume ckpt still overwrites
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,  # less value chase on new rewards
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.003,  # low — prevent std blow-up
        num_learning_epochs=3,  # fewer passes per batch
        num_mini_batches=4,
        learning_rate=1.5e-5,  # very conservative on Full→MuAdapt
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.2,
    )


@configclass
class FootStraightMuPPORunnerCfg(FootPPORunnerCfg):
    """Straight-Mu: partial from 49999; high-μ fast straight / low-μ slow-stable.

    Slightly higher LR than Full→MuAdapt resume (first-layer expand needs signal),
    still conservative vs first foot NaN history.
    """

    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_straight_mu"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.55,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.35,
    )


@configclass
class FootTractionAdaptivePPORunnerCfg(FootPPORunnerCfg):
    """Traction teacher + 0.3-s foot context, partial warm-start from 49999."""

    max_iterations = 16000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_adaptive"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.50,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.30,
    )


@configclass
class FootTractionTeacherPPORunnerCfg(FootTractionAdaptivePPORunnerCfg):
    """Flat-ground privileged-μ teacher warm-started from the 640-D policy."""

    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher"


@configclass
class FootTractionRobustTeacherPPORunnerCfg(FootTractionTeacherPPORunnerCfg):
    """Conservative 2k-iteration Sim2Sim fine-tune from Teacher model_4999."""

    max_iterations = 2000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.25,
    )


@configclass
class FootTractionRobustStabilityTeacherPPORunnerCfg(FootTractionRobustTeacherPPORunnerCfg):
    """Low-LR 500-iteration pass that suppresses rare terminal failures."""

    max_iterations = 500
    save_interval = 50
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust_stability"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.20,
    )


@configclass
class FootTractionRobustShoulderTeacherPPORunnerCfg(
    FootTractionRobustStabilityTeacherPPORunnerCfg
):
    """Conservative 350-iteration shoulder boost from the stable model_7750."""

    max_iterations = 350
    save_interval = 25
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust_shoulder"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.0015,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=7.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.15,
    )


@configclass
class FootTractionRobustShoulderRecoveryTeacherPPORunnerCfg(
    FootTractionRobustShoulderTeacherPPORunnerCfg
):
    """Small trust-region recovery from the fastest zero-fall shoulder candidate."""

    max_iterations = 120
    save_interval = 10
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust_shoulder_recovery"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=0.10,
    )


@configclass
class FootTractionRobustShoulderGuardTeacherPPORunnerCfg(
    FootTractionRobustShoulderRecoveryTeacherPPORunnerCfg
):
    """Final 80-iteration high-mu guard with a tighter trust region."""

    max_iterations = 80
    save_interval = 10
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust_shoulder_guard"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
    )


@configclass
class FootTractionRobustLowMuRecoveryTeacherPPORunnerCfg(
    FootTractionRobustShoulderRecoveryTeacherPPORunnerCfg
):
    """Short isolated pass for the delayed low-mu failure tail."""

    max_iterations = 100
    save_interval = 10
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_teacher_robust_low_mu_recovery"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=0.10,
    )


@configclass
class FootTractionStudentPPORunnerCfg(FootTractionAdaptivePPORunnerCfg):
    """Deployable 640-D student with structured sensor-domain randomization."""

    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_student"


@configclass
class FootTractionLateralGuardTeacherPPORunnerCfg(
    FootTractionRobustShoulderGuardTeacherPPORunnerCfg
):
    """Short trust-region continuation that removes straight-line drift."""

    max_iterations = 100
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_lateral_guard"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
    )


@configclass
class FootTractionHallSwitchStudentPPORunnerCfg(FootTractionAdaptivePPORunnerCfg):
    """Conservative end-to-end PPO for the deployable Hall-only switch actor.

    The actor receives the 1864-D ``proprioception + Hall Bx/By/Bz history``
    observation only.  The asymmetric critic may use simulator-only contact
    and material state to reduce variance during PPO, but none of those
    privileged quantities are part of the exported actor interface.

    This is deliberately a fresh, long-horizon configuration rather than the
    100-iteration Teacher continuation used by the earlier DAgger collection
    task.  It is intended to be warm-started from ``model_49999.pt`` so the
    original nominal gait is preserved while the zero-initialized Hall input
    columns learn transition recovery under a tight trust region.
    """

    max_iterations = 5000
    save_interval = 25
    experiment_name = "unitree_g1_29dof_velocity_foot_traction_hall_switch_student"
    policy = RslRlPpoActorCriticCfg(
        # Keep exploration below the historical foot-policy setting: a large
        # action distribution makes early low-mu rollouts fail before Hall
        # features can receive useful gradients.
        init_noise_std=0.35,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        # Small updates retain the warm-started high-friction gait while the
        # Hall channels learn causal post-contact transition responses.
        clip_param=0.08,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.12,
    )


@configclass
class FootTractionHallHandoffRecoveryPPORunnerCfg(
    FootTractionHallSwitchStudentPPORunnerCfg
):
    """Tight-trust-region continuation for Stage7 capture-step recovery."""

    max_iterations = 1500
    save_interval = 25
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_handoff_recovery"
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
    )


@configclass
class FootTractionHallSpatialTransitionPPORunnerCfg(
    FootTractionHallHandoffRecoveryPPORunnerCfg
):
    """Hall-only PPO on causal high--low--high physical floor patches."""

    max_iterations = 3000
    save_interval = 25
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_transition"
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.06,
        entropy_coef=0.0012,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=8.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=0.10,
    )


@configclass
class FootTractionHallSpatialRetentionPPORunnerCfg(
    FootTractionHallSpatialTransitionPPORunnerCfg
):
    """Stage-S1 trust region: retain the fast gait on the mild course."""

    num_steps_per_env = 48
    max_iterations = 600
    save_interval = 25
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.08,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=0.06,
    )


@configclass
class FootTractionHallSpatialCapturePPORunnerCfg(
    FootTractionHallSpatialTransitionPPORunnerCfg
):
    """Stage-S2 longer rollout for the full high-to-low capture sequence."""

    num_steps_per_env = 64
    max_iterations = 1000
    save_interval = 25
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
    )


@configclass
class FootTractionHallSpatialAnchoredRetentionPPORunnerCfg(
    FootTractionHallSpatialRetentionPPORunnerCfg
):
    """Stage-S1 PPO with the validated fast Teacher anchored on HIGH patches."""

    class_name = "unitree_rl_lab.traction.anchored_ppo:AnchoredOnPolicyRunner"
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_anchored_retention"
    )
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=0.06,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.25,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=5.0e-7,
        anchor_max_learning_rate=4.0e-6,
        stage_aux_loss_coef=0.10,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
    )


@configclass
class FootTractionHallSpatialAnchoredCapturePPORunnerCfg(
    FootTractionHallSpatialCapturePPORunnerCfg
):
    """Stage-S2 PPO: HIGH gait retention while LOW remains reward-driven."""

    class_name = "unitree_rl_lab.traction.anchored_ppo:AnchoredOnPolicyRunner"
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_anchored_capture"
    )
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.30,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=1.0e-6,
        anchor_max_learning_rate=1.0e-5,
        stage_aux_loss_coef=0.15,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
    )


@configclass
class FootTractionHallSpatialFastBaseCapturePPORunnerCfg(
    FootTractionHallSpatialAnchoredCapturePPORunnerCfg
):
    """Medium-course PPO that cannot overwrite the validated fast gait.

    The Gaussian distribution and critic remain native RSL-RL components.
    PPO can update only the bounded Hall capture residual/gate (plus action
    standard deviation); all speedboost112 parameters have ``requires_grad=0``.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_fastbase_capture"
    )
    # Do not rely on RSL-RL's deprecated name inference.  This is inherited by
    # every calibrated/expert FastBase variant and structurally prevents the
    # privileged 570-D critic stream from ever being concatenated into the
    # deployable 1864-D Hall/proprio actor.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = FastBaseHallCaptureActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.08,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.30,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=1.0e-6,
        anchor_max_learning_rate=1.0e-5,
        # The old 0.15 auxiliary coefficient lost against PPO: gate means on
        # both HIGH and LOW rose from 0.119 to roughly 0.48.  First learn a
        # discriminative gate with an immutable zero residual, then let PPO
        # fit the bounded capture action while retaining the same strong BCE.
        stage_aux_loss_coef=1.0,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
        stage_aux_high_end_weight=4.0,
        capture_gate_warmup_updates=50,
        capture_gate_warmup_learning_rate=1.0e-4,
        capture_gate_learning_rate=1.0e-5,
        capture_gate_max_grad_norm=1.0,
        capture_residual_learning_rate=5.0e-5,
        capture_residual_max_grad_norm=0.5,
    )


@configclass
class FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg(
    FootTractionHallSpatialFastBaseCapturePPORunnerCfg
):
    """Explicit deployment-calibrated runner for the three-seed gate fit.

    Seeds 396/397/398 are calibration data and must not be reused for final
    acceptance.  The learned tensors and raw-gate BCE remain identical to the
    uncalibrated runner; only residual authority uses this monotone transform.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "calibrated_fastbase_capture"
    )
    actor = FastBaseHallCaptureActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.08,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
    )


@configclass
class FootTractionHallSpatialCadenceStridePPORunnerCfg(
    FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg
):
    """Long-form FastBase PPO for the non-prescriptive cadence/stride course.

    The frozen speedboost teacher protects the validated nominal gait, while
    the Hall-conditioned gate/residual, critic and action standard deviation
    remain trainable.  This runner deliberately does not inherit the 12-update
    GateBceOnly resume guard: it is a full isolated experiment that may start
    from the config-owned frozen teacher or resume a schema-compatible
    checkpoint.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride"
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    num_steps_per_env = 64
    max_iterations = 1000
    save_interval = 25
    require_fail_closed_training_start: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Once the residual becomes non-zero, ordinary PPO gradients through
        # the multiplicative gate can reopen it on HIGH_END.  The gate must be
        # trained only by its private LOW-vs-HIGH stage BCE; privileged labels
        # remain outside actor observations and deployment export.
        self.algorithm.capture_gate_gradient_mode = "stage_bce_only"


@configclass
class FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg(
    FootTractionHallSpatialCadenceStridePPORunnerCfg
):
    """Long-tail consolidation without reopening the HIGH/LOW Hall gate.

    Start from the audited cadence model55.  The imported action is initially
    bit-identical because the added stability branch has a zero output layer;
    PPO then learns only small observation-triggered corrections while the
    frozen fast teacher and HIGH action anchor retain nominal speed.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_retention"
    )
    actor = FastBaseHallCaptureStabilityActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.06,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        stability_limit=0.25,
        stability_heading_start=0.25,
        stability_heading_full=0.55,
        stability_tilt_start=0.08,
        stability_tilt_full=0.25,
        stability_omega_start=0.60,
        stability_omega_full=1.80,
        stability_turning_yaw_threshold=0.05,
    )
    num_steps_per_env = 64
    max_iterations = 200
    save_interval = 10

    def __post_init__(self):
        super().__post_init__()
        # The warm-start model52 has a nonzero Hall residual, so the legacy
        # gate-only warm-up guard must not run.  Capture branches are frozen
        # anyway; only the stability branch trains.
        self.algorithm.capture_gate_warmup_updates = 0
        self.algorithm.capture_gate_gradient_mode = "stage_bce_only"
        self.algorithm.capture_gate_learning_rate = 5.0e-6
        self.algorithm.capture_residual_learning_rate = 1.0e-5
        self.algorithm.capture_residual_max_grad_norm = 0.10
        # The stability branch must not inherit the KL-adaptive PPO LR: in the
        # first retention run it collapsed to 2e-7 and produced a maximum
        # correction of only 7e-4 actor units after 50 updates.  Give this
        # bounded branch its own auditable LR/clip while leaving the base,
        # Hall gate and Hall residual unchanged.
        # r2 (2e-5, 25 updates) remained safe numerically but reached only
        # 0.0041 actor units at the worst long-course state and left the exact
        # fall trajectory unchanged.  A private 1e-4 LR is still protected by
        # the 0.20 gradient clip, observation authority and 0.25 hard output
        # bound; no other actor branch inherits this rate.
        self.algorithm.stability_residual_learning_rate = 1.0e-4
        self.algorithm.stability_residual_max_grad_norm = 0.20
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.anchor_min_learning_rate = 2.0e-7
        self.algorithm.anchor_max_learning_rate = 3.0e-6
        self.algorithm.clip_param = 0.035
        self.algorithm.max_grad_norm = 0.06


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionPPORunnerCfg(
    FootTractionHallSpatialCadenceStrideRetentionPPORunnerCfg
):
    """Transition retention: frozen Hall adaptation, train only stability.

    Starts from the audited cadence model52.  The Hall gate and capture
    residual (the validated LOW stride/speed adaptation) stay bit-frozen;
    PPO fits only the zero-initialized stability branch against the
    transition heading injection and post-L→H heading/vy convergence terms.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention"
    )
    actor = FastBaseHallCaptureStabilityActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.06,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        stability_limit=0.30,
        stability_heading_start=0.06,
        stability_heading_full=0.30,
        stability_tilt_start=0.12,
        stability_tilt_full=0.30,
        stability_omega_start=0.50,
        stability_omega_full=1.50,
        stability_turning_yaw_threshold=0.05,
        freeze_capture_branches=True,
    )
    num_steps_per_env = 64
    max_iterations = 150
    save_interval = 10

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.capture_gate_gradient_mode = "stage_bce_only"
        # The capture branch optimizer roles run at lr=0 while
        # freeze_capture_branches is active; keep the stored rates positive so
        # the algorithm's validation and resume bookkeeping remain unchanged.
        self.algorithm.capture_gate_learning_rate = 5.0e-6
        self.algorithm.capture_residual_learning_rate = 1.0e-5
        self.algorithm.capture_residual_max_grad_norm = 0.10
        self.algorithm.stability_residual_learning_rate = 1.0e-4
        self.algorithm.stability_residual_max_grad_norm = 0.20
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.anchor_min_learning_rate = 2.0e-7
        self.algorithm.anchor_max_learning_rate = 3.0e-6
        self.algorithm.clip_param = 0.035
        self.algorithm.max_grad_norm = 0.06


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionR2PPORunnerCfg(
    FootTractionHallSpatialCadenceStrideTransitionRetentionPPORunnerCfg
):
    """Round 2: earlier/stronger heading authority, longer convergence fit."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_r2"
    )
    actor = FastBaseHallCaptureStabilityActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.06,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        stability_limit=0.55,
        stability_heading_start=0.03,
        stability_heading_full=0.18,
        stability_tilt_start=0.12,
        stability_tilt_full=0.30,
        stability_omega_start=0.35,
        stability_omega_full=1.20,
        stability_turning_yaw_threshold=0.05,
        freeze_capture_branches=True,
    )
    max_iterations = 400
    save_interval = 20

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.stability_residual_learning_rate = 2.0e-4
        self.algorithm.stability_residual_max_grad_norm = 0.30


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionR3PPORunnerCfg(
    FootTractionHallSpatialCadenceStrideTransitionRetentionR2PPORunnerCfg
):
    """Round 3: earlier authority and more correction headroom for LOW."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_r3"
    )
    actor = FastBaseHallCaptureStabilityActorCfg(
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.06,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        stability_limit=0.75,
        stability_heading_start=0.02,
        stability_heading_full=0.18,
        stability_tilt_start=0.12,
        stability_tilt_full=0.30,
        stability_omega_start=0.35,
        stability_omega_full=1.20,
        stability_turning_yaw_threshold=0.05,
        freeze_capture_branches=True,
    )


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionR4PPORunnerCfg(
    FootTractionHallSpatialCadenceStrideTransitionRetentionR3PPORunnerCfg
):
    """R4 low-mu curriculum runner; R3 authority settings are unchanged."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_r4"
    )


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionR5PPORunnerCfg(
    FootTractionHallSpatialCadenceStrideTransitionRetentionR3PPORunnerCfg
):
    """R5 rebalanced-curriculum runner; R3 authority settings unchanged."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_r5"
    )


@configclass
class FootTractionHallSpatialCadenceStrideTransitionRetentionSlopeStairsPPORunnerCfg(
    FootTractionHallSpatialCadenceStrideTransitionRetentionR5PPORunnerCfg
):
    """Separate ramps/stairs continuation of the R5 retention recipe.

    Same actor composition (FastBase + capture gate/residual + stability
    residual, frozen teacher), same PPO settings; only the experiment
    identity changes so its checkpoints never mix with the R5 model.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_slope_stairs_v1"
    )


@configclass
class FootTractionHallSpatialCadenceStrideHighEndRecoveryExpertPPORunnerCfg(
    FootTractionHallSpatialCadenceStridePPORunnerCfg
):
    """Un-gated HighEnd recovery expert without the failed stability branch."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_high_end_recovery_expert"
    )
    actor = FastBaseHallCaptureStabilityActorCfg(
        class_name=(
            "unitree_rl_lab.traction.fastbase_capture_residual:"
            "FastBaseHallCaptureHighEndRecoveryRslModel"
        ),
        hidden_dims=[1],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.08,
            std_type="scalar",
        ),
        teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        stability_limit=1.25,
        stability_heading_start=0.0,
        stability_heading_full=1.0,
        stability_tilt_start=0.0,
        stability_tilt_full=1.0,
        stability_omega_start=0.0,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
    )
    num_steps_per_env = 64
    # Feasibility-teacher phase: the first audited 1e-3 run changed actions by
    # only 0.016 actor unit (about 0.004 rad at the 0.25 action scale) after 50
    # updates and was physically indistinguishable from the frozen backbone.
    # This branch alone therefore receives a stronger short-run rate with
    # dense checkpoints.  It is not the final bounded deployment residual.
    max_iterations = 50
    save_interval = 5

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.capture_gate_warmup_updates = 0
        self.algorithm.capture_gate_gradient_mode = "joint"
        self.algorithm.stage_aux_loss_coef = 0.0
        self.algorithm.anchor_loss_coef = 0.0
        self.algorithm.capture_gate_learning_rate = 1.0e-6
        self.algorithm.capture_residual_learning_rate = 1.0e-5
        self.algorithm.capture_residual_max_grad_norm = 0.20
        self.algorithm.stability_residual_learning_rate = 5.0e-3
        self.algorithm.stability_residual_max_grad_norm = 1.00
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.anchor_min_learning_rate = 1.0e-6
        self.algorithm.anchor_max_learning_rate = 1.0e-5
        self.algorithm.clip_param = 0.05
        self.algorithm.max_grad_norm = 0.20


@configclass
class FootTractionHallUniformHighFrictionLongBackbonePPORunnerCfg(
    RslRlOnPolicyRunnerCfg
):
    """Conservative full-backbone continuation from original model_49999."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_"
        "uniform_high_friction_long_backbone"
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.12,
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
        gamma=0.995,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=0.10,
    )
    num_steps_per_env = 64
    max_iterations = 1200
    save_interval = 25
    clip_actions = 100.0
    save_exported_policy = True


@configclass
class FootTractionHallUniformHighFrictionLongBackboneWarmupPPORunnerCfg(
    FootTractionHallUniformHighFrictionLongBackbonePPORunnerCfg
):
    """Low-exploration H0 before latency/dynamics/fault hardening."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_"
        "uniform_high_friction_long_backbone_warmup"
    )
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.05,
            std_type="scalar",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-6,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.0010,
        max_grad_norm=0.08,
    )
    max_iterations = 300
    save_interval = 25


@configclass
class FootTractionHighSpeedBackbone482PPORunnerCfg(
    FootTractionHallUniformHighFrictionLongBackboneWarmupPPORunnerCfg
):
    """Train only the deployable 482-D long-horizon high-grip branch."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_"
        "high_speed_backbone_482"
    )
    obs_groups = {"actor": ["high_speed_policy"], "critic": ["critic"]}
    max_iterations = 400
    save_interval = 25


@configclass
class FootTractionHallSpatialCalibratedFastBaseExpertDistillPPORunnerCfg(
    FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg
):
    """LOW-only model6149 direction supervision with conservative residual LR."""

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "calibrated_fastbase_expert_distill"
    )
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.30,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=1.0e-6,
        anchor_max_learning_rate=1.0e-5,
        stage_aux_loss_coef=1.0,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
        stage_aux_high_end_weight=4.0,
        capture_gate_warmup_updates=50,
        capture_gate_warmup_learning_rate=1.0e-4,
        capture_gate_learning_rate=1.0e-5,
        capture_gate_max_grad_norm=1.0,
        # Deliberately separate from the existing 5e-5/0.5 Fast-LR A/B.
        capture_residual_learning_rate=2.0e-5,
        capture_residual_max_grad_norm=0.1,
        low_expert_checkpoint=_LOW_RECOVERY_6149_EXPERT,
        low_expert_expected_sha256=_LOW_RECOVERY_6149_SHA256,
        low_expert_distillation_loss_coef=0.25,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=(0.16, 0.0, 0.0),
        low_expert_residual_gradient_mode="joint",
    )


@configclass
class FootTractionHallSpatialCalibratedFastBaseExpertStrongDirectionPPORunnerCfg(
    FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg
):
    """Isolated strong LOW-expert direction fit; the conservative r1 is unchanged.

    The deployable actor, calibrated gate, HIGH anchor and raw-gate BCE are
    identical to the conservative expert runner.  Only the LOW-only ungated
    residual supervision authority changes: four times the loss coefficient,
    five times the residual learning rate and the already-audited 0.5 private
    residual clip.  Starting from the released model49 gate checkpoint lets
    this branch fit the cached model6149 direction immediately instead of
    spending another 50 updates behind an exact-zero residual warm-up.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "calibrated_fastbase_expert_strong_direction"
    )
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.30,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=1.0e-6,
        anchor_max_learning_rate=1.0e-5,
        stage_aux_loss_coef=1.0,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
        stage_aux_high_end_weight=4.0,
        capture_gate_warmup_updates=50,
        capture_gate_warmup_learning_rate=1.0e-4,
        capture_gate_learning_rate=1.0e-5,
        capture_gate_max_grad_norm=1.0,
        capture_residual_learning_rate=1.0e-4,
        capture_residual_max_grad_norm=0.5,
        low_expert_checkpoint=_LOW_RECOVERY_6149_EXPERT,
        low_expert_expected_sha256=_LOW_RECOVERY_6149_SHA256,
        low_expert_distillation_loss_coef=1.0,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=(0.16, 0.0, 0.0),
        low_expert_residual_gradient_mode="supervised_only",
    )


@configclass
class FootTractionHallSpatialCalibratedFastBaseExpertGateBceOnlyPPORunnerCfg(
    FootTractionHallSpatialCalibratedFastBaseCapturePPORunnerCfg
):
    """Fail-closed private supervision from the untouched released model49.

    The gate receives only the uncalibrated LOW/HIGH stage BCE, with extra
    HIGH_END weight.  The residual receives only HIGH anchor plus LOW model6149
    direction supervision.  PPO therefore cannot reopen the gate at HIGH_END
    or overpower the bounded recovery direction.  This is a new A/B runner;
    conservative r1 and strong-direction r1 remain byte-for-byte configured as
    before.  Start it from model49 without optimizer state.
    """

    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "calibrated_fastbase_expert_gate_bce_only"
    )
    # RSL-RL 5 can still infer these names for legacy configs, but relying on
    # that compatibility path emits a deprecation warning and could silently
    # change which observation group reaches a model after a library upgrade.
    # Keep the privileged 570-D critic group structurally separated from the
    # deployable 1864-D Hall/proprio actor at configuration time.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    # A short, checkpoint-dense causal A/B is intentional: verify that
    # HIGH_END authority closes before allowing another long residual fit.
    max_iterations = 12
    save_interval = 4
    # This experiment is meaningful only as the 12-update continuation of the
    # released gate-warmup model49.  ``train.py`` consumes these fields before
    # the first rollout and refuses fresh/partial/optimizer resumes, a
    # look-alike checkpoint, schema drift, or an unreleased residual branch.
    require_fail_closed_training_start: bool = True
    required_resume_checkpoint_sha256: str = _FASTBASE_GATE_MODEL49_SHA256
    required_resume_checkpoint_iteration: int = 49
    required_capture_gate_completed_updates: int = 50
    required_actor_observation_dim: int = 1864
    required_critic_observation_dim: int = 570
    required_action_dim: int = 29
    required_actor_trailing_feature_mode: str = "motion_feedback"
    required_gate_logit_scale: float = 2.75
    required_gate_logit_bias: float = -3.2
    default_configured_update_count: int = 12
    maximum_allowed_new_updates: int = 12
    algorithm = RslRlAnchoredPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.08,
        anchor_teacher_checkpoint=_SPEEDBOOST112_FROZEN_TEACHER,
        anchor_loss_coef=1.0,
        anchor_delta_cap=0.30,
        anchor_teacher_action_clamp=3.0,
        anchor_policy_observation_group="policy",
        anchor_sensor_age_scale=0.25,
        anchor_min_learning_rate=1.0e-6,
        anchor_max_learning_rate=1.0e-5,
        stage_aux_loss_coef=1.0,
        stage_aux_hidden_dim=64,
        stage_aux_reset_mask_steps=1,
        # Keep the already-validated 4x HIGH_END emphasis so this A/B isolates
        # the causal change (PPO gradient removal) instead of simultaneously
        # retuning an observation-ambiguous LOW->HIGH history boundary.
        stage_aux_high_end_weight=4.0,
        capture_gate_warmup_updates=50,
        capture_gate_warmup_learning_rate=1.0e-4,
        capture_gate_learning_rate=1.0e-5,
        capture_gate_max_grad_norm=1.0,
        capture_gate_gradient_mode="stage_bce_only",
        capture_residual_learning_rate=1.0e-4,
        capture_residual_max_grad_norm=0.5,
        low_expert_checkpoint=_LOW_RECOVERY_6149_EXPERT,
        low_expert_expected_sha256=_LOW_RECOVERY_6149_SHA256,
        low_expert_distillation_loss_coef=1.0,
        low_expert_target_cap=0.20,
        low_expert_smooth_l1_beta=0.05,
        low_expert_command=(0.16, 0.0, 0.0),
        low_expert_residual_gradient_mode="supervised_only",
    )


@configclass
class FootTractionSlopeStairsTeacherPPORunnerCfg(
    FootTractionLateralGuardTeacherPPORunnerCfg
):
    """Trust-region terrain continuation from the selected flat Teacher."""

    max_iterations = 400
    save_interval = 25
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_slope_stairs"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.15,
    )


@configclass
class FootTractionSpeedLateralTeacherPPORunnerCfg(
    FootTractionLateralGuardTeacherPPORunnerCfg
):
    """Conservative continuation for speed accuracy and straight tracking."""

    max_iterations = 200
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_speed_lateral"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.035,
        entropy_coef=0.0004,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=0.06,
    )


@configclass
class FootTractionSpeedLateralV2TeacherPPORunnerCfg(
    FootTractionSpeedLateralTeacherPPORunnerCfg
):
    """Extra-conservative continuation for learned Sim2Sim path correction."""

    max_iterations = 300
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_speed_lateral_v2"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=7.5e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0006,
        max_grad_norm=0.05,
    )


@configclass
class FootTractionSpeedLateralSymmetryTeacherPPORunnerCfg(
    FootTractionSpeedLateralV2TeacherPPORunnerCfg
):
    """Trust-region continuation with learned sagittal equivariance."""

    max_iterations = 240
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_speed_lateral_symmetry"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0006,
        max_grad_norm=0.05,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_g1_29dof_symmetric_states,
            mirror_loss_coeff=0.05,
        ),
    )


@configclass
class FootTractionMotionTeacherPPORunnerCfg(
    FootTractionSpeedLateralSymmetryTeacherPPORunnerCfg
):
    """Motion-feedback continuation for closed-loop lateral correction."""

    max_iterations = 300
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_motion_feedback"
    )


@configclass
class FootTractionMotionStrongSymmetryTeacherPPORunnerCfg(
    FootTractionMotionTeacherPPORunnerCfg
):
    """Learn stronger equivariance while PPO preserves forward locomotion."""

    max_iterations = 180
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_motion_strong_symmetry"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0006,
        max_grad_norm=0.05,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_g1_29dof_motion_symmetric_states,
            mirror_loss_coeff=0.10,
        ),
    )


@configclass
class FootTractionMotionBalancedSymmetryTeacherPPORunnerCfg(
    FootTractionMotionTeacherPPORunnerCfg
):
    """Enforce sagittal equivariance while retaining high-traction tracking."""

    max_iterations = 240
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_motion_balanced_symmetry"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.8,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0006,
        max_grad_norm=0.08,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_g1_29dof_motion_symmetric_states,
            mirror_loss_coeff=0.25,
        ),
    )


@configclass
class FootTractionMotionSwitchTeacherPPORunnerCfg(
    FootTractionMotionBalancedSymmetryTeacherPPORunnerCfg
):
    """Short continuation that learns high<->low traction transitions."""

    max_iterations = 240
    save_interval = 10
    experiment_name = (
        "unitree_g1_29dof_velocity_foot_traction_teacher_motion_switch"
    )
