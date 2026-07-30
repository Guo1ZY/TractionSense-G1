# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from unitree_rl_lab.tasks.locomotion.mdp import (
    compute_g1_29dof_motion_symmetric_states,
    compute_g1_29dof_symmetric_states,
)


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
