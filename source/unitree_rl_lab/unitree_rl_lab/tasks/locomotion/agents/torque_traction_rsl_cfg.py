"""Reproducible RSL-RL configs for the independent torque-traction tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TorqueStudentModelCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.traction_torque.rsl_models:TorqueTractionStudentRslModel"
    latent_dim: int = 16
    temporal_variant: str = "gru"
    freeze_student_policy: bool = True


@configclass
class TorqueTeacherModelCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.traction_torque.rsl_models:TorqueTractionTeacherRslModel"
    latent_dim: int = 16


@configclass
class TorqueTeacherCriticModelCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.traction_torque.rsl_models:TorqueTractionTeacherCriticRslModel"
    latent_dim: int = 16


def _ppo() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.15, entropy_coef=0.004, num_learning_epochs=4, num_mini_batches=4, learning_rate=3.0e-5, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.005, max_grad_norm=0.30)


@configclass
class TorqueTractionTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 20260803
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "g1_29dof_torque_traction_teacher"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    check_for_nan = True
    actor = TorqueTeacherModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.45))
    critic = TorqueTeacherCriticModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = _ppo()


@configclass
class TorqueTractionStudentRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 20260803
    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "g1_29dof_torque_traction_student"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    check_for_nan = True
    actor = TorqueStudentModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.40))
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = _ppo()


@configclass
class TorqueTractionDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 20260803
    num_steps_per_env = 120
    max_iterations = 2000
    save_interval = 50
    experiment_name = "g1_29dof_torque_traction_distillation"
    obs_groups = {"student": ["policy"], "teacher": ["critic"]}
    check_for_nan = True
    student = TorqueStudentModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.10))
    teacher = TorqueTeacherModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0))
    algorithm = RslRlDistillationAlgorithmCfg(num_learning_epochs=2, learning_rate=1.0e-4, gradient_length=15, max_grad_norm=0.5, loss_type="huber")
