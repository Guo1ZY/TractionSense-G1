"""RSL-RL adapters for torque Teacher and deployment Student."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories

from unitree_rl_lab.traction.schema import PRIVILEGED_TRACTION_SCHEMA

from .networks import TorqueTractionStudentPolicy, torque_history_to_legacy_proprio
from .schema import TORQUE_TRACTION_FRAME_SCHEMA
from .teacher_schema import TORQUE_TEACHER_FLAT_DIM, TORQUE_TEACHER_FRAME_DIM, TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_PRIVILEGED_DIM


def _drop_legacy_kwargs(kwargs: dict) -> None:
    for name in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std", "actor_obs_normalization", "critic_obs_normalization"):
        kwargs.pop(name, None)


def torque_teacher_history_to_legacy(history: torch.Tensor, *, critic: bool) -> torch.Tensor:
    if history.shape[-2:] != (TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_FRAME_DIM):
        raise ValueError("Teacher history shape mismatch")
    proprio = history[..., :96]
    actor = torch.cat(tuple(proprio[..., item].flatten(-2) for item in (
        slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 38), slice(38, 67), slice(67, 96)
    )), dim=-1)
    if not critic:
        return actor
    base_slice = PRIVILEGED_TRACTION_SCHEMA.term_slice("base_linear_velocity")
    start = 99 + base_slice.start
    base_velocity = history[..., start : start + 3].flatten(-2)
    return torch.cat((base_velocity, actor), dim=-1)


class TorqueTractionStudentRslModel(MLPModel):
    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, *, latent_dim: int = 16, temporal_variant: str = "gru", freeze_student_policy: bool = True, **kwargs) -> None:
        _drop_legacy_kwargs(kwargs)
        self.head_input_dim = 480 + latent_dim + 3
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != TORQUE_TRACTION_FRAME_SCHEMA.flat_dimension:
            raise ValueError(f"Student observation is {self.obs_dim}, expected 1875")
        from .networks import TorqueTractionStudentCfg
        self.student_policy = TorqueTractionStudentPolicy(TorqueTractionStudentCfg(latent_dim=latent_dim, temporal_variant=temporal_variant, freeze_baseline=True))
        if freeze_student_policy:
            for parameter in self.student_policy.parameters():
                parameter.requires_grad_(False)
        self.latest_output = None
        self.latest_action_residual = None

    def _get_latent_dim(self) -> int:
        return self.head_input_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        flat = super().get_latent(obs, masks, hidden_state)
        leading = flat.shape[:-1]
        history = flat.reshape(-1, 15, 125)
        output = self.student_policy(history)
        self.latest_output = output
        baseline = torque_history_to_legacy_proprio(history)
        baseline_action = self.student_policy.baseline_actor(baseline)
        self.latest_action_residual = output.action - baseline_action
        command = history[:, -1, 6:9]
        latent = torch.cat((baseline, output.traction_latent, command), dim=-1)
        return latent.reshape(*leading, latent.shape[-1])

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        """Add the zero-gated distilled residual to the PPO locomotion mean."""
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks=None, hidden_state=hidden_state)
        mean = self.mlp(latent) + self.latest_action_residual.reshape(*latent.shape[:-1], 29)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self) -> nn.Module:
        return _TorchTorqueStudent(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _TorchTorqueStudent(self)


class _TorqueTeacherBase(MLPModel):
    critic_mode = False

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, *, latent_dim: int = 16, **kwargs) -> None:
        _drop_legacy_kwargs(kwargs)
        self.head_input_dim = (495 if self.critic_mode else 480) + latent_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != TORQUE_TEACHER_FLAT_DIM:
            raise ValueError(f"Teacher observation is {self.obs_dim}, expected {TORQUE_TEACHER_FLAT_DIM}")
        self.privileged_encoder = nn.Sequential(nn.Linear(TORQUE_TEACHER_PRIVILEGED_DIM, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, latent_dim))
        self.latest_traction_latent = None

    def _get_latent_dim(self) -> int:
        return self.head_input_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        flat = super().get_latent(obs, masks, hidden_state)
        leading = flat.shape[:-1]
        history = flat.reshape(-1, TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_FRAME_DIM)
        baseline = torque_teacher_history_to_legacy(history, critic=self.critic_mode)
        latent = self.privileged_encoder(history[:, -1, 99:])
        self.latest_traction_latent = latent
        result = torch.cat((baseline, latent), dim=-1)
        return result.reshape(*leading, result.shape[-1])

    def as_jit(self) -> nn.Module:
        return _TorchTorqueTeacher(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _TorchTorqueTeacher(self)


class TorqueTractionTeacherRslModel(_TorqueTeacherBase):
    critic_mode = False


class TorqueTractionTeacherCriticRslModel(_TorqueTeacherBase):
    critic_mode = True


class _TorchTorqueStudent(nn.Module):
    def __init__(self, model: TorqueTractionStudentRslModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.student = copy.deepcopy(model.student_policy)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic = model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()

    def forward(self, observation: torch.Tensor):
        observation = self.obs_normalizer(observation)
        history = observation.reshape(-1, 15, 125)
        student = self.student(history)
        baseline = torque_history_to_legacy_proprio(history)
        residual = student.action - self.student.baseline_actor(baseline)
        action = self.deterministic(self.mlp(torch.cat((baseline, student.traction_latent, history[:, -1, 6:9]), dim=-1)) + residual)
        return action, student.estimated_force, student.contact_probability, student.slip_probability, student.traction_utilization, student.traction_margin, student.estimator_confidence


class _TorchTorqueTeacher(nn.Module):
    def __init__(self, model: _TorqueTeacherBase) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.privileged_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.critic_mode = model.critic_mode
        self.deterministic = model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        history = self.obs_normalizer(observation).reshape(-1, TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_FRAME_DIM)
        baseline = torque_teacher_history_to_legacy(history, critic=self.critic_mode)
        return self.deterministic(self.mlp(torch.cat((baseline, self.encoder(history[:, -1, 99:])), dim=-1)))
