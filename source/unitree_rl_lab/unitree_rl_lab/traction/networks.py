"""Teacher and temporal tactile-proprioceptive Student networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (
    ACTION_DIM,
    PRIVILEGED_TRACTION_SCHEMA,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)


class StudentEncoderOutput(NamedTuple):
    latent: torch.Tensor
    slip_probability: torch.Tensor
    traction_score: torch.Tensor
    sensor_confidence: torch.Tensor


class StudentPolicyOutput(NamedTuple):
    action_mean: torch.Tensor
    latent: torch.Tensor
    slip_probability: torch.Tensor
    traction_score: torch.Tensor
    sensor_confidence: torch.Tensor
    traction_gate: torch.Tensor


class TeacherPolicyOutput(NamedTuple):
    action_mean: torch.Tensor
    latent: torch.Tensor


@dataclass(frozen=True)
class PrivilegedTractionEncoderCfg:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 64)


class PrivilegedTractionEncoder(nn.Module):
    """Compress physically available privileged traction diagnostics."""

    def __init__(self, cfg: PrivilegedTractionEncoderCfg) -> None:
        super().__init__()
        if cfg.input_dim <= 0 or cfg.latent_dim not in (8, 16):
            raise ValueError("Teacher input must be positive; latent_dim must be 8 or 16")
        layers: list[nn.Module] = []
        input_dim = cfg.input_dim
        for hidden_dim in cfg.hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, cfg.latent_dim))
        self.cfg = cfg
        self.network = nn.Sequential(*layers)

    def forward(self, privileged_traction: torch.Tensor) -> torch.Tensor:
        if privileged_traction.shape[-1] != self.cfg.input_dim:
            raise ValueError(
                f"privileged input has {privileged_traction.shape[-1]} features, "
                f"expected {self.cfg.input_dim}"
            )
        return self.network(torch.nan_to_num(privileged_traction))


class TeacherTractionPolicy(nn.Module):
    """Teacher encoder and 29-D actor using current proprioception and command."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        privileged_input_dim: int = PRIVILEGED_TRACTION_SCHEMA.flat_dimension,
    ) -> None:
        super().__init__()
        self.encoder = PrivilegedTractionEncoder(
            PrivilegedTractionEncoderCfg(
                input_dim=privileged_input_dim,
                latent_dim=latent_dim,
            )
        )
        self.actor = nn.Sequential(
            nn.Linear(96 + 3 + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(
        self,
        current_proprio: torch.Tensor,
        adjusted_command: torch.Tensor,
        privileged_traction: torch.Tensor,
    ) -> TeacherPolicyOutput:
        if current_proprio.shape[-1] != 96 or adjusted_command.shape[-1] != 3:
            raise ValueError("Teacher expects 96-D proprioception and 3-D command")
        latent = self.encoder(privileged_traction)
        action = self.actor(
            torch.cat((current_proprio, adjusted_command, latent), dim=-1)
        )
        return TeacherPolicyOutput(action, latent)


class _SharedFootGru(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        return hidden[-1]


class _SharedFootTcn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)

    def _causal(self, layer: nn.Conv1d, value: torch.Tensor) -> torch.Tensor:
        result = layer(value)
        return result[..., : value.shape[-1]]

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        value = history.transpose(1, 2)
        value = F.elu(self._causal(self.conv1, value))
        value = F.elu(self._causal(self.conv2, value))
        return value[..., -1]


@dataclass(frozen=True)
class TemporalStudentEncoderCfg:
    latent_dim: int = 16
    foot_hidden_dim: int = 32
    proprio_hidden_dim: int = 64
    fusion_hidden_dim: int = 128
    variant: Literal["gru", "tcn"] = "gru"
    confidence_age_scale_s: float = 0.10


class TemporalTactileProprioceptiveStudentEncoder(nn.Module):
    """Shared-foot temporal encoder plus deployable proprioceptive history."""

    def __init__(self, cfg: TemporalStudentEncoderCfg = TemporalStudentEncoderCfg()) -> None:
        super().__init__()
        if cfg.latent_dim not in (8, 16):
            raise ValueError("Student latent_dim must be 8 or 16")
        if cfg.variant == "gru":
            self.shared_foot_encoder: nn.Module = _SharedFootGru(5, cfg.foot_hidden_dim)
        elif cfg.variant == "tcn":
            self.shared_foot_encoder = _SharedFootTcn(5, cfg.foot_hidden_dim)
        else:
            raise ValueError(cfg.variant)
        self.proprio_encoder = nn.GRU(96, cfg.proprio_hidden_dim, batch_first=True)
        fused_dim = 2 * cfg.foot_hidden_dim + cfg.proprio_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, cfg.fusion_hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.fusion_hidden_dim, cfg.latent_dim),
        )
        self.slip_head = nn.Linear(cfg.latent_dim + 2 * cfg.foot_hidden_dim, 2)
        self.traction_head = nn.Linear(cfg.latent_dim, 1)
        self.confidence_head = nn.Linear(cfg.latent_dim, 1)
        self.cfg = cfg
        self._force_slice = TEMPORAL_STUDENT_FRAME_SCHEMA.term_slice(
            "observed_foot_force"
        )
        self._valid_slice = TEMPORAL_STUDENT_FRAME_SCHEMA.term_slice(
            "foot_force_valid"
        )
        self._age_slice = TEMPORAL_STUDENT_FRAME_SCHEMA.term_slice("foot_force_age")

    def forward(self, history: torch.Tensor) -> StudentEncoderOutput:
        expected_dim = TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension
        if history.ndim != 3 or history.shape[-1] != expected_dim:
            raise ValueError(
                f"Student history must be [batch,time,{expected_dim}], "
                f"got {tuple(history.shape)}"
            )
        history = torch.nan_to_num(history)
        force = history[..., self._force_slice].reshape(*history.shape[:2], 2, 3)
        valid = history[..., self._valid_slice]
        age = history[..., self._age_slice].clamp_min(0.0)
        left_input = torch.cat(
            (force[..., 0, :], valid[..., 0:1], age[..., 0:1]), dim=-1
        )
        right_input = torch.cat(
            (force[..., 1, :], valid[..., 1:2], age[..., 1:2]), dim=-1
        )
        left_latent = self.shared_foot_encoder(left_input)
        right_latent = self.shared_foot_encoder(right_input)
        _, proprio_hidden = self.proprio_encoder(history[..., :96])
        proprio_latent = proprio_hidden[-1]
        foot_latent = torch.cat((left_latent, right_latent), dim=-1)
        latent = self.fusion(torch.cat((proprio_latent, foot_latent), dim=-1))
        slip_probability = torch.sigmoid(
            self.slip_head(torch.cat((latent, foot_latent), dim=-1))
        )
        traction_score = torch.sigmoid(self.traction_head(latent))
        learned_confidence = torch.sigmoid(self.confidence_head(latent))
        last_valid = valid[:, -1].mean(dim=-1, keepdim=True)
        last_age = age[:, -1].mean(dim=-1, keepdim=True)
        physical_confidence = last_valid * torch.exp(
            -last_age / self.cfg.confidence_age_scale_s
        )
        sensor_confidence = learned_confidence * physical_confidence
        return StudentEncoderOutput(
            latent,
            slip_probability,
            traction_score,
            sensor_confidence,
        )


class LegacyLocomotionActor(nn.Module):
    """Audited actor architecture used by model_49999."""

    def __init__(self, input_dim: int = 480) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


class GatedTractionPolicy(nn.Module):
    """Baseline locomotion actor plus a zero-initialized traction residual."""

    def __init__(
        self,
        *,
        baseline_input_dim: int = 480,
        encoder_cfg: TemporalStudentEncoderCfg = TemporalStudentEncoderCfg(),
    ) -> None:
        super().__init__()
        self.encoder = TemporalTactileProprioceptiveStudentEncoder(encoder_cfg)
        self.baseline_actor = LegacyLocomotionActor(baseline_input_dim)
        self.traction_residual = nn.Sequential(
            nn.Linear(encoder_cfg.latent_dim + 3, 64),
            nn.ELU(),
            nn.Linear(64, ACTION_DIM),
        )
        nn.init.zeros_(self.traction_residual[-1].weight)
        nn.init.zeros_(self.traction_residual[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(-6.0))

    def forward(
        self,
        baseline_observation: torch.Tensor,
        temporal_history: torch.Tensor,
        adjusted_command: torch.Tensor,
    ) -> StudentPolicyOutput:
        encoder_output = self.encoder(temporal_history)
        baseline_action = self.baseline_actor(baseline_observation)
        gate = torch.sigmoid(self.gate_logit)
        residual = self.traction_residual(
            torch.cat((encoder_output.latent, adjusted_command), dim=-1)
        )
        action = baseline_action + gate * residual
        return StudentPolicyOutput(
            action,
            encoder_output.latent,
            encoder_output.slip_probability,
            encoder_output.traction_score,
            encoder_output.sensor_confidence,
            gate.expand(baseline_action.shape[0], 1),
        )


def temporal_history_to_legacy_proprio(history: torch.Tensor) -> torch.Tensor:
    """Convert the newest five frames to the audited 480-D term-major order."""
    if history.ndim != 3 or history.shape[-1] != 106 or history.shape[1] < 5:
        raise ValueError("history must be [batch,time>=5,106]")
    recent = history[:, -5:]
    slices = (
        slice(0, 3),
        slice(3, 6),
        slice(93, 96),
        slice(6, 35),
        slice(35, 64),
        slice(64, 93),
    )
    return torch.cat(
        [recent[..., term_slice].reshape(history.shape[0], -1) for term_slice in slices],
        dim=-1,
    )


@dataclass(frozen=True)
class DistillationLossCfg:
    latent_weight: float = 1.0
    slip_weight: float = 1.0
    slip_positive_weight: float = 1.0
    action_weight: float = 1.0
    traction_weight: float = 0.2
    confidence_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.slip_positive_weight <= 0.0:
            raise ValueError("slip_positive_weight must be positive")
        if self.confidence_weight < 0.0:
            raise ValueError("confidence_weight must be non-negative")


class DistillationLossOutput(NamedTuple):
    total: torch.Tensor
    ppo: torch.Tensor
    latent: torch.Tensor
    slip: torch.Tensor
    action: torch.Tensor
    traction: torch.Tensor
    confidence: torch.Tensor


def teacher_student_loss(
    *,
    ppo_loss: torch.Tensor,
    student: StudentPolicyOutput,
    teacher_latent: torch.Tensor,
    teacher_action: torch.Tensor,
    slip_label: torch.Tensor,
    traction_target: torch.Tensor | None = None,
    confidence_target: torch.Tensor | None = None,
    cfg: DistillationLossCfg = DistillationLossCfg(),
) -> DistillationLossOutput:
    """PPO + latent + slip + action distillation with optional traction target."""
    latent_loss = F.mse_loss(student.latent, teacher_latent.detach())
    slip_target = slip_label.float()
    slip_sample_weight = torch.where(
        slip_target > 0.5,
        torch.full_like(slip_target, cfg.slip_positive_weight),
        torch.ones_like(slip_target),
    )
    slip_loss = F.binary_cross_entropy(
        student.slip_probability,
        slip_target,
        weight=slip_sample_weight,
    )
    action_loss = F.mse_loss(student.action_mean, teacher_action.detach())
    if traction_target is None:
        traction_loss = torch.zeros((), device=ppo_loss.device)
    else:
        traction_loss = F.mse_loss(student.traction_score, traction_target.detach())
    if confidence_target is None:
        confidence_loss = torch.zeros((), device=ppo_loss.device)
    else:
        confidence_loss = F.mse_loss(
            student.sensor_confidence,
            confidence_target.detach(),
        )
    total = (
        ppo_loss
        + cfg.latent_weight * latent_loss
        + cfg.slip_weight * slip_loss
        + cfg.action_weight * action_loss
        + cfg.traction_weight * traction_loss
        + cfg.confidence_weight * confidence_loss
    )
    return DistillationLossOutput(
        total,
        ppo_loss,
        latent_loss,
        slip_loss,
        action_loss,
        traction_loss,
        confidence_loss,
    )
