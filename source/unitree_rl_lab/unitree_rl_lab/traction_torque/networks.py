"""Teacher, temporal force correction, and deployable torque Student networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from unitree_rl_lab.traction.networks import LegacyLocomotionActor
from unitree_rl_lab.traction.schema import ACTION_DIM

from .schema import (
    LEFT_LEG_ACTION_INDICES,
    RIGHT_LEG_ACTION_INDICES,
    TORQUE_TRACTION_FRAME_SCHEMA,
)


def torque_history_to_legacy_proprio(history: torch.Tensor) -> torch.Tensor:
    """Convert newest five 125-D frames to the baseline 480-D term-major order."""
    if history.ndim != 3 or history.shape[1] < 5 or history.shape[-1] != 125:
        raise ValueError("history must be [batch,time>=5,125]")
    recent = history[:, -5:]
    slices = tuple(TORQUE_TRACTION_FRAME_SCHEMA.term_slice(name) for name in (
        "base_ang_vel", "projected_gravity", "command", "joint_pos_rel",
        "joint_vel", "previous_action",
    ))
    result = torch.cat([recent[..., item].reshape(history.shape[0], -1) for item in slices], dim=-1)
    if result.shape[-1] != 480:
        raise RuntimeError("legacy proprio conversion changed dimension")
    return result


class _SharedTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, variant: Literal["gru", "tcn"]) -> None:
        super().__init__()
        self.variant = variant
        if variant == "gru":
            self.encoder: nn.Module = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif variant == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 3, padding=2), nn.ELU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2), nn.ELU(),
            )
        else:
            raise ValueError(variant)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.variant == "gru":
            _, hidden = self.encoder(value)
            return hidden[-1]
        encoded = value.transpose(1, 2)
        for layer in self.encoder:
            encoded = layer(encoded)
            if isinstance(layer, nn.Conv1d):
                encoded = encoded[..., : value.shape[1]]
        return encoded[..., -1]


@dataclass(frozen=True)
class TemporalForceCorrectorCfg:
    history_frames: int = 15
    hidden_dim: int = 48
    maximum_correction_n: float = 120.0
    variant: Literal["gru", "tcn"] = "gru"


class TemporalForceCorrectionOutput(NamedTuple):
    corrected_force_n: torch.Tensor
    force_delta_n: torch.Tensor
    confidence: torch.Tensor
    correction_gate: torch.Tensor


class TemporalForceCorrector(nn.Module):
    """Shared per-leg causal correction; no truth force is accepted as input."""

    def __init__(self, cfg: TemporalForceCorrectorCfg = TemporalForceCorrectorCfg()) -> None:
        super().__init__()
        # tau6 + analytical force3 + q6 + dq6 + foot velocity2 + IMU3
        self.shared_encoder = _SharedTemporalEncoder(26, cfg.hidden_dim, cfg.variant)
        self.delta_head = nn.Linear(cfg.hidden_dim, 3)
        self.confidence_head = nn.Linear(cfg.hidden_dim, 1)
        self.gate_logit = nn.Parameter(torch.tensor(-6.0))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.cfg = cfg

    def _leg_history(self, history: torch.Tensor, leg: int) -> torch.Tensor:
        tau = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("leg_joint_tau_est")].reshape(*history.shape[:2], 2, 6)[..., leg, :]
        force = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("estimated_foot_force")].reshape(*history.shape[:2], 2, 3)[..., leg, :]
        q = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("joint_pos_rel")]
        dq = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("joint_vel")]
        indices = LEFT_LEG_ACTION_INDICES if leg == 0 else RIGHT_LEG_ACTION_INDICES
        planar = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("foot_planar_velocity")].reshape(*history.shape[:2], 2, 2)[..., leg, :]
        imu = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("imu_linear_acceleration")]
        return torch.cat((tau, force, q[..., list(indices)], dq[..., list(indices)], planar, imu), dim=-1)

    def forward(self, history: torch.Tensor, analytical_force_n: torch.Tensor) -> TemporalForceCorrectionOutput:
        if history.ndim != 3 or history.shape[-1] != 125:
            raise ValueError("force corrector history must be [batch,time,125]")
        if analytical_force_n.shape != (history.shape[0], 6):
            raise ValueError("analytical force must be [batch,6]")
        history = torch.nan_to_num(history)
        latents = torch.stack([self.shared_encoder(self._leg_history(history, leg)) for leg in range(2)], dim=1)
        delta = self.cfg.maximum_correction_n * torch.tanh(self.delta_head(latents))
        gate = torch.sigmoid(self.gate_logit)
        confidence = torch.sigmoid(self.confidence_head(latents)).squeeze(-1)
        corrected = analytical_force_n.reshape(-1, 2, 3) + gate * delta
        return TemporalForceCorrectionOutput(corrected.reshape(-1, 6), delta.reshape(-1, 6), confidence, gate.expand(history.shape[0], 1))


@dataclass(frozen=True)
class TorqueTractionStudentCfg:
    latent_dim: int = 16
    foot_hidden_dim: int = 48
    proprio_hidden_dim: int = 96
    fusion_hidden_dim: int = 128
    temporal_variant: Literal["gru", "tcn"] = "gru"
    freeze_baseline: bool = True
    residual_action_limit: float = 1.0


class TorqueTractionStudentOutput(NamedTuple):
    action: torch.Tensor
    estimated_force: torch.Tensor
    contact_probability: torch.Tensor
    slip_probability: torch.Tensor
    traction_utilization: torch.Tensor
    traction_margin: torch.Tensor
    estimator_confidence: torch.Tensor
    traction_latent: torch.Tensor
    residual_gate: torch.Tensor


class TorqueTractionStudentPolicy(nn.Module):
    """15-frame deployment Student with a preserved 480-D locomotion actor."""

    def __init__(self, cfg: TorqueTractionStudentCfg = TorqueTractionStudentCfg()) -> None:
        super().__init__()
        if cfg.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        # q6+dq6+tau6+force3+contact1+confidence1+planar2+imu3 = 28.
        self.shared_foot_encoder = _SharedTemporalEncoder(28, cfg.foot_hidden_dim, cfg.temporal_variant)
        self.proprio_encoder = _SharedTemporalEncoder(96, cfg.proprio_hidden_dim, cfg.temporal_variant)
        fused_dim = 2 * cfg.foot_hidden_dim + cfg.proprio_hidden_dim
        self.fusion = nn.Sequential(nn.Linear(fused_dim, cfg.fusion_hidden_dim), nn.ELU(), nn.Linear(cfg.fusion_hidden_dim, cfg.latent_dim))
        self.slip_head = nn.Linear(cfg.latent_dim + 2 * cfg.foot_hidden_dim, 2)
        self.utilization_head = nn.Linear(cfg.latent_dim, 2)
        self.margin_head = nn.Linear(cfg.latent_dim, 2)
        self.confidence_head = nn.Linear(cfg.latent_dim + 2, 2)
        self.force_correction_head = nn.Linear(2 * cfg.foot_hidden_dim, 6)
        nn.init.zeros_(self.force_correction_head.weight)
        nn.init.zeros_(self.force_correction_head.bias)
        self.baseline_actor = LegacyLocomotionActor(480)
        self.residual_actor = nn.Sequential(nn.Linear(cfg.latent_dim + 3, 64), nn.ELU(), nn.Linear(64, ACTION_DIM))
        nn.init.zeros_(self.residual_actor[-1].weight)
        nn.init.zeros_(self.residual_actor[-1].bias)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-6.0))
        self.force_gate_logit = nn.Parameter(torch.tensor(-6.0))
        self.cfg = cfg
        if cfg.freeze_baseline:
            for parameter in self.baseline_actor.parameters():
                parameter.requires_grad_(False)

    def _leg_history(self, history: torch.Tensor, leg: int) -> torch.Tensor:
        q = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("joint_pos_rel")]
        dq = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("joint_vel")]
        indices = LEFT_LEG_ACTION_INDICES if leg == 0 else RIGHT_LEG_ACTION_INDICES
        tau = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("leg_joint_tau_est")].reshape(*history.shape[:2], 2, 6)[..., leg, :]
        force = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("estimated_foot_force")].reshape(*history.shape[:2], 2, 3)[..., leg, :]
        contact = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("contact_probability")][..., leg : leg + 1]
        confidence = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("force_estimator_confidence")][..., leg : leg + 1]
        velocity = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("foot_planar_velocity")].reshape(*history.shape[:2], 2, 2)[..., leg, :]
        imu = history[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("imu_linear_acceleration")]
        return torch.cat((q[..., list(indices)], dq[..., list(indices)], tau, force, contact, confidence, velocity, imu), dim=-1)

    def forward(self, history: torch.Tensor, adjusted_command: torch.Tensor | None = None) -> TorqueTractionStudentOutput:
        if history.ndim != 3 or history.shape[1] != 15 or history.shape[2] != 125:
            raise ValueError("Student expects [batch,15,125]")
        history = torch.nan_to_num(history)
        left, right = (self.shared_foot_encoder(self._leg_history(history, leg)) for leg in range(2))
        feet = torch.cat((left, right), dim=-1)
        proprio = self.proprio_encoder(history[..., :96])
        latent = self.fusion(torch.cat((proprio, feet), dim=-1))
        latest = history[:, -1]
        contact = latest[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("contact_probability")].clamp(0, 1)
        analytical_force = latest[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("estimated_foot_force")]
        force_gate = torch.sigmoid(self.force_gate_logit)
        corrected_force = analytical_force + force_gate * torch.tanh(self.force_correction_head(feet))
        slip = torch.sigmoid(self.slip_head(torch.cat((latent, feet), dim=-1)))
        utilization = 3.0 * torch.sigmoid(self.utilization_head(latent))
        margin = 2.0 * torch.tanh(self.margin_head(latent))
        physical_confidence = latest[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("force_estimator_confidence")].clamp(0, 1)
        confidence = torch.sigmoid(self.confidence_head(torch.cat((latent, physical_confidence), dim=-1))) * physical_confidence
        command = latest[..., TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")] if adjusted_command is None else adjusted_command
        baseline = self.baseline_actor(torque_history_to_legacy_proprio(history))
        residual_gate = torch.sigmoid(self.residual_gate_logit)
        residual = self.cfg.residual_action_limit * torch.tanh(self.residual_actor(torch.cat((latent, command), dim=-1)))
        action = baseline + residual_gate * residual
        return TorqueTractionStudentOutput(action, corrected_force, contact, slip, utilization, margin, confidence, latent, residual_gate.expand(history.shape[0], 1))


@dataclass(frozen=True)
class TorqueTractionTeacherCfg:
    privileged_input_dim: int
    latent_dim: int = 16


class TorqueTractionTeacherOutput(NamedTuple):
    action: torch.Tensor
    traction_latent: torch.Tensor
    slip_probability: torch.Tensor
    traction_margin: torch.Tensor
    contact_probability: torch.Tensor
    force_correction_target: torch.Tensor


class TorqueTractionTeacherPolicy(nn.Module):
    """Privileged upper-bound model; never used as the deployable export."""

    def __init__(self, cfg: TorqueTractionTeacherCfg) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(cfg.privileged_input_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, cfg.latent_dim))
        self.baseline_actor = LegacyLocomotionActor(480)
        self.actor_residual = nn.Sequential(nn.Linear(cfg.latent_dim + 3, 64), nn.ELU(), nn.Linear(64, ACTION_DIM))
        nn.init.zeros_(self.actor_residual[-1].weight)
        nn.init.zeros_(self.actor_residual[-1].bias)
        self.slip_head, self.margin_head, self.contact_head = nn.Linear(cfg.latent_dim, 2), nn.Linear(cfg.latent_dim, 2), nn.Linear(cfg.latent_dim, 2)
        self.force_correction_head = nn.Linear(cfg.latent_dim, 6)
        self.cfg = cfg

    def forward(self, baseline_observation: torch.Tensor, command: torch.Tensor, privileged_input: torch.Tensor) -> TorqueTractionTeacherOutput:
        if baseline_observation.shape[-1] != 480 or command.shape[-1] != 3 or privileged_input.shape[-1] != self.cfg.privileged_input_dim:
            raise ValueError("Teacher input shape mismatch")
        latent = self.encoder(torch.nan_to_num(privileged_input))
        action = self.baseline_actor(baseline_observation) + self.actor_residual(torch.cat((latent, command), dim=-1))
        return TorqueTractionTeacherOutput(action, latent, torch.sigmoid(self.slip_head(latent)), 2.0 * torch.tanh(self.margin_head(latent)), torch.sigmoid(self.contact_head(latent)), self.force_correction_head(latent))


@dataclass(frozen=True)
class TorqueDistillationLossCfg:
    action: float = 1.0
    latent: float = 1.0
    slip: float = 1.0
    margin: float = 0.4
    contact: float = 0.5
    force: float = 0.5
    confidence: float = 0.2
    temporal_smoothness: float = 0.05


def torque_student_distillation_loss(*, student: TorqueTractionStudentOutput, teacher: TorqueTractionTeacherOutput, slip_label: torch.Tensor, margin_target: torch.Tensor, contact_label: torch.Tensor, force_target: torch.Tensor, confidence_target: torch.Tensor, previous_student_force: torch.Tensor | None = None, cfg: TorqueDistillationLossCfg = TorqueDistillationLossCfg()) -> dict[str, torch.Tensor]:
    losses = {
        "action": functional.mse_loss(student.action, teacher.action.detach()),
        "latent": functional.mse_loss(student.traction_latent, teacher.traction_latent.detach()),
        "slip": functional.binary_cross_entropy(student.slip_probability, slip_label.float()),
        "margin": functional.smooth_l1_loss(student.traction_margin, margin_target),
        "contact": functional.binary_cross_entropy(student.contact_probability.clamp(1e-5, 1 - 1e-5), contact_label.float()),
        "force": functional.smooth_l1_loss(student.estimated_force, force_target),
        "confidence": functional.mse_loss(student.estimator_confidence, confidence_target),
        "temporal_smoothness": torch.zeros((), device=student.action.device) if previous_student_force is None else functional.smooth_l1_loss(student.estimated_force, previous_student_force),
    }
    total = sum(getattr(cfg, name) * value for name, value in losses.items())
    return {"total": total, **losses}
