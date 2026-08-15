"""Causal traction diagnostics and multi-frame slip-event state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch


class SlipEventState(IntEnum):
    NO_CONTACT = 0
    STABLE_CONTACT = 1
    HIGH_UTILIZATION = 2
    SLIP_CANDIDATE = 3
    CONFIRMED_SLIP = 4
    RECOVERY = 5


@dataclass(frozen=True)
class TractionStateEstimatorCfg:
    dt: float = 0.02
    epsilon_n: float = 1.0
    utilization_high: float = 0.55
    slip_planar_speed_on_m_s: float = 0.12
    slip_planar_speed_off_m_s: float = 0.06
    foot_acceleration_scale_m_s2: float = 2.0
    force_growth_scale_n_s: float = 900.0
    torque_residual_scale_nm: float = 12.0
    imu_motion_scale_m_s2: float = 4.0
    candidate_threshold: float = 0.58
    confirm_duration_s: float = 0.06
    recovery_duration_s: float = 0.20
    contact_threshold: float = 0.50
    lowpass_tau_s: float = 0.06
    lower_bound_decay_per_s: float = 0.02


class TractionEstimatorInput(NamedTuple):
    force_local_n: torch.Tensor
    contact_probability: torch.Tensor
    foot_planar_velocity_m_s: torch.Tensor
    foot_planar_acceleration_m_s2: torch.Tensor
    force_growth_n_s: torch.Tensor
    torque_residual_norm_nm: torch.Tensor
    imu_linear_acceleration_m_s2: torch.Tensor
    estimator_confidence: torch.Tensor


class TractionEstimatorOutput(NamedTuple):
    traction_utilization: torch.Tensor
    slip_probability: torch.Tensor
    traction_margin: torch.Tensor
    friction_lower_bound: torch.Tensor
    slip_event_mu_estimate: torch.Tensor
    estimator_confidence: torch.Tensor
    state: torch.Tensor
    slip_duration_s: torch.Tensor


class TractionStateEstimator:
    """Produces utilization and event estimates, never a continuously true mu."""

    def __init__(self, num_envs: int, *, cfg: TractionStateEstimatorCfg = TractionStateEstimatorCfg(), device: str | torch.device = "cpu") -> None:
        self.num_envs, self.cfg, self.device = num_envs, cfg, torch.device(device)
        shape = (num_envs, 2)
        self.slip_probability = torch.zeros(shape, device=self.device)
        self.state = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.candidate_duration = torch.zeros(shape, device=self.device)
        self.slip_duration = torch.zeros(shape, device=self.device)
        self.recovery_duration = torch.zeros(shape, device=self.device)
        self.friction_lower_bound = torch.zeros(shape, device=self.device)
        self.slip_event_mu = torch.full(shape, float("nan"), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.slip_probability[ids] = 0.0
        self.state[ids] = int(SlipEventState.NO_CONTACT)
        self.candidate_duration[ids] = 0.0
        self.slip_duration[ids] = 0.0
        self.recovery_duration[ids] = 0.0
        self.friction_lower_bound[ids] = 0.0
        self.slip_event_mu[ids] = float("nan")

    def update(self, value: TractionEstimatorInput) -> TractionEstimatorOutput:
        f = torch.nan_to_num(value.force_local_n.to(self.device)).reshape(self.num_envs, 2, 3)
        contact_p = torch.nan_to_num(value.contact_probability.to(self.device)).clamp(0, 1)
        confidence = torch.nan_to_num(value.estimator_confidence.to(self.device)).clamp(0, 1)
        velocity = torch.linalg.vector_norm(torch.nan_to_num(value.foot_planar_velocity_m_s.to(self.device)), dim=-1)
        acceleration = torch.linalg.vector_norm(torch.nan_to_num(value.foot_planar_acceleration_m_s2.to(self.device)), dim=-1)
        growth = torch.nan_to_num(value.force_growth_n_s.to(self.device)).abs()
        residual = torch.nan_to_num(value.torque_residual_norm_nm.to(self.device)).clamp_min(0)
        imu_motion = torch.linalg.vector_norm(torch.nan_to_num(value.imu_linear_acceleration_m_s2.to(self.device)), dim=-1)[:, None]
        normal = f[..., 2].abs()
        tangent = torch.linalg.vector_norm(f[..., :2], dim=-1)
        utilization = (tangent / (normal + self.cfg.epsilon_n)).clamp(0.0, 3.0)
        active = contact_p >= self.cfg.contact_threshold
        speed_score = torch.sigmoid((velocity - self.cfg.slip_planar_speed_on_m_s) / 0.03)
        utilization_score = torch.sigmoid((utilization - self.cfg.utilization_high) / 0.10)
        acceleration_score = 1.0 - torch.exp(-acceleration / self.cfg.foot_acceleration_scale_m_s2)
        growth_score = 1.0 - torch.exp(-growth / self.cfg.force_growth_scale_n_s)
        residual_score = 1.0 - torch.exp(-residual / self.cfg.torque_residual_scale_nm)
        imu_gate = torch.exp(-imu_motion / self.cfg.imu_motion_scale_m_s2)
        raw_slip = active * contact_p * confidence * (
            0.34 * speed_score + 0.28 * utilization_score + 0.14 * acceleration_score
            + 0.12 * growth_score + 0.12 * residual_score
        ) * (0.55 + 0.45 * imu_gate)
        alpha = self.cfg.dt / (self.cfg.dt + self.cfg.lowpass_tau_s)
        self.slip_probability.add_(alpha * (raw_slip - self.slip_probability))

        was_confirmed = self.state == int(SlipEventState.CONFIRMED_SLIP)
        candidate = active & (self.slip_probability >= self.cfg.candidate_threshold)
        self.candidate_duration = torch.where(candidate, self.candidate_duration + self.cfg.dt, torch.zeros_like(self.candidate_duration))
        confirmed = candidate & (self.candidate_duration >= self.cfg.confirm_duration_s)
        just_confirmed = confirmed & ~was_confirmed
        self.slip_event_mu = torch.where(just_confirmed, utilization, self.slip_event_mu)
        self.slip_duration = torch.where(confirmed | was_confirmed, self.slip_duration + self.cfg.dt, torch.zeros_like(self.slip_duration))
        recovered_signal = active & (velocity < self.cfg.slip_planar_speed_off_m_s) & (self.slip_probability < 0.35)
        self.recovery_duration = torch.where(recovered_signal, self.recovery_duration + self.cfg.dt, torch.zeros_like(self.recovery_duration))
        recovering = was_confirmed & ~confirmed & (self.recovery_duration < self.cfg.recovery_duration_s)

        stable = active & ~candidate & ~recovering
        next_state = torch.full_like(self.state, int(SlipEventState.NO_CONTACT))
        next_state = torch.where(stable, torch.full_like(next_state, int(SlipEventState.STABLE_CONTACT)), next_state)
        high = stable & (utilization >= self.cfg.utilization_high)
        next_state = torch.where(high, torch.full_like(next_state, int(SlipEventState.HIGH_UTILIZATION)), next_state)
        next_state = torch.where(candidate, torch.full_like(next_state, int(SlipEventState.SLIP_CANDIDATE)), next_state)
        next_state = torch.where(confirmed, torch.full_like(next_state, int(SlipEventState.CONFIRMED_SLIP)), next_state)
        next_state = torch.where(recovering, torch.full_like(next_state, int(SlipEventState.RECOVERY)), next_state)
        self.state.copy_(next_state)

        decay = max(0.0, 1.0 - self.cfg.lower_bound_decay_per_s * self.cfg.dt)
        lower_candidate = torch.where(active & ~confirmed, utilization, torch.zeros_like(utilization))
        self.friction_lower_bound = torch.maximum(self.friction_lower_bound * decay, lower_candidate)
        capacity_reference = torch.where(torch.isfinite(self.slip_event_mu), self.slip_event_mu, self.friction_lower_bound)
        margin = (capacity_reference - utilization).clamp(-2.0, 2.0)
        return TractionEstimatorOutput(
            utilization, self.slip_probability.clone(), margin,
            self.friction_lower_bound.clone(), self.slip_event_mu.clone(), confidence,
            self.state.clone(), self.slip_duration.clone(),
        )
