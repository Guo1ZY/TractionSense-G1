"""Native-signal traction-adaptive command governor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class TorqueTractionGovernorCfg:
    dt: float = 0.02
    risk_enter: float = 0.48
    risk_exit: float = 0.28
    debounce_s: float = 0.06
    minimum_hold_s: float = 0.20
    persistent_slip_s: float = 0.12
    minimum_confidence: float = 0.25
    fast_down_tau_s: float = 0.08
    slow_recovery_tau_s: float = 0.90
    normal_vx_m_s: float = 1.5
    normal_vy_m_s: float = 0.6
    normal_yaw_rad_s: float = 1.2
    normal_acceleration_m_s2: float = 2.0
    normal_deceleration_m_s2: float = 2.5
    risk_acceleration_m_s2: float = 0.35
    risk_deceleration_m_s2: float = 0.8
    warning_utilization: float = 0.55
    critical_utilization: float = 0.90
    warning_margin: float = 0.18
    critical_margin: float = 0.03
    persistent_speed_scale: float = 0.25
    fallback_speed_scale: float = 0.45
    lateral_minimum_scale: float = 0.25
    yaw_minimum_scale: float = 0.22
    push_off_minimum_scale: float = 0.28
    single_support_risk_gain: float = 1.12

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.fast_down_tau_s <= 0.0 or self.slow_recovery_tau_s <= 0.0:
            raise ValueError("governor timing must be positive")
        if not 0 <= self.risk_exit < self.risk_enter <= 1:
            raise ValueError("governor risk hysteresis is invalid")


class TorqueGovernorOutput(NamedTuple):
    adjusted_command: torch.Tensor
    speed_scale: torch.Tensor
    lateral_scale: torch.Tensor
    yaw_scale: torch.Tensor
    acceleration_limit: torch.Tensor
    deceleration_limit: torch.Tensor
    yaw_limit: torch.Tensor
    push_off_scale: torch.Tensor
    slip_risk: torch.Tensor
    state: torch.Tensor
    safety_flags: torch.Tensor


class TorqueTractionCommandGovernor:
    """Fast risk response and slow recovery using Student/deployable signals only.

    States: 0 normal, 1 utilization limiting, 2 persistent slip, 3
    low-confidence fallback. No ground-truth force, friction, or privileged
    latent is accepted by this API.
    """

    def __init__(self, num_envs: int, *, cfg: TorqueTractionGovernorCfg = TorqueTractionGovernorCfg(), device: str | torch.device = "cpu", enabled: bool = True) -> None:
        self.num_envs, self.cfg, self.device, self.enabled = num_envs, cfg, torch.device(device), enabled
        self.speed_scale = torch.ones(num_envs, 1, device=self.device)
        self.previous_command = torch.zeros(num_envs, 3, device=self.device)
        self.limiting = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.high_risk_duration = torch.zeros(num_envs, device=self.device)
        self.hold_remaining = torch.zeros(num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        self.speed_scale[ids] = 1.0
        self.previous_command[ids] = 0.0
        self.limiting[ids] = False
        self.high_risk_duration[ids] = 0.0
        self.hold_remaining[ids] = 0.0

    def update(self, *, raw_command: torch.Tensor, slip_probability: torch.Tensor, traction_utilization: torch.Tensor, traction_margin: torch.Tensor, contact_probability: torch.Tensor, estimator_confidence: torch.Tensor, foot_relative_velocity: torch.Tensor, slip_duration: torch.Tensor, current_velocity: torch.Tensor) -> TorqueGovernorOutput:
        tensors = (raw_command, slip_probability, traction_utilization, traction_margin, contact_probability, estimator_confidence, foot_relative_velocity, slip_duration, current_velocity)
        shapes = ((self.num_envs, 3), (self.num_envs, 2), (self.num_envs, 2), (self.num_envs, 2), (self.num_envs, 2), (self.num_envs, 2), (self.num_envs, 2, 2), (self.num_envs, 2), (self.num_envs, 3))
        if any(tuple(value.shape) != shape for value, shape in zip(tensors, shapes, strict=True)):
            raise ValueError("torque governor input shape mismatch")
        raw, slip, utilization, margin, contact, confidence, foot_velocity, duration, current = (torch.nan_to_num(value.to(self.device)) for value in tensors)
        del current
        if not self.enabled:
            ones = torch.ones(self.num_envs, 1, device=self.device)
            zeros = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            return TorqueGovernorOutput(raw, ones, ones, ones, ones * self.cfg.normal_acceleration_m_s2, ones * self.cfg.normal_deceleration_m_s2, ones * self.cfg.normal_yaw_rad_s, ones, ones * 0.0, zeros, zeros)

        contact = contact.clamp(0, 1)
        confidence = confidence.clamp(0, 1)
        support_count = (contact >= 0.5).sum(dim=1, keepdim=True)
        supported_confidence = (confidence * contact).sum(dim=1, keepdim=True) / contact.sum(dim=1, keepdim=True).clamp_min(1.0)
        invalid = (support_count == 0) | (supported_confidence < self.cfg.minimum_confidence)
        utilization_risk = ((utilization - self.cfg.warning_utilization) / (self.cfg.critical_utilization - self.cfg.warning_utilization)).clamp(0, 1)
        margin_risk = ((self.cfg.warning_margin - margin) / (self.cfg.warning_margin - self.cfg.critical_margin)).clamp(0, 1)
        foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
        velocity_risk = torch.sigmoid((foot_speed - 0.10) / 0.035)
        per_foot_risk = torch.maximum(slip.clamp(0, 1), torch.maximum(utilization_risk, margin_risk))
        per_foot_risk = torch.maximum(per_foot_risk, 0.6 * velocity_risk)
        support_weight = contact * confidence
        risk = (per_foot_risk * support_weight).sum(dim=1, keepdim=True) / support_weight.sum(dim=1, keepdim=True).clamp_min(0.2)
        risk = torch.where(support_count == 1, (risk * self.cfg.single_support_risk_gain).clamp_max(1.0), risk)
        high = risk[:, 0] >= self.cfg.risk_enter
        self.high_risk_duration = torch.where(high, self.high_risk_duration + self.cfg.dt, torch.zeros_like(self.high_risk_duration))
        enter = self.high_risk_duration >= self.cfg.debounce_s
        self.hold_remaining = torch.where(enter, torch.full_like(self.hold_remaining, self.cfg.minimum_hold_s), (self.hold_remaining - self.cfg.dt).clamp_min(0))
        keep = (risk[:, 0] > self.cfg.risk_exit) | (self.hold_remaining > 0)
        self.limiting = torch.where(self.limiting, keep, enter)
        persistent = duration.max(dim=1, keepdim=True).values >= self.cfg.persistent_slip_s

        # High utilization first restricts acceleration/push-off. Steady speed
        # only drops aggressively after confirmed/persistent slip.
        target_speed = torch.where(persistent, torch.full_like(risk, self.cfg.persistent_speed_scale), 1.0 - 0.30 * risk)
        target_speed = torch.where(self.limiting[:, None], target_speed, torch.ones_like(target_speed))
        target_speed = torch.where(invalid, torch.minimum(target_speed, torch.full_like(target_speed, self.cfg.fallback_speed_scale)), target_speed).clamp(0.2, 1.0)
        down = min(1.0, self.cfg.dt / self.cfg.fast_down_tau_s)
        up = min(1.0, self.cfg.dt / self.cfg.slow_recovery_tau_s)
        alpha = torch.where(target_speed < self.speed_scale, torch.full_like(target_speed, down), torch.full_like(target_speed, up))
        self.speed_scale.add_(alpha * (target_speed - self.speed_scale))
        risk_scale = (1.0 - risk).clamp(0, 1)
        lateral_scale = self.cfg.lateral_minimum_scale + (1 - self.cfg.lateral_minimum_scale) * risk_scale
        yaw_scale = self.cfg.yaw_minimum_scale + (1 - self.cfg.yaw_minimum_scale) * risk_scale
        acceleration = self.cfg.risk_acceleration_m_s2 + (self.cfg.normal_acceleration_m_s2 - self.cfg.risk_acceleration_m_s2) * risk_scale
        deceleration = self.cfg.risk_deceleration_m_s2 + (self.cfg.normal_deceleration_m_s2 - self.cfg.risk_deceleration_m_s2) * risk_scale
        push_off = self.cfg.push_off_minimum_scale + (1 - self.cfg.push_off_minimum_scale) * risk_scale
        yaw_limit = self.cfg.normal_yaw_rad_s * yaw_scale
        target = torch.stack((
            raw[:, 0].clamp(-self.cfg.normal_vx_m_s * self.speed_scale[:, 0], self.cfg.normal_vx_m_s * self.speed_scale[:, 0]),
            raw[:, 1].clamp(-self.cfg.normal_vy_m_s * lateral_scale[:, 0], self.cfg.normal_vy_m_s * lateral_scale[:, 0]),
            raw[:, 2].clamp(-yaw_limit[:, 0], yaw_limit[:, 0]),
        ), dim=1)
        growing = target.abs() > self.previous_command.abs()
        rate = torch.where(growing, acceleration, deceleration)
        axis_scale = torch.cat((torch.ones_like(risk), lateral_scale, yaw_scale), dim=1)
        adjusted = self.previous_command + (target - self.previous_command).clamp(-self.cfg.dt * rate * axis_scale, self.cfg.dt * rate * axis_scale)
        self.previous_command.copy_(adjusted)
        state = torch.where(self.limiting, torch.ones_like(self.high_risk_duration, dtype=torch.long), torch.zeros_like(self.high_risk_duration, dtype=torch.long))
        state = torch.where(persistent[:, 0], torch.full_like(state, 2), state)
        state = torch.where(invalid[:, 0], torch.full_like(state, 3), state)
        flags = invalid[:, 0].long() | (persistent[:, 0].long() << 1)
        return TorqueGovernorOutput(adjusted, self.speed_scale.clone(), lateral_scale, yaw_scale, acceleration, deceleration, yaw_limit, push_off, risk, state, flags)
