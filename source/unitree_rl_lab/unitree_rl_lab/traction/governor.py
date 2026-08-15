"""Slip-aware traction-adaptive command governor shared by all runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class TractionAwareCommandGovernorCfg:
    dt: float = 0.02
    risk_enter: float = 0.45
    risk_exit: float = 0.25
    persistent_slip_s: float = 0.12
    minimum_hold_s: float = 0.20
    risk_debounce_s: float = 0.06
    fast_down_time_constant_s: float = 0.08
    slow_recovery_time_constant_s: float = 0.80
    normal_max_vx: float = 1.5
    normal_max_vy: float = 0.6
    normal_max_yaw: float = 1.2
    minimum_speed_scale: float = 0.22
    persistent_speed_scale: float = 0.22
    invalid_sensor_speed_scale: float = 0.45
    minimum_confidence: float = 0.25
    traction_score_warning: float = 0.55
    traction_score_critical: float = 0.20
    normal_acceleration_limit: float = 2.0
    low_traction_acceleration_limit: float = 0.35
    normal_deceleration_limit: float = 2.5
    low_traction_deceleration_limit: float = 0.80
    lateral_minimum_scale: float = 0.30
    yaw_minimum_scale: float = 0.25
    push_off_minimum_scale: float = 0.30

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("governor dt must be positive")
        if not 0.0 <= self.risk_exit <= self.risk_enter <= 1.0:
            raise ValueError("risk hysteresis thresholds are invalid")
        if self.risk_debounce_s < 0.0:
            raise ValueError("risk debounce must be non-negative")
        if not (
            0.0
            <= self.traction_score_critical
            < self.traction_score_warning
            <= 1.0
        ):
            raise ValueError("traction score thresholds are invalid")
        for value in (
            self.minimum_speed_scale,
            self.persistent_speed_scale,
            self.invalid_sensor_speed_scale,
            self.minimum_confidence,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("governor scales/confidence must be within [0,1]")


class GovernorOutput(NamedTuple):
    adjusted_command: torch.Tensor
    acceleration_limit: torch.Tensor
    deceleration_limit: torch.Tensor
    push_off_scale: torch.Tensor
    speed_scale: torch.Tensor
    yaw_limit: torch.Tensor
    slip_risk: torch.Tensor
    state: torch.Tensor


class TractionAwareCommandGovernor:
    """Fast-down/slow-recovery command governor with safety fallback.

    State codes: 0 normal, 1 traction limiting, 2 persistent slip, 3 sensor
    fallback. Ground-friction truth is intentionally not an input.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: TractionAwareCommandGovernorCfg = TractionAwareCommandGovernorCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.speed_scale = torch.ones((num_envs, 1), device=self.device)
        self.previous_command = torch.zeros((num_envs, 3), device=self.device)
        self.limiting = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.hold_remaining = torch.zeros(num_envs, device=self.device)
        self.risk_high_duration = torch.zeros(num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.speed_scale.fill_(1.0)
            self.previous_command.zero_()
            self.limiting.zero_()
            self.hold_remaining.zero_()
            self.risk_high_duration.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.speed_scale[env_ids] = 1.0
        self.previous_command[env_ids] = 0.0
        self.limiting[env_ids] = False
        self.hold_remaining[env_ids] = 0.0
        self.risk_high_duration[env_ids] = 0.0

    def update(
        self,
        raw_command: torch.Tensor,
        slip_probability: torch.Tensor,
        traction_score: torch.Tensor,
        sensor_confidence: torch.Tensor,
        slip_duration: torch.Tensor,
        current_velocity: torch.Tensor,
    ) -> GovernorOutput:
        expected = {
            "raw_command": (self.num_envs, 3),
            "slip_probability": (self.num_envs, 2),
            "traction_score": (self.num_envs, 1),
            "sensor_confidence": (self.num_envs, 1),
            "slip_duration": (self.num_envs, 2),
            "current_velocity": (self.num_envs, 3),
        }
        values = {
            "raw_command": raw_command,
            "slip_probability": slip_probability,
            "traction_score": traction_score,
            "sensor_confidence": sensor_confidence,
            "slip_duration": slip_duration,
            "current_velocity": current_velocity,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} shape {values[name].shape}, expected {shape}")
        raw = torch.nan_to_num(raw_command.to(self.device))
        slip_probability = torch.nan_to_num(slip_probability.to(self.device)).clamp(0, 1)
        traction_score = torch.nan_to_num(traction_score.to(self.device)).clamp(0, 1)
        confidence = torch.nan_to_num(sensor_confidence.to(self.device)).clamp(0, 1)
        duration = torch.nan_to_num(slip_duration.to(self.device)).clamp_min(0)
        current_velocity = torch.nan_to_num(current_velocity.to(self.device))
        del current_velocity  # validated and reserved for a learned dynamics extension

        # ``traction_score`` is a physical margin, not a probability. A
        # healthy non-zero utilization therefore need not score 1.0. Map the
        # configurable warning/critical band onto risk before fusing it with
        # the probabilistic slip estimate.
        traction_risk = (
            (self.cfg.traction_score_warning - traction_score)
            / (
                self.cfg.traction_score_warning
                - self.cfg.traction_score_critical
            )
        ).clamp(0.0, 1.0)
        learned_risk = torch.maximum(
            slip_probability.max(dim=1, keepdim=True).values,
            traction_risk,
        )
        low_confidence = confidence < self.cfg.minimum_confidence
        # Low confidence does not masquerade as a traction estimate. It selects
        # a distinct conservative fallback below.
        risk = learned_risk * confidence + self.cfg.risk_enter * (1.0 - confidence)
        risk_high = risk[:, 0] >= self.cfg.risk_enter
        self.risk_high_duration = torch.where(
            risk_high,
            self.risk_high_duration + self.cfg.dt,
            torch.zeros_like(self.risk_high_duration),
        )
        enter = self.risk_high_duration >= self.cfg.risk_debounce_s
        exit_allowed = risk[:, 0] <= self.cfg.risk_exit
        self.hold_remaining = torch.where(
            enter,
            torch.full_like(self.hold_remaining, self.cfg.minimum_hold_s),
            (self.hold_remaining - self.cfg.dt).clamp_min(0.0),
        )
        keep = (~exit_allowed) | (self.hold_remaining > 0.0)
        self.limiting = torch.where(self.limiting, keep, enter)

        persistent = duration.max(dim=1, keepdim=True).values >= self.cfg.persistent_slip_s
        target_scale = 1.0 - (1.0 - self.cfg.minimum_speed_scale) * risk
        target_scale = torch.where(
            self.limiting[:, None],
            target_scale,
            torch.ones_like(target_scale),
        )
        target_scale = torch.where(
            persistent,
            torch.minimum(
                target_scale,
                torch.full_like(target_scale, self.cfg.persistent_speed_scale),
            ),
            target_scale,
        )
        target_scale = torch.where(
            low_confidence,
            torch.minimum(
                target_scale,
                torch.full_like(target_scale, self.cfg.invalid_sensor_speed_scale),
            ),
            target_scale,
        ).clamp(self.cfg.minimum_speed_scale, 1.0)

        down_alpha = min(1.0, self.cfg.dt / self.cfg.fast_down_time_constant_s)
        up_alpha = min(1.0, self.cfg.dt / self.cfg.slow_recovery_time_constant_s)
        alpha = torch.where(
            target_scale < self.speed_scale,
            torch.full_like(target_scale, down_alpha),
            torch.full_like(target_scale, up_alpha),
        )
        self.speed_scale.add_(alpha * (target_scale - self.speed_scale))

        lateral_scale = self.cfg.lateral_minimum_scale + (
            1.0 - self.cfg.lateral_minimum_scale
        ) * self.speed_scale
        yaw_scale = self.cfg.yaw_minimum_scale + (
            1.0 - self.cfg.yaw_minimum_scale
        ) * self.speed_scale
        vx_limit = self.cfg.normal_max_vx * self.speed_scale
        vy_limit = self.cfg.normal_max_vy * lateral_scale
        yaw_limit = self.cfg.normal_max_yaw * yaw_scale
        target = torch.stack(
            (
                raw[:, 0].clamp(-vx_limit[:, 0], vx_limit[:, 0]),
                raw[:, 1].clamp(-vy_limit[:, 0], vy_limit[:, 0]),
                raw[:, 2].clamp(-yaw_limit[:, 0], yaw_limit[:, 0]),
            ),
            dim=1,
        )

        acceleration_limit = self.cfg.low_traction_acceleration_limit + (
            self.cfg.normal_acceleration_limit
            - self.cfg.low_traction_acceleration_limit
        ) * self.speed_scale
        deceleration_limit = self.cfg.low_traction_deceleration_limit + (
            self.cfg.normal_deceleration_limit
            - self.cfg.low_traction_deceleration_limit
        ) * self.speed_scale
        growing = target.abs() > self.previous_command.abs()
        rate_limit = torch.where(
            growing,
            acceleration_limit,
            deceleration_limit,
        )
        # Yaw and lateral change rates are additionally scaled by their caps.
        axis_factor = torch.cat(
            (torch.ones_like(self.speed_scale), lateral_scale, yaw_scale), dim=1
        )
        maximum_delta = self.cfg.dt * rate_limit * axis_factor
        delta = (target - self.previous_command).clamp(-maximum_delta, maximum_delta)
        adjusted = self.previous_command + delta
        self.previous_command.copy_(adjusted)

        push_off_scale = self.cfg.push_off_minimum_scale + (
            1.0 - self.cfg.push_off_minimum_scale
        ) * self.speed_scale
        state = torch.where(
            low_confidence[:, 0],
            torch.full((self.num_envs,), 3, device=self.device, dtype=torch.long),
            torch.where(
                persistent[:, 0],
                torch.full((self.num_envs,), 2, device=self.device, dtype=torch.long),
                self.limiting.long(),
            ),
        )
        return GovernorOutput(
            adjusted,
            acceleration_limit,
            deceleration_limit,
            push_off_scale,
            self.speed_scale.clone(),
            yaw_limit,
            risk,
            state,
        )
