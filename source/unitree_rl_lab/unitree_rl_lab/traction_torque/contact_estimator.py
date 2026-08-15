"""Deployable hybrid left/right contact probability and state estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class HybridContactEstimatorCfg:
    dt: float = 0.02
    force_midpoint_n: float = 20.0
    force_scale_n: float = 8.0
    height_midpoint_m: float = 0.055
    height_scale_m: float = 0.020
    vertical_velocity_scale_m_s: float = 0.25
    planar_velocity_scale_m_s: float = 0.55
    torque_scale_nm: float = 18.0
    imu_acceleration_scale_m_s2: float = 12.0
    force_weight: float = 0.40
    height_weight: float = 0.20
    velocity_weight: float = 0.18
    torque_weight: float = 0.14
    history_weight: float = 0.08
    probability_lowpass_tau_s: float = 0.04
    on_threshold: float = 0.62
    off_threshold: float = 0.38
    debounce_s: float = 0.04
    minimum_hold_s: float = 0.08

    def __post_init__(self) -> None:
        weights = (
            self.force_weight
            + self.height_weight
            + self.velocity_weight
            + self.torque_weight
            + self.history_weight
        )
        if abs(weights - 1.0) > 1.0e-6:
            raise ValueError("contact feature weights must sum to one")
        if not 0.0 <= self.off_threshold < self.on_threshold <= 1.0:
            raise ValueError("contact hysteresis thresholds are invalid")
        if self.dt <= 0.0 or self.debounce_s < 0.0 or self.minimum_hold_s < 0.0:
            raise ValueError("contact timing is invalid")


class HybridContactInput(NamedTuple):
    foot_height_m: torch.Tensor
    foot_vertical_velocity_m_s: torch.Tensor
    foot_planar_velocity_m_s: torch.Tensor
    leg_torque_nm: torch.Tensor
    estimated_fz_n: torch.Tensor
    joint_configuration: torch.Tensor
    imu_linear_acceleration_m_s2: torch.Tensor
    gait_phase: torch.Tensor | None = None


class HybridContactOutput(NamedTuple):
    probability: torch.Tensor
    state: torch.Tensor
    candidate_duration_s: torch.Tensor
    hold_remaining_s: torch.Tensor


class HybridContactEstimator:
    """Causal contact state machine using only deployment-available signals."""

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: HybridContactEstimatorCfg = HybridContactEstimatorCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        shape = (num_envs, 2)
        self.probability = torch.zeros(shape, device=self.device)
        self.state = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.candidate_duration = torch.zeros(shape, device=self.device)
        self.hold_remaining = torch.zeros(shape, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.probability.zero_()
            self.state.zero_()
            self.candidate_duration.zero_()
            self.hold_remaining.zero_()
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.probability[ids] = 0.0
        self.state[ids] = False
        self.candidate_duration[ids] = 0.0
        self.hold_remaining[ids] = 0.0

    def update(self, value: HybridContactInput) -> HybridContactOutput:
        expected = {
            "foot_height_m": (self.num_envs, 2),
            "foot_vertical_velocity_m_s": (self.num_envs, 2),
            "foot_planar_velocity_m_s": (self.num_envs, 2, 2),
            "leg_torque_nm": (self.num_envs, 2, 6),
            "estimated_fz_n": (self.num_envs, 2),
            "joint_configuration": (self.num_envs, 2, 6),
            "imu_linear_acceleration_m_s2": (self.num_envs, 3),
        }
        for name, shape in expected.items():
            if tuple(getattr(value, name).shape) != shape:
                raise ValueError(f"{name} shape mismatch")
        height = torch.nan_to_num(value.foot_height_m.to(self.device))
        vertical_velocity = torch.nan_to_num(
            value.foot_vertical_velocity_m_s.to(self.device)
        )
        planar_speed = torch.linalg.vector_norm(
            torch.nan_to_num(value.foot_planar_velocity_m_s.to(self.device)), dim=-1
        )
        torque_norm = torch.linalg.vector_norm(
            torch.nan_to_num(value.leg_torque_nm.to(self.device)), dim=-1
        ) / (6.0**0.5)
        fz = torch.nan_to_num(value.estimated_fz_n.to(self.device)).clamp_min(0.0)
        imu_norm = torch.linalg.vector_norm(
            torch.nan_to_num(value.imu_linear_acceleration_m_s2.to(self.device)), dim=-1
        )[:, None]

        force_score = torch.sigmoid(
            (fz - self.cfg.force_midpoint_n) / self.cfg.force_scale_n
        )
        height_score = torch.sigmoid(
            (self.cfg.height_midpoint_m - height) / self.cfg.height_scale_m
        )
        velocity_score = torch.exp(
            -vertical_velocity.abs() / self.cfg.vertical_velocity_scale_m_s
            -planar_speed / self.cfg.planar_velocity_scale_m_s
        )
        torque_score = torch.sigmoid(
            (torque_norm - 0.25 * self.cfg.torque_scale_nm)
            / self.cfg.torque_scale_nm
        )
        imu_gate = torch.exp(-imu_norm / self.cfg.imu_acceleration_scale_m_s2)
        history_score = self.state.float()
        raw = (
            self.cfg.force_weight * force_score
            + self.cfg.height_weight * height_score
            + self.cfg.velocity_weight * velocity_score
            + self.cfg.torque_weight * torque_score
            + self.cfg.history_weight * history_score
        ) * (0.65 + 0.35 * imu_gate)
        if value.gait_phase is not None:
            phase = torch.nan_to_num(value.gait_phase.to(self.device)).clamp(0.0, 1.0)
            if phase.shape != (self.num_envs, 2):
                raise ValueError("gait_phase shape mismatch")
            raw = 0.9 * raw + 0.1 * phase

        tau = self.cfg.probability_lowpass_tau_s
        alpha = 1.0 if tau <= 0.0 else self.cfg.dt / (self.cfg.dt + tau)
        self.probability.add_(alpha * (raw.clamp(0.0, 1.0) - self.probability))
        above = self.probability >= self.cfg.on_threshold
        self.candidate_duration = torch.where(
            above,
            self.candidate_duration + self.cfg.dt,
            torch.zeros_like(self.candidate_duration),
        )
        enter = self.candidate_duration >= self.cfg.debounce_s
        self.hold_remaining = torch.where(
            enter,
            torch.full_like(self.hold_remaining, self.cfg.minimum_hold_s),
            (self.hold_remaining - self.cfg.dt).clamp_min(0.0),
        )
        keep = (self.probability > self.cfg.off_threshold) | (
            self.hold_remaining > 0.0
        )
        self.state = torch.where(self.state, keep, enter)
        return HybridContactOutput(
            self.probability.clone(),
            self.state.clone(),
            self.candidate_duration.clone(),
            self.hold_remaining.clone(),
        )
