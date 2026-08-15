"""Batched traction diagnostics and hysteretic slip labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass
class TractionDiagnosticsCfg:
    contact_force_on: float = 12.0
    contact_force_off: float = 6.0
    slip_speed_on: float = 0.12
    slip_speed_off: float = 0.06
    minimum_slip_duration: float = 0.04
    dt: float = 0.02
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.contact_force_off <= self.contact_force_on:
            raise ValueError("contact force thresholds must satisfy 0 <= off <= on")
        if not 0.0 <= self.slip_speed_off <= self.slip_speed_on:
            raise ValueError("slip speed thresholds must satisfy 0 <= off <= on")
        if self.minimum_slip_duration < 0.0 or self.dt <= 0.0:
            raise ValueError("invalid duration or dt")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


class TractionDiagnostics(NamedTuple):
    force_normal: torch.Tensor
    force_tangent: torch.Tensor
    friction_utilization: torch.Tensor
    contact: torch.Tensor
    foot_tangent_velocity: torch.Tensor
    slip_speed: torch.Tensor
    slip_label: torch.Tensor
    slip_duration: torch.Tensor
    support_load_ratio: torch.Tensor
    velocity_is_proxy: bool


class TractionDiagnosticsState:
    """Stateful two-foot contact/slip hysteresis without environment loops."""

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: TractionDiagnosticsCfg = TractionDiagnosticsCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.cfg = cfg
        self.contact = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.slip = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.candidate_duration = torch.zeros((num_envs, 2), device=device)
        self.slip_duration = torch.zeros((num_envs, 2), device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.contact.zero_()
            self.slip.zero_()
            self.candidate_duration.zero_()
            self.slip_duration.zero_()
            return
        env_ids = env_ids.to(device=self.contact.device, dtype=torch.long)
        self.contact[env_ids] = False
        self.slip[env_ids] = False
        self.candidate_duration[env_ids] = 0.0
        self.slip_duration[env_ids] = 0.0

    def update(
        self,
        force_local_n: torch.Tensor,
        tangent_velocity_m_s: torch.Tensor,
        *,
        velocity_is_proxy: bool,
    ) -> TractionDiagnostics:
        if force_local_n.shape == (self.num_envs, 6):
            force = force_local_n.reshape(self.num_envs, 2, 3)
        elif force_local_n.shape == (self.num_envs, 2, 3):
            force = force_local_n
        else:
            raise ValueError(
                f"force shape {tuple(force_local_n.shape)}, expected "
                f"{(self.num_envs, 6)} or {(self.num_envs, 2, 3)}"
            )
        if tangent_velocity_m_s.shape == (self.num_envs, 2, 2):
            velocity = tangent_velocity_m_s
        else:
            raise ValueError(
                f"tangent velocity shape {tuple(tangent_velocity_m_s.shape)}, "
                f"expected {(self.num_envs, 2, 2)}"
            )
        force = torch.nan_to_num(force)
        velocity = torch.nan_to_num(velocity)
        normal = force[..., 2].abs()
        tangent = torch.linalg.vector_norm(force[..., :2], dim=-1)
        utilization = tangent / (normal + self.cfg.epsilon)
        slip_speed = torch.linalg.vector_norm(velocity, dim=-1)

        contact_on = normal > self.cfg.contact_force_on
        contact_keep = normal >= self.cfg.contact_force_off
        self.contact = torch.where(self.contact, contact_keep, contact_on)

        entering = self.contact & (slip_speed > self.cfg.slip_speed_on)
        self.candidate_duration = torch.where(
            entering,
            self.candidate_duration + self.cfg.dt,
            torch.zeros_like(self.candidate_duration),
        )
        enter_slip = self.candidate_duration >= self.cfg.minimum_slip_duration
        keep_slip = self.contact & (slip_speed >= self.cfg.slip_speed_off)
        self.slip = torch.where(self.slip, keep_slip, enter_slip)
        self.slip_duration = torch.where(
            self.slip,
            self.slip_duration + self.cfg.dt,
            torch.zeros_like(self.slip_duration),
        )

        load_sum = normal.sum(dim=1, keepdim=True)
        support_ratio = torch.where(
            load_sum > self.cfg.epsilon,
            normal / load_sum.clamp_min(self.cfg.epsilon),
            torch.zeros_like(normal),
        )
        return TractionDiagnostics(
            force_normal=normal,
            force_tangent=tangent,
            friction_utilization=utilization,
            contact=self.contact.clone(),
            foot_tangent_velocity=velocity,
            slip_speed=slip_speed,
            slip_label=self.slip.clone(),
            slip_duration=self.slip_duration.clone(),
            support_load_ratio=support_ratio,
            velocity_is_proxy=velocity_is_proxy,
        )
