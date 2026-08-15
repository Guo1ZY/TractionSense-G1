"""Causal filters for joint acceleration and deployment-native signals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointStateFilterCfg:
    dt: float = 0.02
    acceleration_lowpass_tau_s: float = 0.04
    acceleration_clip_rad_s2: float = 120.0
    torque_lowpass_tau_s: float = 0.02

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.acceleration_lowpass_tau_s < 0.0 or self.torque_lowpass_tau_s < 0.0:
            raise ValueError("filter time constants must be non-negative")
        if self.acceleration_clip_rad_s2 <= 0.0:
            raise ValueError("acceleration clip must be positive")


class JointStateFilter:
    """Batched finite-difference qdd and torque filtering with reset safety."""

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        *,
        cfg: JointStateFilterCfg = JointStateFilterCfg(),
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0 or num_joints <= 0:
            raise ValueError("filter dimensions must be positive")
        self.cfg = cfg
        self.device = torch.device(device)
        shape = (num_envs, num_joints)
        self.previous_velocity = torch.zeros(shape, device=self.device)
        self.acceleration = torch.zeros(shape, device=self.device)
        self.torque = torch.zeros(shape, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.previous_velocity.zero_()
            self.acceleration.zero_()
            self.torque.zero_()
            self.initialized.zero_()
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.previous_velocity[ids] = 0.0
        self.acceleration[ids] = 0.0
        self.torque[ids] = 0.0
        self.initialized[ids] = False

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return 1.0 if tau <= 0.0 else dt / (dt + tau)

    def update(
        self,
        joint_velocity: torch.Tensor,
        tau_est: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_velocity.shape != self.previous_velocity.shape:
            raise ValueError("joint velocity shape mismatch")
        if tau_est.shape != self.torque.shape:
            raise ValueError("joint torque shape mismatch")
        velocity = torch.nan_to_num(joint_velocity.to(self.device))
        torque = torch.nan_to_num(tau_est.to(self.device))
        raw_acceleration = (velocity - self.previous_velocity) / self.cfg.dt
        raw_acceleration = raw_acceleration.clamp(
            -self.cfg.acceleration_clip_rad_s2,
            self.cfg.acceleration_clip_rad_s2,
        )
        raw_acceleration = torch.where(
            self.initialized[:, None],
            raw_acceleration,
            torch.zeros_like(raw_acceleration),
        )
        acc_alpha = self._alpha(self.cfg.dt, self.cfg.acceleration_lowpass_tau_s)
        tau_alpha = self._alpha(self.cfg.dt, self.cfg.torque_lowpass_tau_s)
        self.acceleration.add_(acc_alpha * (raw_acceleration - self.acceleration))
        self.torque.add_(tau_alpha * (torque - self.torque))
        self.previous_velocity.copy_(velocity)
        self.initialized.fill_(True)
        return self.acceleration.clone(), self.torque.clone()
