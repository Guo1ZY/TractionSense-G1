"""Provisional deployment-signal and dynamics-estimator randomization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class TorqueDynamicsRandomizationCfg:
    """All ranges are provisional engineering priors, not G1 measurements."""

    provisional: bool = True
    curriculum_stage: int = 0
    dt: float = 0.02
    maximum_delay_frames: int = 3
    tau_scale_range: tuple[float, float] = (0.94, 1.06)
    tau_episode_bias_nm: tuple[float, float] = (-1.5, 1.5)
    tau_bias_drift_nm_sqrt_s: float = 0.10
    tau_noise_std_nm: float = 0.45
    tau_quantization_nm: float = 0.05
    tau_saturation_nm: float = 120.0
    tau_delay_frames: tuple[int, int] = (0, 2)
    state_dropout_probability: float = 0.002
    joint_position_noise_std_rad: float = 0.0015
    joint_velocity_noise_std_rad_s: float = 0.025
    imu_acceleration_noise_std_m_s2: float = 0.08
    imu_acceleration_bias_m_s2: tuple[float, float] = (-0.12, 0.12)
    imu_delay_frames: tuple[int, int] = (0, 2)
    qdd_noise_std_rad_s2: float = 0.8
    mass_scale_range: tuple[float, float] = (0.94, 1.06)
    inertia_scale_range: tuple[float, float] = (0.92, 1.08)
    com_offset_m_range: tuple[float, float] = (-0.008, 0.008)
    joint_friction_nm_range: tuple[float, float] = (0.0, 0.35)
    motor_damping_scale_range: tuple[float, float] = (0.90, 1.10)
    pd_gain_scale_range: tuple[float, float] = (0.92, 1.08)
    gear_efficiency_range: tuple[float, float] = (0.92, 1.0)
    jacobian_scale_range: tuple[float, float] = (0.97, 1.03)
    jacobian_coupling_std: float = 0.008
    bias_force_scale_range: tuple[float, float] = (0.95, 1.05)

    def __post_init__(self) -> None:
        if not 0 <= self.curriculum_stage <= 5:
            raise ValueError("curriculum_stage must be within [0,5]")
        if self.maximum_delay_frames < max(*self.tau_delay_frames, *self.imu_delay_frames):
            raise ValueError("maximum_delay_frames is smaller than a configured delay")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["stage_semantics"] = {
            "0": "ideal torque and exact estimator dynamics",
            "1": "torque scale, bias, noise and quantization",
            "2": "signal delay and filtering-compatible dropout",
            "3": "mass, inertia, COM, Jacobian and PD/model mismatch",
            "4": "bias drift, dropout, quantization and saturation",
            "5": "combined full provisional randomization",
        }
        return result


class RandomizedTorqueDynamicsSignals(NamedTuple):
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    joint_acceleration: torch.Tensor
    tau_est: torch.Tensor
    imu_linear_acceleration: torch.Tensor
    mass_matrix: torch.Tensor
    bias_force: torch.Tensor
    leg_jacobian: torch.Tensor
    valid: torch.Tensor
    tau_age_s: torch.Tensor
    imu_age_s: torch.Tensor


class TorqueDynamicsObservationModel:
    """Batched randomization of sensing and estimator-model inputs only."""

    def __init__(self, num_envs: int, num_joints: int, num_velocity_dofs: int, *, cfg: TorqueDynamicsRandomizationCfg = TorqueDynamicsRandomizationCfg(), device: str | torch.device = "cpu", seed: int = 0, leg_joint_indices: tuple[tuple[int, ...], tuple[int, ...]] | None = None) -> None:
        self.num_envs, self.num_joints, self.num_velocity_dofs = num_envs, num_joints, num_velocity_dofs
        self.cfg, self.device = cfg, torch.device(device)
        self.leg_joint_indices = leg_joint_indices
        if leg_joint_indices is not None and any(len(indices) != 6 for indices in leg_joint_indices):
            raise ValueError("each leg must specify exactly six joint indices")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        length = cfg.maximum_delay_frames + 1
        self.tau_history = torch.zeros(length, num_envs, num_joints, device=self.device)
        self.imu_history = torch.zeros(length, num_envs, 3, device=self.device)
        self.history_index = 0
        self.tau_drift = torch.zeros(num_envs, num_joints, device=self.device)
        n, j = num_envs, num_joints
        self.tau_scale = torch.ones(n, j, device=self.device)
        self.tau_bias = torch.zeros(n, j, device=self.device)
        self.imu_bias = torch.zeros(n, 3, device=self.device)
        self.tau_delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self.imu_delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self.mass_scale = torch.ones(n, 1, 1, device=self.device)
        self.bias_scale = torch.ones(n, 1, device=self.device)
        self.jacobian_scale = torch.ones(n, 2, 1, 1, device=self.device)
        self.jacobian_coupling = torch.zeros(n, 2, 3, 3, device=self.device)
        self.gear_efficiency = torch.ones(n, j, device=self.device)
        self.reset()

    def _uniform(self, shape: tuple[int, ...], limits: tuple[float, float]) -> torch.Tensor:
        return limits[0] + (limits[1] - limits[0]) * torch.rand(shape, generator=self.generator, device=self.device)

    def _normal(self, shape: tuple[int, ...], std: float) -> torch.Tensor:
        return std * torch.randn(shape, generator=self.generator, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        stage = self.cfg.curriculum_stage
        self.tau_history[:, ids] = 0.0
        self.imu_history[:, ids] = 0.0
        self.tau_drift[ids] = 0.0
        self.tau_scale[ids] = self._uniform((len(ids), self.num_joints), self.cfg.tau_scale_range) if stage >= 1 else 1.0
        self.tau_bias[ids] = self._uniform((len(ids), self.num_joints), self.cfg.tau_episode_bias_nm) if stage >= 1 else 0.0
        self.imu_bias[ids] = self._uniform((len(ids), 3), self.cfg.imu_acceleration_bias_m_s2) if stage >= 1 else 0.0
        if stage >= 2:
            low, high = self.cfg.tau_delay_frames
            self.tau_delay[ids] = torch.randint(low, high + 1, (len(ids),), generator=self.generator, device=self.device)
            low, high = self.cfg.imu_delay_frames
            self.imu_delay[ids] = torch.randint(low, high + 1, (len(ids),), generator=self.generator, device=self.device)
        else:
            self.tau_delay[ids] = 0
            self.imu_delay[ids] = 0
        if stage >= 3:
            self.mass_scale[ids] = self._uniform((len(ids), 1, 1), self.cfg.mass_scale_range)
            self.bias_scale[ids] = self._uniform((len(ids), 1), self.cfg.bias_force_scale_range)
            self.jacobian_scale[ids] = self._uniform((len(ids), 2, 1, 1), self.cfg.jacobian_scale_range)
            coupling = self._normal((len(ids), 2, 3, 3), self.cfg.jacobian_coupling_std)
            coupling.diagonal(dim1=-2, dim2=-1).zero_()
            self.jacobian_coupling[ids] = coupling
            self.gear_efficiency[ids] = self._uniform((len(ids), self.num_joints), self.cfg.gear_efficiency_range)
        else:
            self.mass_scale[ids], self.bias_scale[ids], self.jacobian_scale[ids] = 1.0, 1.0, 1.0
            self.jacobian_coupling[ids], self.gear_efficiency[ids] = 0.0, 1.0

    def _delayed(self, history: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        env = torch.arange(self.num_envs, device=self.device)
        return history[(self.history_index - delay) % history.shape[0], env]

    def update(self, *, joint_position: torch.Tensor, joint_velocity: torch.Tensor, joint_acceleration: torch.Tensor, tau_est: torch.Tensor, imu_linear_acceleration: torch.Tensor, mass_matrix: torch.Tensor, bias_force: torch.Tensor, leg_jacobian: torch.Tensor) -> RandomizedTorqueDynamicsSignals:
        stage = self.cfg.curriculum_stage
        q, dq = torch.nan_to_num(joint_position.to(self.device)).clone(), torch.nan_to_num(joint_velocity.to(self.device)).clone()
        qdd, tau = torch.nan_to_num(joint_acceleration.to(self.device)).clone(), torch.nan_to_num(tau_est.to(self.device)).clone()
        imu = torch.nan_to_num(imu_linear_acceleration.to(self.device)).clone()
        if q.shape != (self.num_envs, self.num_joints) or dq.shape != q.shape or qdd.shape != q.shape or tau.shape != q.shape:
            raise ValueError("joint signal shape mismatch")
        if mass_matrix.shape != (self.num_envs, self.num_velocity_dofs, self.num_velocity_dofs) or bias_force.shape != (self.num_envs, self.num_velocity_dofs):
            raise ValueError("full-body dynamics shape mismatch")
        if leg_jacobian.shape != (self.num_envs, 2, 3, 6):
            raise ValueError("leg Jacobian shape mismatch")
        if stage >= 1:
            q += self._normal(q.shape, self.cfg.joint_position_noise_std_rad)
            dq += self._normal(dq.shape, self.cfg.joint_velocity_noise_std_rad_s)
            qdd += self._normal(qdd.shape, self.cfg.qdd_noise_std_rad_s2)
            tau = tau * self.tau_scale * self.gear_efficiency + self.tau_bias + self._normal(tau.shape, self.cfg.tau_noise_std_nm)
            imu += self.imu_bias + self._normal(imu.shape, self.cfg.imu_acceleration_noise_std_m_s2)
        if stage >= 4:
            self.tau_drift += self._normal(self.tau_drift.shape, self.cfg.tau_bias_drift_nm_sqrt_s * self.cfg.dt**0.5)
            tau += self.tau_drift
            tau = torch.round(tau / self.cfg.tau_quantization_nm) * self.cfg.tau_quantization_nm
            tau.clamp_(-self.cfg.tau_saturation_nm, self.cfg.tau_saturation_nm)
        self.history_index = (self.history_index + 1) % self.tau_history.shape[0]
        self.tau_history[self.history_index], self.imu_history[self.history_index] = tau, imu
        tau, imu = self._delayed(self.tau_history, self.tau_delay), self._delayed(self.imu_history, self.imu_delay)
        valid = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        if stage >= 4:
            valid &= ~(torch.rand((self.num_envs, 2), generator=self.generator, device=self.device) < self.cfg.state_dropout_probability)
            if self.leg_joint_indices is None:
                if self.num_joints != 12:
                    raise RuntimeError("full-joint dropout requires semantic leg_joint_indices")
                tau = torch.where(valid.repeat_interleave(6, dim=1), tau, torch.zeros_like(tau))
            else:
                tau = tau.clone()
                for leg, indices in enumerate(self.leg_joint_indices):
                    tau[:, list(indices)] = torch.where(
                        valid[:, leg : leg + 1],
                        tau[:, list(indices)],
                        torch.zeros_like(tau[:, list(indices)]),
                    )
        mass = torch.nan_to_num(mass_matrix.to(self.device)) * self.mass_scale
        bias = torch.nan_to_num(bias_force.to(self.device)) * self.bias_scale
        jacobian = torch.nan_to_num(leg_jacobian.to(self.device)) * self.jacobian_scale
        jacobian = torch.einsum("blij,bljk->blik", torch.eye(3, device=self.device).view(1, 1, 3, 3) + self.jacobian_coupling, jacobian)
        return RandomizedTorqueDynamicsSignals(q, dq, qdd, tau, imu, mass, bias, jacobian, valid, self.tau_delay.float() * self.cfg.dt, self.imu_delay.float() * self.cfg.dt)
