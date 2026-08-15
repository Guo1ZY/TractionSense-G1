"""Physics-consistent tactile observation randomization for net 3-axis force."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


Range = tuple[float, float]
IntRange = tuple[int, int]


@dataclass
class TactileDomainRandomizationCfg:
    """Central configuration in physical force units (N) and seconds.

    Default ranges are conservative provisional engineering ranges. They are
    not presented as measured statistics because no net-force calibration
    dataset exists in the current real-sensor project.
    """

    dt: float = 0.02
    scale_range: Range = (0.92, 1.08)
    fixed_bias_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    episode_bias_n: Range = (-4.0, 4.0)
    drift_rate_n_sqrt_s: Range = (0.0, 0.6)
    drift_limit_n: float = 6.0
    coupling_offdiag_range: Range = (-0.04, 0.04)
    rotation_deg_range: Range = (-3.0, 3.0)
    delay_steps_range: IntRange = (0, 3)
    lowpass_tau_s_range: Range = (0.0, 0.04)
    noise_floor_n_range: Range = (0.0, 0.8)
    noise_load_fraction_range: Range = (0.0, 0.012)
    saturation_n_range: Range = (450.0, 900.0)
    sample_dropout_probability_range: Range = (0.0, 0.01)
    burst_start_probability_range: Range = (0.0, 0.002)
    burst_length_steps_range: IntRange = (2, 8)
    spike_probability_range: Range = (0.0, 0.001)
    spike_amplitude_n_range: Range = (10.0, 50.0)
    hysteresis_fraction_range: Range = (0.0, 0.03)
    provisional: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.drift_limit_n < 0.0:
            raise ValueError("invalid tactile dt or drift limit")
        for name, limits in vars(self).items():
            if name.endswith("_range"):
                if len(limits) != 2 or limits[1] < limits[0]:
                    raise ValueError(f"invalid {name}={limits}")
        if self.delay_steps_range[0] < 0 or self.burst_length_steps_range[0] < 1:
            raise ValueError("delay and burst length ranges must be non-negative/positive")


@dataclass(frozen=True)
class TactileObservation:
    force_xyz_n: torch.Tensor
    valid: torch.Tensor
    sample_age_s: torch.Tensor


def _uniform(
    shape: tuple[int, ...],
    limits: Range,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low, high = limits
    if low == high:
        return torch.full(shape, float(low), device=device)
    return torch.empty(shape, device=device).uniform_(
        float(low), float(high), generator=generator
    )


def _rotation_matrix_xyz(angles: torch.Tensor) -> torch.Tensor:
    """Return batched intrinsic XYZ small-angle rotation matrices."""
    x, y, z = angles.unbind(dim=-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    result = torch.empty((*angles.shape[:-1], 3, 3), device=angles.device)
    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = cz * sx * sy - cx * sz
    result[..., 0, 2] = sx * sz + cx * cz * sy
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = cx * sy * sz - cz * sx
    result[..., 2, 0] = -sy
    result[..., 2, 1] = cy * sx
    result[..., 2, 2] = cx * cy
    return result


class TactileObservationModel:
    """Vectorized, stateful ideal-force to deployment-force observation model."""

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: TactileDomainRandomizationCfg = TactileDomainRandomizationCfg(),
        device: str | torch.device = "cpu",
        seed: int = 0,
        curriculum_stage: int = 5,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if curriculum_stage not in range(6):
            raise ValueError("curriculum stage must be 0..5")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.curriculum_stage = curriculum_stage
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.max_delay = cfg.delay_steps_range[1]
        self.delay_history = torch.zeros(
            (self.max_delay + 1, num_envs, 2, 3), device=self.device
        )
        self.write_index = 0
        self.filter_state = torch.zeros((num_envs, 2, 3), device=self.device)
        self.last_output = torch.zeros((num_envs, 2, 3), device=self.device)
        self.drift = torch.zeros((num_envs, 2, 3), device=self.device)
        self.age = torch.zeros((num_envs, 2), device=self.device)
        self.burst_remaining = torch.zeros(
            (num_envs, 2), dtype=torch.long, device=self.device
        )
        self.initialized = torch.zeros(
            (num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._allocate_episode_parameters()
        self.reset()

    def _allocate_episode_parameters(self) -> None:
        n = self.num_envs
        self.scale = torch.ones((n, 2, 3), device=self.device)
        self.bias = torch.zeros((n, 2, 3), device=self.device)
        self.drift_rate = torch.zeros((n, 2, 1), device=self.device)
        self.coupling = torch.eye(3, device=self.device).expand(n, 2, 3, 3).clone()
        self.rotation = torch.eye(3, device=self.device).expand(n, 2, 3, 3).clone()
        self.delay_steps = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.tau = torch.zeros((n, 2, 1), device=self.device)
        self.noise_floor = torch.zeros((n, 2, 1), device=self.device)
        self.noise_load_fraction = torch.zeros((n, 2, 1), device=self.device)
        self.saturation = torch.full((n, 2, 3), torch.inf, device=self.device)
        self.sample_dropout_probability = torch.zeros((n, 2), device=self.device)
        self.burst_start_probability = torch.zeros((n, 2), device=self.device)
        self.spike_probability = torch.zeros((n, 2), device=self.device)
        self.spike_amplitude = torch.zeros((n, 2, 1), device=self.device)
        self.hysteresis_fraction = torch.zeros((n, 2, 1), device=self.device)
        self.burst_length = torch.ones((n, 2), dtype=torch.long, device=self.device)

    def set_curriculum_stage(self, stage: int) -> None:
        if stage not in range(6):
            raise ValueError("curriculum stage must be 0..5")
        self.curriculum_stage = stage
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.delay_history.zero_()
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            self.delay_history[:, env_ids] = 0.0
        self.filter_state[env_ids] = 0.0
        self.last_output[env_ids] = 0.0
        self.drift[env_ids] = 0.0
        self.age[env_ids] = 0.0
        self.burst_remaining[env_ids] = 0
        self.initialized[env_ids] = False
        self._sample_episode_parameters(env_ids)

    def _sample_episode_parameters(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        shape3 = (count, 2, 3)
        shape1 = (count, 2, 1)
        stage = self.curriculum_stage

        self.scale[env_ids] = (
            _uniform(
                shape3,
                self.cfg.scale_range,
                device=self.device,
                generator=self.generator,
            )
            if stage >= 1
            else 1.0
        )
        fixed_bias = torch.as_tensor(
            self.cfg.fixed_bias_n, device=self.device
        ).view(1, 1, 3)
        self.bias[env_ids] = (
            fixed_bias
            + _uniform(
                shape3,
                self.cfg.episode_bias_n,
                device=self.device,
                generator=self.generator,
            )
            if stage >= 1
            else 0.0
        )
        self.noise_floor[env_ids] = (
            _uniform(
                shape1,
                self.cfg.noise_floor_n_range,
                device=self.device,
                generator=self.generator,
            )
            if stage >= 1
            else 0.0
        )
        self.noise_load_fraction[env_ids] = (
            _uniform(
                shape1,
                self.cfg.noise_load_fraction_range,
                device=self.device,
                generator=self.generator,
            )
            if stage >= 1
            else 0.0
        )

        if stage >= 2:
            lo, hi = self.cfg.delay_steps_range
            self.delay_steps[env_ids] = torch.randint(
                lo,
                hi + 1,
                (count, 2),
                device=self.device,
                generator=self.generator,
            )
            self.tau[env_ids] = _uniform(
                shape1,
                self.cfg.lowpass_tau_s_range,
                device=self.device,
                generator=self.generator,
            )
        else:
            self.delay_steps[env_ids] = 0
            self.tau[env_ids] = 0.0

        identity = torch.eye(3, device=self.device).expand(count, 2, 3, 3)
        if stage >= 3:
            coupling = _uniform(
                (count, 2, 3, 3),
                self.cfg.coupling_offdiag_range,
                device=self.device,
                generator=self.generator,
            )
            diagonal = torch.arange(3, device=self.device)
            coupling[..., diagonal, diagonal] = 1.0
            self.coupling[env_ids] = coupling
            angles = _uniform(
                shape3,
                self.cfg.rotation_deg_range,
                device=self.device,
                generator=self.generator,
            ) * (math.pi / 180.0)
            self.rotation[env_ids] = _rotation_matrix_xyz(angles)
        else:
            self.coupling[env_ids] = identity
            self.rotation[env_ids] = identity

        if stage >= 4:
            self.drift_rate[env_ids] = _uniform(
                shape1,
                self.cfg.drift_rate_n_sqrt_s,
                device=self.device,
                generator=self.generator,
            )
            self.sample_dropout_probability[env_ids] = _uniform(
                (count, 2),
                self.cfg.sample_dropout_probability_range,
                device=self.device,
                generator=self.generator,
            )
            self.burst_start_probability[env_ids] = _uniform(
                (count, 2),
                self.cfg.burst_start_probability_range,
                device=self.device,
                generator=self.generator,
            )
            self.spike_probability[env_ids] = _uniform(
                (count, 2),
                self.cfg.spike_probability_range,
                device=self.device,
                generator=self.generator,
            )
            self.spike_amplitude[env_ids] = _uniform(
                shape1,
                self.cfg.spike_amplitude_n_range,
                device=self.device,
                generator=self.generator,
            )
            burst_lo, burst_hi = self.cfg.burst_length_steps_range
            self.burst_length[env_ids] = torch.randint(
                burst_lo,
                burst_hi + 1,
                (count, 2),
                device=self.device,
                generator=self.generator,
            )
        else:
            self.drift_rate[env_ids] = 0.0
            self.sample_dropout_probability[env_ids] = 0.0
            self.burst_start_probability[env_ids] = 0.0
            self.spike_probability[env_ids] = 0.0
            self.spike_amplitude[env_ids] = 0.0
            self.burst_length[env_ids] = 1

        if stage >= 5:
            saturation = _uniform(
                (count, 2, 1),
                self.cfg.saturation_n_range,
                device=self.device,
                generator=self.generator,
            )
            self.saturation[env_ids] = saturation.expand(-1, -1, 3)
            self.hysteresis_fraction[env_ids] = _uniform(
                shape1,
                self.cfg.hysteresis_fraction_range,
                device=self.device,
                generator=self.generator,
            )
        else:
            self.saturation[env_ids] = torch.inf
            self.hysteresis_fraction[env_ids] = 0.0

    def __call__(self, true_force_xyz_n: torch.Tensor) -> TactileObservation:
        if true_force_xyz_n.shape == (self.num_envs, 6):
            force = true_force_xyz_n.reshape(self.num_envs, 2, 3)
        elif true_force_xyz_n.shape == (self.num_envs, 2, 3):
            force = true_force_xyz_n
        else:
            raise ValueError(
                f"force shape {tuple(true_force_xyz_n.shape)}, expected "
                f"{(self.num_envs, 6)} or {(self.num_envs, 2, 3)}"
            )
        force = torch.nan_to_num(
            force.to(device=self.device, dtype=torch.float32)
        )
        self.delay_history[self.write_index] = force
        env_index = torch.arange(self.num_envs, device=self.device)[:, None]
        foot_index = torch.arange(2, device=self.device)[None, :]
        read_index = (self.write_index - self.delay_steps) % (self.max_delay + 1)
        delayed = self.delay_history[read_index, env_index, foot_index]
        self.write_index = (self.write_index + 1) % (self.max_delay + 1)

        rotated = torch.matmul(self.rotation, delayed.unsqueeze(-1)).squeeze(-1)
        measurement = torch.matmul(
            self.coupling, rotated.unsqueeze(-1)
        ).squeeze(-1)
        measurement = measurement * self.scale + self.bias

        drift_noise = torch.randn(
            self.drift.shape,
            device=self.device,
            generator=self.generator,
        )
        self.drift.add_(
            drift_noise
            * self.drift_rate
            * math.sqrt(self.cfg.dt)
        ).clamp_(-self.cfg.drift_limit_n, self.cfg.drift_limit_n)
        measurement = measurement + self.drift

        load = torch.linalg.vector_norm(delayed, dim=-1, keepdim=True)
        noise_std = self.noise_floor + self.noise_load_fraction * load
        measurement = measurement + torch.randn(
            measurement.shape,
            device=self.device,
            generator=self.generator,
        ) * noise_std

        spike = torch.rand(
            (self.num_envs, 2),
            device=self.device,
            generator=self.generator,
        ) < self.spike_probability
        spike_direction = torch.randn(
            measurement.shape,
            device=self.device,
            generator=self.generator,
        )
        spike_direction = spike_direction / torch.linalg.vector_norm(
            spike_direction, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        measurement = measurement + (
            spike.unsqueeze(-1) * spike_direction * self.spike_amplitude
        )

        alpha = self.cfg.dt / (self.tau + self.cfg.dt)
        filtered = self.filter_state + alpha * (measurement - self.filter_state)
        filtered = filtered + self.hysteresis_fraction * (
            self.filter_state - filtered
        )
        self.filter_state.copy_(filtered)

        saturated = torch.maximum(
            torch.minimum(filtered, self.saturation), -self.saturation
        )
        sample_dropout = torch.rand(
            (self.num_envs, 2),
            device=self.device,
            generator=self.generator,
        ) < self.sample_dropout_probability
        start_burst = torch.rand(
            (self.num_envs, 2),
            device=self.device,
            generator=self.generator,
        ) < self.burst_start_probability
        self.burst_remaining = torch.where(
            start_burst & (self.burst_remaining == 0),
            self.burst_length,
            self.burst_remaining,
        )
        burst_dropout = self.burst_remaining > 0
        dropout = sample_dropout | burst_dropout
        self.burst_remaining = torch.where(
            burst_dropout,
            (self.burst_remaining - 1).clamp_min(0),
            self.burst_remaining,
        )
        valid = ~dropout
        output = torch.where(valid.unsqueeze(-1), saturated, self.last_output)
        self.last_output.copy_(output)
        self.age = torch.where(
            valid,
            torch.zeros_like(self.age),
            self.age + self.cfg.dt,
        )
        self.initialized |= valid
        return TactileObservation(
            force_xyz_n=output.reshape(self.num_envs, 6),
            valid=valid,
            sample_age_s=self.age.clone(),
        )
