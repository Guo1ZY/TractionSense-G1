"""Hall electronics, sampling, filtering, drift, saturation, and auto-zero."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .hall_sensor_config import HallFootSensorCfg


@dataclass(frozen=True)
class HallSignalOutput:
    """One sampled Hall packet; all magnetic values are tesla."""

    ideal: torch.Tensor
    raw: torch.Tensor
    filtered_absolute: torch.Tensor
    processed: torch.Tensor
    baseline: torch.Tensor
    drift: torch.Tensor
    baseline_ready: torch.Tensor


class HallSensorSignalProcessor:
    """Stateful, vectorized electronic signal model for every Hall channel."""

    def __init__(
        self,
        cfg: HallFootSensorCfg,
        num_envs: int,
        num_sensors: int,
        *,
        device: str | torch.device,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.num_sensors = int(num_sensors)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (self.num_envs, 2, self.num_sensors, 3)
        self.ideal = torch.zeros(shape, device=self.device)
        self.raw = torch.zeros(shape, device=self.device)
        self.filtered_absolute = torch.zeros(shape, device=self.device)
        self.processed = torch.zeros(shape, device=self.device)
        self.baseline = torch.zeros(shape, device=self.device)
        self.drift = torch.zeros(shape, device=self.device)
        self._baseline_sum = torch.zeros(shape, device=self.device)
        self._baseline_count = torch.zeros((*shape[:-1], 1), dtype=torch.long, device=self.device)
        self.baseline_ready = torch.zeros((*shape[:-1], 1), dtype=torch.bool, device=self.device)
        self._filter_initialized = torch.zeros((*shape[:-1], 1), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = self._env_ids(env_ids)
        for value in (
            self.ideal,
            self.raw,
            self.filtered_absolute,
            self.processed,
            self.baseline,
            self.drift,
            self._baseline_sum,
        ):
            value[ids] = 0.0
        self._baseline_count[ids] = 0
        self.baseline_ready[ids] = False
        self._filter_initialized[ids] = False

    def update(
        self,
        ideal_field_t: torch.Tensor,
        dt: float,
        *,
        temperature_c: torch.Tensor | float | None = None,
    ) -> HallSignalOutput:
        if ideal_field_t.shape != self.ideal.shape:
            raise ValueError(f"ideal field shape {tuple(ideal_field_t.shape)} != {tuple(self.ideal.shape)}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.ideal.copy_(torch.nan_to_num(ideal_field_t))

        if self.cfg.drift_std_per_sqrt_s > 0.0:
            random_walk = torch.randn(
                self.drift.shape,
                device=self.device,
                generator=self.generator,
            ) * (self.cfg.drift_std_per_sqrt_s * math.sqrt(dt))
            self.drift.add_(random_walk).clamp_(-self.cfg.drift_limit, self.cfg.drift_limit)

        bias = torch.as_tensor(self.cfg.bias, device=self.device, dtype=self.ideal.dtype)
        noise_std = torch.as_tensor(self.cfg.noise_std, device=self.device, dtype=self.ideal.dtype)
        value = self.ideal + bias + self.drift
        if temperature_c is not None:
            temperature = torch.as_tensor(temperature_c, device=self.device, dtype=self.ideal.dtype)
            while temperature.ndim < self.ideal.ndim - 1:
                temperature = temperature.unsqueeze(-1)
            temperature = temperature.unsqueeze(-1) if temperature.ndim == self.ideal.ndim - 1 else temperature
            coefficient = torch.as_tensor(
                self.cfg.temperature_coefficient,
                device=self.device,
                dtype=self.ideal.dtype,
            )
            value = value + (temperature - self.cfg.reference_temperature_c) * coefficient
        if bool(torch.any(noise_std > 0.0)):
            value = value + torch.randn(
                value.shape,
                device=self.device,
                generator=self.generator,
            ) * noise_std

        lower = torch.as_tensor(self.cfg.saturation_min, device=self.device, dtype=value.dtype)
        upper = torch.as_tensor(self.cfg.saturation_max, device=self.device, dtype=value.dtype)
        value = torch.maximum(torch.minimum(value, upper), lower)
        if self.cfg.resolution > 0.0:
            value = torch.round(value / self.cfg.resolution) * self.cfg.resolution
            value = torch.maximum(torch.minimum(value, upper), lower)
        self.raw.copy_(value)

        if self.cfg.low_pass_cutoff > 0.0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.cfg.low_pass_cutoff * dt)
            candidate = self.filtered_absolute + alpha * (self.raw - self.filtered_absolute)
        else:
            candidate = self.raw
        self.filtered_absolute.copy_(
            torch.where(self._filter_initialized, candidate, self.raw)
        )
        self._filter_initialized.fill_(True)

        if self.cfg.auto_zero:
            collecting = ~self.baseline_ready
            self._baseline_sum.add_(torch.where(collecting, self.filtered_absolute, 0.0))
            self._baseline_count.add_(collecting.to(torch.long))
            just_ready = self._baseline_count >= self.cfg.auto_zero_samples
            denominator = self._baseline_count.clamp_min(1).to(self.ideal.dtype)
            estimated = self._baseline_sum / denominator
            self.baseline.copy_(torch.where(just_ready, estimated, self.baseline))
            self.baseline_ready.logical_or_(just_ready)
            self.processed.copy_(
                torch.where(self.baseline_ready, self.filtered_absolute - self.baseline, 0.0)
            )
        else:
            self.baseline.zero_()
            self.baseline_ready.fill_(True)
            self.processed.copy_(self.filtered_absolute)

        return self.output()

    def output(self) -> HallSignalOutput:
        return HallSignalOutput(
            ideal=self.ideal,
            raw=self.raw,
            filtered_absolute=self.filtered_absolute,
            processed=self.processed,
            baseline=self.baseline,
            drift=self.drift,
            baseline_ready=self.baseline_ready,
        )

    def _env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)
