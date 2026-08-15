"""Per-channel baseline, temperature compensation and response normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import NUM_SENSORS


FORMAT = "g1-foot-magnetic-normalization-v1"
TEMP_X10_MIN = -400
TEMP_X10_MAX = 1250


@dataclass(frozen=True)
class MagneticNormalizer:
    side: str
    baseline_xyz: np.ndarray
    scale_xyz: np.ndarray
    reference_temperature_x10: np.ndarray
    temperature_coefficient_per_x10: np.ndarray
    bad_temperature_channels: tuple[int, ...] = ()
    clip: float = 6.0

    def normalize(
        self, magnetic_xyz: np.ndarray, temperature_x10: np.ndarray
    ) -> np.ndarray:
        magnetic = np.asarray(magnetic_xyz, dtype=np.float64)
        temperature = np.asarray(temperature_x10, dtype=np.float64)
        if magnetic.shape != (NUM_SENSORS, 3):
            raise ValueError("magnetic_xyz must be 15x3")
        if temperature.shape != (NUM_SENSORS,):
            raise ValueError("temperature_x10 must have 15 values")
        thermal_delta = temperature - self.reference_temperature_x10
        invalid = ~np.isfinite(temperature)
        invalid |= (temperature < TEMP_X10_MIN) | (temperature > TEMP_X10_MAX)
        if self.bad_temperature_channels:
            invalid = invalid.copy()
            invalid[list(self.bad_temperature_channels)] = True
        thermal_delta = np.where(invalid, 0.0, thermal_delta)
        compensated_baseline = (
            self.baseline_xyz
            + thermal_delta[:, None] * self.temperature_coefficient_per_x10
        )
        normalized = (magnetic - compensated_baseline) / self.scale_xyz
        if not np.isfinite(normalized).all():
            raise ValueError("normalization produced a non-finite value")
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    @classmethod
    def load(cls, path: Path, expected_side: str | None = None) -> "MagneticNormalizer":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read normalization {path}: {error}") from error
        if payload.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported normalization format")
        side = str(payload.get("side", ""))
        if side not in {"left", "right"} or (expected_side and side != expected_side):
            raise ValueError(f"{path}: side mismatch")
        return cls(
            side=side,
            baseline_xyz=_array(payload.get("baseline_xyz"), (NUM_SENSORS, 3), "baseline_xyz"),
            scale_xyz=_positive_array(payload.get("scale_xyz"), (NUM_SENSORS, 3), "scale_xyz"),
            reference_temperature_x10=_array(
                payload.get("reference_temperature_x10"),
                (NUM_SENSORS,),
                "reference_temperature_x10",
            ),
            temperature_coefficient_per_x10=_array(
                payload.get("temperature_coefficient_per_x10"),
                (NUM_SENSORS, 3),
                "temperature_coefficient_per_x10",
            ),
            bad_temperature_channels=_channel_list(
                payload.get("bad_temperature_channels", [])
            ),
            clip=float(payload.get("clip", 6.0)),
        )


def _channel_list(value) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("bad_temperature_channels must be a list of channel indexes")
    channels = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("bad_temperature_channels must contain integers")
        if not 0 <= item < NUM_SENSORS:
            raise ValueError(f"bad_temperature_channels entry {item} out of range")
        if item not in channels:
            channels.append(item)
    return tuple(channels)


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _positive_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = _array(value, shape, name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return result

