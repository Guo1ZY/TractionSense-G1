"""Per-channel baseline, temperature compensation and response normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import NUM_SENSORS


FORMAT = "g1-foot-magnetic-normalization-v1"


@dataclass(frozen=True)
class MagneticNormalizer:
    side: str
    baseline_xyz: np.ndarray
    scale_xyz: np.ndarray
    reference_temperature_x10: np.ndarray
    temperature_coefficient_per_x10: np.ndarray
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
            clip=float(payload.get("clip", 6.0)),
        )


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

