"""Magnetic-array calibration model used by both capture tools and runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import NUM_SENSORS


FORMAT = "g1-foot-magnetic-calibration-v1"
FEATURE_COUNT = NUM_SENSORS * 3


@dataclass(frozen=True)
class ForceEstimate:
    normal_n: float
    tangent_n: float


@dataclass
class Calibration:
    side: str
    baseline_xyz: np.ndarray
    normal_weights: np.ndarray
    normal_bias: float
    tangent_x_weights: np.ndarray | None = None
    tangent_x_bias: float = 0.0
    tangent_y_weights: np.ndarray | None = None
    tangent_y_bias: float = 0.0
    max_normal_n: float = 800.0
    max_tangent_n: float = 500.0

    @property
    def has_tangent(self) -> bool:
        return self.tangent_x_weights is not None and self.tangent_y_weights is not None

    def estimate(self, magnetic_xyz: np.ndarray) -> ForceEstimate:
        magnetic = np.asarray(magnetic_xyz, dtype=np.float64)
        if magnetic.shape != (NUM_SENSORS, 3):
            raise ValueError(f"magnetic_xyz must have shape {(NUM_SENSORS, 3)}")
        features = (magnetic - self.baseline_xyz).reshape(-1)
        normal = float(features @ self.normal_weights + self.normal_bias)
        normal = float(np.clip(normal, 0.0, self.max_normal_n))
        tangent = 0.0
        if self.has_tangent:
            tangent_x = float(features @ self.tangent_x_weights + self.tangent_x_bias)
            tangent_y = float(features @ self.tangent_y_weights + self.tangent_y_bias)
            tangent = min(math.hypot(tangent_x, tangent_y), self.max_tangent_n)
        if not math.isfinite(normal) or not math.isfinite(tangent):
            raise ValueError("calibration produced a non-finite force")
        return ForceEstimate(normal_n=normal, tangent_n=tangent)

    @classmethod
    def load(cls, path: Path, expected_side: str | None = None) -> "Calibration":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read calibration {path}: {error}") from error
        if payload.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported calibration format")
        side = str(payload.get("side", ""))
        if side not in {"left", "right"}:
            raise ValueError(f"{path}: side must be left or right")
        if expected_side and side != expected_side:
            raise ValueError(f"{path}: calibration is for {side}, expected {expected_side}")
        if not payload.get("model_complete", False):
            raise ValueError(f"{path}: only a zero baseline exists; run fit-normal")
        baseline = _array(payload.get("baseline_xyz"), (NUM_SENSORS, 3), "baseline_xyz")
        normal = payload.get("normal", {})
        normal_weights = _array(normal.get("weights"), (FEATURE_COUNT,), "normal.weights")
        tangent_x = _optional_axis(payload.get("tangent_x"), "tangent_x")
        tangent_y = _optional_axis(payload.get("tangent_y"), "tangent_y")
        limits = payload.get("limits_n", {})
        return cls(
            side=side,
            baseline_xyz=baseline,
            normal_weights=normal_weights,
            normal_bias=float(normal.get("bias", 0.0)),
            tangent_x_weights=None if tangent_x is None else tangent_x[0],
            tangent_x_bias=0.0 if tangent_x is None else tangent_x[1],
            tangent_y_weights=None if tangent_y is None else tangent_y[0],
            tangent_y_bias=0.0 if tangent_y is None else tangent_y[1],
            max_normal_n=float(limits.get("normal", 800.0)),
            max_tangent_n=float(limits.get("tangent", 500.0)),
        )


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {shape}, got {result.shape}")
    return result


def _optional_axis(value: Any, label: str) -> tuple[np.ndarray, float] | None:
    if value is None:
        return None
    return (
        _array(value.get("weights"), (FEATURE_COUNT,), f"{label}.weights"),
        float(value.get("bias", 0.0)),
    )


def calibration_document(
    *,
    side: str,
    baseline_xyz: np.ndarray,
    normal_weights: np.ndarray | None = None,
    normal_bias: float = 0.0,
    metrics: dict[str, float | int] | None = None,
    max_normal_n: float = 800.0,
) -> dict[str, Any]:
    complete = normal_weights is not None
    document: dict[str, Any] = {
        "format": FORMAT,
        "side": side,
        "model_complete": complete,
        "feature": "flatten(magnetic_xyz - baseline_xyz), sensor-major XYZ",
        "baseline_xyz": np.asarray(baseline_xyz, dtype=float).tolist(),
        "normal": None,
        "tangent_x": None,
        "tangent_y": None,
        "limits_n": {"normal": float(max_normal_n), "tangent": 500.0},
        "metrics": metrics or {},
        "note": "Tangent is intentionally zero until a calibrated shear-force dataset is fitted.",
    }
    if complete:
        document["normal"] = {
            "weights": np.asarray(normal_weights, dtype=float).tolist(),
            "bias": float(normal_bias),
        }
    return document

