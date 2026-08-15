#!/usr/bin/env python3
"""Pure NumPy Hall-window features and fail-safe friction decision logic."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


MODEL_FORMAT = "g1-dual-foot-hall-friction-linear-v1"
SENSOR_ORDER = "left,right; P00..P14; Bx,By,Bz"


def feature_names() -> List[str]:
    names: List[str] = []
    for side in ("left", "right"):
        for family in (
            "centered_abs_mean",
            "centered_std",
            "centered_ptp",
            "delta_rms",
            "delta_abs_p90",
        ):
            names.extend(f"{side}_{family}_{axis}" for axis in ("bx", "by", "bz"))
        names.extend(
            [
                f"{side}_site_rms_mean",
                f"{side}_site_rms_std",
                f"{side}_site_rms_max",
                f"{side}_region_dynamic_toe",
                f"{side}_region_dynamic_midfoot",
                f"{side}_region_dynamic_heel",
            ]
        )
    names.extend(
        [
            "lr_axis_std_absdiff_bx",
            "lr_axis_std_absdiff_by",
            "lr_axis_std_absdiff_bz",
            "lr_dynamic_energy_log_ratio",
            "lr_dynamic_energy_correlation",
        ]
    )
    return names


FEATURE_NAMES = feature_names()


def _validate_window(hall_xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(hall_xyz, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (2, 15, 3):
        raise ValueError("hall_xyz must have shape [T,2,15,3]")
    if values.shape[0] < 4:
        raise ValueError("Hall window must contain at least four frames")
    if not np.all(np.isfinite(values)):
        raise ValueError("Hall window contains NaN or infinity")
    return values


def extract_window_features(hall_xyz: np.ndarray) -> np.ndarray:
    """Extract baseline-invariant temporal/spatial features from raw Hall counts."""
    values = _validate_window(hall_xyz)
    result: List[float] = []
    axis_std: List[np.ndarray] = []
    dynamic_energy: List[np.ndarray] = []
    for side_index in range(2):
        foot = values[:, side_index]
        centered = foot - np.median(foot, axis=0, keepdims=True)
        delta = np.diff(foot, axis=0)
        result.extend(np.mean(np.abs(centered), axis=(0, 1)).tolist())
        current_axis_std = np.std(centered, axis=(0, 1))
        axis_std.append(current_axis_std)
        result.extend(current_axis_std.tolist())
        result.extend(np.ptp(centered, axis=(0, 1)).tolist())
        result.extend(np.sqrt(np.mean(delta * delta, axis=(0, 1))).tolist())
        result.extend(np.percentile(np.abs(delta), 90.0, axis=(0, 1)).tolist())

        site_rms = np.sqrt(np.mean(centered * centered, axis=(0, 2)))
        result.extend(
            [float(np.mean(site_rms)), float(np.std(site_rms)), float(np.max(site_rms))]
        )
        delta_norm = np.linalg.norm(delta, axis=2)
        dynamic_energy.append(np.mean(delta_norm, axis=1))
        for start in (0, 5, 10):
            result.append(float(np.sqrt(np.mean(delta[:, start : start + 5] ** 2))))

    result.extend(np.abs(axis_std[0] - axis_std[1]).tolist())
    energy_left = float(np.sqrt(np.mean(dynamic_energy[0] ** 2)))
    energy_right = float(np.sqrt(np.mean(dynamic_energy[1] ** 2)))
    result.append(float(math.log((energy_left + 1.0e-6) / (energy_right + 1.0e-6))))
    if np.std(dynamic_energy[0]) < 1.0e-9 or np.std(dynamic_energy[1]) < 1.0e-9:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(dynamic_energy[0], dynamic_energy[1])[0, 1])
    result.append(correlation)

    features = np.asarray(result, dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError(
            f"feature schema mismatch: {features.shape} != {(len(FEATURE_NAMES),)}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("derived Hall features contain NaN or infinity")
    return features


@dataclass(frozen=True)
class LinearFrictionModel:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    probability_temperature: float
    window_frames: int
    nominal_rate_hz: float
    enter_low_probability: float
    clear_low_probability: float
    validation: Dict[str, Any]
    source_sha256: str

    @classmethod
    def load(cls, path: Path, require_passed_gate: bool = True) -> "LinearFrictionModel":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != MODEL_FORMAT:
            raise ValueError(f"model format must be {MODEL_FORMAT}")
        if document.get("measurement") != "Hall Bx/By/Bz temporal features only":
            raise ValueError("model measurement boundary is not Hall-only")
        if document.get("sensor_order") != SENSOR_ORDER:
            raise ValueError("model sensor order does not match runtime")
        if document.get("feature_names") != FEATURE_NAMES:
            raise ValueError("model feature schema does not match runtime")
        validation = dict(document.get("validation", {}))
        if require_passed_gate and validation.get("passed") is not True:
            raise ValueError("friction model has not passed held-out validation")
        arrays = document.get("linear_model", {})
        mean = np.asarray(arrays.get("mean", []), dtype=np.float64)
        scale = np.asarray(arrays.get("scale", []), dtype=np.float64)
        weight = np.asarray(arrays.get("weight", []), dtype=np.float64)
        expected = (len(FEATURE_NAMES),)
        if mean.shape != expected or scale.shape != expected or weight.shape != expected:
            raise ValueError("linear model vectors have the wrong shape")
        if not np.all(np.isfinite(np.concatenate((mean, scale, weight)))):
            raise ValueError("linear model contains NaN or infinity")
        if np.any(scale <= 0.0):
            raise ValueError("linear model scale values must be positive")
        runtime = document.get("runtime", {})
        model = cls(
            mean=mean,
            scale=scale,
            weight=weight,
            bias=float(arrays.get("bias")),
            probability_temperature=float(arrays.get("probability_temperature", 1.0)),
            window_frames=int(runtime.get("window_frames")),
            nominal_rate_hz=float(runtime.get("nominal_rate_hz")),
            enter_low_probability=float(runtime.get("enter_low_probability", 0.80)),
            clear_low_probability=float(runtime.get("clear_low_probability", 0.20)),
            validation=validation,
            source_sha256=str(document.get("model_sha256", "")),
        )
        if model.window_frames < 4 or not 20.0 <= model.nominal_rate_hz <= 200.0:
            raise ValueError("invalid runtime window/rate")
        if not 0.5 < model.enter_low_probability < 1.0:
            raise ValueError("enter_low_probability must be in (0.5,1)")
        if not 0.0 < model.clear_low_probability < 0.5:
            raise ValueError("clear_low_probability must be in (0,0.5)")
        if not math.isfinite(model.probability_temperature) or model.probability_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive and finite")
        return model

    def probability_low(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != self.mean.shape or not np.all(np.isfinite(values)):
            raise ValueError("features do not match model")
        logit = float(((values - self.mean) / self.scale) @ self.weight + self.bias)
        logit /= self.probability_temperature
        logit = float(np.clip(logit, -40.0, 40.0))
        return float(1.0 / (1.0 + math.exp(-logit)))


@dataclass(frozen=True)
class DecisionOutput:
    state: str
    requested_mode: str
    speed_cap_mps: float
    probability_low: float
    reason: str


class FrictionDecisionStateMachine:
    """Hysteretic semantic mode request; never calls the robot API itself."""

    def __init__(
        self,
        enter_low_probability: float = 0.80,
        clear_low_probability: float = 0.20,
        enter_low_hold_s: float = 0.30,
        clear_low_hold_s: float = 0.80,
        conservative_speed_mps: float = 0.25,
        high_speed_cap_mps: float = 0.80,
    ) -> None:
        if not 0.5 < enter_low_probability < 1.0:
            raise ValueError("enter_low_probability must be in (0.5,1)")
        if not 0.0 < clear_low_probability < 0.5:
            raise ValueError("clear_low_probability must be in (0,0.5)")
        if min(enter_low_hold_s, clear_low_hold_s, conservative_speed_mps) <= 0.0:
            raise ValueError("hold durations and conservative speed must be positive")
        self.enter_low_probability = enter_low_probability
        self.clear_low_probability = clear_low_probability
        self.enter_low_hold_s = enter_low_hold_s
        self.clear_low_hold_s = clear_low_hold_s
        self.conservative_speed_mps = conservative_speed_mps
        self.high_speed_cap_mps = high_speed_cap_mps
        self.state = "STARTUP"
        self._low_evidence_s = 0.0
        self._high_evidence_s = 0.0

    def reset(self) -> None:
        self.state = "STARTUP"
        self._low_evidence_s = 0.0
        self._high_evidence_s = 0.0

    def update(
        self,
        probability_low: float,
        dt_s: float,
        both_feet_healthy: bool,
        model_valid: bool = True,
    ) -> DecisionOutput:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        probability_valid = math.isfinite(probability_low) and 0.0 <= probability_low <= 1.0
        if not both_feet_healthy or not model_valid or not probability_valid:
            self.state = "DEGRADED"
            self._low_evidence_s = 0.0
            self._high_evidence_s = 0.0
            return DecisionOutput(
                state=self.state,
                requested_mode="waist_walk",
                speed_cap_mps=self.conservative_speed_mps,
                probability_low=float(probability_low) if probability_valid else float("nan"),
                reason="sensor_or_model_unhealthy",
            )

        if probability_low >= self.enter_low_probability:
            self._low_evidence_s += dt_s
            self._high_evidence_s = 0.0
        elif probability_low <= self.clear_low_probability:
            self._high_evidence_s += dt_s
            self._low_evidence_s = 0.0
        else:
            self._low_evidence_s = 0.0
            self._high_evidence_s = 0.0

        if self._low_evidence_s >= self.enter_low_hold_s:
            self.state = "LOW"
        elif self._high_evidence_s >= self.clear_low_hold_s:
            self.state = "HIGH"
        elif self.state in ("STARTUP", "DEGRADED"):
            self.state = "UNCERTAIN"

        if self.state == "HIGH":
            return DecisionOutput(
                state="HIGH",
                requested_mode="walkrun",
                speed_cap_mps=self.high_speed_cap_mps,
                probability_low=probability_low,
                reason="sustained_high_friction_evidence",
            )
        if self.state == "LOW":
            return DecisionOutput(
                state="LOW",
                requested_mode="waist_walk",
                speed_cap_mps=self.conservative_speed_mps,
                probability_low=probability_low,
                reason="sustained_low_friction_evidence",
            )
        return DecisionOutput(
            state=self.state,
            requested_mode="waist_walk",
            speed_cap_mps=self.conservative_speed_mps,
            probability_low=probability_low,
            reason="insufficient_hysteresis_evidence",
        )
