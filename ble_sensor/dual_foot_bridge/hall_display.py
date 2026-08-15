"""Display-only filtering for raw dual-foot Hall visualization.

This module deliberately produces dimensionless display intensities from raw
Hall-count changes.  It is not a calibration model and must never be used as
force, pressure, contact-force or friction data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .protocol import NUM_SENSORS


@dataclass(frozen=True)
class HallDisplayLayout:
    outline_uv: np.ndarray
    sensor_uv: np.ndarray
    output_ids: tuple[str, ...]


def load_display_layout(path: Path) -> HallDisplayLayout:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "footsensor15-a4-layout-v1":
        raise ValueError("layout format must be footsensor15-a4-layout-v1")
    sensors = sorted(document.get("sensors", []), key=lambda item: int(item["id"]))
    output_ids = tuple(str(item.get("output_id", "")) for item in sensors)
    expected = tuple(f"P{index:02d}" for index in range(NUM_SENSORS))
    if len(sensors) != NUM_SENSORS or output_ids != expected:
        raise ValueError("layout sensors must remain ordered P00..P14")
    outline = np.asarray(document.get("outline_normalized_uv", []), dtype=np.float32)
    sensor_uv = np.asarray(
        [item.get("normalized_uv", []) for item in sensors], dtype=np.float32
    )
    if outline.ndim != 2 or outline.shape[1:] != (2,) or len(outline) < 3:
        raise ValueError("layout outline must contain at least three UV points")
    if sensor_uv.shape != (NUM_SENSORS, 2):
        raise ValueError("layout sensor positions must have shape (15, 2)")
    if not np.all(np.isfinite(outline)) or not np.all(np.isfinite(sensor_uv)):
        raise ValueError("layout coordinates must be finite")
    if np.any(outline < 0.0) or np.any(outline > 1.0):
        raise ValueError("layout outline UV values must be in [0, 1]")
    if np.any(sensor_uv < 0.0) or np.any(sensor_uv > 1.0):
        raise ValueError("layout sensor UV values must be in [0, 1]")
    return HallDisplayLayout(outline, sensor_uv, output_ids)


@dataclass
class HallFootDisplayFilter:
    """Fixed unloaded baseline, soft deadzone and bounded display smoothing."""

    calibration_target: int = 180
    calibration_window_s: float = 15.0
    calibration_history_s: float = 18.0
    stability_p95_counts_per_s: float = 1.2
    stability_max_counts_per_s: float = 2.5
    min_deadzone_counts: float = 45.0
    baseline: np.ndarray | None = None
    noise_sigma: np.ndarray = field(
        default_factory=lambda: np.full((NUM_SENSORS, 3), 18.0, dtype=np.float64)
    )
    filtered: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_SENSORS, 3), dtype=np.float64)
    )
    calibrating: bool = False
    calibration_samples: list[np.ndarray] = field(default_factory=list)
    calibration_times_s: list[float] = field(default_factory=list)
    calibration_drift_p95: float = float("inf")
    calibration_drift_max: float = float("inf")
    calibration_status: str = "missing"
    valid: bool = False
    _synthetic_time_s: float = 0.0

    def begin_unloaded_baseline(self) -> None:
        self.baseline = None
        self.filtered.fill(0.0)
        self.calibration_samples.clear()
        self.calibration_times_s.clear()
        self.calibration_drift_p95 = float("inf")
        self.calibration_drift_max = float("inf")
        self.calibration_status = "collecting"
        self.calibrating = True
        self.valid = False
        self._synthetic_time_s = 0.0

    @property
    def calibration_progress(self) -> float:
        if not self.calibrating:
            return 1.0 if self.baseline is not None else 0.0
        if len(self.calibration_times_s) < 2:
            return 0.0
        elapsed = self.calibration_times_s[-1] - self.calibration_times_s[0]
        return min(0.99, elapsed / max(self.calibration_window_s, 1.0e-6))

    @property
    def intensity(self) -> np.ndarray:
        return np.linalg.norm(self.filtered, axis=1)

    def update(
        self,
        magnetic_xyz: np.ndarray,
        *,
        valid: bool,
        sample_time_s: float | None = None,
    ) -> None:
        self.valid = bool(valid)
        if not self.valid:
            return
        magnetic = np.asarray(magnetic_xyz, dtype=np.float64)
        if magnetic.shape != (NUM_SENSORS, 3) or not np.all(np.isfinite(magnetic)):
            self.valid = False
            return

        if self.calibrating:
            if sample_time_s is None:
                self._synthetic_time_s += 0.02
                sample_time_s = self._synthetic_time_s
            sample_time_s = float(sample_time_s)
            if not np.isfinite(sample_time_s):
                self.valid = False
                return
            self.calibration_samples.append(magnetic.copy())
            self.calibration_times_s.append(sample_time_s)
            cutoff = sample_time_s - self.calibration_history_s
            while self.calibration_times_s and self.calibration_times_s[0] < cutoff:
                self.calibration_times_s.pop(0)
                self.calibration_samples.pop(0)
            elapsed = self.calibration_times_s[-1] - self.calibration_times_s[0]
            if (
                len(self.calibration_samples) >= self.calibration_target
                and elapsed >= self.calibration_window_s
            ):
                samples = np.stack(self.calibration_samples, axis=0)
                times = np.asarray(self.calibration_times_s, dtype=np.float64)
                centered_time = times - np.mean(times)
                centered_samples = samples - np.mean(samples, axis=0, keepdims=True)
                denominator = max(float(np.dot(centered_time, centered_time)), 1.0e-9)
                slopes = np.sum(
                    centered_samples * centered_time.reshape(-1, 1, 1), axis=0
                ) / denominator
                absolute_slopes = np.abs(slopes).reshape(-1)
                self.calibration_drift_p95 = float(
                    np.percentile(absolute_slopes, 95)
                )
                self.calibration_drift_max = float(np.max(absolute_slopes))
                stable = (
                    self.calibration_drift_p95 <= self.stability_p95_counts_per_s
                    and self.calibration_drift_max <= self.stability_max_counts_per_s
                )
                if stable:
                    recent = times >= times[-1] - 2.0
                    self.baseline = np.median(samples[recent], axis=0)
                    trend = centered_time.reshape(-1, 1, 1) * slopes
                    residual = centered_samples - trend
                    residual_median = np.median(residual, axis=0)
                    mad = np.median(
                        np.abs(residual - residual_median), axis=0
                    )
                    self.noise_sigma = np.maximum(1.4826 * mad, 1.0)
                    self.calibration_samples.clear()
                    self.calibration_times_s.clear()
                    self.calibrating = False
                    self.calibration_status = "locked"
                    self.filtered.fill(0.0)
                else:
                    self.calibration_status = "unstable"
            return
        if self.baseline is None:
            return

        delta = magnetic - self.baseline
        deadzone = np.clip(
            np.maximum(self.min_deadzone_counts, self.noise_sigma * 2.4),
            self.min_deadzone_counts,
            150.0,
        )
        target = np.sign(delta) * np.maximum(np.abs(delta) - deadzone, 0.0)

        # The baseline is intentionally never updated here.  A sustained input
        # therefore stays visible instead of being learned away as "zero".
        rising = np.abs(target) >= np.abs(self.filtered)
        alpha = np.where(rising, 0.22, 0.30)
        self.filtered += alpha * (target - self.filtered)

        # Fast release only when the raw signal is back inside the fixed
        # unloaded deadzone; this removes stale color without erasing a hold.
        quiet = np.linalg.norm(target, axis=1) < self.min_deadzone_counts * 0.25
        self.filtered[quiet] *= 0.55
        self.filtered[np.abs(self.filtered) < 3.0] = 0.0


class DualHallDisplayFilter:
    """Left/right display filters with one shared, comparable color scale."""

    def __init__(self) -> None:
        self.feet = {
            "left": HallFootDisplayFilter(),
            "right": HallFootDisplayFilter(),
        }
        self.shared_scale_counts = 2400.0

    def begin_unloaded_baseline(self) -> None:
        for foot in self.feet.values():
            foot.begin_unloaded_baseline()
        self.shared_scale_counts = 2400.0

    @property
    def baseline_ready(self) -> bool:
        return all(foot.baseline is not None and not foot.calibrating for foot in self.feet.values())

    @property
    def calibration_progress(self) -> float:
        return min(foot.calibration_progress for foot in self.feet.values())

    def update(
        self,
        magnetic: np.ndarray,
        valid: tuple[bool, bool],
        *,
        sample_time_s: float | None = None,
    ) -> None:
        values = np.asarray(magnetic)
        if values.shape != (2, NUM_SENSORS, 3):
            raise ValueError("dual-foot Hall display input must have shape (2, 15, 3)")
        for index, side in enumerate(("left", "right")):
            self.feet[side].update(
                values[index],
                valid=bool(valid[index]),
                sample_time_s=sample_time_s,
            )
        if not self.baseline_ready:
            return
        combined = np.concatenate(
            [self.feet[side].intensity for side in ("left", "right")]
        )
        target = float(np.clip(np.percentile(combined, 95) * 1.25, 600.0, 24000.0))
        alpha = 0.20 if target > self.shared_scale_counts else 0.012
        self.shared_scale_counts += alpha * (target - self.shared_scale_counts)
