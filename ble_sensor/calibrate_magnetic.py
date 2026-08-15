#!/usr/bin/env python3
"""Fit real per-channel magnetic baseline, temperature drift and normalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dual_foot_bridge.normalization import FORMAT, TEMP_X10_MAX, TEMP_X10_MIN
from dual_foot_bridge.protocol import NUM_SENSORS


MAG_COLUMNS = [
    f"mag_{sensor}_{axis}"
    for sensor in range(NUM_SENSORS)
    for axis in ("x", "y", "z")
]
TEMP_COLUMNS = [f"temp_{sensor}_x10" for sensor in range(NUM_SENSORS)]


def load(paths: list[Path], side: str) -> tuple[np.ndarray, np.ndarray]:
    magnetic_rows = []
    temperature_rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = ["side", *TEMP_COLUMNS, *MAG_COLUMNS]
            missing = [name for name in required if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path}: missing columns {missing[:5]}")
            for line, row in enumerate(reader, start=2):
                if row["side"].strip() != side:
                    continue
                try:
                    temperature = [float(row[name]) for name in TEMP_COLUMNS]
                    magnetic = [float(row[name]) for name in MAG_COLUMNS]
                except ValueError as error:
                    raise ValueError(f"{path}:{line}: invalid numeric value") from error
                if np.isfinite(temperature).all() and np.isfinite(magnetic).all():
                    temperature_rows.append(temperature)
                    magnetic_rows.append(magnetic)
    if not magnetic_rows:
        raise ValueError(f"no {side} rows found")
    return (
        np.asarray(magnetic_rows).reshape(-1, NUM_SENSORS, 3),
        np.asarray(temperature_rows).reshape(-1, NUM_SENSORS),
    )


def fit_normalization_document(
    baseline_paths: list[Path],
    motion_paths: list[Path],
    side: str,
    target_range: float = 3.0,
) -> dict:
    """Fit Hall baseline/temperature normalization; no force is estimated."""

    if target_range <= 0:
        raise ValueError("target_range must be positive")
    base_mag, base_temp = load(baseline_paths, side)
    motion_mag, motion_temp = load(motion_paths, side)
    in_range = (base_temp >= TEMP_X10_MIN) & (base_temp <= TEMP_X10_MAX)
    bad_ratio = 1.0 - in_range.mean(axis=0)
    bad_channels = [int(sensor) for sensor in range(NUM_SENSORS) if bad_ratio[sensor] >= 0.5]
    reference_temp = np.mean(base_temp, axis=0)
    if bad_channels:
        pooled_valid = base_temp[in_range]
        pooled_median = float(np.median(pooled_valid)) if pooled_valid.size else 250.0
        for sensor in bad_channels:
            valid_values = base_temp[in_range[:, sensor], sensor]
            reference_temp[sensor] = (
                float(np.median(valid_values)) if valid_values.size else pooled_median
            )
    baseline = np.mean(base_mag, axis=0)
    coefficient = np.zeros((NUM_SENSORS, 3), dtype=np.float64)
    for sensor in range(NUM_SENSORS):
        if sensor in bad_channels:
            continue
        centered_t = base_temp[:, sensor] - reference_temp[sensor]
        denominator = float(centered_t @ centered_t)
        if denominator >= 1.0:
            centered_b = base_mag[:, sensor] - baseline[sensor]
            coefficient[sensor] = (
                centered_t[:, None] * centered_b
            ).sum(axis=0) / denominator
    compensated_baseline = baseline[None] + (
        base_temp - reference_temp[None]
    )[:, :, None] * coefficient[None]
    baseline_residual = base_mag - compensated_baseline
    corrected = motion_mag - (
        baseline[None]
        + (motion_temp - reference_temp[None])[:, :, None] * coefficient[None]
    )
    scale = np.quantile(np.abs(corrected), 0.99, axis=0) / target_range
    scale = np.maximum(scale, 1.0)
    normalized = corrected / scale[None]
    return {
        "format": FORMAT,
        "side": side,
        "measurement": "Hall Bx/By/Bz counts and temperature only",
        "force_conversion": "absent",
        "baseline_xyz": baseline.tolist(),
        "scale_xyz": scale.tolist(),
        "reference_temperature_x10": reference_temp.tolist(),
        "temperature_coefficient_per_x10": coefficient.tolist(),
        "bad_temperature_channels": bad_channels,
        "clip": 6.0,
        "samples": {
            "baseline": int(len(base_mag)),
            "motion": int(len(motion_mag)),
        },
        "diagnostics": {
            "bad_temperature_channels": bad_channels,
            "bad_temperature_policy": (
                "temperature compensation disabled and reference temperature "
                "fallback applied for listed channels"
            ),
            "baseline_residual_abs_p99_counts": float(
                np.quantile(np.abs(baseline_residual), 0.99)
            ),
            "temperature_span_c": float(
                (np.max(base_temp) - np.min(base_temp)) * 0.1
            ),
            "normalized_abs_p95": float(np.quantile(np.abs(normalized), 0.95)),
            "normalized_abs_p99": float(np.quantile(np.abs(normalized), 0.99)),
            "normalized_abs_max": float(np.max(np.abs(normalized))),
            "finite": bool(
                np.isfinite(baseline).all()
                and np.isfinite(scale).all()
                and np.isfinite(coefficient).all()
            ),
        },
    }


def write_normalization_document(document: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--motion", type=Path, nargs="+", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-range", type=float, default=3.0)
    args = parser.parse_args()
    try:
        document = fit_normalization_document(
            args.baseline,
            args.motion,
            args.side,
            args.target_range,
        )
        write_normalization_document(document, args.output)
        print(json.dumps(document["diagnostics"], indent=2))
        print(f"{args.side} normalization -> {args.output}")
        return 0
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
