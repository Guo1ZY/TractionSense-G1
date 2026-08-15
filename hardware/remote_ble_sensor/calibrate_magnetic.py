#!/usr/bin/env python3
"""Fit real per-channel magnetic baseline, temperature drift and normalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dual_foot_bridge.normalization import FORMAT
from dual_foot_bridge.protocol import NUM_SENSORS


MAG_COLUMNS = [
    f"mag_{sensor}_{axis}"
    for sensor in range(NUM_SENSORS)
    for axis in ("x", "y", "z")
]
TEMP_COLUMNS = [f"temp_{sensor}_x10" for sensor in range(NUM_SENSORS)]


def load(
    paths: list[Path],
    side: str,
    sensor_permutation: list[int] | None = None,
    axis_sign: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    magnetic_rows = []
    temperature_rows = []
    for path in paths:
        if path.suffix.casefold() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "hall_xyz" not in data or "temperature_c" not in data:
                    raise ValueError(
                        f"{path}: NPZ must contain hall_xyz and temperature_c"
                    )
                magnetic = np.asarray(data["hall_xyz"], dtype=np.float64)
                temperature = 10.0 * np.asarray(
                    data["temperature_c"], dtype=np.float64
                )
                if magnetic.ndim != 3 or magnetic.shape[1:] != (NUM_SENSORS, 3):
                    raise ValueError(f"{path}: hall_xyz must have shape [N,15,3]")
                if temperature.shape != magnetic.shape[:2]:
                    raise ValueError(f"{path}: temperature_c must have shape [N,15]")
                if "valid" in data:
                    valid = np.asarray(data["valid"], dtype=bool).reshape(-1)
                    if len(valid) != len(magnetic):
                        raise ValueError(f"{path}: valid length mismatch")
                    magnetic = magnetic[valid]
                    temperature = temperature[valid]
                if "metadata" in data:
                    metadata = [str(value) for value in np.asarray(data["metadata"]).reshape(-1)]
                    foot_tags = [value.split("=", 1)[1] for value in metadata if value.startswith("foot_id=")]
                    if foot_tags and foot_tags[-1] != side:
                        raise ValueError(
                            f"{path}: recorded foot_id={foot_tags[-1]!r}, requested {side!r}"
                        )
            order = np.asarray(
                sensor_permutation if sensor_permutation is not None else range(NUM_SENSORS),
                dtype=np.int64,
            )
            sign = np.asarray(axis_sign if axis_sign is not None else (1, 1, 1))
            if sorted(order.tolist()) != list(range(NUM_SENSORS)):
                raise ValueError("--sensor-permutation must contain 0..14 exactly once")
            if sign.shape != (3,) or not np.all(np.isin(sign, (-1, 1))):
                raise ValueError("--axis-sign must contain three values, each -1 or 1")
            magnetic_rows.extend((magnetic[:, order] * sign).tolist())
            temperature_rows.extend(temperature[:, order].tolist())
            continue
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--motion", type=Path, nargs="+", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-range", type=float, default=3.0)
    parser.add_argument(
        "--sensor-permutation",
        type=int,
        nargs=NUM_SENSORS,
        default=list(range(NUM_SENSORS)),
        metavar="CHANNEL",
        help="Wire-to-P00..P14 order for record_raw_hall.py NPZ inputs.",
    )
    parser.add_argument(
        "--axis-sign",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=("SX", "SY", "SZ"),
        help="Hall XYZ sign transform for record_raw_hall.py NPZ inputs.",
    )
    args = parser.parse_args()
    try:
        if args.target_range <= 0:
            raise ValueError("--target-range must be positive")
        base_mag, base_temp = load(
            args.baseline, args.side, args.sensor_permutation, args.axis_sign
        )
        motion_mag, motion_temp = load(
            args.motion, args.side, args.sensor_permutation, args.axis_sign
        )
        reference_temp = np.mean(base_temp, axis=0)
        baseline = np.mean(base_mag, axis=0)
        coefficient = np.zeros((NUM_SENSORS, 3), dtype=np.float64)
        for sensor in range(NUM_SENSORS):
            centered_t = base_temp[:, sensor] - reference_temp[sensor]
            denominator = float(centered_t @ centered_t)
            if denominator >= 1.0:
                centered_b = base_mag[:, sensor] - baseline[sensor]
                coefficient[sensor] = (
                    centered_t[:, None] * centered_b
                ).sum(axis=0) / denominator
        corrected = motion_mag - (
            baseline[None]
            + (motion_temp - reference_temp[None])[:, :, None] * coefficient[None]
        )
        scale = np.quantile(np.abs(corrected), 0.99, axis=0) / args.target_range
        scale = np.maximum(scale, 1.0)
        normalized = corrected / scale[None]
        document = {
            "format": FORMAT,
            "side": args.side,
            "baseline_xyz": baseline.tolist(),
            "scale_xyz": scale.tolist(),
            "reference_temperature_x10": reference_temp.tolist(),
            "temperature_coefficient_per_x10": coefficient.tolist(),
            "clip": 6.0,
            "samples": {
                "baseline": int(len(base_mag)),
                "motion": int(len(motion_mag)),
            },
            "diagnostics": {
                "normalized_abs_p95": float(np.quantile(np.abs(normalized), 0.95)),
                "normalized_abs_p99": float(np.quantile(np.abs(normalized), 0.99)),
                "normalized_abs_max": float(np.max(np.abs(normalized))),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        print(json.dumps(document["diagnostics"], indent=2))
        print(f"{args.side} normalization -> {args.output}")
        return 0
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
