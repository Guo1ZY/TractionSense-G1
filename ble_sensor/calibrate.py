#!/usr/bin/env python3
"""Create per-foot magnetic-to-normal-force calibration JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dual_foot_bridge.calibration import (
    FORMAT,
    FEATURE_COUNT,
    calibration_document,
)
from dual_foot_bridge.protocol import NUM_SENSORS


MAG_COLUMNS = [
    f"mag_{sensor}_{axis}"
    for sensor in range(NUM_SENSORS)
    for axis in ("x", "y", "z")
]


def read_rows(
    paths: list[Path], side: str, require_reference: bool
) -> tuple[np.ndarray, np.ndarray]:
    magnetic_rows: list[list[float]] = []
    references: list[float] = []
    for path in paths:
        try:
            stream = path.open(newline="", encoding="utf-8")
        except OSError as error:
            raise ValueError(f"cannot open {path}: {error}") from error
        with stream:
            reader = csv.DictReader(stream)
            missing = [name for name in ["side", *MAG_COLUMNS] if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path}: missing CSV columns: {', '.join(missing[:5])}")
            for line, row in enumerate(reader, start=2):
                if row["side"].strip() != side:
                    continue
                try:
                    magnetic = [float(row[name]) for name in MAG_COLUMNS]
                    reference_text = row.get("reference_normal_n", "").strip()
                    reference = float(reference_text) if reference_text else float("nan")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line}: invalid numeric value") from error
                if np.all(np.isfinite(magnetic)):
                    if require_reference and not np.isfinite(reference):
                        continue
                    magnetic_rows.append(magnetic)
                    references.append(reference)
    if not magnetic_rows:
        suffix = " with force labels" if require_reference else ""
        raise ValueError(f"no {side} rows{suffix} found")
    return np.asarray(magnetic_rows), np.asarray(references)


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def baseline_command(args: argparse.Namespace) -> int:
    magnetic, _ = read_rows(args.input, args.side, require_reference=False)
    baseline = np.mean(magnetic, axis=0).reshape(NUM_SENSORS, 3)
    document = calibration_document(side=args.side, baseline_xyz=baseline)
    document["baseline_samples"] = int(magnetic.shape[0])
    write_json(args.output, document)
    print(
        f"{args.side}: baseline from {magnetic.shape[0]} frames -> {args.output}\n"
        "This file is intentionally not runtime-valid yet; capture known loads and run fit-normal."
    )
    return 0


def _baseline_from_file(path: Path, side: str) -> np.ndarray:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read baseline {path}: {error}") from error
    if document.get("format") != FORMAT or document.get("side") != side:
        raise ValueError(f"{path}: calibration format/side mismatch")
    baseline = np.asarray(document.get("baseline_xyz"), dtype=np.float64)
    if baseline.shape != (NUM_SENSORS, 3) or not np.all(np.isfinite(baseline)):
        raise ValueError(f"{path}: invalid baseline_xyz")
    return baseline


def fit_normal_command(args: argparse.Namespace) -> int:
    magnetic, reference = read_rows(args.input, args.side, require_reference=True)
    if np.any(reference < 0.0):
        raise ValueError("reference_normal_n cannot be negative")
    unique_loads = np.unique(np.round(reference, decimals=3))
    if unique_loads.size < 3:
        raise ValueError("need at least three distinct force levels including zero")
    if args.baseline:
        baseline = _baseline_from_file(args.baseline, args.side).reshape(-1)
    else:
        zero_mask = reference <= args.zero_threshold_n
        if int(np.count_nonzero(zero_mask)) < args.min_zero_frames:
            raise ValueError(
                f"need at least {args.min_zero_frames} zero-load frames "
                f"(reference <= {args.zero_threshold_n} N), or pass --baseline"
            )
        baseline = np.mean(magnetic[zero_mask], axis=0)
    delta = magnetic - baseline
    scale = np.std(delta, axis=0)
    scale = np.where(scale < 1.0, 1.0, scale)
    design = np.column_stack((delta / scale, np.ones(delta.shape[0])))
    penalty = np.eye(FEATURE_COUNT + 1)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + args.ridge * penalty
    rhs = design.T @ reference
    beta = np.linalg.solve(lhs, rhs)
    weights = beta[:-1] / scale
    bias = float(beta[-1])
    predicted = np.maximum(0.0, delta @ weights + bias)
    residual = predicted - reference
    metrics = {
        "samples": int(reference.size),
        "distinct_loads": int(unique_loads.size),
        "min_reference_n": float(np.min(reference)),
        "max_reference_n": float(np.max(reference)),
        "rmse_n": float(np.sqrt(np.mean(residual**2))),
        "mae_n": float(np.mean(np.abs(residual))),
        "ridge": float(args.ridge),
    }
    document = calibration_document(
        side=args.side,
        baseline_xyz=baseline.reshape(NUM_SENSORS, 3),
        normal_weights=weights,
        normal_bias=bias,
        metrics=metrics,
        max_normal_n=args.max_normal_n,
    )
    write_json(args.output, document)
    print(
        f"{args.side}: {reference.size} frames, {unique_loads.size} load levels, "
        f"RMSE={metrics['rmse_n']:.3f} N, MAE={metrics['mae_n']:.3f} N\n"
        f"runtime calibration -> {args.output}"
    )
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    document = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(
        {
            "format": document.get("format"),
            "side": document.get("side"),
            "model_complete": document.get("model_complete"),
            "metrics": document.get("metrics"),
            "has_tangent_model": bool(
                document.get("tangent_x") and document.get("tangent_y")
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="average unloaded magnetic frames")
    baseline.add_argument("--input", type=Path, nargs="+", required=True)
    baseline.add_argument("--side", choices=("left", "right"), required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.set_defaults(func=baseline_command)

    fit = subparsers.add_parser("fit-normal", help="fit magnetic array to reference force")
    fit.add_argument("--input", type=Path, nargs="+", required=True)
    fit.add_argument("--side", choices=("left", "right"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--baseline", type=Path)
    fit.add_argument("--ridge", type=float, default=1.0)
    fit.add_argument("--zero-threshold-n", type=float, default=1.0)
    fit.add_argument("--min-zero-frames", type=int, default=50)
    fit.add_argument("--max-normal-n", type=float, default=800.0)
    fit.set_defaults(func=fit_normal_command)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.set_defaults(func=inspect_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if getattr(args, "ridge", 0.0) < 0.0:
            raise ValueError("--ridge cannot be negative")
        return args.func(args)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

