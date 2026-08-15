#!/usr/bin/env python3
"""Reject a Hall locomotion checkpoint unless its switch campaign is fall-free.

This gate consumes the phase CSV files emitted by
``scripts/rsl_rl/eval_friction_matrix.py --switch_sequence ...``.  It is a
*release* gate, not a reward or training metric: one fall in any requested
seed rejects the candidate.  The deployed actor remains Hall Bx/By/Bz history
plus proprioception only; CSV-only contact/slip columns are evaluation labels
and never used by this script to alter policy actions.

Run one nominal-Hall suite and one fault-randomized suite.  The latter must be
collected without ``--nominal_magnetic_sensor`` so packet loss, dead channels,
foot dropout and magnetic/TPU domain randomization remain active.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _number(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: missing/non-numeric {field!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite {field}={row[field]!r}")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no phase rows")
    return rows


def _check_run(
    path: Path,
    *,
    high_mu_threshold: float,
    low_mu_threshold: float,
    min_high_tracking_fraction: float,
    max_low_speed: float,
    max_response_s: float,
) -> dict[str, Any]:
    rows = _read_rows(path)
    phases = [_number(row, "phase", path) for row in rows]
    mus = [_number(row, "mu", path) for row in rows]
    falls = [_number(row, "falls", path) for row in rows]
    vx = [_number(row, "steady_vx", path) for row in rows]
    commands = [_number(row, "cmd_vx", path) for row in rows]

    errors: list[str] = []
    if phases != list(range(len(rows))) or len(rows) < 3:
        errors.append("phase rows must be consecutive and contain at least high→low→high")
    if not (mus[0] >= high_mu_threshold and mus[1] <= low_mu_threshold and mus[-1] >= high_mu_threshold):
        errors.append("run does not begin high, enter low, and finish high friction")
    if any(value > 0.0 for value in falls):
        errors.append(f"fall(s) observed: {sum(falls):.0f}")

    high_indices = [index for index, mu in enumerate(mus) if mu >= high_mu_threshold]
    low_indices = [index for index, mu in enumerate(mus) if mu <= low_mu_threshold]
    if not high_indices:
        errors.append("no high-friction phase")
    if not low_indices:
        errors.append("no low-friction phase")
    for index in high_indices:
        target = abs(commands[index]) * min_high_tracking_fraction
        if abs(vx[index]) < target:
            errors.append(
                f"phase {index}: high-friction speed {vx[index]:.3f} < {target:.3f}"
            )
    for index in low_indices:
        if abs(vx[index]) > max_low_speed:
            errors.append(
                f"phase {index}: low-friction speed {vx[index]:.3f} > {max_low_speed:.3f}"
            )

    responses: list[float] = []
    for index, row in enumerate(rows[1:], start=1):
        try:
            response = _number(row, "response_time_s", path)
        except ValueError:
            errors.append(f"phase {index}: missing finite response_time_s")
            continue
        responses.append(response)
        if response > max_response_s:
            errors.append(
                f"phase {index}: response {response:.3f}s > {max_response_s:.3f}s"
            )

    return {
        "path": str(path),
        "phases": len(rows),
        "falls": int(round(sum(falls))),
        "minimum_high_tracking_fraction": (
            min(
                (abs(vx[index]) / max(abs(commands[index]), 1.0e-6) for index in high_indices),
                default=0.0,
            )
        ),
        "maximum_low_speed_m_s": max((abs(vx[index]) for index in low_indices), default=math.inf),
        "maximum_response_s": max(responses, default=math.inf),
        "errors": errors,
        "pass": not errors,
    }


def check_suite(
    paths: list[Path],
    *,
    min_runs: int,
    high_mu_threshold: float,
    low_mu_threshold: float,
    min_high_tracking_fraction: float,
    max_low_speed: float,
    max_response_s: float,
) -> dict[str, Any]:
    reports = [
        _check_run(
            path,
            high_mu_threshold=high_mu_threshold,
            low_mu_threshold=low_mu_threshold,
            min_high_tracking_fraction=min_high_tracking_fraction,
            max_low_speed=max_low_speed,
            max_response_s=max_response_s,
        )
        for path in paths
    ]
    errors: list[str] = []
    if len(paths) < min_runs:
        errors.append(f"only {len(paths)} runs supplied; require at least {min_runs}")
    return {
        "runs": reports,
        "errors": errors,
        "pass": not errors and all(report["pass"] for report in reports),
    }


def build_report(
    *,
    nominal_paths: list[Path],
    fault_paths: list[Path],
    min_runs: int,
    high_mu_threshold: float,
    low_mu_threshold: float,
    min_high_tracking_fraction: float,
    max_low_speed: float,
    max_response_s: float,
) -> dict[str, Any]:
    options = {
        "min_runs": min_runs,
        "high_mu_threshold": high_mu_threshold,
        "low_mu_threshold": low_mu_threshold,
        "min_high_tracking_fraction": min_high_tracking_fraction,
        "max_low_speed": max_low_speed,
        "max_response_s": max_response_s,
    }
    nominal = check_suite(nominal_paths, **options)
    fault = check_suite(fault_paths, **options)
    return {
        "format": "g1-hall-zero-fall-switch-gate-v1",
        "measurement_boundary": "Hall Bx/By/Bz history plus proprioception; no Hall-to-force inverse",
        "nominal_hall": nominal,
        "sensor_fault_randomized": fault,
        "pass": bool(nominal["pass"] and fault["pass"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--fault-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--min-runs-per-suite", type=int, default=3)
    parser.add_argument("--high-mu-threshold", type=float, default=0.75)
    parser.add_argument("--low-mu-threshold", type=float, default=0.20)
    parser.add_argument("--min-high-tracking-fraction", type=float, default=0.70)
    parser.add_argument("--max-low-speed", type=float, default=0.45)
    parser.add_argument("--max-response-s", type=float, default=0.60)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.min_runs_per_suite < 1:
        parser.error("--min-runs-per-suite must be positive")

    try:
        report = build_report(
            nominal_paths=args.nominal_csv,
            fault_paths=args.fault_csv,
            min_runs=args.min_runs_per_suite,
            high_mu_threshold=args.high_mu_threshold,
            low_mu_threshold=args.low_mu_threshold,
            min_high_tracking_fraction=args.min_high_tracking_fraction,
            max_low_speed=args.max_low_speed,
            max_response_s=args.max_response_s,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
