#!/usr/bin/env python3
"""Summarize repeated Isaac magnetic-policy friction matrices and apply gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-low-vx", type=float, default=0.55)
    parser.add_argument("--min-high-vx", type=float, default=0.80)
    parser.add_argument("--min-vx-separation", type=float, default=0.35)
    parser.add_argument("--max-high-abs-vy", type=float, default=0.15)
    parser.add_argument("--max-high-final-lateral", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, float]] = []
    sources: list[str] = []
    for path in args.csv:
        with path.open(newline="", encoding="utf-8") as stream:
            file_rows = [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
        if not file_rows:
            raise RuntimeError(f"No evaluation rows in {path}")
        rows.extend(file_rows)
        sources.append(str(path.resolve()))

    mus = sorted({row["mu"] for row in rows})
    commands = sorted({row["cmd_vx"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    expected_rows = len(mus) * len(commands) * len(seeds)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete matrix: got {len(rows)} rows, expected {expected_rows}"
        )

    endpoint_command = max(commands)
    low_rows = [
        row for row in rows
        if row["mu"] == min(mus) and row["cmd_vx"] == endpoint_command
    ]
    high_rows = [
        row for row in rows
        if row["mu"] == max(mus) and row["cmd_vx"] == endpoint_command
    ]

    def mean(selected: list[dict[str, float]], key: str) -> float:
        return sum(row[key] for row in selected) / len(selected)

    low_vx = mean(low_rows, "mean_vx")
    high_vx = mean(high_rows, "mean_vx")
    separation = high_vx - low_vx
    total_fall_per_env = sum(row["fall_per_env"] for row in rows)
    high_abs_vy = max(row["mean_abs_vy"] for row in high_rows)
    high_final_lateral = max(
        row["final_mean_abs_lateral_pos"] for row in high_rows
    )
    gates = {
        "complete_matrix": len(rows) == expected_rows,
        "zero_falls": total_fall_per_env == 0.0,
        "low_friction_slowdown": low_vx <= args.max_low_vx,
        "high_friction_tracking": high_vx >= args.min_high_vx,
        "friction_speed_separation": separation >= args.min_vx_separation,
        "high_friction_lateral_velocity": high_abs_vy <= args.max_high_abs_vy,
        "high_friction_lateral_displacement": (
            high_final_lateral <= args.max_high_final_lateral
        ),
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "sources": sources,
        "matrix": {
            "seeds": seeds,
            "mus": mus,
            "commands_mps": commands,
            "rows": len(rows),
            "expected_rows": expected_rows,
        },
        "endpoint_at_command_mps": endpoint_command,
        "metrics": {
            "low_friction_mean_vx_mps": low_vx,
            "high_friction_mean_vx_mps": high_vx,
            "high_minus_low_vx_mps": separation,
            "max_high_friction_mean_abs_vy_mps": high_abs_vy,
            "max_high_friction_final_abs_lateral_pos_m": high_final_lateral,
            "sum_fall_per_env": total_fall_per_env,
        },
        "thresholds": {
            "max_low_vx_mps": args.max_low_vx,
            "min_high_vx_mps": args.min_high_vx,
            "min_vx_separation_mps": args.min_vx_separation,
            "max_high_abs_vy_mps": args.max_high_abs_vy,
            "max_high_final_lateral_pos_m": args.max_high_final_lateral,
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
