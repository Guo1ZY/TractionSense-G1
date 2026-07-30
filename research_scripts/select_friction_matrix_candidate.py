#!/usr/bin/env python3
"""Rank Teacher or magnetic-Student candidates from Isaac/MuJoCo matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=CSV[,CSV...]",
        help="Candidate name and one or more Isaac/MuJoCo matrix CSV files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-name", type=Path, required=True)
    parser.add_argument("--max-low-vx", type=float, default=0.55)
    parser.add_argument("--min-high-vx", type=float, default=0.80)
    parser.add_argument("--min-separation", type=float, default=0.35)
    parser.add_argument("--max-high-abs-vy", type=float, default=0.20)
    parser.add_argument("--max-high-lateral", type=float, default=0.25)
    return parser.parse_args()


def finite(value: str | None, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def load_rows(paths: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                rows.append(
                    {
                        "source": str(path.resolve()),
                        "mu": finite(raw.get("mu")),
                        "cmd_vx": finite(raw.get("cmd_vx")),
                        "mean_vx": finite(raw.get("mean_vx")),
                        "mean_abs_vy": finite(raw.get("mean_abs_vy")),
                        "lateral": finite(
                            raw.get(
                                "final_mean_abs_lateral_pos",
                                raw.get(
                                    "lateral_drift",
                                    raw.get("mean_abs_lateral_pos", "0"),
                                ),
                            )
                        ),
                        "slip": finite(raw.get("mean_contact_slip", "0")),
                        "fall": finite(
                            raw.get("fall_per_env", raw.get("fall", "0"))
                        ),
                    }
                )
    if not rows:
        raise RuntimeError(f"no rows loaded from {paths}")
    return rows


def mean(rows: list[dict[str, float | str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(
    name: str,
    paths: list[Path],
    args: argparse.Namespace,
) -> dict:
    rows = load_rows(paths)
    endpoint_command = max(float(row["cmd_vx"]) for row in rows)
    endpoint = [
        row
        for row in rows
        if abs(float(row["cmd_vx"]) - endpoint_command) < 1.0e-6
    ]
    low_mu = min(float(row["mu"]) for row in endpoint)
    high_mu = max(float(row["mu"]) for row in endpoint)
    low = [row for row in endpoint if abs(float(row["mu"]) - low_mu) < 1.0e-6]
    high = [
        row for row in endpoint if abs(float(row["mu"]) - high_mu) < 1.0e-6
    ]
    low_vx = mean(low, "mean_vx")
    high_vx = mean(high, "mean_vx")
    separation = high_vx - low_vx
    high_abs_vy = max(float(row["mean_abs_vy"]) for row in high)
    high_lateral = max(float(row["lateral"]) for row in high)
    falls = sum(float(row["fall"]) for row in rows)
    gates = {
        "zero_falls": falls == 0.0,
        "low_friction_slowdown": low_vx <= args.max_low_vx,
        "high_friction_tracking": high_vx >= args.min_high_vx,
        "friction_speed_separation": separation >= args.min_separation,
        "high_friction_lateral_velocity": high_abs_vy <= args.max_high_abs_vy,
        "high_friction_lateral_displacement": (
            high_lateral <= args.max_high_lateral
        ),
    }
    # Safety remains dominant. Among safe policies, prefer command tracking,
    # traction-conditioned speed separation, and low transverse motion.
    score = (
        12.0 * high_vx
        + 4.0 * separation
        - 5.0 * high_abs_vy
        - 3.0 * high_lateral
        - 1.5 * mean(rows, "slip")
        - 1000.0 * falls
    )
    score -= 20.0 * max(args.min_high_vx - high_vx, 0.0)
    score -= 15.0 * max(low_vx - args.max_low_vx, 0.0)
    return {
        "name": name,
        "qualified": all(gates.values()),
        "score": score,
        "sources": [str(path.resolve()) for path in paths],
        "rows": len(rows),
        "metrics": {
            "endpoint_command_mps": endpoint_command,
            "low_friction_mean_vx_mps": low_vx,
            "high_friction_mean_vx_mps": high_vx,
            "high_minus_low_vx_mps": separation,
            "max_high_friction_mean_abs_vy_mps": high_abs_vy,
            "max_high_friction_lateral_m": high_lateral,
            "sum_fall_per_env": falls,
        },
        "gates": gates,
    }


def main() -> int:
    args = parse_args()
    candidates = []
    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError(f"invalid candidate specification: {specification}")
        name, raw_paths = specification.split("=", 1)
        paths = [Path(item) for item in raw_paths.split(",") if item]
        if not name or not paths:
            raise ValueError(f"invalid candidate specification: {specification}")
        candidates.append(summarize(name, paths, args))

    qualified = [item for item in candidates if item["qualified"]]
    pool = qualified if qualified else candidates
    selected = max(pool, key=lambda item: item["score"])
    report = {
        "status": "PASS" if qualified else "BEST_AVAILABLE",
        "selected": selected["name"],
        "selection_rule": (
            "highest score among fully qualified candidates"
            if qualified
            else "highest safety-weighted score; no candidate passed every gate"
        ),
        "candidates": sorted(candidates, key=lambda item: item["score"], reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.selected_name.parent.mkdir(parents=True, exist_ok=True)
    args.selected_name.write_text(selected["name"] + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
