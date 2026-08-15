#!/usr/bin/env python3
"""Reproducible fixed-policy MuJoCo friction/command/sensor smoke matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulate_python/run_traction_sim2sim.py"

SCENARIOS = (
    {
        "name": "high_friction_forward",
        "friction": 0.9,
        "command": (0.6, 0.0, 0.0),
    },
    {
        "name": "low_friction_forward",
        "friction": 0.10,
        "command": (0.6, 0.0, 0.0),
    },
    {
        "name": "friction_drop",
        "friction": 0.9,
        "transition_friction": 0.08,
        "command": (0.6, 0.0, 0.0),
    },
    {
        "name": "friction_recovery",
        "friction": 0.08,
        "transition_friction": 0.9,
        "command": (0.4, 0.0, 0.0),
    },
    {
        "name": "asymmetric_friction",
        "friction": 0.15,
        "left_friction": 0.15,
        "right_friction": 0.9,
        "command": (0.4, 0.0, 0.0),
    },
    {
        "name": "turn",
        "friction": 0.6,
        "command": (0.3, 0.0, 0.5),
    },
    {
        "name": "lateral",
        "friction": 0.4,
        "command": (0.0, 0.3, 0.0),
    },
    {
        "name": "full_tactile_randomization",
        "friction": 0.8,
        "command": (0.4, 0.0, 0.0),
        "tactile_stage": 5,
    },
    {
        "name": "sensor_invalid",
        "friction": 0.8,
        "command": (0.4, 0.0, 0.0),
        "sensor_invalid": True,
    },
)


def _summarize(path: Path, scenario: str, seed: int) -> dict[str, float | str]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    command = data["raw_command"]
    velocity = data["base_velocity"]
    contact = data["contact_count"] > 0
    slip_speed = data["slip_speed_proxy"]
    slip = contact & (slip_speed > 0.12)
    return {
        "scenario": scenario,
        "seed": float(seed),
        "samples": float(len(command)),
        "fell": float(data["terminated_fall"][0]),
        "minimum_base_height_m": float(
            min(data["base_height_m"].min(), data["final_base_height_m"][0])
        ),
        "velocity_tracking_error_mean_m_s": float(
            np.linalg.norm(command[:, :2] - velocity[:, :2], axis=1).mean()
        ),
        "yaw_tracking_error_mean_rad_s": float(
            np.abs(command[:, 2] - data["base_yaw_rate"]).mean()
        ),
        "maximum_slip_speed_proxy_m_s": float(slip_speed.max(initial=0.0)),
        "slip_proxy_rate": float(slip.mean()),
        "speed_scale_mean": float(data["speed_scale"].mean()),
        "sensor_confidence_mean": float(data["sensor_confidence"].mean()),
        "maximum_action_abs": float(np.abs(data["action"]).max()),
        "nonfinite": float(
            sum(
                np.count_nonzero(~np.isfinite(value))
                for key, value in data.items()
                if key != "metadata"
            )
        ),
        "trajectory": str(path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--duration_s", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260731,))
    parser.add_argument(
        "--disable_governor",
        action="store_true",
        help="Run the proprio baseline/no-governor ablation.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for seed in args.seeds:
            name = str(scenario["name"])
            trajectory = args.output_dir / f"{name}_seed{seed}.npz"
            command = [
                sys.executable,
                str(RUNNER),
                "--policy",
                str(args.policy),
                "--duration_s",
                str(args.duration_s),
                "--friction",
                str(scenario["friction"]),
                "--command",
                *(str(value) for value in scenario["command"]),
                "--seed",
                str(seed),
                "--tactile_stage",
                str(scenario.get("tactile_stage", 0)),
                "--output",
                str(trajectory),
            ]
            if "transition_friction" in scenario:
                command.extend(
                    (
                        "--transition_friction",
                        str(scenario["transition_friction"]),
                    )
                )
            if "left_friction" in scenario:
                command.extend(
                    ("--left_friction", str(scenario["left_friction"]))
                )
            if "right_friction" in scenario:
                command.extend(
                    ("--right_friction", str(scenario["right_friction"]))
                )
            if scenario.get("sensor_invalid", False):
                command.append("--sensor_invalid")
            if args.disable_governor:
                command.append("--disable_governor")
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            (args.output_dir / f"{name}_seed{seed}.log").write_text(
                completed.stdout + completed.stderr,
                encoding="utf-8",
            )
            rows.append(_summarize(trajectory, name, seed))
    fieldnames = list(rows[0])
    with (args.output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": "fixed_policy_no_training_or_finetuning",
        "scenarios": len(SCENARIOS),
        "runs": len(rows),
        "seeds": args.seeds,
        "governor_enabled": not args.disable_governor,
        "nonfinite_total": sum(row["nonfinite"] for row in rows),
        "output": str(args.output_dir.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
