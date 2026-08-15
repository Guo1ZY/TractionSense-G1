#!/usr/bin/env python3
"""Evaluate Isaac/MuJoCo torque-traction NPZ rollouts without changing weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from unitree_rl_lab.traction_torque.evaluation import TORQUE_TRACTION_ABLATIONS, evaluate_rollout_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/evaluation_summary.json"))
    parser.add_argument("--policy_dt", type=float, default=0.02)
    args = parser.parse_args()
    reports = [evaluate_rollout_npz(path, policy_dt_s=args.policy_dt) for path in args.rollouts]
    payload = {"reports": reports, "registered_ablations": TORQUE_TRACTION_ABLATIONS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    columns = ("source", "samples", "survival_time_s", "fell_by_height_threshold", "velocity_tracking_mae_m_s", "yaw_tracking_mae_rad_s", "force_mae_n", "force_rmse_n", "ground_truth_slip_rate", "governor_activation_ratio", "mean_speed_scale", "nonfinite_count")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for report in reports: writer.writerow({name: report[name] for name in columns})
    print(json.dumps({"json": str(args.output.resolve()), "csv": str(csv_path.resolve()), "rollouts": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

