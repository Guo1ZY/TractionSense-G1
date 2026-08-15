#!/usr/bin/env python3
"""Compare fixed MuJoCo torque-traction rollout directories as CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


TRACTION_SOURCE = Path("/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab")
sys.path.insert(0, str(TRACTION_SOURCE))
from unitree_rl_lab.traction_torque.evaluation import evaluate_rollout_npz  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/matrix_comparison.json"))
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        for path in sorted(directory.glob("*.npz")):
            report = evaluate_rollout_npz(path); report["matrix"] = directory.name; report["scenario"] = path.stem
            reports.append(report)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(reports, indent=2, allow_nan=True) + "\n")
    fields = ("matrix", "scenario", "survival_time_s", "fell_by_height_threshold", "velocity_tracking_mae_m_s", "force_mae_n", "ground_truth_slip_rate", "governor_activation_ratio", "mean_speed_scale", "nonfinite_count")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for report in reports: writer.writerow({key: report[key] for key in fields})
    print(json.dumps({"rollouts": len(reports), "output": str(args.output.resolve())}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

