#!/usr/bin/env python3
"""Evaluate recorded Isaac native-signal Student rollouts across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from unitree_rl_lab.traction_torque.evaluation import _binary_metrics, _force_direction_error_deg


def evaluate(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True); metadata = data["metadata"].item()
    estimate = data["estimated_force_local_n"].astype(np.float64); truth = data["true_force_local_n"].astype(np.float64)
    error = estimate - truth; contact = data["true_contact"].astype(bool); contact_probability = data["contact_probability"]
    foot_speed = np.linalg.norm(data["foot_velocity_w_m_s"][..., :2], axis=-1)
    slip_proxy = contact & (foot_speed > 0.12)
    slip_probability = data["slip_probability"]
    terminated = data["terminated"].astype(bool)[..., 0]; truncated = data["truncated"].astype(bool)[..., 0]
    fall = terminated & ~truncated
    command = data["command"]; base_velocity = data["base_velocity_b"]
    action_delta = np.diff(data["action"], axis=0)
    # slip_event_mu_estimate is intentionally NaN until a confirmed event; it
    # has explicit optional-value semantics and is not a numerical failure.
    finite_names = [name for name in data.files if np.asarray(data[name]).dtype.kind in "fci" and name != "slip_event_mu_estimate"]
    return {
        "source": str(path.resolve()), "seed": int(metadata["seed"]), "steps": int(metadata["steps"]), "num_envs": int(metadata["num_envs"]),
        "policy_dt_s": float(metadata["policy_dt_s"]), "randomization_stage": int(metadata["randomization_stage"]),
        "velocity_tracking_mae_xy_m_s": float(np.abs(base_velocity[..., :2] - command[..., :2]).mean()),
        "fall_event_count": int(fall.sum()), "fraction_envs_with_fall_event": float(fall.any(axis=0).mean()),
        "fall_events_per_robot_second": float(fall.sum() / (metadata["steps"] * metadata["num_envs"] * metadata["policy_dt_s"])),
        "timeout_event_count": int(truncated.sum()), "force_mae_n": np.abs(error).mean(axis=(0, 1)).tolist(),
        "force_rmse_n": np.sqrt(np.square(error).mean(axis=(0, 1))).tolist(), "force_direction_error_deg": _force_direction_error_deg(estimate, truth),
        "contact": _binary_metrics(contact_probability, contact), "slip_proxy": _binary_metrics(slip_probability, slip_proxy),
        "ground_truth_slip_proxy_rate": float(slip_proxy.mean()), "mean_action_delta_l2": float(np.linalg.norm(action_delta, axis=-1).mean()),
        "ground_friction_mu": {"min": float(np.nanmin(data["ground_friction_mu"])), "mean": float(np.nanmean(data["ground_friction_mu"])), "max": float(np.nanmax(data["ground_friction_mu"]))},
        "nonfinite_count": sum(int((~np.isfinite(np.asarray(data[name]))).sum()) for name in finite_names),
        "undefined_slip_event_mu_count": int(np.isnan(data["slip_event_mu_estimate"]).sum()),
        "slip_label": "simulated contact AND ankle rigid-body planar-speed proxy > 0.12 m/s",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/isaac_medium_evaluation.json")); args = parser.parse_args()
    reports = [evaluate(path) for path in args.datasets]
    aggregate = {}
    for name in ("velocity_tracking_mae_xy_m_s", "fraction_envs_with_fall_event", "fall_events_per_robot_second", "ground_truth_slip_proxy_rate", "mean_action_delta_l2"):
        values = np.asarray([report[name] for report in reports], dtype=np.float64)
        aggregate[name] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan")}
    payload = {"reports": reports, "aggregate_across_seeds": aggregate, "seed_count": len(reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ("seed", "velocity_tracking_mae_xy_m_s", "fall_event_count", "fraction_envs_with_fall_event", "fall_events_per_robot_second", "ground_truth_slip_proxy_rate", "mean_action_delta_l2", "nonfinite_count")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for report in reports: writer.writerow({name: report[name] for name in fields})
    print(json.dumps({"output": str(args.output.resolve()), "seed_count": len(reports), "aggregate": aggregate}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
