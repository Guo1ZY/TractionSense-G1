#!/usr/bin/env python3
"""Evaluate one or more canonical NPZ trajectories and create CSV/Markdown/plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.evaluation import evaluate_npz  # noqa: E402
from unitree_rl_lab.traction.experiments import (  # noqa: E402
    EXPERIMENTS,
    write_experiment_registry,
)


def _plot(data: dict[str, np.ndarray], output: Path) -> None:
    environment_id = int(data["environment_id"][0, 0])
    indices = np.flatnonzero(data["environment_id"].reshape(-1) == environment_id)
    order = np.argsort(data["timestamp_s"][indices, 0])
    indices = indices[order]
    time = data["timestamp_s"][indices, 0]
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    force = data["ideal_force_xyz_n"][indices]
    for component, label in enumerate(("L_Fx", "L_Fy", "L_Fz")):
        axes[0].plot(time, force[:, component], label=label)
    for component, label in enumerate(("R_Fx", "R_Fy", "R_Fz"), start=3):
        axes[0].plot(time, force[:, component], label=label, linestyle="--")
    axes[0].set_ylabel("ideal force (N)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(time, data["force_normal_n"][indices, 0], label="L Fn")
    axes[1].plot(time, data["force_normal_n"][indices, 1], label="R Fn")
    axes[1].plot(time, data["force_tangent_n"][indices, 0], label="L Ft")
    axes[1].plot(time, data["force_tangent_n"][indices, 1], label="R Ft")
    axes[1].set_ylabel("force (N)")
    axes[1].legend(ncol=4, fontsize=8)
    axes[2].plot(
        time,
        data["friction_utilization"][indices, 0],
        label="L utilization",
    )
    axes[2].plot(
        time,
        data["friction_utilization"][indices, 1],
        label="R utilization",
    )
    axes[2].plot(time, data["slip_speed_proxy"][indices, 0], label="L slip proxy")
    axes[2].plot(time, data["slip_speed_proxy"][indices, 1], label="R slip proxy")
    axes[2].plot(time, data["ground_friction_mu"][indices, 0], label="μ")
    axes[2].legend(ncol=5, fontsize=8)
    axes[3].plot(time, data["command"][indices, 0], label="command vx")
    axes[3].plot(time, data["base_velocity"][indices, 0], label="actual vx")
    axes[3].plot(time, data["command"][indices, 2], label="command yaw")
    if "base_yaw_rate" in data:
        axes[3].plot(time, data["base_yaw_rate"][indices, 0], label="actual yaw")
    axes[3].set_ylabel("command / velocity")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    first_data = None
    for dataset in args.datasets:
        metrics, data = evaluate_npz(dataset)
        rows.append(metrics)
        if first_data is None:
            first_data = data
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Traction evaluation summary",
        "",
        "Values marked `nan` require labels or signals absent from the supplied "
        "trajectory; they are not imputed.",
        "",
    ]
    for row in rows:
        lines.extend((f"## {row['dataset']}", ""))
        for key in fieldnames:
            if key != "dataset":
                lines.append(f"- {key}: {row.get(key, float('nan'))}")
        lines.append("")
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    write_experiment_registry(args.output_dir / "experiment_registry.json")
    if first_data is not None:
        _plot(first_data, args.output_dir / "trajectory.png")
    print(
        {
            "datasets": len(rows),
            "experiment_configs": len(EXPERIMENTS),
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
