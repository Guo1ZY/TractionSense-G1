#!/usr/bin/env python3
"""Plot headline training curves and the constant-mu comparison chart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(headline_csv: Path, out_png: Path) -> None:
    rows = list(csv.DictReader(headline_csv.open()))
    iters = [int(row["iteration"]) for row in rows]
    series = {
        "mean_reward": [float(row["Train/mean_reward"]) for row in rows],
        "episode_length": [float(row["Train/mean_episode_length"]) for row in rows],
        "track_lin_vel_xy": [float(row["Episode_Reward/track_lin_vel_xy"]) for row in rows],
        "bad_orientation": [float(row["Episode_Termination/bad_orientation"]) for row in rows],
        "contact_slip": [float(row["Episode_Reward/contact_point_slip"]) for row in rows],
    }
    fig, axes = plt.subplots(5, 1, figsize=(7, 12), sharex=True)
    for axis, (label, values) in zip(axes, series.items()):
        axis.plot(iters, values, lw=1.2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_matrix(summary_json: Path, out_png: Path) -> None:
    summary = json.loads(summary_json.read_text())
    mu = ["0.8", "0.28", "0.20", "0.10"]
    policies = {"p480": "480-D trained", "r5": "R5 1864-D Hall", "base49999": "model_49999"}
    colors = {"p480": "tab:blue", "r5": "tab:orange", "base49999": "tab:gray"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = ("vx", "fall_events", "heading_rms", "slip")
    for axis, metric in zip(axes.ravel(), metrics):
        for policy, label in policies.items():
            means, stds = [], []
            for m in mu:
                value = summary["constant"][policy][m]
                key = (
                    "mean_fall_events_per_seed" if metric == "fall_events" else metric
                )
                pair = value[key]
                means.append(pair[0] if isinstance(pair, (list, tuple)) else pair)
                stds.append(pair[1] if isinstance(pair, (list, tuple)) else 0.0)
            axis.errorbar(
                range(len(mu)),
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                lw=1.4,
                color=colors[policy],
                label=label,
            )
        axis.set_xticks(range(len(mu)))
        axis.set_xticklabels([f"μ={m}" for m in mu])
        axis.set_title(metric)
        axis.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline_csv", type=Path)
    parser.add_argument("--summary_json", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.headline_csv and args.headline_csv.is_file():
        plot_training(args.headline_csv, args.out_dir / "training_curves.png")
    if args.summary_json and args.summary_json.is_file():
        plot_matrix(args.summary_json, args.out_dir / "constant_mu_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
