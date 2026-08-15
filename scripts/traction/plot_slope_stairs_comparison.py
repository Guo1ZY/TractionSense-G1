#!/usr/bin/env python3
"""Plot paired Hall-policy results on plane, slopes and stairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TERRAINS = ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down")
DISPLAY = ("Flat", "Uphill", "Downhill", "Stairs up", "Stairs down")


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    "terrain": raw["terrain_type"],
                    "phase": float(raw["phase"]),
                    "mu": float(raw["mu"]),
                    "vx": float(raw["steady_vx"]),
                    "slip": float(raw["steady_contact_slip"]),
                    "risk": float(raw["steady_low_traction_probability"]),
                    "falls": float(raw["falls"]),
                }
            )
    return rows


def terrain_metric(
    rows: list[dict[str, float | str]], metric: str, *, total: bool = False
) -> np.ndarray:
    result = []
    for terrain in TERRAINS:
        values = [float(row[metric]) for row in rows if row["terrain"] == terrain]
        result.append(sum(values) if total else float(np.mean(values)))
    return np.asarray(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Previous flat policy")
    parser.add_argument("--candidate-label", default="Slope/stairs Hall policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_rows(args.baseline)
    candidate = read_rows(args.candidate)
    x = np.arange(len(TERRAINS))
    width = 0.36
    colors = ("#8A94A6", "#0072B2")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    panels = (
        ("falls", True, "Fall events (3 friction phases)"),
        ("vx", False, "Mean steady forward speed (m/s)"),
        ("slip", False, "Mean contact-slip proxy (m/s)"),
        ("risk", False, "Hall low-traction probability"),
    )
    for axis, (metric, total, title) in zip(axes.flat, panels):
        old = terrain_metric(baseline, metric, total=total)
        new = terrain_metric(candidate, metric, total=total)
        axis.bar(x - width / 2, old, width, color=colors[0], label=args.baseline_label)
        axis.bar(x + width / 2, new, width, color=colors[1], label=args.candidate_label)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, DISPLAY, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric in ("slip", "risk"):
            axis.axhline(0.0, color="black", linewidth=0.7)
        if metric == "risk":
            axis.axhline(0.65, color="#D55E00", linestyle="--", linewidth=1.2, label="slow-down threshold")
        for offset, values in ((-width / 2, old), (width / 2, new)):
            for index, value in enumerate(values):
                axis.text(
                    index + offset,
                    value,
                    f"{value:.0f}" if metric == "falls" else f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Hall-only magnetic-foot policy: paired Isaac Sim terrain evaluation",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
