#!/usr/bin/env python3
"""Render the Stage-0 torque-traction evidence summary from saved results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts/traction_torque"
MUJOCO_RESULTS = ROOT.parent / "unitree_mujoco/artifacts/traction_torque/matrix_student_distilled_ppo_medium"
OUTPUT = RESULTS / "figures"
COMPONENTS = ("L Fx", "L Fy", "L Fz", "R Fx", "R Fy", "R Fz")
SCENARIO_LABELS = {
    "high_friction_seed20260803": "High μ",
    "low_friction_seed20260803": "Low μ",
    "abrupt_friction_drop_seed20260803": "μ drop",
    "asymmetric_friction_seed20260803": "Asymmetric μ",
    "combined_randomization_seed20260803": "Combined rand.",
}


def _write_source_data(correction: dict, reports: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "source_force_error.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("component", "analytical_mae_n", "corrected_mae_n"))
        for component, analytical, corrected in zip(COMPONENTS, correction["analytical"]["mae_n"], correction["analytical_plus_temporal_correction"]["mae_n"], strict=True):
            writer.writerow((component, analytical, corrected))
    fields = ("scenario", "survival_time_s", "fell", "force_mae_n", "contact_f1", "slip_f1", "governor_activation_ratio", "mean_speed_scale")
    with (OUTPUT / "source_mujoco_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for report in reports:
            writer.writerow({"scenario": report["scenario"], "survival_time_s": report["survival_time_s"], "fell": report["fell_by_height_threshold"], "force_mae_n": report["force_mae_n"], "contact_f1": report["contact"]["f1"], "slip_f1": report["slip"]["f1"], "governor_activation_ratio": report["governor_activation_ratio"], "mean_speed_scale": report["mean_speed_scale"]})


def main() -> int:
    correction = json.loads((RESULTS / "temporal_force_corrector_stage0.json").read_text())
    reports = [value for value in json.loads((ROOT.parent / "unitree_mujoco/artifacts/traction_torque/matrix_student_distilled_ppo_medium_summary.json").read_text()) if value["scenario"] in SCENARIO_LABELS]
    reports = sorted(reports, key=lambda value: tuple(SCENARIO_LABELS).index(value["scenario"]))
    _write_source_data(correction, reports)

    neutral, signal, accent, danger = "#9AA5B1", "#4C78A8", "#72B7B2", "#D66A5E"
    fig, axes = plt.subplots(2, 2, figsize=(7.2047, 4.9213), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(6); width = 0.36
    analytical = np.asarray(correction["analytical"]["mae_n"]); corrected = np.asarray(correction["analytical_plus_temporal_correction"]["mae_n"])
    ax.bar(x - width / 2, analytical, width, color=neutral, label="Analytical")
    ax.bar(x + width / 2, corrected, width, color=signal, label="+ temporal correction")
    ax.set_xticks(x, COMPONENTS); ax.set_ylabel("Force MAE (N)"); ax.set_title("Force correction lowers held-out Stage-0 error")
    ax.legend(ncol=2, loc="upper left"); ax.grid(axis="y", color="#E6E9ED", linewidth=0.5)

    ax = axes[0, 1]
    labels = [SCENARIO_LABELS[report["scenario"]] for report in reports]
    survival = np.asarray([report["survival_time_s"] for report in reports]); fell = np.asarray([report["fell_by_height_threshold"] for report in reports])
    bars = ax.barh(np.arange(len(labels)), survival, color=np.where(fell, danger, accent))
    for bar, failure in zip(bars, fell, strict=True):
        if failure: bar.set_hatch("///")
    ax.axvline(4.0, color="#4D5966", linestyle="--", linewidth=0.8); ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis()
    ax.set_xlabel("Survival time (s; 4 s horizon)"); ax.set_title("Fixed-policy MuJoCo outcomes")
    ax.text(0.99, 0.02, "hatched = fell", transform=ax.transAxes, ha="right", va="bottom", color=danger)

    ax = axes[1, 0]
    x = np.arange(len(labels)); width = 0.25
    contact_f1 = [report["contact"]["f1"] for report in reports]; slip_f1 = [report["slip"]["f1"] for report in reports]; governor = [report["governor_activation_ratio"] for report in reports]
    ax.bar(x - width, contact_f1, width, color=neutral, label="Contact F1")
    ax.bar(x, slip_f1, width, color=signal, label="Slip F1")
    ax.bar(x + width, governor, width, color=accent, label="Governor active")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction / F1"); ax.set_title("Estimator and governor responses")
    ax.legend(ncol=3, loc="upper center"); ax.grid(axis="y", color="#E6E9ED", linewidth=0.5)

    ax = axes[1, 1]
    drop = np.load(MUJOCO_RESULTS / "abrupt_friction_drop_seed20260803.npz", allow_pickle=True)
    time = drop["timestamp_s"]; slip = np.max(drop["slip_probability"], axis=1)
    ax.plot(time, drop["ground_friction_mu"][:, 0], color=neutral, linewidth=1.2, label="Ground μ (truth, metric only)")
    ax.plot(time, slip, color=danger, linewidth=1.2, label="Student max slip p")
    ax.plot(time, drop["speed_scale"], color=signal, linewidth=1.2, label="Governor speed scale")
    ax.axvline(2.0, color="#4D5966", linestyle="--", linewidth=0.8); ax.set_xlim(time.min(), time.max()); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Value"); ax.set_title("Response to abrupt friction drop")
    ax.legend(loc="upper right"); ax.grid(color="#E6E9ED", linewidth=0.5)

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    fig.suptitle("Motor-torque traction adaptation — Stage-0 estimator + PPO-medium policy", fontsize=9, fontweight="bold")
    fig.savefig(OUTPUT / "torque_traction_stage0_summary.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "torque_traction_stage0_summary.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "torque_traction_stage0_summary.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT / "torque_traction_stage0_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    qa = {"backend": "Python/matplotlib", "archetype": "quantitative grid", "final_size_mm": [183, 125], "seed_count": 1, "confidence_intervals": "not shown in summary panel; separate low-mu Stage-5 evaluation uses three seeds", "source_data": ["source_force_error.csv", "source_mujoco_metrics.csv"], "exclusions": "none", "claim_limit": "Stage-0 correction improves held-out force error; nominal low-mu survives one seed, but randomized multi-seed robustness is not validated"}
    (OUTPUT / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps({"output_dir": str(OUTPUT.resolve()), **qa}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
