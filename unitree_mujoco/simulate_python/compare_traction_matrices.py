#!/usr/bin/env python3
"""Compare paired fixed-policy MuJoCo matrices without inventing missing runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    "fell",
    "minimum_base_height_m",
    "velocity_tracking_error_mean_m_s",
    "yaw_tracking_error_mean_rad_s",
    "maximum_slip_speed_proxy_m_s",
    "slip_proxy_rate",
    "speed_scale_mean",
    "sensor_confidence_mean",
    "maximum_action_abs",
    "nonfinite",
)


def _read(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {
        (row["scenario"], int(float(row["seed"]))): row
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate scenario/seed rows")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--baseline_label",
        default="audited proprio actor with the governor disabled",
    )
    parser.add_argument(
        "--candidate_label",
        default="temporal Student with the governor enabled",
    )
    parser.add_argument(
        "--candidate_status",
        default=(
            "The candidate remains a Sim2Sim candidate until the corresponding "
            "training and evaluation evidence is reviewed."
        ),
    )
    args = parser.parse_args()
    baseline = _read(args.baseline)
    candidate = _read(args.candidate)
    if baseline.keys() != candidate.keys():
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        raise ValueError(
            "matrix keys differ: "
            f"missing_baseline={missing_baseline}, "
            f"missing_candidate={missing_candidate}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows: list[dict[str, str | float | int]] = []
    for scenario, seed in sorted(baseline):
        row: dict[str, str | float | int] = {
            "scenario": scenario,
            "seed": seed,
        }
        for metric in METRICS:
            base = float(baseline[(scenario, seed)][metric])
            full = float(candidate[(scenario, seed)][metric])
            row[f"baseline_{metric}"] = base
            row[f"candidate_{metric}"] = full
            row[f"delta_candidate_minus_baseline_{metric}"] = full - base
        paired_rows.append(row)

    with (args.output_dir / "paired_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary_rows = []
    for scenario in sorted({str(row["scenario"]) for row in paired_rows}):
        selected = [row for row in paired_rows if row["scenario"] == scenario]
        for metric in METRICS:
            baseline_values = np.asarray(
                [float(row[f"baseline_{metric}"]) for row in selected]
            )
            candidate_values = np.asarray(
                [float(row[f"candidate_{metric}"]) for row in selected]
            )
            baseline_std = (
                float(baseline_values.std(ddof=1))
                if len(selected) > 1
                else 0.0
            )
            candidate_std = (
                float(candidate_values.std(ddof=1))
                if len(selected) > 1
                else 0.0
            )
            summary_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "seeds": len(selected),
                    "baseline_mean": float(baseline_values.mean()),
                    "baseline_std": baseline_std,
                    "baseline_approx_ci95_half_width": (
                        1.96 * baseline_std / np.sqrt(len(selected))
                    ),
                    "candidate_mean": float(candidate_values.mean()),
                    "candidate_std": candidate_std,
                    "candidate_approx_ci95_half_width": (
                        1.96 * candidate_std / np.sqrt(len(selected))
                    ),
                    "delta_candidate_minus_baseline": float(
                        (candidate_values - baseline_values).mean()
                    ),
                }
            )
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Fixed-policy MuJoCo paired comparison",
        "",
        f"The baseline uses {args.baseline_label}. "
        f"The candidate uses {args.candidate_label}. "
        "Both are fixed: no MuJoCo training or fine-tuning was performed.",
        "",
        "| scenario | falls baseline/candidate | velocity error baseline/candidate | "
        "slip proxy rate baseline/candidate | speed scale baseline/candidate |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in sorted({str(row["scenario"]) for row in paired_rows}):
        by_metric = {
            str(row["metric"]): row
            for row in summary_rows
            if row["scenario"] == scenario
        }
        fall = by_metric["fell"]
        velocity = by_metric["velocity_tracking_error_mean_m_s"]
        slip = by_metric["slip_proxy_rate"]
        speed = by_metric["speed_scale_mean"]
        lines.append(
            f"| {scenario} | {fall['baseline_mean']:.3f}/"
            f"{fall['candidate_mean']:.3f} | "
            f"{velocity['baseline_mean']:.4f}/"
            f"{velocity['candidate_mean']:.4f} | "
            f"{slip['baseline_mean']:.4f}/{slip['candidate_mean']:.4f} | "
            f"{speed['baseline_mean']:.4f}/{speed['candidate_mean']:.4f} |"
        )
    lines.extend(
        (
            "",
            "These are 3-second Sim2Sim runs. Slip is the explicitly named "
            "ankle-rigid-body velocity proxy, not contact-point ground truth.",
            "The CSV includes sample standard deviation and an approximate "
            "normal 95% confidence-interval half-width across seeds; with "
            "three seeds it should be treated as descriptive, not definitive.",
            "",
            args.candidate_status,
            "",
        )
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        {
            "paired_runs": len(paired_rows),
            "scenarios": len({row["scenario"] for row in paired_rows}),
            "output": str(args.output_dir.resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
