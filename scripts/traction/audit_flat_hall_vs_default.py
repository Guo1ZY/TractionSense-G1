#!/usr/bin/env python3
"""Audit a paired flat-ground Hall policy against the original locomotion actor.

Input files are switch-mode phase CSVs produced by ``eval_friction_matrix.py``.
The audit enforces non-inferiority at high friction and improvement at low
friction.  A single rollout is reported, but cannot pass the multi-seed gate;
this prevents a favorable seed from being presented as a general result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        parsed = list(csv.DictReader(stream))
    if not parsed:
        raise ValueError(f"empty phase CSV: {path}")
    required = {"phase", "mu", "steady_vx", "steady_contact_slip", "falls"}
    missing = required - set(parsed[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return [{key: float(value) for key, value in row.items() if key != "phase"} | {"phase": float(row["phase"])} for row in parsed]


def _mean(rows: list[dict[str, float]], key: str, predicate) -> float:
    values = [row[key] for row in rows if predicate(row) and math.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def _paired(hall_path: Path, baseline_path: Path) -> dict[str, object]:
    hall = _rows(hall_path)
    baseline = _rows(baseline_path)
    if len(hall) != len(baseline):
        raise ValueError(f"phase count mismatch: {hall_path} vs {baseline_path}")
    for left, right in zip(hall, baseline, strict=True):
        if left["phase"] != right["phase"] or abs(left["mu"] - right["mu"]) > 1e-6:
            raise ValueError(f"phase/mu mismatch in {hall_path} and {baseline_path}")
    high = lambda row: row["mu"] >= 0.75
    low = lambda row: row["mu"] <= 0.25
    high_vx = _mean(hall, "steady_vx", high)
    base_high_vx = _mean(baseline, "steady_vx", high)
    low_vx = _mean(hall, "steady_vx", low)
    base_low_vx = _mean(baseline, "steady_vx", low)
    low_slip = _mean(hall, "steady_contact_slip", low)
    base_low_slip = _mean(baseline, "steady_contact_slip", low)
    high_slip = _mean(hall, "steady_contact_slip", high)
    base_high_slip = _mean(baseline, "steady_contact_slip", high)
    last_high = [row for row in hall if high(row)][-1]
    last_base_high = [row for row in baseline if high(row)][-1]
    response = [row["response_time_s"] for row in hall if "response_time_s" in row and math.isfinite(row["response_time_s"])]
    return {
        "hall_csv": str(hall_path),
        "baseline_csv": str(baseline_path),
        "hall_high_vx": high_vx,
        "baseline_high_vx": base_high_vx,
        "high_vx_relative_change": (high_vx - base_high_vx) / max(abs(base_high_vx), 1e-6),
        "hall_last_high_vx": last_high["steady_vx"],
        "baseline_last_high_vx": last_base_high["steady_vx"],
        "last_high_vx_relative_change": (last_high["steady_vx"] - last_base_high["steady_vx"]) / max(abs(last_base_high["steady_vx"]), 1e-6),
        "hall_low_vx": low_vx,
        "baseline_low_vx": base_low_vx,
        "hall_low_slip": low_slip,
        "baseline_low_slip": base_low_slip,
        "low_slip_ratio": low_slip / max(base_low_slip, 1e-6),
        "low_slip_reduction": 1.0 - low_slip / max(base_low_slip, 1e-6),
        "hall_high_slip": high_slip,
        "baseline_high_slip": base_high_slip,
        "hall_falls": int(sum(row["falls"] for row in hall)),
        "baseline_falls": int(sum(row["falls"] for row in baseline)),
        "hall_max_response_s": max(response, default=float("nan")),
    }


def _bootstrap(values: np.ndarray, seed: int = 20260809) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _write_figure(path: Path, records: list[dict[str, object]]) -> None:
    labels = [f"seed {record['seed']}" for record in records]
    x = np.arange(len(records))
    hall_low_slip = np.asarray([record["hall_low_slip"] for record in records], dtype=float)
    base_low_slip = np.asarray([record["baseline_low_slip"] for record in records], dtype=float)
    hall_high = np.asarray([record["hall_high_vx"] for record in records], dtype=float)
    base_high = np.asarray([record["baseline_high_vx"] for record in records], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, base_high, width, color="#9099a8", label="原始 locomotion")
    axes[0].bar(x + width / 2, hall_high, width, color="#159957", label="Hall 自适应")
    axes[0].set_title("高摩擦速度：非劣化约束")
    axes[0].set_ylabel("稳态 vx (m/s)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, base_low_slip, width, color="#9099a8")
    axes[1].bar(x + width / 2, hall_low_slip, width, color="#159957")
    axes[1].set_title("低摩擦滑移：必须改善")
    axes[1].set_ylabel("接触脚滑移 (m/s)")
    axes[1].set_xticks(x, labels)
    axes[2].bar(x - width / 2, base_high - hall_high, width, color="#d95f02")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Hall 相对原版的高摩擦速度差")
    axes[2].set_ylabel("原版 - Hall (m/s)")
    axes[2].set_xticks(x, labels)
    figure.suptitle("Flat-ground Hall vs original locomotion acceptance audit", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hall", type=Path, action="append", required=True, help="Hall phase CSV; repeat per seed")
    parser.add_argument("--baseline", type=Path, action="append", required=True, help="paired original phase CSV; repeat per seed")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--max-low-slip-ratio", type=float, default=0.75)
    parser.add_argument("--min-high-vx-relative", type=float, default=-0.05)
    parser.add_argument("--max-response-s", type=float, default=1.0)
    args = parser.parse_args()
    if len(args.hall) != len(args.baseline):
        raise ValueError("--hall and --baseline must have the same count")
    if len(args.hall) < 1:
        raise ValueError("at least one paired rollout is required")
    records: list[dict[str, object]] = []
    for index, (hall_path, baseline_path) in enumerate(zip(args.hall, args.baseline, strict=True)):
        record = _paired(hall_path, baseline_path)
        record["seed"] = index
        records.append(record)
    low_slip_ratios = np.asarray([record["low_slip_ratio"] for record in records], dtype=float)
    high_changes = np.asarray([record["high_vx_relative_change"] for record in records], dtype=float)
    response_values = np.asarray([record["hall_max_response_s"] for record in records], dtype=float)
    gates = {
        "minimum_seed_count": len(records) >= args.min_seeds,
        "low_friction_slip_improves": bool(np.all(low_slip_ratios <= args.max_low_slip_ratio)),
        "high_friction_speed_noninferior": bool(np.all(high_changes >= args.min_high_vx_relative)),
        "no_more_falls_than_original": bool(all(record["hall_falls"] <= record["baseline_falls"] for record in records)),
        "switch_response_bound": bool(np.all(np.isfinite(response_values)) and np.all(response_values <= args.max_response_s)),
    }
    report = {
        "format": "flat-hall-vs-default-audit-v1",
        "definition": {
            "default": "model_49999 original 480-D proprio actor",
            "hall_policy": "1864-D proprio + dual-foot 15x3 Hall B history",
            "forbidden_policy_inputs": ["ground_mu", "normal_force", "tangential_force", "contact_slip_truth"],
        },
        "thresholds": {
            "minimum_seed_count": args.min_seeds,
            "max_low_slip_ratio": args.max_low_slip_ratio,
            "min_high_vx_relative_change": args.min_high_vx_relative,
            "max_response_s": args.max_response_s,
        },
        "records": records,
        "statistics": {
            "low_slip_ratio_ci95": _bootstrap(low_slip_ratios),
            "high_vx_relative_change_ci95": _bootstrap(high_changes),
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "NEEDS_TRAINING",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for record in records for key in record})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    _write_figure(args.output_figure, records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
