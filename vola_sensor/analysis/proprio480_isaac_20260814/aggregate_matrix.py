#!/usr/bin/env python3
"""Aggregate the per-seed matrix JSONs into a mean±std comparison table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POLICIES = ("p480", "r5", "base49999")
CONSTANT_MU = ("0.8", "0.28", "0.20", "0.10")


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return mean, math.sqrt(variance)


def load_matrix(dir_path: Path, label: str) -> dict:
    seeds = (450, 451, 452)
    result = {}
    for mu in CONSTANT_MU:
        rows = []
        for seed in seeds:
            path = dir_path / f"{label}_constant_mu{mu}_seed{seed}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            segment = payload["segments"][0]
            rows.append(
                {
                    "fall_event_count": payload["aggregate"]["fall_event_count"],
                    "survival": payload["aggregate"]["survival_completion_count"],
                    "n": payload["run"]["num_envs"],
                    "vx": segment["mean_vx_m_s"],
                    "vy_rms": segment["vy_rms_m_s"],
                    "heading_rms": segment["heading_rms_rad"],
                    "tilt_rms": segment["tilt_rms"],
                    "slip": segment["mean_slip_m_s"],
                    "cadence": segment["step_frequency_hz"],
                    "step_length": segment["mean_step_length_m"],
                }
            )
        result[mu] = rows

    time_hlh = []
    for seed in seeds:
        path = dir_path / f"{label}_time_hlh_seed{seed}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        time_hlh.append(
            {
                "fall_event_count": payload["aggregate"]["fall_event_count"],
                "survival": payload["aggregate"]["survival_completion_count"],
                "n": payload["run"]["num_envs"],
                "segments": [
                    {
                        "vx": seg["mean_vx_m_s"],
                        "vy_rms": seg["vy_rms_m_s"],
                        "heading_rms": seg["heading_rms_rad"],
                        "slip": seg["mean_slip_m_s"],
                        "cadence": seg["step_frequency_hz"],
                        "step_length": seg["mean_step_length_m"],
                    }
                    for seg in payload["segments"]
                ],
            }
        )
    result["time_hlh"] = time_hlh
    return result


def aggregate_rows(rows: list[dict], key: str) -> tuple:
    values = [row.get(key) for row in rows]
    return _mean_std([value for value in values if value is not None])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    summary = {"constant": {}, "time_hlh": {}}
    for policy in POLICIES:
        data = load_matrix(args.eval_dir, policy)
        summary["constant"][policy] = {}
        for mu in CONSTANT_MU:
            rows = data[mu]
            summary["constant"][policy][mu] = {
                "seeds_complete": len(rows),
                "mean_fall_events_per_seed": aggregate_rows(rows, "fall_event_count")[0],
                "survival": aggregate_rows(rows, "survival")[0],
                "n_envs_per_seed": rows[0]["n"] if rows else None,
                "vx": aggregate_rows(rows, "vx"),
                "vy_rms": aggregate_rows(rows, "vy_rms"),
                "heading_rms": aggregate_rows(rows, "heading_rms"),
                "tilt_rms": aggregate_rows(rows, "tilt_rms"),
                "slip": aggregate_rows(rows, "slip"),
                "cadence": aggregate_rows(rows, "cadence"),
                "step_length": aggregate_rows(rows, "step_length"),
            }
        rows = data["time_hlh"]
        seg_agg = []
        for seg_idx in range(3):
            values = {}
            for key in ("vx", "vy_rms", "heading_rms", "slip", "cadence", "step_length"):
                values[key] = aggregate_rows(
                    [
                        {"v": row["segments"][seg_idx][key]}
                        for row in rows
                        if len(row["segments"]) > seg_idx
                    ],
                    "v",
                )
            seg_agg.append(values)
        summary["time_hlh"][policy] = {
            "seeds_complete": len(rows),
            "mean_fall_events_per_seed": aggregate_rows(rows, "fall_event_count")[0],
            "survival": aggregate_rows(rows, "survival")[0],
            "segments": seg_agg,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
