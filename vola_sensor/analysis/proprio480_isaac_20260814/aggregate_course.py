#!/usr/bin/env python3
"""Aggregate position-course JSONs from two schemas into one comparison table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def bare480_row(path: Path) -> dict:
    payload = json.loads(path.read_text())
    aggregate = payload["aggregate"]
    return {
        "total": payload["run"]["num_envs"],
        "completed": _num(aggregate.get("completion_count")),
        "fall_events": _num(aggregate.get("fall_event_count")),
        "unique_falls": _num(aggregate.get("unique_env_first_fall_count")),
        "vx_high": _num(aggregate.get("mean_body_vx_m_s")),
        "vx_low": _num(aggregate.get("low_mean_body_vx_m_s")),
        "vx_high_end": _num(aggregate.get("high_end_mean_body_vx_m_s")),
        "heading_rms": _num(aggregate.get("heading_rms_rad")),
        "heading_low": _num(aggregate.get("low_heading_rms_rad")),
        "heading_high_end": _num(aggregate.get("high_end_heading_rms_rad")),
        "step_high": None,
        "step_low": None,
        "cadence_high": None,
        "cadence_low": None,
    }


def r5_row(path: Path) -> dict:
    payload = json.loads(path.read_text())
    nr = payload["natural_rollout"]
    vx = nr.get("mean_body_vx_m_s") or {}
    regions = nr.get("gait_adaptation", {}).get("regions", {})
    drift = ((nr.get("transition_response") or {}).get("drift_gate") or {})
    per_region = drift.get("per_region") or {}
    high = regions.get("high_start", {})
    low = regions.get("low", {})
    high_end = regions.get("high_end", {})
    return {
        "total": payload["num_envs"],
        "completed": _num(nr.get("completed_hlh_envs")),
        "fall_events": _num(nr.get("fall_events")),
        "unique_falls": _num(nr.get("fall_envs")),
        "vx_high": _num(vx.get("high_start")),
        "vx_low": _num(vx.get("low")),
        "vx_high_end": _num(vx.get("high_end")),
        "heading_rms": _num(drift.get("aggregate_heading_rms_rad")),
        "heading_low": _num(
            (per_region.get("low") or {}).get("heading_rms_rad")
        ),
        "heading_high_end": _num(
            (per_region.get("high_end") or {}).get("heading_rms_rad")
        ),
        "step_high": _num(high.get("mean_step_length_m")),
        "step_low": _num(low.get("mean_step_length_m")),
        "cadence_high": _num(high.get("step_frequency_hz")),
        "cadence_low": _num(low.get("step_frequency_hz")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    groups = {}
    for path in sorted(args.dir.glob("*.json")):
        name = path.stem
        payload = json.loads(path.read_text())
        if "natural_rollout" in payload:
            row = r5_row(path)
        elif "aggregate" in payload and "run" in payload:
            row = bare480_row(path)
        else:
            print(f"skip unknown schema: {path}")
            continue
        groups.setdefault(name.rsplit("_seed", 1)[0], []).append(row)

    summary = {}
    for name, rows in sorted(groups.items()):
        keys = (
            "completed", "fall_events", "unique_falls", "vx_high", "vx_low",
            "vx_high_end", "heading_rms", "heading_low", "heading_high_end",
            "step_high", "step_low", "cadence_high", "cadence_low",
        )
        summary[name] = {
            "seeds": len(rows),
            "total_per_seed": rows[0]["total"] if rows else None,
        }
        for key in keys:
            summary[name][key] = {
                "mean": _mean([row[key] for row in rows]),
                "values": [row[key] for row in rows],
            }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
