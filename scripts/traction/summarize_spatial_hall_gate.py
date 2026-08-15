#!/usr/bin/env python3
"""Summarize fixed-condition spatial Hall rollouts without hiding falls.

The evaluator intentionally keeps a fall as a failure event.  This helper
aggregates independent JSON summaries and reports both rollout-level survival
and the speed separation between the high/low/high patches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="unknown")
    args = parser.parse_args()

    rows = []
    for path in args.summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rollout = payload["natural_rollout"]
        rows.append(
            {
                "file": str(path),
                "num_envs": int(payload["num_envs"]),
                "fall_events": int(rollout["fall_events"]),
                "fall_envs": int(rollout["fall_envs"]),
                "completed_hlh_envs": int(rollout["completed_hlh_envs"]),
                "nan_detected": bool(rollout["nan_detected"]),
                "speed_m_s": rollout["mean_body_vx_m_s"],
            }
        )
    total_envs = sum(row["num_envs"] for row in rows)
    total_fall_envs = sum(row["fall_envs"] for row in rows)
    total_fall_events = sum(row["fall_events"] for row in rows)
    total_hlh = sum(row["completed_hlh_envs"] for row in rows)
    speed_rows = [row["speed_m_s"] for row in rows]
    means = {
        key: sum(float(row[key]) for row in speed_rows) / len(speed_rows)
        for key in ("high_start", "low", "high_end")
    }
    report = {
        "format": "spatial-hall-fixed-condition-gate-v1",
        "candidate": args.candidate,
        "rollout_count": len(rows),
        "total_env_rollouts": total_envs,
        "unique_fall_env_rollouts": total_fall_envs,
        "fall_event_count": total_fall_events,
        "completed_hlh_env_rollouts": total_hlh,
        "survival_fraction": 1.0 - total_fall_envs / max(total_envs, 1),
        "hlh_completion_fraction": total_hlh / max(total_envs, 1),
        "nan_free": all(not row["nan_detected"] for row in rows),
        "mean_speed_m_s": means,
        "low_to_high_start_gap_m_s": means["high_start"] - means["low"],
        "high_end_recovery_fraction": means["high_end"] / max(means["high_start"], 1.0e-9),
        "formal_zero_fall_gate": total_fall_envs == 0 and total_fall_events == 0,
        "source_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["nan_free"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
