#!/usr/bin/env python3
"""Score a transition-retention candidate against the five gates.

Gates:
  G1 low-friction adaptation retained  (stride still shortens in LOW)
  G2 speed recovery retained           (high-end recovery median <= ~1.2 s)
  G3 heading retention                 (course heading RMS down vs model52)
  G4 lateral retention                 (max/final |y| tightened vs model52)
  G5 dynamics safety                   (course dynamics falls <= baseline)

Reference rows are the ±15 m A3-v2 JSONs already present in this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/home/mosense/guo_1/vola_sensor/analysis/acceptance_20260812")


def load(name: str):
    path = OUT / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"corrupt {name}: {exc}")
        return None


def course_row(name: str):
    d = load(name)
    if d is None:
        return None
    nr = d.get("natural_rollout") or {}
    tr = nr.get("transition_response") or {}
    dg = tr.get("drift_gate") or {}
    ga = nr.get("gait_adaptation") or {}
    regions = ga.get("regions") or {}
    vx = nr.get("mean_body_vx_m_s") or {}
    rec = (tr.get("absolute_high_recovery") or {}).get("time_s") or {}
    return {
        "completed_hlh": nr.get("completed_hlh_envs"),
        "fall_envs": nr.get("fall_envs"),
        "dynamic_fall": tr.get("dynamic_fall_envs"),
        "edge_exit": tr.get("edge_exit_fall_envs"),
        "vx_high": vx.get("high_start"),
        "vx_low": vx.get("low"),
        "vx_high_end": vx.get("high_end"),
        "step_high": (regions.get("high_start") or {}).get("mean_step_length_m"),
        "step_low": (regions.get("low") or {}).get("mean_step_length_m"),
        "rec_median": rec.get("median"),
        "heading_rms": dg.get("aggregate_heading_rms_rad"),
        "vy_rms": dg.get("aggregate_body_vy_rms_m_s"),
        "max_abs_y": dg.get("max_abs_cross_track_m"),
        "mean_final_y": dg.get("mean_final_cross_track_m"),
        "max_abs_final_y": dg.get("max_abs_final_cross_track_m"),
    }


def mean_over_seeds(rows, key):
    values = [row[key] for row in rows if row is not None and row.get(key) is not None]
    return sum(values) / len(values) if values else None


def total_over_seeds(rows, key):
    values = [row[key] for row in rows if row is not None and row.get(key) is not None]
    return sum(values) if values else None


def report(name: str, rows, ref_rows, base_rows=None):
    if any(row is None for row in rows):
        print(f"{name}: MISSING RUNS")
        return None
    print(f"== {name} ==")
    print(
        f"  HLH {total_over_seeds(rows, 'completed_hlh')}/48 | "
        f"falls {total_over_seeds(rows, 'fall_envs')} "
        f"(dyn {total_over_seeds(rows, 'dynamic_fall')}, "
        f"edge {total_over_seeds(rows, 'edge_exit')})"
    )
    print(
        f"  vx H/L/He {mean_over_seeds(rows, 'vx_high'):.3f}/"
        f"{mean_over_seeds(rows, 'vx_low'):.3f}/"
        f"{mean_over_seeds(rows, 'vx_high_end'):.3f} | "
        f"step H->L {mean_over_seeds(rows, 'step_high'):.3f}->"
        f"{mean_over_seeds(rows, 'step_low'):.3f} m | "
        f"rec {mean_over_seeds(rows, 'rec_median'):.3f} s"
    )
    print(
        f"  heading_rms {mean_over_seeds(rows, 'heading_rms'):.3f} rad | "
        f"vy_rms {mean_over_seeds(rows, 'vy_rms'):.3f} m/s | "
        f"mean final_y {mean_over_seeds(rows, 'mean_final_y'):.2f} m | "
        f"max|y| {mean_over_seeds(rows, 'max_abs_y'):.2f} m"
    )
    gates = {
        "G1 low adaptation": (
            mean_over_seeds(rows, "step_low") is not None
            and mean_over_seeds(rows, "step_low")
            <= 0.75 * (mean_over_seeds(rows, "step_high") or 1.0)
        ),
        "G2 recovery": (
            mean_over_seeds(rows, "rec_median") is not None
            and mean_over_seeds(rows, "rec_median") <= 1.2
        ),
        # Staged absolute targets: the user's milestone is heading RMS in
        # 0.2-0.25 rad (model52 course value is 0.636 rad) and eventually
        # |Δy| < 1 m.  G4 keeps every env inside the ±15 m floor with margin.
        "G3 heading": (
            mean_over_seeds(rows, "heading_rms") is not None
            and mean_over_seeds(rows, "heading_rms") <= 0.25
        ),
        "G4 lateral": (
            mean_over_seeds(rows, "max_abs_y") is not None
            and mean_over_seeds(rows, "max_abs_y") <= 12.0
            and mean_over_seeds(rows, "mean_final_y") is not None
            and abs(mean_over_seeds(rows, "mean_final_y")) <= 5.0
        ),
        "G5 dynamics": (
            total_over_seeds(rows, "dynamic_fall") is not None
            and total_over_seeds(rows, "dynamic_fall")
            <= (
                total_over_seeds(base_rows, "dynamic_fall")
                if base_rows is not None
                else 0
            )
        ),
    }
    for gate, passed in gates.items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")
    return gates


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if not label:
        print("usage: gate_transition_retention.py <label>")
        return 2
    ref = [course_row(f"gate52_long30m_seed{s}.json") for s in (450, 451, 452)]
    base = [course_row(f"baseline49999_long30m_seed{s}.json") for s in (450, 451, 452)]
    cand = [
        course_row(
            f"transition_retention/{label}_long30m_seed{s}.json"
        )
        for s in (450, 451, 452)
    ]
    print("Reference model52 (A3-v2 ±15 m):")
    report("model52", ref, ref, base)
    print()
    print("Baseline 49999 (A3-v2 ±15 m):")
    report("baseline49999", base, ref, base)
    print()
    gates = report(f"candidate {label}", cand, ref, base)
    if gates is None:
        return 1
    print()
    passed = all(gates.values())
    print(f"CANDIDATE GATES: {'ALL PASS' if passed else 'NOT ALL PASS'}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
