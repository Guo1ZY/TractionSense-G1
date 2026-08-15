#!/usr/bin/env python3
"""Summarize the independent MuJoCo Hall-only cross-simulation checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hall", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hall, baseline, fault = map(read, (args.hall, args.baseline, args.fault))
    hall_speed = hall["mean_vx_by_phase"]
    report = {
        "format": "mujoco-hall-crosssim-gate-v1",
        "measurement_boundary": hall["measurement_boundary"],
        "hall_normal": {
            "fell": hall["fell"],
            "nonfinite": hall["nonfinite"],
            "mean_vx_by_phase": hall_speed,
            "risk_by_phase": hall["mean_risk_by_phase"],
            "status": hall["status"],
        },
        "original_baseline": {
            "fell": baseline["fell"],
            "nonfinite": baseline["nonfinite"],
            "mean_vx_by_phase": baseline["mean_vx_by_phase"],
            "status": baseline["status"],
        },
        "hall_full_fault": {
            "fell": fault["fell"],
            "nonfinite": fault["nonfinite"],
            "mean_applied_vx_by_phase": fault["mean_applied_vx_by_phase"],
            "risk_by_phase": fault["mean_risk_by_phase"],
            "status": fault["status"],
        },
        # Keep the strict high/low speed gate visible.  It is intentionally
        # not relabeled PASS when the current spatial actor only adapts weakly
        # in the independent MuJoCo dynamics.
        "gates": {
            "hall_survives_normal": not hall["fell"] and hall["nonfinite"] == 0,
            "baseline_fails_same_case": bool(baseline["fell"]),
            "hall_full_fault_fail_safe": fault["status"] == "PASS",
            "hall_speed_separation_ge_0_08_m_s": hall_speed[0] - hall_speed[1] >= 0.08,
            "sim2sim_performance_gate": hall["status"] == "PASS",
        },
        "status": (
            "SAFETY_AND_FAULT_PASS_SPEED_CALIBRATION_PENDING"
            if (not hall["fell"] and fault["status"] == "PASS")
            else "CROSSSIM_FAIL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "CROSSSIM_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
