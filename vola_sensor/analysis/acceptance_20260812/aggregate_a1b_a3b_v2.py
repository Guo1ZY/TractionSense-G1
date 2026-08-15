#!/usr/bin/env python3
"""Aggregate the A1b/A3-v2 ±15 m floor runs into drift-gate tables."""

from __future__ import annotations

import json
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


def f(value, nd=3):
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    return f"{value:.{nd}f}"


def uniform_rows():
    rows = []
    for policy, prefix in (
        ("model_49999", "model49999"),
        ("model_52", "model52"),
    ):
        for seed in (550, 551, 552):
            d = load(f"{prefix}_uniform70s_w30_seed{seed}.json")
            if d is None:
                rows.append({"policy": policy, "seed": seed, "status": "missing"})
                continue
            a = d.get("aggregate") or {}
            env = d.get("environment") or {}
            rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "status": d.get("status"),
                    "envs": d.get("num_envs"),
                    "duration_s": d.get("requested_duration_s"),
                    "falls": a.get("fall_event_count"),
                    "timeouts": a.get("timeout_count"),
                    "survival_frac": a.get("survival_fraction"),
                    "mean_survival_s": a.get("mean_survival_s"),
                    "mean_vx": a.get("mean_body_vx_m_s"),
                    "min_env_vx": a.get("minimum_per_env_mean_vx_m_s"),
                    "mean_vy": a.get("mean_body_vy_m_s"),
                    "vy_rms": a.get("body_vy_rms_m_s"),
                    "bias_frac": (
                        abs(a.get("mean_body_vy_m_s"))
                        / a.get("body_vy_rms_m_s")
                        if a.get("body_vy_rms_m_s", 0) > 1.0e-9
                        else None
                    ),
                    "heading_rms": a.get("heading_rms_rad"),
                    "mean_final_y": a.get("mean_final_cross_track_m"),
                    "max_abs_final_y": a.get("maximum_abs_final_cross_track_m"),
                    "p95_abs_y": a.get("p95_abs_cross_track_m"),
                    "width_m": (env.get("floor") or {})
                    .get("friction_high_start", {})
                    .get("size_m", [None, None])[1],
                    "length_m": (env.get("floor") or {})
                    .get("friction_high_end", {})
                    .get("size_m", [None])[0],
                }
            )
    return rows


def course_rows():
    rows = []
    for policy, prefix in (
        ("model_52", "gate52"),
        ("baseline_49999", "baseline49999"),
    ):
        for seed in (450, 451, 452):
            d = load(f"{prefix}_long30m_seed{seed}.json")
            if d is None:
                rows.append({"policy": policy, "seed": seed, "status": "missing"})
                continue
            nr = d.get("natural_rollout") or {}
            tr = nr.get("transition_response") or {}
            dg = tr.get("drift_gate") or {}
            vx = nr.get("mean_body_vx_m_s") or {}
            rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "envs": d.get("num_envs"),
                    "steps": nr.get("steps_run"),
                    "completed_hlh": nr.get("completed_hlh_envs"),
                    "fall_envs": nr.get("fall_envs"),
                    "fall_events": nr.get("fall_events"),
                    "edge_exit": tr.get("edge_exit_fall_envs"),
                    "dynamic_fall": tr.get("dynamic_fall_envs"),
                    "vx_high_start": vx.get("high_start"),
                    "vx_low": vx.get("low"),
                    "vx_high_end": vx.get("high_end"),
                    "mean_vy": dg.get("aggregate_mean_body_vy_m_s"),
                    "vy_rms": dg.get("aggregate_body_vy_rms_m_s"),
                    "bias_frac": dg.get("lateral_bias_fraction"),
                    "heading_rms": dg.get("aggregate_heading_rms_rad"),
                    "max_abs_y": dg.get("max_abs_cross_track_m"),
                    "p95_abs_y": dg.get("p95_abs_cross_track_m"),
                    "mean_final_y": dg.get("mean_final_cross_track_m"),
                    "max_abs_final_y": dg.get("max_abs_final_cross_track_m"),
                    "half_width_m": dg.get("course_half_width_m"),
                    "drift_gate_pass": dg.get("pass"),
                    "recovery_median_s": (
                        (tr.get("absolute_high_recovery") or {})
                        .get("time_s", {})
                        .get("median")
                    ),
                }
            )
    return rows


def main() -> int:
    print("== Pure high-friction 70 s, ±15 m floor ==")
    header = [
        "policy", "seed", "envs", "falls", "timeout", "surv", "meanSurv_s",
        "vx", "minVx", "meanVy", "vyRMS", "bias", "psiRMS",
        "finalY", "maxFinalY", "p95Y",
    ]
    print("  ".join(h.ljust(11 if i == 0 else 8) for i, h in enumerate(header)))
    for row in uniform_rows():
        values = [
            row.get("policy"), row.get("seed"), row.get("envs"),
            row.get("falls"), row.get("timeouts"), row.get("survival_frac"),
            row.get("mean_survival_s"), row.get("mean_vx"), row.get("min_env_vx"),
            row.get("mean_vy"), row.get("vy_rms"), row.get("bias_frac"),
            row.get("heading_rms"), row.get("mean_final_y"),
            row.get("max_abs_final_y"), row.get("p95_abs_y"),
        ]
        print("  ".join((f(v).ljust(11 if i == 0 else 8)) for i, v in enumerate(values)))

    print()
    print("== A3-v2 long H→L→H course, ±15 m floor ==")
    header2 = [
        "policy", "seed", "HLH", "falls", "edgeExit", "dynFall",
        "vxH", "vxL", "vxHe", "meanVy", "vyRMS", "bias",
        "psiRMS", "maxY", "p95Y", "finalY", "recMed_s", "driftPass",
    ]
    print("  ".join(h.ljust(9 if i == 0 else 7) for i, h in enumerate(header2)))
    for row in course_rows():
        values = [
            row.get("policy"), row.get("seed"), row.get("completed_hlh"),
            row.get("fall_envs"), row.get("edge_exit"), row.get("dynamic_fall"),
            row.get("vx_high_start"), row.get("vx_low"), row.get("vx_high_end"),
            row.get("mean_vy"), row.get("vy_rms"), row.get("bias_frac"),
            row.get("heading_rms"), row.get("max_abs_y"), row.get("p95_abs_y"),
            row.get("mean_final_y"), row.get("recovery_median_s"),
            row.get("drift_gate_pass"),
        ]
        print("  ".join((f(v).ljust(9 if i == 0 else 7)) for i, v in enumerate(values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
