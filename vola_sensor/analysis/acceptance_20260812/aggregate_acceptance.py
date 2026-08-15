#!/usr/bin/env python3
"""Aggregate acceptance_20260812 runs into a gated acceptance report."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/home/mosense/guo_1/vola_sensor/analysis/acceptance_20260812")


def load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        print(f"corrupt {name}: {exc}")
        return None


def course_row(f, label):
    d = load(f)
    if d is None:
        return {"run": label, "status": "missing"}
    nr = d.get("natural_rollout") or {}
    tr = nr.get("transition_response") or {}
    fast = nr.get("fastbase_capture_diagnostics") or {}
    row = {
        "run": label,
        "task": d["task"].split("-")[-1],
        "seed": d["seed"],
        "envs": d["num_envs"],
        "steps": nr.get("steps_run"),
        "completed_hlh": nr.get("completed_hlh_envs"),
        "falls": nr.get("fall_envs"),
        "fall_events": nr.get("fall_events"),
        "vx_high_start": nr.get("mean_body_vx_m_s", {}).get("high_start"),
        "vx_low": nr.get("mean_body_vx_m_s", {}).get("low"),
        "vx_high_end": nr.get("mean_body_vx_m_s", {}).get("high_end"),
        "low_entry_speed": (tr.get("low_entry_speed_m_s") or {}).get("mean"),
        "decel_0_5s": ((tr.get("deceleration_after_low_contact") or {}).get("0.5s") or {}).get("deceleration_m_s", {}).get("mean"),
        "decel_1_0s": ((tr.get("deceleration_after_low_contact") or {}).get("1s") or {}).get("deceleration_m_s", {}).get("mean"),
        "recovery_frac": (tr.get("absolute_high_recovery") or {}).get("recovery_fraction"),
        "recovery_mean_s": ((tr.get("absolute_high_recovery") or {}).get("time_s") or {}).get("mean"),
        "recovery_median_s": ((tr.get("absolute_high_recovery") or {}).get("time_s") or {}).get("median"),
        "gate_auc": (fast.get("low_vs_high_auc") or {}).get("effective_gate"),
        "gate_activation_frac": (fast.get("low_activation") or {}).get("activation_fraction"),
        "hall_mode": "hardened" if (d.get("hall_fault_profile") or {}).get("requested_hardened") else "nominal",
        "course": nr.get("course_geometry", {}).get("long_course"),
    }
    return row


def uniform_row(f, label):
    d = load(f)
    if d is None:
        return {"run": label, "status": "missing"}
    agg = d.get("aggregate") or {}
    gates = d.get("gates") or {}
    env = d.get("environment") or {}
    row = {
        "run": label,
        "status": d.get("status"),
        "width_m": (env.get("floor") or {}).get("friction_high_start", {}).get("size_m", [None, None])[1],
        "seed": d.get("seed"),
        "envs": d.get("num_envs"),
        "duration_s": d.get("requested_duration_s"),
        "fall_event_count": agg.get("fall_event_count"),
        "first_fall_envs": agg.get("unique_env_first_fall_count"),
        "survival_fraction": agg.get("survival_fraction"),
        "mean_survival_s": agg.get("mean_survival_s"),
        "mean_vx": agg.get("mean_body_vx_m_s"),
        "min_env_vx": agg.get("minimum_per_env_mean_vx_m_s"),
        "heading_rms": agg.get("heading_rms_rad"),
        "body_vy_rms": agg.get("body_vy_rms_m_s"),
        "angvel_rms": agg.get("angular_velocity_rms_rad_s"),
        "action_sat": agg.get("action_saturation_fraction"),
        "cross_track_p95": agg.get("p95_abs_cross_track_m"),
        "gate_pass": gates.get("pass"),
        "gates": {k: v for k, v in gates.items() if k != "pass"},
    }
    return row


rows = []
for s in (450, 451, 452, 453, 454, 455):
    rows.append(course_row(f"gate52_medium_nominal_seed{s}.json", f"m52-medium-nom-s{s}"))
for s in (450, 451, 452, 453, 454, 455):
    rows.append(course_row(f"gate52_medium_hardened_seed{s}.json", f"m52-medium-hard-s{s}"))
for s in (450, 451, 452):
    rows.append(course_row(f"gate52_long_seed{s}.json", f"m52-long-s{s}"))
    rows.append(course_row(f"baseline49999_long_seed{s}.json", f"base-long-s{s}"))
rows.append(course_row("original_unitree_medium_nominal_seed497_32env.json", "base-medium-s497(32env)"))

urows = []
for s in (550, 551, 552):
    urows.append(uniform_row(f"model49999_uniform_width12_seed{s}.json", f"m49999-width12-s{s}"))
urows.append(uniform_row("model49999_uniform_width32_seed550_ref.json", "m49999-width3.2-s550-ref"))

def fmt(x, nd=3):
    return "None" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

print("== Course (H->L->H) runs ==")
hdr = ["run", "hall", "seed", "envs", "HLH", "falls", "vxH", "vxL", "vxHe",
       "lowEnt", "dec05", "dec10", "recFrac", "recMed_s", "auc", "actFrac", "long"]
print("  ".join(h.ljust(14 if i == 0 else 7) for i, h in enumerate(hdr)))
for r in rows:
    vals = [r.get("run"), r.get("hall_mode"), r.get("seed"), r.get("envs"),
            r.get("completed_hlh"), r.get("falls"), r.get("vx_high_start"), r.get("vx_low"),
            r.get("vx_high_end"), r.get("low_entry_speed"), r.get("decel_0_5s"),
            r.get("decel_1_0s"), r.get("recovery_frac"), r.get("recovery_median_s"),
            r.get("gate_auc"), r.get("gate_activation_frac"),
            "L" if r.get("course") else "S"]
    print("  ".join((fmt(v).ljust(14 if i == 0 else 7)) for i, v in enumerate(vals)))

print()
print("== Uniform high-friction 30 s runs (model_49999) ==")
hdr2 = ["run", "width", "seed", "envs", "falls", "firstFallEnv", "surv", "meanSurv_s",
        "meanVx", "minVx", "headingRMS", "vyRMS", "angRMS", "actSat", "ct95", "gate"]
print("  ".join(h.ljust(16 if i == 0 else 11) for i, h in enumerate(hdr2)))
for r in urows:
    vals = [r.get("run"), r.get("width_m"), r.get("seed"), r.get("envs"),
            r.get("fall_event_count"), r.get("first_fall_envs"), r.get("survival_fraction"),
            r.get("mean_survival_s"), r.get("mean_vx"), r.get("min_env_vx"),
            r.get("heading_rms"), r.get("body_vy_rms"), r.get("angvel_rms"),
            r.get("action_sat"), r.get("cross_track_p95"), r.get("gate_pass")]
    print("  ".join((fmt(v, 3 if i > 5 else 1)).ljust(16 if i == 0 else 11) for i, v in enumerate(vals)))

print()
print("== Acceptance criteria (from CURRENT_MODEL_POLICY_SUMMARY) ==")
print("""
1. wide road, same command/seed/DR vs original baseline  : A1 (uniform 12m) + A3 (long H->L->H)
2. nominal + Hall fault multi-seed                       : A2 (medium 6 seeds nominal+hardened)
3. first-fall survival + cumulative reset dual report    : in course rows (falls per env, events)
4. high-friction speed loss <= 5%                        : vxH vs baseline/command
5. low-friction 0 fall, cadence/stride adaptation        : decel + vxL + AUC
6. recovery latency on return to high                    : recFrac/recMed_s
7. lateral/heading/vy/angvel/action saturation bounded   : uniform rows + A3
""")
