#!/usr/bin/env python3
"""Audit the frozen r4 actor + r16 Hall-risk candidate.

The audit deliberately separates simulation safety/adaptation acceptance from
the more ambitious speed-tracking targets.  A simulation PASS never implies
hardware validation; the packaged policy remains inactive until live Hall
normalization, bridge preflight and overhead-harness tests are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SLOT = (
    ROOT
    / "deploy/robots/g1_29dof/config/policy/velocity/"
    "hall_traction_r4_r16_candidate"
)
PLATEN = (
    ROOT
    / "artifacts/hall_traction_execution/final_platen_validation.summary.json"
)
ISAAC_SWITCH = (
    ROOT
    / "artifacts/hall_traction_execution/isaac_r15_final_governor_seed76.csv"
)
ISAAC_ZERO = (
    ROOT
    / "artifacts/hall_traction_execution/isaac_r16_zero_micro_seed77.csv"
)
ISAAC_MICRO = (
    ROOT
    / "artifacts/hall_traction_execution/isaac_r16_micro_pulse045_seed77.csv"
)
MUJOCO_ROOT = ROOT.parent / "unitree_mujoco/artifacts/hall_magnetic_crosssim"
MUJOCO_SWITCH_CASES = (
    "r15_eval_straight_seed20260830.summary.json",
    "r15_eval_straight_seed20260832.summary.json",
    "r15_eval_straight_seed20260844.summary.json",
    "r15_eval_straight_seed20260851.summary.json",
    "r15_eval_turn_seed20260852.summary.json",
    "r16_accept_unseen_straight_seed20260863.summary.json",
    "r16_accept_unseen_turn_seed20260864.summary.json",
)
MUJOCO_LOW = "r16_accept_lowonly_mu010_seed20260865.summary.json"
MUJOCO_FAULT = "r16_accept_fullfault_lowcrawl_seed20260866.summary.json"
MUJOCO_ZERO = "r16_accept_zero_lowcrawl_seed20260867.summary.json"
MUJOCO_MICRO_NPZ = "r16_accept_micro_v005_lowcrawl_seed20260868.npz"
CONTROLLER_CONFIG = ROOT / "deploy/robots/g1_29dof/config/config.yaml"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is non-finite")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def switch_adapts(case: dict) -> bool:
    applied = [float(value) for value in case["mean_applied_vx_by_phase"]]
    velocity = [float(value) for value in case["mean_vx_by_phase"]]
    return bool(
        len(applied) >= 3
        and applied[1] <= min(applied[0], applied[-1]) - 0.05
        and velocity[-1] >= velocity[1] + 0.05
    )


def audit(slot: Path = SLOT) -> dict:
    manifest = load_json(slot / "metadata/install_manifest.json")
    schema = load_json(slot / "metadata/observation_schema.json")
    action = load_json(slot / "metadata/action_training_summary.json")
    risk = load_json(slot / "metadata/risk_training_summary.json")
    platen = load_json(PLATEN)
    isaac_switch = load_csv(ISAAC_SWITCH)
    isaac_zero = load_csv(ISAAC_ZERO)
    isaac_micro = load_csv(ISAAC_MICRO)
    mujoco_switch = [
        load_json(MUJOCO_ROOT / name) for name in MUJOCO_SWITCH_CASES
    ]
    mujoco_low = load_json(MUJOCO_ROOT / MUJOCO_LOW)
    mujoco_fault = load_json(MUJOCO_ROOT / MUJOCO_FAULT)
    mujoco_zero = load_json(MUJOCO_ROOT / MUJOCO_ZERO)
    with np.load(MUJOCO_ROOT / MUJOCO_MICRO_NPZ, allow_pickle=False) as data:
        micro_vx = np.asarray(data["base_velocity"], dtype=np.float64)[:, 0]
        micro_applied = np.asarray(data["applied_command"], dtype=np.float64)[:, 0]
        micro_fall = np.asarray(data["terminated_fall"], dtype=np.float64)

    policy_model = slot / "exported/policy.onnx"
    risk_model = slot / "exported/hall_risk.onnx"
    deploy_text = (slot / "params/deploy.yaml").read_text(encoding="utf-8")
    forbidden = set(schema.get("forbidden_student_inputs", ()))

    isaac_low = min(isaac_switch, key=lambda row: number(row, "mu"))
    isaac_high_initial = isaac_switch[0]
    isaac_high_final = isaac_switch[-1]
    isaac_falls = sum(int(number(row, "falls")) for row in isaac_switch)
    zero_row = min(isaac_zero, key=lambda row: abs(number(row, "cmd_vx")))
    micro_row = min(isaac_micro, key=lambda row: abs(number(row, "cmd_vx") - 0.05))

    selected_mujoco = [*mujoco_switch, mujoco_low, mujoco_fault, mujoco_zero]
    gates = {
        "actor_offline_pass": action.get("overall") == "PASS",
        "risk_offline_pass": risk.get("overall") == "PASS"
        and all(risk.get("gates", {}).values()),
        "hall_only_schema": (
            schema.get("input_dimension") == 1864
            and schema.get("output_dimension") == 29
            and {"normal_force", "tangential_force", "ground_friction_mu"}
            <= forbidden
        ),
        "embedded_tpu_platen_pass": platen.get("pass") is True,
        "isaac_no_falls": isaac_falls == 0,
        "isaac_low_mu_slows": (
            number(isaac_low, "steady_vx")
            <= number(isaac_high_initial, "steady_vx") - 0.05
            and number(isaac_low, "steady_applied_vx_command")
            <= number(isaac_high_initial, "steady_applied_vx_command") - 0.05
        ),
        "isaac_high_mu_recovers": (
            number(isaac_high_final, "steady_vx")
            >= number(isaac_low, "steady_vx") + 0.05
        ),
        "isaac_exact_zero": (
            abs(number(zero_row, "mean_applied_vx_command")) <= 1.0e-6
            and number(zero_row, "fall_per_env") == 0.0
        ),
        "isaac_microstep_launches": (
            number(micro_row, "mean_vx") >= 0.03
            and number(micro_row, "fall_per_env") == 0.0
        ),
        "mujoco_switch_matrix_adapts": all(
            case.get("status") == "PASS"
            and case.get("fell") is False
            and case.get("nonfinite") == 0
            and switch_adapts(case)
            for case in mujoco_switch
        ),
        "mujoco_extreme_low_mu_safe": (
            mujoco_low.get("status") == "PASS"
            and mujoco_low.get("fell") is False
            and max(mujoco_low["mean_applied_vx_by_phase"]) <= 0.20
        ),
        "full_hall_loss_stops": (
            mujoco_fault.get("fault_mode") == "full"
            and mujoco_fault.get("fell") is False
            and max(map(abs, mujoco_fault["mean_applied_vx_by_phase"])) <= 1.0e-6
            and min(mujoco_fault["mean_risk_by_phase"]) >= 0.999999
        ),
        "mujoco_exact_zero": (
            max(map(abs, mujoco_zero["mean_applied_vx_by_phase"])) <= 1.0e-6
            and mujoco_zero.get("fell") is False
        ),
        "mujoco_microstep_net_forward": (
            float(micro_vx.mean()) >= 0.02
            and float(micro_vx.sum() * 0.02) >= 0.20
            and float(np.abs(micro_applied).mean()) <= 0.12
            and not bool(np.any(micro_fall > 0.5))
        ),
        "package_checksums": (
            sha256(policy_model) == manifest.get("policy_sha256")
            and sha256(risk_model) == manifest.get("risk_sha256")
        ),
        "final_governor_parameters": all(
            text in deploy_text
            for text in (
                "probability_low_enter: 0.40",
                "probability_high_enter: 0.38",
                "low_hold_s: 0.40",
                "probe_s: 1.00",
                "probe_relative_clear_drop: 0.05",
                "crawl_pulse_s: 0.45",
            )
        ),
        "package_inactive": (
            manifest.get("default_selected") is False
            and manifest.get("governor_enabled_in_artifact") is False
            and "traction_governor:\n      enabled: false" in deploy_text
            and "hall_traction_r4_r16_candidate"
            not in CONTROLLER_CONFIG.read_text(encoding="utf-8")
        ),
    }
    simulation_pass = all(gates.values())

    recovered_vx = number(isaac_high_final, "steady_vx")
    speed_gap = recovered_vx - number(isaac_low, "steady_vx")
    response_s = number(isaac_high_final, "response_time_s")
    performance_targets = {
        "isaac_recovered_vx_ge_75pct_command": recovered_vx >= 0.45,
        "isaac_high_low_speed_gap_ge_0p35": speed_gap >= 0.35,
        "isaac_recovery_within_1s": response_s <= 1.0,
    }
    performance_pass = all(performance_targets.values())
    return {
        "format": "g1-hall-traction-r4-r16-acceptance-v1",
        "status": (
            "SIMULATION_SAFETY_ADAPTATION_PASS_PERFORMANCE_TARGET_PARTIAL_REAL_HARNESS_REQUIRED"
            if simulation_pass and not performance_pass
            else "SIMULATION_ACCEPTANCE_PASS_REAL_HARNESS_REQUIRED"
            if simulation_pass
            else "SIMULATION_ACCEPTANCE_FAIL"
        ),
        "simulation_safety_adaptation_pass": simulation_pass,
        "research_performance_target_pass": performance_pass,
        "hardware_validated": False,
        "measurement_boundary": (
            "runtime input is Hall Bx/By/Bz response + proprioceptive history; "
            "contact force and friction are simulator mechanics/offline labels only"
        ),
        "mechanical_model": (
            "one local TPU material state per Hall site; four embedded magnetic "
            "inclusions share that deformation and have no independent rigid-body DOF"
        ),
        "candidate": str(slot.resolve()),
        "gates": gates,
        "performance_targets": performance_targets,
        "metrics": {
            "risk_nominal_balanced_accuracy": risk["nominal"]["balanced_accuracy"],
            "risk_randomized_balanced_accuracy": risk["randomized"]["balanced_accuracy"],
            "risk_onnx_parity_max_abs": risk["onnx_parity_max_abs"],
            "platen_unload_max_abs_t": platen["unload_max_abs_T"],
            "isaac_falls": isaac_falls,
            "isaac_high_initial_vx_m_s": number(isaac_high_initial, "steady_vx"),
            "isaac_low_vx_m_s": number(isaac_low, "steady_vx"),
            "isaac_recovered_vx_m_s": recovered_vx,
            "isaac_recovery_s": response_s,
            "isaac_zero_applied_vx_m_s": number(zero_row, "mean_applied_vx_command"),
            "isaac_microstep_vx_m_s": number(micro_row, "mean_vx"),
            "mujoco_microstep_mean_vx_m_s": float(micro_vx.mean()),
            "mujoco_microstep_displacement_m": float(micro_vx.sum() * 0.02),
            "mujoco_cases": [case["trajectory"] for case in selected_mujoco],
        },
        "remaining_gate": (
            "Collect real dual-foot raw Hall/temperature data and normalization, "
            "pass live F0M1 preflight, then complete staged overhead-harness tests."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=Path, default=SLOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts/hall_traction_execution/final_r4_r16_acceptance.json",
    )
    args = parser.parse_args()
    try:
        report = audit(args.slot)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"[ERROR] {error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["simulation_safety_adaptation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
