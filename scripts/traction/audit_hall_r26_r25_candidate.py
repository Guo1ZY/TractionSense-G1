#!/usr/bin/env python3
"""Audit the inactive r26 Hall recovery + r25 Hall risk deployment candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLOT = ROOT / (
    "deploy/robots/g1_29dof/config/policy/velocity/"
    "hall_traction_r26_r25_candidate"
)
CONTINUATION = ROOT / "artifacts/hall_traction_continuation"
MUJOCO = ROOT.parent / "unitree_mujoco/artifacts/hall_magnetic_crosssim"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def phase_value(rows: list[dict[str, str]], phase: int, key: str) -> float:
    return float(next(row for row in rows if int(row["phase"]) == phase)[key])


def main() -> int:
    manifest = load_json(SLOT / "metadata/install_manifest.json")
    schema = load_json(SLOT / "metadata/observation_schema.json")
    recovery = load_json(SLOT / "metadata/action_training_summary.json")
    risk = load_json(SLOT / "metadata/risk_training_summary.json")
    latency = load_json(CONTINUATION / "r26_r25_cpu_latency.json")
    platen = load_json(
        ROOT / "artifacts/hall_traction_execution/final_platen_validation.summary.json"
    )
    isaac_106 = load_csv(CONTINUATION / "r26_isaac_randomized_seed106.csv")
    isaac_107 = load_csv(CONTINUATION / "r26_isaac_randomized_seed107.csv")
    isaac_turn = load_csv(CONTINUATION / "r26_isaac_turn_randomized_seed108.csv")
    isaac_control = load_csv(CONTINUATION / "r4_r25_isaac_control_seed106.csv")
    mujoco_cases = [
        load_json(MUJOCO / name)
        for name in (
            "r26_closedloop_nominal_seed20260837.summary.json",
            "r26_closedloop_randomized_seed20260838.summary.json",
            "r26_closedloop_full_fault_seed20260839.summary.json",
            "r26_closedloop_turn_randomized_seed20260840.summary.json",
            "r26_closedloop_left_fault_turn_seed20260841.summary.json",
        )
    ]
    deploy_text = (SLOT / "params/deploy.yaml").read_text(encoding="utf-8")
    controller_text = (
        ROOT / "deploy/robots/g1_29dof/config/config.yaml"
    ).read_text(encoding="utf-8")
    forbidden = set(schema.get("forbidden_student_inputs", ()))
    isaac_cases = (isaac_106, isaac_107, isaac_turn)
    isaac_falls = sum(
        int(float(row["falls"])) for rows in isaac_cases for row in rows
    )
    low_vx = [phase_value(rows, 1, "steady_vx") for rows in isaac_cases]
    high_vx = [
        0.5 * (
            phase_value(rows, 0, "steady_vx")
            + phase_value(rows, 2, "steady_vx")
        )
        for rows in isaac_cases
    ]
    low_slip_r26 = phase_value(isaac_106, 1, "steady_contact_slip")
    low_slip_control = phase_value(isaac_control, 1, "steady_contact_slip")

    gates = {
        "recovery_offline_pass": recovery.get("status") == "PASS"
        and all(recovery.get("gates", {}).values()),
        "risk_offline_pass": risk.get("overall") == "PASS"
        and all(risk.get("gates", {}).values()),
        "hall_only_schema": schema.get("input_dimension") == 1864
        and schema.get("output_dimension") == 29
        and {"normal_force", "tangential_force", "ground_friction_mu"}
        <= forbidden,
        "tpu_magnetic_platen_pass": platen.get("pass") is True,
        "isaac_192_environment_phases_no_fall": isaac_falls == 0,
        "isaac_low_friction_adapts": all(
            high - low >= 0.08 for high, low in zip(high_vx, low_vx)
        ),
        "isaac_recovery_does_not_increase_slip": low_slip_r26 <= low_slip_control,
        "mujoco_straight_turn_random_fault_pass": all(
            case.get("status") == "PASS"
            and case.get("fell") is False
            and case.get("nonfinite") == 0
            for case in mujoco_cases
        ),
        "full_or_single_foot_loss_stops": all(
            max(map(abs, case["mean_applied_vx_by_phase"])) <= 1.0e-6
            and min(case["mean_risk_by_phase"]) >= 0.99
            for case in mujoco_cases
            if case.get("fault_mode") in {"full", "left", "right"}
        ),
        "package_checksums": sha256(SLOT / "exported/policy.onnx")
        == manifest.get("policy_sha256")
        and sha256(SLOT / "exported/hall_risk.onnx")
        == manifest.get("risk_sha256"),
        "cpu_inference_deadline_margin": latency.get("pass") is True,
        "governor_matches_cross_sim": all(
            item in deploy_text
            for item in (
                "low_speed_limit: 0.22",
                "high_speed_limit: 0.60",
                "probability_critical_enter: 0.95",
                "critical_hold_s: 0.04",
                "state_reference_ema_alpha: 0.002",
                "relative_low_rise: 0.20",
                "probe_s: 1.60",
                "probe_speed_limit: 0.50",
            )
        ),
        "package_inactive": manifest.get("default_selected") is False
        and manifest.get("governor_enabled_in_artifact") is False
        and "traction_governor:\n      enabled: false" in deploy_text
        and "hall_traction_r26_r25_candidate" not in controller_text,
    }
    simulation_pass = all(gates.values())
    report = {
        "format": "g1-hall-traction-r26-r25-acceptance-v1",
        "status": (
            "SIMULATION_SAFETY_ADAPTATION_PASS_REAL_CALIBRATION_AND_HARNESS_REQUIRED"
            if simulation_pass
            else "SIMULATION_ACCEPTANCE_FAIL"
        ),
        "simulation_safety_adaptation_pass": simulation_pass,
        "hardware_validated": False,
        "measurement_boundary": (
            "runtime observations are Hall Bx/By/Bz response histories and "
            "proprioception only; no Hall-to-force or Hall-to-friction inverse"
        ),
        "candidate": str(SLOT.resolve()),
        "gates": gates,
        "metrics": {
            "risk_nominal_balanced_accuracy": risk["nominal"]["balanced_accuracy"],
            "risk_randomized_balanced_accuracy": risk["randomized"]["balanced_accuracy"],
            "recovery_active_improvement": recovery["nominal"][
                "active_improvement_fraction"
            ],
            "recovery_randomized_active_improvement": recovery["randomized"][
                "active_improvement_fraction"
            ],
            "cpu_combined_mean_inference_ms": latency["combined_mean_ms"],
            "cpu_combined_p99_bound_ms": latency["policy"]["p99_ms"]
            + latency["risk"]["p99_ms"],
            "isaac_falls": isaac_falls,
            "isaac_high_vx_m_s": high_vx,
            "isaac_low_vx_m_s": low_vx,
            "isaac_seed106_low_slip_r26": low_slip_r26,
            "isaac_seed106_low_slip_r4_control": low_slip_control,
            "mujoco_cases": [case["trajectory"] for case in mujoco_cases],
        },
        "remaining_hardware_gate": (
            "record unloaded and loaded/motion Hall+temperature data on the real "
            "assembled soles, generate left/right normalization, pass live F0M1 "
            "preflight, then complete staged overhead-harness tests"
        ),
    }
    output = CONTINUATION / "final_r26_r25_acceptance.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if simulation_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
