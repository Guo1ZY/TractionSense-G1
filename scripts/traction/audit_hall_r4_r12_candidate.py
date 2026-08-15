#!/usr/bin/env python3
"""Audit the frozen r4 actor + r12 Hall-risk simulation candidate.

PASS means that the reproducible simulation, model-boundary, packaging and
fail-safe gates passed.  It never means that hardware validation passed; the
report always retains an explicit real-harness gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLOT = (
    ROOT
    / "deploy/robots/g1_29dof/config/policy/velocity/"
    "hall_traction_r4_r12_candidate"
)
ACTION_SUMMARY = SLOT / "metadata/action_training_summary.json"
RISK_SUMMARY = SLOT / "metadata/risk_training_summary.json"
PLATEN = (
    ROOT
    / "artifacts/hall_traction_execution/final_platen_validation.summary.json"
)
ISAAC_SWITCH = (
    ROOT
    / "artifacts/hall_traction_execution/"
    "isaac_r12_verified_envelope_v06_mu020_seed70.csv"
)
ISAAC_LOW_SPEED = (
    ROOT
    / "artifacts/hall_traction_execution/"
    "isaac_r12_low_speed_zero_v005_seed71.csv"
)
MUJOCO_ROOT = ROOT.parent / "unitree_mujoco/artifacts/hall_magnetic_crosssim"
MUJOCO_CASES = (
    "r11_reprobe2500_randomized_switch_seed20260830.summary.json",
    "r12_randomized_switch_seed20260835.summary.json",
    "r12_turn_randomized_switch_seed20260836.summary.json",
    "r12_full_fault_seed20260837.summary.json",
)
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
        raise ValueError(f"non-finite {key}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit(slot: Path = SLOT) -> dict:
    manifest = load_json(slot / "metadata/install_manifest.json")
    schema = load_json(slot / "metadata/observation_schema.json")
    action = load_json(ACTION_SUMMARY if slot == SLOT else slot / "metadata/action_training_summary.json")
    risk = load_json(RISK_SUMMARY if slot == SLOT else slot / "metadata/risk_training_summary.json")
    platen = load_json(PLATEN)
    switch = load_csv(ISAAC_SWITCH)
    low_speed = load_csv(ISAAC_LOW_SPEED)
    policy = slot / "exported/policy.onnx"
    risk_model = slot / "exported/hall_risk.onnx"
    deploy_text = (slot / "params/deploy.yaml").read_text(encoding="utf-8")

    switch_falls = sum(int(number(row, "falls")) for row in switch)
    low_phase = min(switch, key=lambda row: number(row, "mu"))
    final_phase = switch[-1]
    zero = min(low_speed, key=lambda row: abs(number(row, "cmd_vx")))
    micro = min(low_speed, key=lambda row: abs(number(row, "cmd_vx") - 0.05))
    mujoco = [load_json(MUJOCO_ROOT / name) for name in MUJOCO_CASES]
    forbidden = set(schema.get("forbidden_student_inputs", ()))

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
        "isaac_switch_no_falls": switch_falls == 0,
        "isaac_low_mu_limited": number(low_phase, "steady_vx") <= 0.25,
        "isaac_high_mu_recovers": number(final_phase, "steady_vx")
        >= number(low_phase, "steady_vx") + 0.05,
        "exact_zero_stays_zero": abs(number(zero, "mean_applied_vx_command"))
        <= 1.0e-6,
        "microstep_launches_without_fall": number(micro, "mean_vx") >= 0.03
        and number(micro, "fall_per_env") == 0.0,
        "mujoco_selected_cases_pass": all(
            item.get("status") == "PASS"
            and item.get("fell") is False
            and item.get("nonfinite") == 0
            for item in mujoco
        ),
        "full_hall_loss_stops": (
            mujoco[-1]["fault_mode"] == "full"
            and max(map(abs, mujoco[-1]["mean_applied_vx_by_phase"])) <= 1.0e-6
            and min(mujoco[-1]["mean_risk_by_phase"]) >= 0.999999
        ),
        "package_checksums": sha256(policy) == manifest.get("policy_sha256")
        and sha256(risk_model) == manifest.get("risk_sha256"),
        "package_inactive": manifest.get("default_selected") is False
        and manifest.get("governor_enabled_in_artifact") is False
        and "traction_governor:\n      enabled: false" in deploy_text
        and "hall_traction_r4_r12_candidate" not in CONTROLLER_CONFIG.read_text(
            encoding="utf-8"
        ),
    }
    simulation_pass = all(gates.values())
    return {
        "format": "g1-hall-traction-r4-r12-acceptance-v1",
        "status": (
            "SIMULATION_ACCEPTANCE_PASS_REAL_HARNESS_REQUIRED"
            if simulation_pass
            else "SIMULATION_ACCEPTANCE_FAIL"
        ),
        "simulation_pass": simulation_pass,
        "hardware_validated": False,
        "measurement_boundary": (
            "runtime input is Hall Bx/By/Bz response + proprioceptive history; "
            "contact force/friction exist only as simulator mechanics or offline labels"
        ),
        "mechanical_model": (
            "one local TPU deformation state per Hall site; four magnets are "
            "embedded inclusions that share that material motion"
        ),
        "candidate": str(slot.resolve()),
        "gates": gates,
        "metrics": {
            "risk_nominal_balanced_accuracy": risk["nominal"]["balanced_accuracy"],
            "risk_randomized_balanced_accuracy": risk["randomized"]["balanced_accuracy"],
            "risk_onnx_parity_max_abs": risk["onnx_parity_max_abs"],
            "platen_unload_max_abs_t": platen["unload_max_abs_T"],
            "isaac_switch_falls": switch_falls,
            "isaac_low_mu_vx_m_s": number(low_phase, "steady_vx"),
            "isaac_recovered_vx_m_s": number(final_phase, "steady_vx"),
            "isaac_zero_applied_vx_m_s": number(zero, "mean_applied_vx_command"),
            "isaac_microstep_vx_m_s": number(micro, "mean_vx"),
            "mujoco_cases": [item["trajectory"] for item in mujoco],
        },
        "remaining_gate": (
            "Collect real dual-foot Hall/temperature normalization data, pass "
            "live preflight, then complete overhead-harness staged acceptance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=Path, default=SLOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts/hall_traction_execution/final_r4_r12_acceptance.json",
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
    return 0 if report["simulation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
