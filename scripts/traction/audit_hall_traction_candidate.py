#!/usr/bin/env python3
"""Apply reproducible Hall-only deployment and MuJoCo acceptance gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "source/unitree_rl_lab/config/hall_traction_acceptance.json"
DEFAULT_SLOT = ROOT / "deploy/robots/g1_29dof/config/policy/velocity/layout_magnetic_v2"
DEFAULT_STEADY = (
    ROOT
    / "logs/evaluations/mujoco_layout_magnetic_v2"
    / "final_full_v6_consolidated_20260806/matrix.csv"
)
DEFAULT_SWITCH = (
    ROOT
    / "logs/evaluations/mujoco_layout_magnetic_v2"
    / "final_switch_v9_20260806/switch_phases.csv"
)
DEFAULT_MUJOCO = ROOT.parent / "unitree_mujoco/simulate/src/main.cc"
DEFAULT_MATRIX_RUNNER = ROOT.parent / "scripts/mujoco_friction_speed_matrix.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def number(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}={row[name]!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_steady(rows: list[dict[str, str]], cfg: dict[str, Any]) -> dict[str, Any]:
    falls = sum(int(number(row, "fall")) for row in rows)
    stable_fraction = sum(row["stable"].strip().upper() == "PASS" for row in rows) / len(rows)
    minimum_valid = min(number(row, "sensor_valid") for row in rows)
    selected = [
        row
        for row in rows
        if number(row, "mu") >= float(cfg["high_friction_threshold"])
        and abs(number(row, "cmd_vx")) >= float(cfg["tracking_command_threshold_m_s"])
    ]
    relative_errors = [
        abs(number(row, "mean_vx") - number(row, "cmd_vx"))
        / max(abs(number(row, "cmd_vx")), 1.0e-6)
        for row in selected
    ]
    max_abs_vy = max((number(row, "mean_abs_vy") for row in selected), default=math.inf)
    gates = {
        "fall_count": falls <= int(cfg["maximum_falls"]),
        "stable_fraction": stable_fraction >= float(cfg["minimum_stable_fraction"]),
        "sensor_valid": minimum_valid >= float(cfg["minimum_sensor_valid"]),
        "high_friction_tracking": bool(selected)
        and max(relative_errors) <= float(cfg["maximum_high_friction_relative_vx_error"]),
        "high_friction_lateral_velocity": bool(selected)
        and max_abs_vy <= float(cfg["maximum_high_friction_abs_vy_m_s"]),
    }
    return {
        "rows": len(rows),
        "falls": falls,
        "stable_fraction": stable_fraction,
        "minimum_sensor_valid": minimum_valid,
        "high_friction_cases": len(selected),
        "maximum_high_friction_relative_vx_error": max(relative_errors, default=math.inf),
        "maximum_high_friction_abs_vy_m_s": max_abs_vy,
        "gates": gates,
        "pass": all(gates.values()),
    }


def check_switch(rows: list[dict[str, str]], cfg: dict[str, Any]) -> dict[str, Any]:
    falls = sum(int(number(row, "fall")) for row in rows)
    stable_fraction = sum(row["stable"].strip().upper() == "PASS" for row in rows) / len(rows)
    response = [number(row, "response_time_s") for row in rows if row["response_time_s"].lower() != "nan"]
    gates = {
        "fall_count": falls <= int(cfg["maximum_falls"]),
        "stable_fraction": stable_fraction >= float(cfg["minimum_stable_fraction"]),
        "response_time": len(response) == len(rows)
        and max(response) <= float(cfg["maximum_response_time_s"]),
    }
    return {
        "rows": len(rows),
        "falls": falls,
        "stable_fraction": stable_fraction,
        "maximum_response_time_s": max(response, default=math.inf),
        "gates": gates,
        "pass": all(gates.values()),
    }


def check_slot(slot: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    schema = load_json(slot / "metadata/observation_schema.json")
    manifest = load_json(slot / "metadata/install_manifest.json")
    training = load_json(slot / "metadata/training_summary.json")
    onnx = slot / "exported/policy.onnx"
    forbidden = set(cfg["measurement_boundary"]["forbidden_student_inputs"])
    declared_forbidden = set(schema.get("forbidden_student_inputs", ()))
    gates = {
        "input_dimension": int(schema.get("input_dimension", -1))
        == int(cfg["deployment"]["input_dimension"]),
        "output_dimension": int(schema.get("output_dimension", -1))
        == int(cfg["deployment"]["output_dimension"]),
        "forbidden_schema_complete": forbidden <= declared_forbidden,
        "manifest_hall_only": manifest.get("student_uses_force") is False
        and manifest.get("student_uses_ground_friction") is False,
        "onnx_exists": onnx.is_file(),
        "onnx_parity": float(training.get("fallback_max_abs", math.inf))
        <= float(cfg["deployment"]["sensor_loss_fallback_max_abs"])
        and float(training.get("onnx_parity_max_abs", math.inf))
        <= float(cfg["deployment"]["onnx_parity_max_abs"]),
    }
    return {
        "slot": str(slot),
        "onnx_sha256": sha256(onnx) if onnx.is_file() else None,
        "gates": gates,
        "pass": all(gates.values()),
    }


def check_source(paths: list[Path], patterns: list[str]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text:
                findings.append({"path": str(path), "pattern": pattern})
    return {"findings": findings, "pass": not findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--slot", type=Path, default=DEFAULT_SLOT)
    parser.add_argument("--steady", type=Path, default=DEFAULT_STEADY)
    parser.add_argument("--switch", type=Path, default=DEFAULT_SWITCH)
    parser.add_argument("--mujoco-source", type=Path, default=DEFAULT_MUJOCO)
    parser.add_argument("--matrix-runner", type=Path, default=DEFAULT_MATRIX_RUNNER)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if cfg.get("format") != "g1-hall-traction-acceptance-v1":
        raise ValueError("unsupported acceptance config")
    report = {
        "format": "g1-hall-traction-audit-v1",
        "config": str(args.config.resolve()),
        "steady": check_steady(load_csv(args.steady), cfg["steady"]),
        "switch": check_switch(load_csv(args.switch), cfg["switch"]),
        "deployment": check_slot(args.slot, cfg),
        "source_boundary": check_source(
            [args.mujoco_source, args.matrix_runner],
            list(cfg["mujoco_source_forbidden_patterns"]),
        ),
    }
    report["pass"] = all(
        report[name]["pass"]
        for name in ("steady", "switch", "deployment", "source_boundary")
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
