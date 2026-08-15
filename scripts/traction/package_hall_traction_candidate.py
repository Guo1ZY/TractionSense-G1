#!/usr/bin/env python3
"""Build a reproducible, inactive Hall-only G1 deployment candidate.

The packager copies an already selected locomotion actor and an independent
causal Hall-risk model into a new policy slot.  It never edits the controller's
active ``config.yaml`` and writes ``enabled: false`` for the governor.  Real
motion therefore still requires two explicit operator actions: select this
slot and enable the governor after the live Hall preflight passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION = (
    ROOT
    / "artifacts/hall_traction_continuation/"
    "hall_recovery_round26_r25_features"
)
DEFAULT_RISK = (
    ROOT
    / "artifacts/hall_traction_continuation/"
    "hall_risk_round25_active_crosssim"
)
DEFAULT_SCHEMA = (
    ROOT
    / "artifacts/hall_traction_execution/"
    "layout_magnetic_physical_round4_low_speed_observability"
)
DEFAULT_TEMPLATE = (
    ROOT
    / "deploy/robots/g1_29dof/config/policy/velocity/"
    "layout_magnetic_v2/params/deploy.yaml"
)
DEFAULT_SLOT = (
    ROOT
    / "deploy/robots/g1_29dof/config/policy/velocity/"
    "hall_traction_r26_r25_candidate"
)
CONTROLLER_CONFIG = ROOT / "deploy/robots/g1_29dof/config/config.yaml"

MOTION_TRAILING_FEATURE_MODE = "motion_feedback"
MOTION_TRAILING_FEATURE_NAMES = [
    "body_lateral_velocity",
    "relative_heading_error",
]
HALL_CHANNEL_FRAME = (
    "per_site_hall_ic_local_xyz; P00..P14 mounting yaw is explicit and "
    "right-foot mirror sign is part of the source convention"
)
_SENSOR_AGE_YAML_BLOCK = """  foot_sensor_age_lr:
    params: {}
    clip: [0.0, 1.0]
    scale: null
    history_length: 1
"""
_MOTION_FEEDBACK_YAML_BLOCK = """  lateral_motion_feedback:
    params: {asset_name: robot, lateral_velocity_clip: 1.5, heading_error_clip: 1.0}
    clip: [-1.5, 1.5]
    scale: null
    history_length: 1
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def deploy_yaml(template: str) -> str:
    """Build an inactive Motion-Hall deployment YAML.

    Spatial Hall actors are trained with ``[body_vy, relative_heading]`` in
    columns 1862:1864.  The historical Hall template used packet age in those
    columns.  Silently preserving that term produces a valid 1864-vector with
    the wrong meaning, so require and replace the exact block here.
    """
    old = """    ranges:
      lin_vel_x: [-0.3, 1.0]
      lin_vel_y: [0.0, 0.0]
      ang_vel_z: [0.0, 0.0]
      heading: null
"""
    new = """    # Simulation-verified candidate envelope.  The Hall governor is
    # deliberately OFF in the artifact and must be enabled only after live
    # F0M1 preflight.  Hall channels are magnetic Bx/By/Bz response only.
    ranges:
      lin_vel_x: [-0.3, 0.6]
      lin_vel_y: [-0.2, 0.2]
      ang_vel_z: [-0.6, 0.6]
      heading: null
    traction_governor:
      enabled: false
      mode: auto
      lock_lateral_yaw: false
      low_speed_limit: 0.22
      high_speed_limit: 0.60
      critical_speed_limit: 0.0
      low_lateral_limit: 0.05
      high_lateral_limit: 0.35
      low_yaw_limit: 0.15
      high_yaw_limit: 0.80
      accel_rate: 1.50
      decel_rate: 1.00
      probability_low_enter: 0.65
      probability_high_enter: 0.55
      probability_critical_enter: 0.95
      critical_hold_s: 0.04
      probability_ema_alpha: 0.20
      state_reference_ema_alpha: 0.002
      relative_low_rise: 0.20
      relative_high_drop: 0.20
      low_hold_s: 0.10
      high_hold_s: 0.10
      feedback_timeout_s: 0.25
      min_detection_command: 0.20
      startup_command_threshold: 0.02
      warmup_s: 0.20
      probe_s: 1.60
      probe_speed_limit: 0.50
      low_reprobe_s: 10.00
      probe_relative_clear_drop: 0.20
      crawl_pulse_s: 0.45
      crawl_min_hold_s: 0.25
      launch_accel_rate: 1.50
      tracking_low_enter: 0.55
      tracking_high_enter: 0.30
"""
    if old not in template:
        raise ValueError("deployment template command block changed")
    result = template.replace(old, new, 1)
    if "foot_magnetic_array:" not in result:
        raise ValueError("deployment template is not the 1864-D Hall schema")
    if result.count(_SENSOR_AGE_YAML_BLOCK) != 1:
        raise ValueError(
            "deployment template must contain exactly one canonical "
            "foot_sensor_age_lr block"
        )
    result = result.replace(
        _SENSOR_AGE_YAML_BLOCK, _MOTION_FEEDBACK_YAML_BLOCK, 1
    )
    if "foot_sensor_age_lr:" in result:
        raise ValueError("Motion-Hall deployment must not retain packet-age actor slots")
    if result.count("lateral_motion_feedback:") != 1:
        raise ValueError("Motion-Hall deployment requires one lateral_motion_feedback term")
    return result


def validate_motion_schema(schema: dict) -> None:
    """Reject dimension-correct but semantically incompatible actor schemas."""

    if schema.get("input_dimension") != 1864 or schema.get("output_dimension") != 29:
        raise ValueError("candidate must use the exact 1864 -> 29 Hall policy schema")
    if schema.get("trailing_feature_mode") != MOTION_TRAILING_FEATURE_MODE:
        raise ValueError(
            "candidate schema must explicitly declare trailing_feature_mode="
            "motion_feedback; missing/legacy packet-age schemas are incompatible"
        )
    if schema.get("trailing_feature_names") != MOTION_TRAILING_FEATURE_NAMES:
        raise ValueError(
            "candidate schema columns 1862:1864 must be "
            "[body_lateral_velocity, relative_heading_error]"
        )
    if schema.get("hall_frame") != HALL_CHANNEL_FRAME:
        raise ValueError(
            "candidate Hall axes must explicitly use the per-site Hall-IC local "
            "frame; a foot-local label is incompatible with the simulated and "
            "raw BLE Bx/By/Bz channels"
        )
    if len(schema.get("hall_axis_yaw_deg", ())) != 15:
        raise ValueError("candidate schema must declare all 15 Hall-site yaw angles")
    if schema.get("right_foot_axis_sign") != [1.0, -1.0, 1.0]:
        raise ValueError("candidate schema has the wrong right-foot Hall axis sign")
    slices = schema.get("slices")
    if not isinstance(slices, dict):
        raise ValueError("candidate schema must declare observation slices")
    expected = {
        "proprio": [0, 480],
        "magnetic_history": [480, 1830],
        "sample_period_history": [1830, 1860],
        "valid_lr": [1860, 1862],
        "trailing_features": [1862, 1864],
        "motion_feedback": [1862, 1864],
    }
    for name, bounds in expected.items():
        if slices.get(name) != bounds:
            raise ValueError(
                f"candidate schema slice {name!r} must be {bounds}, got "
                f"{slices.get(name)!r}"
            )
    if "age_lr" in slices:
        raise ValueError("motion_feedback schema must not label columns 1862:1864 as age_lr")


def build(
    action_dir: Path,
    risk_dir: Path,
    schema_dir: Path,
    template: Path,
    slot: Path,
    controller_config: Path,
) -> dict:
    action_model = action_dir / "policy.onnx"
    # Newer risk training names the artifact by its measured quantity.  Keep
    # the deployment filename stable while accepting the causal slip-risk
    # exporter used by the current Hall pipeline.
    risk_model = risk_dir / "hall_risk.onnx"
    if not risk_model.is_file():
        risk_model = risk_dir / "hall_slip_risk.onnx"
    schema_path = schema_dir / "observation_schema.json"
    action_summary_path = action_dir / "training_summary.json"
    risk_summary_path = risk_dir / "training_summary.json"
    # Never manufacture an acceptance summary from a checkpoint directory.
    # A missing gate report is a missing required input, not an implicit PASS.
    # This keeps packaging read-only with respect to all source artifacts and
    # prevents an unevaluated spatial checkpoint from reaching a robot slot.
    required = (
        action_model,
        risk_model,
        schema_path,
        action_summary_path,
        risk_summary_path,
        template,
        controller_config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing candidate inputs: " + ", ".join(missing))

    schema = load_json(schema_path)
    action_summary = load_json(action_summary_path)
    risk_summary = load_json(risk_summary_path)
    validate_motion_schema(schema)
    action_pass = action_summary.get("overall", action_summary.get("status")) == "PASS"
    risk_pass = risk_summary.get("overall", risk_summary.get("status")) == "PASS"
    if not action_pass or not risk_pass:
        raise ValueError("both action and risk candidates must pass offline gates")
    forbidden = set(schema.get("forbidden_student_inputs", ()))
    if not {"normal_force", "tangential_force", "ground_friction_mu"} <= forbidden:
        raise ValueError("observation schema does not declare the Hall-only boundary")

    relative_slot = slot.resolve().relative_to(
        (ROOT / "deploy/robots/g1_29dof").resolve()
    ).as_posix()
    active_text = controller_config.read_text(encoding="utf-8")
    if f"policy_dir: {relative_slot}" in active_text:
        raise RuntimeError("refusing to overwrite a currently selected candidate slot")

    exported = slot / "exported"
    metadata = slot / "metadata"
    params = slot / "params"
    for directory in (exported, metadata, params):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(action_model, exported / "policy.onnx")
    shutil.copy2(risk_model, exported / "hall_risk.onnx")
    shutil.copy2(schema_path, metadata / "observation_schema.json")
    shutil.copy2(action_summary_path, metadata / "action_training_summary.json")
    shutil.copy2(risk_summary_path, metadata / "risk_training_summary.json")
    (params / "deploy.yaml").write_text(
        deploy_yaml(template.read_text(encoding="utf-8")), encoding="utf-8"
    )

    manifest = {
        "format": "g1-hall-traction-candidate-v2",
        "status": "simulation_safety_adaptation_pass_performance_target_partial_real_harness_pending",
        "default_selected": False,
        "governor_enabled_in_artifact": False,
        "student_uses_force": False,
        "student_uses_ground_friction": False,
        "measurement": "dual-foot normalized Hall Bx/By/Bz response and proprioception",
        "mechanics": "TPU local deformation with four embedded magnetic inclusions per Hall site",
        "policy_input_shape": [1, 1864],
        "policy_output_shape": [1, 29],
        "trailing_feature_mode": MOTION_TRAILING_FEATURE_MODE,
        "trailing_feature_names": MOTION_TRAILING_FEATURE_NAMES,
        "risk_input_shape": [1, 1864],
        "risk_output_shape": [1, 1],
        "policy_sha256": sha256(exported / "policy.onnx"),
        "risk_sha256": sha256(exported / "hall_risk.onnx"),
        "action_checkpoint_sha256": action_summary.get(
            "checkpoint_sha256", action_summary.get("artifacts", {}).get("checkpoint_sha256")
        ),
        "risk_checkpoint_sha256": risk_summary.get(
            "checkpoint_sha256", risk_summary.get("artifacts", {}).get("checkpoint_sha256")
        ),
        "action_policy_type": "risk_gated_hall_recovery_r26",
        "risk_model_variant": risk_summary.get("model_variant", "layout_encoder"),
        "activation_gates": [
            "complete real unloaded/motion normalization for both feet",
            "pass live F0M1 stream preflight",
            "select G1_POLICY_DIR explicitly",
            "set G1_TRACTION_GOVERNOR=1 and G1_TRACTION_MODE=auto explicitly",
            "perform overhead-harness acceptance before free walking",
        ],
        "rollback_policy": "unset G1_POLICY_DIR and G1_TRACTION_GOVERNOR; controller config is unchanged",
    }
    (metadata / "install_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (slot / "README.md").write_text(
        (
            "# Hall-only G1 traction candidate (r26 recovery + r25 risk)\n\n"
            "This slot is intentionally inactive. Hall inputs are only normalized "
            "Bx/By/Bz response histories; neither model accepts force, contact truth "
            "or friction truth. Four magnetic inclusions per Hall site are embedded "
            "inside TPU in the simulation forward model. Simulation safety and "
            "friction adaptation passed, while the ambitious high-speed tracking/"
            "one-second recovery target remains partial.\n\n"
            "Do not edit the global controller configuration for first bring-up. "
            "Use the guarded harness launcher and complete its live preflight.\n"
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--risk-dir", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--slot", type=Path, default=DEFAULT_SLOT)
    parser.add_argument("--controller-config", type=Path, default=CONTROLLER_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build(
            args.action_dir,
            args.risk_dir,
            args.schema_dir,
            args.template,
            args.slot,
            args.controller_config,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
