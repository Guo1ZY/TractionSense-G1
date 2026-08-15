from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from unitree_rl_lab.traction.layout_magnetic_student import (
    schema_for_trailing_feature_mode,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/traction/package_hall_traction_candidate.py"
SPEC = importlib.util.spec_from_file_location("package_hall_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_deploy_yaml_is_inactive_hall_only_and_turn_capable() -> None:
    template = MODULE.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    value = MODULE.deploy_yaml(template)
    assert "enabled: false" in value
    assert "mode: auto" in value
    assert "low_reprobe_s: 10.00" in value
    assert "probability_low_enter: 0.65" in value
    assert "probability_high_enter: 0.55" in value
    assert "low_hold_s: 0.10" in value
    assert "probe_s: 1.60" in value
    assert "probe_relative_clear_drop: 0.20" in value
    assert "lin_vel_y: [-0.2, 0.2]" in value
    assert "ang_vel_z: [-0.6, 0.6]" in value
    assert "foot_magnetic_array:" in value
    assert value.count("lateral_motion_feedback:") == 1
    assert "foot_sensor_age_lr:" not in value
    assert "lateral_velocity_clip: 1.5" in value
    assert "heading_error_clip: 1.0" in value


def _motion_schema() -> dict:
    return schema_for_trailing_feature_mode("motion_feedback").to_dict()


def _fake_candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    action = tmp_path / "action"
    risk = tmp_path / "risk"
    schema = tmp_path / "schema"
    for directory in (action, risk, schema):
        directory.mkdir()
    (action / "policy.onnx").write_bytes(b"test-policy")
    (risk / "hall_risk.onnx").write_bytes(b"test-risk")
    (action / "training_summary.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (risk / "training_summary.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (schema / "observation_schema.json").write_text(
        json.dumps(_motion_schema()), encoding="utf-8"
    )
    return action, risk, schema


def test_build_never_selects_candidate(tmp_path: Path) -> None:
    action, risk, schema = _fake_candidate_inputs(tmp_path)
    slot = ROOT / "deploy/robots/g1_29dof" / "test-output-placeholder"
    # Only exercise the explicit selection guard without copying large ONNX
    # artifacts into pytest's temporary tree.
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        "policy_dir: test-output-placeholder\n", encoding="utf-8"
    )
    try:
        MODULE.build(
            action,
            risk,
            schema,
            MODULE.DEFAULT_TEMPLATE,
            slot,
            selected,
        )
    except RuntimeError as error:
        assert "currently selected" in str(error)
    else:
        raise AssertionError("selected candidate must be rejected")


def test_build_rejects_missing_action_gate_without_fabricating_pass(tmp_path: Path) -> None:
    action, risk, schema = _fake_candidate_inputs(tmp_path)
    summary = action / "training_summary.json"
    summary.unlink()
    slot = ROOT / "deploy/robots/g1_29dof" / "test-output-placeholder"
    controller = tmp_path / "controller.yaml"
    controller.write_text("policy_dir: config/policy/velocity/v0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="training_summary.json"):
        MODULE.build(
            action,
            risk,
            schema,
            MODULE.DEFAULT_TEMPLATE,
            slot,
            controller,
        )
    assert not summary.exists()
    assert not (action / "spatial_action_training_summary.json").exists()


def test_source_candidates_declare_measurement_boundary() -> None:
    schema = _motion_schema()
    forbidden = set(schema["forbidden_student_inputs"])
    assert {"normal_force", "tangential_force", "ground_friction_mu"} <= forbidden
    MODULE.validate_motion_schema(schema)


def test_packager_rejects_legacy_or_mislabeled_trailing_channels() -> None:
    legacy = _motion_schema()
    legacy["trailing_feature_mode"] = "sensor_age"
    legacy["trailing_feature_names"] = ["sensor_age_left", "sensor_age_right"]
    legacy["slices"].pop("motion_feedback")
    legacy["slices"]["age_lr"] = [1862, 1864]
    with pytest.raises(ValueError, match="motion_feedback"):
        MODULE.validate_motion_schema(legacy)

    mislabeled = _motion_schema()
    mislabeled["slices"]["age_lr"] = [1862, 1864]
    with pytest.raises(ValueError, match="age_lr"):
        MODULE.validate_motion_schema(mislabeled)


def test_validate_motion_schema_rejects_false_foot_aligned_axis_label() -> None:
    schema = _motion_schema()
    schema["hall_frame"] = "foot_local_aligned: +x toe, +y robot-left, +z up"
    with pytest.raises(ValueError, match="per-site Hall-IC local"):
        MODULE.validate_motion_schema(schema)
