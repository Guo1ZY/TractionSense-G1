#!/usr/bin/env python3
"""Source-level invariants for the final dual-foot magnetic observation."""

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
OBSERVATIONS = (
    ROOT / "deploy/include/isaaclab/envs/mdp/observations/observations.h"
)
EXPORTER = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "utils"
    / "export_deploy_cfg.py"
)


def _load_format_value():
    tree = ast.parse(EXPORTER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "format_value"
    )
    namespace = {"math": __import__("math"), "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(EXPORTER), "exec"), namespace)
    return namespace["format_value"]


def test_policy_layout_is_exactly_1864_without_privileged_mu() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    block = source.split("class FootTractionMagneticObservationsCfg", 1)[1].split(
        "class RobotFootTractionMagneticStudentEnvCfg", 1
    )[0]
    assert "ground_friction_mu" not in block.split("class CriticCfg", 1)[0]
    assert "foot_magnetic_array = ObsTerm" in block
    assert "history_length=15" in block.split(
        "foot_magnetic_array = ObsTerm", 1
    )[1].split("foot_sample_period_lr", 1)[0]
    assert "history_length=15" in block.split(
        "foot_sample_period_lr = ObsTerm", 1
    )[1].split("foot_sensor_valid_lr", 1)[0]
    assert 480 + 15 * 2 * 15 * 3 + 15 * 2 + 4 == 1864


def test_cpp_deploy_registers_all_final_sensor_terms() -> None:
    source = OBSERVATIONS.read_text(encoding="utf-8")
    for term in (
        "foot_magnetic_array",
        "foot_sample_period_lr",
        "foot_sensor_valid_lr",
        "foot_sensor_age_lr",
    ):
        assert f"REGISTER_OBSERVATION({term})" in source
    assert "REGISTER_OBSERVATION(lateral_motion_feedback)" in source


def test_motion_task_replaces_age_slots_with_motion_feedback() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    block = source.split("class FootTractionMagneticMotionObservationsCfg", 1)[1].split(
        "class RobotFootTractionMagneticMotionStudentEnvCfg", 1
    )[0]
    assert "func=mdp.lateral_motion_feedback" in block
    assert '"lateral_velocity_clip": 1.5' in block
    assert '"heading_error_clip": 1.0' in block
    # The inherited attribute name is kept only to preserve the final term's
    # position; its callable must never remain the Hall packet-age function.
    assert "func=mdp.hall_sensor_age_lr" not in block


def test_generic_deploy_export_uses_motion_callable_semantics_not_legacy_alias() -> None:
    exporter = EXPORTER.read_text(encoding="utf-8")
    assert "def deploy_observation_name(" in exporter
    assert 'config_name == "foot_sensor_age_lr"' in exporter
    assert 'function_name == "lateral_motion_feedback"' in exporter
    assert 'return "lateral_motion_feedback"' in exporter
    assert "deploy_name = deploy_observation_name(obs_name, obs_cfg.func)" in exporter
    assert 'cfg["observations"][deploy_name] = term_cfg' in exporter


def test_deploy_value_tree_is_safe_yaml_without_python_tags() -> None:
    format_value = _load_format_value()
    value = format_value(
        {
            "tuple": (1.0, np.float32(2.0)),
            "array": np.asarray([3, 4], dtype=np.int64),
            "all_indices": slice(None),
            "nested": {"enabled": True, "missing": None},
        }
    )
    dumped = yaml.safe_dump(value, sort_keys=False)
    assert "!!python" not in dumped
    assert yaml.safe_load(dumped) == {
        "tuple": [1.0, 2.0],
        "array": [3, 4],
        "all_indices": [],
        "nested": {"enabled": True, "missing": None},
    }


def test_partial_slice_fails_closed_in_deploy_export() -> None:
    format_value = _load_format_value()
    with pytest.raises(ValueError, match="partial slice selectors"):
        format_value({"joint_ids": slice(1, 5, 2)})


def test_nonfinite_deploy_numbers_fail_closed() -> None:
    format_value = _load_format_value()
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite numeric value"):
            format_value({"gain": value})


def test_exporter_uses_safe_dump() -> None:
    exporter = EXPORTER.read_text(encoding="utf-8")
    assert "yaml.safe_dump(cfg" in exporter
    assert "yaml.dump(cfg" not in exporter
