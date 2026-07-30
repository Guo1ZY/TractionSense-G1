#!/usr/bin/env python3
"""Source-level invariants for the final dual-foot magnetic observation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
OBSERVATIONS = (
    ROOT / "deploy/include/isaaclab/envs/mdp/observations/observations.h"
)


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
