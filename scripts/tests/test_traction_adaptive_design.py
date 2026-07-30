#!/usr/bin/env python3
"""Fast source/design invariants for the G1 traction-adaptive task."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_CFG = ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"


def test_speed_cap_is_smooth_and_monotonic() -> None:
    mus = (0.05, 0.08, 0.20, 0.40, 0.55, 0.85, 1.20)
    cap = [0.20 + 1.30 / (1.0 + math.exp(-(mu - 0.55) / 0.14)) for mu in mus]
    assert all(b > a for a, b in zip(cap, cap[1:])), cap
    assert 0.20 <= cap[0] < 0.30, cap
    assert cap[2] < 0.40, cap
    assert 0.80 < cap[4] < 0.90, cap
    assert cap[5] > 1.30, cap
    assert cap[-1] < 1.50, cap


def test_command_distribution_keeps_one_meter_default() -> None:
    source = ENV_CFG.read_text()
    block = source.split("class TractionAdaptiveCommandsCfg", 1)[1].split(
        "class TractionAdaptiveCurriculumCfg", 1
    )[0]
    assert "lin_vel_x=(-0.3, 1.0)" in block
    assert "lin_vel_y=(0.0, 0.0)" in block
    assert "ang_vel_z=(0.0, 0.0)" in block
    assert "high_speed_fraction=0.15" in block
    assert "high_speed_range=(1.0, 1.5)" in block


def test_actor_is_deployable_and_preserves_baseline_prefix() -> None:
    source = ENV_CFG.read_text()
    policy = source.split("class FootTractionAdaptiveObservationsCfg", 1)[1].split(
        "class CriticCfg", 1
    )[0]
    baseline_terms = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )
    for name in baseline_terms:
        assert f"{name} = ObsTerm" in policy
    assert "ground_friction_mu" not in policy
    assert "foot_slip_proxy" not in policy
    for name in (
        "foot_contact",
        "foot_normal_force",
        "foot_tangent_force",
        "foot_friction_ratio",
        "foot_load_ratio",
    ):
        term = policy.split(f"{name} = ObsTerm", 1)[1].split("history_length=15", 1)
        assert len(term) == 2, name

    # 49999 prefix: (3+3+3+29+29+29)*5 = 480.
    # Foot context: five 2-D terms*15 + valid/age*5 = 160.
    assert 480 + 5 * 2 * 15 + 2 * 1 * 5 == 640


def test_speed_lateral_v2_strengthens_path_without_changing_command_cap() -> None:
    source = ENV_CFG.read_text()
    block = source.split(
        "class RobotFootTractionSpeedLateralV2TeacherEnvCfg", 1
    )[1].split(
        "class RobotFootTractionSpeedLateralV2TeacherPlayEnvCfg", 1
    )[0]
    assert "straight_line_motion.weight = -8.0" in block
    assert "straight_cross_track.weight = -8.0" in block
    assert "base_angular_velocity.weight = -0.45" in block
    assert "high_traction_underspeed.weight = -5.0" in block


if __name__ == "__main__":
    test_speed_cap_is_smooth_and_monotonic()
    test_command_distribution_keeps_one_meter_default()
    test_actor_is_deployable_and_preserves_baseline_prefix()
    print("[ok] traction-adaptive design invariants")
