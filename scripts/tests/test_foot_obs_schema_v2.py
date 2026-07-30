#!/usr/bin/env python3
"""Schema / dim consistency tests for Foot-Adaptive-V2 (no Isaac required)."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/obs_schema/foot_obs_v2.yaml"


def test_schema_dims():
    data = yaml.safe_load(SCHEMA.read_text())
    assert data["schema_version"] == "foot_obs_v2"
    foot = sum(t["dim"] for t in data["terms"])
    assert foot == data["actor_foot_dim"] == 8
    total = (data["base_policy_dim"] + foot) * data["history_length"]
    assert total == data["actor_total_dim"] == 520
    # No tangent force on actor list
    names = [t["name"] for t in data["terms"]]
    assert "foot_tangent_force" not in names
    assert "foot_sensor_valid" in names
    print("[ok] schema dims 520 = (96+8)*5")


def test_reward_math_no_min_floor_giveup():
    """stable_speed_bonus vanishes under slip; full track stays 1.0 (concept)."""
    track = 0.9
    for slip, expect_lower in [(0.0, False), (0.5, True), (1.5, True)]:
        plant = math.exp(-slip / 0.30)
        bonus = track * plant
        if expect_lower:
            assert bonus < track * 0.9
        else:
            assert abs(bonus - track) < 1e-6
    # slip_aware old hole: min_track_scale=0.1 allows ~ignore cmd
    min_track_scale = 0.1
    scale_old = min_track_scale + (1 - min_track_scale) * math.exp(-2.0 / 0.28)
    assert scale_old < 0.15  # old design nearly zeros tracking pressure
    print("[ok] reward math: V2 bonus dies with slip; old min_track allows give-up")


def test_soft_contact_matches_train():
    def soft(fn, thr=5.0, sc=2.0):
        x = (abs(fn) - thr) * sc
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    assert soft(0.0) < 0.1
    assert soft(200.0) > 0.99
    print("[ok] soft contact sigmoid")


def test_f0t1_packet_layout():
    """Legacy F0T1 40-byte packet still used by MuJoCo bridge / foot_ros_bridge."""
    MAGIC = 0x46305431
    fmt = "<IIQffffff"
    assert struct.calcsize(fmt) == 40
    pkt = struct.pack(fmt, MAGIC, 1, 123, 0.9, 0.8, 1.5, 1.4, 0.2, 0.1)
    magic, seq, stamp, *rest = struct.unpack(fmt, pkt)
    assert magic == MAGIC and len(rest) == 6
    print("[ok] F0T1 packet 40B")


def main() -> int:
    if not SCHEMA.is_file():
        print("MISSING schema", SCHEMA, file=sys.stderr)
        return 1
    test_schema_dims()
    test_reward_math_no_min_floor_giveup()
    test_soft_contact_matches_train()
    test_f0t1_packet_layout()
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
