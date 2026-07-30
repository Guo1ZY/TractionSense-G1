from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "research_scripts")
)
from check_real_magnetic_preflight import (  # noqa: E402
    Checks,
    FINAL_SLOT,
    PROFILE,
    check_config,
    check_controller,
    check_health,
    check_packet,
)


def test_complete_preflight_fixture(tmp_path: Path) -> None:
    normalization = tmp_path / "normalization"
    normalization.mkdir()
    for side in ("left", "right"):
        (normalization / f"{side}.json").write_text(
            json.dumps({"side": side}), encoding="utf-8"
        )
    config = tmp_path / "config.magnetic.json"
    config.write_text(
        json.dumps(
            {
                "format": "g1-dual-foot-ble-config-v1",
                "left": {
                    "address": "AA:00",
                    "normalization": "normalization/left.json",
                },
                "right": {
                    "address": "BB:00",
                    "normalization": "normalization/right.json",
                },
            }
        ),
        encoding="utf-8",
    )
    feet = {
        side: {
            "connected": True,
            "fresh": True,
            "normalized": True,
            "age_s": 0.01,
            "sample_period_s": 0.02,
            "frames": 100,
            "rejected_frames": 0,
            "last_error": "",
        }
        for side in ("left", "right")
    }
    health = tmp_path / "health.json"
    health.write_text(
        json.dumps(
            {
                "format": "g1-dual-foot-magnetic-health-v1",
                "publishing_f0m1": True,
                "wall_time_ns": time.time_ns(),
                "feet": feet,
            }
        ),
        encoding="utf-8",
    )
    profile_mean = sum(PROFILE) / len(PROFILE)
    magnetic = []
    for _foot in range(2):
        for value in PROFILE:
            gain = value / profile_mean
            magnetic.extend((0.2 * gain, -0.1 * gain, 1.0 * gain))
    packet = tmp_path / "packet.bin"
    packet.write_bytes(
        struct.pack(
            "<IIQffffff90f",
            0x46304D31,
            10,
            time.time_ns(),
            1.0,
            1.0,
            0.01,
            0.01,
            0.02,
            0.02,
            *magnetic,
        )
    )
    controller = tmp_path / "config.yaml"
    controller.write_text(
        f"policy_dir: {FINAL_SLOT}\n",
        encoding="utf-8",
    )

    checks = Checks()
    check_config(config, checks)
    check_health(health, 0.20, 0.10, checks)
    check_packet(packet, 0.20, 0.10, checks)
    check_controller(controller, True, checks)
    assert not checks.failures
    assert any("FAST eligible" in note for note in checks.notes)


def test_placeholder_addresses_fail(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "format": "g1-dual-foot-ble-config-v1",
                "left": {"address": "REPLACE_LEFT", "normalization": ""},
                "right": {"address": "REPLACE_RIGHT", "normalization": ""},
            }
        ),
        encoding="utf-8",
    )
    checks = Checks()
    check_config(config, checks)
    assert checks.failures
