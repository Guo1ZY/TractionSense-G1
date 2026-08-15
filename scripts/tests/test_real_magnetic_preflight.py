from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/home/mosense/guo/scripts")
from check_real_magnetic_preflight import (  # noqa: E402
    Checks,
    DEFAULT_CANDIDATE,
    FINAL_SLOT,
    NORMALIZATION_FORMAT,
    check_config,
    check_candidate,
    check_controller,
    check_health,
    check_packet,
    check_stream,
)


def test_complete_preflight_fixture(tmp_path: Path) -> None:
    normalization = tmp_path / "normalization"
    normalization.mkdir()
    for side in ("left", "right"):
        (normalization / f"{side}.json").write_text(
            json.dumps(
                {
                    "format": NORMALIZATION_FORMAT,
                    "side": side,
                    "baseline_xyz": [[100.0, 200.0, 300.0] for _ in range(15)],
                    "scale_xyz": [[10.0, 10.0, 10.0] for _ in range(15)],
                    "reference_temperature_x10": [250.0 for _ in range(15)],
                    "temperature_coefficient_per_x10": [
                        [0.0, 0.0, 0.0] for _ in range(15)
                    ],
                    "samples": {"baseline": 500, "motion": 500},
                }
            ),
            encoding="utf-8",
        )
    config = tmp_path / "config.magnetic.json"
    config.write_text(
        json.dumps(
            {
                "format": "g1-dual-foot-ble-config-v1",
                "left": {
                    "address": "AA:00",
                    "adapter": "hci0",
                    "normalization": "normalization/left.json",
                },
                "right": {
                    "address": "BB:00",
                    "adapter": "hci1",
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
                "synchronization": {
                    "method": "nearest_host_monotonic",
                    "hardware_timestamp_available": False,
                    "synchronized": True,
                    "max_pair_skew_s": 0.010,
                    "holdback_s": 0.020,
                    "last_pair_skew_s": 0.003,
                    "last_pair_age_s": 0.010,
                    "recent_pair_rate_hz": 50.0,
                    "synchronized_pairs": 100,
                },
            }
        ),
        encoding="utf-8",
    )
    magnetic = [0.2, -0.1, 1.0] * 30
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
    assert any("no force is inferred" in note for note in checks.notes)


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


def test_packaged_candidate_is_inactive_and_checksum_valid() -> None:
    checks = Checks()
    check_candidate(DEFAULT_CANDIDATE, checks)
    assert not checks.failures
    assert any("Hall-to-force inference is absent" in note for note in checks.notes)


def test_stream_monitor_requires_causal_sequence_advancement(tmp_path: Path) -> None:
    packet = tmp_path / "packet.bin"
    magnetic = [0.0] * 90
    stop = threading.Event()

    def publish() -> None:
        sequence = 1
        while not stop.is_set():
            packet.write_bytes(
                struct.pack(
                    "<IIQffffff90f",
                    0x46304D31,
                    sequence,
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
            sequence += 1
            time.sleep(0.01)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        checks = Checks()
        check_stream(packet, 0.15, checks)
    finally:
        stop.set()
        thread.join(timeout=1.0)
    assert not checks.failures
    assert any("packet timing is causal" in note for note in checks.notes)
