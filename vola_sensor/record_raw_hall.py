#!/usr/bin/env python3
"""Record FootSensor15 BLE frames as raw Hall/temperature NPZ.

This tool intentionally does not emit force: the current project contains no
measured Hall[15,3] -> net Fx/Fy/Fz calibration model.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np


TRACTION_SOURCE = Path(
    "/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab"
)
BLE_PROTOCOL_SOURCE = TRACTION_SOURCE / "unitree_rl_lab" / "traction" / "ble.py"


def _load_ble_protocol():
    """Load the standalone wire parser without importing the training stack.

    Importing ``unitree_rl_lab.traction`` also imports Torch-based training
    diagnostics.  Raw BLE capture must remain usable in the lightweight sensor
    virtual environment, where Torch is intentionally not installed.
    """
    module_name = "_footsensor15_ble_protocol"
    spec = importlib.util.spec_from_file_location(module_name, BLE_PROTOCOL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load BLE protocol from {BLE_PROTOCOL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ble_protocol = _load_ble_protocol()
BleFrameParser = _ble_protocol.BleFrameParser
DEVICE_NAME = _ble_protocol.DEVICE_NAME
FRAME_LENGTH = _ble_protocol.FRAME_LENGTH
NOTIFY_CHARACTERISTIC_UUID = _ble_protocol.NOTIFY_CHARACTERISTIC_UUID


class RawHallRecorder:
    def __init__(
        self,
        foot_id: str,
        *,
        source_address: str | None = None,
        adapter: str | None = None,
    ) -> None:
        self.parser = BleFrameParser(foot_id=foot_id)
        self.source_address = source_address or ""
        self.adapter = adapter or ""
        self.timestamp: list[float] = []
        self.sequence: list[int] = []
        self.header_byte_1_u8: list[int] = []
        self.hall_xyz: list[np.ndarray] = []
        self.temperature_c: list[np.ndarray] = []
        self.valid: list[bool] = []

    def feed(self, data: bytes, *, timestamp: float | None = None) -> int:
        frames = self.parser.feed(data, timestamp=timestamp)
        for frame in frames:
            self.timestamp.append(frame.timestamp)
            self.sequence.append(frame.sequence)
            self.header_byte_1_u8.append(
                -1 if frame.header_byte_1 is None else frame.header_byte_1
            )
            self.hall_xyz.append(frame.hall_xyz.copy())
            self.temperature_c.append(frame.temperature.copy())
            self.valid.append(frame.valid)
        return len(frames)

    def save(self, path: Path) -> None:
        if not self.hall_xyz:
            raise ValueError("no complete sensor frames were received")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            timestamp_s=np.asarray(self.timestamp, dtype=np.float64),
            sequence=np.asarray(self.sequence, dtype=np.int64),
            header_byte_1_u8=np.asarray(self.header_byte_1_u8, dtype=np.int16),
            hall_xyz=np.asarray(self.hall_xyz, dtype=np.int32),
            temperature_c=np.asarray(self.temperature_c, dtype=np.float32),
            valid=np.asarray(self.valid, dtype=bool),
            metadata=np.asarray(
                [
                    "sensor=FootSensor15",
                    f"foot_id={self.parser.foot_id}",
                    f"source_address={self.source_address}",
                    f"adapter={self.adapter}",
                    "force_calibration=absent",
                    "hall_axes=wire_order_before_canonical_layout_transform",
                    "timestamp=host_monotonic",
                    "sequence=host_monotonic_counter",
                    "header_byte_1_u8=firmware_semantics_unknown_not_a_frame_counter",
                ]
            ),
        )
        temporary.replace(path)


async def _record_live(
    recorder: RawHallRecorder,
    *,
    duration_s: float,
    address: str | None,
    adapter: str | None,
) -> None:
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as error:
        raise RuntimeError(
            "live BLE recording requires bleak; install vis/requirements.txt"
        ) from error
    device = None
    if address:
        device = address
    else:
        devices = await BleakScanner.discover(timeout=8.0)
        device = next(
            (
                candidate
                for candidate in devices
                if candidate.name == DEVICE_NAME
            ),
            None,
        )
    if device is None:
        raise RuntimeError(f"BLE device {DEVICE_NAME!r} was not found")

    def notification(_: object, data: bytearray) -> None:
        recorder.feed(bytes(data), timestamp=time.monotonic())

    client_options: dict[str, object] = {}
    if adapter:
        if "bluez" in inspect.signature(BleakClient).parameters:
            client_options["bluez"] = {"adapter": adapter}
        else:
            client_options["adapter"] = adapter
    async with BleakClient(device, **client_options) as client:
        await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification)
        await asyncio.sleep(duration_s)
        await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)


def _self_test() -> dict[str, object]:
    payload = bytearray(FRAME_LENGTH)
    payload[0:4] = bytes((0x7D, 0x00, 0xF0, 0x02))
    for channel in range(15):
        struct.pack_into(
            ">hhhh",
            payload,
            4 + channel * 8,
            250 + channel,
            channel,
            -channel,
            100 + channel,
        )
    recorder = RawHallRecorder("left")
    parsed = recorder.feed(bytes(payload), timestamp=12.5)
    return {
        "parsed_frames": parsed,
        "hall_shape": list(recorder.hall_xyz[0].shape),
        "temperature_first_c": float(recorder.temperature_c[0][0]),
        "sequence": recorder.sequence[0],
        "header_byte_1_u8": recorder.header_byte_1_u8[0],
        "calibrated_force_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--foot_id", choices=("left", "right"), default="left")
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--address")
    parser.add_argument(
        "--adapter",
        help="BlueZ adapter, for example hci0; required for unambiguous dual-foot capture",
    )
    parser.add_argument("--binary_input", type=Path)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --self_test is used")
    recorder = RawHallRecorder(
        args.foot_id,
        source_address=args.address,
        adapter=args.adapter,
    )
    if args.binary_input:
        recorder.feed(args.binary_input.read_bytes(), timestamp=time.monotonic())
    else:
        asyncio.run(
            _record_live(
                recorder,
                duration_s=args.duration_s,
                address=args.address,
                adapter=args.adapter,
            )
        )
    recorder.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frames": len(recorder.hall_xyz),
                "rejected_bytes": recorder.parser.rejected_bytes,
                "calibrated_force_created": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
