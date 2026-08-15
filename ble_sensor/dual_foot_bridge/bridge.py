"""Headless dual-BLE bridge from FootSensor15 magnetic arrays to g1_ctrl."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import csv
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibration, ForceEstimate
from .ipc import F0T1Writer, atomic_write
from .protocol import (
    CHAR_UUID,
    DEVICE_NAME,
    FrameError,
    FrameParser,
    NUM_SENSORS,
    transform_magnetic,
)


DEFAULT_OUT = Path("/tmp/g1_foot_rl_obs.bin")
DEFAULT_HEALTH = Path("/tmp/g1_foot_ble_health.json")
MAG_COLUMNS = [
    f"mag_{sensor}_{axis}"
    for sensor in range(NUM_SENSORS)
    for axis in ("x", "y", "z")
]
TEMP_COLUMNS = [f"temp_{sensor}_x10" for sensor in range(NUM_SENSORS)]


@dataclass
class FootState:
    side: str
    address: str
    adapter: str = ""
    device_name: str = ""
    connected: bool = False
    frames: int = 0
    rejected_frames: int = 0
    reconnects: int = 0
    saturation_frames: int = 0
    last_saturation_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_SENSORS, 3), dtype=bool)
    )
    last_error: str = ""
    last_wall_ns: int = 0
    last_monotonic_ns: int = 0
    last_monotonic: float = 0.0
    source_sequence: int = 0
    sample_period_s: float = 0.02
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    temperature_x10: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_SENSORS, dtype=np.int32)
    )
    magnetic_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_SENSORS, 3), dtype=np.float64)
    )
    raw_magnetic_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_SENSORS, 3), dtype=np.int64)
    )
    force: ForceEstimate = field(default_factory=lambda: ForceEstimate(0.0, 0.0))

    def age(self, now: float | None = None) -> float:
        if self.last_monotonic <= 0.0:
            return float("inf")
        return (time.monotonic() if now is None else now) - self.last_monotonic


class RawCsvLogger:
    def __init__(
        self,
        path: Path | None,
        references_n: dict[str, float | None],
    ) -> None:
        self._stream = None
        self._writer = None
        self.references_n = references_n
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.stat().st_size > 0
            self._stream = path.open("a", newline="", encoding="utf-8")
            fieldnames = [
                "wall_time_ns",
                "monotonic_ns",
                "monotonic_s",
                "side",
                "device_name",
                "address",
                "adapter",
                "source_sequence",
                "sample_period_s",
                "valid",
                "reference_normal_n",
                *TEMP_COLUMNS,
                *MAG_COLUMNS,
            ]
            self._writer = csv.DictWriter(self._stream, fieldnames=fieldnames)
            if not exists:
                self._writer.writeheader()

    def write(self, state: FootState) -> None:
        if self._writer is None:
            return
        row: dict[str, Any] = {
            "wall_time_ns": state.last_wall_ns,
            "monotonic_ns": state.last_monotonic_ns,
            "monotonic_s": f"{state.last_monotonic:.9f}",
            "side": state.side,
            "device_name": state.device_name,
            "address": state.address,
            "adapter": state.adapter,
            "source_sequence": state.source_sequence,
            "sample_period_s": f"{state.sample_period_s:.9f}",
            "valid": 1,
            "reference_normal_n": (
                ""
                if self.references_n[state.side] is None
                else self.references_n[state.side]
            ),
        }
        row.update(
            {
                name: int(value)
                for name, value in zip(TEMP_COLUMNS, state.temperature_x10)
            }
        )
        row.update(
            {
                name: int(value)
                for name, value in zip(MAG_COLUMNS, state.raw_magnetic_xyz.reshape(-1))
            }
        )
        self._writer.writerow(row)
        if state.frames % 20 == 0:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()


class SensorPipeline:
    def __init__(
        self,
        side: str,
        side_config: dict[str, Any],
        state: FootState,
        raw_logger: RawCsvLogger,
        ema_alpha: float,
    ) -> None:
        self.side = side
        self.state = state
        self.raw_logger = raw_logger
        self.parser = FrameParser()
        self.permutation = list(side_config.get("sensor_permutation", range(NUM_SENSORS)))
        self.axis_sign = list(side_config.get("axis_sign", [1, 1, 1]))
        self.ema_alpha = ema_alpha
        self._filtered: np.ndarray | None = None
        transform_magnetic(
            np.zeros((NUM_SENSORS, 3)), self.permutation, self.axis_sign
        )

    def reset(self) -> None:
        self.parser.reset()
        self._filtered = None
        self.state.frame_times.clear()

    def receive(self, data: bytes) -> None:
        try:
            frame = self.parser.parse(data)
            magnetic = transform_magnetic(
                frame.magnetic_xyz, self.permutation, self.axis_sign
            )
            if self._filtered is None:
                self._filtered = magnetic
            else:
                self._filtered += self.ema_alpha * (magnetic - self._filtered)
            self.state.frame_times.append(frame.received_monotonic)
            cutoff = frame.received_monotonic - 2.0
            while self.state.frame_times and self.state.frame_times[0] < cutoff:
                self.state.frame_times.popleft()
            if len(self.state.frame_times) >= 2:
                elapsed = self.state.frame_times[-1] - self.state.frame_times[0]
                if elapsed > 0.0:
                    self.state.sample_period_s = elapsed / (
                        len(self.state.frame_times) - 1
                    )
            self.state.source_sequence = frame.source_sequence
            self.state.last_wall_ns = frame.received_wall_ns
            self.state.last_monotonic_ns = frame.received_monotonic_ns
            self.state.last_monotonic = frame.received_monotonic
            self.state.temperature_x10 = frame.temperature_x10
            self.state.raw_magnetic_xyz = magnetic.astype(np.int64, copy=True)
            self.state.magnetic_xyz = self._filtered.copy()
            self.state.frames += 1
            if frame.saturation_xyz is not None and bool(np.any(frame.saturation_xyz)):
                self.state.saturation_frames += 1
                self.state.last_saturation_xyz = frame.saturation_xyz
            self.raw_logger.write(self.state)
        except (FrameError, ValueError) as error:
            self.state.rejected_frames += 1
            self.state.last_error = str(error)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load config {path}: {error}") from error
    if config.get("format") != "g1-dual-foot-ble-config-v1":
        raise ValueError("config format must be g1-dual-foot-ble-config-v1")
    for side in ("left", "right"):
        if not isinstance(config.get(side), dict):
            raise ValueError(f"missing {side} configuration")
    names = {
        side: str(config[side].get("device_name", side)).strip()
        for side in ("left", "right")
    }
    if not all(names.values()) or names["left"].casefold() == names["right"].casefold():
        raise ValueError("left/right device_name values must be non-empty and different")
    return config


def _resolve(base: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else base / path


def _load_calibrations(
    config: dict[str, Any], config_dir: Path, raw_only: bool
) -> dict[str, Calibration | None]:
    result: dict[str, Calibration | None] = {}
    for side in ("left", "right"):
        if raw_only:
            result[side] = None
            continue
        value = str(config[side].get("calibration", "")).strip()
        if not value:
            raise ValueError(f"{side}.calibration is required unless --raw-only is used")
        result[side] = Calibration.load(_resolve(config_dir, value), expected_side=side)
    return result


async def _sensor_loop(
    side: str,
    side_config: dict[str, Any],
    characteristic_uuid: str,
    reconnect_delay_s: float,
    state: FootState,
    pipeline: SensorPipeline,
    stop: asyncio.Event,
    notify_ready: set[str] | None = None,
    notify_barrier: asyncio.Event | None = None,
) -> None:
    try:
        from bleak import BleakClient
    except ImportError as error:
        raise RuntimeError("missing Python package 'bleak'; install requirements.txt") from error

    while not stop.is_set():
        client = None
        try:
            pipeline.reset()
            client_options = {"adapter": state.adapter} if state.adapter else {}
            client = BleakClient(state.address, **client_options)
            await client.connect()
            state.connected = True
            state.last_error = ""

            def on_notify(_sender: Any, data: bytearray) -> None:
                pipeline.receive(bytes(data))

            if notify_ready is not None and notify_barrier is not None:
                notify_ready.add(side)
                if len(notify_ready) >= 2:
                    notify_barrier.set()
                try:
                    await asyncio.wait_for(notify_barrier.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    notify_barrier.set()
            await client.start_notify(characteristic_uuid, on_notify)
            while client.is_connected and not stop.is_set():
                await asyncio.sleep(0.1)
            if not stop.is_set():
                state.last_error = "BLE disconnected"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            state.last_error = f"{type(error).__name__}: {error}"
        finally:
            state.connected = False
            if client is not None:
                try:
                    if client.is_connected:
                        await client.disconnect()
                except Exception:
                    pass
        if not stop.is_set():
            state.reconnects += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=reconnect_delay_s)
            except asyncio.TimeoutError:
                pass


def _health_document(
    states: dict[str, FootState],
    calibrations: dict[str, Calibration | None],
    source_timeout_s: float,
    publishing: bool,
) -> dict[str, Any]:
    now = time.monotonic()
    feet: dict[str, Any] = {}
    for side, state in states.items():
        age = state.age(now)
        feet[side] = {
            "device_name": state.device_name,
            "address": state.address,
            "adapter": state.adapter or None,
            "connected": state.connected,
            "fresh": age <= source_timeout_s,
            "age_s": None if not math.isfinite(age) else round(age, 6),
            "frames": state.frames,
            "rejected_frames": state.rejected_frames,
            "reconnects": state.reconnects,
            "saturation_frames": state.saturation_frames,
            "source_sequence": state.source_sequence,
            "temperature_c_min": (
                None
                if state.frames == 0
                else round(float(np.min(state.temperature_x10)) * 0.1, 2)
            ),
            "magnetic_min": (
                None if state.frames == 0 else round(float(np.min(state.magnetic_xyz)), 3)
            ),
            "magnetic_max": (
                None if state.frames == 0 else round(float(np.max(state.magnetic_xyz)), 3)
            ),
            "normal_n": round(state.force.normal_n, 3),
            "tangent_n": round(state.force.tangent_n, 3),
            "calibrated": calibrations[side] is not None,
            "last_error": state.last_error,
        }
    return {
        "format": "g1-dual-foot-ble-health-v1",
        "wall_time_ns": time.time_ns(),
        "publishing_f0t1": publishing,
        "feet": feet,
    }


async def _publish_loop(
    config: dict[str, Any],
    states: dict[str, FootState],
    calibrations: dict[str, Calibration | None],
    writer: F0T1Writer | None,
    health_path: Path,
    stop: asyncio.Event,
) -> None:
    ble_config = config.get("ble", {})
    output_config = config.get("output", {})
    source_timeout_s = float(ble_config.get("source_timeout_s", 0.20))
    rate_hz = float(output_config.get("rate_hz", 50.0))
    print_hz = float(output_config.get("print_hz", 1.0))
    if source_timeout_s <= 0.0 or source_timeout_s >= 0.25:
        raise ValueError("ble.source_timeout_s must be in (0, 0.25)")
    if rate_hz <= 0.0 or rate_hz > 200.0:
        raise ValueError("output.rate_hz must be in (0, 200]")
    period = 1.0 / rate_hz
    last_print = 0.0
    last_health = 0.0
    while not stop.is_set():
        loop_started = time.monotonic()
        fresh = all(state.age(loop_started) <= source_timeout_s for state in states.values())
        calibrated = all(calibrations[side] is not None for side in ("left", "right"))
        publishing = bool(writer is not None and fresh and calibrated)
        if publishing:
            try:
                for side in ("left", "right"):
                    calibration = calibrations[side]
                    assert calibration is not None
                    states[side].force = calibration.estimate(states[side].magnetic_xyz)
                writer.write(
                    states["left"].force.normal_n,
                    states["right"].force.normal_n,
                    states["left"].force.tangent_n,
                    states["right"].force.tangent_n,
                )
            except ValueError as error:
                publishing = False
                for state in states.values():
                    state.last_error = f"force estimate rejected: {error}"
        if loop_started - last_health >= 0.2:
            last_health = loop_started
            health = _health_document(
                states, calibrations, source_timeout_s, publishing
            )
            atomic_write(
                health_path,
                (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode(),
            )
        if print_hz > 0.0 and loop_started - last_print >= 1.0 / print_hz:
            last_print = loop_started
            left, right = states["left"], states["right"]
            print(
                f"F0T1={'ON' if publishing else 'OFF'} "
                f"BLE={int(left.connected)}/{int(right.connected)} "
                f"age={left.age(loop_started):.3f}/{right.age(loop_started):.3f}s "
                f"Fn={left.force.normal_n:.1f}/{right.force.normal_n:.1f}N "
                f"frames={left.frames}/{right.frames}",
                flush=True,
            )
        remaining = period - (time.monotonic() - loop_started)
        if remaining > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass


async def run_hardware(args: argparse.Namespace, config: dict[str, Any]) -> int:
    config_dir = args.config.resolve().parent
    calibrations = _load_calibrations(config, config_dir, args.raw_only)
    ble_config = config.get("ble", {})
    output_config = config.get("output", {})
    addresses = {
        side: str(config[side].get("address", "")).strip()
        for side in ("left", "right")
    }
    if not all(addresses.values()):
        raise ValueError("left.address and right.address are required")
    if addresses["left"].casefold() == addresses["right"].casefold():
        raise ValueError("left and right must use different BLE addresses")
    out = args.out or _resolve(
        config_dir, str(output_config.get("path", DEFAULT_OUT))
    )
    health_path = args.health or _resolve(
        config_dir, str(output_config.get("health_path", DEFAULT_HEALTH))
    )
    if out.exists():
        out.unlink()
    logger = RawCsvLogger(
        args.record,
        {"left": args.reference_left, "right": args.reference_right},
    )
    states = {
        side: FootState(
            side=side,
            address=addresses[side],
            adapter=str(config[side].get("adapter", "")).strip(),
            device_name=str(config[side].get("device_name", side)).strip(),
        )
        for side in ("left", "right")
    }
    ema_alpha = float(ble_config.get("ema_alpha", 0.25))
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ble.ema_alpha must be in (0, 1]")
    pipelines = {
        side: SensorPipeline(
            side, config[side], states[side], logger, ema_alpha
        )
        for side in ("left", "right")
    }
    writer = None
    if not args.raw_only:
        writer = F0T1Writer(
            out,
            contact_threshold_n=float(output_config.get("contact_threshold_n", 5.0)),
            max_force_n=float(output_config.get("max_force_n", 500.0)),
        )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass
    notify_ready: set[str] = set()
    notify_barrier = asyncio.Event()
    tasks = [
        # Connect both GATT clients first, then enable both Notify streams together.
        # This reduces the controller scheduling bias observed when one foot starts early.
        asyncio.create_task(
            _sensor_loop(
                side,
                config[side],
                str(ble_config.get("characteristic_uuid", CHAR_UUID)),
                float(ble_config.get("reconnect_delay_s", 2.0)),
                states[side],
                pipelines[side],
                stop,
                notify_ready,
                notify_barrier,
            ),
            name=f"BLE-{side}",
        )
        for side in ("left", "right")
    ]
    tasks.append(
        asyncio.create_task(
            _publish_loop(config, states, calibrations, writer, health_path, stop),
            name="publisher",
        )
    )
    if args.duration > 0.0:
        async def stop_later() -> None:
            await asyncio.sleep(args.duration)
            stop.set()
        tasks.append(asyncio.create_task(stop_later(), name="duration"))
    try:
        while not stop.is_set():
            done, _ = await asyncio.wait(tasks, timeout=0.2, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.close()
        if out.exists():
            out.unlink()
    return 0


async def run_simulation(args: argparse.Namespace, config: dict[str, Any]) -> int:
    config_dir = args.config.resolve().parent
    output_config = config.get("output", {})
    out = args.out or _resolve(
        config_dir, str(output_config.get("path", DEFAULT_OUT))
    )
    if out == DEFAULT_OUT and not args.allow_sim_output:
        raise ValueError(
            "refusing to write simulated data to the live default path; "
            "use --out /tmp/g1_foot_sim.bin or explicitly pass --allow-sim-output"
        )
    writer = F0T1Writer(out)
    start = time.monotonic()
    duration = args.duration if args.duration > 0.0 else 3.0
    while time.monotonic() - start < duration:
        phase = 2.0 * math.pi * 1.2 * (time.monotonic() - start)
        left = max(0.0, 300.0 + 180.0 * math.sin(phase))
        right = max(0.0, 300.0 - 180.0 * math.sin(phase))
        writer.write(left, right, 0.0, 0.0)
        await asyncio.sleep(0.02)
    print(f"simulation wrote valid 40-byte F0T1 packets to {out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive two FootSensor15 BLE devices and feed g1_ctrl"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--record", type=Path, help="append every magnetic frame to CSV")
    parser.add_argument("--reference-left", type=float)
    parser.add_argument("--reference-right", type=float)
    parser.add_argument("--raw-only", action="store_true", help="never create F0T1 output")
    parser.add_argument("--simulate", action="store_true", help="offline F0T1 test; no BLE")
    parser.add_argument("--allow-sim-output", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _load_config(args.config)
        if args.record is not None and not args.record.is_absolute():
            args.record = Path.cwd() / args.record
        return asyncio.run(
            run_simulation(args, config) if args.simulate else run_hardware(args, config)
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
