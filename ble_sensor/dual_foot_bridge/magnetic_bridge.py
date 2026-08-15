"""Live dual-BLE normalized magnetic bridge using the F0M1 packet."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from .bridge import (
    DEFAULT_HEALTH,
    DEFAULT_OUT,
    FootState,
    RawCsvLogger,
    SensorPipeline,
    _health_document,
    _load_config,
    _resolve,
    _sensor_loop,
)
from .capture_ipc import F0R1Writer, MISSING_AGE_S, PairedCsvLogger
from .ipc import atomic_write
from .magnetic_ipc import F0M1Writer
from .normalization import MagneticNormalizer
from .protocol import CHAR_UUID


def load_normalizers(
    config: dict, config_dir: Path, raw_only: bool
) -> dict[str, MagneticNormalizer | None]:
    result = {}
    for side in ("left", "right"):
        if raw_only:
            result[side] = None
            continue
        value = str(config[side].get("normalization", "")).strip()
        if not value:
            raise ValueError(f"{side}.normalization is required unless --raw-only is used")
        result[side] = MagneticNormalizer.load(
            _resolve(config_dir, value), expected_side=side
        )
    return result


async def publish(
    config: dict,
    states: dict[str, FootState],
    normalizers: dict[str, MagneticNormalizer | None],
    writer: F0M1Writer | None,
    capture_writer: F0R1Writer | None,
    paired_logger: PairedCsvLogger,
    health_path: Path,
    stop: asyncio.Event,
    both_fresh_event: asyncio.Event | None = None,
) -> None:
    ble = config.get("ble", {})
    output = config.get("output", {})
    timeout = float(ble.get("source_timeout_s", 0.20))
    if not 0.0 < timeout < 0.25:
        raise ValueError("ble.source_timeout_s must be in (0, 0.25)")
    rate = float(output.get("rate_hz", 50.0))
    if not 0.0 < rate <= 200.0:
        raise ValueError("output.rate_hz must be in (0, 200]")
    print_hz = float(output.get("print_hz", 1.0))
    last_print = 0.0
    last_health = 0.0
    while not stop.is_set():
        publish_monotonic_ns = time.monotonic_ns()
        publish_wall_ns = time.time_ns()
        started = publish_monotonic_ns * 1.0e-9
        age = {side: states[side].age(started) for side in ("left", "right")}
        valid = {
            side: bool(
                states[side].connected
                and states[side].frames > 0
                and age[side] <= timeout
            )
            for side in ("left", "right")
        }
        fresh = all(valid.values())
        if fresh and both_fresh_event is not None:
            both_fresh_event.set()
        if capture_writer is not None:
            capture_sample = capture_writer.write(
                states["left"].raw_magnetic_xyz,
                states["right"].raw_magnetic_xyz,
                states["left"].temperature_x10,
                states["right"].temperature_x10,
                publish_wall_ns=publish_wall_ns,
                publish_monotonic_ns=publish_monotonic_ns,
                frame_wall_ns=(
                    states["left"].last_wall_ns,
                    states["right"].last_wall_ns,
                ),
                frame_monotonic_ns=(
                    states["left"].last_monotonic_ns,
                    states["right"].last_monotonic_ns,
                ),
                source_sequence=(
                    states["left"].source_sequence,
                    states["right"].source_sequence,
                ),
                valid=(valid["left"], valid["right"]),
                age_s=(
                    age["left"] if states["left"].frames else MISSING_AGE_S,
                    age["right"] if states["right"].frames else MISSING_AGE_S,
                ),
                period_s=(
                    states["left"].sample_period_s,
                    states["right"].sample_period_s,
                ),
            )
            paired_logger.write(capture_sample)
        ready = all(normalizers[side] is not None for side in ("left", "right"))
        publishing = bool(writer and fresh and ready)
        if publishing:
            normalized = {}
            for side in ("left", "right"):
                normalizer = normalizers[side]
                assert normalizer is not None
                normalized[side] = normalizer.normalize(
                    states[side].magnetic_xyz, states[side].temperature_x10
                )
            writer.write(
                normalized["left"],
                normalized["right"],
                valid_left=1.0,
                valid_right=1.0,
                age_left_s=age["left"],
                age_right_s=age["right"],
                period_left_s=states["left"].sample_period_s,
                period_right_s=states["right"].sample_period_s,
            )
        if started - last_health >= 0.2:
            last_health = started
            # Reuse common BLE health and add the actual measured sample period.
            health = _health_document(states, normalizers, timeout, publishing)
            health["format"] = "g1-dual-foot-magnetic-health-v1"
            health["publishing_f0m1"] = health.pop("publishing_f0t1")
            health["publishing_f0r1"] = capture_writer is not None
            for side in ("left", "right"):
                # Hall-only path: never expose legacy Hall-to-force estimates.
                health["feet"][side].pop("normal_n", None)
                health["feet"][side].pop("tangent_n", None)
                health["feet"][side]["sample_period_s"] = round(
                    states[side].sample_period_s, 6
                )
                health["feet"][side]["sample_rate_hz"] = round(
                    1.0 / max(states[side].sample_period_s, 1.0e-6), 3
                )
                health["feet"][side]["normalized"] = normalizers[side] is not None
            atomic_write(
                health_path,
                (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode(),
            )
        if print_hz > 0 and started - last_print >= 1.0 / print_hz:
            last_print = started
            print(
                f"F0M1={'ON' if publishing else 'OFF'} "
                f"F0R1={'ON' if capture_writer is not None else 'OFF'} "
                f"BLE={int(states['left'].connected)}/{int(states['right'].connected)} "
                f"Hz={1/states['left'].sample_period_s:.1f}/"
                f"{1/states['right'].sample_period_s:.1f} "
                f"age={age['left']:.3f}/{age['right']:.3f}s",
                flush=True,
            )
        remaining = 1.0 / rate - (time.monotonic() - started)
        if remaining > 0.0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass


async def run(args: argparse.Namespace, config: dict) -> int:
    config_dir = args.config.resolve().parent
    normalizers = load_normalizers(config, config_dir, args.raw_only)
    addresses = {
        side: str(config[side].get("address", "")).strip()
        for side in ("left", "right")
    }
    if not all(addresses.values()) or addresses["left"].casefold() == addresses["right"].casefold():
        raise ValueError("two different non-empty BLE addresses are required")
    output_cfg = config.get("output", {})
    out = args.out or _resolve(config_dir, str(output_cfg.get("path", DEFAULT_OUT)))
    health = args.health or _resolve(
        config_dir, str(output_cfg.get("health_path", DEFAULT_HEALTH))
    )
    if out.exists():
        out.unlink()
    logger = RawCsvLogger(
        args.record, {"left": args.reference_left, "right": args.reference_right}
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
    ble = config.get("ble", {})
    alpha = float(ble.get("ema_alpha", 0.25))
    pipelines = {
        side: SensorPipeline(side, config[side], states[side], logger, alpha)
        for side in ("left", "right")
    }
    writer = None if args.raw_only else F0M1Writer(out)
    capture_out = getattr(args, "capture_out", None)
    if capture_out is None:
        configured_capture = str(output_cfg.get("raw_capture_path", "")).strip()
        if configured_capture:
            capture_out = _resolve(config_dir, configured_capture)
    if capture_out is not None and capture_out.exists():
        capture_out.unlink()
    paired_record = getattr(args, "paired_record", None)
    if paired_record is not None and capture_out is None:
        raise ValueError("--paired-record requires --capture-out")
    capture_writer = None if capture_out is None else F0R1Writer(capture_out)
    paired_logger = PairedCsvLogger(paired_record)
    stop = asyncio.Event()
    stop_reason = {"value": "running"}

    def request_stop(reason: str) -> None:
        if not stop.is_set():
            stop_reason["value"] = reason
        stop.set()

    both_fresh_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                signum,
                request_stop,
                f"signal:{signal.Signals(signum).name}",
            )
        except NotImplementedError:
            pass
    notify_ready: set[str] = set()
    notify_barrier = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _sensor_loop(
                side,
                config[side],
                str(ble.get("characteristic_uuid", CHAR_UUID)),
                float(ble.get("reconnect_delay_s", 2.0)),
                states[side],
                pipelines[side],
                stop,
                notify_ready,
                notify_barrier,
            )
        )
        for side in ("left", "right")
    ]
    tasks.append(
        asyncio.create_task(
            publish(
                config,
                states,
                normalizers,
                writer,
                capture_writer,
                paired_logger,
                health,
                stop,
                both_fresh_event,
            )
        )
    )
    if args.duration > 0:
        async def stop_later() -> None:
            if bool(getattr(args, "duration_after_ready", False)):
                ready_timeout_s = float(getattr(args, "ready_timeout_s", 30.0))
                if ready_timeout_s <= 0.0:
                    raise ValueError("ready_timeout_s must be positive")
                try:
                    await asyncio.wait_for(
                        both_fresh_event.wait(), timeout=ready_timeout_s
                    )
                except asyncio.TimeoutError as error:
                    stop_reason["value"] = "ready_timeout"
                    raise RuntimeError(
                        f"dual-foot BLE did not become fresh within {ready_timeout_s:.1f}s"
                    ) from error
            await asyncio.sleep(args.duration)
            request_stop("duration_complete")
        tasks.append(asyncio.create_task(stop_later()))
    try:
        while not stop.is_set():
            done, _ = await asyncio.wait(tasks, timeout=0.2, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    if stop_reason["value"] == "running":
                        stop_reason["value"] = "task_exception"
                    raise task.exception()
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.close()
        paired_logger.close()
        if out.exists():
            out.unlink()
        if capture_out is not None and capture_out.exists():
            capture_out.unlink()
        setattr(args, "stop_reason", stop_reason["value"])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--capture-out",
        type=Path,
        help="atomic F0R1 raw Hall/temperature snapshot for robot-side collectors",
    )
    parser.add_argument(
        "--paired-record",
        type=Path,
        help="50 Hz synchronized left/right raw Hall CSV",
    )
    parser.add_argument("--reference-left", type=float)
    parser.add_argument("--reference-right", type=float)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--duration-after-ready",
        action="store_true",
        help="start --duration only after both feet are connected and fresh",
    )
    parser.add_argument(
        "--ready-timeout", dest="ready_timeout_s", type=float, default=30.0
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for name in ("record", "paired_record"):
            path = getattr(args, name)
            if path is not None and not path.is_absolute():
                setattr(args, name, Path.cwd() / path)
        return asyncio.run(run(args, _load_config(args.config)))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
