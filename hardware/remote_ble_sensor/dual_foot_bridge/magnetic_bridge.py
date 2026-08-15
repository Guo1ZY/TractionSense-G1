"""Live dual-BLE normalized magnetic bridge using the F0M1 packet."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

from .bridge import (
    DEFAULT_HEALTH,
    DEFAULT_OUT,
    FootState,
    HallSample,
    RawCsvLogger,
    SensorPipeline,
    _load_config,
    _resolve,
    _sensor_loop,
    configured_adapters,
)
from .ipc import atomic_write
from .magnetic_ipc import F0M1Writer
from .normalization import MagneticNormalizer
from .protocol import CHAR_UUID


@dataclass(frozen=True)
class SynchronizedHallPair:
    left: HallSample
    right: HallSample
    skew_s: float


class DualFootSynchronizer:
    """Nearest-neighbor pairing on same-host BLE receive timestamps.

    FootSensor15 frames contain no hardware sampling timestamp, so this is
    explicitly host-receive synchronization rather than sensor-clock sync.
    """

    def __init__(
        self,
        max_pair_skew_s: float,
        source_timeout_s: float,
        holdback_s: float = 0.0,
    ) -> None:
        if not 0.0 < max_pair_skew_s <= source_timeout_s:
            raise ValueError(
                "ble.max_pair_skew_s must be in (0, ble.source_timeout_s]"
            )
        if not 0.0 <= holdback_s < source_timeout_s:
            raise ValueError(
                "ble.sync_holdback_s must be in [0, ble.source_timeout_s)"
            )
        self.max_pair_skew_s = max_pair_skew_s
        self.source_timeout_s = source_timeout_s
        self.holdback_s = holdback_s
        self.last_used = {"left": -math.inf, "right": -math.inf}
        self.last_pair: SynchronizedHallPair | None = None
        self.synchronized_pairs = 0
        self.sync_misses = 0
        self.max_observed_pair_skew_s = 0.0
        self._recent_pair_times: deque[float] = deque(maxlen=256)

    def match(
        self, states: dict[str, FootState], now: float
    ) -> SynchronizedHallPair | None:
        cutoff = now - self.holdback_s
        available: dict[str, list[HallSample]] = {}
        for side in ("left", "right"):
            available[side] = [
                sample
                for sample in states[side].samples
                if sample.received_monotonic > self.last_used[side]
                and sample.received_monotonic <= cutoff
                and 0.0 <= now - sample.received_monotonic <= self.source_timeout_s
            ]
        candidates: list[SynchronizedHallPair] = []
        for left in available["left"]:
            for right in available["right"]:
                skew = abs(left.received_monotonic - right.received_monotonic)
                if skew <= self.max_pair_skew_s:
                    candidates.append(SynchronizedHallPair(left, right, skew))
        if not candidates:
            self.sync_misses += 1
            return None

        # Decimate the ~100 Hz source streams onto the 50 Hz publisher by
        # taking the newest feasible pair.  The hard skew bound is checked
        # before freshness is used as a tie-breaker.
        pair = max(
            candidates,
            key=lambda item: (
                min(
                    item.left.received_monotonic,
                    item.right.received_monotonic,
                ),
                -item.skew_s,
            ),
        )
        self.last_used["left"] = pair.left.received_monotonic
        self.last_used["right"] = pair.right.received_monotonic
        self.last_pair = pair
        self.synchronized_pairs += 1
        self.max_observed_pair_skew_s = max(
            self.max_observed_pair_skew_s, pair.skew_s
        )
        self._recent_pair_times.append(now)
        return pair

    def health_document(self, now: float, synchronized: bool) -> dict:
        pair = self.last_pair
        recent = [stamp for stamp in self._recent_pair_times if now - stamp <= 2.0]
        pair_rate_hz = 0.0
        if len(recent) >= 2 and recent[-1] > recent[0]:
            pair_rate_hz = (len(recent) - 1) / (recent[-1] - recent[0])
        if pair is None:
            pair_age_s = None
            pair_skew_s = None
            left_wall_ns = None
            right_wall_ns = None
        else:
            pair_age_s = max(
                0.0,
                now
                - min(
                    pair.left.received_monotonic,
                    pair.right.received_monotonic,
                ),
            )
            pair_skew_s = pair.skew_s
            left_wall_ns = pair.left.received_wall_ns
            right_wall_ns = pair.right.received_wall_ns
        return {
            "method": "nearest_host_monotonic",
            "hardware_timestamp_available": False,
            "synchronized": synchronized,
            "max_pair_skew_s": round(self.max_pair_skew_s, 6),
            "holdback_s": round(self.holdback_s, 6),
            "last_pair_skew_s": (
                None if pair_skew_s is None else round(pair_skew_s, 6)
            ),
            "max_observed_pair_skew_s": round(
                self.max_observed_pair_skew_s, 6
            ),
            "last_pair_age_s": (
                None if pair_age_s is None else round(pair_age_s, 6)
            ),
            "left_receive_wall_ns": left_wall_ns,
            "right_receive_wall_ns": right_wall_ns,
            "synchronized_pairs": self.synchronized_pairs,
            "sync_misses": self.sync_misses,
            "recent_pair_rate_hz": round(pair_rate_hz, 3),
        }

    def is_synchronized(self, now: float, grace_s: float) -> bool:
        pair = self.last_pair
        if pair is None:
            return False
        oldest_stamp = min(
            pair.left.received_monotonic,
            pair.right.received_monotonic,
        )
        return 0.0 <= now - oldest_stamp <= grace_s


def _magnetic_health_document(
    states: dict[str, FootState],
    normalizers: dict[str, MagneticNormalizer | None],
    source_timeout_s: float,
    publishing: bool,
    synchronizer: DualFootSynchronizer | None = None,
    synchronized: bool = False,
) -> dict:
    """Describe only observable Hall/temperature/link health.

    In particular this document must not inherit the legacy F0T1 bridge's
    normal/tangential-force fields: F0M1 has no such measurement.
    """
    now = time.monotonic()
    feet = {}
    for side, state in states.items():
        age = state.age(now)
        feet[side] = {
            "address": state.address,
            "adapter": state.adapter or None,
            "connected": state.connected,
            "fresh": age <= source_timeout_s,
            "age_s": None if not math.isfinite(age) else round(age, 6),
            "frames": state.frames,
            "rejected_frames": state.rejected_frames,
            "reconnects": state.reconnects,
            "source_sequence": state.source_sequence,
            "source_sequence_semantics": "header_byte_1_unknown_not_gap_counter",
            "temperature_c_min": (
                None
                if state.frames == 0
                else round(float(np.min(state.temperature_x10)) * 0.1, 2)
            ),
            "hall_raw_count_min": (
                None if state.frames == 0 else round(float(np.min(state.magnetic_xyz)), 3)
            ),
            "hall_raw_count_max": (
                None if state.frames == 0 else round(float(np.max(state.magnetic_xyz)), 3)
            ),
            # Current raw diagnostics stay in the health side-channel only;
            # the policy IPC still contains normalized Hall Bx/By/Bz and no
            # force-like quantity.  These arrays let preflight detect wiring,
            # saturation and temperature faults before enabling motion.
            "hall_raw_count_xyz": (
                None
                if state.frames == 0
                else np.asarray(state.magnetic_xyz, dtype=float).tolist()
            ),
            "temperature_c": (
                None
                if state.frames == 0
                else (0.1 * np.asarray(state.temperature_x10, dtype=float)).tolist()
            ),
            "normalized": normalizers[side] is not None,
            "sample_period_s": round(state.sample_period_s, 6),
            "sample_rate_hz": round(1.0 / max(state.sample_period_s, 1.0e-6), 3),
            "last_error": state.last_error,
        }
    document = {
        "format": "g1-dual-foot-magnetic-health-v1",
        "wall_time_ns": time.time_ns(),
        "publishing_f0m1": publishing,
        "measurement": "baseline/temperature compensated normalized Hall response",
        "raw_unit": "device_count",
        "force_available": False,
        "feet": feet,
    }
    if synchronizer is not None:
        document["synchronization"] = synchronizer.health_document(
            now, synchronized
        )
    return document


def load_normalizers(
    config: dict, config_dir: Path, raw_only: bool
) -> dict[str, MagneticNormalizer | None]:
    result = {}
    for side in ("left", "right"):
        value = str(config[side].get("normalization", "")).strip()
        if not value:
            if raw_only:
                result[side] = None
                continue
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
    health_path: Path,
    stop: asyncio.Event,
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
    synchronizer = DualFootSynchronizer(
        max_pair_skew_s=float(ble.get("max_pair_skew_s", 0.010)),
        source_timeout_s=timeout,
        holdback_s=float(ble.get("sync_holdback_s", 0.0)),
    )
    sync_grace_s = min(
        timeout,
        max(2.0 / rate, synchronizer.holdback_s + 2.0 * synchronizer.max_pair_skew_s),
    )
    last_print = 0.0
    last_health = 0.0
    while not stop.is_set():
        started = time.monotonic()
        age = {side: states[side].age(started) for side in ("left", "right")}
        fresh = all(age[side] <= timeout and states[side].connected for side in age)
        pair = synchronizer.match(states, started) if fresh else None
        synchronized = synchronizer.is_synchronized(started, sync_grace_s)
        ready = all(normalizers[side] is not None for side in ("left", "right"))
        stream_healthy = bool(writer and fresh and ready and synchronized)
        if writer is not None and fresh and ready and pair is not None:
            normalized = {}
            pair_samples = {"left": pair.left, "right": pair.right}
            pair_age = {
                side: max(0.0, started - pair_samples[side].received_monotonic)
                for side in ("left", "right")
            }
            for side, sample in pair_samples.items():
                normalizer = normalizers[side]
                assert normalizer is not None
                normalized[side] = normalizer.normalize(
                    sample.magnetic_xyz, sample.temperature_x10
                )
            writer.write(
                normalized["left"],
                normalized["right"],
                valid_left=1.0,
                valid_right=1.0,
                age_left_s=pair_age["left"],
                age_right_s=pair_age["right"],
                period_left_s=pair.left.sample_period_s,
                period_right_s=pair.right.sample_period_s,
            )
        if started - last_health >= 0.2:
            last_health = started
            health = _magnetic_health_document(
                states,
                normalizers,
                timeout,
                stream_healthy,
                synchronizer,
                synchronized,
            )
            atomic_write(
                health_path,
                (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode(),
            )
        if print_hz > 0 and started - last_print >= 1.0 / print_hz:
            last_print = started
            sync_text = (
                f"SYNC=1 skew={synchronizer.last_pair.skew_s * 1000.0:.2f}ms"
                if synchronized and synchronizer.last_pair is not None
                else "SYNC=0 skew=NA"
            )
            print(
                f"F0M1={'ON' if stream_healthy else 'OFF'} "
                f"BLE={int(states['left'].connected)}/{int(states['right'].connected)} "
                f"Hz={1/states['left'].sample_period_s:.1f}/"
                f"{1/states['right'].sample_period_s:.1f} "
                f"age={age['left']:.3f}/{age['right']:.3f}s "
                f"{sync_text}",
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
    adapters = configured_adapters(config)
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
        side: FootState(side=side, address=addresses[side], adapter=adapters[side])
        for side in ("left", "right")
    }
    ble = config.get("ble", {})
    alpha = float(ble.get("ema_alpha", 0.25))
    pipelines = {
        side: SensorPipeline(side, config[side], states[side], logger, alpha)
        for side in ("left", "right")
    }
    writer = None if args.raw_only else F0M1Writer(out)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass
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
            )
        )
        for side in ("left", "right")
    ]
    tasks.append(
        asyncio.create_task(publish(config, states, normalizers, writer, health, stop))
    )
    if args.duration > 0:
        async def stop_later() -> None:
            await asyncio.sleep(args.duration)
            stop.set()
        tasks.append(asyncio.create_task(stop_later()))
    try:
        while not stop.is_set():
            done, _ = await asyncio.wait(tasks, timeout=0.2, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.close()
        if out.exists():
            out.unlink()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--reference-left", type=float)
    parser.add_argument("--reference-right", type=float)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args, _load_config(args.config)))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
