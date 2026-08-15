#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FootSensor15 双脚 BLE 实时可视化。

默认按唯一广播名 ``left`` 和 ``right`` 同时连接两只脚，不再按地址排序猜测
物理左右。也可用 --left-address / --right-address 固定物理设备。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

if "--screenshot" in sys.argv and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import ble_viz_dashboard_demo as single


APP_TITLE = "模感科技 · 双足 Hall 磁场监测"
APP_VERSION = "DUAL BLE REALTIME 03 · FIXED BASELINE"
LOGICAL_SIZE = (1440, 900)
MIN_WINDOW_SIZE = (1120, 700)
FPS = 60
SIDES = ("left", "right")
SIDE_CN = {"left": "左脚", "right": "右脚"}

C = single.C
SensorFrame = single.SensorFrame
Metrics = single.Metrics
FrameStreamDecoder = single.FrameStreamDecoder
Fonts = single.Fonts
Button = single.Button
VideoRecorder = single.VideoRecorder
SENSOR_POS = single.SENSOR_POS
DEFAULT_LAYOUT_PATH = (
    Path(getattr(sys, "_MEIPASS")) / "config" / "sensor_layout_a4_15.json"
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent / "config" / "sensor_layout_a4_15.json"
)
INSOLE_OUTLINE = single.INSOLE_OUTLINE
REGION_NAMES = single.REGION_NAMES
REGION_INDICES = single.REGION_INDICES
CHIP_XY_ROTATIONS = single.CHIP_XY_ROTATIONS
NUM_SENSORS = single.NUM_SENSORS
DEVICE_NAME = single.DEVICE_NAME
DEFAULT_DEVICE_NAMES = {"left": "left", "right": "right"}
CHAR_UUID = single.CHAR_UUID

rounded_rect = single.rounded_rect
blit_text = single.blit_text
mix_color = single.mix_color
force_color = single.force_color


def load_sensor_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "footsensor15-a4-layout-v1":
        raise ValueError(f"{path}: 不支持的布局格式")
    sensors = sorted(document.get("sensors", []), key=lambda item: int(item.get("id", -1)))
    if [int(item.get("id", -1)) for item in sensors] != list(range(1, 16)):
        raise ValueError(f"{path}: 传感器编号必须恰好为 1..15")
    positions = np.asarray([item.get("normalized_uv") for item in sensors], dtype=np.float32)
    if positions.shape != (NUM_SENSORS, 2) or not np.isfinite(positions).all():
        raise ValueError(f"{path}: normalized_uv 必须是有限的 15x2 数组")
    if np.any(positions < 0.0) or np.any(positions > 1.0):
        raise ValueError(f"{path}: normalized_uv 必须位于 [0,1]")
    outline = np.asarray(document.get("outline_normalized_uv"), dtype=np.float32)
    if outline.ndim != 2 or outline.shape[1:] != (2,) or outline.shape[0] < 16:
        raise ValueError(f"{path}: outline_normalized_uv 必须是至少 16 点的 Nx2 数组")
    if not np.isfinite(outline).all() or np.any(outline < 0.0) or np.any(outline > 1.0):
        raise ValueError(f"{path}: outline_normalized_uv 必须是 [0,1] 内的有限坐标")
    return positions, outline


def load_sensor_positions(path: Path) -> np.ndarray:
    """兼容既有调用；双脚界面本身同时读取 A4 外轮廓。"""
    return load_sensor_layout(path)[0]


@dataclass
class DeviceSlot:
    side: str
    address: str = ""
    name: str = ""
    expected_name: str = ""
    status: str = "等待扫描"
    detail: str = ""
    connected: bool = False
    decoder: FrameStreamDecoder = field(default_factory=FrameStreamDecoder)
    latest: Optional[SensorFrame] = None
    notification_count: int = 0
    valid_frame_count: int = 0
    last_payload_len: int = 0
    last_valid_at: float = 0.0
    frames: deque[SensorFrame] = field(
        default_factory=lambda: deque(maxlen=64), repr=False
    )


class DualBLESource:
    """在一个 asyncio 后台线程内扫描并并发连接两台同名传感器。"""

    is_demo = False

    def __init__(
        self,
        *,
        left_address: Optional[str] = None,
        right_address: Optional[str] = None,
        left_adapter: Optional[str] = None,
        right_adapter: Optional[str] = None,
        left_name: str = "left",
        right_name: str = "right",
    ) -> None:
        self.requested_addresses = {
            "left": (left_address or "").strip(),
            "right": (right_address or "").strip(),
        }
        configured_addresses = list(self.requested_addresses.values())
        if (
            all(configured_addresses)
            and configured_addresses[0].casefold() == configured_addresses[1].casefold()
        ):
            raise ValueError("左右脚不能配置为同一个 BLE 地址")
        self.requested_adapters = {
            "left": (left_adapter or "").strip(),
            "right": (right_adapter or "").strip(),
        }
        configured_adapters = list(self.requested_adapters.values())
        if any(configured_adapters) and not all(configured_adapters):
            raise ValueError("left-adapter 与 right-adapter 需要同时提供")
        if configured_adapters[0] and configured_adapters[0] == configured_adapters[1]:
            raise ValueError("左右脚必须使用不同蓝牙适配器")
        names = {"left": left_name.strip(), "right": right_name.strip()}
        if not all(names.values()) or names["left"].casefold() == names["right"].casefold():
            raise ValueError("左右脚 BLE 广播名必须非空且互不相同")
        self.device_names = names
        self.slots = {
            side: DeviceSlot(side, expected_name=names[side]) for side in SIDES
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self.status = "准备扫描双脚"
        self.detail = f"寻找 {names['left']} / {names['right']}"
        self._notify_barrier: Optional[asyncio.Event] = None
        self._initial_ready: set[str] = set()
        self._expected_sides: set[str] = set()
        self._last_pair: Optional[tuple[SensorFrame, SensorFrame]] = None
        self._display_pair: Optional[tuple[SensorFrame, SensorFrame]] = None
        self._last_used_timestamp = {"left": -math.inf, "right": -math.inf}
        self._max_pair_skew_s = 0.010
        self._sync_grace_s = 0.040
        self._last_pair_skew_s = math.inf
        self._pair_times: deque[float] = deque(maxlen=256)
        self._display_period_s = 1.0 / 50.0
        self._next_display_at = 0.0

    @property
    def connected_count(self) -> int:
        with self.lock:
            return sum(slot.connected for slot in self.slots.values())

    @property
    def connected(self) -> bool:
        return self.connected_count == 2

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> bool:
        self.stop_event.set()
        loop = self._loop
        task = self._main_task
        if loop is not None and task is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        return not (self.thread and self.thread.is_alive())

    def sample(self) -> dict[str, Optional[SensorFrame]]:
        with self.lock:
            now = time.monotonic()
            # 先把 UI 两次轮询之间积累的帧全部按时间顺序一一配对。旧实现每次
            # render 只拿一对最新帧，因而把约 100 Hz 的原始流人为统计成 40 Hz。
            left_frames = [
                frame
                for frame in self.slots["left"].frames
                if frame.timestamp > self._last_used_timestamp["left"]
            ]
            right_frames = [
                frame
                for frame in self.slots["right"].frames
                if frame.timestamp > self._last_used_timestamp["right"]
            ]
            li = ri = 0
            while li < len(left_frames) and ri < len(right_frames):
                left = left_frames[li]
                right = right_frames[ri]
                signed_skew = left.timestamp - right.timestamp
                if abs(signed_skew) <= self._max_pair_skew_s:
                    self._last_used_timestamp["left"] = left.timestamp
                    self._last_used_timestamp["right"] = right.timestamp
                    self._last_pair = (left, right)
                    self._last_pair_skew_s = abs(signed_skew)
                    self._pair_times.append(max(left.timestamp, right.timestamp))
                    li += 1
                    ri += 1
                elif signed_skew < 0.0:
                    # 左帧已经早到不可能再和当前或未来右帧配对。
                    self._last_used_timestamp["left"] = left.timestamp
                    li += 1
                else:
                    self._last_used_timestamp["right"] = right.timestamp
                    ri += 1

            while self._pair_times and now - self._pair_times[0] > 1.0:
                self._pair_times.popleft()
            pair_rate = 0.0
            if len(self._pair_times) >= 2:
                duration = self._pair_times[-1] - self._pair_times[0]
                if duration > 0.0:
                    pair_rate = (len(self._pair_times) - 1) / duration

            def raw_rate(side: str) -> float:
                recent = [
                    frame.timestamp
                    for frame in self.slots[side].frames
                    if now - frame.timestamp <= 1.0
                ]
                if len(recent) < 2 or recent[-1] <= recent[0]:
                    return 0.0
                return (len(recent) - 1) / (recent[-1] - recent[0])

            if self._last_pair is not None:
                self.detail = (
                    f"L{raw_rate('left'):.0f}/R{raw_rate('right'):.0f}原始 · "
                    f"配对{pair_rate:.0f} · 显示50 · Δ{self._last_pair_skew_s * 1000.0:.1f}ms"
                )

            # 可视化/控制观察节拍固定为 50 Hz；只在节拍到达时取最新真实配对，
            # 中间 render 复用同一 seq，FootRuntime 不会把它冒充成新采样。
            if now >= self._next_display_at:
                skipped = max(0, int((now - self._next_display_at) / self._display_period_s))
                self._next_display_at += (skipped + 1) * self._display_period_s
                if self._next_display_at < now or self._next_display_at - now > 0.1:
                    self._next_display_at = now + self._display_period_s
                self._display_pair = self._last_pair

            pair = self._display_pair
            if pair is None or now - min(pair[0].timestamp, pair[1].timestamp) > self._sync_grace_s:
                return {"left": None, "right": None}
            return {
                side: SensorFrame(
                    pair[index].xyz.copy(),
                    pair[index].temp_x10.copy(),
                    pair[index].timestamp,
                    pair[index].seq,
                )
                for index, side in enumerate(SIDES)
            }

    def slot_snapshot(self, side: str) -> DeviceSlot:
        with self.lock:
            slot = self.slots[side]
            return DeviceSlot(
                side=side,
                address=slot.address,
                name=slot.name,
                expected_name=slot.expected_name,
                status=slot.status,
                detail=slot.detail,
                connected=slot.connected,
                decoder=slot.decoder,
                latest=slot.latest,
                notification_count=slot.notification_count,
                valid_frame_count=slot.valid_frame_count,
                last_payload_len=slot.last_payload_len,
                last_valid_at=slot.last_valid_at,
            )

    def swap(self) -> None:
        """交换 UI 左右映射；连接回调仍跟随各自 DeviceSlot，不会串流。"""
        with self.lock:
            self.slots["left"], self.slots["right"] = (
                self.slots["right"],
                self.slots["left"],
            )
            self.slots["left"].side = "left"
            self.slots["right"].side = "right"
            self._last_pair = None
            self._display_pair = None
            self._last_used_timestamp = {"left": -math.inf, "right": -math.inf}
            self._pair_times.clear()
            self._next_display_at = 0.0

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._run())
        self._main_task = task
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.status = "双脚 BLE 线程异常"
            self.detail = str(exc)[:88]
        finally:
            pending = [item for item in asyncio.all_tasks(loop) if not item.done()]
            for item in pending:
                item.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._main_task = None
            self._loop = None

    async def _discover(self):
        from bleak import BleakScanner

        self.status = "正在扫描两只脚"
        self.detail = (
            f"寻找 {self.device_names['left']} / {self.device_names['right']} · 最长 12 秒"
        )
        discovered = await BleakScanner.discover(timeout=12.0, return_adv=True)
        matches = {side: [] for side in SIDES}
        for device, advertisement in discovered.values():
            name = (
                getattr(device, "name", "")
                or getattr(advertisement, "local_name", "")
                or ""
            ).strip()
            for side in SIDES:
                if name.casefold() == self.device_names[side].casefold():
                    matches[side].append(device)
        return {
            side: sorted(
                {getattr(d, "address", str(d)): d for d in matches[side]}.values(),
                key=lambda item: str(getattr(item, "address", item)).casefold(),
            )
            for side in SIDES
        }

    async def _run(self) -> None:
        try:
            from bleak import BleakClient
        except ImportError:
            self.status = "缺少 bleak"
            self.detail = "请执行 pip install bleak"
            return

        configured = [self.requested_addresses[side] for side in SIDES]
        if any(configured) and not all(configured):
            self.status = "地址配置不完整"
            self.detail = "left-address 与 right-address 需要同时提供"
            return

        if all(configured):
            targets = {
                side: self.requested_addresses[side]
                for side in SIDES
            }
        else:
            try:
                devices = await self._discover()
            except Exception as exc:
                self.status = "扫描失败"
                self.detail = str(exc)[:88]
                return
            if not any(devices.values()):
                self.status = "未发现足底传感器"
                self.detail = (
                    f"没有找到 {self.device_names['left']} / {self.device_names['right']}"
                )
                for slot in self.slots.values():
                    slot.status = "未发现设备"
                return
            targets = {
                side: devices[side][0]
                for side in SIDES
                if devices[side]
            }
            duplicates = [side for side in SIDES if len(devices[side]) > 1]
            if duplicates:
                self.status = "广播名冲突"
                self.detail = f"多个设备使用同一名称：{', '.join(duplicates)}"
                return
            if len(targets) < 2:
                missing = [side for side in SIDES if side not in targets]
                self.status = "只发现 1 只脚"
                self.detail = f"缺少唯一广播名：{', '.join(missing)}"

        self._expected_sides = {side for side in SIDES if targets.get(side) is not None}
        self._initial_ready.clear()
        self._notify_barrier = asyncio.Event()
        tasks = []
        for side in SIDES:
            target = targets.get(side)
            if target is None:
                self.slots[side].status = "等待另一只脚"
                continue
            tasks.append(
                asyncio.create_task(
                    self._connect_slot_with_retry(
                        self.slots[side],
                        target,
                        BleakClient,
                        self.requested_adapters[side],
                    )
                )
            )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_slot_with_retry(
        self, slot: DeviceSlot, target, client_cls, adapter: str = ""
    ) -> None:
        """Keep one physical foot on its configured adapter until shutdown."""
        attempt = 0
        while not self.stop_event.is_set():
            await self._connect_slot(slot, target, client_cls, adapter)
            if self.stop_event.is_set():
                return
            attempt += 1
            with self.lock:
                retry_in_progress = "InProgress" in slot.detail
                slot.status = "等待自动重连"
                if retry_in_progress:
                    slot.detail = f"蓝牙控制器忙 · 重试 {attempt}"
                elif not slot.detail:
                    slot.detail = f"连接中断 · 重试 {attempt}"
            await asyncio.sleep(min(1.0 + attempt * 0.35, 3.0))

    async def _connect_slot(
        self, slot: DeviceSlot, target, client_cls, adapter: str = ""
    ) -> None:
        address = getattr(target, "address", target)
        with self.lock:
            slot.address = str(address)
            slot.name = getattr(target, "name", None) or slot.expected_name
            slot.status = "正在连接"
            slot.detail = f"{slot.address} · {adapter}" if adapter else slot.address
        try:
            client_options = {}
            if adapter:
                if "bluez" in inspect.signature(client_cls).parameters:
                    client_options["bluez"] = {"adapter": adapter}
                else:
                    client_options["adapter"] = adapter
            async with client_cls(target, **client_options) as client:
                with self.lock:
                    slot.decoder.reset()
                    slot.frames.clear()
                    slot.connected = True
                    slot.status = "BLE 已连接"
                    slot.detail = (
                        f"{slot.address} · {adapter}" if adapter else slot.address
                    )
                connected_at = time.monotonic()

                # 等两路 GATT 都连好再同时开启 Notify，减少单侧先连造成的调度偏置。
                barrier = self._notify_barrier
                if barrier is not None and not barrier.is_set():
                    self._initial_ready.add(slot.side)
                    if self._initial_ready >= self._expected_sides:
                        barrier.set()
                    try:
                        await asyncio.wait_for(barrier.wait(), timeout=4.0)
                    except asyncio.TimeoutError:
                        barrier.set()

                def on_notify(_sender, payload: bytearray) -> None:
                    frames = slot.decoder.feed(bytes(payload))
                    now = time.monotonic()
                    with self.lock:
                        slot.notification_count += 1
                        slot.last_payload_len = len(payload)
                        if frames:
                            slot.valid_frame_count += len(frames)
                            slot.last_valid_at = now
                            slot.frames.extend(frames)
                            slot.latest = frames[-1]

                await client.start_notify(CHAR_UUID, on_notify)
                while not self.stop_event.is_set() and client.is_connected:
                    now = time.monotonic()
                    with self.lock:
                        if slot.valid_frame_count == 0 and now - connected_at > 1.8:
                            slot.status = "已连接·无有效数据"
                            slot.detail = (
                                f"Notify {slot.notification_count} · "
                                f"末包 {slot.last_payload_len} B"
                            )
                        elif slot.last_valid_at and now - slot.last_valid_at > 1.2:
                            slot.status = "BLE 数据暂停"
                            slot.detail = f"距末帧 {now - slot.last_valid_at:.1f} s"
                        elif slot.valid_frame_count:
                            slot.status = "BLE 实时数据"
                            slot.detail = f"有效帧 {slot.valid_frame_count}"
                    count = self.connected_count
                    self.status = "双脚 BLE 实时" if count == 2 else f"已连接 {count}/2"
                    if self._last_pair is None:
                        self.detail = "等待左右帧时间配对"
                    await asyncio.sleep(0.08)
                if client.is_connected:
                    await client.stop_notify(CHAR_UUID)
        except Exception as exc:
            with self.lock:
                slot.status = "连接中断"
                slot.detail = str(exc)[:72]
        finally:
            with self.lock:
                slot.connected = False


class DualDemoSource:
    """左右脚相差半个步态周期的离线界面诊断数据。"""

    is_demo = True

    def __init__(self) -> None:
        self.rng = np.random.default_rng(20260803)
        self.base = {
            side: self.rng.normal(0.0, 1450.0, (NUM_SENSORS, 3)).astype(np.float32)
            for side in SIDES
        }
        self.bias = {
            side: self.rng.uniform(0.86, 1.14, NUM_SENSORS).astype(np.float32)
            for side in SIDES
        }
        self.started_at = time.monotonic()
        self.last_emit = 0.0
        self.seq = 0
        self.latest = {side: None for side in SIDES}
        self.status = "双脚仿真数据"
        self.detail = "左右脚交替步态 · 50 Hz"

    @property
    def connected_count(self) -> int:
        return 2

    @property
    def connected(self) -> bool:
        return True

    def start(self) -> None:
        return

    def stop(self) -> bool:
        return True

    def swap(self) -> None:
        self.base["left"], self.base["right"] = self.base["right"], self.base["left"]
        self.bias["left"], self.bias["right"] = self.bias["right"], self.bias["left"]

    def slot_snapshot(self, side: str) -> DeviceSlot:
        return DeviceSlot(
            side=side,
            address=f"DEMO-{side.upper()}",
            status="仿真实时数据",
            detail="50 Hz",
            connected=True,
            valid_frame_count=self.seq,
        )

    def sample(self) -> dict[str, Optional[SensorFrame]]:
        now = time.monotonic()
        if now - self.last_emit < 1.0 / 50.0:
            return self.latest.copy()
        self.last_emit = now
        t = now - self.started_at
        self.seq += 1
        for side in SIDES:
            signal = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
            if t > 1.15:
                cycle = 1.62
                phase_shift = 0.0 if side == "left" else 0.50
                phase = ((t - 1.15) / cycle + phase_shift) % 1.0
                if phase < 0.58:
                    u = phase / 0.58
                    envelope = math.sin(math.pi * u) ** 0.35
                    center_y = 0.88 - 0.79 * (u * u * (3.0 - 2.0 * u))
                    longitudinal = np.exp(-((SENSOR_POS[:, 1] - center_y) / 0.22) ** 2)
                    intensity = 4300.0 * envelope * (0.22 + 0.78 * longitudinal) * self.bias[side]
                    lateral_sign = -1.0 if side == "left" else 1.0
                    signal[:, 0] = intensity * (
                        lateral_sign * 0.12 * np.sin(2.0 * math.pi * u)
                        + (SENSOR_POS[:, 0] - 0.5) * 0.20
                    )
                    signal[:, 1] = intensity * (0.17 - 0.29 * u)
                    signal[:, 2] = intensity * (
                        0.80 + 0.08 * np.cos(SENSOR_POS[:, 0] * 7.0)
                    )
            noise = self.rng.normal(0.0, 20.0, (NUM_SENSORS, 3)).astype(np.float32)
            xyz = np.rint(self.base[side] + signal + noise).astype(np.int32)
            temp = np.rint(
                279.0
                + (1.2 if side == "right" else 0.0)
                + 3.0 * np.sin(t * 0.1 + np.arange(NUM_SENSORS) * 0.27)
            ).astype(np.int32)
            self.latest[side] = SensorFrame(xyz, temp, now, self.seq)
        return self.latest.copy()


class FootRuntime:
    """一只脚独立的基线、滤波、指标、热图和 IDW 矢量状态。"""

    HEAT_W = 150
    HEAT_H = 430

    def __init__(self, side: str) -> None:
        self.side = side
        self.last_seq = -1
        self.last_frame_time = 0.0
        self.frame_intervals: deque[float] = deque(maxlen=80)
        self.raw_xyz = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.filtered = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.noise_sigma = np.full((NUM_SENSORS, 3), 18.0, dtype=np.float32)
        self.temp = np.full(NUM_SENSORS, np.nan, dtype=np.float32)
        self.baseline: Optional[np.ndarray] = None
        self.calibration_samples: list[np.ndarray] = []
        self.calibrating = True
        self.calibration_target = 75
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.metrics = Metrics()
        # 显式迟滞死区：小噪声不触发，已触发通道回到较低门限后才关闭。
        # 这些量仍是 Hall 原始 counts，不是力、压力或摩擦系数。
        self.deadzone_release_counts = 160.0
        self.deadzone_engage_counts = 280.0
        self.output_floor_counts = 100.0
        self.filter_alpha_rise = 0.22
        self.filter_alpha_fall = 0.34
        self.active_components = np.zeros((NUM_SENSORS, 3), dtype=bool)
        self._last_watchdog_at = time.monotonic()
        self.mag_history_short: deque[float] = deque(maxlen=55)
        self.history_peak: deque[float] = deque(maxlen=240)
        self.history_mean: deque[float] = deque(maxlen=240)
        self.last_history_at = 0.0

        yy, xx = np.mgrid[0 : self.HEAT_H, 0 : self.HEAT_W]
        self.grid_x = xx / (self.HEAT_W - 1)
        self.grid_y = yy / (self.HEAT_H - 1)
        self.foot_mask = self._build_foot_mask()
        kernels = []
        for sx, sy in SENSOR_POS:
            dist2 = (self.grid_x - sx) ** 2 + (self.grid_y - sy) ** 2
            kernels.append(np.exp(-dist2 / (2.0 * 0.145 * 0.145)))
        kernel_stack = np.asarray(kernels, dtype=np.float32)
        kernel_sum = np.maximum(np.sum(kernel_stack, axis=0), 1e-5)
        self.heat_weights = (
            kernel_stack / kernel_sum[None, :, :] * self.foot_mask[None, :, :]
        ).astype(np.float32)
        self.force_field_ema = np.zeros((self.HEAT_H, self.HEAT_W), dtype=np.float32)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self._heat_surface_cache: Optional[pygame.Surface] = None
        self._heat_scale_cache = math.nan

        self.vector_grid_size: Optional[tuple[int, int]] = None
        self.vector_points = np.zeros((0, 2), dtype=np.float32)
        self.vector_weights = np.zeros((0, NUM_SENSORS), dtype=np.float32)
        self.vector_valid = np.zeros((0,), dtype=bool)
        self.vector_prev = np.zeros((0, 2), dtype=np.float32)
        self.vector_values = np.zeros((0, 2), dtype=np.float32)
        self.vector_visible = np.zeros((0,), dtype=bool)
        self.vector_seq = -2

    def _build_foot_mask(self) -> np.ndarray:
        surface = pygame.Surface((self.HEAT_W, self.HEAT_H))
        surface.fill((0, 0, 0))
        points = [
            (round(float(x) * (self.HEAT_W - 1)), round(float(y) * (self.HEAT_H - 1)))
            for x, y in INSOLE_OUTLINE
        ]
        pygame.draw.polygon(surface, (255, 255, 255), points)
        return pygame.surfarray.array2d(surface).T != 0

    def request_calibration(self) -> None:
        self.calibration_samples.clear()
        self.calibrating = True
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.baseline = None
        self.filtered.fill(0.0)
        self.active_components.fill(False)
        self.force_field_ema.fill(0.0)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self._heat_surface_cache = None
        self._heat_scale_cache = math.nan
        self.vector_prev.fill(0.0)
        self.vector_values.fill(0.0)
        self.vector_visible.fill(False)
        self.vector_seq = -2

    def finish_calibration(self) -> None:
        if not self.calibration_samples:
            return
        samples = np.stack(self.calibration_samples, axis=0)
        self.baseline = np.median(samples, axis=0).astype(np.float32)
        # MAD 对偶发按压/丢包尖峰更稳健，避免短暂异常把死区永久抬高。
        mad = np.median(np.abs(samples - self.baseline[None, :, :]), axis=0)
        self.noise_sigma = np.maximum(1.4826 * mad, 1.0).astype(np.float32)
        self.calibration_samples.clear()
        self.calibrating = False

    def update_watchdog(self, *, paused: bool = False) -> None:
        now = time.monotonic()
        if self.calibrating:
            if (
                self.calibration_last_sample_at > 0.0
                and now - self.calibration_last_sample_at >= 0.50
            ):
                # 断流/重连不能用残缺空载样本生成基线。清除此轮数据，恢复后
                # 必须重新收满一个连续稳定窗口。
                self.calibration_samples.clear()
                self.calibration_first_sample_at = 0.0
                self.calibration_last_sample_at = 0.0
            self._last_watchdog_at = now
            return

        dt = min(max(now - self._last_watchdog_at, 0.0), 0.1)
        self._last_watchdog_at = now
        if paused or self.last_frame_time <= 0.0:
            return
        stale_for = now - self.last_frame_time
        if stale_for <= 0.12:
            return

        # 断流时不保留最后一个伪“接触”状态：约 0.2 s 平滑退回零，避免
        # 无数据或重连间隙中的热图/箭头乱跳。该衰减仅用于显示端。
        retention = math.exp(-dt / 0.085)
        self.filtered *= retention
        self.force_field_ema *= retention
        self.force_field_seq = -2
        self._heat_surface_cache = None
        self._heat_scale_cache = math.nan
        self.vector_prev *= retention
        self.vector_values *= retention
        self.filtered[np.abs(self.filtered) < 4.0] = 0.0
        if stale_for >= 0.50:
            self.filtered.fill(0.0)
            self.force_field_ema.fill(0.0)
            self.vector_prev.fill(0.0)
            self.vector_values.fill(0.0)
            self.vector_visible.fill(False)
            self.active_components.fill(False)
            self.frame_intervals.clear()
        self._update_metrics(record_sample=False)

    def process(self, frame: Optional[SensorFrame], *, paused: bool) -> None:
        if frame is None or frame.seq == self.last_seq or paused:
            return
        self.last_seq = frame.seq
        if self.last_frame_time > 0.0:
            dt = frame.timestamp - self.last_frame_time
            if 0.001 < dt < 1.0:
                self.frame_intervals.append(dt)
        self.last_frame_time = frame.timestamp
        self.raw_xyz = frame.xyz.astype(np.float32)
        temp_c = frame.temp_x10.astype(np.float32) / 10.0
        temp_c[(temp_c < -40.0) | (temp_c > 125.0)] = np.nan
        self.temp = temp_c

        if self.calibrating:
            now = time.monotonic()
            if not self.calibration_samples:
                self.calibration_first_sample_at = now
            self.calibration_last_sample_at = now
            self.calibration_samples.append(self.raw_xyz.copy())
            elapsed = now - self.calibration_first_sample_at
            enough = len(self.calibration_samples) >= self.calibration_target and elapsed >= 1.25
            low_rate = len(self.calibration_samples) >= 45 and elapsed >= 2.00
            if enough or low_rate:
                self.finish_calibration()
            return
        if self.baseline is None:
            return

        delta = self.raw_xyz - self.baseline
        release_zone = np.clip(
            np.maximum(self.deadzone_release_counts, self.noise_sigma * 3.2),
            self.deadzone_release_counts,
            220.0,
        )
        engage_zone = np.clip(
            np.maximum(self.deadzone_engage_counts, self.noise_sigma * 5.0),
            self.deadzone_engage_counts,
            320.0,
        )
        absolute = np.abs(delta)
        self.active_components |= absolute >= engage_zone
        self.active_components &= absolute > release_zone
        target = np.where(
            self.active_components,
            np.sign(delta) * np.maximum(absolute - release_zone, 0.0),
            0.0,
        )
        below_output_floor = np.abs(target) < self.output_floor_counts
        target[below_output_floor] = 0.0
        self.active_components[below_output_floor] = False
        same_direction = np.sign(target) == np.sign(self.filtered)
        growing = same_direction & (np.abs(target) > np.abs(self.filtered))
        alpha = np.where(
            growing,
            self.filter_alpha_rise,
            self.filter_alpha_fall,
        )
        self.filtered += alpha * (target - self.filtered)
        quiet_components = np.abs(target) < self.output_floor_counts
        self.filtered[quiet_components] *= 0.58
        self.filtered[np.abs(self.filtered) < 4.0] = 0.0

        # 基线只能由明确的空载校准建立，运行时保持不变。Hall 本身无法区分
        # “持续静态加载”和“卸载后的 TPU/磁片回程残余”；任何自动追基线都会
        # 在某些情况下把真实持续响应吃掉。需要清除真实残余时，确认双脚完全
        # 卸载后按 B 重新采集空载基线。原始 BLE Bx/By/Bz 始终不被修改。
        self._update_metrics()

    def _update_metrics(self, *, record_sample: bool = True) -> None:
        mags = np.linalg.norm(self.filtered, axis=1)
        peak = float(np.max(mags))
        active_threshold = max(90.0, peak * 0.09)
        total = float(np.sum(mags))
        weights = mags + 1e-6
        cop = (
            float(np.sum(SENSOR_POS[:, 0] * weights) / np.sum(weights)),
            float(np.sum(SENSOR_POS[:, 1] * weights) / np.sum(weights)),
        )
        region_raw = [float(np.sum(mags[idx])) for idx in REGION_INDICES]
        region_sum = max(sum(region_raw), 1e-6)
        regions = tuple(value / region_sum for value in region_raw)
        if record_sample:
            self.mag_history_short.append(total)
        if len(self.mag_history_short) >= 8:
            arr = np.asarray(self.mag_history_short)
            stability = float(np.clip(100.0 - np.std(arr) / max(np.mean(arr), 1.0) * 85.0, 0.0, 100.0))
        else:
            stability = 100.0
        valid_temp = self.temp[np.isfinite(self.temp)]
        mean_temp = float(np.mean(valid_temp)) if valid_temp.size else float("nan")
        min_temp = float(np.min(valid_temp)) if valid_temp.size else float("nan")
        hz = 1.0 / float(np.mean(self.frame_intervals)) if self.frame_intervals else 0.0
        components = tuple(float(v) for v in np.mean(self.filtered, axis=0))
        self.metrics = Metrics(
            peak=peak,
            active=int(np.count_nonzero(mags >= active_threshold)),
            mean_temp=mean_temp,
            min_temp=min_temp,
            hz=hz,
            stability=stability,
            hall_components=components,
            cop=cop,
            region_loads=regions,
        )
        now = time.monotonic()
        if record_sample and now - self.last_history_at >= 0.08:
            self.last_history_at = now
            self.history_peak.append(peak)
            self.history_mean.append(total / NUM_SENSORS)

    @property
    def load(self) -> float:
        return float(np.sum(np.linalg.norm(self.filtered, axis=1)))

    def build_heat_surface(self, display_vmax: Optional[float] = None) -> pygame.Surface:
        values_raw = np.linalg.norm(self.filtered, axis=1)
        scale_changed = (
            display_vmax is not None
            and (
                not math.isfinite(self._heat_scale_cache)
                or abs(float(display_vmax) - self._heat_scale_cache) > 1.0
            )
        )
        is_new = self.force_field_seq != self.last_seq or scale_changed
        if not is_new and self._heat_surface_cache is not None:
            return self._heat_surface_cache
        if display_vmax is None:
            target_vmax = max(850.0, float(np.percentile(values_raw, 95)) * 1.12)
            # 独立显示模式采用当前帧即时量程，避免历史峰值导致颜色长期不恢复。
            self.force_vmax = target_vmax
        else:
            # 双脚默认共用同一个当前帧 counts 色标，不做左右隐式增益。
            self.force_vmax = float(display_vmax)
        self.force_vmax = float(np.clip(self.force_vmax, 850.0, 9000.0))
        self._heat_scale_cache = self.force_vmax
        values = np.clip(values_raw / self.force_vmax, 0.0, 1.0) ** 0.72
        field = np.tensordot(values, self.heat_weights, axes=(0, 0)).astype(np.float32)
        mask_f = self.foot_mask.astype(np.float32)
        pf = np.pad(field, 1, mode="constant")
        pm = np.pad(mask_f, 1, mode="constant")
        numerator = (
            pf[1:-1, 1:-1] * 4.0 + pf[:-2, 1:-1] + pf[2:, 1:-1]
            + pf[1:-1, :-2] + pf[1:-1, 2:]
        )
        denominator = (
            pm[1:-1, 1:-1] * 4.0 + pm[:-2, 1:-1] + pm[2:, 1:-1]
            + pm[1:-1, :-2] + pm[1:-1, 2:]
        )
        field = np.where(self.foot_mask, numerator / np.maximum(denominator, 1e-5), 0.0)
        if is_new:
            heat_alpha = np.where(field > self.force_field_ema, 0.24, 0.38)
            self.force_field_ema += heat_alpha * (field - self.force_field_ema)
            self.force_field_seq = self.last_seq
        field = self.force_field_ema

        stop_p = np.array([0.0, 0.18, 0.40, 0.62, 0.82, 1.0])
        stop_c = np.array(
            [
                (235, 243, 244), (190, 220, 224), (111, 185, 191),
                (226, 207, 123), (231, 143, 96), (210, 85, 98),
            ],
            dtype=np.float32,
        )
        rgb = np.zeros((self.HEAT_H, self.HEAT_W, 3), dtype=np.uint8)
        for channel in range(3):
            rgb[:, :, channel] = np.interp(field, stop_p, stop_c[:, channel]).astype(np.uint8)
        alpha = np.where(self.foot_mask, 246, 0).astype(np.uint8)
        surface = pygame.Surface((self.HEAT_W, self.HEAT_H), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = alpha.T
        self._heat_surface_cache = surface
        return surface

    def outline_surface(self) -> pygame.Surface:
        surface = pygame.Surface((self.HEAT_W, self.HEAT_H), pygame.SRCALPHA)
        mask = self.foot_mask
        edge = mask & (
            ~np.roll(mask, 1, axis=0) | ~np.roll(mask, -1, axis=0)
            | ~np.roll(mask, 1, axis=1) | ~np.roll(mask, -1, axis=1)
        )
        rgba = np.zeros((self.HEAT_H, self.HEAT_W, 4), dtype=np.uint8)
        rgba[edge] = (*C.CYAN_2, 220)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        return surface

    def _inside_foot(self, x: float, y: float) -> bool:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        ix = int(np.clip(x * (self.HEAT_W - 1), 0, self.HEAT_W - 1))
        iy = int(np.clip(y * (self.HEAT_H - 1), 0, self.HEAT_H - 1))
        return bool(self.foot_mask[iy, ix])

    def _prepare_vectors(self, rect: pygame.Rect) -> None:
        size = (rect.w, rect.h)
        if self.vector_grid_size == size:
            return
        step = 25.0
        xs = np.arange(step / 2.0, float(rect.w), step, dtype=np.float32)
        ys = np.arange(step / 2.0, float(rect.h), step, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1).astype(np.float32)
        sensor_pixels = SENSOR_POS * np.array([rect.w, rect.h], dtype=np.float32)
        diff = points[:, None, :] - sensor_pixels[None, :, :]
        weights = 1.0 / (np.sum(diff * diff, axis=2) + 36.0)
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
        valid = np.array(
            [self._inside_foot(float(x / rect.w), float(y / rect.h)) for x, y in points],
            dtype=bool,
        )
        count = len(points)
        self.vector_grid_size = size
        self.vector_points = points
        self.vector_weights = weights.astype(np.float32)
        self.vector_valid = valid
        self.vector_prev = np.zeros((count, 2), dtype=np.float32)
        self.vector_values = np.zeros((count, 2), dtype=np.float32)
        self.vector_visible = np.zeros((count,), dtype=bool)
        self.vector_seq = -2

    def _sensor_xy(self) -> np.ndarray:
        delta_xy = -self.filtered[:, :2]
        cos_r = np.cos(CHIP_XY_ROTATIONS)
        sin_r = np.sin(CHIP_XY_ROTATIONS)
        return np.column_stack(
            (
                cos_r * delta_xy[:, 0] - sin_r * delta_xy[:, 1],
                sin_r * delta_xy[:, 0] + cos_r * delta_xy[:, 1],
            )
        ).astype(np.float32)

    def draw_vectors(self, canvas: pygame.Surface, rect: pygame.Rect, *, mirrored: bool) -> None:
        self._prepare_vectors(rect)
        vectors = self.vector_weights @ self._sensor_xy()
        if self.vector_seq != self.last_seq:
            magnitude2 = np.sum(vectors * vectors, axis=1)
            changes = np.linalg.norm(vectors - self.vector_prev, axis=1)
            raw_on = (magnitude2 >= 64.0) | (changes >= 4.0)
            raw_off = (magnitude2 < 36.0) & (changes < 1.4)
            self.vector_visible = np.where(self.vector_visible, ~raw_off, raw_on)
            self.vector_prev[:] = vectors
            self.vector_values[:] = vectors
            self.vector_seq = self.last_seq

        for point, valid in zip(self.vector_points, self.vector_valid):
            if not valid:
                continue
            px = rect.w - point[0] if mirrored else point[0]
            pygame.draw.circle(canvas, (74, 142, 122), (round(rect.x + px), round(rect.y + point[1])), 2)

        for point, vector, valid, visible in zip(
            self.vector_points, self.vector_values, self.vector_valid, self.vector_visible
        ):
            if not valid or not visible:
                continue
            px = rect.w - point[0] if mirrored else point[0]
            display = vector / 40.0
            if mirrored:
                display = display * np.array([-1.0, 1.0], dtype=np.float32)
            length = float(np.linalg.norm(display))
            if length > 18.0:
                display *= 18.0 / length
            start = np.array([rect.x + px, rect.y + point[1]], dtype=np.float32)
            end = start + display
            start_px = (round(float(start[0])), round(float(start[1])))
            end_px = (round(float(end[0])), round(float(end[1])))
            pygame.draw.line(canvas, (47, 158, 112), start_px, end_px, 2)
            pygame.draw.circle(canvas, (47, 158, 112), end_px, 2)


class DualDashboard:
    def __init__(
        self,
        source: DualBLESource | DualDemoSource,
        *,
        left_address: Optional[str] = None,
        right_address: Optional[str] = None,
        left_adapter: Optional[str] = None,
        right_adapter: Optional[str] = None,
        left_name: str = "left",
        right_name: str = "right",
        screenshot_path: Optional[str] = None,
        screenshot_frame: int = 150,
    ) -> None:
        pygame.init()
        pygame.display.set_caption(APP_TITLE)
        self.screen = pygame.display.set_mode(LOGICAL_SIZE, pygame.RESIZABLE)
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.source = source
        self.left_address = left_address
        self.right_address = right_address
        self.left_adapter = left_adapter
        self.right_adapter = right_adapter
        self.left_name = left_name
        self.right_name = right_name
        self.feet = {side: FootRuntime(side) for side in SIDES}
        self.shared_hall_vmax = 850.0
        self.running = True
        self.paused = False
        self.show_heat = True
        self.show_vectors = True
        self.show_ids = True
        self.show_cop = False
        self.shared_heat_scale = False
        self.mouse_logical = (-100, -100)
        self.toast_text = ""
        self.toast_until = 0.0
        self.screenshot_path = screenshot_path
        self.screenshot_frame = max(5, screenshot_frame)
        self.rendered_frames = 0
        self.saved_requested_shot: Optional[Path] = None
        self.recorder = VideoRecorder(LOGICAL_SIZE, fps=30)

        self.buttons = {
            "connect": Button((790, 21, 112, 42), "重新连接", "D"),
            "swap": Button((912, 21, 104, 42), "交换左右", "X"),
            "calibrate": Button((1026, 21, 88, 42), "空载归零", "B"),
            "pause": Button((1124, 21, 88, 42), "暂停", "SP"),
            "record": Button((1222, 21, 92, 42), "录制", "F9"),
            "shot": Button((1324, 21, 92, 42), "截图", "S"),
        }
        self.brand_logo: Optional[pygame.Surface] = None
        logo_path = Path(__file__).resolve().parent / "assets" / "mosense_logo.png"
        try:
            self.brand_logo = pygame.image.load(str(logo_path)).convert()
        except (pygame.error, OSError):
            pass

    def toast(self, text: str, duration: float = 2.4) -> None:
        self.toast_text = text
        self.toast_until = time.monotonic() + duration

    def request_calibration(self) -> None:
        for foot in self.feet.values():
            foot.request_calibration()
        self.toast("正在同步采集固定空载基线：请确认两只脚完全卸载", 3.5)

    def reconnect(self) -> None:
        if not self.source.stop():
            self.toast("旧 BLE 连接尚未释放，已取消本次重连", 3.5)
            return
        self.source = DualBLESource(
            left_address=self.left_address,
            right_address=self.right_address,
            left_adapter=self.left_adapter,
            right_adapter=self.right_adapter,
            left_name=self.left_name,
            right_name=self.right_name,
        )
        self.source.start()
        self.feet = {side: FootRuntime(side) for side in SIDES}
        self.toast("正在扫描并连接两只脚", 3.0)

    def swap_feet(self) -> None:
        self.source.swap()
        self.feet["left"], self.feet["right"] = self.feet["right"], self.feet["left"]
        self.feet["left"].side = "left"
        self.feet["right"].side = "right"
        self.toast("已交换左右脚映射")

    def toggle_heat_scale(self) -> None:
        self.shared_heat_scale = not self.shared_heat_scale
        for foot in self.feet.values():
            foot.force_field_seq = -2
            foot._heat_surface_cache = None
            foot._heat_scale_cache = math.nan
        mode = "共享 counts 色标，可比较绝对响应" if self.shared_heat_scale else "每脚独立即时色标，仅比较分布形态"
        self.toast(mode, 3.2)

    def save_screenshot(self, path: Optional[Path] = None) -> Path:
        if path is None:
            folder = Path.cwd() / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"footsense_dual_{datetime.now():%Y%m%d_%H%M%S}.png"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.canvas, str(path))
        self.saved_requested_shot = path
        self.toast(f"截图已保存：{path.name}")
        return path

    def toggle_recording(self) -> None:
        if self.recorder.active:
            path = self.recorder.stop()
            if path is not None:
                self.toast(f"视频已保存：{path.name}", 3.0)
            return
        folder = Path.cwd() / "recordings"
        path = folder / f"footsense_dual_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        if self.recorder.start(path):
            self.toast("开始录制双脚画面")
        else:
            self.toast(self.recorder.error or "无法开始录制", 3.5)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                width = max(MIN_WINDOW_SIZE[0], event.w)
                height = max(MIN_WINDOW_SIZE[1], event.h)
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_d:
                    self.reconnect()
                elif event.key == pygame.K_x:
                    self.swap_feet()
                elif event.key == pygame.K_b:
                    self.request_calibration()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.toast("画面已暂停" if self.paused else "画面继续")
                elif event.key == pygame.K_h:
                    self.show_heat = not self.show_heat
                elif event.key == pygame.K_m:
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_c:
                    self.show_cop = not self.show_cop
                elif event.key == pygame.K_g:
                    self.toggle_heat_scale()
                elif event.key == pygame.K_r:
                    for foot in self.feet.values():
                        foot.history_peak.clear()
                        foot.history_mean.clear()
                elif event.key == pygame.K_F9:
                    self.toggle_recording()
                elif event.key == pygame.K_s:
                    self.save_screenshot()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = self._screen_to_logical(event.pos)
                if self.buttons["connect"].rect.collidepoint(p):
                    self.reconnect()
                elif self.buttons["swap"].rect.collidepoint(p):
                    self.swap_feet()
                elif self.buttons["calibrate"].rect.collidepoint(p):
                    self.request_calibration()
                elif self.buttons["pause"].rect.collidepoint(p):
                    self.paused = not self.paused
                elif self.buttons["record"].rect.collidepoint(p):
                    self.toggle_recording()
                elif self.buttons["shot"].rect.collidepoint(p):
                    self.save_screenshot()
        self.mouse_logical = self._screen_to_logical(pygame.mouse.get_pos())

    def _screen_to_logical(self, pos: tuple[int, int]) -> tuple[int, int]:
        sw, sh = self.screen.get_size()
        scale = min(sw / LOGICAL_SIZE[0], sh / LOGICAL_SIZE[1])
        out_w, out_h = LOGICAL_SIZE[0] * scale, LOGICAL_SIZE[1] * scale
        ox, oy = (sw - out_w) / 2.0, (sh - out_h) / 2.0
        return (
            int((pos[0] - ox) / max(scale, 1e-6)),
            int((pos[1] - oy) / max(scale, 1e-6)),
        )

    def process(self) -> None:
        samples = self.source.sample()
        for side in SIDES:
            self.feet[side].process(samples.get(side), paused=self.paused)
            self.feet[side].update_watchdog(paused=self.paused)
        # 即时共享色标参考：参考 detail 版本的 VMAX_ALPHA=1 思路，避免某只脚
        # 历史超量程把自己的色标长期撑大，造成左右“权重不同”的错觉。
        all_magnitudes = np.concatenate(
            [np.linalg.norm(self.feet[side].filtered, axis=1) for side in SIDES]
        )
        self.shared_hall_vmax = float(
            np.clip(max(850.0, float(np.percentile(all_magnitudes, 95)) * 1.12), 850.0, 9000.0)
        )

    def _balance(self) -> tuple[float, float, float]:
        left = self.feet["left"].load
        right = self.feet["right"].load
        total = left + right
        if total < 1.0:
            return 0.5, 0.5, 100.0
        left_ratio = left / total
        right_ratio = right / total
        symmetry = max(0.0, 100.0 * (1.0 - abs(left - right) / total))
        return left_ratio, right_ratio, symmetry

    def draw_header(self) -> None:
        pygame.draw.line(self.canvas, C.BORDER, (24, 78), (1416, 78), 1)
        logo_rect = pygame.Rect(24, 9, 198, 60)
        rounded_rect(self.canvas, logo_rect, (33, 31, 31), 10)
        if self.brand_logo is not None:
            logo = pygame.transform.smoothscale(self.brand_logo, (189, 60))
            self.canvas.blit(logo, logo.get_rect(center=logo_rect.center))
        else:
            blit_text(self.canvas, self.fonts.subtitle, "模感科技", C.WHITE, logo_rect.center, anchor="center")
        blit_text(self.canvas, self.fonts.title, "双足 Hall 磁场监测", C.TEXT, (242, 20))
        blit_text(self.canvas, self.fonts.tiny, f"MoSense Technology · {APP_VERSION}", C.TEXT_3, (242, 53))

        status_rect = pygame.Rect(600, 21, 180, 42)
        rounded_rect(self.canvas, status_rect, C.PANEL, 11, C.BORDER)
        count = self.source.connected_count
        dot = C.GREEN if count == 2 else (C.YELLOW if count == 1 else C.RED)
        pygame.draw.circle(self.canvas, dot, (618, 42), 5)
        status = "双脚已连接" if count == 2 else f"设备 {count}/2"
        if self.source.is_demo:
            status = "双脚仿真"
        blit_text(self.canvas, self.fonts.small, status, C.TEXT, (631, 31))
        self.buttons["pause"].text = "继续" if self.paused else "暂停"
        self.buttons["record"].text = "停止" if self.recorder.active else "录制"
        for button in self.buttons.values():
            button.draw(self.canvas, self.fonts, self.mouse_logical)

    def draw_metric_cards(self) -> None:
        left = self.feet["left"].metrics
        right = self.feet["right"].metrics
        left_ratio, right_ratio, symmetry = self._balance()
        hz_values = [value for value in (left.hz, right.hz) if value > 0.0]
        hz = min(hz_values) if hz_values else 0.0
        cards = [
            ("左脚 Hall 峰值", f"{left.peak:,.0f}", "counts", C.BLUE, "左脚 15 点磁场变化模长"),
            ("右脚 Hall 峰值", f"{right.peak:,.0f}", "counts", C.ORANGE, "右脚 15 点磁场变化模长"),
            ("左右响应", f"{left_ratio * 100:.0f}:{right_ratio * 100:.0f}", "%", C.CYAN, "左 : 右 相对 Hall 响应"),
            ("响应对称", f"{symmetry:.0f}", "%", C.GREEN, "基于双脚 Hall 总响应差异"),
            (
                "节拍 / 新鲜帧",
                f"50/{hz:.0f}",
                "Hz",
                C.MAGENTA,
                self.source.detail,
            ),
        ]
        x0, y, gap, width, height = 24, 91, 12, 268, 83
        for i, (label, value, unit, accent, hint) in enumerate(cards):
            rect = pygame.Rect(x0 + i * (width + gap), y, width, height)
            rounded_rect(self.canvas, rect, C.PANEL, 13, C.BORDER)
            pygame.draw.rect(self.canvas, accent, (rect.x, rect.y + 12, 3, rect.h - 24), border_radius=2)
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_2, (rect.x + 17, rect.y + 9))
            value_rect = blit_text(self.canvas, self.fonts.metric, value, C.TEXT, (rect.x + 17, rect.y + 25))
            blit_text(self.canvas, self.fonts.tiny, unit, accent, (value_rect.right + 8, rect.y + 43), anchor="midleft")
            blit_text(self.canvas, self.fonts.tiny, hint[:34], C.TEXT_3, (rect.x + 17, rect.bottom - 17))

    def _draw_region_bars(self, side: str, x: int, y: int, width: int) -> None:
        foot = self.feet[side]
        colors = (C.ORANGE, C.CYAN, C.BLUE)
        blit_text(self.canvas, self.fonts.tiny, "区域响应占比", C.TEXT_2, (x, y))
        for i, (name, value, color) in enumerate(zip(REGION_NAMES, foot.metrics.region_loads, colors)):
            yy = y + 31 + i * 58
            blit_text(self.canvas, self.fonts.tiny, name, C.TEXT_2, (x, yy))
            blit_text(self.canvas, self.fonts.tiny, f"{value * 100:4.1f}%", C.TEXT, (x + width, yy), anchor="topright")
            track = pygame.Rect(x, yy + 24, width, 6)
            pygame.draw.rect(self.canvas, C.GRID, track, border_radius=4)
            fill = track.copy()
            fill.width = max(2, round(track.w * value))
            pygame.draw.rect(self.canvas, color, fill, border_radius=4)

    def _draw_one_foot(self, side: str, rect: pygame.Rect) -> None:
        foot = self.feet[side]
        mirrored = side == "left"
        if self.show_heat:
            display_vmax = self.shared_hall_vmax if self.shared_heat_scale else None
            surface = foot.build_heat_surface(display_vmax)
        else:
            surface = pygame.Surface((foot.HEAT_W, foot.HEAT_H), pygame.SRCALPHA)
            rgba = np.zeros((foot.HEAT_H, foot.HEAT_W, 4), dtype=np.uint8)
            rgba[foot.foot_mask] = (241, 246, 244, 255)
            pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
            pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        outline = foot.outline_surface()
        if mirrored:
            surface = pygame.transform.flip(surface, True, False)
            outline = pygame.transform.flip(outline, True, False)
        self.canvas.blit(pygame.transform.smoothscale(surface, rect.size), rect.topleft)
        if self.show_vectors:
            foot.draw_vectors(self.canvas, rect, mirrored=mirrored)
        self.canvas.blit(pygame.transform.smoothscale(outline, rect.size), rect.topleft)

        mags = np.linalg.norm(foot.filtered, axis=1)
        peak_ref = max(3000.0, float(np.max(mags)))
        display_positions = SENSOR_POS.copy()
        if mirrored:
            display_positions[:, 0] = 1.0 - display_positions[:, 0]
        # 三组真实十字布局：上、左、中、右、下；连线仅表示 PCB 点位关系。
        for group_start in (0, 5, 10):
            top, left, center, right, bottom = display_positions[group_start : group_start + 5]
            pixel = lambda point: (
                round(rect.x + float(point[0]) * rect.w),
                round(rect.y + float(point[1]) * rect.h),
            )
            pygame.draw.aaline(self.canvas, C.BORDER, pixel(top), pixel(bottom))
            pygame.draw.aaline(self.canvas, C.BORDER, pixel(left), pixel(right))

        for output_index, ((nx, ny), mag) in enumerate(zip(SENSOR_POS, mags)):
            display_x = 1.0 - float(nx) if mirrored else float(nx)
            x = rect.x + display_x * rect.w
            y = rect.y + float(ny) * rect.h
            strength = float(np.clip(mag / peak_ref, 0.0, 1.0))
            color = mix_color(C.CYAN_2, C.ORANGE, strength)
            pygame.draw.circle(self.canvas, C.WHITE, (round(x), round(y)), 5)
            pygame.draw.circle(self.canvas, C.INK, (round(x), round(y)), 4, 1)
            pygame.draw.circle(self.canvas, color, (round(x), round(y)), 2)
            if self.show_ids:
                label_x = round(x) - 7 if mirrored else round(x) + 7
                anchor = "topright" if mirrored else "topleft"
                blit_text(
                    self.canvas,
                    self.fonts.tiny,
                    f"P{output_index:02d}",
                    C.TEXT,
                    (label_x, round(y) - 8),
                    anchor=anchor,
                )
        if self.show_cop and foot.load > 1.0:
            cop_x, cop_y = foot.metrics.cop
            display_x = 1.0 - cop_x if mirrored else cop_x
            x = rect.x + display_x * rect.w
            y = rect.y + cop_y * rect.h
            pygame.draw.circle(self.canvas, C.INK, (round(x), round(y)), 9, 1)
            pygame.draw.line(self.canvas, C.INK, (x - 11, y), (x + 11, y), 1)
            pygame.draw.line(self.canvas, C.INK, (x, y - 11), (x, y + 11), 1)

    def draw_feet_card(self) -> None:
        card = pygame.Rect(24, 190, 1008, 668)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "左右脚同步 Hall 磁场", C.TEXT, (44, 205))
        scale_text = (
            "左右共享 counts 色标（可比较绝对响应）"
            if self.shared_heat_scale
            else "每脚独立即时色标（仅比较各自分布）"
        )
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"Hall 响应热图 + 15 点磁场矢量 · {scale_text} · A4 1:1",
            C.TEXT_3,
            (44, 231),
        )
        pygame.draw.line(self.canvas, C.BORDER, (528, 252), (528, 838), 1)

        foot_rects = {
            "left": pygame.Rect(195, 300, 198, 520),
            "right": pygame.Rect(651, 300, 198, 520),
        }
        for side in SIDES:
            foot = self.feet[side]
            slot = self.source.slot_snapshot(side)
            x = 44 if side == "left" else 550
            accent = C.BLUE if side == "left" else C.ORANGE
            pygame.draw.circle(self.canvas, accent, (x + 6, 274), 5)
            blit_text(self.canvas, self.fonts.subtitle, SIDE_CN[side], C.TEXT, (x + 18, 260))
            identity = slot.name or slot.expected_name
            address = slot.address or "等待发现设备"
            blit_text(
                self.canvas,
                self.fonts.tiny,
                f"{identity} · {address}",
                C.TEXT_3,
                (x + 18, 286),
            )
            self._draw_one_foot(side, foot_rects[side])
            bars_x = 45 if side == "left" else 872
            self._draw_region_bars(side, bars_x, 350, 118)
            temp = "--" if not np.isfinite(foot.metrics.min_temp) else f"{foot.metrics.min_temp:.1f} °C"
            blit_text(self.canvas, self.fonts.tiny, "传感器温度", C.TEXT_3, (bars_x, 565))
            blit_text(self.canvas, self.fonts.small, temp, C.TEXT, (bars_x, 588))
            blit_text(self.canvas, self.fonts.tiny, "活跃通道", C.TEXT_3, (bars_x, 630))
            blit_text(self.canvas, self.fonts.small, f"{foot.metrics.active} / {NUM_SENSORS}", C.TEXT, (bars_x, 653))
            if foot.calibrating:
                progress = len(foot.calibration_samples)
                pill = pygame.Rect(x + 18, 810, 130, 25)
                rounded_rect(self.canvas, pill, C.PANEL_2, 8, C.YELLOW)
                blit_text(self.canvas, self.fonts.tiny, f"校准 {progress}/{foot.calibration_target}", C.TEXT_2, pill.center, anchor="center")
            else:
                pill = pygame.Rect(x + 18, 810, 130, 25)
                rounded_rect(self.canvas, pill, C.PANEL_2, 8, C.GREEN)
                blit_text(self.canvas, self.fonts.tiny, "空载基线已锁定", C.TEXT_2, pill.center, anchor="center")

        blit_text(self.canvas, self.fonts.tiny, "低", C.TEXT_3, (449, 307))
        for i in range(140):
            pygame.draw.line(self.canvas, force_color(i / 139.0), (449, 329 + i), (458, 329 + i))
        blit_text(self.canvas, self.fonts.tiny, "高", C.TEXT_3, (449, 476))
        pygame.draw.circle(self.canvas, (74, 142, 122), (453, 543), 2)
        blit_text(self.canvas, self.fonts.tiny, "IDW", C.TEXT_3, (467, 534))
        blit_text(self.canvas, self.fonts.tiny, "网格", C.TEXT_3, (467, 552))

    def draw_balance_card(self) -> None:
        card = pygame.Rect(1048, 190, 368, 224)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "双脚 Hall 响应平衡", C.TEXT, (1066, 205))
        left_ratio, right_ratio, symmetry = self._balance()
        blit_text(self.canvas, self.fonts.hero, f"{left_ratio * 100:.0f}", C.BLUE, (1080, 246))
        blit_text(self.canvas, self.fonts.tiny, "% 左脚", C.TEXT_2, (1082, 292))
        blit_text(self.canvas, self.fonts.hero, f"{right_ratio * 100:.0f}", C.ORANGE, (1395, 246), anchor="topright")
        blit_text(self.canvas, self.fonts.tiny, "右脚 %", C.TEXT_2, (1395, 292), anchor="topright")
        bar = pygame.Rect(1080, 327, 304, 14)
        pygame.draw.rect(self.canvas, C.GRID, bar, border_radius=7)
        left_w = round(bar.w * left_ratio)
        pygame.draw.rect(self.canvas, C.BLUE, (bar.x, bar.y, left_w, bar.h), border_radius=7)
        pygame.draw.rect(self.canvas, C.ORANGE, (bar.x + left_w, bar.y, bar.w - left_w, bar.h), border_radius=7)
        pygame.draw.line(self.canvas, C.WHITE, (bar.centerx, bar.y - 3), (bar.centerx, bar.bottom + 3), 2)
        blit_text(self.canvas, self.fonts.tiny, f"对称度 {symmetry:.0f}%", C.TEXT_2, (1080, 361))
        blit_text(self.canvas, self.fonts.tiny, "相对 Hall 响应，不是力/压力", C.TEXT_3, (1080, 386))

    def draw_device_card(self) -> None:
        card = pygame.Rect(1048, 430, 368, 190)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "双路蓝牙状态", C.TEXT, (1066, 445))
        for i, side in enumerate(SIDES):
            slot = self.source.slot_snapshot(side)
            y = 483 + i * 62
            color = C.GREEN if slot.connected else C.RED
            pygame.draw.circle(self.canvas, color, (1074, y + 7), 5)
            blit_text(self.canvas, self.fonts.small, SIDE_CN[side], C.TEXT, (1087, y - 3))
            blit_text(self.canvas, self.fonts.tiny, slot.status, C.TEXT_2, (1144, y))
            identity = slot.name or slot.expected_name
            address = slot.address or slot.detail or "尚未发现"
            blit_text(
                self.canvas,
                self.fonts.tiny,
                f"{identity} · {address}"[:34],
                C.TEXT_3,
                (1087, y + 26),
            )
        blit_text(self.canvas, self.fonts.tiny, "唯一名称 left / right · 地址二次校验", C.TEXT_3, (1066, 594))

    def _draw_series(self, rect: pygame.Rect, values: deque[float], color) -> None:
        if len(values) < 2:
            return
        arr = np.asarray(values, dtype=np.float32)
        both_peaks = list(self.feet["left"].history_peak) + list(self.feet["right"].history_peak)
        hi = max(4200.0, float(np.percentile(both_peaks or [1.0], 98)) * 1.10)
        xs = np.linspace(rect.x, rect.right, len(arr))
        ys = rect.bottom - np.clip(arr / hi, 0.0, 1.0) * rect.h
        points = [(round(x), round(y)) for x, y in zip(xs, ys)]
        pygame.draw.aalines(self.canvas, color, False, points)
        pygame.draw.lines(self.canvas, color, False, points, 2)

    def draw_trend_card(self) -> None:
        card = pygame.Rect(1048, 636, 368, 222)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "左右脚实时趋势", C.TEXT, (1066, 651))
        blit_text(self.canvas, self.fonts.tiny, "每脚 15 通道平均响应 · R 清空", C.TEXT_3, (1066, 677))
        chart = pygame.Rect(1066, 718, 332, 116)
        for i in range(4):
            y = chart.y + round(chart.h * i / 3)
            pygame.draw.line(self.canvas, C.GRID, (chart.x, y), (chart.right, y), 1)
        for i in range(7):
            x = chart.x + round(chart.w * i / 6)
            pygame.draw.line(self.canvas, C.GRID, (x, chart.y), (x, chart.bottom), 1)
        self._draw_series(chart, self.feet["left"].history_mean, C.BLUE)
        self._draw_series(chart, self.feet["right"].history_mean, C.ORANGE)
        pygame.draw.line(self.canvas, C.BLUE, (1272, 660), (1288, 660), 2)
        blit_text(self.canvas, self.fonts.tiny, "左脚", C.TEXT_2, (1293, 652))
        pygame.draw.line(self.canvas, C.ORANGE, (1340, 660), (1356, 660), 2)
        blit_text(self.canvas, self.fonts.tiny, "右脚", C.TEXT_2, (1361, 652))

    def draw_footer(self) -> None:
        text = "D 重连   X 交换左右   B 空载归零   G 色标模式   SPACE 暂停   H 热图   M 磁矢量   C 响应中心   I 编号   F9 录制   S 截图   ESC 退出"
        blit_text(self.canvas, self.fonts.tiny, text, C.TEXT_3, (24, 874))
        source = "DEMO / 双脚仿真" if self.source.is_demo else "BLE / DUAL REALTIME"
        blit_text(self.canvas, self.fonts.tiny, source, C.TEXT_3, (1416, 874), anchor="topright")

    def draw_overlay(self) -> None:
        if self.paused:
            pill = pygame.Rect(654, 93, 132, 34)
            rounded_rect(self.canvas, pill, C.YELLOW, 10)
            blit_text(self.canvas, self.fonts.small, "已暂停", C.INK, pill.center, anchor="center")
        if self.toast_text and time.monotonic() < self.toast_until:
            image = self.fonts.small.render(self.toast_text, True, C.TEXT)
            toast = pygame.Rect(0, 0, image.get_width() + 34, 42)
            toast.midbottom = (LOGICAL_SIZE[0] // 2, 851)
            rounded_rect(self.canvas, toast, C.PANEL, 12, C.CYAN_2)
            self.canvas.blit(image, image.get_rect(center=toast.center))

    def render(self) -> None:
        self.canvas.fill(C.BG)
        glow = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(glow, (128, 195, 181, 20), (420, 430), 410)
        pygame.draw.circle(glow, (154, 176, 206, 16), (1280, 650), 310)
        self.canvas.blit(glow, (0, 0))
        self.draw_header()
        self.draw_metric_cards()
        self.draw_feet_card()
        self.draw_balance_card()
        self.draw_device_card()
        self.draw_trend_card()
        self.draw_footer()
        self.draw_overlay()

        sw, sh = self.screen.get_size()
        scale = min(sw / LOGICAL_SIZE[0], sh / LOGICAL_SIZE[1])
        target = (max(1, round(LOGICAL_SIZE[0] * scale)), max(1, round(LOGICAL_SIZE[1] * scale)))
        scaled = pygame.transform.smoothscale(self.canvas, target)
        self.screen.fill(C.BG_2)
        self.screen.blit(scaled, ((sw - target[0]) // 2, (sh - target[1]) // 2))
        pygame.display.flip()

    def run(self) -> Optional[Path]:
        self.source.start()
        try:
            while self.running:
                self.handle_events()
                self.process()
                self.render()
                self.recorder.capture(self.canvas)
                self.rendered_frames += 1
                if self.screenshot_path and self.rendered_frames >= self.screenshot_frame:
                    self.save_screenshot(Path(self.screenshot_path).resolve())
                    break
                self.clock.tick(FPS)
        finally:
            if self.recorder.active:
                self.recorder.stop()
            self.source.stop()
            pygame.quit()
        return self.saved_requested_shot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FootSensor15 双脚 BLE 实时可视化")
    parser.add_argument(
        "--mode",
        choices=("demo", "ble"),
        default="ble",
        help="ble=同时连接两只真实脚（默认），demo=双脚交替步态仿真",
    )
    parser.add_argument("--left-address", help="固定左脚 BLE 地址")
    parser.add_argument("--right-address", help="固定右脚 BLE 地址")
    parser.add_argument("--left-adapter", help="固定左脚蓝牙控制器，例如 hci0")
    parser.add_argument("--right-adapter", help="固定右脚蓝牙控制器，例如 hci1")
    parser.add_argument("--left-name", default="left", help="左脚 BLE 广播名")
    parser.add_argument("--right-name", default="right", help="右脚 BLE 广播名")
    parser.add_argument(
        "--layout", type=Path, default=DEFAULT_LAYOUT_PATH, help="A4 1:1 的 15 点布局 JSON"
    )
    parser.add_argument("--screenshot", metavar="PNG", help="保存预览截图后退出")
    parser.add_argument("--screenshot-frame", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    global SENSOR_POS, INSOLE_OUTLINE
    args = parse_args()
    try:
        SENSOR_POS, INSOLE_OUTLINE = load_sensor_layout(args.layout)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] 无法加载传感器布局：{error}", file=sys.stderr)
        return 2
    if args.mode == "demo":
        source: DualBLESource | DualDemoSource = DualDemoSource()
    else:
        source = DualBLESource(
            left_address=args.left_address,
            right_address=args.right_address,
            left_adapter=args.left_adapter,
            right_adapter=args.right_adapter,
            left_name=args.left_name,
            right_name=args.right_name,
        )
    app = DualDashboard(
        source,
        left_address=args.left_address,
        right_address=args.right_address,
        left_adapter=args.left_adapter,
        right_adapter=args.right_adapter,
        left_name=args.left_name,
        right_name=args.right_name,
        screenshot_path=args.screenshot,
        screenshot_frame=args.screenshot_frame,
    )
    shot = app.run()
    if shot is not None:
        print(shot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
