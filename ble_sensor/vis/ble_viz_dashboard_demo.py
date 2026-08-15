#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FootSensor15 Dual-Foot Real-Time BLE Dashboard

Features
--------
1. Binds BLE devices advertised as ``left`` and ``right`` without ambiguous ordering.
2. Keeps Notify decoding, baselines, filters, heatmaps, and trends independent.
3. Shows bilateral load distribution, regional balance, IDW field vectors, and CoP.
4. Supports MP4 recording through FFmpeg.

Run
---
    ../.venv/bin/python ble_viz_dashboard_demo.py

Shortcuts
---------
    B        Recalibrate (keep both insoles unloaded)
    D        Reconnect both configured FootSensor15 devices
    X        Swap left/right device mapping and reconnect
    Space    Pause / resume
    H        Show / hide heatmaps
    M        Show / hide magnetic vectors
    C        Show / hide center of pressure (hidden by default)
    I        Show / hide sensor IDs
    F9       Start / stop MP4 recording
    S        Save a screenshot under screenshots/
    R        Clear trends
    Esc      Exit
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

# 允许在无桌面的 CI / 服务器里生成静态截图。
if "--screenshot" in sys.argv and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# 配置与数据模型
# ---------------------------------------------------------------------------

APP_TITLE = "Dual-Foot Multiphysics Monitor"
APP_VERSION = "DUAL BLE REALTIME 02"
LOGICAL_SIZE = (1440, 900)
MIN_WINDOW_SIZE = (1120, 700)
FPS = 60
SAMPLE_HZ = 100
SAMPLE_INTERVAL_S = 1.0 / SAMPLE_HZ
NUM_SENSORS = 15

LEGACY_DEVICE_NAME = "FootSensor15"
DEFAULT_DEVICE_NAMES = {"left": "left", "right": "right"}
# The file still contains the legacy single-foot dashboard for compatibility.
DEVICE_NAME = LEGACY_DEVICE_NAME
CHAR_UUID = "0000ab01-0000-1000-8000-00805f9b34fb"
FRAME_LEN = 125
FRAME_HEADER = 0x7D
DATA_OFFSET = 4
DATA_BYTES = 120
ENDIAN_FMT = ">hhhh"  # T(°C×10), X, Y, Z

# 传感器在足底轮廓中的归一化位置。
# 依据用户给出的 15 点排布示意图重建；脚尖在上，脚跟在下。
SENSOR_POS = np.array(
    [
        (0.516, 0.129),
        (0.299, 0.201),
        (0.516, 0.203),
        (0.748, 0.205),
        (0.518, 0.305),
        (0.487, 0.435),
        (0.285, 0.514),
        (0.498, 0.517),
        (0.741, 0.522),
        (0.504, 0.593),
        (0.504, 0.751),
        (0.323, 0.809),
        (0.504, 0.817),
        (0.708, 0.813),
        (0.504, 0.882),
    ],
    dtype=np.float32,
)
DEFAULT_LAYOUT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "sensor_layout_a4_15.json"
)


def load_sensor_positions(path: Path) -> np.ndarray:
    """Load the numbered 15-point A4 layout without changing channel order."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "footsensor15-a4-layout-v1":
        raise ValueError(f"{path}: unsupported sensor layout format")
    sensors = document.get("sensors")
    if not isinstance(sensors, list) or len(sensors) != NUM_SENSORS:
        raise ValueError(f"{path}: sensors must contain exactly {NUM_SENSORS} entries")
    ordered = sorted(sensors, key=lambda item: int(item.get("id", -1)))
    if [int(item.get("id", -1)) for item in ordered] != list(range(1, 16)):
        raise ValueError(f"{path}: sensor ids must be exactly 1..15")
    positions = np.asarray(
        [item.get("normalized_uv") for item in ordered], dtype=np.float32
    )
    if positions.shape != (NUM_SENSORS, 2) or not np.isfinite(positions).all():
        raise ValueError(f"{path}: normalized_uv must be a finite 15x2 array")
    if np.any(positions < 0.0) or np.any(positions > 1.0):
        raise ValueError(f"{path}: normalized_uv values must be in [0, 1]")
    return positions

# 与 ble_viz_superres_hot 依赖的原实现一致：先把各芯片局部 XY
# 旋转到鞋垫统一坐标，再进行规则网格 IDW 插值。
_EF = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0], dtype=np.float32)
CHIP_XY_ROTATIONS = np.array(
    [
        _EF[0], _EF[1], 0.0, np.pi / 2, _EF[4],
        _EF[0], _EF[1], _EF[2], _EF[3],
        _EF[0], _EF[4],
        _EF[1], _EF[2], _EF[3], _EF[4],
    ],
    dtype=np.float32,
)

# 从用户提供的鞋垫 STEP 最大水平面提取并归一化的外轮廓。
# STEP 原始 Y 最小端是脚尖；这里已映射成屏幕上方，避免前后颠倒。
INSOLE_OUTLINE = np.array(
    [
        (0.9082, 0.8817),
        (0.9187, 0.8403),
        (0.9186, 0.8360),
        (0.9090, 0.7607),
        (0.9067, 0.7335),
        (0.9035, 0.6707),
        (0.9157, 0.5917),
        (0.9298, 0.5606),
        (0.9831, 0.4878),
        (1.0000, 0.4541),
        (0.9994, 0.4232),
        (0.9906, 0.2614),
        (0.9786, 0.2107),
        (0.9381, 0.1347),
        (0.8841, 0.0853),
        (0.8156, 0.0489),
        (0.7334, 0.0232),
        (0.6409, 0.0076),
        (0.5495, 0.0009),
        (0.4912, 0.0000),
        (0.4089, 0.0024),
        (0.3026, 0.0139),
        (0.2150, 0.0338),
        (0.1446, 0.0613),
        (0.0884, 0.0970),
        (0.0456, 0.1419),
        (0.0121, 0.2143),
        (0.0000, 0.3064),
        (0.0026, 0.4153),
        (0.0046, 0.4758),
        (0.0187, 0.5056),
        (0.0613, 0.5638),
        (0.0906, 0.6414),
        (0.0939, 0.7365),
        (0.0876, 0.8391),
        (0.0999, 0.8839),
        (0.1378, 0.9203),
        (0.2022, 0.9530),
        (0.2868, 0.9780),
        (0.3811, 0.9934),
        (0.4816, 1.0000),
        (0.5259, 1.0000),
        (0.6307, 0.9929),
        (0.7334, 0.9749),
        (0.8097, 0.9510),
        (0.8666, 0.9218),
    ],
    dtype=np.float32,
)

REGION_NAMES = ("Forefoot", "Arch", "Heel")
REGION_INDICES = (
    np.arange(0, 5),
    np.arange(5, 10),
    np.arange(10, 15),
)


class C:
    """低饱和、浅色、带一点薄荷绿的淡雅主题。"""

    BG = (241, 246, 245)
    BG_2 = (232, 239, 238)
    PANEL = (252, 253, 251)
    PANEL_2 = (244, 248, 246)
    PANEL_HOVER = (234, 243, 240)
    BORDER = (207, 220, 217)
    GRID = (222, 231, 229)
    TEXT = (47, 65, 72)
    TEXT_2 = (92, 111, 116)
    TEXT_3 = (143, 158, 160)
    CYAN = (67, 160, 153)
    CYAN_2 = (117, 183, 175)
    BLUE = (103, 145, 190)
    GREEN = (104, 176, 141)
    YELLOW = (216, 177, 94)
    ORANGE = (225, 142, 104)
    RED = (214, 104, 111)
    MAGENTA = (166, 131, 183)
    INK = (54, 77, 84)
    WHITE = (255, 255, 255)


@dataclass
class SensorFrame:
    xyz: np.ndarray
    temp_x10: np.ndarray
    timestamp: float
    seq: int
    source_session: int = 0


@dataclass
class Metrics:
    peak: float = 0.0
    active: int = 0
    mean_temp: float = float("nan")
    min_temp: float = float("nan")
    hz: float = 0.0
    stability: float = 100.0
    resultant: tuple[float, float] = (0.0, 0.0)
    force_components: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cop: tuple[float, float] = (0.5, 0.5)
    region_loads: tuple[float, float, float] = (0.0, 0.0, 0.0)
    region_totals: tuple[float, float, float] = (0.0, 0.0, 0.0)
    total: float = 0.0


# ---------------------------------------------------------------------------
# BLE 与仿真数据源
# ---------------------------------------------------------------------------


class Int16Unwrapper:
    """把连续 int16 流展开为 int32，避免跨 ±32768 时出现整幅跳变。"""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.last: Optional[np.ndarray] = None
        self.ext: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last = None
        self.ext = None

    def push(self, wire: np.ndarray) -> np.ndarray:
        current = np.asarray(wire, dtype=np.int32)
        if self.last is None:
            self.last = current.copy()
            self.ext = current.copy()
            return self.ext.copy()
        delta = (current - self.last + 32768) % 65536 - 32768
        self.ext = np.clip(self.ext.astype(np.int64) + delta, -2_000_000, 2_000_000).astype(
            np.int32
        )
        self.last = current.copy()
        return self.ext.copy()


class FrameParser:
    def __init__(self) -> None:
        self.unwrap = Int16Unwrapper((NUM_SENSORS, 3))
        self.seq = 0

    def parse(self, data: bytes) -> Optional[SensorFrame]:
        if len(data) < FRAME_LEN:
            return None
        if data[0] != FRAME_HEADER or data[2] != 0xF0 or data[3] != 0x02:
            return None

        raw = data[DATA_OFFSET : DATA_OFFSET + DATA_BYTES]
        xyz_wire = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
        temp = np.zeros(NUM_SENSORS, dtype=np.int32)
        try:
            for i in range(NUM_SENSORS):
                t_x10, x, y, z = struct.unpack_from(ENDIAN_FMT, raw, i * 8)
                temp[i] = t_x10
                xyz_wire[i] = (x, y, z)
        except struct.error:
            return None

        self.seq += 1
        return SensorFrame(
            xyz=self.unwrap.push(xyz_wire),
            temp_x10=temp,
            timestamp=time.monotonic(),
            seq=self.seq,
        )


class FrameStreamDecoder:
    """从 BLE Notify 字节流中恢复完整 125 字节帧，兼容拆包、粘包和错位。"""

    def __init__(self) -> None:
        self.parser = FrameParser()
        self.buffer = bytearray()
        self.discarded_bytes = 0

    def reset(self) -> None:
        self.parser.unwrap.reset()
        self.buffer.clear()
        self.discarded_bytes = 0

    def feed(self, payload: bytes) -> list[SensorFrame]:
        self.buffer.extend(payload)
        frames: list[SensorFrame] = []
        while True:
            try:
                header_at = self.buffer.index(FRAME_HEADER)
            except ValueError:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if header_at:
                self.discarded_bytes += header_at
                del self.buffer[:header_at]
            if len(self.buffer) < FRAME_LEN:
                break
            if self.buffer[2] != 0xF0 or self.buffer[3] != 0x02:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            candidate = bytes(self.buffer[:FRAME_LEN])
            del self.buffer[:FRAME_LEN]
            frame = self.parser.parse(candidate)
            if frame is not None:
                frames.append(frame)
        # 防止持续异常数据无限占用内存。
        if len(self.buffer) > FRAME_LEN * 4:
            self.discarded_bytes += len(self.buffer) - FRAME_LEN
            del self.buffer[:-FRAME_LEN]
        return frames


class DemoSource:
    """可复现的步态仿真，用来先评审 UI，不冒充真实物理量。"""

    def __init__(self) -> None:
        rng = np.random.default_rng(20260731)
        self.base = rng.normal(0.0, 1600.0, (NUM_SENSORS, 3)).astype(np.float32)
        self.bias = rng.uniform(0.82, 1.16, NUM_SENSORS).astype(np.float32)
        self.rng = rng
        self.started_at = time.monotonic()
        self.last_emit = 0.0
        self.seq = 0
        self.latest: Optional[SensorFrame] = None
        self.status = "Demo"
        self.detail = "无需硬件 · 50 Hz"
        self.connected = True

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def sample(self) -> Optional[SensorFrame]:
        now = time.monotonic()
        if now - self.last_emit < 1.0 / 50.0:
            return self.latest
        self.last_emit = now
        t = now - self.started_at

        # 前 1.15 s 保持空载，便于启动时自动校准。
        signal = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        if t > 1.15:
            cycle = 1.65
            phase = ((t - 1.15) % cycle) / cycle
            if phase < 0.76:
                u = phase / 0.76
                envelope = math.sin(math.pi * u) ** 0.35
                center_y = 0.88 - 0.79 * (u * u * (3.0 - 2.0 * u))
                longitudinal = np.exp(-((SENSOR_POS[:, 1] - center_y) / 0.22) ** 2)
                support = 0.25 + 0.75 * longitudinal
                intensity = 4100.0 * envelope * support * self.bias
                lateral = np.sin(2.0 * math.pi * u + SENSOR_POS[:, 1] * 2.2)
                signal[:, 0] = intensity * (0.15 * lateral + (SENSOR_POS[:, 0] - 0.5) * 0.22)
                signal[:, 1] = intensity * (0.18 - 0.31 * u)
                signal[:, 2] = intensity * (0.80 + 0.08 * np.cos(SENSOR_POS[:, 0] * 7.0))

        noise = self.rng.normal(0.0, 20.0, (NUM_SENSORS, 3)).astype(np.float32)
        xyz = np.rint(self.base + signal + noise).astype(np.int32)
        temp = np.rint(
            281.0
            + 4.0 * np.sin(t * 0.11 + np.arange(NUM_SENSORS) * 0.29)
            + self.rng.normal(0.0, 0.18, NUM_SENSORS)
        ).astype(np.int32)
        self.seq += 1
        self.latest = SensorFrame(xyz, temp, now, self.seq)
        return self.latest


class BLESource:
    """Bleak 后台线程。所有错误都保留到 UI 状态区，不再静默吞掉。"""

    def __init__(self) -> None:
        self.decoder = FrameStreamDecoder()
        self.latest: Optional[SensorFrame] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.status = "准备扫描"
        self.detail = DEVICE_NAME
        self.connected = False
        self.notification_count = 0
        self.valid_frame_count = 0
        self.last_payload_len = 0
        self.last_valid_at = 0.0

    def start(self) -> None:
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            # UI 内切换数据源时不能被 BLE 扫描阻塞；后台线程是 daemon，
            # 收到事件后会自行结束扫描/连接。
            self.thread.join(timeout=0.25)

    def sample(self) -> Optional[SensorFrame]:
        with self.lock:
            if self.latest is None:
                return None
            return SensorFrame(
                self.latest.xyz.copy(),
                self.latest.temp_x10.copy(),
                self.latest.timestamp,
                self.latest.seq,
            )

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.connected = False
            self.status = "BLE 线程异常"
            self.detail = str(exc)[:72]

    async def _run(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            self.status = "缺少 bleak"
            self.detail = "请执行 pip install bleak"
            return

        self.status = "正在扫描"
        self.detail = f"寻找 {DEVICE_NAME} · 最长 10 秒"
        try:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
        except Exception as exc:
            self.status = "扫描失败"
            self.detail = str(exc)[:72]
            return
        if device is None:
            self.status = "未发现设备"
            self.detail = f"请确认 {DEVICE_NAME} 已上电并允许蓝牙访问"
            return

        self.status = "正在连接"
        self.detail = getattr(device, "address", DEVICE_NAME)
        try:
            async with BleakClient(device) as client:
                self.decoder.reset()
                self.connected = True
                self.status = "BLE 已连接"
                self.detail = getattr(device, "address", DEVICE_NAME)
                connected_at = time.monotonic()

                def on_notify(_sender, payload: bytearray) -> None:
                    self.notification_count += 1
                    self.last_payload_len = len(payload)
                    frames = self.decoder.feed(bytes(payload))
                    if frames:
                        self.valid_frame_count += len(frames)
                        self.last_valid_at = time.monotonic()
                        with self.lock:
                            self.latest = frames[-1]

                await client.start_notify(CHAR_UUID, on_notify)
                while not self.stop_event.is_set() and client.is_connected:
                    now = time.monotonic()
                    if self.valid_frame_count == 0 and now - connected_at > 1.8:
                        self.status = "已连接·无有效数据"
                        self.detail = (
                            f"Notify {self.notification_count} 次 · "
                            f"末包 {self.last_payload_len} B · "
                            f"丢弃 {self.decoder.discarded_bytes} B"
                        )
                    elif self.last_valid_at and now - self.last_valid_at > 1.2:
                        self.status = "BLE 数据已暂停"
                        self.detail = (
                            f"有效帧 {self.valid_frame_count} · "
                            f"距末帧 {now - self.last_valid_at:.1f}s"
                        )
                    elif self.valid_frame_count:
                        self.status = "BLE 实时数据"
                        self.detail = (
                            f"有效帧 {self.valid_frame_count} · "
                            f"Notify {self.notification_count}"
                        )
                    await asyncio.sleep(0.08)
                if client.is_connected:
                    await client.stop_notify(CHAR_UUID)
        except Exception as exc:
            self.status = "连接中断"
            self.detail = str(exc)[:72]
        finally:
            self.connected = False


# ---------------------------------------------------------------------------
# 双脚数据源
# ---------------------------------------------------------------------------


SIDES = ("left", "right")
SIDE_NAMES = {"left": "Left", "right": "Right"}


class DemoFootSource:
    """单只脚的可复现仿真通道；左右脚使用不同随机种子和步态相位。"""

    def __init__(
        self,
        side: str,
        cadence_spm: float = 96.0,
        *,
        cadence_profile: Optional[list[tuple[float, float]]] = None,
        walking_start_s: float = 1.15,
    ) -> None:
        self.side = side
        self.cadence_spm = float(np.clip(cadence_spm, 50.0, 170.0))
        self.cadence_profile = sorted(cadence_profile or [])
        self.walking_start_s = max(0.5, float(walking_start_s))
        self.current_cadence_spm = self.cadence_spm
        rng = np.random.default_rng(20260731 if side == "left" else 20260801)
        self.base = rng.normal(0.0, 1600.0, (NUM_SENSORS, 3)).astype(np.float32)
        self.bias = rng.uniform(0.91, 1.09, NUM_SENSORS).astype(np.float32)
        self.drift_phase = rng.uniform(0.0, 2.0 * math.pi, (NUM_SENSORS, 3))
        self.rng = rng
        self.started_at = time.monotonic()
        self.model_time_s = 0.0
        self.simulation_time_s: Optional[float] = None
        self.phase_cycles = 0.0 if side == "left" else 0.50
        self.last_emit = 0.0
        self.seq = 0
        self.latest: Optional[SensorFrame] = None
        self.status = "Demo"
        self.detail = f"50 Hz · {self.cadence_spm:.0f} SPM"
        self.connected = True
        self.address = "DEMO-L" if side == "left" else "DEMO-R"

    @property
    def raw_hz(self) -> float:
        return 50.0

    def _cadence_at(self, walking_time: float) -> float:
        if not self.cadence_profile:
            return self.cadence_spm
        times = [point[0] for point in self.cadence_profile]
        rates = [point[1] for point in self.cadence_profile]
        return float(np.clip(np.interp(walking_time, times, rates), 50.0, 170.0))

    def set_simulation_time(self, elapsed_s: Optional[float]) -> None:
        self.simulation_time_s = None if elapsed_s is None else max(0.0, float(elapsed_s))

    def sample(self) -> Optional[SensorFrame]:
        now = time.monotonic()
        if now - self.last_emit < 1.0 / 50.0:
            return self.latest
        self.last_emit = now
        t = (
            now - self.started_at
            if self.simulation_time_s is None
            else self.simulation_time_s
        )
        model_dt = float(np.clip(t - self.model_time_s, 0.0, 0.10))
        self.model_time_s = t
        signal = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        if t > self.walking_start_s:
            walking_time = t - self.walking_start_s
            self.current_cadence_spm = self._cadence_at(walking_time)
            self.phase_cycles += model_dt * self.current_cadence_spm / 120.0
            phase = self.phase_cycles % 1.0
            cycle_index = math.floor(self.phase_cycles)
            stance_fraction = float(
                np.interp(self.current_cadence_spm, (50.0, 170.0), (0.68, 0.57))
            )
            if phase < stance_fraction:
                u = phase / stance_fraction
                smooth_u = u * u * (3.0 - 2.0 * u)
                onset = float(np.clip(u / 0.055, 0.0, 1.0))
                toe_off = float(np.clip((1.0 - u) / 0.070, 0.0, 1.0))
                contact_gate = onset * onset * (3.0 - 2.0 * onset)
                contact_gate *= toe_off * toe_off * (3.0 - 2.0 * toe_off)

                # Approximate the vertical ground-reaction M-shape: heel strike,
                # mid-stance support, and a stronger forefoot push-off peak.
                heel_peak = math.exp(-((u - 0.11) / 0.115) ** 2)
                mid_support = math.exp(-((u - 0.44) / 0.31) ** 2)
                push_peak = math.exp(-((u - 0.79) / 0.145) ** 2)
                load_profile = contact_gate * (
                    0.68 * heel_peak + 0.50 * mid_support + 0.92 * push_peak
                )

                center_y = 0.89 - 0.80 * smooth_u
                contact_width = 0.16 + 0.08 * math.sin(math.pi * u)
                longitudinal = np.exp(
                    -((SENSOR_POS[:, 1] - center_y) / contact_width) ** 2
                )
                # Some load remains distributed across the rest of the foot.
                support = 0.20 + 0.80 * longitudinal
                side_gain = 1.025 if self.side == "left" else 0.975
                stride_variation = 1.0 + 0.025 * math.sin(
                    cycle_index * 1.71 + (0.4 if self.side == "left" else 1.2)
                )
                speed_gain = float(
                    np.interp(self.current_cadence_spm, (50.0, 170.0), (0.88, 1.12))
                )
                intensity = (
                    5200.0
                    * speed_gain
                    * side_gain
                    * stride_variation
                    * load_profile
                    * support
                    * self.bias
                )

                lateral = np.sin(math.pi * u + SENSOR_POS[:, 1] * 1.7)
                side_sign = -1.0 if self.side == "left" else 1.0
                signal[:, 0] = intensity * (
                    side_sign * 0.07 * lateral
                    + (SENSOR_POS[:, 0] - 0.5) * 0.18
                )
                # Fore/aft shear changes sign between braking and propulsion.
                signal[:, 1] = intensity * (0.16 - 0.34 * smooth_u)
                signal[:, 2] = intensity * (
                    0.88 + 0.055 * np.cos(SENSOR_POS[:, 0] * 7.0)
                )

        # Independent sensor noise plus small temperature-like baseline drift.
        drift = 11.0 * np.sin(t * 0.19 + self.drift_phase)
        noise = self.rng.normal(0.0, 15.0, (NUM_SENSORS, 3)).astype(np.float32)
        xyz = np.rint(self.base + drift + signal + noise).astype(np.int32)
        temp = np.rint(
            281.0
            + 4.0 * np.sin(t * 0.11 + np.arange(NUM_SENSORS) * 0.29)
            + self.rng.normal(0.0, 0.18, NUM_SENSORS)
        ).astype(np.int32)
        self.seq += 1
        self.detail = f"50 Hz · {self.current_cadence_spm:.0f} SPM"
        self.latest = SensorFrame(xyz, temp, now, self.seq)
        return self.latest


class DualDemoSource:
    def __init__(
        self,
        cadence_spm: float = 96.0,
        *,
        cadence_profile: Optional[list[tuple[float, float]]] = None,
        walking_start_s: float = 1.15,
    ) -> None:
        self.cadence_spm = float(np.clip(cadence_spm, 50.0, 170.0))
        self.feet = {
            side: DemoFootSource(
                side,
                self.cadence_spm,
                cadence_profile=cadence_profile,
                walking_start_s=walking_start_s,
            )
            for side in SIDES
        }

    @property
    def current_cadence_spm(self) -> float:
        return float(np.mean([foot.current_cadence_spm for foot in self.feet.values()]))

    def set_simulation_time(self, elapsed_s: Optional[float]) -> None:
        for foot in self.feet.values():
            foot.set_simulation_time(elapsed_s)

    @property
    def connected(self) -> bool:
        return True

    @property
    def status(self) -> str:
        return "Dual-foot demo"

    @property
    def detail(self) -> str:
        return f"Independent feet · 50 Hz · {self.current_cadence_spm:.0f} SPM"

    @property
    def addresses(self) -> dict[str, str]:
        return {side: self.feet[side].address for side in SIDES}

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def sample(self) -> dict[str, Optional[SensorFrame]]:
        return {side: self.feet[side].sample() for side in SIDES}


class BLEFootChannel:
    """一只物理鞋垫的独立状态；解码器绝不与另一只脚共享。"""

    def __init__(self, side: str, address: str = "", device_name: str = "") -> None:
        self.side = side
        self.address = address.strip().upper()
        self.device_name = device_name.strip() or DEFAULT_DEVICE_NAMES[side]
        self.observed_name = ""
        self.decoder = FrameStreamDecoder()
        self.latest: Optional[SensorFrame] = None
        self.frame_buffer: deque[SensorFrame] = deque(maxlen=24)
        self.lock = threading.Lock()
        self.connected = False
        self.status = "Waiting"
        self.detail = self.address or self.device_name
        self.notification_count = 0
        self.valid_frame_count = 0
        self.source_session = 0
        self.output_seq = 0
        self.last_payload_len = 0
        self.last_valid_at = 0.0
        self.frame_times: deque[float] = deque(maxlen=240)
        self.output_times: deque[float] = deque(maxlen=180)
        self.mtu_size = 23

    @property
    def raw_hz(self) -> float:
        """BLE 解码层实测帧率，不受 UI 主循环抽样速度影响。"""
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 1e-6 else 0.0

    @property
    def output_hz(self) -> float:
        """Rate of synchronized samples delivered to the dashboard."""
        if len(self.output_times) < 2:
            return 0.0
        elapsed = self.output_times[-1] - self.output_times[0]
        return (len(self.output_times) - 1) / elapsed if elapsed > 1e-6 else 0.0

    def begin_session(self) -> None:
        """Drop stale timing and samples whenever this physical link reconnects."""
        self.decoder.reset()
        with self.lock:
            self.latest = None
            self.frame_buffer.clear()
            self.frame_times.clear()
            self.output_times.clear()
            self.output_seq = 0
        self.notification_count = 0
        self.valid_frame_count = 0
        self.last_payload_len = 0
        self.last_valid_at = 0.0

    def snapshot(self, target_time: Optional[float] = None) -> Optional[SensorFrame]:
        sampled_at = time.monotonic()
        target_time = sampled_at if target_time is None else target_time
        with self.lock:
            if not self.frame_buffer:
                return None
            frames = list(self.frame_buffer)
            newest = frames[-1]
            older = newest
            newer = newest
            if target_time <= frames[0].timestamp:
                older = newer = frames[0]
            elif target_time < newest.timestamp:
                for index in range(1, len(frames)):
                    if frames[index].timestamp >= target_time:
                        older = frames[index - 1]
                        newer = frames[index]
                        break

            span = newer.timestamp - older.timestamp
            if span > 1e-6 and older is not newer:
                amount = float(np.clip((target_time - older.timestamp) / span, 0.0, 1.0))
                xyz = older.xyz.astype(np.float32) + amount * (
                    newer.xyz.astype(np.float32) - older.xyz.astype(np.float32)
                )
                temp = older.temp_x10.astype(np.float32) + amount * (
                    newer.temp_x10.astype(np.float32) - older.temp_x10.astype(np.float32)
                )
            else:
                xyz = newer.xyz.copy()
                temp = newer.temp_x10.copy()
            self.output_seq += 1
            self.output_times.append(sampled_at)
            # Preserve the real source timestamp after a stall so the display
            # watchdog can decay the signal instead of holding an old frame.
            output_timestamp = (
                newest.timestamp
                if sampled_at - newest.timestamp > 0.30
                else target_time
            )
            return SensorFrame(
                xyz,
                temp,
                output_timestamp,
                self.output_seq,
                newest.source_session,
            )


class DualBLESource:
    """一次扫描后按两个唯一地址连接，左右 Notify/缓冲/序列完全隔离。"""

    def __init__(
        self,
        addresses: Optional[dict[str, str]] = None,
        *,
        device_names: Optional[dict[str, str]] = None,
        scan_timeout: float = 8.0,
    ) -> None:
        addresses = addresses or {}
        device_names = device_names or DEFAULT_DEVICE_NAMES
        clean = {side: str(addresses.get(side, "")).strip().upper() for side in SIDES}
        names = {
            side: str(device_names.get(side, DEFAULT_DEVICE_NAMES[side])).strip()
            for side in SIDES
        }
        if not all(names.values()):
            raise ValueError("left and right BLE device names must be non-empty")
        if names["left"].casefold() == names["right"].casefold():
            raise ValueError("left and right BLE device names must be different")
        if clean["left"] and clean["right"] and clean["left"] == clean["right"]:
            raise ValueError("left and right must use different BLE addresses")
        self.feet = {
            side: BLEFootChannel(side, clean[side], names[side]) for side in SIDES
        }
        self.scan_timeout = max(2.0, float(scan_timeout))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._status = "Ready to scan"
        self._detail = f"{names['left']} / {names['right']}"
        self._notify_barrier: Optional[asyncio.Event] = None
        self._initial_ready: set[str] = set()
        self._expected_sides: set[str] = set()
        self.resample_latency_s = 0.050

    @property
    def addresses(self) -> dict[str, str]:
        return {side: self.feet[side].address for side in SIDES}

    @property
    def device_names(self) -> dict[str, str]:
        return {side: self.feet[side].device_name for side in SIDES}

    @property
    def connected(self) -> bool:
        return all(self.feet[side].connected for side in SIDES)

    @property
    def status(self) -> str:
        count = sum(int(self.feet[side].connected) for side in SIDES)
        return f"BLE {count}/2 connected" if count else self._status

    @property
    def detail(self) -> str:
        return self._detail

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def sample(self) -> dict[str, Optional[SensorFrame]]:
        target_time = time.monotonic() - self.resample_latency_s
        return {
            side: self.feet[side].snapshot(target_time) for side in SIDES
        }

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._status = "Dual BLE error"
            self._detail = str(exc)[:96]
            for channel in self.feet.values():
                channel.connected = False

    async def _run(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            self._status = "Bleak missing"
            self._detail = "Run: pip install bleak"
            return

        self._status = "Scanning both insoles"
        expected = " / ".join(self.feet[side].device_name for side in SIDES)
        self._detail = f"Looking for {expected} · {self.scan_timeout:.0f} s"
        for channel in self.feet.values():
            channel.status = "Scanning"
        try:
            discovered = await BleakScanner.discover(
                timeout=self.scan_timeout,
                return_adv=True,
            )
        except Exception as exc:
            self._status = "BLE scan failed"
            self._detail = str(exc)[:96]
            return
        if self.stop_event.is_set():
            return

        candidates = []
        observed_names: dict[str, str] = {}
        for _key, (device, advertisement) in discovered.items():
            observed_name = (advertisement.local_name or device.name or "").strip()
            address_key = device.address.casefold()
            candidates.append(device)
            observed_names[address_key] = observed_name
        candidates.sort(key=lambda item: item.address.casefold())
        by_address = {device.address.casefold(): device for device in candidates}
        assigned: dict[str, object] = {}
        used: set[str] = set()

        # 有配置时只认地址，不因两个设备同名而偷偷换绑。
        for side in SIDES:
            wanted = self.feet[side].address.casefold()
            if not wanted:
                continue
            device = by_address.get(wanted)
            if device is not None:
                assigned[side] = device
                used.add(device.address.casefold())
                self.feet[side].observed_name = observed_names.get(wanted, "")
            else:
                self.feet[side].status = "Configured device missing"
                self.feet[side].detail = self.feet[side].address

        # 未配置地址时严格按唯一广播名 left/right 绑定，不再按地址排序猜左右脚。
        for side in SIDES:
            if side in assigned or self.feet[side].address:
                continue
            expected_name = self.feet[side].device_name.casefold()
            device = next(
                (
                    d
                    for d in candidates
                    if d.address.casefold() not in used
                    and observed_names.get(d.address.casefold(), "").casefold()
                    == expected_name
                ),
                None,
            )
            if device is not None:
                assigned[side] = device
                used.add(device.address.casefold())
                self.feet[side].address = device.address.upper()
                self.feet[side].observed_name = observed_names.get(
                    device.address.casefold(), ""
                )

        if len({self.feet[s].address for s in assigned}) != len(assigned):
            self._status = "Address conflict"
            self._detail = "Left and right cannot use the same device"
            return
        if not assigned:
            self._status = "No insole found"
            self._detail = f"No BLE devices named {expected} detected"
            return

        self._status = "Connecting both feet"
        self._detail = " · ".join(
            f"{self.feet[side].device_name} {self.feet[side].address[-5:]}"
            for side in assigned
        )
        self._expected_sides = set(assigned)
        self._initial_ready.clear()
        self._notify_barrier = asyncio.Event()
        tasks = [
            asyncio.create_task(self._foot_loop(side, device, BleakClient))
            for side, device in assigned.items()
        ]
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.10)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for channel in self.feet.values():
                channel.connected = False

    async def _foot_loop(self, side: str, device, client_class) -> None:
        channel = self.feet[side]
        while not self.stop_event.is_set():
            channel.status = "Connecting"
            channel.detail = channel.address
            try:
                async with client_class(device) as client:
                    channel.source_session += 1
                    source_session = channel.source_session
                    channel.begin_session()
                    channel.connected = True
                    channel.status = "Syncing Notify"
                    channel.detail = channel.address
                    connected_at = time.monotonic()
                    print(f"[BLE] {SIDE_NAMES[side]} connected {channel.address}", flush=True)

                    # 初次启动时先等两个 GATT 客户端都连好，再尽量同时打开
                    # Notify，避免先连接者提前独占更高的链路调度优先级。
                    barrier = self._notify_barrier
                    if barrier is not None and not barrier.is_set():
                        self._initial_ready.add(side)
                        if self._initial_ready >= self._expected_sides:
                            barrier.set()
                        try:
                            await asyncio.wait_for(barrier.wait(), timeout=4.0)
                        except asyncio.TimeoutError:
                            barrier.set()

                    def on_notify(
                        _sender,
                        payload: bytearray,
                        target=channel,
                        session=source_session,
                    ) -> None:
                        target.notification_count += 1
                        target.last_payload_len = len(payload)
                        frames = target.decoder.feed(bytes(payload))
                        if frames:
                            for frame in frames:
                                frame.source_session = session
                            target.valid_frame_count += len(frames)
                            target.last_valid_at = time.monotonic()
                            target.frame_times.extend(frame.timestamp for frame in frames)
                            with target.lock:
                                for frame in frames:
                                    target.frame_buffer.append(frame)
                                target.latest = frames[-1]

                    await client.start_notify(CHAR_UUID, on_notify)
                    while not self.stop_event.is_set() and client.is_connected:
                        now = time.monotonic()
                        if channel.valid_frame_count == 0 and now - connected_at > 1.8:
                            channel.status = "Connected · no frames"
                            channel.detail = (
                                f"Notify {channel.notification_count} · "
                                f"last packet {channel.last_payload_len} B"
                            )
                        elif channel.last_valid_at and now - channel.last_valid_at > 1.2:
                            channel.status = "Data stalled"
                            channel.detail = f"last frame {now - channel.last_valid_at:.1f}s ago"
                        else:
                            channel.status = "Live"
                            channel.detail = (
                                f"frames {channel.valid_frame_count} · "
                                f"Notify {channel.notification_count}"
                            )
                        await asyncio.sleep(0.10)
                    if client.is_connected:
                        await client.stop_notify(CHAR_UUID)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                channel.status = "Disconnected"
                channel.detail = str(exc)[:72]
            finally:
                channel.connected = False
            if not self.stop_event.is_set():
                await asyncio.sleep(1.5)


# ---------------------------------------------------------------------------
# 小型绘图工具
# ---------------------------------------------------------------------------


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    return tuple(int(lerp(a[i], b[i], t)) for i in range(3))


def heat_color(value: float) -> tuple[int, int, int]:
    """低饱和磁响应色图：雾蓝 → 薄荷 → 杏黄 → 珊瑚。"""
    stops = (
        (0.00, (222, 235, 237)),
        (0.22, (164, 207, 205)),
        (0.45, (126, 190, 172)),
        (0.66, (226, 204, 133)),
        (0.84, (230, 155, 112)),
        (1.00, (211, 104, 111)),
    )
    v = float(np.clip(value, 0.0, 1.0))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            return mix_color(c0, c1, (v - p0) / max(p1 - p0, 1e-6))
    return stops[-1][1]


def force_color(value: float) -> tuple[int, int, int]:
    """受力热图色标：冷色低响应，暖色高响应。"""
    stops = (
        (0.00, (235, 243, 244)),
        (0.18, (190, 220, 224)),
        (0.40, (111, 185, 191)),
        (0.62, (226, 207, 123)),
        (0.82, (231, 143, 96)),
        (1.00, (210, 85, 98)),
    )
    v = float(np.clip(value, 0.0, 1.0))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            return mix_color(c0, c1, (v - p0) / max(p1 - p0, 1e-6))
    return stops[-1][1]


def rounded_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color,
    radius: int = 14,
    border: Optional[tuple[int, int, int]] = None,
    border_width: int = 1,
) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, border_width, border_radius=radius)


def blit_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color,
    pos: tuple[int, int],
    *,
    anchor: str = "topleft",
) -> pygame.Rect:
    image = font.render(str(text), True, color)
    rect = image.get_rect()
    setattr(rect, anchor, pos)
    surface.blit(image, rect)
    return rect


def draw_arrow(
    surface: pygame.Surface,
    start: tuple[float, float],
    end: tuple[float, float],
    color,
    width: int = 2,
    head: float = 7.0,
) -> None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    pygame.draw.aaline(surface, color, (sx, sy), (ex, ey))
    if width > 1:
        pygame.draw.line(surface, color, (round(sx), round(sy)), (round(ex), round(ey)), width)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (ex - ux * head + px * head * 0.52, ey - uy * head + py * head * 0.52)
    p2 = (ex - ux * head - px * head * 0.52, ey - uy * head - py * head * 0.52)
    pygame.draw.polygon(surface, color, [(ex, ey), p1, p2])


def find_cjk_font() -> Optional[str]:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    )
    return next((p for p in candidates if Path(p).exists()), None)


class Fonts:
    def __init__(self) -> None:
        font_path = find_cjk_font()
        self.tiny = pygame.font.Font(font_path, 13)
        self.small = pygame.font.Font(font_path, 15)
        self.body = pygame.font.Font(font_path, 17)
        self.label = pygame.font.Font(font_path, 18)
        self.subtitle = pygame.font.Font(font_path, 20)
        self.metric = pygame.font.Font(font_path, 30)
        self.title = pygame.font.Font(font_path, 26)
        self.hero = pygame.font.Font(font_path, 42)


class Button:
    def __init__(self, rect: tuple[int, int, int, int], text: str, shortcut: str):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.shortcut = shortcut

    def draw(self, surface: pygame.Surface, fonts: Fonts, mouse: tuple[int, int]) -> None:
        hover = self.rect.collidepoint(mouse)
        rounded_rect(
            surface,
            self.rect,
            C.PANEL_HOVER if hover else C.PANEL_2,
            11,
            C.CYAN_2 if hover else C.BORDER,
        )
        blit_text(surface, fonts.small, self.text, C.TEXT, (self.rect.x + 14, self.rect.centery), anchor="midleft")
        key_rect = pygame.Rect(self.rect.right - 35, self.rect.y + 8, 25, self.rect.height - 16)
        rounded_rect(surface, key_rect, C.BG_2, 6)
        blit_text(surface, fonts.tiny, self.shortcut, C.TEXT_2, key_rect.center, anchor="center")


class VideoRecorder:
    """把逻辑画布以 RGB24 管道送给 FFmpeg，稳定输出 H.264 MP4。"""

    def __init__(self, size: tuple[int, int], fps: int = 30) -> None:
        self.size = size
        self.fps = fps
        self.ffmpeg = self._find_ffmpeg()
        self.process: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None
        self.started_at = 0.0
        self.last_capture_at = 0.0
        self.frames_written = 0
        self.error = ""

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Prefer a system FFmpeg, then fall back to imageio's bundled binary."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        try:
            import imageio_ffmpeg

            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            return bundled if bundled and Path(bundled).is_file() else None
        except (ImportError, OSError, RuntimeError):
            return None

    @property
    def active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at) if self.active else 0.0

    def start(self, path: Path) -> bool:
        if self.active:
            return True
        if self.ffmpeg is None:
            self.error = "FFmpeg not found; MP4 recording is unavailable"
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.size
        command = [
            self.ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self.process = None
            self.error = str(exc)
            return False
        self.path = path
        self.started_at = time.monotonic()
        self.last_capture_at = 0.0
        self.frames_written = 0
        self.error = ""
        return True

    def capture(self, surface: pygame.Surface, now: Optional[float] = None) -> None:
        if not self.active or self.process is None or self.process.stdin is None:
            return
        now = time.monotonic() if now is None else now
        if self.last_capture_at and now - self.last_capture_at < 1.0 / self.fps:
            return
        self.last_capture_at = now
        try:
            self.process.stdin.write(pygame.image.tobytes(surface, "RGB"))
            self.frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            self.error = f"Recording interrupted: {exc}"
            self.stop()

    def stop(self) -> Optional[Path]:
        process = self.process
        path = self.path
        if process is None:
            return path
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            self.process = None
        return path


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------


class SingleFootDashboardLegacy:
    def __init__(
        self,
        source: DemoSource | BLESource,
        *,
        screenshot_path: Optional[str] = None,
        screenshot_frame: int = 150,
    ) -> None:
        pygame.init()
        pygame.display.set_caption(APP_TITLE)
        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode(LOGICAL_SIZE, flags)
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.source = source
        self.running = True
        self.paused = False
        self.show_heat = True
        self.show_field_lines = True
        self.show_ids = True
        self.show_force_cop = False
        self.mouse_logical = (-100, -100)

        self.last_seq = -1
        self.last_frame_time = 0.0
        self.frame_intervals: deque[float] = deque(maxlen=80)
        self.raw_xyz = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.signal_target = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.filtered = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.noise_sigma = np.full((NUM_SENSORS, 3), 18.0, dtype=np.float32)
        # 小死区保留微小响应；较慢的自适应 EMA 和归零滞回负责抑制抖动。
        self.min_deadzone_counts = 45.0
        self.filter_alpha_rise = 0.13
        self.filter_alpha_fall = 0.075
        self.temp = np.full(NUM_SENSORS, np.nan, dtype=np.float32)
        self.baseline: Optional[np.ndarray] = None
        self.calibration_samples: list[np.ndarray] = []
        self.calibrating = True
        self.calibration_target = 10
        self.calibration_started_at = time.monotonic()
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.calibration_message_until = 0.0
        self.metrics = Metrics()
        self.last_display_at = time.monotonic()
        self.mag_history_short: deque[float] = deque(maxlen=55)
        self.history_peak: deque[float] = deque(maxlen=240)
        self.history_total: deque[float] = deque(maxlen=240)
        self.history_temp: deque[float] = deque(maxlen=240)
        self.history_clock = 0.0
        self.last_history_at = 0.0
        self.last_loop_at = time.monotonic()
        self.display_intervals: deque[float] = deque(maxlen=120)

        self.buttons = {
            "connect": Button((814, 21, 118, 42), "连接设备", "D"),
            "calibrate": Button((942, 21, 90, 42), "校准", "B"),
            "pause": Button((1042, 21, 90, 42), "暂停", "SP"),
            "record": Button((1142, 21, 126, 42), "录制视频", "F9"),
            "shot": Button((1278, 21, 138, 42), "保存截图", "S"),
        }
        self.force_cop_toggle_rect = pygame.Rect(390, 250, 120, 28)
        self.recorder = VideoRecorder(LOGICAL_SIZE, fps=30)
        self.last_recorded_path: Optional[Path] = None
        self.brand_logo: Optional[pygame.Surface] = None
        logo_path = Path(__file__).resolve().parent / "assets" / "mosense_logo.png"
        try:
            self.brand_logo = pygame.image.load(str(logo_path)).convert()
        except (pygame.error, OSError):
            self.brand_logo = None

        self.screenshot_path = screenshot_path
        self.screenshot_frame = max(5, screenshot_frame)
        self.rendered_frames = 0
        self.saved_requested_shot: Optional[Path] = None
        self.toast_text = ""
        self.toast_until = 0.0

        # 足底热图低分辨率缓存；显示时平滑放大。
        self.heat_w, self.heat_h = 180, 480
        yy, xx = np.mgrid[0 : self.heat_h, 0 : self.heat_w]
        self.heat_grid_x = xx / (self.heat_w - 1)
        self.heat_grid_y = yy / (self.heat_h - 1)
        self.foot_mask = self._build_foot_mask(self.heat_w, self.heat_h)
        kernels = []
        for sx, sy in SENSOR_POS:
            dist2 = (self.heat_grid_x - sx) ** 2 + (self.heat_grid_y - sy) ** 2
            kernels.append(np.exp(-dist2 / (2.0 * 0.145 * 0.145)))
        self.heat_kernels = np.asarray(kernels, dtype=np.float32)
        self.heat_kernel_sum = np.maximum(
            np.sum(self.heat_kernels, axis=0),
            1e-5,
        )
        self.static_foot_cache: dict[tuple[str, tuple[int, int], str], pygame.Surface] = {}
        self.force_field_ema = np.zeros((self.heat_h, self.heat_w), dtype=np.float32)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self.last_heat_at = 0.0
        self.force_surface_cache: Optional[pygame.Surface] = None
        self.force_surface_version = 0
        self.display_surface_cache: Optional[pygame.Surface] = None
        self.display_surface_version = -1
        self.display_surface_size: Optional[tuple[int, int]] = None
        # 超分辨磁场网格状态：沿用原实现的上一帧变化检测和箭头滞回。
        self.magnetic_grid_size: Optional[tuple[int, int]] = None
        self.magnetic_grid_points = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_weights = np.zeros((0, NUM_SENSORS), dtype=np.float32)
        self.magnetic_grid_valid = np.zeros((0,), dtype=bool)
        self.magnetic_grid_prev = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_vectors = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_visible = np.zeros((0,), dtype=bool)
        self.magnetic_grid_seq = -2

    @staticmethod
    def _build_foot_mask(width: int, height: int) -> np.ndarray:
        mask_surface = pygame.Surface((width, height))
        mask_surface.fill((0, 0, 0))
        points = [
            (round(float(x) * (width - 1)), round(float(y) * (height - 1)))
            for x, y in INSOLE_OUTLINE
        ]
        pygame.draw.polygon(mask_surface, (255, 255, 255), points)
        return pygame.surfarray.array2d(mask_surface).T != 0

    def request_calibration(self) -> None:
        self.calibration_samples.clear()
        self.calibrating = True
        self.calibration_started_at = time.monotonic()
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.baseline = None
        self.filtered.fill(0.0)
        self.signal_target.fill(0.0)
        self.last_display_at = time.monotonic()
        self.force_field_ema.fill(0.0)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self.last_heat_at = 0.0
        self.force_surface_cache = None
        self.force_surface_version += 1
        self.display_surface_cache = None
        self.display_surface_version = -1
        self.display_surface_size = None
        self.magnetic_grid_prev.fill(0.0)
        self.magnetic_grid_vectors.fill(0.0)
        self.magnetic_grid_visible.fill(False)
        self.magnetic_grid_seq = -2
        self.toast("正在采集空载基线，请保持足底无外力")

    def clear_history(self) -> None:
        self.history_peak.clear()
        self.history_total.clear()
        self.history_temp.clear()
        self.toast("实时趋势已清空")

    def toast(self, text: str, duration: float = 2.2) -> None:
        self.toast_text = text
        self.toast_until = time.monotonic() + duration

    def save_screenshot(self, path: Optional[Path] = None) -> Path:
        if path is None:
            folder = Path.cwd() / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"footsense_{datetime.now():%Y%m%d_%H%M%S}.png"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.canvas, str(path))
        self.saved_requested_shot = path
        self.toast(f"截图已保存：{path.name}")
        return path

    def toggle_device_connection(self) -> None:
        previous = self.source
        previous.stop()
        self.source = BLESource()
        self.source.start()
        message = f"正在重新扫描 {DEVICE_NAME}"

        self.last_seq = -1
        self.last_frame_time = 0.0
        self.frame_intervals.clear()
        self.metrics = Metrics()
        self.request_calibration()
        self.toast(message, 3.0)

    def _finish_calibration(self, *, interrupted: bool = False) -> None:
        """用已经收到的真实帧完成基线；数据断续时也不能把 UI 永久锁住。"""
        if not self.calibration_samples:
            return
        samples = np.stack(self.calibration_samples, axis=0)
        sample_count = len(self.calibration_samples)
        self.baseline = np.median(samples, axis=0).astype(np.float32)
        self.noise_sigma = np.maximum(np.std(samples, axis=0), 1.0).astype(np.float32)
        self.calibration_samples.clear()
        self.calibrating = False
        self.calibration_message_until = time.monotonic() + 1.8
        if interrupted:
            self.toast(f"数据间断，已用 {sample_count} 帧建立基线", 2.8)
        else:
            self.toast("基线校准完成")

    def _update_calibration_watchdog(self) -> None:
        """收到少量帧后若数据暂停，使用已有帧收尾，避免校准界面卡死。"""
        if (
            self.calibrating
            and self.calibration_samples
            and self.calibration_last_sample_at > 0.0
            and time.monotonic() - self.calibration_last_sample_at >= 1.0
        ):
            self._finish_calibration(interrupted=True)

    def toggle_recording(self) -> None:
        if self.recorder.active:
            path = self.recorder.stop()
            self.last_recorded_path = path
            if path is not None:
                self.toast(f"视频已保存：{path.name}", 3.0)
            return
        folder = Path.cwd() / "recordings"
        path = folder / f"footsense_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        if self.recorder.start(path):
            self.toast("开始录制 1440×900 MP4")
        else:
            self.toast(self.recorder.error or "无法开始录制", 3.5)

    def process_frame(self, frame: Optional[SensorFrame]) -> None:
        if frame is None or frame.seq == self.last_seq or self.paused:
            return
        self.last_seq = frame.seq
        if self.last_frame_time > 0:
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
            enough_frames = len(self.calibration_samples) >= self.calibration_target
            low_rate_timeout = (
                len(self.calibration_samples) >= 4
                and now - self.calibration_first_sample_at >= 0.35
            )
            if enough_frames or low_rate_timeout:
                self._finish_calibration()
            return
        if self.baseline is None:
            return

        delta = self.raw_xyz - self.baseline
        # 小型软死区：门限只去除基线噪声，不吞掉细微信号。
        deadzone = np.clip(
            np.maximum(self.min_deadzone_counts, self.noise_sigma * 2.4),
            self.min_deadzone_counts,
            150.0,
        )
        target = np.sign(delta) * np.maximum(np.abs(delta) - deadzone, 0.0)

        # 上升稍快、回落稍慢的自适应 EMA，配合近零归位抑制来回闪动。
        alpha = np.where(
            np.abs(target) > np.abs(self.filtered),
            self.filter_alpha_rise,
            self.filter_alpha_fall,
        )
        self.filtered += alpha * (target - self.filtered)
        quiet = np.linalg.norm(target, axis=1) < self.min_deadzone_counts * 1.25
        self.filtered[quiet] *= 0.82
        self.filtered[np.abs(self.filtered) < 4.0] = 0.0
        self._update_metrics()

    def _update_metrics(self) -> None:
        mags = np.linalg.norm(self.filtered, axis=1)
        peak = float(np.max(mags))
        # 动态显示上限主要用于归一化，不参与原始值计算。
        active_threshold = max(90.0, peak * 0.09)
        active = int(np.count_nonzero(mags >= active_threshold))
        total = float(np.sum(mags))
        # 相对受力采用三轴磁响应合成量；真实录屏中显著变化并不只出现在 Z。
        force_proxy = mags
        weights = force_proxy + mags * 0.12 + 1e-6
        cop_x = float(np.sum(SENSOR_POS[:, 0] * weights) / np.sum(weights))
        cop_y = float(np.sum(SENSOR_POS[:, 1] * weights) / np.sum(weights))

        components = tuple(float(v) for v in np.mean(self.filtered, axis=0))
        lateral = float(np.sum(self.filtered[:, 0]))
        fore_aft = float(np.sum(self.filtered[:, 1]))
        resultant_scale = max(total, 1.0)
        resultant = (
            float(np.clip(lateral / resultant_scale * 3.0, -1.0, 1.0)),
            float(np.clip(fore_aft / resultant_scale * 3.0, -1.0, 1.0)),
        )

        region_raw = [float(np.sum(force_proxy[idx])) for idx in REGION_INDICES]
        region_sum = max(sum(region_raw), 1e-6)
        regions = tuple(v / region_sum for v in region_raw)

        self.mag_history_short.append(total)
        if len(self.mag_history_short) >= 8:
            arr = np.asarray(self.mag_history_short)
            cv = float(np.std(arr) / max(np.mean(arr), 1.0))
            stability = float(np.clip(100.0 - cv * 85.0, 0.0, 100.0))
        else:
            stability = 100.0

        valid_temp = self.temp[np.isfinite(self.temp)]
        mean_temp = float(np.mean(valid_temp)) if valid_temp.size else float("nan")
        min_temp = float(np.min(valid_temp)) if valid_temp.size else float("nan")
        hz = 1.0 / float(np.mean(self.frame_intervals)) if self.frame_intervals else 0.0
        self.metrics = Metrics(
            peak=peak,
            active=active,
            mean_temp=mean_temp,
            min_temp=min_temp,
            hz=hz,
            stability=stability,
            resultant=resultant,
            force_components=components,
            cop=(cop_x, cop_y),
            region_loads=regions,
        )

        now = time.monotonic()
        if now - self.last_history_at >= 0.08:
            self.last_history_at = now
            self.history_peak.append(peak)
            self.history_total.append(total / NUM_SENSORS)
            self.history_temp.append(mean_temp)

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
                    self.toggle_device_connection()
                elif event.key == pygame.K_b:
                    self.request_calibration()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.toast("Display paused" if self.paused else "Display resumed")
                elif event.key == pygame.K_h:
                    self.show_heat = not self.show_heat
                    self.toast("热力图已开启" if self.show_heat else "热力图已隐藏")
                elif event.key == pygame.K_m:
                    self.show_field_lines = not self.show_field_lines
                    self.toast("磁场矢量已开启" if self.show_field_lines else "磁场矢量已隐藏")
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_c:
                    self.show_force_cop = not self.show_force_cop
                    self.toast("力重心已显示" if self.show_force_cop else "力重心已隐藏")
                elif event.key == pygame.K_F9:
                    self.toggle_recording()
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                elif event.key == pygame.K_r:
                    self.clear_history()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = self._screen_to_logical(event.pos)
                if self.buttons["connect"].rect.collidepoint(p):
                    self.toggle_device_connection()
                elif self.buttons["calibrate"].rect.collidepoint(p):
                    self.request_calibration()
                elif self.buttons["pause"].rect.collidepoint(p):
                    self.paused = not self.paused
                    self.toast("画面已暂停" if self.paused else "画面继续")
                elif self.buttons["record"].rect.collidepoint(p):
                    self.toggle_recording()
                elif self.buttons["shot"].rect.collidepoint(p):
                    self.save_screenshot()
                elif self.force_cop_toggle_rect.collidepoint(p):
                    self.show_force_cop = not self.show_force_cop
                    self.toast("力重心已显示" if self.show_force_cop else "力重心已隐藏")

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

    def draw_header(self) -> None:
        pygame.draw.line(self.canvas, C.BORDER, (24, 78), (1416, 78), 1)
        logo_rect = pygame.Rect(24, 9, 198, 60)
        rounded_rect(self.canvas, logo_rect, (33, 31, 31), 10)
        if self.brand_logo is not None:
            logo = pygame.transform.smoothscale(self.brand_logo, (189, 60))
            self.canvas.blit(logo, logo.get_rect(center=logo_rect.center))
        else:
            blit_text(self.canvas, self.fonts.subtitle, "模感科技", C.WHITE, logo_rect.center, anchor="center")
        blit_text(self.canvas, self.fonts.title, "足底多物理场监测", C.TEXT, (242, 20))
        blit_text(self.canvas, self.fonts.tiny, f"MoSense Technology · {APP_VERSION}", C.TEXT_3, (242, 53))

        dot = C.GREEN if self.source.connected else C.RED
        status_rect = pygame.Rect(624, 21, 180, 42)
        rounded_rect(self.canvas, status_rect, C.PANEL, 11, C.BORDER)
        pygame.draw.circle(self.canvas, dot, (642, 42), 5)
        blit_text(self.canvas, self.fonts.small, self.source.status, C.TEXT, (655, 31))
        self.buttons["connect"].text = "重新连接"
        self.buttons["pause"].text = "继续" if self.paused else "暂停"
        self.buttons["record"].text = "停止录制" if self.recorder.active else "录制视频"
        for button in self.buttons.values():
            button.draw(self.canvas, self.fonts, self.mouse_logical)
        if self.recorder.active:
            pygame.draw.circle(self.canvas, C.RED, (1162, 27), 5)
            elapsed = int(self.recorder.elapsed)
            blit_text(
                self.canvas,
                self.fonts.tiny,
                f"REC {elapsed // 60:02d}:{elapsed % 60:02d}",
                C.RED,
                (1216, 68),
                anchor="midtop",
            )

    def draw_metric_cards(self) -> None:
        cards = [
            ("峰值响应", f"{self.metrics.peak:,.0f}", "counts", C.ORANGE, "当前 15 通道最大模长"),
            ("活跃通道", f"{self.metrics.active}", f"/ {NUM_SENSORS}", C.CYAN, "按动态噪声门限统计"),
            ("信号稳定度", f"{self.metrics.stability:.0f}", "%", C.GREEN, "短时总响应波动估计"),
            (
                "环境温度",
                "--" if not np.isfinite(self.metrics.min_temp) else f"{self.metrics.min_temp:.1f}",
                "°C · MIN",
                C.BLUE,
                "15 通道有效温度最小值",
            ),
            ("数据刷新", f"{self.metrics.hz:.1f}", "Hz", C.MAGENTA, self.source.detail),
        ]
        x0, y, gap, width, height = 24, 91, 12, 268, 83
        for i, (label, value, unit, accent, hint) in enumerate(cards):
            rect = pygame.Rect(x0 + i * (width + gap), y, width, height)
            rounded_rect(self.canvas, rect, C.PANEL, 13, C.BORDER)
            pygame.draw.rect(
                self.canvas,
                accent,
                (rect.x, rect.y + 12, 3, rect.height - 24),
                border_radius=2,
            )
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_2, (rect.x + 17, rect.y + 9))
            value_rect = blit_text(
                self.canvas,
                self.fonts.metric,
                value,
                C.TEXT,
                (rect.x + 17, rect.y + 25),
            )
            blit_text(
                self.canvas,
                self.fonts.tiny,
                unit,
                accent,
                (value_rect.right + 8, rect.y + 43),
                anchor="midleft",
            )
            blit_text(self.canvas, self.fonts.tiny, hint, C.TEXT_3, (rect.x + 17, rect.bottom - 17))

    def _build_scalar_surface(self, raw_values: np.ndarray, *, kind: str) -> pygame.Surface:
        """把 15 个离散值平滑成鞋垫内连续标量场。"""
        raw_values = np.maximum(np.asarray(raw_values, dtype=np.float32), 0.0)
        is_new_force_sample = kind == "force" and self.force_field_seq != self.last_seq
        if kind == "force":
            # 量程快速跟随增大的力、缓慢回落，避免每帧自动拉满造成颜色不变。
            target_vmax = max(850.0, float(np.percentile(raw_values, 95)) * 1.12)
            if is_new_force_sample:
                rate = 0.08 if target_vmax > self.force_vmax else 0.006
                self.force_vmax += rate * (target_vmax - self.force_vmax)
                self.force_vmax = float(np.clip(self.force_vmax, 850.0, 9000.0))
            peak_ref = self.force_vmax
            # 轻压区适度提升，重压区仍保留进入暖色的空间。
            values = np.clip(raw_values / peak_ref, 0.0, 1.0) ** 0.72
        else:
            peak_ref = max(3600.0, float(np.percentile(raw_values, 95)) * 1.05)
            values = np.clip(raw_values / peak_ref, 0.0, 1.0)
        field = np.zeros((self.heat_h, self.heat_w), dtype=np.float32)
        weight = np.zeros_like(field)
        sigma = 0.145 if kind == "force" else 0.090
        for (sx, sy), value in zip(SENSOR_POS, values):
            dist2 = (self.heat_grid_x - sx) ** 2 + (self.heat_grid_y - sy) ** 2
            w = np.exp(-dist2 / (2.0 * sigma * sigma))
            field += w * float(value)
            weight += w
        field = np.divide(field, np.maximum(weight, 1e-5))
        field *= self.foot_mask

        if kind == "force":
            # 一次掩膜感知的邻域平滑，消除传感器 Voronoi 式分块边界。
            mask_f = self.foot_mask.astype(np.float32)
            padded_f = np.pad(field, 1, mode="constant")
            padded_m = np.pad(mask_f, 1, mode="constant")
            numerator = (
                padded_f[1:-1, 1:-1] * 4.0
                + padded_f[:-2, 1:-1]
                + padded_f[2:, 1:-1]
                + padded_f[1:-1, :-2]
                + padded_f[1:-1, 2:]
            )
            denominator = (
                padded_m[1:-1, 1:-1] * 4.0
                + padded_m[:-2, 1:-1]
                + padded_m[2:, 1:-1]
                + padded_m[1:-1, :-2]
                + padded_m[1:-1, 2:]
            )
            field = np.where(
                self.foot_mask,
                numerator / np.maximum(denominator, 1e-5),
                0.0,
            )
            # 热图拥有独立时间滤波，不让颜色跟着单帧数据跳变。
            if is_new_force_sample:
                self.force_field_ema += 0.18 * (field - self.force_field_ema)
                self.force_field_seq = self.last_seq
            field = self.force_field_ema

        rgb = np.zeros((self.heat_h, self.heat_w, 3), dtype=np.uint8)
        if kind == "force":
            stop_p = np.array([0.0, 0.18, 0.40, 0.62, 0.82, 1.0])
            stop_c = np.array(
                [
                    (235, 243, 244),
                    (190, 220, 224),
                    (111, 185, 191),
                    (226, 207, 123),
                    (231, 143, 96),
                    (210, 85, 98),
                ],
                dtype=np.float32,
            )
        else:
            stop_p = np.array([0.0, 0.22, 0.45, 0.66, 0.84, 1.0])
            stop_c = np.array(
                [
                    (222, 235, 237),
                    (164, 207, 205),
                    (126, 190, 172),
                    (226, 204, 133),
                    (230, 155, 112),
                    (211, 104, 111),
                ],
                dtype=np.float32,
            )
        for channel in range(3):
            rgb[:, :, channel] = np.interp(field, stop_p, stop_c[:, channel]).astype(np.uint8)
        alpha = np.where(self.foot_mask, 246, 0).astype(np.uint8)

        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = alpha.T
        return surface

    def _inside_foot(self, x: float, y: float) -> bool:
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return False
        ix = int(np.clip(x * (self.heat_w - 1), 0, self.heat_w - 1))
        iy = int(np.clip(y * (self.heat_h - 1), 0, self.heat_h - 1))
        return bool(self.foot_mask[iy, ix])

    def _prepare_magnetic_idw_grid(self, rect: pygame.Rect) -> None:
        """构建与原 superres_hot 相同结构的规则网格及归一化 IDW 权重。"""
        size = (rect.w, rect.h)
        if self.magnetic_grid_size == size:
            return

        # 原实现使用 29 px；当前磁场鞋垫更窄，采用 22 px 保持相近视觉密度。
        step = 22.0
        xs = np.arange(step / 2.0, float(rect.w), step, dtype=np.float32)
        ys = np.arange(step / 2.0, float(rect.h), step, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1).astype(np.float32)

        sensor_pixels = SENSOR_POS * np.array([rect.w, rect.h], dtype=np.float32)
        diff = points[:, None, :] - sensor_pixels[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        # GRID_IDW_POWER=2.0，GRID_IDW_EPS_PX=6.0。
        weights = 1.0 / (dist2 + 6.0 * 6.0)
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
        valid = np.array(
            [
                self._inside_foot(float(x / rect.w), float(y / rect.h))
                for x, y in points
            ],
            dtype=bool,
        )

        count = len(points)
        self.magnetic_grid_size = size
        self.magnetic_grid_points = points
        self.magnetic_grid_weights = weights.astype(np.float32)
        self.magnetic_grid_valid = valid
        self.magnetic_grid_prev = np.zeros((count, 2), dtype=np.float32)
        self.magnetic_grid_vectors = np.zeros((count, 2), dtype=np.float32)
        self.magnetic_grid_visible = np.zeros((count,), dtype=bool)
        self.magnetic_grid_seq = -2

    def _sensor_xy_in_foot_coordinates(self) -> np.ndarray:
        """复制原实现的芯片坐标翻转与安装角旋转。"""
        delta_xy = -self.filtered[:, :2]
        cos_r = np.cos(CHIP_XY_ROTATIONS)
        sin_r = np.sin(CHIP_XY_ROTATIONS)
        return np.column_stack(
            (
                cos_r * delta_xy[:, 0] - sin_r * delta_xy[:, 1],
                sin_r * delta_xy[:, 0] + cos_r * delta_xy[:, 1],
            )
        ).astype(np.float32)

    def draw_magnetic_idw_arrows(self, rect: pygame.Rect) -> None:
        """规则网格 IDW + 固定锚点 + 条件箭头滞回，复刻 superres_hot 画法。"""
        self._prepare_magnetic_idw_grid(rect)
        if not len(self.magnetic_grid_points):
            return

        vectors = self.magnetic_grid_weights @ self._sensor_xy_in_foot_coordinates()
        is_new_sample = self.magnetic_grid_seq != self.last_seq
        if is_new_sample:
            # 原实现参数：开 8、关 6、变化开 4、变化关 4×0.35。
            magnitude2 = np.sum(vectors * vectors, axis=1)
            changes = np.linalg.norm(vectors - self.magnetic_grid_prev, axis=1)
            raw_on = (magnitude2 >= 8.0 * 8.0) | (changes >= 4.0)
            raw_off = (magnitude2 < 6.0 * 6.0) & (changes < 4.0 * 0.35)
            self.magnetic_grid_visible = np.where(
                self.magnetic_grid_visible,
                ~raw_off,
                raw_on,
            )
            self.magnetic_grid_prev[:] = vectors
            self.magnetic_grid_vectors[:] = vectors
            self.magnetic_grid_seq = self.last_seq

        anchor_color = (80, 145, 124)
        arrow_color = (55, 168, 119)
        for point, valid in zip(self.magnetic_grid_points, self.magnetic_grid_valid):
            if not valid:
                continue
            center = (round(rect.x + float(point[0])), round(rect.y + float(point[1])))
            pygame.draw.circle(self.canvas, anchor_color, center, 2)

        # 原实现 SCALING_FACTOR=40；仅限制极端异常包的显示长度，不改变方向。
        for point, vector, valid, visible in zip(
            self.magnetic_grid_points,
            self.magnetic_grid_vectors,
            self.magnetic_grid_valid,
            self.magnetic_grid_visible,
        ):
            if not valid or not visible:
                continue
            display_vec = vector / 40.0
            length = float(np.linalg.norm(display_vec))
            if length > 21.0:
                display_vec *= 21.0 / length
            start = np.array([rect.x + point[0], rect.y + point[1]], dtype=np.float32)
            end = start + display_vec
            start_px = (round(float(start[0])), round(float(start[1])))
            end_px = (round(float(end[0])), round(float(end[1])))
            pygame.draw.line(self.canvas, arrow_color, start_px, end_px, 2)
            pygame.draw.circle(self.canvas, arrow_color, end_px, 2)

    def draw_foot_panel(self) -> None:
        card = pygame.Rect(24, 190, 1008, 668)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "鞋垫多物理场", C.TEXT, (44, 205))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "左：相对受力强度   ·   右：15 点 IDW 超分辨磁场矢量   ·   脚尖始终朝上",
            C.TEXT_3,
            (44, 231),
        )
        pygame.draw.line(self.canvas, C.BORDER, (528, 249), (528, 844), 1)

        blit_text(self.canvas, self.fonts.small, "受力强度", C.TEXT, (45, 253))
        blit_text(self.canvas, self.fonts.tiny, "|ΔB| 三轴合成 · 冷色低、暖色高", C.TEXT_3, (45, 276))
        toggle_hover = self.force_cop_toggle_rect.collidepoint(self.mouse_logical)
        rounded_rect(
            self.canvas,
            self.force_cop_toggle_rect,
            C.PANEL_HOVER if toggle_hover else C.PANEL_2,
            8,
            C.CYAN_2 if self.show_force_cop else C.BORDER,
        )
        pygame.draw.circle(
            self.canvas,
            C.CYAN if self.show_force_cop else C.TEXT_3,
            (self.force_cop_toggle_rect.x + 13, self.force_cop_toggle_rect.centery),
            4,
        )
        blit_text(
            self.canvas,
            self.fonts.tiny,
            ("隐藏" if self.show_force_cop else "显示") + "力重心  C",
            C.TEXT_2,
            (self.force_cop_toggle_rect.x + 23, self.force_cop_toggle_rect.y + 5),
        )
        blit_text(self.canvas, self.fonts.small, "磁场分布", C.TEXT, (550, 253))
        field_mode = "规则网格 IDW · 固定锚点 · 滞回矢量箭头"
        blit_text(self.canvas, self.fonts.tiny, field_mode, C.TEXT_3, (550, 276))

        force_rect = pygame.Rect(274, 296, 232, 550)
        magnetic_rect = pygame.Rect(786, 296, 232, 550)
        mags = np.linalg.norm(self.filtered, axis=1)
        force_values = mags

        # 左侧力强度图
        force_base = (
            self._build_scalar_surface(force_values, kind="force")
            if self.show_heat
            else self._mask_surface((237, 243, 239, 255))
        )
        self.canvas.blit(pygame.transform.smoothscale(force_base, force_rect.size), force_rect.topleft)
        self.canvas.blit(
            pygame.transform.smoothscale(self._outline_surface(C.CYAN_2), force_rect.size),
            force_rect.topleft,
        )
        if self.show_force_cop:
            cop_x = force_rect.x + self.metrics.cop[0] * force_rect.w
            cop_y = force_rect.y + self.metrics.cop[1] * force_rect.h
            pygame.draw.circle(self.canvas, C.INK, (round(cop_x), round(cop_y)), 9, 1)
            pygame.draw.circle(self.canvas, C.INK, (round(cop_x), round(cop_y)), 2)
            pygame.draw.line(self.canvas, C.INK, (cop_x - 12, cop_y), (cop_x + 12, cop_y), 1)
            pygame.draw.line(self.canvas, C.INK, (cop_x, cop_y - 12), (cop_x, cop_y + 12), 1)
            blit_text(self.canvas, self.fonts.tiny, "力重心", C.INK, (cop_x + 10, cop_y - 17))
        blit_text(self.canvas, self.fonts.tiny, "区域力占比", C.TEXT_2, (45, 337))

        # 区域力占比与单调色标
        region_colors = (C.ORANGE, C.CYAN, C.BLUE)
        for i, (name, value, color) in enumerate(
            zip(REGION_NAMES, self.metrics.region_loads, region_colors)
        ):
            y = 365 + i * 51
            blit_text(self.canvas, self.fonts.tiny, name, C.TEXT_2, (45, y))
            blit_text(self.canvas, self.fonts.small, f"{value * 100:4.1f}%", C.TEXT, (244, y - 2), anchor="topright")
            track = pygame.Rect(45, y + 23, 199, 6)
            pygame.draw.rect(self.canvas, C.GRID, track, border_radius=4)
            fill = track.copy()
            fill.width = max(2, round(track.w * value))
            pygame.draw.rect(self.canvas, color, fill, border_radius=4)

        blit_text(self.canvas, self.fonts.tiny, "轻", C.TEXT_3, (45, 641))
        for i in range(151):
            pygame.draw.line(
                self.canvas,
                force_color(i / 150.0),
                (68 + i, 644),
                (68 + i, 653),
            )
        blit_text(self.canvas, self.fonts.tiny, "重", C.TEXT_3, (226, 641))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "未进行 N 标定，仅用于相对变化",
            C.TEXT_3,
            (45, 672),
        )

        # 右侧只保留中性鞋垫底图、IDW 网格箭头和 15 个实测采样点。
        magnetic_base = self._mask_surface((245, 248, 247, 255))
        self.canvas.blit(
            pygame.transform.smoothscale(magnetic_base, magnetic_rect.size),
            magnetic_rect.topleft,
        )
        if self.show_field_lines:
            self.draw_magnetic_idw_arrows(magnetic_rect)
        self.canvas.blit(
            pygame.transform.smoothscale(self._outline_surface(C.BLUE), magnetic_rect.size),
            magnetic_rect.topleft,
        )

        peak_ref = max(3000.0, float(np.max(mags)))
        for idx, ((nx, ny), vec, mag) in enumerate(zip(SENSOR_POS, self.filtered, mags), start=1):
            x = magnetic_rect.x + float(nx) * magnetic_rect.w
            y = magnetic_rect.y + float(ny) * magnetic_rect.h
            strength = float(np.clip(mag / peak_ref, 0.0, 1.0))
            color = mix_color(C.CYAN_2, C.CYAN, strength)
            pygame.draw.circle(self.canvas, C.WHITE, (round(x), round(y)), 6)
            pygame.draw.circle(self.canvas, C.INK, (round(x), round(y)), 5, 1)
            pygame.draw.circle(self.canvas, color, (round(x), round(y)), 3)
            if self.show_ids:
                blit_text(
                    self.canvas,
                    self.fonts.tiny,
                    f"{idx:02d}",
                    C.TEXT,
                    (round(x) + 7, round(y) - 8),
                )

        pygame.draw.circle(self.canvas, (80, 145, 124), (550, 653), 2)
        blit_text(self.canvas, self.fonts.tiny, "IDW 网格锚点", C.TEXT_3, (560, 644))
        pygame.draw.line(self.canvas, (55, 168, 119), (663, 660), (677, 646), 2)
        pygame.draw.circle(self.canvas, (55, 168, 119), (677, 646), 2)
        blit_text(self.canvas, self.fonts.tiny, "磁响应矢量", C.TEXT_3, (685, 644))

        y = 695
        for label, enabled, color in (
            ("力热图 H", self.show_heat, C.ORANGE),
            ("磁矢量 M", self.show_field_lines, C.CYAN),
            ("编号 I", self.show_ids, C.BLUE),
        ):
            rect = pygame.Rect(550, y, 122, 24)
            rounded_rect(self.canvas, rect, C.PANEL_2, 7)
            pygame.draw.circle(self.canvas, color if enabled else C.TEXT_3, (563, y + 12), 4)
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_2, (573, y + 4))
            y += 28

    def _mask_surface(self, color: tuple[int, int, int, int]) -> pygame.Surface:
        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        rgba = np.zeros((self.heat_h, self.heat_w, 4), dtype=np.uint8)
        rgba[self.foot_mask] = color
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        return surface

    def _outline_surface(self, color=C.CYAN_2) -> pygame.Surface:
        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        mask = self.foot_mask
        edge = mask & (
            ~np.roll(mask, 1, axis=0)
            | ~np.roll(mask, -1, axis=0)
            | ~np.roll(mask, 1, axis=1)
            | ~np.roll(mask, -1, axis=1)
        )
        rgba = np.zeros((self.heat_h, self.heat_w, 4), dtype=np.uint8)
        rgba[edge] = (*color, 210)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        return surface

    def draw_force_card(self) -> None:
        card = pygame.Rect(1048, 190, 368, 244)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "受力向量示意", C.TEXT, (1066, 205))
        blit_text(self.canvas, self.fonts.tiny, "磁响应分量映射 · 未标定为 N", C.TEXT_3, (1066, 230))

        center = np.array([1125.0, 334.0])
        basis_x = np.array([0.88, 0.44])
        basis_y = np.array([-0.82, 0.52])
        basis_z = np.array([0.0, -1.0])
        for basis, label in ((basis_x, "X"), (basis_y, "Y"), (basis_z, "Z")):
            end = center + basis * 50.0
            draw_arrow(self.canvas, center, end, C.BORDER, 1, 5)
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_3, tuple(end.astype(int)), anchor="center")

        components = np.asarray(self.metrics.force_components, dtype=np.float32)
        max_component = max(420.0, float(np.max(np.abs(components))))
        axes = (basis_x, basis_y, basis_z)
        colors = (C.ORANGE, C.BLUE, C.CYAN)
        for value, basis, color in zip(components, axes, colors):
            end = center + basis * (float(value) / max_component) * 40.0
            draw_arrow(self.canvas, center, end, color, 2, 7)

        projected = (
            basis_x * components[0]
            + basis_y * components[1]
            + basis_z * components[2]
        )
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm > 1.0:
            projected = projected / max(projected_norm, max_component) * 54.0
            draw_arrow(self.canvas, center, center + projected, C.MAGENTA, 3, 9)
        pygame.draw.circle(self.canvas, C.INK, tuple(center.astype(int)), 4)

        # 三轴有符号条形，中心线为零。
        labels = ("Fx", "Fy", "Fz")
        for i, (label, value, color) in enumerate(zip(labels, components, colors)):
            y = 272 + i * 43
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_2, (1204, y))
            blit_text(self.canvas, self.fonts.tiny, f"{value:+.0f}", C.TEXT, (1397, y), anchor="topright")
            track = pygame.Rect(1231, y + 6, 130, 7)
            pygame.draw.rect(self.canvas, C.GRID, track, border_radius=4)
            pygame.draw.line(self.canvas, C.TEXT_3, (track.centerx, track.y - 2), (track.centerx, track.bottom + 2), 1)
            half = track.w // 2
            bar = round(np.clip(abs(float(value)) / max_component, 0.0, 1.0) * half)
            if value >= 0:
                fill = pygame.Rect(track.centerx, track.y, max(2, bar), track.h)
            else:
                fill = pygame.Rect(track.centerx - max(2, bar), track.y, max(2, bar), track.h)
            pygame.draw.rect(self.canvas, color, fill, border_radius=4)

        state = "暂停" if self.paused else ("校准中" if self.calibrating else "实时")
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"采集 {state} · α {self.filter_alpha_rise:.2f}/{self.filter_alpha_fall:.3f} · DZ {self.min_deadzone_counts:.0f}",
            C.TEXT_3,
            (1066, 409),
        )

    def draw_channel_card(self) -> None:
        card = pygame.Rect(1048, 450, 368, 220)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "15 通道状态", C.TEXT, (1066, 465))
        blit_text(self.canvas, self.fonts.tiny, "圆环=|ΔB| · 条形=|ΔBz|", C.TEXT_3, (1066, 490))

        mags = np.linalg.norm(self.filtered, axis=1)
        peak_ref = max(3000.0, float(np.max(mags)))
        for i in range(NUM_SENSORS):
            col, row = i % 5, i // 5
            x = 1065 + col * 68
            y = 518 + row * 45
            strength = float(np.clip(mags[i] / peak_ref, 0.0, 1.0))
            color = heat_color(strength)
            pygame.draw.circle(self.canvas, C.GRID, (x + 9, y + 9), 8)
            pygame.draw.arc(
                self.canvas,
                color,
                pygame.Rect(x + 1, y + 1, 16, 16),
                math.pi / 2,
                math.pi / 2 + 2.0 * math.pi * max(strength, 0.025),
                2,
            )
            blit_text(self.canvas, self.fonts.tiny, f"{i + 1:02d}", C.TEXT, (x + 9, y + 9), anchor="center")
            track = pygame.Rect(x + 21, y + 6, 36, 5)
            pygame.draw.rect(self.canvas, C.GRID, track, border_radius=3)
            z_strength = float(np.clip(abs(self.filtered[i, 2]) / peak_ref, 0.0, 1.0))
            fill = track.copy()
            fill.width = max(2, round(track.w * z_strength))
            pygame.draw.rect(self.canvas, force_color(z_strength), fill, border_radius=3)
            blit_text(
                self.canvas,
                self.fonts.tiny,
                f"{mags[i]:.0f}",
                C.TEXT_3,
                (x + 21, y + 17),
            )

    def draw_trend_card(self) -> None:
        card = pygame.Rect(1048, 686, 368, 172)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "实时趋势", C.TEXT, (1066, 701))
        blit_text(self.canvas, self.fonts.tiny, "最近约 19 秒 · R 清空", C.TEXT_3, (1066, 726))

        chart = pygame.Rect(1066, 751, 332, 88)
        for i in range(4):
            y = chart.y + round(chart.h * i / 3)
            pygame.draw.line(self.canvas, C.GRID, (chart.x, y), (chart.right, y), 1)
        for i in range(7):
            x = chart.x + round(chart.w * i / 6)
            pygame.draw.line(self.canvas, C.GRID, (x, chart.y), (x, chart.bottom), 1)

        self._draw_series(chart, self.history_total, C.CYAN, 1.0)
        self._draw_series(chart, self.history_peak, C.ORANGE, 1.0)
        blit_text(self.canvas, self.fonts.tiny, "平均", C.CYAN, (1281, 706))
        pygame.draw.line(self.canvas, C.CYAN, (1264, 713), (1277, 713), 2)
        blit_text(self.canvas, self.fonts.tiny, "峰值", C.ORANGE, (1350, 706))
        pygame.draw.line(self.canvas, C.ORANGE, (1333, 713), (1346, 713), 2)

    def _draw_series(
        self,
        rect: pygame.Rect,
        values: deque[float],
        color,
        _width: float,
    ) -> None:
        if len(values) < 2:
            return
        arr = np.asarray(values, dtype=np.float32)
        hi = max(4200.0, float(np.percentile(np.asarray(self.history_peak or [1]), 98)) * 1.10)
        xs = np.linspace(rect.x, rect.right, len(arr))
        ys = rect.bottom - np.clip(arr / hi, 0.0, 1.0) * rect.h
        points = [(round(x), round(y)) for x, y in zip(xs, ys)]
        pygame.draw.aalines(self.canvas, color, False, points)
        if len(points) >= 2:
            pygame.draw.lines(self.canvas, color, False, points, 2)

    def draw_footer(self) -> None:
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "D 重新连接   B 校准   SPACE 暂停   H 力热图   M 磁矢量   C 力重心   I 编号   F9 录制   S 截图   ESC 退出",
            C.TEXT_3,
            (24, 874),
        )
        source = "DEMO / 仿真" if isinstance(self.source, DemoSource) else "BLE / REALTIME"
        blit_text(self.canvas, self.fonts.tiny, source, C.TEXT_3, (1416, 874), anchor="topright")

    def draw_overlay(self) -> None:
        now = time.monotonic()
        show_calibration = self.calibrating and bool(self.calibration_samples)
        if show_calibration:
            panel = pygame.Rect(451, 442, 540, 112)
            veil = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
            veil.fill((226, 233, 231, 170))
            self.canvas.blit(veil, (0, 0))
            rounded_rect(self.canvas, panel, C.PANEL, 16, C.YELLOW)
            blit_text(self.canvas, self.fonts.subtitle, "正在建立空载基线", C.TEXT, (721, 463), anchor="midtop")
            blit_text(
                self.canvas,
                self.fonts.small,
                "请保持足底无外力，完成后将自动进入实时显示",
                C.TEXT_2,
                (721, 496),
                anchor="midtop",
            )
            progress = pygame.Rect(501, 528, 440, 7)
            pygame.draw.rect(self.canvas, C.GRID, progress, border_radius=4)
            fill = progress.copy()
            fill.width = round(
                progress.w
                * np.clip(len(self.calibration_samples) / self.calibration_target, 0.0, 1.0)
            )
            pygame.draw.rect(self.canvas, C.YELLOW, fill, border_radius=4)
        elif self.paused:
            pill = pygame.Rect(654, 93, 132, 34)
            rounded_rect(self.canvas, pill, C.YELLOW, 10)
            blit_text(self.canvas, self.fonts.small, "已暂停", C.INK, pill.center, anchor="center")

        if self.toast_text and now < self.toast_until and not show_calibration:
            image = self.fonts.small.render(self.toast_text, True, C.TEXT)
            toast = pygame.Rect(0, 0, image.get_width() + 34, 42)
            toast.midbottom = (LOGICAL_SIZE[0] // 2, 851)
            rounded_rect(self.canvas, toast, C.PANEL, 12, C.CYAN_2)
            self.canvas.blit(image, image.get_rect(center=toast.center))

    def render(self) -> None:
        self.canvas.fill(C.BG)
        # 很轻的背景光晕，让面板层级更明确。
        glow = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(glow, (128, 195, 181, 20), (415, 430), 410)
        pygame.draw.circle(glow, (154, 176, 206, 16), (1280, 650), 310)
        self.canvas.blit(glow, (0, 0))

        self.draw_header()
        self.draw_metric_cards()
        self.draw_foot_panel()
        self.draw_force_card()
        self.draw_channel_card()
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
                self.process_frame(self.source.sample())
                self._update_calibration_watchdog()
                self.render()
                self.recorder.capture(self.canvas)
                self.rendered_frames += 1
                if self.screenshot_path and self.rendered_frames >= self.screenshot_frame:
                    path = self.save_screenshot(Path(self.screenshot_path).resolve())
                    # 重绘一次，使截图保存提示不影响目标截图。
                    self.saved_requested_shot = path
                    break
                self.clock.tick(FPS)
        finally:
            if self.recorder.active:
                self.last_recorded_path = self.recorder.stop()
            self.source.stop()
            pygame.quit()
        return self.saved_requested_shot


class FootViewState:
    """UI 侧的一只脚状态；基线、滤波、量程和历史全部彼此独立。"""

    def __init__(self, side: str, heat_shape: tuple[int, int]) -> None:
        self.side = side
        self.source_session = -1
        self.last_seq = -1
        self.last_frame_time = 0.0
        self.frame_intervals: deque[float] = deque(maxlen=80)
        self.raw_xyz = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.signal_target = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.filtered = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.noise_sigma = np.full((NUM_SENSORS, 3), 18.0, dtype=np.float32)
        self.temp = np.full(NUM_SENSORS, np.nan, dtype=np.float32)
        self.baseline: Optional[np.ndarray] = None
        self.calibration_samples: list[np.ndarray] = []
        self.calibrating = True
        self.calibration_target = 24
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.contact_active = False
        self.contact_peak_total = 0.0
        self.contact_peak_sensor = 0.0
        self.release_candidate_duration = 0.0
        self.release_zero_remaining = 0.0
        self.metrics = Metrics()
        self.last_display_at = time.monotonic()
        self.mag_history_short: deque[float] = deque(maxlen=55)
        self.history_total: deque[float] = deque(maxlen=240)
        heat_h, heat_w = heat_shape
        self.force_field_ema = np.zeros((heat_h, heat_w), dtype=np.float32)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self.last_heat_at = 0.0
        self.force_surface_cache: Optional[pygame.Surface] = None
        self.force_surface_version = 0
        self.display_surface_cache: Optional[pygame.Surface] = None
        self.display_surface_version = -1
        self.display_surface_size: Optional[tuple[int, int]] = None
        self.magnetic_grid_size: Optional[tuple[int, int]] = None
        self.magnetic_grid_points = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_weights = np.zeros((0, NUM_SENSORS), dtype=np.float32)
        self.magnetic_grid_valid = np.zeros((0,), dtype=bool)
        self.magnetic_grid_prev = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_vectors = np.zeros((0, 2), dtype=np.float32)
        self.magnetic_grid_visible = np.zeros((0,), dtype=bool)
        self.magnetic_grid_seq = -2

    def reset_calibration(self) -> None:
        self.calibration_samples.clear()
        self.calibrating = True
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.baseline = None
        self.contact_active = False
        self.contact_peak_total = 0.0
        self.contact_peak_sensor = 0.0
        self.release_candidate_duration = 0.0
        self.release_zero_remaining = 0.0
        self.last_seq = -1
        self.last_frame_time = 0.0
        self.frame_intervals.clear()
        self.filtered.fill(0.0)
        self.signal_target.fill(0.0)
        self.last_display_at = time.monotonic()
        self.force_field_ema.fill(0.0)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
        self.last_heat_at = 0.0
        self.force_surface_cache = None
        self.force_surface_version += 1
        self.display_surface_cache = None
        self.display_surface_version = -1
        self.display_surface_size = None
        self.magnetic_grid_prev.fill(0.0)
        self.magnetic_grid_vectors.fill(0.0)
        self.magnetic_grid_visible.fill(False)
        self.magnetic_grid_seq = -2
        self.metrics = Metrics()


class WeightControlWindow:
    """Independent Tk process for adjusting bilateral load weights."""

    MIN_WEIGHT = 0.10
    MAX_WEIGHT = 3.00

    def __init__(self, initial_weights: dict[str, float]) -> None:
        self.initial_weights = dict(initial_weights)
        self.context = mp.get_context("spawn")
        self.updates = self.context.Queue()
        self.stop_event = self.context.Event()
        self.process: Optional[mp.Process] = None

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.stop_event.clear()
        self.process = self.context.Process(
            target=self._run_process,
            args=(self.initial_weights, self.updates, self.stop_event),
            name="load-weight-controls",
            daemon=True,
        )
        self.process.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is None:
            return
        self.process.join(timeout=1.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)

    def close_queue(self) -> None:
        self.updates.close()
        self.updates.join_thread()

    @staticmethod
    def _run_process(initial_weights, updates, stop_event) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk

            min_weight = WeightControlWindow.MIN_WEIGHT
            max_weight = WeightControlWindow.MAX_WEIGHT

            root = tk.Tk()
            root.title("Left / Right Load Weights")
            root.resizable(False, False)
            root.configure(background="#101720")

            window_width, window_height = 460, 286
            screen_width = root.winfo_screenwidth()
            x_value = max(20, screen_width - window_width - 35)
            root.geometry(f"{window_width}x{window_height}+{x_value}+95")

            style = ttk.Style(root)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("Weights.TFrame", background="#101720")
            style.configure(
                "Weights.TLabel",
                background="#101720",
                foreground="#dce8ee",
                font=("DejaVu Sans", 11),
            )
            style.configure(
                "Title.Weights.TLabel",
                background="#101720",
                foreground="#f1f6f8",
                font=("DejaVu Sans", 14, "bold"),
            )
            style.configure(
                "Value.Weights.TLabel",
                background="#101720",
                foreground="#8bd7cb",
                font=("DejaVu Sans Mono", 11, "bold"),
            )
            style.configure("Weights.Horizontal.TScale", background="#101720")

            content = ttk.Frame(root, padding=(22, 18), style="Weights.TFrame")
            content.pack(fill="both", expand=True)
            ttk.Label(
                content,
                text="Load Balance Weights",
                style="Title.Weights.TLabel",
            ).grid(row=0, column=0, columnspan=3, sticky="w")
            ttk.Label(
                content,
                text="Adjusts displayed load ratios; raw sensor data is unchanged.",
                style="Weights.TLabel",
            ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 15))

            values = {
                side: tk.DoubleVar(value=initial_weights.get(side, 1.0))
                for side in SIDES
            }
            value_labels: dict[str, ttk.Label] = {}
            ratio_value = tk.StringVar()

            def refresh_ratio() -> None:
                left_value = max(min_weight, float(values["left"].get()))
                right_value = max(min_weight, float(values["right"].get()))
                total = left_value + right_value
                ratio_value.set(
                    f"Weight ratio   L {left_value / total * 100:5.1f}%"
                    f"   /   R {right_value / total * 100:5.1f}%"
                )
                for side in SIDES:
                    value_labels[side].configure(text=f"{values[side].get():.2f}")

            def send_value(side: str, raw_value: str) -> None:
                value = round(
                    min(max_weight, max(min_weight, float(raw_value))),
                    2,
                )
                refresh_ratio()
                updates.put(("set", side, value))

            for row, side in enumerate(SIDES, start=2):
                ttk.Label(
                    content,
                    text=SIDE_NAMES[side],
                    style="Weights.TLabel",
                ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
                scale = ttk.Scale(
                    content,
                    from_=min_weight,
                    to=max_weight,
                    orient="horizontal",
                    length=285,
                    variable=values[side],
                    command=lambda raw, current_side=side: send_value(current_side, raw),
                    style="Weights.Horizontal.TScale",
                )
                scale.grid(row=row, column=1, sticky="ew", pady=8)
                value_labels[side] = ttk.Label(
                    content,
                    text=f"{values[side].get():.2f}",
                    width=5,
                    anchor="e",
                    style="Value.Weights.TLabel",
                )
                value_labels[side].grid(row=row, column=2, sticky="e", padx=(12, 0))

            ttk.Label(
                content,
                textvariable=ratio_value,
                style="Weights.TLabel",
            ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 8))

            def reset_weights() -> None:
                for side in SIDES:
                    values[side].set(1.0)
                    updates.put(("set", side, 1.0))
                refresh_ratio()

            ttk.Button(content, text="Reset to 1 : 1", command=reset_weights).grid(
                row=5,
                column=0,
                columnspan=3,
                sticky="e",
                pady=(4, 0),
            )
            content.columnconfigure(1, weight=1)
            refresh_ratio()

            def poll_stop() -> None:
                if stop_event.is_set():
                    root.destroy()
                    return
                root.after(100, poll_stop)

            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.after(100, poll_stop)
            root.after(100, lambda: root.attributes("-topmost", True))
            root.after(900, lambda: root.attributes("-topmost", False))
            root.mainloop()
        except Exception as exc:  # Tk failures must not stop BLE visualization.
            updates.put(("error", str(exc)))


class DualDashboard:
    def __init__(
        self,
        source: DualDemoSource | DualBLESource,
        *,
        scan_timeout: float = 8.0,
        load_weights: Optional[dict[str, float]] = None,
        settings_path: Optional[Path] = None,
        screenshot_path: Optional[str] = None,
        screenshot_frame: int = 150,
        auto_record_path: Optional[Path] = None,
        auto_record_seconds: float = 0.0,
        auto_record_delay: float = 0.8,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Dual-Foot Multiphysics Monitor")
        self.screen = pygame.display.set_mode(LOGICAL_SIZE, pygame.RESIZABLE)
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.source = source
        self.scan_timeout = scan_timeout
        supplied_weights = load_weights or {}
        self.load_weights = {
            side: float(np.clip(supplied_weights.get(side, 1.0), 0.10, 3.00))
            for side in SIDES
        }
        self.settings_path = settings_path
        self.running = True
        self.paused = False
        self.show_heat = True
        self.show_field_lines = True
        self.show_ids = True
        self.show_force_cop = False
        self.mouse_logical = (-100, -100)
        self.min_deadzone_counts = 45.0
        self.filter_alpha_rise = 0.13
        self.filter_alpha_fall = 0.075
        self.last_history_at = 0.0
        self.last_loop_at = time.monotonic()
        self.next_sample_at = 0.0
        self.display_intervals: deque[float] = deque(maxlen=120)

        self.heat_w, self.heat_h = 180, 480
        yy, xx = np.mgrid[0 : self.heat_h, 0 : self.heat_w]
        self.heat_grid_x = xx / (self.heat_w - 1)
        self.heat_grid_y = yy / (self.heat_h - 1)
        self.foot_mask = self._build_foot_mask(self.heat_w, self.heat_h)
        kernels = []
        for sx, sy in SENSOR_POS:
            dist2 = (self.heat_grid_x - sx) ** 2 + (self.heat_grid_y - sy) ** 2
            kernels.append(np.exp(-dist2 / (2.0 * 0.145 * 0.145)))
        self.heat_kernels = np.asarray(kernels, dtype=np.float32)
        self.heat_kernel_sum = np.maximum(np.sum(self.heat_kernels, axis=0), 1e-5)
        self.static_foot_cache: dict[tuple[str, tuple[int, int], str], pygame.Surface] = {}
        self.feet = {
            side: FootViewState(side, (self.heat_h, self.heat_w)) for side in SIDES
        }

        self.buttons = {
            "connect": Button((722, 20, 112, 42), "Connect", "D"),
            "calibrate": Button((844, 20, 120, 42), "Calibrate", "B"),
            "pause": Button((974, 20, 90, 42), "Pause", "SP"),
            "record": Button((1074, 20, 104, 42), "Record", "F9"),
            "shot": Button((1188, 20, 136, 42), "Screenshot", "S"),
        }
        self.weight_window = (
            None
            if screenshot_path is not None or auto_record_path is not None
            else WeightControlWindow(self.load_weights)
        )
        self.weight_updates = (
            queue.Queue() if self.weight_window is None else self.weight_window.updates
        )
        self.weight_save_due_at = 0.0
        self.weight_save_pending = False
        self.recorder = VideoRecorder(LOGICAL_SIZE, fps=30)
        self.last_recorded_path: Optional[Path] = None
        self.auto_record_path = auto_record_path.resolve() if auto_record_path else None
        self.auto_record_seconds = max(1.0, float(auto_record_seconds))
        self.auto_record_delay = max(0.0, float(auto_record_delay))
        self.auto_record_started_at = 0.0
        self.screenshot_path = screenshot_path
        self.screenshot_frame = max(5, screenshot_frame)
        self.rendered_frames = 0
        self.saved_requested_shot: Optional[Path] = None
        self.toast_text = ""
        self.toast_until = 0.0

    @staticmethod
    def _build_foot_mask(width: int, height: int) -> np.ndarray:
        mask_surface = pygame.Surface((width, height))
        mask_surface.fill((0, 0, 0))
        points = [
            (round(float(x) * (width - 1)), round(float(y) * (height - 1)))
            for x, y in INSOLE_OUTLINE
        ]
        pygame.draw.polygon(mask_surface, (255, 255, 255), points)
        return pygame.surfarray.array2d(mask_surface).T != 0

    def toast(self, text_value: str, duration: float = 2.2) -> None:
        self.toast_text = text_value
        self.toast_until = time.monotonic() + duration

    def _save_load_weights(self) -> None:
        if self.settings_path is None:
            return
        try:
            document = {}
            if self.settings_path.exists():
                document = json.loads(self.settings_path.read_text(encoding="utf-8"))
            document["left_weight"] = round(self.load_weights["left"], 2)
            document["right_weight"] = round(self.load_weights["right"], 2)
            temporary = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.settings_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.toast(f"Unable to save weights: {exc}", 3.0)

    def _process_weight_updates(self) -> None:
        changed = False
        while True:
            try:
                message = self.weight_updates.get_nowait()
            except queue.Empty:
                break
            if message[0] == "error":
                self.toast(f"Unable to open weight controls: {message[1]}", 4.0)
                continue
            if message[0] != "set" or message[1] not in SIDES:
                continue
            side, value = message[1], float(message[2])
            value = round(float(np.clip(value, 0.10, 3.00)), 2)
            if value != self.load_weights[side]:
                self.load_weights[side] = value
                changed = True
        if changed:
            self.weight_save_pending = True
            self.weight_save_due_at = time.monotonic() + 0.25
        if self.weight_save_pending and time.monotonic() >= self.weight_save_due_at:
            self._save_load_weights()
            self.weight_save_pending = False

    def request_calibration(self) -> None:
        for state in self.feet.values():
            state.reset_calibration()
        self.toast("Capturing independent baselines; keep both insoles unloaded", 3.0)

    def clear_history(self) -> None:
        for state in self.feet.values():
            state.history_total.clear()
        self.toast("Dual-foot trends cleared")

    def save_screenshot(self, path: Optional[Path] = None) -> Path:
        if path is None:
            folder = Path.cwd() / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"dual_foot_{datetime.now():%Y%m%d_%H%M%S}.png"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.canvas, str(path))
        self.saved_requested_shot = path
        self.toast(f"Screenshot saved: {path.name}")
        return path

    def toggle_recording(self) -> None:
        if self.recorder.active:
            path = self.recorder.stop()
            self.last_recorded_path = path
            if path is not None:
                self.toast(f"Video saved: {path.name}", 3.0)
            return
        path = Path.cwd() / "recordings" / f"dual_foot_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        if self.recorder.start(path):
            self.toast("Dual-foot recording started")
        else:
            self.toast(self.recorder.error or "Unable to start recording", 3.5)

    def reconnect(self, *, swap: bool = False) -> None:
        if isinstance(self.source, DualDemoSource):
            if swap:
                self.source.feet["left"], self.source.feet["right"] = (
                    self.source.feet["right"],
                    self.source.feet["left"],
                )
                self.toast("Demo left/right mapping swapped")
            else:
                self.toast("Reconnect is not required in demo mode")
            self.request_calibration()
            return
        addresses = self.source.addresses
        device_names = self.source.device_names
        if swap:
            addresses = {"left": addresses["right"], "right": addresses["left"]}
            device_names = {
                "left": device_names["right"],
                "right": device_names["left"],
            }
        self.source.stop()
        self.source = DualBLESource(
            addresses,
            device_names=device_names,
            scan_timeout=self.scan_timeout,
        )
        self.source.start()
        self.request_calibration()
        if swap:
            self.toast("Swapping left/right devices and reconnecting", 3.5)
        else:
            self.toast("Reconnecting by unique left/right addresses", 3.0)

    def _finish_calibration(self, state: FootViewState, *, interrupted: bool = False) -> None:
        if not state.calibration_samples:
            return
        samples = np.stack(state.calibration_samples, axis=0)
        count = len(state.calibration_samples)
        state.baseline = np.median(samples, axis=0).astype(np.float32)
        state.noise_sigma = np.maximum(np.std(samples, axis=0), 1.0).astype(np.float32)
        state.signal_target.fill(0.0)
        state.filtered.fill(0.0)
        state.contact_active = False
        state.contact_peak_total = 0.0
        state.contact_peak_sensor = 0.0
        state.release_candidate_duration = 0.0
        state.release_zero_remaining = 0.0
        state.calibration_samples.clear()
        state.calibrating = False
        if interrupted:
            self.toast(
                f"{SIDE_NAMES[state.side]} data stalled; baseline built from {count} frames",
                2.8,
            )

    def _load_target(
        self,
        state: FootViewState,
        delta: np.ndarray,
        deadzone: np.ndarray,
        frame_dt: float,
    ) -> np.ndarray:
        """Suppress unloaded drift while preserving a deliberate press transition."""
        target = np.sign(delta) * np.maximum(np.abs(delta) - deadzone, 0.0)
        magnitudes = np.linalg.norm(target, axis=1)
        total = float(np.sum(magnitudes))
        peak = float(np.max(magnitudes))
        active_sensors = int(np.count_nonzero(magnitudes >= 75.0))

        if state.release_zero_remaining > 0.0:
            # After a confirmed lift, rapidly absorb the mechanical/magnetic
            # rebound before allowing a new contact. Without this lockout a
            # large residual can immediately retrigger the entry threshold.
            state.release_zero_remaining = max(
                0.0,
                state.release_zero_remaining - frame_dt,
            )
            tracking_alpha = 1.0 - math.exp(
                -float(np.clip(frame_dt, 0.001, 0.20)) / 0.35
            )
            state.baseline += tracking_alpha * delta
            return np.zeros_like(target)

        if state.contact_active:
            state.contact_peak_total = max(state.contact_peak_total, total)
            state.contact_peak_sensor = max(state.contact_peak_sensor, peak)
            # Hysteresis prevents a held load from flickering at the entry threshold.
            fully_quiet = peak < 130.0 and (total < 400.0 or active_sensors < 2)
            unloaded_from_peak = (
                state.contact_peak_total >= 1_200.0
                and total <= max(900.0, state.contact_peak_total * 0.38)
                and peak <= max(350.0, state.contact_peak_sensor * 0.55)
            )
            if unloaded_from_peak:
                state.release_candidate_duration += frame_dt
            else:
                state.release_candidate_duration = 0.0
            if fully_quiet or state.release_candidate_duration >= 0.18:
                state.contact_active = False
                state.release_candidate_duration = 0.0
                state.release_zero_remaining = 0.70
        elif peak >= 280.0 or (total >= 850.0 and active_sensors >= 3):
            state.contact_active = True
            state.contact_peak_total = total
            state.contact_peak_sensor = peak
            state.release_candidate_duration = 0.0

        if state.contact_active:
            return target

        if state.release_zero_remaining > 0.0:
            tracking_alpha = 1.0 - math.exp(
                -float(np.clip(frame_dt, 0.001, 0.20)) / 0.35
            )
            state.baseline += tracking_alpha * delta
            return np.zeros_like(target)

        # While there is no credible contact, follow slow Hall/temperature drift.
        # A 4 s time constant is fast enough to remove idle wandering but too slow
        # to absorb an ordinary press, which crosses the contact gate first.
        tracking_alpha = 1.0 - math.exp(-float(np.clip(frame_dt, 0.001, 0.20)) / 4.0)
        state.baseline += tracking_alpha * delta
        return np.zeros_like(target)

    def _update_calibration_watchdog(self) -> None:
        now = time.monotonic()
        for state in self.feet.values():
            if (
                state.calibrating
                and state.calibration_samples
                and state.calibration_last_sample_at > 0.0
                and now - state.calibration_last_sample_at >= 1.0
            ):
                self._finish_calibration(state, interrupted=True)

    def process_frames(self, frames: dict[str, Optional[SensorFrame]]) -> None:
        if self.paused:
            return
        for side in SIDES:
            frame = frames.get(side)
            state = self.feet[side]
            if frame is None:
                continue
            if frame.source_session != state.source_session:
                reconnected = state.source_session >= 0 and frame.source_session > 0
                state.source_session = frame.source_session
                state.reset_calibration()
                if reconnected:
                    self.toast(
                        f"{SIDE_NAMES[side]} reconnected; rebuilding unloaded baseline",
                        3.0,
                    )
            if frame.seq == state.last_seq:
                continue
            state.last_seq = frame.seq
            frame_dt = 1.0 / 50.0
            if state.last_frame_time > 0.0:
                dt = frame.timestamp - state.last_frame_time
                if 0.001 < dt < 1.0:
                    state.frame_intervals.append(dt)
                    frame_dt = dt
            state.last_frame_time = frame.timestamp
            state.raw_xyz = frame.xyz.astype(np.float32)
            temp_c = frame.temp_x10.astype(np.float32) / 10.0
            temp_c[(temp_c < -40.0) | (temp_c > 125.0)] = np.nan
            state.temp = temp_c

            if state.calibrating:
                now = time.monotonic()
                if not state.calibration_samples:
                    state.calibration_first_sample_at = now
                state.calibration_last_sample_at = now
                state.calibration_samples.append(state.raw_xyz.copy())
                elapsed = now - state.calibration_first_sample_at
                enough = (
                    len(state.calibration_samples) >= state.calibration_target
                    and elapsed >= 0.45
                )
                low_rate = (
                    len(state.calibration_samples) >= 12
                    and elapsed >= 0.90
                )
                if enough or low_rate:
                    self._finish_calibration(state)
                continue
            if state.baseline is None:
                continue

            delta = state.raw_xyz - state.baseline
            deadzone = np.clip(
                np.maximum(self.min_deadzone_counts, state.noise_sigma * 2.4),
                self.min_deadzone_counts,
                150.0,
            )
            target = self._load_target(state, delta, deadzone, frame_dt)
            # BLE 线程只更新目标值；显示状态在统一 UI 时钟中连续插值。
            # 因此 45 Hz 和 80 Hz 的输入都以同样的 60 FPS 平滑呈现。
            state.signal_target[:] = target

    def advance_display(self) -> None:
        if self.paused:
            return
        now = time.monotonic()
        for state in self.feet.values():
            dt = float(np.clip(now - state.last_display_at, 1.0 / 240.0, 0.10))
            state.last_display_at = now
            target = state.signal_target
            # 数据断流时平滑回零，不能无限保持最后一帧。
            if state.last_frame_time <= 0.0 or now - state.last_frame_time > 0.30:
                target = np.zeros_like(state.signal_target)
            rising = np.abs(target) > np.abs(state.filtered)
            alpha_rise = 1.0 - math.exp(-dt / 0.085)
            alpha_fall = 1.0 - math.exp(-dt / 0.145)
            alpha = np.where(rising, alpha_rise, alpha_fall)
            state.filtered += alpha * (target - state.filtered)
            quiet = np.linalg.norm(target, axis=1) < self.min_deadzone_counts * 1.25
            state.filtered[quiet] *= math.exp(-dt / 0.095)
            state.filtered[np.abs(state.filtered) < 4.0] = 0.0
            self._update_metrics(state)

        if now - self.last_history_at >= 0.08:
            self.last_history_at = now
            for state in self.feet.values():
                state.history_total.append(state.metrics.total / NUM_SENSORS)

    def _update_metrics(self, state: FootViewState) -> None:
        mags = np.linalg.norm(state.filtered, axis=1)
        peak = float(np.max(mags))
        active_threshold = max(90.0, peak * 0.09)
        active = int(np.count_nonzero(mags >= active_threshold))
        total = float(np.sum(mags))
        weights = mags + 1e-6
        cop_x = float(np.sum(SENSOR_POS[:, 0] * weights) / np.sum(weights))
        cop_y = float(np.sum(SENSOR_POS[:, 1] * weights) / np.sum(weights))
        region_raw = tuple(float(np.sum(mags[idx])) for idx in REGION_INDICES)
        region_sum = max(sum(region_raw), 1e-6)
        regions = tuple(value / region_sum for value in region_raw)
        state.mag_history_short.append(total)
        if len(state.mag_history_short) >= 8:
            values = np.asarray(state.mag_history_short)
            cv = float(np.std(values) / max(np.mean(values), 1.0))
            stability = float(np.clip(100.0 - cv * 85.0, 0.0, 100.0))
        else:
            stability = 100.0
        valid_temp = state.temp[np.isfinite(state.temp)]
        mean_temp = float(np.mean(valid_temp)) if valid_temp.size else float("nan")
        min_temp = float(np.min(valid_temp)) if valid_temp.size else float("nan")
        channel = self._channel(state.side)
        hz = float(getattr(channel, "output_hz", channel.raw_hz))
        components = tuple(float(value) for value in np.mean(state.filtered, axis=0))
        state.metrics = Metrics(
            peak=peak,
            active=active,
            mean_temp=mean_temp,
            min_temp=min_temp,
            hz=hz,
            stability=stability,
            force_components=components,
            cop=(cop_x, cop_y),
            region_loads=regions,
            region_totals=region_raw,
            total=total,
        )

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
                    self.reconnect(swap=True)
                elif event.key == pygame.K_b:
                    self.request_calibration()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.toast("Display paused" if self.paused else "Display resumed")
                elif event.key == pygame.K_h:
                    self.show_heat = not self.show_heat
                elif event.key == pygame.K_m:
                    self.show_field_lines = not self.show_field_lines
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_c:
                    self.show_force_cop = not self.show_force_cop
                elif event.key == pygame.K_F9:
                    self.toggle_recording()
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                elif event.key == pygame.K_r:
                    self.clear_history()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                point = self._screen_to_logical(event.pos)
                if self.buttons["connect"].rect.collidepoint(point):
                    self.reconnect()
                elif self.buttons["calibrate"].rect.collidepoint(point):
                    self.request_calibration()
                elif self.buttons["pause"].rect.collidepoint(point):
                    self.paused = not self.paused
                elif self.buttons["record"].rect.collidepoint(point):
                    self.toggle_recording()
                elif self.buttons["shot"].rect.collidepoint(point):
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

    def _balance(self) -> tuple[float, float]:
        left = self.feet["left"].metrics.total * self.load_weights["left"]
        right = self.feet["right"].metrics.total * self.load_weights["right"]
        total = left + right
        # 空载时只剩滤波尾差，不能把几十 counts 的噪声显示成严重偏载。
        if (
            total < 600.0
            or (
                self.feet["left"].metrics.active == 0
                and self.feet["right"].metrics.active == 0
            )
        ):
            return 0.5, 0.5
        return left / total, right / total

    def _channel(self, side: str):
        return self.source.feet[side]

    def _display_x(self, side: str, normalized_x: float) -> float:
        return 1.0 - normalized_x if side == "left" else normalized_x

    def _display_surface(self, side: str, surface: pygame.Surface) -> pygame.Surface:
        return pygame.transform.flip(surface, True, False) if side == "left" else surface

    def _inside_foot(self, x_value: float, y_value: float) -> bool:
        if x_value < 0.0 or x_value > 1.0 or y_value < 0.0 or y_value > 1.0:
            return False
        ix = int(np.clip(x_value * (self.heat_w - 1), 0, self.heat_w - 1))
        iy = int(np.clip(y_value * (self.heat_h - 1), 0, self.heat_h - 1))
        return bool(self.foot_mask[iy, ix])

    def _mask_surface(self, color: tuple[int, int, int, int]) -> pygame.Surface:
        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        rgba = np.zeros((self.heat_h, self.heat_w, 4), dtype=np.uint8)
        rgba[self.foot_mask] = color
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        return surface

    def _outline_surface(self, color=C.CYAN_2) -> pygame.Surface:
        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        mask = self.foot_mask
        edge = mask & (
            ~np.roll(mask, 1, axis=0)
            | ~np.roll(mask, -1, axis=0)
            | ~np.roll(mask, 1, axis=1)
            | ~np.roll(mask, -1, axis=1)
        )
        rgba = np.zeros((self.heat_h, self.heat_w, 4), dtype=np.uint8)
        rgba[edge] = (*color, 220)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = rgba[:, :, 3].T
        return surface

    def _build_force_surface(self, state: FootViewState) -> pygame.Surface:
        now = time.monotonic()
        if state.force_surface_cache is not None and now - state.last_heat_at < 1.0 / 30.0:
            return state.force_surface_cache
        heat_dt = 1.0 / 30.0 if state.last_heat_at <= 0.0 else min(now - state.last_heat_at, 0.10)
        state.last_heat_at = now
        raw_values = np.linalg.norm(state.filtered, axis=1).astype(np.float32)
        target_vmax = max(850.0, float(np.percentile(raw_values, 95)) * 1.12)
        vmax_tau = 0.20 if target_vmax > state.force_vmax else 2.8
        vmax_alpha = 1.0 - math.exp(-heat_dt / vmax_tau)
        state.force_vmax += vmax_alpha * (target_vmax - state.force_vmax)
        state.force_vmax = float(np.clip(state.force_vmax, 850.0, 9000.0))
        values = np.clip(raw_values / state.force_vmax, 0.0, 1.0) ** 0.72
        # 这里用逐元素归约，避免小矩阵 tensordot 启动大量 BLAS 线程反而拖慢 UI。
        field = np.sum(
            self.heat_kernels * values[:, None, None],
            axis=0,
            dtype=np.float32,
        )
        field = np.divide(field, self.heat_kernel_sum) * self.foot_mask
        mask_f = self.foot_mask.astype(np.float32)
        padded_f = np.pad(field, 1, mode="constant")
        padded_m = np.pad(mask_f, 1, mode="constant")
        numerator = (
            padded_f[1:-1, 1:-1] * 4.0
            + padded_f[:-2, 1:-1]
            + padded_f[2:, 1:-1]
            + padded_f[1:-1, :-2]
            + padded_f[1:-1, 2:]
        )
        denominator = (
            padded_m[1:-1, 1:-1] * 4.0
            + padded_m[:-2, 1:-1]
            + padded_m[2:, 1:-1]
            + padded_m[1:-1, :-2]
            + padded_m[1:-1, 2:]
        )
        field = np.where(self.foot_mask, numerator / np.maximum(denominator, 1e-5), 0.0)
        field_alpha = 1.0 - math.exp(-heat_dt / 0.09)
        state.force_field_ema += field_alpha * (field - state.force_field_ema)
        state.force_field_seq = state.last_seq
        field = state.force_field_ema
        stop_p = np.array([0.0, 0.18, 0.40, 0.62, 0.82, 1.0])
        stop_c = np.array(
            [
                (235, 243, 244),
                (190, 220, 224),
                (111, 185, 191),
                (226, 207, 123),
                (231, 143, 96),
                (210, 85, 98),
            ],
            dtype=np.float32,
        )
        rgb = np.zeros((self.heat_h, self.heat_w, 3), dtype=np.uint8)
        for channel in range(3):
            rgb[:, :, channel] = np.interp(field, stop_p, stop_c[:, channel]).astype(np.uint8)
        alpha = np.where(self.foot_mask, 246, 0).astype(np.uint8)
        surface = pygame.Surface((self.heat_w, self.heat_h), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgb, (1, 0, 2))
        pygame.surfarray.pixels_alpha(surface)[:, :] = alpha.T
        state.force_surface_cache = surface
        state.force_surface_version += 1
        return state.force_surface_cache

    def _prepare_magnetic_grid(self, state: FootViewState, rect: pygame.Rect) -> None:
        if state.magnetic_grid_size == rect.size:
            return
        step = 28.0
        xs = np.arange(step / 2.0, float(rect.w), step, dtype=np.float32)
        ys = np.arange(step / 2.0, float(rect.h), step, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1).astype(np.float32)
        sensor_pixels = SENSOR_POS * np.array([rect.w, rect.h], dtype=np.float32)
        diff = points[:, None, :] - sensor_pixels[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        weights = 1.0 / (dist2 + 36.0)
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-6)
        valid = np.array(
            [self._inside_foot(float(x / rect.w), float(y / rect.h)) for x, y in points],
            dtype=bool,
        )
        count = len(points)
        state.magnetic_grid_size = rect.size
        state.magnetic_grid_points = points
        state.magnetic_grid_weights = weights.astype(np.float32)
        state.magnetic_grid_valid = valid
        state.magnetic_grid_prev = np.zeros((count, 2), dtype=np.float32)
        state.magnetic_grid_vectors = np.zeros((count, 2), dtype=np.float32)
        state.magnetic_grid_visible = np.zeros((count,), dtype=bool)
        state.magnetic_grid_seq = -2

    def _sensor_xy(self, state: FootViewState) -> np.ndarray:
        delta_xy = -state.filtered[:, :2]
        cos_r = np.cos(CHIP_XY_ROTATIONS)
        sin_r = np.sin(CHIP_XY_ROTATIONS)
        return np.column_stack(
            (
                cos_r * delta_xy[:, 0] - sin_r * delta_xy[:, 1],
                sin_r * delta_xy[:, 0] + cos_r * delta_xy[:, 1],
            )
        ).astype(np.float32)

    def draw_magnetic_arrows(
        self,
        state: FootViewState,
        rect: pygame.Rect,
    ) -> None:
        self._prepare_magnetic_grid(state, rect)
        vectors = state.magnetic_grid_weights @ self._sensor_xy(state)
        # filtered 已由统一显示时钟插值，所以矢量也应每个显示帧更新，
        # 不能再次被较慢一侧的 BLE 序列号卡住。
        magnitude2 = np.sum(vectors * vectors, axis=1)
        changes = np.linalg.norm(vectors - state.magnetic_grid_prev, axis=1)
        raw_on = (magnitude2 >= 64.0) | (changes >= 4.0)
        raw_off = (magnitude2 < 36.0) & (changes < 1.4)
        state.magnetic_grid_visible = np.where(
            state.magnetic_grid_visible,
            ~raw_off,
            raw_on,
        )
        state.magnetic_grid_prev[:] = vectors
        state.magnetic_grid_vectors[:] = vectors
        state.magnetic_grid_seq = state.last_seq
        mirrored = state.side == "left"
        for point, vector, valid, visible in zip(
            state.magnetic_grid_points,
            state.magnetic_grid_vectors,
            state.magnetic_grid_valid,
            state.magnetic_grid_visible,
        ):
            if not valid:
                continue
            local_x = rect.w - float(point[0]) if mirrored else float(point[0])
            start = np.array([rect.x + local_x, rect.y + point[1]], dtype=np.float32)
            pygame.draw.circle(self.canvas, (80, 145, 124), tuple(start.astype(int)), 2)
            if not visible:
                continue
            display_vector = vector / 40.0
            if mirrored:
                display_vector[0] *= -1.0
            length = float(np.linalg.norm(display_vector))
            if length > 22.0:
                display_vector *= 22.0 / length
            end = start + display_vector
            pygame.draw.line(
                self.canvas,
                (55, 168, 119),
                tuple(start.astype(int)),
                tuple(end.astype(int)),
                2,
            )
            pygame.draw.circle(self.canvas, (55, 168, 119), tuple(end.astype(int)), 2)

    def draw_header(self) -> None:
        pygame.draw.line(self.canvas, C.BORDER, (24, 78), (1416, 78), 1)
        # 按需求去掉品牌 Logo，仅保留功能标题。
        blit_text(self.canvas, self.fonts.title, "Dual-Foot Sensor Monitor", C.TEXT, (24, 15))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "Independent BLE streams · load and magnetic vectors",
            C.TEXT_3,
            (24, 50),
        )
        for side, x_value in (("left", 402), ("right", 562)):
            channel = self._channel(side)
            rect = pygame.Rect(x_value, 20, 150, 42)
            rounded_rect(self.canvas, rect, C.PANEL, 11, C.BORDER)
            color = C.GREEN if channel.connected else C.RED
            pygame.draw.circle(self.canvas, color, (rect.x + 17, rect.centery), 5)
            status_label = "Online" if channel.connected else channel.status[:8]
            blit_text(
                self.canvas,
                self.fonts.small,
                f"{SIDE_NAMES[side]} {status_label}",
                C.TEXT,
                (rect.x + 29, rect.y + 10),
            )
        self.buttons["pause"].text = "Resume" if self.paused else "Pause"
        self.buttons["record"].text = "Stop" if self.recorder.active else "Record"
        for button in self.buttons.values():
            button.draw(self.canvas, self.fonts, self.mouse_logical)
        if self.recorder.active:
            pygame.draw.circle(self.canvas, C.RED, (1082, 26), 5)

    def draw_metric_cards(self) -> None:
        left = self.feet["left"].metrics
        right = self.feet["right"].metrics
        display_fps = (
            1.0 / float(np.mean(self.display_intervals))
            if self.display_intervals
            else 0.0
        )
        raw_left_hz = float(self._channel("left").raw_hz)
        raw_right_hz = float(self._channel("right").raw_hz)
        left_share, right_share = self._balance()
        if abs(left_share - right_share) < 0.03:
            balance_hint = "Load is nearly balanced"
        elif left_share > right_share:
            balance_hint = f"Left +{(left_share - right_share) * 100:.1f} percentage points"
        else:
            balance_hint = f"Right +{(right_share - left_share) * 100:.1f} percentage points"
        cards = [
            (
                "Left Total Response",
                f"{left.total / 1000.0:.1f}",
                "kcounts",
                C.CYAN,
                f"Peak {left.peak:,.0f} · active {left.active}/{NUM_SENSORS}",
            ),
            (
                "Right Total Response",
                f"{right.total / 1000.0:.1f}",
                "kcounts",
                C.ORANGE,
                f"Peak {right.peak:,.0f} · active {right.active}/{NUM_SENSORS}",
            ),
            (
                "Bilateral Load Split",
                f"{left_share * 100:.0f}:{right_share * 100:.0f}",
                "L : R",
                C.GREEN,
                balance_hint,
            ),
            (
                "Synchronized Data Rate",
                f"{left.hz:.0f}/{right.hz:.0f}",
                "Hz",
                C.MAGENTA,
                f"Raw links {raw_left_hz:.0f}/{raw_right_hz:.0f} Hz · display {display_fps:.0f} FPS",
            ),
        ]
        x0, y_value, gap, width, height = 24, 91, 12, 339, 82
        for index, (label, value, unit, accent, hint) in enumerate(cards):
            rect = pygame.Rect(x0 + index * (width + gap), y_value, width, height)
            rounded_rect(self.canvas, rect, C.PANEL, 13, C.BORDER)
            pygame.draw.rect(
                self.canvas,
                accent,
                (rect.x, rect.y + 12, 3, rect.height - 24),
                border_radius=2,
            )
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_2, (rect.x + 17, rect.y + 8))
            value_rect = blit_text(
                self.canvas,
                self.fonts.metric,
                value,
                C.TEXT,
                (rect.x + 17, rect.y + 24),
            )
            blit_text(
                self.canvas,
                self.fonts.tiny,
                unit,
                accent,
                (value_rect.right + 8, rect.y + 42),
                anchor="midleft",
            )
            blit_text(self.canvas, self.fonts.tiny, hint, C.TEXT_3, (rect.x + 17, rect.bottom - 17))

    def _static_scaled_foot(
        self,
        side: str,
        size: tuple[int, int],
        kind: str,
    ) -> pygame.Surface:
        key = (side, size, kind)
        cached = self.static_foot_cache.get(key)
        if cached is not None:
            return cached
        if kind == "outline":
            source = self._outline_surface(C.CYAN_2)
        else:
            source = self._mask_surface((240, 246, 244, 255))
        source = self._display_surface(side, source)
        cached = pygame.transform.smoothscale(source, size)
        self.static_foot_cache[key] = cached
        return cached

    def draw_single_foot(self, state: FootViewState, rect: pygame.Rect) -> None:
        side = state.side
        if self.show_heat:
            base = self._build_force_surface(state)
            if (
                state.display_surface_cache is None
                or state.display_surface_version != state.force_surface_version
                or state.display_surface_size != rect.size
            ):
                base = self._display_surface(side, base)
                state.display_surface_cache = pygame.transform.smoothscale(base, rect.size)
                state.display_surface_version = state.force_surface_version
                state.display_surface_size = rect.size
            display_base = state.display_surface_cache
        else:
            display_base = self._static_scaled_foot(side, rect.size, "neutral")
        self.canvas.blit(display_base, rect.topleft)
        if self.show_field_lines:
            self.draw_magnetic_arrows(state, rect)
        self.canvas.blit(self._static_scaled_foot(side, rect.size, "outline"), rect.topleft)

        if self.show_force_cop and state.metrics.total > 1.0:
            cop_x = rect.x + self._display_x(side, state.metrics.cop[0]) * rect.w
            cop_y = rect.y + state.metrics.cop[1] * rect.h
            pygame.draw.circle(self.canvas, C.INK, (round(cop_x), round(cop_y)), 10, 1)
            pygame.draw.circle(self.canvas, C.INK, (round(cop_x), round(cop_y)), 2)
            pygame.draw.line(self.canvas, C.INK, (cop_x - 13, cop_y), (cop_x + 13, cop_y), 1)
            pygame.draw.line(self.canvas, C.INK, (cop_x, cop_y - 13), (cop_x, cop_y + 13), 1)

        mags = np.linalg.norm(state.filtered, axis=1)
        peak_ref = max(3000.0, float(np.max(mags)))
        for index, ((nx, ny), magnitude) in enumerate(zip(SENSOR_POS, mags), start=1):
            x_value = rect.x + self._display_x(side, float(nx)) * rect.w
            y_value = rect.y + float(ny) * rect.h
            strength = float(np.clip(magnitude / peak_ref, 0.0, 1.0))
            color = mix_color(C.CYAN_2, C.ORANGE, strength)
            pygame.draw.circle(self.canvas, C.WHITE, (round(x_value), round(y_value)), 7)
            pygame.draw.circle(self.canvas, C.INK, (round(x_value), round(y_value)), 6, 1)
            pygame.draw.circle(self.canvas, color, (round(x_value), round(y_value)), 4)
            if self.show_ids:
                offset = -25 if side == "left" else 9
                blit_text(
                    self.canvas,
                    self.fonts.tiny,
                    f"{index:02d}",
                    C.TEXT,
                    (round(x_value) + offset, round(y_value) - 8),
                )

    def draw_feet_panel(self) -> None:
        card = pygame.Rect(24, 188, 932, 670)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "Real-Time Dual-Insole Distribution", C.TEXT, (44, 204))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "Heatmap = triaxial relative load · green arrows = IDW field direction · toes up",
            C.TEXT_3,
            (44, 230),
        )
        foot_rects = {
            # STEP 顶视包围盒 80.039 × 215.021 mm，宽长比 0.3722。
            # 576 px 高对应 214 px 宽，避免旧版 270 px 造成横向压扁感。
            "left": pygame.Rect(170, 278, 214, 576),
            "right": pygame.Rect(596, 278, 214, 576),
        }
        for side in SIDES:
            state = self.feet[side]
            channel = self._channel(side)
            rect = foot_rects[side]
            accent = C.CYAN if side == "left" else C.ORANGE
            blit_text(
                self.canvas,
                self.fonts.subtitle,
                SIDE_NAMES[side],
                C.TEXT,
                (rect.centerx, 273),
                anchor="midbottom",
            )
            identity = getattr(channel, "observed_name", "") or getattr(
                channel, "device_name", side
            )
            suffix = channel.address[-8:] if channel.address else "awaiting"
            address_label = f"{identity} · {suffix}"
            blit_text(
                self.canvas,
                self.fonts.tiny,
                address_label,
                accent,
                (rect.centerx, rect.y + 8),
                anchor="midtop",
            )
            self.draw_single_foot(state, rect)

    def _draw_region_balance(self, name: str, y_value: int, left_value: float, right_value: float) -> None:
        total = left_value + right_value
        left_share = 0.5 if total < 120.0 else left_value / total
        right_share = 1.0 - left_share
        blit_text(self.canvas, self.fonts.tiny, name, C.TEXT_2, (994, y_value))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"L {left_share * 100:.0f}%",
            C.CYAN,
            (1058, y_value),
        )
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"{right_share * 100:.0f}% R",
            C.ORANGE,
            (1392, y_value),
            anchor="topright",
        )
        track = pygame.Rect(1058, y_value + 22, 334, 10)
        pygame.draw.rect(self.canvas, C.ORANGE, track, border_radius=5)
        left_rect = track.copy()
        left_rect.width = max(1, round(track.w * left_share))
        pygame.draw.rect(self.canvas, C.CYAN, left_rect, border_radius=5)
        pygame.draw.line(self.canvas, C.WHITE, (track.centerx, track.y - 2), (track.centerx, track.bottom + 2), 1)

    def _draw_connection_card(self, side: str, rect: pygame.Rect) -> None:
        channel = self._channel(side)
        state = self.feet[side]
        accent = C.CYAN if side == "left" else C.ORANGE
        rounded_rect(self.canvas, rect, C.PANEL_2, 11, C.BORDER)
        pygame.draw.circle(
            self.canvas,
            C.GREEN if channel.connected else C.RED,
            (rect.x + 15, rect.y + 17),
            5,
        )
        blit_text(self.canvas, self.fonts.small, SIDE_NAMES[side], C.TEXT, (rect.x + 27, rect.y + 7))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            channel.status,
            accent,
            (rect.right - 10, rect.y + 9),
            anchor="topright",
        )
        identity = getattr(channel, "observed_name", "") or getattr(
            channel, "device_name", side
        )
        address_label = channel.address[-8:] if channel.address else "Unassigned"
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"{identity} · {address_label}",
            C.TEXT_3,
            (rect.x + 12, rect.y + 34),
        )
        blit_text(
            self.canvas,
            self.fonts.tiny,
            f"{state.metrics.hz:.1f} sync Hz",
            C.TEXT_2,
            (rect.right - 10, rect.y + 34),
            anchor="topright",
        )

    def _draw_trend_series(self, rect: pygame.Rect, values: deque[float], color, hi: float) -> None:
        if len(values) < 2:
            return
        arr = np.asarray(values, dtype=np.float32)
        xs = np.linspace(rect.x, rect.right, len(arr))
        ys = rect.bottom - np.clip(arr / max(hi, 1.0), 0.0, 1.0) * rect.h
        points = [(round(x_value), round(y_value)) for x_value, y_value in zip(xs, ys)]
        pygame.draw.aalines(self.canvas, color, False, points)
        pygame.draw.lines(self.canvas, color, False, points, 2)

    def draw_balance_panel(self) -> None:
        card = pygame.Rect(972, 188, 444, 670)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        left_share, right_share = self._balance()
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "15-point triaxial comparison · not calibrated in N",
            C.TEXT_3,
            (994, 211),
        )
        blit_text(
            self.canvas,
            self.fonts.hero,
            f"{left_share * 100:.0f}  :  {right_share * 100:.0f}",
            C.TEXT,
            (1194, 244),
            anchor="midtop",
        )
        blit_text(self.canvas, self.fonts.tiny, "Left", C.CYAN, (994, 289))
        blit_text(self.canvas, self.fonts.tiny, "Right", C.ORANGE, (1392, 289), anchor="topright")
        track = pygame.Rect(994, 311, 398, 16)
        pygame.draw.rect(self.canvas, C.ORANGE, track, border_radius=8)
        left_fill = track.copy()
        left_fill.width = max(1, round(track.w * left_share))
        pygame.draw.rect(self.canvas, C.CYAN, left_fill, border_radius=8)
        pygame.draw.line(self.canvas, C.WHITE, (track.centerx, track.y - 3), (track.centerx, track.bottom + 3), 2)
        difference = abs(left_share - right_share) * 100.0
        if difference < 3.0:
            result = "Balanced"
            result_color = C.GREEN
        elif left_share > right_share:
            result = f"Left  +{difference:.1f}%"
            result_color = C.CYAN
        else:
            result = f"Right  +{difference:.1f}%"
            result_color = C.ORANGE
        blit_text(
            self.canvas,
            self.fonts.subtitle,
            result,
            result_color,
            (1194, 342),
            anchor="midtop",
        )

        blit_text(self.canvas, self.fonts.small, "Regional Load Balance", C.TEXT, (994, 395))
        left_regions = self.feet["left"].metrics.region_totals
        right_regions = self.feet["right"].metrics.region_totals
        for index, name in enumerate(REGION_NAMES):
            self._draw_region_balance(
                name,
                421 + index * 42,
                left_regions[index] * self.load_weights["left"],
                right_regions[index] * self.load_weights["right"],
            )

        blit_text(self.canvas, self.fonts.small, "Independent Device Links", C.TEXT, (994, 551))
        self._draw_connection_card("left", pygame.Rect(994, 577, 194, 65))
        self._draw_connection_card("right", pygame.Rect(1198, 577, 194, 65))

        blit_text(self.canvas, self.fonts.small, "Dual-Foot Trend", C.TEXT, (994, 660))
        blit_text(self.canvas, self.fonts.tiny, "Last ~19 s · R to clear", C.TEXT_3, (1120, 663))
        chart = pygame.Rect(994, 692, 398, 130)
        for index in range(4):
            y_value = chart.y + round(chart.h * index / 3)
            pygame.draw.line(self.canvas, C.GRID, (chart.x, y_value), (chart.right, y_value), 1)
        for index in range(7):
            x_value = chart.x + round(chart.w * index / 6)
            pygame.draw.line(self.canvas, C.GRID, (x_value, chart.y), (x_value, chart.bottom), 1)
        histories = [np.asarray(self.feet[side].history_total or [0.0]) for side in SIDES]
        hi = max(4200.0, *(float(np.percentile(values, 98)) * 1.10 for values in histories))
        self._draw_trend_series(chart, self.feet["left"].history_total, C.CYAN, hi)
        self._draw_trend_series(chart, self.feet["right"].history_total, C.ORANGE, hi)
        pygame.draw.line(self.canvas, C.CYAN, (1264, 671), (1278, 671), 2)
        blit_text(self.canvas, self.fonts.tiny, "L", C.CYAN, (1283, 662))
        pygame.draw.line(self.canvas, C.ORANGE, (1323, 671), (1337, 671), 2)
        blit_text(self.canvas, self.fonts.tiny, "R", C.ORANGE, (1342, 662))

    def draw_footer(self) -> None:
        # Keep the bottom edge clean in both the live UI and exported videos.
        return

    def draw_overlay(self) -> None:
        now = time.monotonic()
        sampling = [
            state for state in self.feet.values() if state.calibrating and state.calibration_samples
        ]
        if sampling:
            veil = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
            veil.fill((226, 233, 231, 150))
            self.canvas.blit(veil, (0, 0))
            panel = pygame.Rect(456, 410, 528, 164)
            rounded_rect(self.canvas, panel, C.PANEL, 16, C.YELLOW)
            blit_text(
                self.canvas,
                self.fonts.subtitle,
                "Building Independent Baselines",
                C.TEXT,
                (720, 429),
                anchor="midtop",
            )
            blit_text(
                self.canvas,
                self.fonts.small,
                "Keep both insoles unloaded; each foot calibrates independently",
                C.TEXT_2,
                (720, 462),
                anchor="midtop",
            )
            for index, side in enumerate(SIDES):
                state = self.feet[side]
                y_value = 501 + index * 30
                blit_text(self.canvas, self.fonts.tiny, SIDE_NAMES[side], C.TEXT_2, (490, y_value - 4))
                progress = pygame.Rect(540, y_value, 400, 8)
                pygame.draw.rect(self.canvas, C.GRID, progress, border_radius=4)
                fill = progress.copy()
                fill.width = round(
                    progress.w
                    * np.clip(
                        len(state.calibration_samples) / state.calibration_target,
                        0.0,
                        1.0,
                    )
                )
                pygame.draw.rect(
                    self.canvas,
                    C.CYAN if side == "left" else C.ORANGE,
                    fill,
                    border_radius=4,
                )
        elif self.paused:
            pill = pygame.Rect(654, 93, 132, 34)
            rounded_rect(self.canvas, pill, C.YELLOW, 10)
            blit_text(self.canvas, self.fonts.small, "PAUSED", C.INK, pill.center, anchor="center")

        if self.toast_text and now < self.toast_until and not sampling:
            image = self.fonts.small.render(self.toast_text, True, C.TEXT)
            toast_rect = pygame.Rect(0, 0, image.get_width() + 34, 42)
            toast_rect.midbottom = (LOGICAL_SIZE[0] // 2, 851)
            rounded_rect(self.canvas, toast_rect, C.PANEL, 12, C.CYAN_2)
            self.canvas.blit(image, image.get_rect(center=toast_rect.center))

    def render(self) -> None:
        self.canvas.fill(C.BG)
        glow = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(glow, (128, 195, 181, 20), (350, 470), 420)
        pygame.draw.circle(glow, (225, 170, 125, 14), (830, 480), 400)
        self.canvas.blit(glow, (0, 0))
        self.draw_header()
        self.draw_metric_cards()
        self.draw_feet_panel()
        self.draw_balance_panel()
        self.draw_footer()
        self.draw_overlay()
        sw, sh = self.screen.get_size()
        scale = min(sw / LOGICAL_SIZE[0], sh / LOGICAL_SIZE[1])
        target = (
            max(1, round(LOGICAL_SIZE[0] * scale)),
            max(1, round(LOGICAL_SIZE[1] * scale)),
        )
        scaled = (
            self.canvas
            if target == LOGICAL_SIZE
            else pygame.transform.smoothscale(self.canvas, target)
        )
        self.screen.fill(C.BG_2)
        self.screen.blit(scaled, ((sw - target[0]) // 2, (sh - target[1]) // 2))
        pygame.display.flip()

    def run(self) -> Optional[Path]:
        run_started_at = time.monotonic()
        try:
            if self.weight_window is not None:
                self.weight_window.start()
            self.source.start()
            while self.running:
                loop_now = time.monotonic()
                loop_dt = loop_now - self.last_loop_at
                self.last_loop_at = loop_now
                if 0.001 < loop_dt < 0.2:
                    self.display_intervals.append(loop_dt)
                self.handle_events()
                self._process_weight_updates()
                if (
                    self.auto_record_started_at > 0.0
                    and isinstance(self.source, DualDemoSource)
                ):
                    video_time = self.recorder.frames_written / self.recorder.fps
                    self.source.set_simulation_time(self.auto_record_delay + video_time)
                sample_now = time.monotonic()
                if self.next_sample_at < sample_now - SAMPLE_INTERVAL_S * 4.0:
                    self.next_sample_at = sample_now
                while self.next_sample_at <= sample_now:
                    self.process_frames(self.source.sample())
                    self.next_sample_at += SAMPLE_INTERVAL_S
                self.advance_display()
                self._update_calibration_watchdog()
                self.render()
                if (
                    self.auto_record_path is not None
                    and self.auto_record_started_at <= 0.0
                    and loop_now - run_started_at >= self.auto_record_delay
                ):
                    if not self.recorder.start(self.auto_record_path):
                        raise RuntimeError(self.recorder.error)
                    self.auto_record_started_at = loop_now
                self.recorder.capture(self.canvas)
                self.rendered_frames += 1
                if (
                    self.auto_record_started_at > 0.0
                    and self.recorder.frames_written
                    >= math.ceil(self.auto_record_seconds * self.recorder.fps)
                ):
                    self.last_recorded_path = self.recorder.stop()
                    break
                if self.screenshot_path and self.rendered_frames >= self.screenshot_frame:
                    path = self.save_screenshot(Path(self.screenshot_path).resolve())
                    self.saved_requested_shot = path
                    break
                self.clock.tick(FPS)
        finally:
            if self.weight_window is not None:
                self.weight_window.stop()
            self._process_weight_updates()
            if self.weight_save_pending:
                self._save_load_weights()
                self.weight_save_pending = False
            if self.recorder.active:
                self.last_recorded_path = self.recorder.stop()
            if self.weight_window is not None:
                self.weight_window.close_queue()
            self.source.stop()
            pygame.quit()
        return self.saved_requested_shot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FootSensor15 dual-foot real-time BLE dashboard")
    parser.add_argument(
        "--mode",
        choices=("demo", "ble"),
        default="ble",
        help="ble=connect real FootSensor15 devices (default); demo=offline UI preview",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("dual_foot_dashboard.json"),
        help="left/right BLE address config; defaults to dual_foot_dashboard.json",
    )
    parser.add_argument("--left-address", help="temporarily override the left BLE address")
    parser.add_argument("--right-address", help="temporarily override the right BLE address")
    parser.add_argument("--left-name", help="left-foot BLE advertising name (default: left)")
    parser.add_argument("--right-name", help="right-foot BLE advertising name (default: right)")
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT_PATH,
        help="numbered 15-point A4 sensor layout JSON",
    )
    parser.add_argument("--left-weight", type=float, help="initial left load weight (0.10 to 3.00)")
    parser.add_argument("--right-weight", type=float, help="initial right load weight (0.10 to 3.00)")
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=8.0,
        help="BLE scan timeout in seconds (default: 8)",
    )
    parser.add_argument(
        "--screenshot",
        metavar="PNG",
        help="save a screenshot after several frames and exit",
    )
    parser.add_argument(
        "--screenshot-frame",
        type=int,
        default=150,
        help="frames rendered before automatic screenshot (default: 150)",
    )
    parser.add_argument(
        "--demo-cadence",
        type=float,
        default=96.0,
        help="demo walking cadence in total steps per minute (default: 96)",
    )
    parser.add_argument(
        "--record-demo",
        type=Path,
        metavar="MP4",
        help="automatically record demo mode to MP4 and exit",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=8.0,
        help="automatic demo recording duration in seconds (default: 8)",
    )
    parser.add_argument(
        "--record-delay",
        type=float,
        default=0.8,
        help="warm-up before automatic recording in seconds (default: 0.8)",
    )
    return parser.parse_args()


def load_dashboard_addresses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"left": "", "right": ""}
    document = json.loads(path.read_text(encoding="utf-8"))
    addresses = {}
    for side in SIDES:
        direct = document.get(f"{side}_address", "")
        nested = document.get(side, {})
        nested_value = nested.get("address", "") if isinstance(nested, dict) else ""
        addresses[side] = str(direct or nested_value).strip().upper()
    if addresses["left"] and addresses["left"] == addresses["right"]:
        raise ValueError("left_address and right_address must be different")
    return addresses


def load_dashboard_device_names(path: Path) -> dict[str, str]:
    names = dict(DEFAULT_DEVICE_NAMES)
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        for side in SIDES:
            direct = document.get(f"{side}_device_name", "")
            nested = document.get(side, {})
            nested_value = (
                nested.get("device_name", "") if isinstance(nested, dict) else ""
            )
            names[side] = str(direct or nested_value or names[side]).strip()
    if not all(names.values()):
        raise ValueError("left_device_name and right_device_name must be non-empty")
    if names["left"].casefold() == names["right"].casefold():
        raise ValueError("left_device_name and right_device_name must be different")
    return names


def load_dashboard_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"left": 1.0, "right": 1.0}
    document = json.loads(path.read_text(encoding="utf-8"))
    weights = {
        side: float(document.get(f"{side}_weight", 1.0)) for side in SIDES
    }
    if any(not 0.10 <= value <= 3.00 for value in weights.values()):
        raise ValueError("left_weight and right_weight must be within 0.10 to 3.00")
    return weights


def main() -> int:
    global SENSOR_POS
    args = parse_args()
    try:
        SENSOR_POS = load_sensor_positions(args.layout)
        load_weights = load_dashboard_weights(args.config)
        if args.left_weight is not None:
            load_weights["left"] = args.left_weight
        if args.right_weight is not None:
            load_weights["right"] = args.right_weight
        if any(not 0.10 <= value <= 3.00 for value in load_weights.values()):
            raise ValueError("load weights must be within 0.10 to 3.00")
        if args.record_demo is not None and args.mode != "demo":
            raise ValueError("--record-demo requires --mode demo")
        if not 50.0 <= args.demo_cadence <= 170.0:
            raise ValueError("--demo-cadence must be within 50 to 170 SPM")
        if args.record_seconds < 1.0:
            raise ValueError("--record-seconds must be at least 1.0")
        if args.mode == "demo":
            source: DualDemoSource | DualBLESource = DualDemoSource(args.demo_cadence)
        else:
            addresses = load_dashboard_addresses(args.config)
            device_names = load_dashboard_device_names(args.config)
            if args.left_address:
                addresses["left"] = args.left_address.strip().upper()
            if args.right_address:
                addresses["right"] = args.right_address.strip().upper()
            if args.left_name:
                device_names["left"] = args.left_name.strip()
            if args.right_name:
                device_names["right"] = args.right_name.strip()
            source = DualBLESource(
                addresses,
                device_names=device_names,
                scan_timeout=args.scan_timeout,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Unable to load dual-foot BLE config: {exc}", file=sys.stderr)
        return 2
    app = DualDashboard(
        source,
        scan_timeout=args.scan_timeout,
        load_weights=load_weights,
        settings_path=args.config,
        screenshot_path=args.screenshot,
        screenshot_frame=args.screenshot_frame,
        auto_record_path=args.record_demo,
        auto_record_seconds=args.record_seconds,
        auto_record_delay=args.record_delay,
    )
    shot = app.run()
    if shot is not None:
        print(shot)
    if args.record_demo is not None and app.last_recorded_path is not None:
        print(app.last_recorded_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
