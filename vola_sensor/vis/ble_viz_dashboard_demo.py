#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FootSensor15 实时 BLE 可视化 Dashboard

特点
----
1. 单文件运行，不依赖原项目中缺失的 foot_hall_* 等模块。
2. 默认直接扫描并连接真实 FootSensor15，不需要额外启动参数。
3. BLE Notify 支持拆包、粘包和错位后的自动重新同步。
4. 支持基线校准、Hall 响应热图、IDW 磁场箭头、三轴响应示意和实时趋势。
5. 支持调用 FFmpeg 录制 MP4 视频。

运行
----
    ../.venv/bin/python ble_viz_dashboard_demo.py

快捷键
------
    B        重新校准（校准时请确保足底无外力）
    D        重新连接 FootSensor15
    Space    暂停 / 继续
    H        显示 / 隐藏热力图
    M        显示 / 隐藏磁场矢量
    C        显示 / 隐藏 Hall 响应中心（默认隐藏）
    I        显示 / 隐藏传感器编号
    F9       开始 / 停止录制 MP4
    S        保存截图到当前目录 screenshots/
    R        清空趋势
    Esc      退出
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
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

APP_TITLE = "模感科技 · 足底多物理场监测"
APP_VERSION = "BLE REALTIME 01"
LOGICAL_SIZE = (1440, 900)
MIN_WINDOW_SIZE = (1120, 700)
FPS = 60
NUM_SENSORS = 15

DEVICE_NAME = "FootSensor15"
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

REGION_NAMES = ("前掌", "足弓", "足跟")
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


@dataclass
class Metrics:
    peak: float = 0.0
    active: int = 0
    mean_temp: float = float("nan")
    min_temp: float = float("nan")
    hz: float = 0.0
    stability: float = 100.0
    resultant: tuple[float, float] = (0.0, 0.0)
    hall_components: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cop: tuple[float, float] = (0.5, 0.5)
    region_loads: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# BLE 与仿真数据源
# ---------------------------------------------------------------------------


class Int16Unwrapper:
    """保守展开连续 int16 流，避免把普通大跳变误判成跨量程环绕。

    只在相邻线传值之差超过 60000 counts、也就是确实位于 int16 两端时，
    才补偿 ±65536。该门限与实机参考可视化一致；重连时由 decoder.reset()
    重新锚定，避免沿用上一连接的圈数。
    """

    def __init__(self, shape: tuple[int, ...], wrap_threshold: int = 60000):
        self.shape = shape
        self.wrap_threshold = int(wrap_threshold)
        self.last: Optional[np.ndarray] = None
        self.ext: Optional[np.ndarray] = None
        self.wrap_events = 0

    def reset(self) -> None:
        self.last = None
        self.ext = None
        self.wrap_events = 0

    def push(self, wire: np.ndarray) -> np.ndarray:
        current = np.asarray(wire, dtype=np.int32)
        if self.last is None:
            self.last = current.copy()
            self.ext = current.copy()
            return self.ext.copy()
        delta = current.astype(np.int64) - self.last.astype(np.int64)
        abs_delta = np.abs(delta)
        wraps_negative = (abs_delta > self.wrap_threshold) & (delta < 0)
        wraps_positive = (abs_delta > self.wrap_threshold) & (delta > 0)
        step = delta.copy()
        step = np.where(wraps_negative, delta + 65536, step)
        step = np.where(wraps_positive, delta - 65536, step)
        self.wrap_events += int(np.count_nonzero(wraps_negative | wraps_positive))
        self.ext = np.clip(self.ext.astype(np.int64) + step, -2_000_000, 2_000_000).astype(
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
        self.status = "仿真数据"
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self.status = "准备扫描"
        self.detail = DEVICE_NAME
        self.connected = False
        self.notification_count = 0
        self.valid_frame_count = 0
        self.last_payload_len = 0
        self.last_valid_at = 0.0

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
            self.connected = False
            self.status = "BLE 线程异常"
            self.detail = str(exc)[:72]
        finally:
            pending = [item for item in asyncio.all_tasks(loop) if not item.done()]
            for item in pending:
                item.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._main_task = None
            self._loop = None

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
    """Hall 响应热图色标：冷色低响应，暖色高响应。"""
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
        self.ffmpeg = shutil.which("ffmpeg")
        self.process: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None
        self.started_at = 0.0
        self.last_capture_at = 0.0
        self.error = ""

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
            self.error = "未找到 FFmpeg，无法生成 MP4"
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
        except (BrokenPipeError, OSError) as exc:
            self.error = f"录制中断：{exc}"
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


class Dashboard:
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
        self.calibration_target = 75
        self.calibration_started_at = time.monotonic()
        self.calibration_first_sample_at = 0.0
        self.calibration_last_sample_at = 0.0
        self.calibration_message_until = 0.0
        self.metrics = Metrics()
        self.mag_history_short: deque[float] = deque(maxlen=55)
        self.history_peak: deque[float] = deque(maxlen=240)
        self.history_total: deque[float] = deque(maxlen=240)
        self.history_temp: deque[float] = deque(maxlen=240)
        self.history_clock = 0.0
        self.last_history_at = 0.0

        self.buttons = {
            "connect": Button((814, 21, 118, 42), "连接设备", "D"),
            "calibrate": Button((942, 21, 90, 42), "空载归零", "B"),
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
        self.force_field_ema = np.zeros((self.heat_h, self.heat_w), dtype=np.float32)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
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
        self.force_field_ema.fill(0.0)
        self.force_field_seq = -2
        self.force_vmax = 3200.0
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

    def _finish_calibration(self) -> None:
        """Use one continuous unloaded window to establish a fixed baseline."""
        if not self.calibration_samples:
            return
        samples = np.stack(self.calibration_samples, axis=0)
        self.baseline = np.median(samples, axis=0).astype(np.float32)
        self.noise_sigma = np.maximum(np.std(samples, axis=0), 1.0).astype(np.float32)
        self.calibration_samples.clear()
        self.calibrating = False
        self.calibration_message_until = time.monotonic() + 1.8
        self.toast("固定空载基线校准完成")

    def _update_calibration_watchdog(self) -> None:
        """Discard an interrupted window; never calibrate from partial data."""
        if (
            self.calibrating
            and self.calibration_last_sample_at > 0.0
            and time.monotonic() - self.calibration_last_sample_at >= 0.50
        ):
            self.calibration_samples.clear()
            self.calibration_first_sample_at = 0.0
            self.calibration_last_sample_at = 0.0

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
            enough_frames = (
                len(self.calibration_samples) >= self.calibration_target
                and now - self.calibration_first_sample_at >= 1.25
            )
            low_rate_timeout = (
                len(self.calibration_samples) >= 45
                and now - self.calibration_first_sample_at >= 2.0
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
        # 热图采用三轴 Hall 变化模长；显著变化并不只出现在 Z。
        response_magnitude = mags
        weights = response_magnitude + mags * 0.12 + 1e-6
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

        region_raw = [float(np.sum(response_magnitude[idx])) for idx in REGION_INDICES]
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
            hall_components=components,
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
                    self.toast("画面已暂停" if self.paused else "画面继续")
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
                    self.toast("Hall 响应中心已显示" if self.show_force_cop else "Hall 响应中心已隐藏")
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
                    self.toast("Hall 响应中心已显示" if self.show_force_cop else "Hall 响应中心已隐藏")

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
            # 量程快速跟随增大的 Hall 响应、缓慢回落，避免每帧拉满。
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
            "左：Hall 三轴响应强度   ·   右：15 点 IDW 磁场矢量   ·   脚尖始终朝上",
            C.TEXT_3,
            (44, 231),
        )
        pygame.draw.line(self.canvas, C.BORDER, (528, 249), (528, 844), 1)

        blit_text(self.canvas, self.fonts.small, "Hall 响应强度", C.TEXT, (45, 253))
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
            ("隐藏" if self.show_force_cop else "显示") + "响应中心  C",
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

        # 左侧仅显示 Hall 三轴变化模长，不表示力或压力。
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
            blit_text(self.canvas, self.fonts.tiny, "Hall 响应中心", C.INK, (cop_x + 10, cop_y - 17))
        blit_text(self.canvas, self.fonts.tiny, "区域 Hall 响应占比", C.TEXT_2, (45, 337))

        # 区域 Hall 响应占比与单调色标
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

        blit_text(self.canvas, self.fonts.tiny, "低", C.TEXT_3, (45, 641))
        for i in range(151):
            pygame.draw.line(
                self.canvas,
                force_color(i / 150.0),
                (68 + i, 644),
                (68 + i, 653),
            )
        blit_text(self.canvas, self.fonts.tiny, "高", C.TEXT_3, (226, 641))
        blit_text(
            self.canvas,
            self.fonts.tiny,
            "原始 Hall counts 相对变化，不是力/压力",
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
            ("Hall热图 H", self.show_heat, C.ORANGE),
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

    def draw_hall_component_card(self) -> None:
        card = pygame.Rect(1048, 190, 368, 244)
        rounded_rect(self.canvas, card, C.PANEL, 16, C.BORDER)
        blit_text(self.canvas, self.fonts.label, "Hall 三轴响应示意", C.TEXT, (1066, 205))
        blit_text(self.canvas, self.fonts.tiny, "平均 ΔB 分量 · 原始 counts", C.TEXT_3, (1066, 230))

        center = np.array([1125.0, 334.0])
        basis_x = np.array([0.88, 0.44])
        basis_y = np.array([-0.82, 0.52])
        basis_z = np.array([0.0, -1.0])
        for basis, label in ((basis_x, "X"), (basis_y, "Y"), (basis_z, "Z")):
            end = center + basis * 50.0
            draw_arrow(self.canvas, center, end, C.BORDER, 1, 5)
            blit_text(self.canvas, self.fonts.tiny, label, C.TEXT_3, tuple(end.astype(int)), anchor="center")

        components = np.asarray(self.metrics.hall_components, dtype=np.float32)
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
        labels = ("Bx", "By", "Bz")
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
            "D 重新连接   B 空载归零   SPACE 暂停   H Hall热图   M 磁矢量   C 响应中心   I 编号   F9 录制   S 截图   ESC 退出",
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
        self.draw_hall_component_card()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FootSensor15 实时 BLE 足底可视化 Dashboard")
    parser.add_argument(
        "--mode",
        choices=("demo", "ble"),
        default="ble",
        help="ble=连接真实 FootSensor15（默认），demo=仅用于离线界面诊断",
    )
    parser.add_argument(
        "--screenshot",
        metavar="PNG",
        help="运行若干帧后保存截图并自动退出，适合无桌面预览",
    )
    parser.add_argument(
        "--screenshot-frame",
        type=int,
        default=150,
        help="自动截图前绘制的帧数，默认 150",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = DemoSource() if args.mode == "demo" else BLESource()
    app = Dashboard(
        source,
        screenshot_path=args.screenshot,
        screenshot_frame=args.screenshot_frame,
    )
    shot = app.run()
    if shot is not None:
        print(shot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
