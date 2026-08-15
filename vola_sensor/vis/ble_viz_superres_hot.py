#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLE 足底传感器可视化（超分辨）：通过 BLE 连接 FootSensor15，固定使用 IDW 网格密集箭头。

稀疏 15 箭头可视化请用同目录 ble_viz.py（foot_hall_viz_region.py）。

数据流：参考 ble_debug.py (BLE Notify -> 125 字节帧解析)
数据处理：同目录 foot_hall_viz_region_superres_hot.py
  - 卡尔曼滤波、基准校准、区域检测、16 扇区方向补偿
  - 超分辨：IDW 规则网格；足底格点固定绿锚点（与箭头同色）+ 滞回条件箭头，减轻闪烁
  - 热力图与实时 XYZ 曲线（与 hot 版本一致）
  - 窗口内按 P：开关终端周期调试输出（象限/分区、[GridMask] 等）

依赖: pip install bleak numpy pygame

打包体积（PyInstaller）：
  foot_hall_viz_region_superres_hot 已对 cv2、pyserial 延迟加载；不要用 --collect-all pygame。
  推荐：在干净 venv 只装 bleak numpy pygame pyinstaller，于 tools 目录用精简 spec（排除未用大包 + UPX）：
    cd ...\tools
    pyinstaller pyinstaller_ble_superres_hot.spec
  输出 dist\\FootSensorBLESuperresHot.exe。若 UPX 导致无法启动，编辑 spec 将 upx=False。
  命令行等价（无 spec 时）可仍加 --exclude-module cv2 --exclude-module serial。
  体积下限主要由 numpy / pygame / bleak 决定；要再小只能换技术栈或接受 onedir 分发。
"""

import asyncio
import os
import struct
import sys
import threading
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    sys.exit(1)

import numpy as np

from ble_int16_unwrap import Int16StreamUnwrap3D
from ble_value_bounds import (
    clamp_mag_ext_i32,
    clamp_temp_wire_i32,
    clamp_wire_i16,
    clip_mag_ext_array,
    min_valid_temp_c,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# 全局静默：屏蔽当前进程（含依赖模块）所有 print 输出
_stdio_devnull = open(os.devnull, "w", encoding="utf-8")
sys.stdout = _stdio_devnull
sys.stderr = _stdio_devnull

import pygame

# ================= BLE 配置（与 ble_debug.py 一致）=================
DEVICE_NAME = "FootSensor15"
CHAR_UUID = "0000ab01-0000-1000-8000-00805f9b34fb"
FRAME_LEN = 125
FRAME_HEADER = 0x7D
DATA_OFFSET = 4
DATA_BYTES = 120
NUM_SENSORS = 15
# T 为帧内 °C×10（int16），与 ble_debug / ble_packet.c 一致
ENDIAN_FMT = ">hhhh"

_unwrap_xyz = Int16StreamUnwrap3D(NUM_SENSORS)

# ================= 导入同目录 foot_hall_viz_region_superres_hot（含热力图与实时曲线）=================
import foot_hall_viz_region_superres_hot as fhv

# 与 foot_hall 一致：可视化共享缓冲为 int32（磁场取整；温度为 °C×10）
fhv.shared_data = np.zeros_like(fhv.shared_data, dtype=np.int32)
fhv.shared_temp = np.zeros_like(fhv.shared_temp, dtype=np.int32)

# 使用 foot_hall_viz_region_superres_hot 的共享状态（卡尔曼滤波、基准、锁等）
shared_data = fhv.shared_data
shared_temp = fhv.shared_temp
data_lock = fhv.data_lock
sensor_filters = fhv.sensor_filters

# 热力图仅显示红色阈值：gray>=阈值显示红色，其余不显示任何颜色
HEATMAP_RED_GRAY_THRESHOLD = 180  # 0~255，越大越“严格”


def _red_only_colormap_bgr(gray_u8: np.ndarray) -> np.ndarray:
    h, w = gray_u8.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    red_mask = gray_u8 >= int(HEATMAP_RED_GRAY_THRESHOLD)
    out[red_mask] = (0, 0, 255)  # BGR 红色
    return out


# 覆盖 hot 模块内部色图函数：不改其余热力图/曲线逻辑
fhv._colormap_jet_bgr = _red_only_colormap_bgr

# 进一步覆盖热力图构建：非红区 alpha=0，保留底图；红区才半透明叠加
_orig_build_superres_heatmap = fhv._build_superres_heatmap


def _build_superres_heatmap_red_overlay(
    grid_pts, grid_valid, grid_vec, prev_heat_mags, grid_nx, grid_ny, prev_vmax
):
    heat_bgr, gray, vmax, filt_mags, a_h = _orig_build_superres_heatmap(
        grid_pts, grid_valid, grid_vec, prev_heat_mags, grid_nx, grid_ny, prev_vmax
    )
    if heat_bgr is None or gray is None:
        return heat_bgr, gray, vmax, filt_mags, a_h

    red_mask = gray >= int(HEATMAP_RED_GRAY_THRESHOLD)
    # 非红区不参与混合（alpha=0），因此会显示原始底图
    gray = np.where(red_mask, gray, 0).astype(np.uint8)
    heat_bgr[~red_mask] = (0, 0, 0)
    return heat_bgr, gray, vmax, filt_mags, a_h


fhv._build_superres_heatmap = _build_superres_heatmap_red_overlay

# 超分辨绿点/箭头：底图固定锚点 + 条件箭头，并滞回箭头开关，抑制阈值抖动闪烁
fhv.GRID_DRAW_BASE_DOTS = True
fhv.GRID_ARROW_USE_HYSTERESIS = True
fhv.GRID_BASE_DOT_RADIUS = 2
fhv.GRID_ARROW_HYST_LO_FRAC = 0.65

# 足底图最左侧小面板：仅显示 15 通道 T 的最小值（宽/高按文字自适应）
_AMBIENT_MARGIN = 10
_AMBIENT_PAD_X = 8
_AMBIENT_PAD_Y = 6


def _draw_ambient_temp_overlay(
    window,
    _font,
    font_large,
    temps,
    *,
    img_offset_x,
):
    """在足底图区域最左侧绘制 15 通道 T 的最小值（°C），仅数值、无标题。"""
    if temps is None or len(temps) < NUM_SENSORS:
        return
    # shared_temp 为 °C×10（int32），显示时换算为 °C
    arr = np.asarray(temps, dtype=np.int64).reshape(-1)
    min_t = min_valid_temp_c(arr, num_sensors=NUM_SENSORS)

    if not np.isfinite(min_t):
        val_s = "--  °C"
    else:
        val_s = f"{min_t:.2f} °C"
    val = font_large.render(val_s, True, (120, 220, 255))

    panel_w = val.get_width() + _AMBIENT_PAD_X * 2
    panel_h = _AMBIENT_PAD_Y * 2 + val.get_height()

    x0 = img_offset_x + _AMBIENT_MARGIN
    y0 = _AMBIENT_MARGIN

    try:
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((25, 28, 35, 210))
        window.blit(panel, (x0, y0))
    except Exception:
        pygame.draw.rect(
            window,
            (40, 44, 52),
            (x0, y0, panel_w, panel_h),
        )
    pygame.draw.rect(
        window,
        (180, 190, 200),
        (x0, y0, panel_w, panel_h),
        1,
    )

    tx = x0 + _AMBIENT_PAD_X
    ty = y0 + _AMBIENT_PAD_Y
    window.blit(val, (tx, ty))


def _parse_and_update(data: bytes):
    """解析 BLE 帧并更新 shared_data/shared_temp（与 foot_hall_viz_region 串口解析逻辑一致）"""
    if len(data) < FRAME_LEN:
        return
    if data[0] != FRAME_HEADER or data[2] != 0xF0 or data[3] != 0x02:
        return
    raw = data[DATA_OFFSET : DATA_OFFSET + DATA_BYTES]
    wire_xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
    wire_t = np.zeros(NUM_SENSORS, dtype=np.int32)
    for i in range(NUM_SENSORS):
        chunk_s = raw[i * 8 : (i + 1) * 8]
        if len(chunk_s) < 8:
            return
        try:
            t_x10, x, y, z = struct.unpack(ENDIAN_FMT, chunk_s)
            wire_t[i] = clamp_wire_i16(t_x10)
            wire_xyz[i, 0] = clamp_wire_i16(x)
            wire_xyz[i, 1] = clamp_wire_i16(y)
            wire_xyz[i, 2] = clamp_wire_i16(z)
        except struct.error:
            return

    ext_xyz = _unwrap_xyz.push_wire_xyz(wire_xyz)
    ext_xyz = clip_mag_ext_array(ext_xyz.copy())

    temp_xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
    temp_t = np.zeros(NUM_SENSORS, dtype=np.int32)
    for i in range(NUM_SENSORS):
        tx, ty, tz = (int(ext_xyz[i, 0]), int(ext_xyz[i, 1]), int(ext_xyz[i, 2]))
        fx, fy, fz = sensor_filters[i].filter((float(tx), float(ty), float(tz)))
        cx = clamp_mag_ext_i32(fx)
        cy = clamp_mag_ext_i32(fy)
        cz = clamp_mag_ext_i32(fz)
        if cx is None or cy is None or cz is None:
            return
        temp_xyz[i, 0] = cx
        temp_xyz[i, 1] = cy
        temp_xyz[i, 2] = cz
        temp_t[i] = clamp_temp_wire_i32(wire_t[i])
    # 脚跟区域的已知 X 方向修正只执行一次。此前在滤波前和滤波后各
    # 交换一次，两个交换相互抵消。
    x12, x13 = temp_xyz[12, 0].item(), temp_xyz[13, 0].item()
    temp_xyz[12, 0], temp_xyz[13, 0] = x13, x12
    with data_lock:
        np.copyto(shared_data, temp_xyz)
        np.copyto(shared_temp, temp_t)


def _ble_worker_thread(device, stop_event, do_calibrate_list):
    """在子线程中运行 BLE 客户端，订阅 Notify 并更新 shared_data"""
    async def _run():
        client = None
        try:
            client = BleakClient(device)
            await client.connect()
            _unwrap_xyz.reset()

            def on_notify(sender, data: bytearray):
                _parse_and_update(bytes(data))

            await client.start_notify(CHAR_UUID, on_notify)

            # 连接成功后 1 秒触发自动校准（相当于自动按 B 键）
            await asyncio.sleep(1.0)
            if do_calibrate_list is not None and not stop_event.is_set():
                do_calibrate_list[0] = True

            while not stop_event.is_set():
                await asyncio.sleep(0.1)

            await client.stop_notify(CHAR_UUID)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            if client and client.is_connected:
                await client.disconnect()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run())


def _find_footsensor():
    """扫描并返回 FootSensor15 设备"""
    return asyncio.run(
        BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    )


def main():
    device = _find_footsensor()
    if not device:
        return

    stop_event = threading.Event()
    do_calibrate_list = [False]  # BLE 连接成功 1 秒后自动设为 True，触发校准
    worker_factory = lambda ev: threading.Thread(
        target=_ble_worker_thread, args=(device, ev, do_calibrate_list), daemon=True
    ).start()

    # 等待首帧数据到达，便于校准
    time.sleep(0.5)

    # 超分辨可视化：IDW 网格插值 + 适度平滑，连接成功 1 秒后自动校准
    fhv.main(
        worker_factory=worker_factory,
        post_smooth_xy_alpha=0.28,
        post_smooth_z_alpha=0.25,
        do_calibrate_list=do_calibrate_list,
        frame_overlay=_draw_ambient_temp_overlay,
    )


if __name__ == "__main__":
    main()
