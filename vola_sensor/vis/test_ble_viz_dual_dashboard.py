#!/usr/bin/env python3
"""双脚 Hall 可视化的无硬件回归测试。"""

from __future__ import annotations

import os
from pathlib import Path
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from ble_viz_dashboard_demo import Int16Unwrapper
from ble_viz_dual_dashboard import (
    DEFAULT_LAYOUT_PATH,
    DualBLESource,
    FootRuntime,
    NUM_SENSORS,
    SensorFrame,
    load_sensor_layout,
)


def make_frame(seq: int, xyz: np.ndarray, timestamp: float | None = None) -> SensorFrame:
    return SensorFrame(
        xyz=np.asarray(xyz, dtype=np.int32),
        temp_x10=np.full(NUM_SENSORS, 250, dtype=np.int32),
        timestamp=time.monotonic() if timestamp is None else timestamp,
        seq=seq,
    )


class LayoutTest(unittest.TestCase):
    def test_a4_layout_keeps_p00_to_p14_order_and_outline(self) -> None:
        positions, outline = load_sensor_layout(Path(DEFAULT_LAYOUT_PATH))
        self.assertEqual(positions.shape, (15, 2))
        self.assertGreaterEqual(outline.shape[0], 16)
        # 三组中心分别为 P02、P07、P12，且沿脚尖到脚跟排列。
        self.assertLess(positions[2, 1], positions[7, 1])
        self.assertLess(positions[7, 1], positions[12, 1])


class Int16UnwrapperTest(unittest.TestCase):
    def test_only_endpoint_crossing_is_unwrapped_and_returns(self) -> None:
        unwrap = Int16Unwrapper((1, 3))
        first = np.asarray([[32700, 30000, -30000]], dtype=np.int32)
        second = np.asarray([[-32700, -10000, 10000]], dtype=np.int32)
        third = first.copy()
        np.testing.assert_array_equal(unwrap.push(first), first)
        np.testing.assert_array_equal(
            unwrap.push(second),
            np.asarray([[32836, -10000, 10000]], dtype=np.int32),
        )
        np.testing.assert_array_equal(unwrap.push(third), first)
        self.assertEqual(unwrap.wrap_events, 2)

    def test_reset_reanchors_after_reconnect(self) -> None:
        unwrap = Int16Unwrapper((1, 3))
        unwrap.push(np.asarray([[32700, 0, 0]], dtype=np.int32))
        unwrap.push(np.asarray([[-32700, 0, 0]], dtype=np.int32))
        unwrap.reset()
        restarted = np.asarray([[-12000, 500, -500]], dtype=np.int32)
        np.testing.assert_array_equal(unwrap.push(restarted), restarted)


class FootRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.foot = FootRuntime("left")
        self.foot.calibrating = False
        self.foot.baseline = np.zeros((NUM_SENSORS, 3), dtype=np.float32)
        self.foot.noise_sigma.fill(20.0)
        self.seq = 0

    def push_z(self, value: int, repeats: int = 1) -> None:
        for _ in range(repeats):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[0, 2] = value
            self.foot.process(make_frame(self.seq, xyz), paused=False)

    def test_hysteresis_deadzone_rejects_idle_noise(self) -> None:
        for _ in range(20):
            self.seq += 1
            xyz = np.full((NUM_SENSORS, 3), 200, dtype=np.int32)
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.assertEqual(self.foot.metrics.peak, 0.0)
        self.assertFalse(np.any(self.foot.active_components))

    def test_press_then_static_residual_is_not_absorbed_into_baseline(self) -> None:
        for _ in range(12):
            self.push_z(2000)
            self.foot.build_heat_surface()
        self.assertGreater(self.foot.metrics.peak, 1000.0)
        self.assertGreater(float(np.max(self.foot.force_field_ema)), 0.1)
        for _ in range(120):
            self.push_z(650)
            self.foot.build_heat_surface()
        self.assertGreater(self.foot.metrics.peak, 300.0)
        self.assertEqual(float(self.foot.baseline[0, 2]), 0.0)

    def test_direct_small_static_response_is_not_auto_zeroed(self) -> None:
        self.push_z(650, repeats=120)
        self.assertGreater(self.foot.metrics.peak, 300.0)
        self.assertEqual(float(self.foot.baseline[0, 2]), 0.0)

    def test_sustained_foot_wide_load_never_auto_zeroes(self) -> None:
        for _ in range(600):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[:, 2] = 1000
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.assertEqual(float(self.foot.baseline[0, 2]), 0.0)
        self.assertGreater(self.foot.metrics.peak, 700.0)
        self.assertGreater(self.foot.load, 10_000.0)

    def test_foot_wide_peak_drop_does_not_claim_unload(self) -> None:
        for _ in range(20):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[:, 2] = 1000
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        for _ in range(300):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[:, 2] = 400
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.assertEqual(float(self.foot.baseline[0, 2]), 0.0)
        self.assertGreater(self.foot.metrics.peak, 200.0)

    def test_explicit_empty_recalibration_changes_baseline(self) -> None:
        self.foot.request_calibration()
        for _ in range(self.foot.calibration_target):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[:, 2] = 400
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.foot.finish_calibration()
        self.assertEqual(float(self.foot.baseline[0, 2]), 400.0)
        for _ in range(20):
            self.seq += 1
            xyz = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
            xyz[:, 2] = 400
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.assertEqual(self.foot.metrics.peak, 0.0)

    def test_no_data_watchdog_clears_stale_display(self) -> None:
        self.foot.filtered.fill(500.0)
        self.foot.last_frame_time = time.monotonic() - 0.6
        self.foot._last_watchdog_at = time.monotonic() - 0.1
        self.foot.update_watchdog()
        self.assertFalse(np.any(self.foot.filtered))
        self.assertEqual(self.foot.metrics.peak, 0.0)

    def test_interrupted_calibration_discards_partial_window(self) -> None:
        self.foot.request_calibration()
        for _ in range(20):
            self.seq += 1
            xyz = np.full((NUM_SENSORS, 3), 25, dtype=np.int32)
            self.foot.process(make_frame(self.seq, xyz), paused=False)
        self.assertEqual(len(self.foot.calibration_samples), 20)
        self.foot.calibration_last_sample_at = time.monotonic() - 0.6
        self.foot.update_watchdog()
        self.assertTrue(self.foot.calibrating)
        self.assertIsNone(self.foot.baseline)
        self.assertEqual(self.foot.calibration_samples, [])

    def test_shared_heat_scale_is_applied_without_peak_memory(self) -> None:
        self.foot.filtered[0, 2] = 4000.0
        self.foot.last_seq = 1
        self.foot.build_heat_surface(display_vmax=5000.0)
        self.assertEqual(self.foot.force_vmax, 5000.0)
        self.foot.filtered.fill(0.0)
        self.foot.last_seq = 2
        self.foot.build_heat_surface(display_vmax=850.0)
        self.assertEqual(self.foot.force_vmax, 850.0)


class PairingTest(unittest.TestCase):
    def test_same_ble_address_cannot_be_both_feet(self) -> None:
        with self.assertRaisesRegex(ValueError, "同一个 BLE 地址"):
            DualBLESource(left_address="AA:BB", right_address="aa:bb")

    def test_stop_before_start_is_complete(self) -> None:
        source = DualBLESource(left_address="LEFT", right_address="RIGHT")
        self.assertTrue(source.stop())

    def test_async_100_hz_streams_are_counted_before_50_hz_display(self) -> None:
        source = DualBLESource(
            left_address="LEFT",
            right_address="RIGHT",
            left_adapter="hci0",
            right_adapter="hci1",
        )
        now = time.monotonic()
        zeros = np.zeros((NUM_SENSORS, 3), dtype=np.int32)
        for index in range(64):
            stamp = now - 0.631 + index * 0.010
            source.slots["left"].frames.append(make_frame(index + 1, zeros, stamp))
            source.slots["right"].frames.append(make_frame(index + 1, zeros, stamp + 0.003))
        sample = source.sample()
        self.assertIsNotNone(sample["left"])
        self.assertIsNotNone(sample["right"])
        self.assertIn("L100/R100原始", source.detail)
        self.assertIn("配对100", source.detail)
        self.assertIn("显示50", source.detail)


if __name__ == "__main__":
    pygame.init()
    try:
        unittest.main()
    finally:
        pygame.quit()
