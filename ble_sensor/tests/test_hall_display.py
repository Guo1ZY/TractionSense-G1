from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dual_foot_bridge.hall_display import (
    DualHallDisplayFilter,
    HallFootDisplayFilter,
    load_display_layout,
)
from dual_foot_bridge.protocol import NUM_SENSORS


class DisplayFilterTests(unittest.TestCase):
    def calibrate(self, display: DualHallDisplayFilter) -> None:
        display.begin_unloaded_baseline()
        zero = np.zeros((2, NUM_SENSORS, 3), dtype=np.int64)
        for index in range(800):
            display.update(zero, (True, True), sample_time_s=index * 0.02)
        self.assertTrue(display.baseline_ready)

    def test_sustained_hall_change_is_never_learned_away(self) -> None:
        display = DualHallDisplayFilter()
        self.calibrate(display)
        held = np.zeros((2, NUM_SENSORS, 3), dtype=np.int64)
        held[:, 2, 0] = 1200
        for index in range(500):
            display.update(held, (True, True), sample_time_s=20.0 + index * 0.02)
        self.assertGreater(display.feet["left"].intensity[2], 1000.0)
        self.assertGreater(display.feet["right"].intensity[2], 1000.0)

    def test_release_returns_display_to_zero_quickly(self) -> None:
        display = DualHallDisplayFilter()
        self.calibrate(display)
        held = np.zeros((2, NUM_SENSORS, 3), dtype=np.int64)
        held[:, 4, 2] = 2000
        for index in range(30):
            display.update(held, (True, True), sample_time_s=20.0 + index * 0.02)
        zero = np.zeros_like(held)
        for index in range(24):
            display.update(zero, (True, True), sample_time_s=21.0 + index * 0.02)
        self.assertLess(display.feet["left"].intensity[4], 4.0)
        self.assertLess(display.feet["right"].intensity[4], 4.0)

    def test_equal_left_right_changes_use_equal_intensity(self) -> None:
        display = DualHallDisplayFilter()
        self.calibrate(display)
        value = np.full((2, NUM_SENSORS, 3), 400, dtype=np.int64)
        for index in range(20):
            display.update(value, (True, True), sample_time_s=20.0 + index * 0.02)
        np.testing.assert_allclose(
            display.feet["left"].intensity,
            display.feet["right"].intensity,
        )

    def test_invalid_side_does_not_consume_baseline_samples(self) -> None:
        foot = HallFootDisplayFilter(calibration_target=2)
        foot.begin_unloaded_baseline()
        data = np.zeros((NUM_SENSORS, 3))
        foot.update(data, valid=False)
        self.assertEqual(len(foot.calibration_samples), 0)

    def test_unloaded_linear_drift_blocks_baseline_lock(self) -> None:
        display = DualHallDisplayFilter()
        display.begin_unloaded_baseline()
        for index in range(1000):
            time_s = index * 0.02
            values = np.zeros((2, NUM_SENSORS, 3), dtype=np.float64)
            values[0, 7, 0] = 4.0 * time_s
            display.update(values, (True, True), sample_time_s=time_s)
        self.assertFalse(display.baseline_ready)
        self.assertEqual(display.feet["left"].calibration_status, "unstable")
        self.assertGreater(display.feet["left"].calibration_drift_max, 3.5)
        self.assertIsNotNone(display.feet["right"].baseline)


class LayoutTests(unittest.TestCase):
    def test_layout_requires_p00_to_p14_order(self) -> None:
        document = {
            "format": "footsensor15-a4-layout-v1",
            "outline_normalized_uv": [[0, 0], [1, 0], [0, 1]],
            "sensors": [
                {"id": index + 1, "output_id": f"P{index:02d}", "normalized_uv": [0.5, 0.5]}
                for index in range(NUM_SENSORS)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            layout = load_display_layout(path)
        self.assertEqual(layout.output_ids[0], "P00")
        self.assertEqual(layout.output_ids[-1], "P14")


if __name__ == "__main__":
    unittest.main()
