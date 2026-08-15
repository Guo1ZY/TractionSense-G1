from __future__ import annotations

import csv
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dual_foot_bridge.bridge import FootState, RawCsvLogger, SensorPipeline
from dual_foot_bridge.capture_ipc import (
    F0R1Writer,
    PACKET_SIZE,
    PairedCsvLogger,
    read_packet,
)
from dual_foot_bridge.protocol import NUM_SENSORS, make_test_frame
from capture_robot_hall import _summarize_raw_frame_timing


class CaptureIpcTests(unittest.TestCase):
    def test_f0r1_round_trip_preserves_left_right_and_timestamps(self) -> None:
        left = np.arange(NUM_SENSORS * 3, dtype=np.int64).reshape(NUM_SENSORS, 3)
        right = -left - 1000
        left_temp = np.arange(NUM_SENSORS, dtype=np.int32) + 230
        right_temp = np.arange(NUM_SENSORS, dtype=np.int32) + 260
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            writer = F0R1Writer(path)
            writer.write(
                left,
                right,
                left_temp,
                right_temp,
                publish_wall_ns=10,
                publish_monotonic_ns=20,
                frame_wall_ns=(11, 12),
                frame_monotonic_ns=(18, 19),
                source_sequence=(250, 7),
                valid=(True, False),
                age_s=(0.002, 1.0e9),
                period_s=(0.01, 0.02),
            )
            self.assertEqual(path.stat().st_size, PACKET_SIZE)
            sample = read_packet(path)
        np.testing.assert_array_equal(sample.magnetic[0], left)
        np.testing.assert_array_equal(sample.magnetic[1], right)
        np.testing.assert_array_equal(sample.temperature_x10[0], left_temp)
        np.testing.assert_array_equal(sample.temperature_x10[1], right_temp)
        self.assertEqual(sample.frame_monotonic_ns, (18, 19))
        self.assertEqual(sample.valid, (True, False))

    def test_paired_csv_uses_explicit_side_and_p00_order(self) -> None:
        magnetic = np.zeros((NUM_SENSORS, 3), dtype=np.int64)
        magnetic[0] = (101, 102, 103)
        temperature = np.arange(NUM_SENSORS, dtype=np.int32) + 240
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            packet_path = directory_path / "capture.bin"
            csv_path = directory_path / "paired.csv"
            sample = F0R1Writer(packet_path).write(
                magnetic,
                -magnetic,
                temperature,
                temperature + 10,
                publish_wall_ns=100,
                publish_monotonic_ns=200,
                frame_wall_ns=(90, 91),
                frame_monotonic_ns=(180, 170),
                source_sequence=(1, 2),
                valid=(True, True),
                age_s=(0.02, 0.03),
                period_s=(0.01, 0.011),
            )
            logger = PairedCsvLogger(csv_path)
            logger.write(sample)
            logger.close()
            with csv_path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["left_P00_bx"], "101")
        self.assertEqual(row["left_P00_by"], "102")
        self.assertEqual(row["left_P00_bz"], "103")
        self.assertEqual(row["right_P00_bx"], "-101")
        self.assertEqual(row["left_right_frame_skew_ns"], "10")


class RawFrameLoggerTests(unittest.TestCase):
    def test_raw_csv_is_not_the_ema_filtered_runtime_value(self) -> None:
        side_config = {
            "sensor_permutation": list(range(NUM_SENSORS)),
            "axis_sign": [1, 1, 1],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            state = FootState(side="left", address="left-address", adapter="hci0")
            logger = RawCsvLogger(path, {"left": None, "right": None})
            pipeline = SensorPipeline("left", side_config, state, logger, ema_alpha=0.25)
            first = np.full((NUM_SENSORS, 3), 100, dtype=np.int16)
            second = np.full((NUM_SENSORS, 3), 200, dtype=np.int16)
            pipeline.receive(make_test_frame(first, sequence=1))
            pipeline.receive(make_test_frame(second, sequence=2))
            logger.close()
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]["mag_0_x"], "200")
        self.assertEqual(int(state.raw_magnetic_xyz[0, 0]), 200)
        self.assertAlmostEqual(float(state.magnetic_xyz[0, 0]), 125.0)
        self.assertGreater(state.last_monotonic_ns, 0)
        self.assertLess(time.monotonic_ns() - state.last_monotonic_ns, 1_000_000_000)


class RawTimingSummaryTests(unittest.TestCase):
    def test_keeps_left_right_notify_timing_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text(
                "side,monotonic_ns\n"
                "left,0\nleft,10000000\nleft,20000000\n"
                "right,0\nright,20000000\nright,100000000\n",
                encoding="utf-8",
            )
            result = _summarize_raw_frame_timing(path)
        self.assertEqual(result["left"]["mean_rate_hz"], 100.0)
        self.assertEqual(result["right"]["mean_rate_hz"], 20.0)
        self.assertEqual(result["right"]["intervals_ge_40ms"], 1)


if __name__ == "__main__":
    unittest.main()
