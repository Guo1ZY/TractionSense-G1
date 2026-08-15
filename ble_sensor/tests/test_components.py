from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dual_foot_bridge.calibration import Calibration, calibration_document
from dual_foot_bridge.ipc import F0T1Writer, PACKET, read_packet
from dual_foot_bridge.magnetic_ipc import (
    F0M1Writer,
    PACKET_SIZE as MAGNETIC_PACKET_SIZE,
    read_packet as read_magnetic_packet,
)
from dual_foot_bridge.protocol import (
    FrameError,
    FrameParser,
    NUM_SENSORS,
    Int16Unwrapper3D,
    make_test_frame,
    transform_magnetic,
)


class ProtocolTests(unittest.TestCase):
    def test_parse_and_unwrap_are_per_parser(self) -> None:
        first = np.full((NUM_SENSORS, 3), 32760, dtype=np.int16)
        wrapped = np.full((NUM_SENSORS, 3), -32760, dtype=np.int16)
        parser_a = FrameParser()
        parser_b = FrameParser()
        parsed_first = parser_a.parse(make_test_frame(first, sequence=7))
        parsed_wrapped = parser_a.parse(make_test_frame(wrapped, sequence=8))
        independent = parser_b.parse(make_test_frame(wrapped, sequence=9))
        self.assertEqual(parsed_first.source_sequence, 7)
        self.assertTrue(np.all(parsed_wrapped.magnetic_xyz == 32776))
        self.assertTrue(np.all(independent.magnetic_xyz == -32760))

    def test_over_range_wraps_back_and_flags_saturation(self) -> None:
        unwrapper = Int16Unwrapper3D()
        unwrapper.push(np.zeros((NUM_SENSORS, 3), dtype=np.int64))
        unwrapper.push(np.full((NUM_SENSORS, 3), 30000, dtype=np.int64))
        result = unwrapper.push(np.full((NUM_SENSORS, 3), -30000, dtype=np.int64))
        self.assertTrue(np.all(np.abs(result) <= 32768))
        self.assertIsNotNone(unwrapper.last_saturation)
        self.assertTrue(bool(np.any(unwrapper.last_saturation)))

    def test_rail_value_flags_saturation_without_changing_value(self) -> None:
        unwrapper = Int16Unwrapper3D()
        unwrapper.push(np.zeros((NUM_SENSORS, 3), dtype=np.int64))
        at_rail = np.full((NUM_SENSORS, 3), 32767, dtype=np.int64)
        result = unwrapper.push(at_rail)
        self.assertTrue(bool(np.all(unwrapper.last_saturation)))
        self.assertTrue(np.all(result == 32767))

    def test_reject_short_frame(self) -> None:
        with self.assertRaises(FrameError):
            FrameParser().parse(b"\x7d")

    def test_transform(self) -> None:
        values = np.arange(NUM_SENSORS * 3).reshape(NUM_SENSORS, 3)
        order = list(reversed(range(NUM_SENSORS)))
        result = transform_magnetic(values, order, [1, -1, 1])
        np.testing.assert_array_equal(result[0], values[-1] * [1, -1, 1])


class CalibrationTests(unittest.TestCase):
    def test_estimate_and_round_trip(self) -> None:
        baseline = np.full((NUM_SENSORS, 3), 10.0)
        weights = np.zeros(NUM_SENSORS * 3)
        weights[0] = 2.0
        document = calibration_document(
            side="left",
            baseline_xyz=baseline,
            normal_weights=weights,
            normal_bias=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "left.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            model = Calibration.load(path, expected_side="left")
        magnetic = baseline.copy()
        magnetic[0, 0] += 3.0
        estimate = model.estimate(magnetic)
        self.assertAlmostEqual(estimate.normal_n, 7.0)
        self.assertEqual(estimate.tangent_n, 0.0)


class IpcTests(unittest.TestCase):
    def test_packet_layout_and_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foot.bin"
            writer = F0T1Writer(path)
            writer.write(100.0, 250.0, 12.0, 0.0)
            self.assertEqual(path.stat().st_size, PACKET.size)
            sample = read_packet(path)
            self.assertEqual(sample.sequence, 0)
            self.assertEqual(sample.contact_left, 1.0)
            self.assertAlmostEqual(sample.normal_left_policy, 1.0)
            self.assertAlmostEqual(sample.normal_right_policy, 2.5)
            self.assertLess(time.time_ns() - sample.stamp_ns, 1_000_000_000)

    def test_magnetic_packet_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "magnetic.bin"
            left = np.arange(NUM_SENSORS * 3, dtype=np.float32).reshape(NUM_SENSORS, 3) / 10
            right = -left
            F0M1Writer(path).write(
                left,
                right,
                valid_left=1.0,
                valid_right=1.0,
                age_left_s=0.01,
                age_right_s=0.02,
                period_left_s=0.02,
                period_right_s=0.05,
            )
            self.assertEqual(path.stat().st_size, MAGNETIC_PACKET_SIZE)
            sample = read_magnetic_packet(path)
            self.assertEqual(sample.magnetic.shape, (2, NUM_SENSORS, 3))
            np.testing.assert_allclose(sample.magnetic[0], np.clip(left, -6, 6))
            np.testing.assert_allclose(sample.magnetic[1], np.clip(right, -6, 6))


if __name__ == "__main__":
    unittest.main()
