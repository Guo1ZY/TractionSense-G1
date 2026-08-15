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
from dual_foot_bridge.magnetic_bridge import (
    DualFootSynchronizer,
    _magnetic_health_document,
)
from dual_foot_bridge.bridge import HallSample, FootState, configured_adapters
from dual_foot_bridge.protocol import (
    FrameError,
    FrameParser,
    NUM_SENSORS,
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

    def test_ordinary_large_jump_is_not_misclassified_as_wrap(self) -> None:
        first = np.full((NUM_SENSORS, 3), 30000, dtype=np.int16)
        second = np.full((NUM_SENSORS, 3), -10000, dtype=np.int16)
        parser = FrameParser()
        parser.parse(make_test_frame(first, sequence=1))
        parsed = parser.parse(make_test_frame(second, sequence=2))
        self.assertTrue(np.all(parsed.magnetic_xyz == -10000))

    def test_reject_short_frame(self) -> None:
        with self.assertRaises(FrameError):
            FrameParser().parse(b"\x7d")

    def test_transform(self) -> None:
        values = np.arange(NUM_SENSORS * 3).reshape(NUM_SENSORS, 3)
        order = list(reversed(range(NUM_SENSORS)))
        result = transform_magnetic(values, order, [1, -1, 1])
        np.testing.assert_array_equal(result[0], values[-1] * [1, -1, 1])


class AdapterConfigurationTests(unittest.TestCase):
    def test_two_explicit_adapters_remain_side_specific(self) -> None:
        config = {"left": {"adapter": "hci0"}, "right": {"adapter": "hci1"}}
        self.assertEqual(
            configured_adapters(config), {"left": "hci0", "right": "hci1"}
        )

    def test_partial_or_shared_adapter_assignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            configured_adapters({"left": {"adapter": "hci0"}, "right": {}})
        with self.assertRaisesRegex(ValueError, "must be different"):
            configured_adapters(
                {"left": {"adapter": "hci0"}, "right": {"adapter": "hci0"}}
            )


class DualFootSynchronizationTests(unittest.TestCase):
    @staticmethod
    def _sample(stamp: float, value: float) -> HallSample:
        return HallSample(
            received_wall_ns=int(stamp * 1.0e9),
            received_monotonic=stamp,
            source_sequence=0,
            sample_period_s=0.01,
            temperature_x10=np.full(NUM_SENSORS, 250, dtype=np.int32),
            magnetic_xyz=np.full((NUM_SENSORS, 3), value, dtype=np.float64),
        )

    def test_newest_pair_within_bound_preserves_left_right_identity(self) -> None:
        states = {
            side: FootState(side=side, address=f"{side}-address")
            for side in ("left", "right")
        }
        states["left"].samples.extend(
            (self._sample(9.980, 1.0), self._sample(10.000, 2.0))
        )
        states["right"].samples.extend(
            (self._sample(9.981, 3.0), self._sample(10.007, 4.0))
        )
        synchronizer = DualFootSynchronizer(
            max_pair_skew_s=0.010, source_timeout_s=0.20
        )
        pair = synchronizer.match(states, now=10.010)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertAlmostEqual(pair.skew_s, 0.007)
        self.assertTrue(np.all(pair.left.magnetic_xyz == 2.0))
        self.assertTrue(np.all(pair.right.magnetic_xyz == 4.0))
        self.assertTrue(synchronizer.is_synchronized(now=10.020, grace_s=0.025))
        self.assertFalse(synchronizer.is_synchronized(now=10.050, grace_s=0.025))
        self.assertIsNone(synchronizer.match(states, now=10.020))

    def test_pair_over_skew_bound_is_not_released(self) -> None:
        states = {
            side: FootState(side=side, address=f"{side}-address")
            for side in ("left", "right")
        }
        states["left"].samples.append(self._sample(5.000, 1.0))
        states["right"].samples.append(self._sample(5.011, 2.0))
        synchronizer = DualFootSynchronizer(
            max_pair_skew_s=0.010, source_timeout_s=0.20
        )
        self.assertIsNone(synchronizer.match(states, now=5.020))
        self.assertEqual(synchronizer.synchronized_pairs, 0)
        self.assertEqual(synchronizer.sync_misses, 1)

    def test_holdback_waits_for_both_adapter_streams(self) -> None:
        states = {
            side: FootState(side=side, address=f"{side}-address")
            for side in ("left", "right")
        }
        states["left"].samples.append(self._sample(3.000, 1.0))
        states["right"].samples.append(self._sample(3.006, 2.0))
        synchronizer = DualFootSynchronizer(
            max_pair_skew_s=0.010,
            source_timeout_s=0.20,
            holdback_s=0.020,
        )
        self.assertIsNone(synchronizer.match(states, now=3.015))
        pair = synchronizer.match(states, now=3.030)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertAlmostEqual(pair.skew_s, 0.006)


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

    def test_magnetic_health_never_exposes_force_fields(self) -> None:
        states = {
            side: FootState(side=side, address=f"{side}-address")
            for side in ("left", "right")
        }
        document = _magnetic_health_document(
            states,
            {"left": None, "right": None},
            source_timeout_s=0.20,
            publishing=False,
        )
        self.assertFalse(document["force_available"])
        self.assertEqual(document["raw_unit"], "device_count")
        for foot in document["feet"].values():
            self.assertNotIn("normal_n", foot)
            self.assertNotIn("tangent_n", foot)


if __name__ == "__main__":
    unittest.main()
