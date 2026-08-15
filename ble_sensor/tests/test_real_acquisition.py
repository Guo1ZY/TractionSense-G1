from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibrate_magnetic import (
    MAG_COLUMNS,
    TEMP_COLUMNS,
    fit_normalization_document,
)
from capture_magnetic_dataset import _selected_phases
from dual_foot_bridge.bridge import _load_calibrations, _load_config
from dual_foot_bridge.magnetic_bridge import load_normalizers
from dual_foot_bridge.normalization import MagneticNormalizer


def _write_csv(path: Path, *, motion: bool, bad_temp_channel: int | None = None) -> None:
    fieldnames = ["side", *TEMP_COLUMNS, *MAG_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for side_offset, side in enumerate(("left", "right")):
            for frame in range(30):
                temperature = np.full(15, 240 + frame, dtype=float)
                if bad_temp_channel is not None:
                    temperature[bad_temp_channel] = -9980.0
                magnetic = np.zeros((15, 3), dtype=float)
                for sensor in range(15):
                    magnetic[sensor] = (
                        1000.0
                        + side_offset * 50.0
                        + sensor * 3.0
                        + np.arange(3)
                        + 0.4 * (temperature[sensor] - 250.0)
                    )
                if motion:
                    magnetic += (frame - 15) * np.array([2.0, -1.5, 3.0])
                row = {"side": side}
                row.update(dict(zip(TEMP_COLUMNS, temperature)))
                row.update(dict(zip(MAG_COLUMNS, magnetic.reshape(-1))))
                writer.writerow(row)


class IdentityConfigTests(unittest.TestCase):
    def test_unique_left_right_names_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            base = {
                "format": "g1-dual-foot-ble-config-v1",
                "left": {"device_name": "left"},
                "right": {"device_name": "right"},
            }
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(_load_config(path)["left"]["device_name"], "left")
            base["right"]["device_name"] = "LEFT"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_config(path)

    def test_raw_only_never_requires_future_calibration_files(self) -> None:
        config = {
            "left": {
                "normalization": "missing-left.json",
                "calibration": "missing-left-force.json",
            },
            "right": {
                "normalization": "missing-right.json",
                "calibration": "missing-right-force.json",
            },
        }
        self.assertEqual(
            load_normalizers(config, Path("/does/not/exist"), raw_only=True),
            {"left": None, "right": None},
        )
        self.assertEqual(
            _load_calibrations(config, Path("/does/not/exist"), raw_only=True),
            {"left": None, "right": None},
        )


class MagneticNormalizationTests(unittest.TestCase):
    def test_fit_contains_only_hall_temperature_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.csv"
            motion = Path(directory) / "motion.csv"
            _write_csv(baseline, motion=False)
            _write_csv(motion, motion=True)
            document = fit_normalization_document([baseline], [motion], "left")
        self.assertEqual(document["force_conversion"], "absent")
        self.assertTrue(document["diagnostics"]["finite"])
        self.assertEqual(np.asarray(document["baseline_xyz"]).shape, (15, 3))
        self.assertEqual(np.asarray(document["scale_xyz"]).shape, (15, 3))
        self.assertNotIn("normal_force", document)
        self.assertNotIn("tangential_force", document)

    def test_bad_temperature_channel_disables_compensation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.csv"
            motion = Path(directory) / "motion.csv"
            _write_csv(baseline, motion=False, bad_temp_channel=0)
            _write_csv(motion, motion=True, bad_temp_channel=0)
            document = fit_normalization_document([baseline], [motion], "right")
        self.assertEqual(document["bad_temperature_channels"], [0])
        self.assertEqual(document["diagnostics"]["bad_temperature_channels"], [0])
        coefficient = np.asarray(document["temperature_coefficient_per_x10"])
        self.assertTrue(np.all(coefficient[0] == 0.0))
        self.assertGreaterEqual(
            float(document["reference_temperature_x10"][0]), -400.0
        )
        self.assertLessEqual(float(document["reference_temperature_x10"][0]), 1250.0)
        self.assertGreater(float(document["scale_xyz"][0][0]), 0.0)

    def test_normalizer_skips_bad_channel_temperature_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.csv"
            motion = Path(directory) / "motion.csv"
            _write_csv(baseline, motion=False, bad_temp_channel=0)
            _write_csv(motion, motion=True, bad_temp_channel=0)
            document = fit_normalization_document([baseline], [motion], "right")
            path = Path(directory) / "right.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            normalizer = MagneticNormalizer.load(path, expected_side="right")
            self.assertEqual(normalizer.bad_temperature_channels, (0,))
            magnetic = np.zeros((15, 3), dtype=np.float64)
            temperature = np.full(15, 250, dtype=np.float64)
            temperature[0] = -9980.0
            result = normalizer.normalize(magnetic, temperature)
        self.assertEqual(result.shape, (15, 3))
        self.assertTrue(np.isfinite(result).all())

    def test_quick_plan_keeps_physical_phase_labels(self) -> None:
        phases = _selected_phases(["baseline_unloaded", "shear_x"], quick=True)
        self.assertEqual([phase.key for phase in phases], ["baseline_unloaded", "shear_x"])
        self.assertLessEqual(max(phase.duration_s for phase in phases), 12.0)


if __name__ == "__main__":
    unittest.main()
