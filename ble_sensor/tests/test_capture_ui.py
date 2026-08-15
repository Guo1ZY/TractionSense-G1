from __future__ import annotations

import unittest

from capture_robot_hall_ui import (
    PHASES,
    SAFETY_LABELS,
    assess_capture_quality,
    extract_json_document,
    preflight_blockers,
    required_checks_complete,
    safe_component,
)


class UiDefinitionTests(unittest.TestCase):
    def test_phase_keys_are_unique_and_checks_exist(self) -> None:
        self.assertEqual(len(PHASES), len({phase.key for phase in PHASES}))
        self.assertEqual(
            [phase.key for phase in PHASES],
            ["suspended_unloaded", "dual_standing", "walking_straight"],
        )
        for phase in PHASES:
            self.assertGreater(phase.duration_s, 0)
            self.assertTrue(set(phase.safety).issubset(SAFETY_LABELS))

    def test_required_checks_only_cover_current_phase(self) -> None:
        phase = PHASES[0]
        checks = {key: True for key in phase.safety}
        self.assertTrue(required_checks_complete(phase, checks))
        checks[phase.safety[0]] = False
        self.assertFalse(required_checks_complete(phase, checks))

    def test_safe_component_cannot_make_nested_path(self) -> None:
        self.assertEqual(safe_component("../01 baseline / foot"), "01_baseline_foot")


class PreflightParsingTests(unittest.TestCase):
    def test_extract_json_from_output(self) -> None:
        text = 'prefix\n{"ready": false, "missing_adapters": ["hci1"]}\nsuffix'
        self.assertEqual(extract_json_document(text)["missing_adapters"], ["hci1"])

    def test_blockers_preserve_adapter_and_process_details(self) -> None:
        document = {
            "ready": False,
            "missing_adapters": ["hci1"],
            "competing_ble_processes": [
                {"pid": 123, "script": "ble_viz_superres_hot_detail.py"}
            ],
        }
        blockers = preflight_blockers(document)
        self.assertTrue(any("hci1" in value for value in blockers))
        self.assertTrue(any("123" in value for value in blockers))


class QualityGateTests(unittest.TestCase):
    @staticmethod
    def manifest(rate: float = 110.0, valid: int = 1000, rows: int = 1000):
        return {
            "status": "complete",
            "paired_rows": {
                "rows_after_ready": rows,
                "both_valid_after_ready": valid,
                "abs_frame_skew_ns_p95": 10_000_000,
            },
            "final_health": {
                "feet": {
                    "left": {"sample_rate_hz": rate, "rejected_frames": 0},
                    "right": {"sample_rate_hz": rate, "rejected_frames": 0},
                }
            },
        }

    def test_pass(self) -> None:
        self.assertEqual(assess_capture_quality(self.manifest())[0], "PASS")

    def test_review_for_sub_target_rate_or_skew(self) -> None:
        manifest = self.manifest(rate=85.0)
        manifest["paired_rows"]["abs_frame_skew_ns_p95"] = 25_000_000
        grade, reasons = assess_capture_quality(manifest)
        self.assertEqual(grade, "REVIEW")
        self.assertTrue(any("时间戳" in reason for reason in reasons))

    def test_fail_for_invalid_or_bad_frames(self) -> None:
        manifest = self.manifest(valid=900)
        self.assertEqual(assess_capture_quality(manifest)[0], "FAIL")
        manifest = self.manifest()
        manifest["final_health"]["feet"]["right"]["rejected_frames"] = 1
        self.assertEqual(assess_capture_quality(manifest)[0], "FAIL")

    def test_fail_for_low_rate_or_large_timestamp_skew(self) -> None:
        self.assertEqual(assess_capture_quality(self.manifest(rate=79.9))[0], "FAIL")
        manifest = self.manifest()
        manifest["paired_rows"]["abs_frame_skew_ns_p95"] = 50_000_001
        self.assertEqual(assess_capture_quality(manifest)[0], "FAIL")

    def test_non_complete_status_fails(self) -> None:
        manifest = self.manifest()
        manifest["status"] = "incomplete"
        self.assertEqual(assess_capture_quality(manifest)[0], "FAIL")


if __name__ == "__main__":
    unittest.main()
