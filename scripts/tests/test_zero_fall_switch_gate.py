from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "traction"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from zero_fall_switch_gate import build_report


FIELDS = ["phase", "mu", "cmd_vx", "steady_vx", "response_time_s", "falls"]


def _write(path: Path, *, falls: int = 0, response: float = 0.4) -> Path:
    rows = [
        {"phase": 0, "mu": 0.8, "cmd_vx": 0.8, "steady_vx": 0.68, "response_time_s": "nan", "falls": 0},
        {"phase": 1, "mu": 0.08, "cmd_vx": 0.8, "steady_vx": 0.20, "response_time_s": response, "falls": falls},
        {"phase": 2, "mu": 0.8, "cmd_vx": 0.8, "steady_vx": 0.66, "response_time_s": response, "falls": 0},
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_zero_fall_gate_requires_every_nominal_and_fault_run_to_pass(tmp_path: Path) -> None:
    nominal = [_write(tmp_path / f"nominal_{i}.csv") for i in range(3)]
    faults = [_write(tmp_path / f"fault_{i}.csv") for i in range(3)]
    report = build_report(
        nominal_paths=nominal,
        fault_paths=faults,
        min_runs=3,
        high_mu_threshold=0.75,
        low_mu_threshold=0.20,
        min_high_tracking_fraction=0.70,
        max_low_speed=0.45,
        max_response_s=0.60,
    )
    assert report["pass"]


def test_zero_fall_gate_rejects_one_fall_even_if_other_runs_pass(tmp_path: Path) -> None:
    nominal = [_write(tmp_path / f"nominal_{i}.csv") for i in range(3)]
    faults = [_write(tmp_path / "fault_0.csv", falls=1)] + [
        _write(tmp_path / f"fault_{i}.csv") for i in range(1, 3)
    ]
    report = build_report(
        nominal_paths=nominal,
        fault_paths=faults,
        min_runs=3,
        high_mu_threshold=0.75,
        low_mu_threshold=0.20,
        min_high_tracking_fraction=0.70,
        max_low_speed=0.45,
        max_response_s=0.60,
    )
    assert not report["pass"]
    assert report["sensor_fault_randomized"]["runs"][0]["falls"] == 1
