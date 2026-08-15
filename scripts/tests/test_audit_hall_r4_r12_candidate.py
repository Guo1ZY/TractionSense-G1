from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/traction/audit_hall_r4_r12_candidate.py"
SPEC = importlib.util.spec_from_file_location("audit_hall_r4_r12", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_candidate_passes_simulation_but_not_hardware_gate() -> None:
    report = MODULE.audit()
    assert report["simulation_pass"]
    assert report["hardware_validated"] is False
    assert report["status"] == "SIMULATION_ACCEPTANCE_PASS_REAL_HARNESS_REQUIRED"
    assert report["gates"]["full_hall_loss_stops"]
    assert report["gates"]["package_inactive"]
