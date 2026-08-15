from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/traction/audit_hall_r4_r16_candidate.py"
SPEC = importlib.util.spec_from_file_location("audit_hall_r4_r16", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_passes_sim_safety_but_keeps_real_and_performance_gates() -> None:
    report = MODULE.audit()
    assert report["simulation_safety_adaptation_pass"]
    assert report["research_performance_target_pass"] is False
    assert report["hardware_validated"] is False
    assert report["status"] == (
        "SIMULATION_SAFETY_ADAPTATION_PASS_PERFORMANCE_TARGET_PARTIAL_"
        "REAL_HARNESS_REQUIRED"
    )
    assert report["gates"]["full_hall_loss_stops"]
    assert report["gates"]["package_inactive"]
    assert report["gates"]["mujoco_microstep_net_forward"]
