from __future__ import annotations

from run_hall_magnetic_sim2sim import adaptive_recovery_gate


def test_recovery_gate_rejects_stationary_false_positive() -> None:
    assert not adaptive_recovery_gate(
        0.6,
        (0.8, 0.2, 0.8),
        [0.36, 0.02, 0.002],
        [0.60, 0.08, 0.136],
        [0.26, 0.66, 0.62],
    )


def test_recovery_gate_accepts_meaningful_speed_and_risk_recovery() -> None:
    assert adaptive_recovery_gate(
        0.6,
        (0.8, 0.2, 0.8),
        [0.36, 0.06, 0.35],
        [0.60, 0.09, 0.60],
        [0.25, 0.60, 0.22],
    )
