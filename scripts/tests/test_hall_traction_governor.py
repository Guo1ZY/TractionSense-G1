from __future__ import annotations

import torch

from unitree_rl_lab.traction.hall_governor import (
    HIGH,
    LOW,
    UNKNOWN,
    HallTractionGovernor,
    HallTractionGovernorCfg,
)


def test_unknown_is_conservative_then_high_evidence_releases_speed() -> None:
    cfg = HallTractionGovernorCfg(
        unknown_warmup_s=0.10,
        probe_duration_s=0.10,
        low_reprobe_s=0.10,
    )
    governor = HallTractionGovernor(2, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.8, 0.2, 0.5], [0.8, 0.2, 0.5]])
    for _ in range(10):
        output, state = governor.update(requested, torch.tensor([0.1, 0.9]))
    assert state.tolist() == [HIGH, LOW]
    assert output[0, 0] <= cfg.probe_speed_limit + 1.0e-5
    assert output[1, 0] <= 0.10001
    for _ in range(45):
        output, state = governor.update(requested, torch.tensor([0.1, 0.9]))
    assert state.tolist() == [HIGH, LOW]
    assert output[0, 0] > 0.10
    assert output[1, 0] <= 0.10001


def test_invalid_or_nan_risk_never_releases_unknown_state() -> None:
    governor = HallTractionGovernor(2, 0.02, "cpu")
    requested = torch.full((2, 3), 1.0)
    for _ in range(100):
        output, state = governor.update(
            requested,
            torch.tensor([float("nan"), 0.0]),
            valid=torch.tensor([True, False]),
        )
    assert state.tolist() == [LOW, LOW]
    assert torch.allclose(output[:, 0], torch.zeros(2), atol=1.0e-6)


def test_low_state_limits_forward_lateral_and_yaw_with_deceleration() -> None:
    cfg = HallTractionGovernorCfg(
        high_hold_s=0.02,
        low_hold_s=0.02,
        unknown_warmup_s=0.02,
        probe_duration_s=0.02,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.9, 0.3, 0.7]])
    for _ in range(30):
        output, state = governor.update(requested, torch.tensor([0.0]))
    assert state.item() == HIGH
    assert output[0, 0] > cfg.low_speed_limit
    for _ in range(50):
        output, state = governor.update(requested, torch.tensor([0.70]))
    assert state.item() == LOW
    assert torch.allclose(
        output,
        torch.tensor(
            [[cfg.low_speed_limit, cfg.low_lateral_limit, cfg.low_yaw_limit]]
        ),
        atol=1.0e-5,
    )


def test_reset_is_per_environment() -> None:
    governor = HallTractionGovernor(3, 0.02, "cpu")
    request = torch.full((3, 3), 0.5)
    for _ in range(80):
        governor.update(request, torch.zeros(3))
    governor.reset(torch.tensor([1]))
    assert governor.state.tolist() == [HIGH, UNKNOWN, HIGH]
    assert torch.all(governor.output_command[1] == 0.0)


def test_small_nonzero_request_gets_one_launch_pulse_but_zero_stays_zero() -> None:
    cfg = HallTractionGovernorCfg(
        unknown_warmup_s=0.04,
        probe_duration_s=0.12,
        probe_speed_limit=0.25,
        launch_accel_rate=2.0,
    )
    governor = HallTractionGovernor(2, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
    maximum = torch.zeros(2)
    for _ in range(12):
        output, _ = governor.update(requested, torch.tensor([0.10, 0.10]))
        maximum = torch.maximum(maximum, output[:, 0])
    assert maximum[0] > 0.05
    assert maximum[0] <= cfg.probe_speed_limit + 1.0e-6
    assert maximum[1] == 0.0


def test_critical_risk_aborts_launch_and_has_final_authority() -> None:
    cfg = HallTractionGovernorCfg(
        unknown_warmup_s=0.02,
        probe_duration_s=0.40,
        launch_accel_rate=2.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.05, 0.0, 0.0]])
    for _ in range(5):
        output, _ = governor.update(requested, torch.tensor([0.10]))
    assert governor.probing.item()
    assert output[0, 0] > 0.0
    for _ in range(20):
        output, state = governor.update(requested, torch.tensor([0.95]))
    assert not governor.probing.item()
    assert state.item() == LOW
    assert torch.allclose(output, torch.zeros_like(output), atol=1.0e-6)


def test_low_state_periodically_reprobes_under_persistent_moderate_risk() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.50,
        probability_high_enter=0.30,
        probability_critical_enter=0.90,
        low_hold_s=0.02,
        unknown_warmup_s=0.02,
        probe_duration_s=0.20,
        low_reprobe_s=0.08,
        probe_speed_limit=0.25,
        launch_accel_rate=10.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.60, 0.0, 0.0]])
    governor.state.fill_(LOW)
    governor.state_probability_reference.fill_(0.70)
    governor.state_reference_initialized.fill_(True)

    # The moderate score enters LOW and stays above the LOW threshold.  The
    # old logic reset its timer forever; the corrected logic performs a
    # bounded active re-probe so recovered traction can become observable.
    for _ in range(4):
        output, state = governor.update(requested, torch.tensor([0.70]))
    assert state.item() == LOW
    assert governor.probing.item()
    assert 0.0 < output[0, 0] <= cfg.probe_speed_limit + 1.0e-6

    # Critical Hall risk has final authority and cancels that probe.
    previous = output[0, 0].item()
    governor.update(requested, torch.tensor([0.95]))
    output, state = governor.update(requested, torch.tensor([0.95]))
    assert state.item() == LOW
    assert not governor.probing.item()
    assert output[0, 0].item() < previous
    for _ in range(30):
        output, state = governor.update(requested, torch.tensor([0.95]))
    assert torch.allclose(output, torch.zeros_like(output), atol=1.0e-6)


def test_probe_relative_clear_resets_stale_low_evidence() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.50,
        probability_high_enter=0.20,
        probability_critical_enter=0.90,
        low_hold_s=0.06,
        high_hold_s=0.40,
        unknown_warmup_s=1.0,
        probe_duration_s=0.08,
        low_reprobe_s=0.02,
        probe_relative_clear_drop=0.10,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    governor.state.fill_(LOW)
    requested = torch.tensor([[0.60, 0.0, 0.0]])

    # Start at p=0.70, then the bounded probe reduces risk to 0.55.  Absolute
    # calibration is still above high_enter, but the causal drop is clear.
    governor.update(requested, torch.tensor([0.70]))
    assert governor.probing.item()
    for _ in range(4):
        _, state = governor.update(requested, torch.tensor([0.55]))
    assert state.item() == HIGH
    assert not governor.probing.item()

    # Old LOW hold evidence must not demote HIGH on the very next sample.
    _, state = governor.update(requested, torch.tensor([0.55]))
    assert state.item() == HIGH


def test_high_state_crawl_uses_pulses_for_sub_deadzone_command() -> None:
    cfg = HallTractionGovernorCfg(
        unknown_warmup_s=0.02,
        probe_duration_s=0.04,
        high_hold_s=0.02,
        crawl_pulse_s=0.10,
        crawl_min_hold_s=0.10,
        launch_accel_rate=4.0,
        linear_decel_rate=4.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.05, 0.0, 0.0]])
    values = []
    for _ in range(100):
        output, state = governor.update(requested, torch.tensor([0.05]))
        values.append(float(output[0, 0]))
    assert state.item() == HIGH
    assert max(values) > cfg.min_detection_command
    assert min(values[-50:]) < 0.01
    assert 0.02 < sum(values[-50:]) / 50.0 < 0.12


def test_low_state_micro_command_uses_mean_preserving_active_sensing() -> None:
    cfg = HallTractionGovernorCfg(
        probability_low_enter=0.40,
        probability_high_enter=0.30,
        probability_critical_enter=0.90,
        low_hold_s=0.02,
        unknown_warmup_s=0.02,
        probe_duration_s=0.04,
        low_reprobe_s=10.0,
        crawl_pulse_s=0.10,
        crawl_min_hold_s=0.10,
        launch_accel_rate=4.0,
        linear_decel_rate=4.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.05, 0.0, 0.0]])
    values = []
    for _ in range(100):
        output, state = governor.update(requested, torch.tensor([0.60]))
        values.append(float(output[0, 0]))
    assert state.item() == LOW
    assert max(values) > cfg.min_detection_command
    assert 0.02 < sum(values[-50:]) / 50.0 < 0.12

    # The same state must never create motion from a true zero request.
    for _ in range(20):
        output, _ = governor.update(torch.zeros_like(requested), torch.tensor([0.60]))
    assert float(output.abs().max()) == 0.0

    # Critical evidence has final authority over active-sensing pulses.
    governor.update(requested, torch.tensor([0.95]))
    output, _ = governor.update(requested, torch.tensor([0.95]))
    assert float(output.abs().max()) == 0.0


def test_single_critical_probability_spike_does_not_abort_probe() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_critical_enter=0.90,
        critical_hold_s=0.04,
        unknown_warmup_s=0.02,
        probe_duration_s=0.20,
        launch_accel_rate=4.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.60, 0.0, 0.0]])
    governor.update(requested, torch.tensor([0.10]))
    assert governor.probing.item()
    output, state = governor.update(requested, torch.tensor([0.95]))
    assert governor.probing.item()
    assert state.item() == UNKNOWN
    assert output[0, 0] > 0.0


def test_state_reference_detects_relative_risk_rise() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.80,
        probability_high_enter=0.20,
        probability_critical_enter=0.99,
        relative_low_rise=0.12,
        relative_high_drop=0.12,
        low_hold_s=0.04,
        unknown_warmup_s=0.02,
        probe_duration_s=0.04,
        low_reprobe_s=10.0,
        reference_settle_s=0.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.60, 0.0, 0.0]])
    for _ in range(4):
        _, state = governor.update(requested, torch.tensor([0.10]))
    assert state.item() == HIGH

    # p=0.30 is below the absolute LOW threshold, but its sustained increase
    # over the per-environment high-traction reference is causal evidence.
    for _ in range(2):
        _, state = governor.update(requested, torch.tensor([0.30]))
    assert state.item() == LOW


def test_opt_in_relative_low_floor_detects_small_but_sharp_risk_change() -> None:
    """The early guard may use a separate floor from the release threshold."""

    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.70,
        probability_high_enter=0.45,
        probability_critical_enter=0.90,
        relative_low_rise=0.04,
        relative_low_min_probability=0.05,
        low_hold_s=0.02,
        unknown_warmup_s=1.0,
        low_reprobe_s=10.0,
        reference_settle_s=0.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    governor.state.fill_(HIGH)
    governor.state_probability_reference.fill_(0.01)
    governor.state_reference_initialized.fill_(True)
    requested = torch.tensor([[0.60, 0.0, 0.0]])

    # The score has not crossed the (deliberately high) release threshold,
    # but its causal jump is sufficient to enter the safe low-speed state.
    _, state = governor.update(requested, torch.tensor([0.08]))
    assert state.item() == LOW


def test_high_reference_settles_loaded_gait_before_relative_braking() -> None:
    """A quiet probe must not make the first loaded gait cycle look unsafe."""

    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.90,
        probability_high_enter=0.70,
        probability_critical_enter=0.99,
        unknown_warmup_s=0.02,
        probe_duration_s=0.04,
        reference_settle_s=0.10,
        reference_settle_alpha=1.0,
        state_reference_ema_alpha=0.01,
        relative_low_rise=0.08,
        relative_low_min_probability=0.05,
        low_hold_s=0.02,
        low_reprobe_s=10.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.60, 0.0, 0.0]])

    # The bounded probe observes a quieter score, then normal loaded gait
    # rises to 0.30.  The short settling window adapts the reference instead
    # of falsely entering LOW.
    for _ in range(4):
        _, state = governor.update(requested, torch.tensor([0.10]))
    assert state.item() == HIGH
    for _ in range(5):
        _, state = governor.update(requested, torch.tensor([0.30]))
    assert state.item() == HIGH
    assert governor.state_probability_reference.item() > 0.28

    # Once the walking reference is established, a genuine sharp causal Hall
    # risk increase retains the usual LOW protection.
    for _ in range(2):
        _, state = governor.update(requested, torch.tensor([0.45]))
    assert state.item() == LOW


def test_prebrake_limits_command_before_low_state_is_confirmed() -> None:
    """A causal Hall-risk jump may brake, without falsely changing state."""

    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.90,
        probability_high_enter=0.20,
        probability_critical_enter=0.99,
        reference_settle_s=0.0,
        relative_low_rise=0.80,
        low_hold_s=0.20,
        unknown_warmup_s=1.0,
        prebrake_probability=0.50,
        prebrake_relative_rise=0.10,
        prebrake_speed_limit=0.05,
        linear_accel_rate=100.0,
        linear_decel_rate=100.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    governor.state.fill_(HIGH)
    governor.state_probability_reference.fill_(0.30)
    governor.state_reference_initialized.fill_(True)
    governor.state_settle_time_s.fill_(1.0)
    requested = torch.tensor([[0.80, 0.0, 0.0]])

    output, state = governor.update(requested, torch.tensor([0.55]))
    assert state.item() == HIGH
    assert governor.prebrake_active.item()
    assert output[0, 0] <= 0.05001


def test_state_reference_detects_relative_recovery_drop() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.80,
        probability_high_enter=0.20,
        probability_critical_enter=0.99,
        relative_low_rise=0.12,
        relative_high_drop=0.12,
        low_hold_s=0.04,
        high_hold_s=0.04,
        unknown_warmup_s=1.0,
        low_reprobe_s=10.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    requested = torch.tensor([[0.60, 0.0, 0.0]])
    governor.state.fill_(LOW)
    governor.state_probability_reference.fill_(0.90)
    governor.state_reference_initialized.fill_(True)
    state = governor.state
    assert state.item() == LOW

    # p=0.70 is above the absolute HIGH threshold.  Its sustained drop from
    # the established low-traction reference still releases the recovered
    # surface after the configured hold time.
    for _ in range(2):
        _, state = governor.update(requested, torch.tensor([0.70]))
    assert state.item() == HIGH


def test_opt_in_absolute_clear_releases_only_after_sustained_safe_risk() -> None:
    """A calibrated prospective-risk head may explicitly opt into release."""

    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_low_enter=0.60,
        probability_high_enter=0.30,
        probability_critical_enter=0.90,
        high_hold_s=0.06,
        low_hold_s=0.02,
        unknown_warmup_s=1.0,
        allow_absolute_high_clear=True,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    governor.state.fill_(LOW)
    governor.state_probability_reference.fill_(0.80)
    governor.state_reference_initialized.fill_(True)
    requested = torch.tensor([[0.60, 0.0, 0.0]])

    # One safe frame is not sufficient; the normal HIGH hold still protects
    # against a gait-phase dip in the Hall-derived risk score.
    governor.update(requested, torch.tensor([0.20]))
    assert governor.state.item() == LOW
    for _ in range(3):
        _, state = governor.update(requested, torch.tensor([0.20]))
    assert state.item() == HIGH


def test_absolute_clear_is_disabled_by_default() -> None:
    cfg = HallTractionGovernorCfg(
        probability_ema_alpha=1.0,
        probability_high_enter=0.30,
        high_hold_s=0.02,
        unknown_warmup_s=1.0,
    )
    governor = HallTractionGovernor(1, 0.02, "cpu", cfg)
    governor.state.fill_(LOW)
    governor.state_probability_reference.fill_(0.20)
    governor.state_reference_initialized.fill_(True)
    requested = torch.tensor([[0.60, 0.0, 0.0]])
    for _ in range(8):
        _, state = governor.update(requested, torch.tensor([0.20]))
    assert state.item() == LOW
