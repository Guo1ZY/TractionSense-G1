"""CPU-only behavioral tests for the spatial H--L--H latch and success gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
    / "spatial_friction_state.py"
)


def _load_state_module():
    spec = importlib.util.spec_from_file_location("spatial_friction_state_pure", STATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bool(*values: bool):
    return torch.tensor(values, dtype=torch.bool)


def test_low_contact_latches_through_flight_until_real_high_end_contact() -> None:
    state = _load_state_module()
    stage = torch.tensor(
        [state.SPATIAL_HIGH_START, state.SPATIAL_HIGH_START], dtype=torch.long
    )

    # Contacting HighEnd before ever touching Low is not a valid traversal.
    stage = state.advance_spatial_course_stage(
        stage,
        _bool(False, False),
        _bool(True, False),
        _bool(False, False),
    )
    assert stage.tolist() == [state.SPATIAL_HIGH_START, state.SPATIAL_HIGH_START]

    stage = state.advance_spatial_course_stage(
        stage,
        _bool(True, True),
        _bool(False, False),
        _bool(False, False),
    )
    assert stage.tolist() == [state.SPATIAL_LOW, state.SPATIAL_LOW]

    # No filtered contact represents flight, not a return to high friction.
    stage = state.advance_spatial_course_stage(
        stage,
        _bool(False, False),
        _bool(False, False),
        _bool(False, False),
    )
    assert stage.tolist() == [state.SPATIAL_LOW, state.SPATIAL_LOW]

    # Split Low/HighEnd contact is conservative; Low wins in the first env.
    stage = state.advance_spatial_course_stage(
        stage,
        _bool(True, False),
        _bool(True, True),
        _bool(False, False),
    )
    assert stage.tolist() == [state.SPATIAL_LOW, state.SPATIAL_HIGH_END]


def test_reset_always_returns_to_high_start() -> None:
    state = _load_state_module()
    stage = torch.tensor(
        [state.SPATIAL_LOW, state.SPATIAL_HIGH_END], dtype=torch.long
    )
    stage = state.advance_spatial_course_stage(
        stage,
        _bool(True, False),
        _bool(True, True),
        _bool(True, True),
    )
    assert stage.tolist() == [state.SPATIAL_HIGH_START, state.SPATIAL_HIGH_START]


def test_success_requires_prior_low_current_high_end_contact_and_progress() -> None:
    state = _load_state_module()
    stage = torch.tensor(
        [
            state.SPATIAL_HIGH_END,
            state.SPATIAL_HIGH_END,
            state.SPATIAL_LOW,
            state.SPATIAL_HIGH_END,
        ],
        dtype=torch.long,
    )
    success = state.spatial_course_success_mask(
        stage,
        _bool(True, False, True, True),
        torch.tensor([2.60, 2.80, 2.90, 2.59]),
        minimum_local_x=2.60,
    )
    assert success.tolist() == [True, False, False, False]


def test_state_machine_rejects_mismatched_shapes_and_non_boolean_evidence() -> None:
    state = _load_state_module()
    stage = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        state.advance_spatial_course_stage(
            stage,
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    with pytest.raises(TypeError):
        state.advance_spatial_course_stage(
            stage,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )


def test_low_capture_timing_latches_first_filtered_contact_and_freezes_after_low() -> None:
    state = _load_state_module()
    high = torch.tensor([state.SPATIAL_HIGH_START], dtype=torch.long)
    low = torch.tensor([state.SPATIAL_LOW], dtype=torch.long)
    high_end = torch.tensor([state.SPATIAL_HIGH_END], dtype=torch.long)
    entry_step = torch.tensor([-1], dtype=torch.long)
    entry_speed = torch.zeros(1)
    elapsed = torch.zeros(1)

    entry_step, entry_speed, elapsed = state.update_low_capture_timing(
        high,
        low,
        torch.tensor([100], dtype=torch.long),
        torch.tensor([0.80]),
        entry_step,
        entry_speed,
        elapsed,
        _bool(False),
        control_dt=0.02,
    )
    assert entry_step.tolist() == [100]
    assert entry_speed.tolist() == pytest.approx([0.80])
    assert elapsed.tolist() == pytest.approx([0.0])

    entry_step, entry_speed, elapsed = state.update_low_capture_timing(
        low,
        low,
        torch.tensor([140], dtype=torch.long),
        torch.tensor([0.46]),
        entry_step,
        entry_speed,
        elapsed,
        _bool(False),
        control_dt=0.02,
    )
    assert entry_step.tolist() == [100]
    assert entry_speed.tolist() == pytest.approx([0.80])
    assert elapsed.tolist() == pytest.approx([0.80])

    entry_step, entry_speed, elapsed = state.update_low_capture_timing(
        low,
        high_end,
        torch.tensor([145], dtype=torch.long),
        torch.tensor([0.44]),
        entry_step,
        entry_speed,
        elapsed,
        _bool(False),
        control_dt=0.02,
    )
    assert elapsed.tolist() == pytest.approx([0.90])
    _, _, frozen = state.update_low_capture_timing(
        high_end,
        high_end,
        torch.tensor([170], dtype=torch.long),
        torch.tensor([0.75]),
        entry_step,
        entry_speed,
        elapsed,
        _bool(False),
        control_dt=0.02,
    )
    assert frozen.tolist() == pytest.approx([0.90])


def test_capture_requires_consecutive_stable_samples_and_latches_success() -> None:
    state = _load_state_module()
    low = torch.tensor([state.SPATIAL_LOW, state.SPATIAL_LOW], dtype=torch.long)
    count = torch.zeros(2, dtype=torch.long)
    success = _bool(False, False)

    # The second environment contains invalid velocity and must never count.
    for sample in range(6):
        count, success, new, timely = state.update_low_capture_stability(
            low,
            torch.tensor([0.48, float("nan")]),
            count,
            success,
            torch.tensor([0.70 + 0.02 * sample, 0.50]),
            _bool(False, False),
            target_speed=0.45,
            speed_tolerance=0.05,
            stable_steps=6,
            deadline_s=1.0,
        )
    assert count.tolist() == [6, 0]
    assert success.tolist() == [True, False]
    assert new.tolist() == [True, False]
    assert timely.tolist() == [True, False]

    # Completion is a one-step pulse; the success latch survives HighEnd.
    high_end = torch.tensor(
        [state.SPATIAL_HIGH_END, state.SPATIAL_HIGH_END], dtype=torch.long
    )
    count, success, new, timely = state.update_low_capture_stability(
        high_end,
        torch.tensor([0.80, 0.80]),
        count,
        success,
        torch.tensor([1.10, 1.10]),
        _bool(False, False),
        target_speed=0.45,
        speed_tolerance=0.05,
        stable_steps=6,
        deadline_s=1.0,
    )
    assert count.tolist() == [0, 0]
    assert success.tolist() == [True, False]
    assert not new.any()
    assert not timely.any()


def test_capture_speed_envelope_reaches_target_at_deadline_and_stays_finite() -> None:
    state = _load_state_module()
    envelope = state.capture_speed_envelope(
        torch.tensor([0.0, 0.45, 0.90, 1.20, float("nan")]),
        torch.tensor([0.80, 0.80, 0.80, 0.80, float("inf")]),
        target_speed=0.24,
        deadline_s=0.90,
    )
    assert envelope.tolist()[:4] == pytest.approx([0.80, 0.52, 0.24, 0.24])
    assert torch.isfinite(envelope).all()
    assert envelope[-1].item() == pytest.approx(0.24)


def test_capture_reset_clears_timing_and_stability() -> None:
    state = _load_state_module()
    low = torch.tensor([state.SPATIAL_LOW], dtype=torch.long)
    high = torch.tensor([state.SPATIAL_HIGH_START], dtype=torch.long)
    entry_step, entry_speed, elapsed = state.update_low_capture_timing(
        low,
        high,
        torch.tensor([42], dtype=torch.long),
        torch.tensor([0.5]),
        torch.tensor([10], dtype=torch.long),
        torch.tensor([0.8]),
        torch.tensor([0.64]),
        _bool(True),
        control_dt=0.02,
    )
    assert entry_step.tolist() == [-1]
    assert entry_speed.tolist() == [0.0]
    assert elapsed.tolist() == [0.0]
    count, success, new, timely = state.update_low_capture_stability(
        high,
        torch.tensor([0.2]),
        torch.tensor([5], dtype=torch.long),
        _bool(True),
        elapsed,
        _bool(True),
        target_speed=0.24,
        speed_tolerance=0.05,
        stable_steps=6,
        deadline_s=0.9,
    )
    assert count.tolist() == [0]
    assert not success.any() and not new.any() and not timely.any()


def test_transition_retention_latch_freezes_heading_at_low_entry() -> None:
    state = _load_state_module()
    high_start = torch.tensor([state.SPATIAL_HIGH_START], dtype=torch.long)
    low = torch.tensor([state.SPATIAL_LOW], dtype=torch.long)
    high_end = torch.tensor([state.SPATIAL_HIGH_END], dtype=torch.long)
    entry, elapsed = state.update_transition_retention_latch(
        high_start,
        low,
        torch.tensor([0.12]),
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
        _bool(False),
        control_dt=0.02,
    )
    assert entry.tolist() == pytest.approx([0.12])
    assert elapsed.tolist() == [0.0]
    # Retained through HIGH_END and not relatched by later LOW contact.
    entry, elapsed = state.update_transition_retention_latch(
        low,
        high_end,
        torch.tensor([-0.35]),
        entry,
        torch.tensor([0.0]),
        _bool(False),
        control_dt=0.02,
    )
    assert entry.tolist() == pytest.approx([0.12])
    assert elapsed.tolist() == pytest.approx([0.02])
    # Reset clears both buffers.
    entry, elapsed = state.update_transition_retention_latch(
        high_end,
        high_start,
        torch.tensor([0.0]),
        entry,
        elapsed,
        _bool(True),
        control_dt=0.02,
    )
    assert torch.isnan(entry).all()
    assert elapsed.tolist() == [0.0]


def test_transition_stage_heading_weight_peaks_in_low_and_decays() -> None:
    state = _load_state_module()
    stage = torch.tensor(
        [
            state.SPATIAL_HIGH_START,
            state.SPATIAL_LOW,
            state.SPATIAL_HIGH_END,
            state.SPATIAL_HIGH_END,
        ],
        dtype=torch.long,
    )
    weight = state.transition_stage_heading_weight(
        stage,
        torch.tensor([0.0, 0.0, 0.0, 9.0]),
        low_weight=1.0,
        high_start_weight=0.1,
        high_end_peak_weight=1.0,
        high_end_decay_s=3.0,
    )
    assert weight.tolist()[0] == pytest.approx(0.1)
    assert weight.tolist()[1] == pytest.approx(1.0)
    assert weight.tolist()[2] == pytest.approx(1.0)
    assert weight.tolist()[3] == pytest.approx(
        0.1 + 0.9 * torch.exp(torch.tensor(-3.0)).item()
    )
