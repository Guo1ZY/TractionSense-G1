from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction import (  # noqa: E402
    TactileDomainRandomizationCfg,
    TactileObservationModel,
    TemporalHistoryBuffer,
    TractionDiagnosticsCfg,
    TractionDiagnosticsState,
)


def _deterministic_cfg(**overrides) -> TactileDomainRandomizationCfg:
    base = TactileDomainRandomizationCfg(
        scale_range=(1.0, 1.0),
        fixed_bias_n=(0.0, 0.0, 0.0),
        episode_bias_n=(0.0, 0.0),
        drift_rate_n_sqrt_s=(0.0, 0.0),
        coupling_offdiag_range=(0.0, 0.0),
        rotation_deg_range=(0.0, 0.0),
        delay_steps_range=(0, 0),
        lowpass_tau_s_range=(0.0, 0.0),
        noise_floor_n_range=(0.0, 0.0),
        noise_load_fraction_range=(0.0, 0.0),
        saturation_n_range=(1.0e6, 1.0e6),
        sample_dropout_probability_range=(0.0, 0.0),
        burst_start_probability_range=(0.0, 0.0),
        burst_length_steps_range=(2, 2),
        spike_probability_range=(0.0, 0.0),
        spike_amplitude_n_range=(0.0, 0.0),
        hysteresis_fraction_range=(0.0, 0.0),
    )
    return replace(base, **overrides)


def test_history_is_time_major_and_reset_has_no_episode_leakage() -> None:
    history = TemporalHistoryBuffer(2, 3, 2)
    history.append(torch.tensor([[1.0, 2.0], [10.0, 20.0]]))
    history.append(torch.tensor([[3.0, 4.0], [30.0, 40.0]]))
    assert torch.equal(
        history.flatten()[0],
        torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    history.reset(torch.tensor([0]), initial=torch.tensor([[7.0, 8.0]]))
    assert torch.equal(
        history.flatten()[0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 8.0]),
    )
    assert torch.equal(
        history.flatten()[1],
        torch.tensor([0.0, 0.0, 10.0, 20.0, 30.0, 40.0]),
    )


def test_ideal_tactile_stage_is_exact_and_signed() -> None:
    model = TactileObservationModel(
        2,
        cfg=_deterministic_cfg(),
        seed=11,
        curriculum_stage=0,
    )
    force = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, 6.0],
            [-10.0, 20.0, 30.0, 40.0, -50.0, 60.0],
        ]
    )
    observation = model(force)
    assert torch.equal(observation.force_xyz_n, force)
    assert observation.valid.all()
    assert torch.count_nonzero(observation.sample_age_s) == 0


def test_tactile_delay_is_causal_and_reset_clears_delay_line() -> None:
    model = TactileObservationModel(
        1,
        cfg=_deterministic_cfg(delay_steps_range=(2, 2)),
        seed=12,
        curriculum_stage=2,
    )
    first = torch.full((1, 6), 10.0)
    second = torch.full((1, 6), 20.0)
    third = torch.full((1, 6), 30.0)
    assert torch.count_nonzero(model(first).force_xyz_n) == 0
    assert torch.count_nonzero(model(second).force_xyz_n) == 0
    assert torch.equal(model(third).force_xyz_n, first)
    model.reset()
    assert torch.count_nonzero(model(third).force_xyz_n) == 0


def test_tactile_dropout_holds_value_marks_invalid_and_increments_age() -> None:
    model = TactileObservationModel(
        1,
        cfg=_deterministic_cfg(
            sample_dropout_probability_range=(1.0, 1.0)
        ),
        seed=13,
        curriculum_stage=4,
    )
    observation1 = model(torch.full((1, 6), 10.0))
    observation2 = model(torch.full((1, 6), 20.0))
    assert not observation1.valid.any()
    assert not observation2.valid.any()
    assert torch.count_nonzero(observation2.force_xyz_n) == 0
    assert torch.allclose(observation1.sample_age_s, torch.full((1, 2), 0.02))
    assert torch.allclose(observation2.sample_age_s, torch.full((1, 2), 0.04))
    model.reset()
    assert torch.count_nonzero(model.last_output) == 0
    assert torch.count_nonzero(model.age) == 0


def test_full_tactile_model_is_batched_finite_and_reproducible() -> None:
    cfg = TactileDomainRandomizationCfg()
    first = TactileObservationModel(64, cfg=cfg, seed=20260731)
    second = TactileObservationModel(64, cfg=cfg, seed=20260731)
    generator = torch.Generator().manual_seed(7)
    force = torch.randn((64, 6), generator=generator) * 200.0
    for _ in range(20):
        out_first = first(force)
        out_second = second(force)
        assert torch.equal(out_first.force_xyz_n, out_second.force_xyz_n)
        assert torch.equal(out_first.valid, out_second.valid)
        assert torch.equal(out_first.sample_age_s, out_second.sample_age_s)
        assert torch.isfinite(out_first.force_xyz_n).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tactile_model_runs_vectorized_on_cuda() -> None:
    model = TactileObservationModel(
        4096,
        cfg=TactileDomainRandomizationCfg(),
        device="cuda:0",
        seed=20260731,
    )
    force = torch.randn((4096, 6), device="cuda:0") * 200.0
    for _ in range(10):
        observation = model(force)
    assert observation.force_xyz_n.device.type == "cuda"
    assert torch.isfinite(observation.force_xyz_n).all()


def test_contact_and_slip_hysteresis_with_minimum_duration() -> None:
    cfg = TractionDiagnosticsCfg(
        contact_force_on=10.0,
        contact_force_off=5.0,
        slip_speed_on=0.10,
        slip_speed_off=0.05,
        minimum_slip_duration=0.04,
        dt=0.02,
    )
    state = TractionDiagnosticsState(1, cfg=cfg)
    force = torch.tensor([[[3.0, 4.0, 20.0], [0.0, 0.0, 0.0]]])
    moving = torch.tensor([[[0.11, 0.0], [0.0, 0.0]]])
    first = state.update(force, moving, velocity_is_proxy=True)
    assert first.contact[0, 0] and not first.contact[0, 1]
    assert not first.slip_label.any()
    second = state.update(force, moving, velocity_is_proxy=True)
    assert second.slip_label[0, 0]
    assert second.velocity_is_proxy
    assert second.force_normal[0, 0] == 20.0
    assert second.force_tangent[0, 0] == 5.0
    assert second.friction_utilization[0, 0] > 0.249
    assert torch.allclose(second.support_load_ratio, torch.tensor([[1.0, 0.0]]))

    # Hysteresis keeps slip between off and on, then exits below off.
    middle = torch.tensor([[[0.07, 0.0], [0.0, 0.0]]])
    assert state.update(force, middle, velocity_is_proxy=True).slip_label[0, 0]
    slow = torch.tensor([[[0.04, 0.0], [0.0, 0.0]]])
    assert not state.update(force, slow, velocity_is_proxy=True).slip_label[0, 0]
    # Contact remains at 7 N, then drops below the off threshold.
    force[..., 2] = torch.tensor([[7.0, 0.0]])
    assert state.update(force, slow, velocity_is_proxy=True).contact[0, 0]
    force[..., 2] = torch.tensor([[4.0, 0.0]])
    assert not state.update(force, slow, velocity_is_proxy=True).contact[0, 0]
