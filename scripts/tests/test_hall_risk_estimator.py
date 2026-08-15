from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from unitree_rl_lab.traction.hall_risk_estimator import (
    INVARIANT_FEATURE_DIM,
    BaselineInvariantHallTractionRiskEstimator,
    HallTractionRiskEstimator,
    build_hall_risk_estimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (
    AGE_SLICE,
    INPUT_DIM,
    VALID_SLICE,
)


ROOT = Path(__file__).resolve().parents[2]
TRACTION_SCRIPTS = ROOT / "scripts" / "traction"
if str(TRACTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACTION_SCRIPTS))
from train_hall_risk_estimator import load_parts
from train_hall_slip_risk_estimator import prospective_risk_target, trajectory_ids


def test_hall_risk_shape_and_range():
    model = HallTractionRiskEstimator().eval()
    observation = torch.randn(4, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = 0.0
    with torch.inference_mode():
        risk = model(observation)
    assert risk.shape == (4, 1)
    assert torch.isfinite(risk).all()
    assert torch.all((risk >= 0.0) & (risk <= 1.0))


def test_missing_feet_are_maximum_risk_even_with_nonfinite_input():
    model = HallTractionRiskEstimator().eval()
    observation = torch.zeros(3, INPUT_DIM)
    observation[0, 500] = float("nan")
    observation[:, VALID_SLICE] = 0.0
    observation[:, AGE_SLICE] = 1.0
    with torch.inference_mode():
        risk = model(observation)
    torch.testing.assert_close(risk, torch.ones_like(risk), atol=0.0, rtol=0.0)


def test_normal_ble_packet_age_keeps_full_confidence():
    health = torch.tensor(
        [[1.0, 1.0, 0.10, 0.20], [1.0, 1.0, 0.40, 0.40], [1.0, 1.0, 0.80, 0.80]]
    )
    confidence = HallTractionRiskEstimator.physical_confidence(health)
    torch.testing.assert_close(confidence[:2], torch.ones((2, 1)))
    assert 0.0 < confidence[2, 0] < 1.0


def test_baseline_invariant_risk_features_and_checkpoint_factory():
    observation = torch.randn(4, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = 0.0
    raw, health = BaselineInvariantHallTractionRiskEstimator.raw_features(observation)
    assert raw.shape == (4, INVARIANT_FEATURE_DIM)
    assert health.shape == (4, 4)
    model = BaselineInvariantHallTractionRiskEstimator().eval()
    restored = build_hall_risk_estimator(
        {"model_variant": "baseline_invariant", "model": model.state_dict()}
    ).eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(observation), restored(observation))


def test_risk_loader_accepts_mujoco_hall_only_dataset(tmp_path: Path):
    path = tmp_path / "unitree_mujoco_hall_rollout.npz"
    np.savez_compressed(
        path,
        obs=np.zeros((3, INPUT_DIM), dtype=np.float32),
        mu=np.asarray((0.2, 0.8, 0.5), dtype=np.float32),
        sample_weight=np.ones(3, dtype=np.float32),
    )
    data = load_parts([path], crosssim_weight=4.0)
    assert data["obs"].shape == (3, INPUT_DIM)
    np.testing.assert_array_equal(data["sample_weight"], 4.0)


def test_risk_loader_uses_per_sample_switch_transition_not_filename(tmp_path: Path):
    path = tmp_path / "active_probe_rollout.npz"
    count = 4
    np.savez_compressed(
        path,
        obs=np.zeros((count, INPUT_DIM), dtype=np.float32),
        teacher_obs=np.zeros((count, 641), dtype=np.float32),
        mu=np.asarray((0.8, 0.2, 0.2, 0.8), dtype=np.float32),
        cmd_vx=np.full(count, 0.6, dtype=np.float32),
        sample_weight=np.ones(count, dtype=np.float32),
        time_since_switch_s=np.asarray((0.02, 0.50, 1.01, 2.00), dtype=np.float32),
    )
    data = load_parts([path])
    np.testing.assert_array_equal(
        data["is_switch"], np.asarray((1.0, 1.0, 0.0, 0.0), dtype=np.float32)
    )


def test_future_slip_labels_do_not_cross_matrix_rollout_resets():
    # Two cells deliberately reuse env=0 and reset their step counter.  A
    # fall in cell 1 must not make the identically numbered state in cell 0
    # risky merely because a matrix evaluator reuses vector-environment ids.
    env = np.asarray((0, 0, 0, 0), dtype=np.int32)
    step = np.asarray((0, 1, 0, 1), dtype=np.int32)
    rollout = np.asarray((1, 1, 2, 2), dtype=np.int32)
    groups = trajectory_ids(env, step, rollout)
    target = prospective_risk_target(
        np.zeros(4, dtype=np.float32),
        np.asarray((False, False, False, True)),
        groups,
        step,
        horizon_steps=1,
        pre_fall_steps=1,
        slip_threshold=0.25,
        slip_quantile=0.75,
    )
    assert np.all(target[:2] == 0.0)
    assert target[2] == 1.0


def test_future_slip_labels_do_not_cross_managed_reset_with_monotonic_step():
    """A managed Isaac reset does not necessarily reset the global counter."""

    # This is the form emitted by the evaluator after a fall: the same vector
    # environment row keeps an increasing evaluator step, while its saved
    # rollout segment changes.  Without the segment id, a fall after reset
    # would incorrectly label the previous physical episode as risky.
    env = np.asarray((0, 0, 0, 0), dtype=np.int32)
    step = np.asarray((10, 11, 12, 13), dtype=np.int32)
    rollout = np.asarray((7_000_000, 7_000_000, 7_000_001, 7_000_001))
    groups = trajectory_ids(env, step, rollout)
    target = prospective_risk_target(
        np.zeros(4, dtype=np.float32),
        np.asarray((False, False, False, True)),
        groups,
        step,
        horizon_steps=1,
        pre_fall_steps=1,
        slip_threshold=0.25,
        slip_quantile=0.75,
    )
    assert np.all(target[:2] == 0.0)
    assert target[2] == 1.0
