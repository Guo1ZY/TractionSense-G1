from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.evaluation import evaluate_npz  # noqa: E402
from unitree_rl_lab.traction.experiments import (  # noqa: E402
    EXPERIMENTS,
    experiment_by_id,
)


def _trajectory() -> dict[str, np.ndarray]:
    # Stored as Isaac collection order: all environments at t0, then all at t1.
    environment_id = np.tile(np.arange(2), 4)[:, None]
    timestamp = np.repeat(np.arange(4) * 0.02, 2)[:, None]
    slip_label = np.zeros((8, 2), dtype=np.float32)
    slip_label[[2, 4], 0] = 1.0  # env 0: one two-step event.
    slip_probability = np.full((8, 2), 0.1, dtype=np.float32)
    slip_probability[4, 0] = 0.9  # detected one sample after onset.
    done = np.zeros((8, 1), dtype=bool)
    terminated = np.zeros_like(done)
    truncated = np.zeros_like(done)
    terminated[6] = True
    truncated[7] = True
    command = np.tile(np.asarray([[1.0, 0.0, 0.2]], dtype=np.float32), (8, 1))
    base_velocity = np.tile(
        np.asarray([[0.8, 0.0, 0.0]], dtype=np.float32),
        (8, 1),
    )
    traction_target = 1.0 - slip_label.max(axis=1, keepdims=True)
    return {
        "timestamp_s": timestamp,
        "environment_id": environment_id,
        "command": command,
        "base_velocity": base_velocity,
        "base_yaw_rate": np.full((8, 1), 0.1, dtype=np.float32),
        "projected_gravity": np.zeros((8, 3), dtype=np.float32),
        "rollout_action": np.arange(8 * 29, dtype=np.float32).reshape(8, 29),
        "episode_done": terminated | truncated,
        "terminated": terminated,
        "truncated": truncated,
        "slip_speed_proxy": slip_label * 0.2,
        "slip_label": slip_label,
        "predicted_slip_probability": slip_probability,
        "traction_target": traction_target,
        "predicted_traction_score": traction_target.copy(),
        "sensor_valid": np.ones((8, 2), dtype=np.float32),
        "predicted_sensor_confidence": np.ones((8, 1), dtype=np.float32),
        "teacher_latent": np.zeros((8, 16), dtype=np.float32),
        "student_latent": np.ones((8, 16), dtype=np.float32),
        "joint_velocity": np.full((8, 29), 2.0, dtype=np.float32),
        "joint_torque": np.full((8, 29), 3.0, dtype=np.float32),
    }


def test_required_experiment_and_ablation_registry_is_complete() -> None:
    assert len(EXPERIMENTS) == 25
    assert len({entry.identifier for entry in EXPERIMENTS}) == 25
    assert experiment_by_id("full_method").use_governor
    assert experiment_by_id("proprio_baseline").force_mode == "proprio_only"
    assert experiment_by_id("gru_vs_tcn").matrix_values == ("gru", "tcn")


def test_evaluation_computes_events_detection_energy_and_episode_metrics(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.npz"
    np.savez_compressed(trajectory, **_trajectory())
    metrics, _ = evaluate_npz(trajectory)
    assert metrics["samples"] == 8
    assert metrics["environments"] == 2
    assert metrics["completed_episodes"] == 2
    assert metrics["fall_rate"] == pytest.approx(0.5)
    assert metrics["episode_length_mean_s"] == pytest.approx(0.08)
    assert metrics["velocity_tracking_error_mean_m_s"] == pytest.approx(0.2)
    assert metrics["yaw_tracking_error_mean_rad_s"] == pytest.approx(0.1)
    assert metrics["slip_event_count"] == 1
    assert metrics["slip_duration_s"] == pytest.approx(0.04)
    assert metrics["slip_distance_proxy_m"] == pytest.approx(0.008)
    assert metrics["slip_detection_delay_s"] == pytest.approx(0.02)
    # One positive is deliberately tied with negatives before delayed detection.
    assert metrics["slip_auc"] == pytest.approx(0.75)
    assert metrics["mean_absolute_mechanical_power_w"] == pytest.approx(174.0)
    assert metrics["mechanical_energy_per_environment_j"] == pytest.approx(13.92)
    assert metrics["latent_mse"] == pytest.approx(1.0)
    assert metrics["sensor_confidence_brier"] == pytest.approx(0.0)


def test_evaluation_rejects_noncanonical_archive(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, command=np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_npz(path)
