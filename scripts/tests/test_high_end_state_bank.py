from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/traction/high_end_state_bank.py"
)
SPEC = importlib.util.spec_from_file_location("high_end_state_bank_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bank_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bank_module
SPEC.loader.exec_module(bank_module)


def _arrays(count: int = 3, seed: int = 510) -> dict[str, np.ndarray]:
    observation = np.zeros((count, 1864), dtype=np.float32)
    command = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
    observation[:, 30:45] = np.tile(command, 5)
    observation[:, 1860:1862] = 1.0
    root_pose = np.zeros((count, 7), dtype=np.float32)
    root_pose[:, 2] = 0.75
    root_pose[:, 3] = 1.0
    return {
        "root_pose_local": root_pose,
        "root_velocity": np.zeros((count, 6), dtype=np.float32),
        "joint_pos": np.zeros((count, 29), dtype=np.float32),
        "joint_vel": np.zeros((count, 29), dtype=np.float32),
        "observation": observation,
        "motion_feedback_initial_yaw": np.zeros(count, dtype=np.float32),
        "straight_heading_reference_xy": np.tile(
            np.asarray([1.0, 0.0], dtype=np.float32), (count, 1)
        ),
        "straight_track_origin_local_xy": np.zeros((count, 2), dtype=np.float32),
        "straight_track_lateral_axis": np.tile(
            np.asarray([0.0, 1.0], dtype=np.float32), (count, 1)
        ),
        "hall_local_deformation": np.zeros((count, 2, 15, 6), dtype=np.float32),
        "hall_loading_history": np.zeros((count, 2, 15, 4, 6), dtype=np.float32),
        "hall_signal_filtered_absolute": np.zeros((count, 2, 15, 3), dtype=np.float32),
        "hall_signal_processed": np.zeros((count, 2, 15, 3), dtype=np.float32),
        "hall_signal_baseline": np.zeros((count, 2, 15, 3), dtype=np.float32),
        "hall_signal_drift": np.zeros((count, 2, 15, 3), dtype=np.float32),
        "hall_policy_history": np.zeros((count, 2, 3, 15, 3), dtype=np.float32),
        "hall_policy_gain": np.ones((count, 2, 15, 3), dtype=np.float32),
        "hall_policy_cross_axis": np.tile(
            np.eye(3, dtype=np.float32), (count, 2, 15, 1, 1)
        ),
        "hall_policy_zero_residual": np.zeros((count, 2, 15, 3), dtype=np.float32),
        "hall_policy_channel_keep": np.ones((count, 2, 15, 1), dtype=np.float32),
        "hall_policy_foot_keep": np.ones((count, 2, 1, 1), dtype=np.float32),
        "hall_policy_delay_steps": np.zeros((count, 2), dtype=np.int64),
        "hall_reported_sample_period": np.full((count, 2), 0.02, dtype=np.float32),
        "source_seed": np.full(count, seed, dtype=np.int64),
        "source_env_id": np.arange(count, dtype=np.int64),
        "source_rollout_step": np.arange(100, 100 + count, dtype=np.int64),
        "time_to_fall_s": np.full(count, 1.5, dtype=np.float32),
        "state_kind": np.ones(count, dtype=np.int64),
    }


def _write(path: Path, *, seed: int = 510, role: str | None = None) -> None:
    metadata = {
        "schema_version": bank_module.SCHEMA_VERSION,
        "dataset_role": role or bank_module.TRAINING_ROLE,
        "source_seeds": [seed],
        "excluded_locked_seeds": [500],
        "actor_observation_dim": 1864,
        "measurement_boundary": "Hall Bx/By/Bz only; no force conversion",
    }
    np.savez_compressed(
        path,
        **_arrays(seed=seed),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_v2_loader_accepts_training_bank_and_preserves_1864_abi(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    _write(path)
    bank = bank_module.load_high_end_state_bank(path, device="cpu")
    assert bank.sample_count == 3
    assert bank.arrays["observation"].shape == (3, 1864)
    histories = bank_module.policy_history_terms(bank.arrays["observation"])
    assert list(histories) == list(bank_module.POLICY_HISTORY_LAYOUT)
    assert histories["last_action"].shape == (3, 5, 29)
    assert histories["foot_magnetic_array"].shape == (3, 15, 90)


def test_loader_rejects_locked_seed_and_old_unversioned_bank(tmp_path: Path) -> None:
    locked = tmp_path / "locked.npz"
    _write(locked, seed=500)
    with pytest.raises(ValueError, match="locked acceptance seed leakage"):
        bank_module.load_high_end_state_bank(locked, device="cpu")

    old = tmp_path / "old.npz"
    np.savez_compressed(
        old,
        root_pose_local=np.zeros((2, 7), dtype=np.float32),
        root_velocity=np.zeros((2, 6), dtype=np.float32),
        joint_pos=np.zeros((2, 29), dtype=np.float32),
        joint_vel=np.zeros((2, 29), dtype=np.float32),
        observation=np.zeros((2, 1864), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing metadata_json"):
        bank_module.load_high_end_state_bank(old, device="cpu")


def test_loader_rejects_command_history_or_single_foot_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    arrays = _arrays()
    arrays["observation"][0, 30] = 0.7
    metadata = {
        "schema_version": bank_module.SCHEMA_VERSION,
        "dataset_role": bank_module.TRAINING_ROLE,
        "source_seeds": [510],
        "excluded_locked_seeds": [500],
    }
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="command history is inconsistent"):
        bank_module.load_high_end_state_bank(path, device="cpu")

    arrays = _arrays()
    arrays["observation"][1, 1861] = 0.0
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="requires two valid Hall feet"):
        bank_module.load_high_end_state_bank(path, device="cpu")


class _FakeCircularBuffer:
    def __init__(self, length: int, batch: int, width: int, pointer: int):
        self.max_length = length
        self.batch_size = batch
        self._device = "cpu"
        self._pointer = pointer
        self._buffer = torch.full((length, batch, width), -99.0)
        self._num_pushes = torch.zeros(batch, dtype=torch.long)

    @property
    def buffer(self) -> torch.Tensor:
        raw = self._buffer.clone()
        raw = torch.roll(raw, shifts=self.max_length - self._pointer - 1, dims=0)
        return raw.transpose(0, 1)


@pytest.mark.parametrize("pointer", [0, 1, 4])
def test_logical_ring_restore_respects_pointer_and_subset(pointer: int) -> None:
    ring = _FakeCircularBuffer(length=5, batch=4, width=3, pointer=pointer)
    before_other = ring.buffer[[0, 2]].clone()
    ids = torch.tensor([1, 3])
    logical = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    bank_module.seed_circular_buffer_logical(ring, ids, logical)
    assert torch.equal(ring.buffer[ids], logical)
    assert torch.equal(ring.buffer[[0, 2]], before_other)
    assert torch.equal(ring._num_pushes, torch.tensor([0, 5, 0, 5]))


def test_recovery_task_alone_uses_specialized_env_and_v2_path() -> None:
    registry = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py"
    ).read_text()
    assert registry.count("HighEndRecoveryRLEnv") == 1
    baseline_block = registry.split("# Training-only recoverable-disturbance", 1)[0]
    assert 'id="Unitree-G1-29dof-Velocity"' in baseline_block
    assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' in baseline_block

    cfg = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
    ).read_text()
    block = cfg.split(
        "class RobotFootTractionMagneticMotionSpatialFrictionCadenceStrideHighEndRecoveryExpertEnvCfg",
        1,
    )[1].split("@configclass", 1)[0]
    assert "model55_seed500_high_end_state_bank.npz" not in block
    assert "high_end_state_bank_train_510_513_v2.npz" in block
    assert '"state_bank_required_role": "training_high_end_state_bank"' in block
    assert "self.events.reset_robot_joints = None" in block


def test_high_end_reset_masks_stale_contact_until_context_finalize() -> None:
    spatial = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/spatial_friction.py"
    ).read_text()
    recovery = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/high_end_recovery_env.py"
    ).read_text()
    assert 'env, "_high_end_recovery_pending_sample_ids", None' in spatial
    assert "reset = reset | (pending_bank_rows[ids] >= 0)" in spatial
    assert "self.spatial_course_stage_buf[ids] = SPATIAL_HIGH_END" in recovery
    assert "self.spatial_low_contact_buf[ids] = False" in recovery
    assert "self.spatial_high_end_contact_buf[ids] = False" in recovery


def test_training_teacher_authority_is_explicitly_full_not_sensor_confidence() -> None:
    source = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/traction/fastbase_capture_residual.py"
    ).read_text()
    block = source.split("class FastBaseHallCaptureHighEndRecoveryResidual", 1)[1].split(
        "class FastBaseHallCaptureHighEndRecoveryRslModel", 1
    )[0]
    assert "torch.ones" in block
    assert "sensor_confidence" not in block

    runner_source = (
        ROOT
        / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py"
    ).read_text()
    runner_block = runner_source.split(
        "class FootTractionHallSpatialCadenceStrideHighEndRecoveryExpertPPORunnerCfg",
        1,
    )[1].split("@configclass", 1)[0]
    assert "max_iterations = 50" in runner_block
    assert "save_interval = 5" in runner_block
    assert "stability_residual_learning_rate = 5.0e-3" in runner_block


def test_builder_creates_training_bank_from_nonlocked_v2_dump(tmp_path: Path) -> None:
    arrays = _arrays(count=8, seed=510)
    arrays["source_rollout_step"] = np.arange(0, 40, 5, dtype=np.int64)
    arrays["time_to_fall_s"] = np.asarray(
        [2.5, 2.0, 1.5, 1.0, -1.0, -1.0, -1.0, -1.0],
        dtype=np.float32,
    )
    arrays["source_episode_fall"] = arrays["time_to_fall_s"] >= 0.0
    dump_metadata = {
        "schema_version": "high_end_recovery_state_dump.v2",
        "dataset_role": "training_high_end_state_dump",
        "source_seeds": [510],
        "excluded_locked_seeds": [500],
    }
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        **arrays,
        metadata_json=np.asarray(json.dumps(dump_metadata)),
    )
    output = tmp_path / "bank.npz"
    script = ROOT / "scripts/traction/build_high_end_recovery_state_bank.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "source/unitree_rl_lab")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--max-states",
            "8",
            "--minimum-gap-steps",
            "1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    bank = bank_module.load_high_end_state_bank(output, device="cpu")
    assert bank.sample_count == 5
    assert set(bank.arrays["state_kind"].tolist()) == {0, 1}
    assert bank.metadata["selection"]["near_selected"] == 4
    assert bank.metadata["selection"]["nominal_selected"] == 1


def test_builder_uses_far_pre_failure_rows_without_claiming_episode_success(
    tmp_path: Path,
) -> None:
    arrays = _arrays(count=8, seed=511)
    arrays["source_rollout_step"] = np.arange(0, 40, 5, dtype=np.int64)
    arrays["time_to_fall_s"] = np.asarray(
        [6.0, 5.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5], dtype=np.float32
    )
    arrays["source_episode_fall"] = np.ones(8, dtype=bool)
    metadata = {
        "schema_version": "high_end_recovery_state_dump.v2",
        "dataset_role": "training_high_end_state_dump",
        "source_seeds": [511],
        "excluded_locked_seeds": [500],
    }
    source = tmp_path / "all_eventually_fall.npz"
    np.savez_compressed(
        source, **arrays, metadata_json=np.asarray(json.dumps(metadata))
    )
    output = tmp_path / "bank.npz"
    script = ROOT / "scripts/traction/build_high_end_recovery_state_bank.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "source/unitree_rl_lab")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--max-states",
            "8",
            "--minimum-gap-steps",
            "1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    bank = bank_module.load_high_end_state_bank(output, device="cpu")
    assert set(bank.arrays["state_kind"].tolist()) == {0, 1}
    selected_ttf = bank.arrays["time_to_fall_s"].numpy()
    nominal_ttf = selected_ttf[bank.arrays["state_kind"].numpy() == 0]
    assert np.all(nominal_ttf >= 4.0)
    assert "not an acceptance success" in str(
        bank.metadata["selection"]["nominal_definition"]
    )


def test_evaluator_v2_dump_contains_complete_context_and_leak_guard() -> None:
    source = (ROOT / "scripts/rsl_rl/eval_spatial_friction_course.py").read_text()
    for field in (
        "hall_loading_history",
        "hall_policy_history",
        "motion_feedback_initial_yaw",
        "straight_heading_reference_xy",
        "straight_track_origin_local_xy",
        "source_episode_fall",
        "time_to_fall_s",
    ):
        assert field in source
    assert '"training_high_end_state_dump"' in source
    assert "is locked for acceptance and cannot be used" in source
    assert 'course_stage = getattr(base_env, "spatial_course_stage_buf", None)' in source
    assert 'float(_runtime_course_geometry()["low_end_x_m"])' in source
    state_keep_block = source.split("state_keep = (", 1)[1].split(")\n", 1)[0]
    assert "course_stage == 2" in state_keep_block
    assert "local_x_before >= 3.0" not in state_keep_block
