from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research_scripts/fine_tune_shared_magnetic_dagger.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("magnetic_speed_lateral", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_conditioned_teacher_command_only_changes_selected_rows() -> None:
    observation = np.zeros((3, 641), dtype=np.float32)
    observation[:, MODULE.COMMAND_VX_INDICES] = np.asarray(
        [[0.8], [0.8], [0.5]], dtype=np.float32
    )
    observation[:, -1] = np.asarray([0.8, 0.5, 1.2], dtype=np.float32)
    conditioned, count = MODULE.conditioned_teacher_observations(
        observation,
        mu=np.asarray([0.8, 0.5, 1.2], dtype=np.float32),
        command_vx=np.asarray([0.8, 0.8, 0.5], dtype=np.float32),
        command_scale=1.1,
        mu_threshold=0.75,
        command_threshold=0.70,
    )
    assert count == 1
    assert np.allclose(conditioned[0, MODULE.COMMAND_VX_INDICES], 0.88)
    assert np.allclose(conditioned[1:], observation[1:])
    assert np.allclose(conditioned[:, -1], observation[:, -1])


def test_full_observation_mirror_is_an_involution() -> None:
    generator = torch.Generator().manual_seed(8291)
    observation = torch.randn(
        4, MODULE.INPUT_DIM, generator=generator, dtype=torch.float32
    )
    restored = MODULE.mirror_observation(MODULE.mirror_observation(observation))
    assert torch.allclose(restored, observation)


def test_motion_feedback_mirror_swaps_valid_and_negates_lateral_state() -> None:
    observation = torch.zeros(1, MODULE.INPUT_DIM, dtype=torch.float32)
    observation[0, -4:] = torch.tensor([0.2, 0.9, -0.35, 0.12])
    mirrored = MODULE.mirror_observation(observation, motion_feedback=True)
    assert torch.allclose(
        mirrored[0, -4:],
        torch.tensor([0.9, 0.2, 0.35, -0.12]),
    )
    restored = MODULE.mirror_observation(mirrored, motion_feedback=True)
    assert torch.allclose(restored, observation)


def test_teacher_mix_is_small_on_ice_and_large_on_high_grip() -> None:
    weights = MODULE.teacher_mix_weights(
        mu=np.asarray([0.08, 1.20, 1.20], dtype=np.float32),
        command_vx=np.asarray([1.00, 1.00, 0.30], dtype=np.float32),
        low=0.0,
        high=0.30,
        mu_threshold=0.75,
        command_threshold=0.70,
    )
    assert weights[0] < 0.002
    assert weights[1] > 0.290
    assert weights[2] < 0.003


def test_foot_encoder_only_freezes_every_other_parameter() -> None:
    model = MODULE.SharedMagneticPolicy()
    MODULE.configure_trainable_parameters(
        model,
        freeze_foot_encoder=False,
        actor_head_only=False,
        foot_encoder_only=True,
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("foot_encoder.") for name in trainable)
    assert not any(
        parameter.requires_grad for parameter in model.actor.parameters()
    )


def test_dagger_loader_reads_failure_priority(tmp_path: Path) -> None:
    path = tmp_path / "dagger.npz"
    np.savez(
        path,
        obs=np.zeros((3, MODULE.INPUT_DIM), dtype=np.float32),
        teacher_obs=np.zeros((3, MODULE.OLD_INPUT_DIM + 1), dtype=np.float32),
        mu=np.asarray([0.08, 0.8, 1.2], dtype=np.float32),
        cmd_vx=np.asarray([1.0, 1.0, 0.5], dtype=np.float32),
        sample_weight=np.asarray([1.0, 8.0, 4.0], dtype=np.float32),
    )
    _, _, _, _, priority = MODULE.load(path)
    assert np.array_equal(priority, np.asarray([1.0, 8.0, 4.0]))
