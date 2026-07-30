from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "research_scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "export_estimator_guided_magnetic_teacher.py"
SPEC = importlib.util.spec_from_file_location("estimator_guided_teacher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
from evaluate_friction_estimator import FrictionEstimator


def estimator_checkpoint() -> dict:
    estimator = FrictionEstimator(MODULE.INPUT_DIM)
    return {
        "model": estimator.state_dict(),
        "feature_indices": np.arange(MODULE.INPUT_DIM, dtype=np.int64),
        "mean": np.zeros(MODULE.INPUT_DIM, dtype=np.float32),
        "scale": np.ones(MODULE.INPUT_DIM, dtype=np.float32),
        "input_dim": MODULE.INPUT_DIM,
    }


def test_nominal_hall_proxy_reconstructs_force_history() -> None:
    policy = MODULE.EstimatorGuidedTeacher(
        nn.Linear(MODULE.TEACHER_DIM, 29),
        estimator_checkpoint(),
    ).eval()
    observation = torch.zeros(2, MODULE.INPUT_DIM)
    normal = 1.7
    tangent = 0.35
    profile = torch.from_numpy(MODULE.PROFILE)
    mixing = torch.from_numpy(MODULE.MIXING)
    signal = profile[:, None] * (
        normal * mixing[:, 0] + tangent * mixing[:, 1]
    )
    frame = 5.0 * torch.tanh(signal / 5.0)
    magnetic = frame.reshape(1, 1, 1, MODULE.SENSORS, MODULE.AXES).repeat(
        2, MODULE.HISTORY, MODULE.FEET, 1, 1
    )
    observation[:, MODULE.BASE_DIM : MODULE.BASE_DIM + MODULE.MAGNETIC_DIM] = (
        magnetic.reshape(2, -1)
    )
    observation[:, 1860:1862] = 1.0
    teacher_observation = policy.teacher_observation(observation)
    reconstructed_normal = teacher_observation[:, 510:540]
    reconstructed_tangent = teacher_observation[:, 540:570]
    assert teacher_observation.shape == (2, MODULE.TEACHER_DIM)
    assert torch.allclose(
        reconstructed_normal, torch.full_like(reconstructed_normal, normal), atol=1e-4
    )
    assert torch.allclose(
        reconstructed_tangent, torch.full_like(reconstructed_tangent, tangent), atol=1e-4
    )
    assert policy(observation).shape == (2, 29)


def test_teacher_observation_mirror_is_an_involution() -> None:
    observation = torch.randn(
        4, MODULE.TEACHER_DIM, generator=torch.Generator().manual_seed(8267)
    )
    restored = MODULE.mirror_teacher_observation(
        MODULE.mirror_teacher_observation(observation)
    )
    assert torch.allclose(restored, observation)


def test_teacher_symmetry_ensemble_preserves_output_shape() -> None:
    teacher = nn.Sequential(nn.Linear(MODULE.TEACHER_DIM, 128), nn.ELU(), nn.Linear(128, 29))
    ensemble = MODULE.TeacherLateralSymmetryEnsemble(
        teacher, lateral_weight=0.75, arm_weight=0.25
    )
    assert ensemble(torch.randn(3, MODULE.TEACHER_DIM)).shape == (3, 29)
