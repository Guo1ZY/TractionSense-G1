from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research_scripts/train_magnetic_traction_student.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("magnetic_train", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_magnetic_observation_shape_and_no_mu_leak() -> None:
    old = torch.zeros(4, 640)
    old[:, 510:540] = 2.0
    old[:, 540:570] = 0.4
    old[:, 630:635] = 1.0
    converted = MODULE.magnetic_observation(old, stochastic=False)
    assert converted.shape == (4, 1840)
    assert torch.allclose(converted[:, :480], old[:, :480])
    assert torch.allclose(converted[:, -10:-5], old[:, 630:635])
    assert torch.allclose(converted[:, -5:], old[:, 635:640])
    assert torch.isfinite(converted).all()


def test_magnetic_proxy_responds_to_force_history() -> None:
    unloaded = torch.zeros(2, 640)
    loaded = unloaded.clone()
    loaded[:, 480:510] = 1.0
    loaded[:, 510:540] = 2.0
    unloaded_mag = MODULE.magnetic_observation(unloaded, stochastic=False)
    loaded_mag = MODULE.magnetic_observation(loaded, stochastic=False)
    assert loaded_mag[:, 480:1830].abs().mean() > unloaded_mag[:, 480:1830].abs().mean()
