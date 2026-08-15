from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/train_shared_magnetic_policy.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("shared_magnetic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_external_and_internal_shapes() -> None:
    model = MODULE.SharedMagneticPolicy().eval()
    observation = torch.zeros(3, MODULE.INPUT_DIM)
    action = model(observation)
    fused, left, right = model.encode(observation)
    assert action.shape == (3, 29)
    assert fused.shape == (3, 548)
    assert left.shape == right.shape == (3, 32)
    assert model.foot_encoder is model.foot_encoder


def test_proxy_contains_independent_health_slots() -> None:
    old = torch.zeros(2, 640)
    old[:, 630:635] = 1.0
    converted = MODULE.proxy_input(old, stochastic=False)
    assert converted.shape == (2, 1864)
    assert torch.all(converted[:, -4:-2] == 1.0)
    assert torch.all(converted[:, -2:] == 0.0)

