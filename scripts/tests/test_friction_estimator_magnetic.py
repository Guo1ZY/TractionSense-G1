from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research_scripts/evaluate_friction_estimator.py"
)
SPEC = importlib.util.spec_from_file_location("friction_estimator_magnetic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_1864_feature_partitions_do_not_leak_or_drop_channels() -> None:
    all_indices = MODULE.feature_indices("all", 1864)
    proprio = MODULE.feature_indices("proprio", 1864)
    foot = MODULE.feature_indices("foot", 1864)
    assert np.array_equal(all_indices, np.arange(1864))
    assert np.array_equal(proprio, np.arange(480))
    assert np.array_equal(foot, np.arange(480, 1864))
    assert np.intersect1d(proprio, foot).size == 0
