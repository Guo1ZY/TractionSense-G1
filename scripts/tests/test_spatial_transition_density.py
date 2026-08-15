"""CPU/static tests for the transition-dense Medium H-L-H training task."""

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
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"
    / "velocity_foot_env_cfg.py"
)
REGISTRY = ENV_CFG.with_name("__init__.py")


def _load_state_module():
    spec = importlib.util.spec_from_file_location(
        "spatial_friction_state_dense_test", STATE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def test_stratified_reset_samples_all_three_safe_high_start_bands() -> None:
    state = _load_state_module()
    local_x, band = state.stratified_high_patch_reset_x(
        torch.tensor([0.01, 0.24, 0.26, 0.64, 0.66, 0.99]),
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        x_bands=((-1.70, -1.35), (-1.20, -0.85), (-0.75, -0.45)),
        band_probabilities=(0.25, 0.40, 0.35),
        low_boundary_x=0.0,
        minimum_high_margin=0.30,
    )
    assert band.tolist() == [0, 0, 1, 1, 2, 2]
    assert local_x.tolist() == pytest.approx(
        [-1.70, -1.35, -1.20, -0.85, -0.75, -0.45]
    )
    assert torch.all(local_x <= -0.30)


def test_stratified_reset_rejects_a_band_that_can_start_in_low() -> None:
    state = _load_state_module()
    with pytest.raises(ValueError, match="too close to or inside LOW"):
        state.stratified_high_patch_reset_x(
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            x_bands=((-0.20, 0.10),),
            band_probabilities=(1.0,),
            low_boundary_x=0.0,
            minimum_high_margin=0.25,
        )


def test_dense_cfg_changes_only_training_reset_and_exposes_no_shortcut() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    dense = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionPlayEnvCfg",
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionMediumEnvCfg" in dense
    assert "reset_root_state_spatial_stratified" in dense
    assert '"band_probabilities": (0.25, 0.40, 0.35)' in dense
    for band in ("(-1.70, -1.35)", "(-1.20, -0.85)", "(-0.75, -0.45)"):
        assert band in dense
    assert '"minimum_high_margin": 0.30' in dense
    assert "self.observations" not in dense
    assert "root_pos" not in dense
    assert "episode_length_buf" not in dense
    assert "friction_mu" not in dense


def test_dense_fastbase_task_uses_original_medium_play_course() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    task = _block(
        source,
        "TractionMagneticMotionStudent-SpatialFrictionMediumDenseFastBaseCapture",
        'id="Unitree-G1-29dof-Velocity-Foot-TractionTeacher-SpeedLateral"',
    )
    assert "RobotFootTractionMagneticMotionSpatialFrictionMediumDenseEnvCfg" in task
    assert "RobotFootTractionMagneticMotionSpatialFrictionMediumPlayEnvCfg" in task
    assert "FootTractionHallSpatialFastBaseCapturePPORunnerCfg" in task
    assert "MediumDensePlay" not in task


def test_original_evaluation_reset_and_hlh_geometry_remain_unchanged() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    scene = _block(source, "class HallSpatialFrictionSceneCfg", "class HallSpatialMildFrictionSceneCfg")
    assert 'size_x=2.0' in scene
    assert 'size_x=1.0' in scene
    assert 'center_x=-1.0' in scene
    assert 'center_x=0.5' in scene
    assert 'center_x=2.0' in scene

    base = _block(
        source,
        "class RobotFootTractionMagneticMotionSpatialFrictionEnvCfg",
        "class RobotFootTractionMagneticMotionSpatialFrictionMildEnvCfg",
    )
    assert '"x": (-1.70, -1.40)' in base
    assert '"minimum_local_x": 2.60' in source
