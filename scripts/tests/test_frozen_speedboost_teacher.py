from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    INPUT_DIM,
    KNOWN_SPEEDBOOST112_SHA256,
    OUTPUT_DIM,
    TRAILING_SLICE,
    FrozenSpeedBoostTeacher,
    adapt_teacher_observation,
    load_frozen_speedboost_teacher,
    save_frozen_speedboost_teacher,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO_ROOT / "scripts" / "traction" / "convert_speedboost_teacher.py"
CANONICAL_ONNX = REPO_ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_dynamic.onnx"
FIXED_ONNX = (
    REPO_ROOT
    / "artifacts"
    / "final_hall_policy_20260806"
    / "remote"
    / "traction_magnetic_speedboost112_guard"
    / "exported"
    / "policy.onnx"
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_speedboost_teacher_for_test", CONVERTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_is_frozen_and_has_exact_external_schema() -> None:
    model = FrozenSpeedBoostTeacher()
    assert not model.training
    assert len(model.state_dict()) == 80
    assert all(not parameter.requires_grad for parameter in model.parameters())

    model.train(True)
    assert not model.training
    observation = torch.zeros(3, INPUT_DIM)
    with torch.no_grad():
        action = model(observation)
    assert action.shape == (3, OUTPUT_DIM)
    assert torch.isfinite(action).all()


def test_motion_feedback_adapter_only_changes_copy_of_trailing_channels() -> None:
    observation = torch.randn(4, INPUT_DIM)
    original = observation.clone()
    sensor_age = torch.tensor([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.9, 1.2]])
    adapted = adapt_teacher_observation(
        observation,
        policy_trailing_feature_mode="motion_feedback",
        sensor_age_lr=sensor_age,
    )
    assert adapted.data_ptr() != observation.data_ptr()
    assert torch.equal(observation, original)
    assert torch.equal(adapted[:, : TRAILING_SLICE.start], original[:, : TRAILING_SLICE.start])
    assert torch.equal(adapted[:, TRAILING_SLICE], sensor_age.clamp(0.0, 1.0))

    with pytest.raises(ValueError, match="require training-only sensor_age_lr"):
        adapt_teacher_observation(observation, policy_trailing_feature_mode="motion_feedback")


def test_explicit_assume_fresh_adapter_path() -> None:
    observation = torch.randn(2, INPUT_DIM)
    original = observation.clone()
    adapted = adapt_teacher_observation(
        observation,
        policy_trailing_feature_mode="motion_feedback",
        assume_fresh_if_motion_feedback=True,
    )
    assert torch.equal(adapted[:, TRAILING_SLICE], torch.zeros(2, 2))
    assert torch.equal(observation, original)


def test_checkpoint_roundtrip_preserves_state_and_rejects_hash_drift(tmp_path: Path) -> None:
    torch.manual_seed(112)
    model = FrozenSpeedBoostTeacher()
    path = tmp_path / "teacher.pt"
    save_frozen_speedboost_teacher(
        path,
        model,
        source_onnx_sha256=KNOWN_SPEEDBOOST112_SHA256,
        source_graph={"node_count": 417, "initializer_count": 82},
        parity={"passed": True, "max_abs_error": 1.0e-6},
        provenance={"test": True},
    )
    loaded = load_frozen_speedboost_teacher(path)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_frozen_speedboost_teacher(path, expected_source_sha256="0" * 64)


@pytest.mark.skipif(not CANONICAL_ONNX.is_file(), reason="canonical speedboost112 ONNX artifact is not present")
def test_canonical_417_node_conversion_and_onnxruntime_parity() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    converter = _load_converter()
    source_sha256 = converter.sha256_file(CANONICAL_ONNX)
    assert source_sha256 == KNOWN_SPEEDBOOST112_SHA256

    model_proto = converter.onnx.load(CANONICAL_ONNX, load_external_data=True)
    signature = converter.validate_canonical_graph(model_proto, source_sha256)
    assert signature["node_count"] == 417
    assert signature["initializer_count"] == 82
    teacher, audit = converter.recover_model(model_proto)
    assert audit == {
        "state_tensor_count": 80,
        "initializer_count": 82,
        "direct_or_mapped_state_initializers": 80,
        "derived_initializers": ["onnx::Where_740", "onnx::Mul_741"],
        "all_initializers_accounted": True,
    }
    assert torch.equal(teacher.command_mask.nonzero().flatten(), torch.tensor([30, 33, 36, 39, 42]))
    assert teacher.config.stable_uses_boosted_command is True
    assert teacher.config.boost_factor == pytest.approx(1.12, abs=1.0e-7)
    assert teacher.config.traction_center == pytest.approx(0.65, abs=1.0e-7)
    assert torch.allclose(teacher.stable_weight[[4, 6, 8, 12, 13, 14, 15, 22]], torch.full((8,), 0.75))
    assert torch.allclose(teacher.stable_weight[[5, 17, 18, 19, 23, 26, 27, 28]], torch.ones(8))

    if FIXED_ONNX.is_file():
        fixed_report = converter.compare_initializer_payloads(model_proto, FIXED_ONNX)
        assert fixed_report["bitwise_equal"] is True
        assert fixed_report["initializer_count"] == 82

    parity = converter.verify_runtime_parity(
        teacher,
        CANONICAL_ONNX,
        batch_sizes=(1, 4),
        seed=112,
        tolerance=2.0e-5,
    )
    assert parity["passed"] is True
    assert parity["max_abs_error"] < 2.0e-5
