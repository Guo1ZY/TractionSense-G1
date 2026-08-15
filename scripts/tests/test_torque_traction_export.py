"""TorchScript/ONNX contract and simulator-boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from unitree_rl_lab.traction_torque.networks import TorqueTractionStudentPolicy

MUJOCO_DIR = Path(__file__).resolve().parents[3] / "unitree_mujoco/simulate_python"


class _ExportHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.student = TorqueTractionStudentPolicy().eval()

    def forward(self, history: torch.Tensor):
        result = self.student(history)
        return result[:7]


def test_exported_torchscript_and_onnx_batch_contract(tmp_path: Path) -> None:
    torch.manual_seed(8); module = _ExportHeads().eval()
    example = torch.randn(2, 15, 125)
    torchscript_path = tmp_path / "student.ts"; torch.jit.trace(module, example).save(str(torchscript_path))
    onnx_path = tmp_path / "student.onnx"
    names = ("action", "estimated_force", "contact_probability", "slip_probability", "traction_utilization", "traction_margin", "estimator_confidence")
    torch.onnx.export(module, example, onnx_path, input_names=["history"], output_names=list(names), dynamic_axes={"history": {0: "batch"}, **{name: {0: "batch"} for name in names}}, opset_version=17)
    torchscript = torch.jit.load(str(torchscript_path)).eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    history = torch.linspace(-1.0, 1.0, 3 * 15 * 125).reshape(3, 15, 125)
    with torch.inference_mode():
        torch_output = torchscript(history)
    onnx_output = session.run(None, {"history": history.numpy()})
    expected_shapes = ((3, 29), (3, 6), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2))
    for actual, converted, shape in zip(torch_output, onnx_output, expected_shapes, strict=True):
        assert tuple(actual.shape) == shape
        assert converted.shape == shape
        assert np.isfinite(converted).all()
        np.testing.assert_allclose(actual.numpy(), converted, atol=5e-5, rtol=5e-5)


def test_mujoco_policy_path_cannot_read_contact_truth() -> None:
    estimator_source = (MUJOCO_DIR / "torque_force_estimator.py").read_text()
    runner_source = (MUJOCO_DIR / "run_torque_traction_sim2sim.py").read_text()
    assert "mujoco.mj_contactForce(" not in estimator_source
    # Truth is deliberately a separate adapter and must be called after the
    # policy input/history has already been formed.
    assert runner_source.index("history = torch.from_numpy") < runner_source.index("truth = self.truth.read")
