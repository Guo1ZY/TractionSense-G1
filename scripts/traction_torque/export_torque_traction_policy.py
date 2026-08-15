#!/usr/bin/env python3
"""Export a fixed 15x125 torque-traction Student package."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn

from unitree_rl_lab.traction.schema import ACTION_DIM, FORCE_FRAME, FORCE_ORDER, G1_29DOF_JOINT_ORDER, POLICY_DT_S
from unitree_rl_lab.traction_torque.governor import TorqueTractionGovernorCfg
from unitree_rl_lab.traction_torque.networks import TorqueTractionStudentPolicy
from unitree_rl_lab.traction_torque.networks import torque_history_to_legacy_proprio
from unitree_rl_lab.traction_torque.randomization import TorqueDynamicsRandomizationCfg
from unitree_rl_lab.traction_torque.schema import TORQUE_TRACTION_FRAME_SCHEMA


class ExportableTorqueStudent(nn.Module):
    def __init__(self, student: TorqueTractionStudentPolicy) -> None:
        super().__init__(); self.student = student

    def forward(self, history: torch.Tensor):
        output = self.student(history)
        return (output.action, output.estimated_force, output.contact_probability, output.slip_probability, output.traction_utilization, output.traction_margin, output.estimator_confidence)


class ExportableRslTorqueStudent(nn.Module):
    """Deployment heads plus the PPO-trained 499-D deterministic actor mean."""

    def __init__(self, student: TorqueTractionStudentPolicy, actor_state: dict[str, torch.Tensor]) -> None:
        super().__init__(); self.student = student
        self.actor = nn.Sequential(nn.Linear(499, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29))
        mapped = {}
        for index, source_index in zip((0, 2, 4, 6), (0, 2, 4, 6), strict=True):
            mapped[f"{index}.weight"] = actor_state[f"mlp.{source_index}.weight"]
            mapped[f"{index}.bias"] = actor_state[f"mlp.{source_index}.bias"]
        self.actor.load_state_dict(mapped, strict=True)

    def forward(self, history: torch.Tensor):
        output = self.student(history)
        command = history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")]
        baseline = torque_history_to_legacy_proprio(history)
        distilled_residual = output.action - self.student.baseline_actor(baseline)
        action = self.actor(torch.cat((baseline, output.traction_latent, command), dim=-1)) + distilled_residual
        return (action, output.estimated_force, output.contact_probability, output.slip_probability, output.traction_utilization, output.traction_margin, output.estimator_confidence)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_baseline(student: TorqueTractionStudentPolicy, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = state.get("actor_state_dict", state.get("model_state_dict", state))
    expected = student.baseline_actor.state_dict()
    selected = {name: actor[name] for name in expected if name in actor and actor[name].shape == expected[name].shape}
    if set(selected) != set(expected):
        raise KeyError(f"baseline actor checkpoint missing: {sorted(set(expected) - set(selected))}")
    student.baseline_actor.load_state_dict(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_checkpoint", type=Path, default=Path("model/rl/model_49999.pt"))
    parser.add_argument("--student_checkpoint", type=Path)
    parser.add_argument("--rsl_checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/traction_torque/export_warm_start"))
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.student_checkpoint and args.rsl_checkpoint:
        parser.error("--student_checkpoint and --rsl_checkpoint are mutually exclusive")
    student = TorqueTractionStudentPolicy()
    _load_baseline(student, args.baseline_checkpoint)
    status = "baseline_warm_start_untrained_traction_heads"
    if args.student_checkpoint:
        checkpoint = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("student_state_dict", checkpoint.get("model_state_dict", checkpoint))
        student.load_state_dict(state, strict=True)
        status = "trained_candidate_checkpoint_loaded"
    exportable: nn.Module = ExportableTorqueStudent(student)
    if args.rsl_checkpoint:
        checkpoint = torch.load(args.rsl_checkpoint, map_location="cpu", weights_only=False)
        actor_state = checkpoint["actor_state_dict"]
        student.load_state_dict({name.removeprefix("student_policy."): value for name, value in actor_state.items() if name.startswith("student_policy.")}, strict=True)
        exportable = ExportableRslTorqueStudent(student, actor_state)
        status = "on_policy_rsl_candidate_checkpoint_loaded"
    student.eval(); exportable.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.randn(2, 15, 125)
    with torch.inference_mode(): reference = exportable(example)
    traced = torch.jit.trace(exportable, example)
    torchscript_path = args.output_dir / "torque_traction_student.ts"
    traced.save(str(torchscript_path))
    onnx_path = args.output_dir / "torque_traction_student.onnx"
    torch.onnx.export(
        exportable, example, onnx_path,
        input_names=["history"],
        output_names=["action", "estimated_force", "contact_probability", "slip_probability", "traction_utilization", "traction_margin", "estimator_confidence"],
        dynamic_axes={"history": {0: "batch"}, **{name: {0: "batch"} for name in ("action", "estimated_force", "contact_probability", "slip_probability", "traction_utilization", "traction_margin", "estimator_confidence")}},
        opset_version=17,
    )
    with torch.inference_mode(): torchscript_output = torch.jit.load(str(torchscript_path))(example)
    torchscript_max_error = max((a - b).abs().max().item() for a, b in zip(reference, torchscript_output, strict=True))
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"history": example.numpy()})
    onnx_max_error = max(float(np.max(np.abs(a.detach().numpy() - b))) for a, b in zip(reference, onnx_output, strict=True))
    schema_path = args.output_dir / "observation_schema.json"; TORQUE_TRACTION_FRAME_SCHEMA.write_json(schema_path)
    (args.output_dir / "joint_order.json").write_text(json.dumps({"action_dimension": ACTION_DIM, "joint_order": G1_29DOF_JOINT_ORDER, "action_scale_rad": 0.25}, indent=2) + "\n")
    (args.output_dir / "force_frame.json").write_text(json.dumps({"order": FORCE_ORDER, "frame": FORCE_FRAME, "axes": {"x": "toe", "y": "robot-left", "z": "up"}, "analytical_output_unit": "N", "policy_force_input_unit": "N/(robot_mass*9.81)"}, indent=2) + "\n")
    (args.output_dir / "dynamics_estimator.json").write_text(json.dumps({"equation": "M(q)qdd+h=S^T*tau+J^T*F", "root_dofs": 6, "velocity_dofs": 35, "solver": "weighted temporally regularized least squares", "deployment_inputs": ["q", "dq", "filtered_qdd", "tau_est", "IMU", "foot_Jacobian"], "contact_truth_input": False}, indent=2) + "\n")
    (args.output_dir / "torque_randomization.json").write_text(json.dumps(TorqueDynamicsRandomizationCfg(curriculum_stage=5).to_dict(), indent=2) + "\n")
    (args.output_dir / "command_governor.json").write_text(json.dumps(asdict(TorqueTractionGovernorCfg()), indent=2) + "\n")
    metadata = {
        "status": status, "seed": args.seed, "policy_input": ["batch", 15, 125],
        "history_order": "time-major oldest-to-newest", "control_frequency_hz": 1.0 / POLICY_DT_S,
        "outputs": {"action": 29, "estimated_force": 6, "contact_probability": 2, "slip_probability": 2, "traction_utilization": 2, "traction_margin": 2, "estimator_confidence": 2},
        "student_contains_privileged_simulator_input": False,
        "parity": {"torchscript_max_abs_error": torchscript_max_error, "onnx_max_abs_error": onnx_max_error},
    }
    metadata_path = args.output_dir / "metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    readme = f"""# Motor-Torque Traction Student export\n\nStatus: `{status}`.\n\nInput is float32 `[batch,15,125]`, oldest to newest. The package does not accept ContactSensor force, ground-truth friction, or privileged slip labels. `estimated_force` in this network output follows the normalized policy force channels; the analytical runtime retains force in Newtons before normalization.\n\nTorchScript parity max error: `{torchscript_max_error:.9g}`. ONNX parity max error: `{onnx_max_error:.9g}`. This export performs no real-robot control.\n"""
    (args.output_dir / "README.md").write_text(readme)
    hashes = {path.name: _sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    (args.output_dir / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **metadata, "sha256": hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
