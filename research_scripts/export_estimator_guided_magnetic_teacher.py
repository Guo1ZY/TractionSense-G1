#!/usr/bin/env python3
"""Export a calibrated 1864-D magnetic front-end around the Oracle Teacher.

This is an intermediate DAgger/runtime policy, not a privileged policy: exact
friction is never an input.  The module reconstructs the Teacher's foot-force
history from the normalized 15xXYZ Hall proxy, estimates friction from the
same causal 1864-D observation, and calls the frozen 641-D Teacher.  A
calibration/health gate falls back to the proven shared-magnetic Student when
the Hall array is stale, invalid, or inconsistent with its calibrated spatial
profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from distill_traction_student import load_actor
from evaluate_friction_estimator import (
    FrictionEstimator,
    NormalizedFrictionEstimator,
)
from export_confidence_gated_magnetic_policy import CalibrationGatedPolicy
from export_jointwise_magnetic_ensemble import load_policy
from train_shared_magnetic_policy import (
    AXES,
    BASE_DIM,
    FEET,
    HISTORY,
    INPUT_DIM,
    MAGNETIC_DIM,
    SENSORS,
)

TEACHER_DIM = 641
JOINTS = 29
JOINT_MIRROR_INDEX = torch.tensor(
    [
        2, 3, 0, 1, 4, 26, 8, 9, 6, 7, 10, 20, 14, 15, 12,
        13, 25, 27, 28, 23, 11, 24, 22, 19, 21, 16, 5, 17, 18,
    ],
    dtype=torch.long,
)
JOINT_MIRROR_SIGN = torch.tensor(
    [
        1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1,
    ],
    dtype=torch.float32,
)
LATERAL_JOINTS = (4, 6, 8, 12, 13, 14, 15, 22)
LATERAL_ARM_JOINTS = (5, 17, 18, 19, 23, 26, 27, 28)
PROFILE = np.asarray(
    [
        0.70, 0.76, 0.70,
        0.76, 0.82, 0.76,
        0.82, 0.88, 0.82,
        0.88, 0.94, 0.88,
        0.94, 1.00, 0.94,
    ],
    dtype=np.float32,
)
PROFILE /= PROFILE.mean()
MIXING = np.asarray(
    [
        [0.14, 1.00],
        [-0.10, 0.42],
        [1.00, 0.12],
    ],
    dtype=np.float32,
)
MIXING_PINV = np.linalg.pinv(MIXING).astype(np.float32)


def mirror_joints(value: torch.Tensor) -> torch.Tensor:
    index = JOINT_MIRROR_INDEX.to(value.device)
    sign = JOINT_MIRROR_SIGN.to(device=value.device, dtype=value.dtype)
    return value.index_select(-1, index) * sign


def mirror_teacher_observation(
    observation: torch.Tensor,
    motion_feedback: bool = False,
) -> torch.Tensor:
    """Reflect a 641-D Oracle observation across the sagittal plane."""

    if observation.ndim != 2 or observation.shape[1] != TEACHER_DIM:
        raise ValueError(f"expected Nx{TEACHER_DIM}, got {tuple(observation.shape)}")
    mirrored = observation.clone()
    mirrored[:, 0:15] = (
        observation[:, 0:15].reshape(-1, 5, 3)
        * observation.new_tensor([-1.0, 1.0, -1.0])
    ).reshape(-1, 15)
    mirrored[:, 15:30] = (
        observation[:, 15:30].reshape(-1, 5, 3)
        * observation.new_tensor([1.0, -1.0, 1.0])
    ).reshape(-1, 15)
    mirrored[:, 30:45] = (
        observation[:, 30:45].reshape(-1, 5, 3)
        * observation.new_tensor([1.0, -1.0, -1.0])
    ).reshape(-1, 15)
    for start in (45, 190, 335):
        joint_history = observation[:, start : start + 5 * JOINTS].reshape(
            -1, 5, JOINTS
        )
        mirrored[:, start : start + 5 * JOINTS] = mirror_joints(
            joint_history
        ).reshape(-1, 5 * JOINTS)
    for start in (480, 510, 540, 570, 600):
        feet = observation[:, start : start + HISTORY * FEET].reshape(
            -1, HISTORY, FEET
        )
        mirrored[:, start : start + HISTORY * FEET] = feet[:, :, [1, 0]].reshape(
            -1, HISTORY * FEET
        )
    if motion_feedback:
        mirrored[:, 630:640] = -observation[:, 630:640]
    return mirrored


class TeacherLateralSymmetryEnsemble(nn.Module):
    """Use mirrored Teacher consistency only on lateral-control joints."""

    def __init__(
        self,
        teacher: nn.Module,
        lateral_weight: float,
        arm_weight: float,
        motion_feedback: bool = False,
        motion_gate_center: float = -1.0,
        motion_gate_sharpness: float = 30.0,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        mask = torch.zeros(JOINTS)
        mask[list(LATERAL_JOINTS)] = float(lateral_weight)
        mask[list(LATERAL_ARM_JOINTS)] = float(arm_weight)
        self.register_buffer("symmetry_weight", mask)
        self.motion_feedback = bool(motion_feedback)
        self.motion_gate_center = float(motion_gate_center)
        self.motion_gate_sharpness = float(motion_gate_sharpness)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        direct = self.teacher(observation)
        mirrored = mirror_joints(
            self.teacher(
                mirror_teacher_observation(
                    observation,
                    motion_feedback=self.motion_feedback,
                )
            )
        )
        symmetric = 0.5 * (direct + mirrored)
        symmetry_weight = self.symmetry_weight
        if self.motion_feedback and self.motion_gate_center >= 0.0:
            current_motion = observation[:, 638:640].abs().amax(dim=1)
            gate = torch.sigmoid(
                (current_motion - self.motion_gate_center)
                * self.motion_gate_sharpness
            )
            symmetry_weight = gate.unsqueeze(1) * symmetry_weight.unsqueeze(0)
        return torch.lerp(direct, symmetric, symmetry_weight)


class EstimatorGuidedTeacher(nn.Module):
    """Convert the causal magnetic observation into a deployable Teacher input."""

    def __init__(
        self,
        teacher: nn.Module,
        estimator_checkpoint: dict,
        motion_feedback: bool = False,
        mu_scale: float = 1.0,
        mu_bias: float = 0.0,
        high_mu_command_scale: float = 1.0,
        high_mu_command_center: float = 0.75,
        high_mu_command_sharpness: float = 20.0,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.motion_feedback = bool(motion_feedback)
        self.mu_scale = float(mu_scale)
        self.mu_bias = float(mu_bias)
        self.high_mu_command_scale = float(high_mu_command_scale)
        self.high_mu_command_center = float(high_mu_command_center)
        self.high_mu_command_sharpness = float(high_mu_command_sharpness)
        command_mask = torch.zeros(BASE_DIM)
        command_mask[[30, 33, 36, 39, 42]] = 1.0
        self.register_buffer("command_vx_mask", command_mask)
        estimator = FrictionEstimator(int(estimator_checkpoint["input_dim"]))
        estimator.load_state_dict(estimator_checkpoint["model"], strict=True)
        self.estimator = NormalizedFrictionEstimator(
            estimator,
            np.asarray(estimator_checkpoint["mean"], dtype=np.float32),
            np.asarray(estimator_checkpoint["scale"], dtype=np.float32),
        )
        indices = np.asarray(estimator_checkpoint["feature_indices"], dtype=np.int64)
        self.register_buffer("estimator_indices", torch.from_numpy(indices))
        self.register_buffer(
            "profile",
            torch.from_numpy(PROFILE).reshape(1, 1, 1, SENSORS, 1),
        )
        self.register_buffer("mixing_pinv", torch.from_numpy(MIXING_PINV))

    def teacher_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected Nx{INPUT_DIM}, got {tuple(observation.shape)}")
        magnetic = observation[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        clipped = torch.clamp(magnetic / 5.0, -0.999999, 0.999999)
        # Equivalent to atanh(), expressed with ONNX opset-17 primitives.
        inverse_tanh = 0.5 * torch.log((1.0 + clipped) / (1.0 - clipped))
        signal = 5.0 * inverse_tanh
        axis_signal = (signal / self.profile).mean(dim=3)
        forces = torch.matmul(axis_signal, self.mixing_pinv.transpose(0, 1))
        forces = torch.clamp(forces, 0.0, 5.0)
        normal = forces[..., 0]
        tangent = forces[..., 1]
        force_magnitude_n = 100.0 * torch.sqrt(normal.square() + tangent.square())
        contact = torch.sigmoid((force_magnitude_n - 5.0) * 2.0)
        ratio = torch.clamp(tangent / (normal + 0.05), 0.0, 2.0)
        total_normal = normal.sum(dim=2, keepdim=True)
        load = torch.where(
            total_normal > 1.0e-6,
            normal / torch.clamp(total_normal, min=1.0e-6),
            torch.full_like(normal, 0.5),
        )
        estimator_input = observation.index_select(1, self.estimator_indices)
        estimated_mu = torch.clamp(
            self.mu_scale * self.estimator(estimator_input) + self.mu_bias,
            0.0,
            1.2,
        ).unsqueeze(1)
        base = observation[:, :BASE_DIM]
        if self.high_mu_command_scale != 1.0:
            high_mu_gate = torch.sigmoid(
                (estimated_mu - self.high_mu_command_center)
                * self.high_mu_command_sharpness
            )
            gain = 1.0 + (
                self.high_mu_command_scale - 1.0
            ) * high_mu_gate
            base = base * (
                1.0
                + (gain - 1.0) * self.command_vx_mask.unsqueeze(0)
            )
        if self.motion_feedback:
            deploy_feedback = observation[:, 1862:1864].repeat(1, 5)
        else:
            valid = observation[:, 1860:1862].amin(dim=1, keepdim=True)
            age = observation[:, 1862:1864].amax(dim=1, keepdim=True)
            deploy_feedback = torch.cat(
                (valid.repeat(1, 5), age.repeat(1, 5)),
                dim=1,
            )
        teacher_observation = torch.cat(
            (
                base,
                contact.reshape(-1, HISTORY * FEET),
                normal.reshape(-1, HISTORY * FEET),
                tangent.reshape(-1, HISTORY * FEET),
                ratio.reshape(-1, HISTORY * FEET),
                load.reshape(-1, HISTORY * FEET),
                deploy_feedback,
                estimated_mu,
            ),
            dim=1,
        )
        return teacher_observation

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.teacher(self.teacher_observation(observation))


def build_runtime(payload: dict) -> nn.Module:
    teacher = load_actor(Path(payload["teacher_onnx"]), TEACHER_DIM).eval()
    metrics = payload["metrics"]
    teacher = TeacherLateralSymmetryEnsemble(
        teacher,
        float(metrics.get("teacher_lateral_symmetry", 0.0)),
        float(metrics.get("teacher_arm_symmetry", 0.0)),
        bool(metrics.get("motion_feedback", False)),
        float(metrics.get("symmetry_motion_center", -1.0)),
        float(metrics.get("symmetry_motion_sharpness", 30.0)),
    ).eval()
    estimator_checkpoint = torch.load(
        payload["estimator_pt"], map_location="cpu", weights_only=False
    )
    guided = EstimatorGuidedTeacher(
        teacher,
        estimator_checkpoint,
        motion_feedback=bool(metrics.get("motion_feedback", False)),
        mu_scale=float(metrics.get("mu_scale", 1.0)),
        mu_bias=float(metrics.get("mu_bias", 0.0)),
        high_mu_command_scale=float(
            metrics.get("high_mu_command_scale", 1.0)
        ),
        high_mu_command_center=float(
            metrics.get("high_mu_command_center", 0.75)
        ),
        high_mu_command_sharpness=float(
            metrics.get("high_mu_command_sharpness", 20.0)
        ),
    ).eval()
    safe = load_policy(Path(payload["safe_checkpoint"]))
    return CalibrationGatedPolicy(
        safe,
        guided,
        float(metrics["residual_center"]),
        float(metrics["residual_sharpness"]),
        float(metrics["evidence_center"]),
        float(metrics["evidence_sharpness"]),
        motion_feedback=bool(metrics.get("motion_feedback", False)),
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", required=True, type=Path)
    parser.add_argument("--estimator-pt", required=True, type=Path)
    parser.add_argument("--safe", required=True, type=Path)
    parser.add_argument("--residual-center", type=float, default=0.06)
    parser.add_argument("--residual-sharpness", type=float, default=150.0)
    parser.add_argument("--evidence-center", type=float, default=0.15)
    parser.add_argument("--evidence-sharpness", type=float, default=50.0)
    parser.add_argument("--teacher-lateral-symmetry", type=float, default=0.0)
    parser.add_argument("--teacher-arm-symmetry", type=float, default=0.0)
    parser.add_argument(
        "--motion-feedback",
        action="store_true",
        help="Interpret final two 1864-D channels as [body_vy, relative_heading]",
    )
    parser.add_argument(
        "--mu-scale",
        type=float,
        default=1.0,
        help="Monotonic Sim2Sim calibration applied to estimated mu before Teacher",
    )
    parser.add_argument("--mu-bias", type=float, default=0.0)
    parser.add_argument("--high-mu-command-scale", type=float, default=1.0)
    parser.add_argument("--high-mu-command-center", type=float, default=0.75)
    parser.add_argument("--high-mu-command-sharpness", type=float, default=20.0)
    parser.add_argument(
        "--symmetry-motion-center",
        type=float,
        default=-1.0,
        help="If >=0, activate Teacher symmetry only as |vy/yaw| exceeds this",
    )
    parser.add_argument("--symmetry-motion-sharpness", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.teacher_lateral_symmetry <= 1.0:
        raise ValueError("--teacher-lateral-symmetry must be in [0, 1]")
    if not 0.0 <= args.teacher_arm_symmetry <= 1.0:
        raise ValueError("--teacher-arm-symmetry must be in [0, 1]")
    if args.mu_scale <= 0.0:
        raise ValueError("--mu-scale must be positive")
    if args.high_mu_command_scale < 1.0:
        raise ValueError("--high-mu-command-scale must be >= 1")
    estimator_checkpoint = torch.load(
        args.estimator_pt, map_location="cpu", weights_only=False
    )
    teacher = TeacherLateralSymmetryEnsemble(
        load_actor(args.teacher_onnx, TEACHER_DIM).eval(),
        args.teacher_lateral_symmetry,
        args.teacher_arm_symmetry,
        args.motion_feedback,
        args.symmetry_motion_center,
        args.symmetry_motion_sharpness,
    ).eval()
    guided = EstimatorGuidedTeacher(
        teacher,
        estimator_checkpoint,
        motion_feedback=args.motion_feedback,
        mu_scale=args.mu_scale,
        mu_bias=args.mu_bias,
        high_mu_command_scale=args.high_mu_command_scale,
        high_mu_command_center=args.high_mu_command_center,
        high_mu_command_sharpness=args.high_mu_command_sharpness,
    ).eval()
    model = CalibrationGatedPolicy(
        load_policy(args.safe),
        guided,
        args.residual_center,
        args.residual_sharpness,
        args.evidence_center,
        args.evidence_sharpness,
        motion_feedback=args.motion_feedback,
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "calibration-gated estimator-guided magnetic Teacher",
        "policy_type": "estimator_guided_magnetic_teacher",
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "estimator_pt": str(args.estimator_pt.resolve()),
        "safe": str(args.safe.resolve()),
        "safe_checkpoint": str(args.safe.resolve()),
        "residual_center": args.residual_center,
        "residual_sharpness": args.residual_sharpness,
        "evidence_center": args.evidence_center,
        "evidence_sharpness": args.evidence_sharpness,
        "teacher_lateral_symmetry": args.teacher_lateral_symmetry,
        "teacher_arm_symmetry": args.teacher_arm_symmetry,
        "motion_feedback": args.motion_feedback,
        "motion_feedback_history": "current [body_vy, relative_heading] repeated for 5 Teacher frames",
        "mu_scale": args.mu_scale,
        "mu_bias": args.mu_bias,
        "symmetry_motion_center": args.symmetry_motion_center,
        "symmetry_motion_sharpness": args.symmetry_motion_sharpness,
        "high_mu_command_scale": args.high_mu_command_scale,
        "high_mu_command_center": args.high_mu_command_center,
        "high_mu_command_sharpness": args.high_mu_command_sharpness,
        "friction_input": "estimated from causal 1864-D observation",
        "privileged_mu_at_inference": False,
        "input_dim": INPUT_DIM,
        "output_dim": 29,
    }
    payload = {
        "policy_type": "estimator_guided_magnetic_teacher",
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "estimator_pt": str(args.estimator_pt.resolve()),
        "safe_checkpoint": str(args.safe.resolve()),
        "metrics": manifest,
        "input_dim": INPUT_DIM,
    }
    torch.save(payload, args.output_dir / "shared_magnetic_policy.pt")
    torch.onnx.export(
        model,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
