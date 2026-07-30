#!/usr/bin/env python3
"""Export a calibration-confidence gate between safe and Sim2Sim actors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from export_jointwise_magnetic_ensemble import JointwiseEnsemble, load_policy
from train_shared_magnetic_policy import (
    AXES,
    BASE_DIM,
    FEET,
    HISTORY,
    INPUT_DIM,
    MAGNETIC_DIM,
    SENSORS,
)

PROFILE = (
    0.70, 0.76, 0.70,
    0.76, 0.82, 0.76,
    0.82, 0.88, 0.82,
    0.88, 0.94, 0.88,
    0.94, 1.00, 0.94,
)


class CalibrationGatedPolicy(nn.Module):
    """Use fast policy only for fresh, profile-consistent magnetic arrays."""

    def __init__(
        self,
        safe: nn.Module,
        fast: nn.Module,
        residual_center: float,
        residual_sharpness: float,
        evidence_center: float,
        evidence_sharpness: float,
        motion_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.safe = safe
        self.fast = fast
        profile = torch.tensor(PROFILE)
        profile = profile / profile.mean()
        self.register_buffer(
            "profile", profile.reshape(1, 1, 1, SENSORS, 1)
        )
        self.residual_center = float(residual_center)
        self.residual_sharpness = float(residual_sharpness)
        self.evidence_center = float(evidence_center)
        self.evidence_sharpness = float(evidence_sharpness)
        self.motion_feedback = bool(motion_feedback)

    def confidence(self, observation: torch.Tensor) -> torch.Tensor:
        magnetic = observation[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        normalized = magnetic / self.profile
        sensor_mean = normalized.mean(dim=3, keepdim=True)
        residual = torch.abs(normalized - sensor_mean).mean(dim=(1, 2, 3, 4))
        evidence = torch.abs(sensor_mean).mean(dim=(1, 2, 3, 4))
        score = residual / (evidence + 0.05)
        calibration = torch.sigmoid(
            (self.residual_center - score) * self.residual_sharpness
        )
        has_evidence = torch.sigmoid(
            (evidence - self.evidence_center) * self.evidence_sharpness
        )
        valid = observation[:, 1860:1862].amin(dim=1)
        if self.motion_feedback:
            # The final two slots are [body_vy, relative_heading], not sensor
            # age. Magnetic evidence and valid_lr still guard this branch.
            fresh = torch.ones_like(valid)
        else:
            normalized_age = observation[:, 1862:1864].amax(dim=1)
            fresh = 1.0 - normalized_age.square()
        return (calibration * has_evidence * valid * fresh).clamp(0.0, 1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        safe_action = self.safe(observation)
        fast_action = self.fast(observation)
        confidence = self.confidence(observation).unsqueeze(1)
        return torch.lerp(safe_action, fast_action, confidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--stable", required=True, type=Path)
    parser.add_argument("--lateral-weight", type=float, default=0.75)
    parser.add_argument("--arm-weight", type=float, default=1.0)
    parser.add_argument("--residual-center", type=float, default=0.06)
    parser.add_argument("--residual-sharpness", type=float, default=150.0)
    parser.add_argument("--evidence-center", type=float, default=0.15)
    parser.add_argument("--evidence-sharpness", type=float, default=50.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    safe = load_policy(args.safe)
    fast = JointwiseEnsemble(
        load_policy(args.fast),
        load_policy(args.stable),
        args.lateral_weight,
        args.arm_weight,
    )
    model = CalibrationGatedPolicy(
        safe,
        fast,
        args.residual_center,
        args.residual_sharpness,
        args.evidence_center,
        args.evidence_sharpness,
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "magnetic calibration-confidence safety gate",
        "safe": str(args.safe.resolve()),
        "fast": str(args.fast.resolve()),
        "stable": str(args.stable.resolve()),
        "stable_lateral_weight": args.lateral_weight,
        "stable_arm_weight": args.arm_weight,
        "residual_center": args.residual_center,
        "residual_sharpness": args.residual_sharpness,
        "evidence_center": args.evidence_center,
        "evidence_sharpness": args.evidence_sharpness,
        "health_gate": "min(valid_lr) * (1 - max(age_lr)^2)",
        "input_dim": INPUT_DIM,
        "output_dim": 29,
    }
    torch.save(
        {
            "metrics": manifest,
            "input_dim": INPUT_DIM,
            "safe_checkpoint": str(args.safe.resolve()),
            "fast_checkpoint": str(args.fast.resolve()),
            "stable_checkpoint": str(args.stable.resolve()),
        },
        args.output_dir / "shared_magnetic_policy.pt",
    )
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
