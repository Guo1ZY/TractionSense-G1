#!/usr/bin/env python3
"""Export a calibration-gated magnetic policy with high-friction speed boost.

The deploy command remains capped at 1.0 m/s.  This wrapper only compensates
the actor's conservative tracking error by increasing the command values seen
inside the fast actor when all of these conditions are met:

* both magnetic feet are valid and fresh;
* the magnetic array is consistent with its calibration profile;
* the learned traction estimate is high; and
* the forward command is already in the high-speed regime.

Invalid, stale, or uncalibrated magnetic input continues to use the unchanged
safe actor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from export_confidence_gated_magnetic_policy import CalibrationGatedPolicy
from export_jointwise_magnetic_ensemble import (
    LATERAL_ARM_JOINTS,
    LATERAL_JOINTS,
    load_policy,
)
from train_shared_magnetic_policy import INPUT_DIM


COMMAND_VX_INDICES = (30, 33, 36, 39, 42)


class HighFrictionSpeedBoostPolicy(CalibrationGatedPolicy):
    """Apply a smooth, confidence-gated command boost to the fast branch."""

    def __init__(
        self,
        safe: nn.Module,
        fast: nn.Module,
        stable: nn.Module,
        lateral_weight: float,
        arm_weight: float,
        residual_center: float,
        residual_sharpness: float,
        evidence_center: float,
        evidence_sharpness: float,
        boost_factor: float,
        traction_center: float,
        traction_sharpness: float,
        command_center: float,
        command_sharpness: float,
        stable_uses_boosted_command: bool,
    ) -> None:
        # CalibrationGatedPolicy owns the tested calibration/health gate.
        super().__init__(
            safe,
            fast,
            residual_center,
            residual_sharpness,
            evidence_center,
            evidence_sharpness,
        )
        self.stable = stable
        stable_weight = torch.zeros(29)
        stable_weight[list(LATERAL_JOINTS)] = float(lateral_weight)
        stable_weight[list(LATERAL_ARM_JOINTS)] = float(arm_weight)
        self.register_buffer("stable_weight", stable_weight)
        command_mask = torch.zeros(INPUT_DIM)
        command_mask[list(COMMAND_VX_INDICES)] = 1.0
        self.register_buffer("command_mask", command_mask)
        self.boost_factor = float(boost_factor)
        self.traction_center = float(traction_center)
        self.traction_sharpness = float(traction_sharpness)
        self.command_center = float(command_center)
        self.command_sharpness = float(command_sharpness)
        self.stable_uses_boosted_command = bool(stable_uses_boosted_command)

    def boost_gate(self, observation: torch.Tensor) -> torch.Tensor:
        predicted_mu, _ = self.fast.auxiliary(observation)
        traction = torch.sigmoid(
            (predicted_mu - self.traction_center) * self.traction_sharpness
        )
        command_vx = observation[:, list(COMMAND_VX_INDICES)].mean(dim=1)
        high_command = torch.sigmoid(
            (command_vx - self.command_center) * self.command_sharpness
        )
        return (self.confidence(observation) * traction * high_command).clamp(
            0.0, 1.0
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        calibration_confidence = self.confidence(observation)
        boost = (
            (self.boost_factor - 1.0)
            * self.boost_gate(observation)
        ).unsqueeze(1)
        boosted_observation = observation * (
            1.0 + boost * self.command_mask.unsqueeze(0)
        )
        fast_action = self.fast(boosted_observation)
        stable_observation = (
            boosted_observation
            if self.stable_uses_boosted_command
            else observation
        )
        stable_action = self.stable(stable_observation)
        corrected_action = torch.lerp(
            fast_action, stable_action, self.stable_weight
        )
        safe_action = self.safe(observation)
        return torch.lerp(
            safe_action,
            corrected_action,
            calibration_confidence.unsqueeze(1),
        )


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
    parser.add_argument("--boost-factor", type=float, default=1.10)
    parser.add_argument("--traction-center", type=float, default=0.65)
    parser.add_argument("--traction-sharpness", type=float, default=10.0)
    parser.add_argument("--command-center", type=float, default=0.70)
    parser.add_argument("--command-sharpness", type=float, default=15.0)
    parser.add_argument(
        "--stable-uses-boosted-command",
        action="store_true",
        help="Also compensate the stable branch; faster but potentially more lateral motion",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not 1.0 <= args.boost_factor <= 1.25:
        raise ValueError("--boost-factor must be in [1.0, 1.25]")
    if not 0.0 <= args.lateral_weight <= 1.0:
        raise ValueError("--lateral-weight must be in [0, 1]")
    if not 0.0 <= args.arm_weight <= 1.0:
        raise ValueError("--arm-weight must be in [0, 1]")

    model = HighFrictionSpeedBoostPolicy(
        load_policy(args.safe),
        load_policy(args.fast),
        load_policy(args.stable),
        args.lateral_weight,
        args.arm_weight,
        args.residual_center,
        args.residual_sharpness,
        args.evidence_center,
        args.evidence_sharpness,
        args.boost_factor,
        args.traction_center,
        args.traction_sharpness,
        args.command_center,
        args.command_sharpness,
        args.stable_uses_boosted_command,
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "calibration-gated high-friction speed compensation",
        "safe": str(args.safe.resolve()),
        "fast": str(args.fast.resolve()),
        "stable": str(args.stable.resolve()),
        "stable_lateral_weight": args.lateral_weight,
        "stable_arm_weight": args.arm_weight,
        "residual_center": args.residual_center,
        "residual_sharpness": args.residual_sharpness,
        "evidence_center": args.evidence_center,
        "evidence_sharpness": args.evidence_sharpness,
        "boost_factor": args.boost_factor,
        "traction_center": args.traction_center,
        "traction_sharpness": args.traction_sharpness,
        "command_center": args.command_center,
        "command_sharpness": args.command_sharpness,
        "command_vx_indices": list(COMMAND_VX_INDICES),
        "stable_branch_command": (
            "boosted internal command"
            if args.stable_uses_boosted_command
            else "unmodified external command"
        ),
        "external_command_cap_mps": 1.0,
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
