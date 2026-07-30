#!/usr/bin/env python3
"""Export a fast-policy actor with lateral joints corrected by a stable actor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_shared_magnetic_policy import INPUT_DIM, SharedMagneticPolicy

LATERAL_JOINTS = (4, 6, 8, 12, 13, 14, 15, 22)
LATERAL_ARM_JOINTS = (5, 17, 18, 19, 23, 26, 27, 28)


class JointwiseEnsemble(nn.Module):
    def __init__(
        self,
        fast: SharedMagneticPolicy,
        stable: SharedMagneticPolicy,
        lateral_weight: float,
        arm_weight: float,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.stable = stable
        mask = torch.zeros(29)
        mask[list(LATERAL_JOINTS)] = float(lateral_weight)
        mask[list(LATERAL_ARM_JOINTS)] = float(arm_weight)
        self.register_buffer("stable_weight", mask)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        fast_action = self.fast(observation)
        stable_action = self.stable(observation)
        return torch.lerp(fast_action, stable_action, self.stable_weight)


def load_policy(path: Path) -> SharedMagneticPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = SharedMagneticPolicy().eval()
    policy.load_state_dict(payload["model"], strict=True)
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--stable", required=True, type=Path)
    parser.add_argument("--lateral-weight", required=True, type=float)
    parser.add_argument("--arm-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.lateral_weight <= 1.0:
        raise ValueError("--lateral-weight must be in [0, 1]")
    if not 0.0 <= args.arm_weight <= 1.0:
        raise ValueError("--arm-weight must be in [0, 1]")
    model = JointwiseEnsemble(
        load_policy(args.fast),
        load_policy(args.stable),
        args.lateral_weight,
        args.arm_weight,
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "jointwise fast/stable magnetic-policy ensemble",
        "fast": str(args.fast.resolve()),
        "stable": str(args.stable.resolve()),
        "stable_lateral_weight": args.lateral_weight,
        "stable_arm_weight": args.arm_weight,
        "stable_joint_indices_isaac": list(LATERAL_JOINTS),
        "stable_arm_joint_indices_isaac": list(LATERAL_ARM_JOINTS),
        "input_dim": INPUT_DIM,
        "output_dim": 29,
    }
    # The runtime artifact is the ONNX ensemble; these references make the two
    # source checkpoints explicit without pretending it is one trainable state.
    torch.save(
        {
            "metrics": manifest,
            "input_dim": INPUT_DIM,
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
