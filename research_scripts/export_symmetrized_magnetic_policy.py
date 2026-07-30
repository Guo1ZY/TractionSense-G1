#!/usr/bin/env python3
"""Export a left/right symmetry ensemble around a shared magnetic policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fine_tune_shared_magnetic_dagger import mirror_joints, mirror_observation
from train_shared_magnetic_policy import INPUT_DIM, SharedMagneticPolicy


class SymmetryEnsemble(nn.Module):
    def __init__(self, policy: SharedMagneticPolicy, mirror_weight: float) -> None:
        super().__init__()
        self.policy = policy
        self.mirror_weight = float(mirror_weight)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        direct = self.policy(observation)
        reflected = mirror_joints(self.policy(mirror_observation(observation)))
        return torch.lerp(direct, reflected, self.mirror_weight)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--mirror-weight", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.mirror_weight <= 0.5:
        raise ValueError("--mirror-weight must be in [0, 0.5]")

    payload = torch.load(args.base, map_location="cpu", weights_only=False)
    policy = SharedMagneticPolicy().eval()
    policy.load_state_dict(payload["model"], strict=True)
    model = SymmetryEnsemble(policy, args.mirror_weight).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "left/right inference symmetry ensemble",
        "base": str(args.base.resolve()),
        "mirror_weight": args.mirror_weight,
        "input_dim": INPUT_DIM,
        "output_dim": 29,
        "exact_equivariance": args.mirror_weight == 0.5,
    }
    # Keep the underlying trainable policy state for reproducibility.  Runtime
    # uses the exported ensemble ONNX described by the extra metadata.
    torch.save(
        {
            "model": policy.state_dict(),
            "metrics": manifest,
            "input_dim": INPUT_DIM,
            "runtime_wrapper": manifest,
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
