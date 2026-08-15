#!/usr/bin/env python3
"""Export a torque-gated selector between the safe and official G1 actors.

The selector remains a pure-proprioception deployment policy: its only input is
the 915-D causal observation used by the torque student.  The learned friction
head gates the unchanged 480-D official actor only for sufficiently large
forward commands and sufficiently strong high-traction evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_proprio_traction_policy import INPUT_DIM, ProprioTractionPolicy  # noqa: E402


OFFICIAL_INPUT_DIM = 480
COMMAND_X_INDEX = 42


def load_official_actor(path: Path) -> nn.Sequential:
    """Rebuild the stock actor from its ONNX initializers without editing it."""
    graph = onnx.load(path).graph
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    prefixes = ("actor.0", "actor.2", "actor.4", "actor.6")
    layers: list[nn.Module] = []
    for index, prefix in enumerate(prefixes):
        weight = torch.from_numpy(np.array(tensors[f"{prefix}.weight"], copy=True))
        bias = torch.from_numpy(np.array(tensors[f"{prefix}.bias"], copy=True))
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
        if index != len(prefixes) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).eval()
    if actor(torch.zeros(1, OFFICIAL_INPUT_DIM)).shape != (1, 29):
        raise ValueError(f"unexpected official actor dimensions in {path}")
    return actor


class TorqueGatedPolicySelector(nn.Module):
    """Smoothly select the stock fast gait only after traction is observable."""

    def __init__(
        self,
        safe_policy: ProprioTractionPolicy,
        official_actor: nn.Module,
        mu_center: float,
        mu_sharpness: float,
        command_center: float,
        command_sharpness: float,
        maximum_fast_blend: float,
    ) -> None:
        super().__init__()
        self.safe_policy = safe_policy
        self.official_actor = official_actor
        self.mu_center = mu_center
        self.mu_sharpness = mu_sharpness
        self.command_center = command_center
        self.command_sharpness = command_sharpness
        self.maximum_fast_blend = maximum_fast_blend

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        safe_action = self.safe_policy(observation)
        fast_action = self.official_actor(observation[:, :OFFICIAL_INPUT_DIM])
        estimated_mu = self.safe_policy.predict_mu(observation)
        command = observation[:, COMMAND_X_INDEX]
        traction_gate = torch.sigmoid(
            self.mu_sharpness * (estimated_mu - self.mu_center)
        )
        command_gate = torch.sigmoid(
            self.command_sharpness * (command - self.command_center)
        )
        blend = (
            self.maximum_fast_blend * traction_gate * command_gate
        ).unsqueeze(-1)
        return safe_action + blend * (fast_action - safe_action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-checkpoint", type=Path, required=True)
    parser.add_argument("--official-onnx", type=Path, required=True)
    parser.add_argument("--source-slot", type=Path, required=True)
    parser.add_argument("--install-slot", type=Path, required=True)
    parser.add_argument("--mu-center", type=float, default=0.28)
    parser.add_argument("--mu-sharpness", type=float, default=20.0)
    parser.add_argument("--command-center", type=float, default=0.60)
    parser.add_argument("--command-sharpness", type=float, default=18.0)
    parser.add_argument("--maximum-fast-blend", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.safe_checkpoint, map_location="cpu", weights_only=False)
    safe_policy = ProprioTractionPolicy(payload["mean"], payload["scale"])
    safe_policy.load_state_dict(payload["model"], strict=True)
    safe_policy.eval()
    official_actor = load_official_actor(args.official_onnx)
    selector = TorqueGatedPolicySelector(
        safe_policy=safe_policy,
        official_actor=official_actor,
        mu_center=args.mu_center,
        mu_sharpness=args.mu_sharpness,
        command_center=args.command_center,
        command_sharpness=args.command_sharpness,
        maximum_fast_blend=args.maximum_fast_blend,
    ).eval()

    exported = args.install_slot / "exported"
    params = args.install_slot / "params"
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, INPUT_DIM), dtype=torch.float32)
    torch.onnx.export(
        selector,
        example,
        exported / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )
    torch.jit.trace(selector, example).save(str(exported / "policy.ts"))
    shutil.copy2(
        args.source_slot / "exported/friction_estimator.onnx",
        exported / "friction_estimator.onnx",
    )
    shutil.copy2(args.source_slot / "params/deploy.yaml", params / "deploy.yaml")
    (args.install_slot / "checkpoint.txt").write_text(
        str(args.safe_checkpoint.resolve()) + "\n", encoding="utf-8"
    )
    manifest = {
        "method": "torque-estimated-friction smooth policy selector",
        "input_dim": INPUT_DIM,
        "safe_checkpoint": str(args.safe_checkpoint.resolve()),
        "official_actor_unchanged": str(args.official_onnx.resolve()),
        "mu_center": args.mu_center,
        "mu_sharpness": args.mu_sharpness,
        "command_center": args.command_center,
        "command_sharpness": args.command_sharpness,
        "maximum_fast_blend": args.maximum_fast_blend,
        "foot_sensor_required": False,
        "magnetic_sensor_required": False,
    }
    (args.install_slot / "install_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
