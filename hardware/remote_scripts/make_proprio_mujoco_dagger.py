#!/usr/bin/env python3
"""Create a capture slot and convert its MuJoCo observations to DAgger pairs.

The capture policy receives 915 torque-only inputs plus the exact 160-D foot
bridge stream, but its action path deliberately slices and uses only the first
915 values.  The appended stream is recorded solely to construct a same-state
641-D frozen-Teacher label after the rollout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_proprio_traction_policy import INPUT_DIM, ProprioTractionPolicy  # noqa: E402

FOOT_DIM = 160
CAPTURE_DIM = INPUT_DIM + FOOT_DIM


class CapturePolicy(nn.Module):
    def __init__(self, policy: ProprioTractionPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy(observation[:, :INPUT_DIM])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("make-slot")
    make.add_argument("--checkpoint", type=Path, required=True)
    make.add_argument("--source-slot", type=Path, required=True)
    make.add_argument("--teacher-template", type=Path, required=True)
    make.add_argument("--capture-slot", type=Path, required=True)
    convert = subparsers.add_parser("convert")
    convert.add_argument("--capture-npz", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_slot(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ProprioTractionPolicy(payload["mean"], payload["scale"])
    policy.load_state_dict(payload["model"], strict=True)
    model = CapturePolicy(policy.eval()).eval()
    args.capture_slot.mkdir(parents=True, exist_ok=True)
    (args.capture_slot / "exported").mkdir(exist_ok=True)
    (args.capture_slot / "params").mkdir(exist_ok=True)
    example = torch.zeros((1, CAPTURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        args.capture_slot / "exported/policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )

    source_text = (args.source_slot / "params/deploy.yaml").read_text(encoding="utf-8").rstrip()
    teacher_text = args.teacher_template.read_text(encoding="utf-8")
    if "\n  foot_contact:" not in teacher_text or "\n  effective_friction_mu:" not in teacher_text:
        raise ValueError("teacher template lacks canonical foot/effective-mu markers")
    foot_tail = "  foot_contact:" + teacher_text.split("\n  foot_contact:", 1)[1]
    foot_tail = foot_tail.split("\n  effective_friction_mu:", 1)[0].rstrip()
    deploy = source_text + "\n" + foot_tail + "\n"
    (args.capture_slot / "params/deploy.yaml").write_text(deploy, encoding="utf-8")
    (args.capture_slot / "checkpoint.txt").write_text(
        str(args.checkpoint.resolve()) + "\n", encoding="utf-8"
    )
    (args.capture_slot / "install_manifest.json").write_text(
        json.dumps(
            {
                "method": "evaluation-only same-state MuJoCo DAgger capture",
                "input_dim": CAPTURE_DIM,
                "action_input_slice": [0, INPUT_DIM],
                "record_only_foot_slice": [INPUT_DIM, CAPTURE_DIM],
                "real_deploy": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def convert(args: argparse.Namespace) -> None:
    with np.load(args.capture_npz) as data:
        capture = np.asarray(data["obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        command = np.asarray(data["cmd_vx"], dtype=np.float32).reshape(-1)
        wall_time = np.asarray(data.get("wall_time", np.arange(len(mu))), dtype=np.float64)
    if capture.ndim != 2 or capture.shape[1] != CAPTURE_DIM:
        raise ValueError(f"expected Nx{CAPTURE_DIM} capture data, got {capture.shape}")
    # Teacher layout is its original 480-D base, the exact captured 160-D
    # foot bridge, then simulation-only true mu.
    teacher = np.concatenate(
        (capture[:, :480], capture[:, INPUT_DIM:CAPTURE_DIM], mu[:, None]), axis=1
    ).astype(np.float32)
    if teacher.shape[1] != 641:
        raise AssertionError(teacher.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=capture[:, :INPUT_DIM].astype(np.float32),
        teacher_obs=teacher,
        mu=mu,
        cmd_vx=command,
        wall_time=wall_time,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "samples": int(len(mu)),
                "policy_dim": INPUT_DIM,
                "teacher_dim": int(teacher.shape[1]),
                "finite": bool(np.isfinite(teacher).all()),
            },
            indent=2,
        )
    )


def main() -> int:
    args = parse_args()
    if args.command == "make-slot":
        make_slot(args)
    else:
        convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
