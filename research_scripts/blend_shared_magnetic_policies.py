#!/usr/bin/env python3
"""Linearly blend two compatible shared-magnetic checkpoints and export ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_shared_magnetic_policy import INPUT_DIM, SharedMagneticPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=False)
    base_state = base["model"]
    tuned_state = tuned["model"]
    if base_state.keys() != tuned_state.keys():
        raise ValueError("checkpoint parameter keys differ")
    blended = {}
    for name in base_state:
        left, right = base_state[name], tuned_state[name]
        if left.shape != right.shape:
            raise ValueError(f"{name}: shape mismatch")
        blended[name] = torch.lerp(left, right, args.alpha)
    model = SharedMagneticPolicy().eval()
    model.load_state_dict(blended, strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "linear policy blend",
        "base": str(args.base.resolve()),
        "tuned": str(args.tuned.resolve()),
        "alpha_tuned": args.alpha,
        "input_dim": INPUT_DIM,
        "output_dim": 29,
    }
    torch.save(
        {"model": model.state_dict(), "metrics": manifest, "input_dim": INPUT_DIM},
        args.output_dir / "shared_magnetic_policy.pt",
    )
    torch.onnx.export(
        model,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
