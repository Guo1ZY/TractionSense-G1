#!/usr/bin/env python3
"""Export the audited 480-D original G1 actor behind the 1864-D interface.

The wrapper intentionally slices only ``observation[:, :480]``.  Hall field,
sampling-period and health channels are ignored, making this an exact
proprioceptive baseline for paired Isaac comparisons with the magnetic policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from unitree_rl_lab.traction.layout_magnetic_student import INPUT_DIM
from unitree_rl_lab.traction.networks import LegacyLocomotionActor


class ProprioBaseline1864(nn.Module):
    """Compatibility wrapper that has no path from Hall input to action."""

    def __init__(self, actor: LegacyLocomotionActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation[:, :480])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/rl/model_49999.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{args.checkpoint} has no actor_state_dict")
    mlp_state = {
        key: value for key, value in state.items() if key.startswith("mlp.")
    }
    actor = LegacyLocomotionActor(480).eval()
    actor.load_state_dict(mlp_state, strict=True)
    wrapper = ProprioBaseline1864(actor).eval()

    generator = torch.Generator().manual_seed(args.seed)
    sample = torch.randn(3, INPUT_DIM, generator=generator) * 0.25
    with torch.inference_mode():
        reference = wrapper(sample).numpy()
        hall_changed = sample.clone()
        hall_changed[:, 480:] = 10.0 * torch.randn(
            hall_changed[:, 480:].shape, generator=generator
        )
        hall_invariance_error = float(
            torch.max(torch.abs(wrapper(sample) - wrapper(hall_changed))).item()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample[:1],
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: sample.numpy()},
    )[0]
    parity_error = float(np.max(np.abs(reference - actual)))
    finite = bool(np.isfinite(actual).all())
    passed = finite and parity_error <= 2.0e-5 and hall_invariance_error == 0.0
    report = {
        "policy": "original_proprioceptive_baseline",
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "input_dim": INPUT_DIM,
        "consumed_input_slice": [0, 480],
        "ignored_input_slice": [480, INPUT_DIM],
        "output_dim": 29,
        "hall_to_action_path": False,
        "hall_invariance_max_abs_error": hall_invariance_error,
        "onnx_parity_max_abs_error": parity_error,
        "finite": finite,
        "status": "PASS" if passed else "FAIL",
    }
    manifest = args.output.with_suffix(".json")
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("proprio baseline export validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
