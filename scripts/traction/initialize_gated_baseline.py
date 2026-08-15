#!/usr/bin/env python3
"""Create a canonical 1590-D deploy package checkpoint that is exactly baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.networks import GatedTractionPolicy  # noqa: E402
from unitree_rl_lab.traction.schema import (  # noqa: E402
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline_checkpoint",
        type=Path,
        default=ROOT / "model" / "rl" / "model_49999.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.baseline_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    policy = GatedTractionPolicy()
    actor_state = {
        key: value
        for key, value in checkpoint["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    policy.baseline_actor.load_state_dict(actor_state, strict=True)
    policy.eval()

    history = torch.randn(
        8,
        TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
        TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
    )
    history[..., 102:104] = 1.0
    history[..., 104:106] = 0.0
    from unitree_rl_lab.traction.networks import temporal_history_to_legacy_proprio

    baseline_observation = temporal_history_to_legacy_proprio(history)
    with torch.inference_mode():
        expected = policy.baseline_actor(baseline_observation)
        actual = policy(
            baseline_observation,
            history,
            history[:, -1, 93:96],
        ).action_mean
    maximum_error = float((actual - expected).abs().max())
    if maximum_error != 0.0:
        raise RuntimeError(
            f"zero-initialized traction residual changed baseline by {maximum_error}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student_policy_state_dict": policy.state_dict(),
        "epoch": -1,
        "seed": args.seed,
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "schema_version": TEMPORAL_STUDENT_FRAME_SCHEMA.schema_version,
        "metrics": {
            "baseline_action_max_abs_error": maximum_error,
            "traction_residual_initialization": "exact_zero",
        },
    }
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline_action_max_abs_error": maximum_error,
                "action_dimension": 29,
                "student_dimension": TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
