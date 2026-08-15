#!/usr/bin/env python3
"""Actual model_49999 480/495 -> 510/525 behavior-equivalence regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.schema import (  # noqa: E402
    legacy_actor_schema,
    legacy_critic_schema,
    old_to_new_flat_index,
)
from unitree_rl_lab.utils.partial_checkpoint import merge_state_dicts  # noqa: E402


def _expanded_template(
    state: dict[str, torch.Tensor],
    input_dimension: int,
) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in state.items()}
    weight = result["mlp.0.weight"]
    result["mlp.0.weight"] = torch.empty(
        (weight.shape[0], input_dimension),
        dtype=weight.dtype,
        device=weight.device,
    )
    return result


def _mlp(state: dict[str, torch.Tensor], observation: torch.Tensor) -> torch.Tensor:
    x = observation
    for index in (0, 2, 4):
        x = F.elu(
            F.linear(
                x,
                state[f"mlp.{index}.weight"],
                state[f"mlp.{index}.bias"],
            )
        )
    return F.linear(x, state["mlp.6.weight"], state["mlp.6.bias"])


def _regress(
    state: dict[str, torch.Tensor],
    old_schema,
    new_schema,
    *,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, object]]:
    template = _expanded_template(state, new_schema.flat_dimension)
    merged, stats = merge_state_dicts(template, state, verbose=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    old_observation = torch.randn(
        (batch_size, old_schema.flat_dimension),
        generator=generator,
        dtype=state["mlp.0.weight"].dtype,
    )
    mapping = torch.as_tensor(
        old_to_new_flat_index(old_schema, new_schema),
        dtype=torch.long,
    )
    new_observation = torch.zeros(
        (batch_size, new_schema.flat_dimension),
        dtype=old_observation.dtype,
    )
    new_observation[:, mapping] = old_observation
    old_output = _mlp(state, old_observation)
    new_output = _mlp(merged, new_observation)
    error = (new_output - old_output).abs()
    return (
        {
            "maximum_absolute_error": float(error.max().item()),
            "mean_absolute_error": float(error.mean().item()),
        },
        stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "model" / "rl" / "model_49999.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "checkpoint_migration_20260731.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    actor_error, actor_stats = _regress(
        checkpoint["actor_state_dict"],
        legacy_actor_schema(include_force=False),
        legacy_actor_schema(include_force=True),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    critic_error, critic_stats = _regress(
        checkpoint["critic_state_dict"],
        legacy_critic_schema(include_force=False),
        legacy_critic_schema(include_force=True),
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": int(checkpoint.get("iter", -1)),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "actor_action_mean_error": actor_error,
        "critic_value_error": critic_error,
        "actor_load_stats": actor_stats,
        "critic_load_stats": critic_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    # A 480-wide and a 510-wide float32 GEMM may accumulate in a different
    # order even though the extra columns and values are exactly zero.
    assert actor_error["maximum_absolute_error"] <= 5.0e-5
    assert critic_error["maximum_absolute_error"] <= 2.0e-4
    assert actor_stats["mapping"][0]["source"] == "canonical_schema"
    assert critic_stats["mapping"][0]["source"] == "canonical_schema"
    assert actor_stats["new_initialized"][0]["count"] == 30
    assert critic_stats["new_initialized"][0]["count"] == 30


if __name__ == "__main__":
    main()
