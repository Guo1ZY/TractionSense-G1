#!/usr/bin/env python3
"""Prepare a same-shape Teacher checkpoint for motion-feedback continuation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    actor = payload["actor_state_dict"]
    first_layer = actor["mlp.0.weight"]
    if tuple(first_layer.shape) != (512, 641):
        raise ValueError(f"unexpected first layer: {tuple(first_layer.shape)}")

    # Columns 630:635 previously saw nominal validity=1 and 635:640 nominal
    # age=0. Fold that nominal contribution into the bias before clearing the
    # columns, so the derived policy is exactly behavior-preserving at the
    # start of motion-feedback training.
    actor["mlp.0.bias"].add_(first_layer[:, 630:635].sum(dim=1))
    first_layer[:, 630:640].zero_()
    payload["optimizer_state_dict"] = None
    infos = dict(payload.get("infos") or {})
    infos.update(
        {
            "derived_from": str(args.checkpoint.expanduser().resolve()),
            "motion_feedback_columns": [630, 640],
            "motion_feedback": [
                "body_lateral_velocity",
                "relative_heading_error",
            ],
        }
    )
    payload["infos"] = infos

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
