#!/usr/bin/env python3
"""Aggregate Student-visited Isaac rollouts for PPO-Teacher DAgger labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [np.load(path, allow_pickle=True) for path in args.datasets]
    if not sources:
        raise ValueError("at least one rollout is required")
    metadata = [source["metadata"].item() for source in sources]
    steps = int(metadata[0]["steps"])
    if any(int(item["steps"]) != steps for item in metadata):
        raise ValueError("DAgger rollouts must use the same step count")
    required = {"student_history", "teacher_history", "true_force_local_n", "true_contact"}
    if any(not required.issubset(source.files) for source in sources):
        raise KeyError(f"each DAgger rollout requires {sorted(required)}")
    keys = set(sources[0].files) - {"metadata"}
    if any((set(source.files) - {"metadata"}) != keys for source in sources):
        raise ValueError("DAgger rollout schemas differ")
    output: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [np.asarray(source[key]) for source in sources]
        if any(array.shape[0] != steps or array.ndim < 2 for array in arrays):
            raise ValueError(f"{key} is not a [steps, environments, ...] tensor")
        output[key] = np.concatenate(arrays, axis=1)
    total_envs = int(output["student_history"].shape[1])
    output["environment_id"] = np.broadcast_to(
        np.arange(total_envs, dtype=output["environment_id"].dtype)[None, :, None],
        (steps, total_envs, 1),
    ).copy()
    aggregate_metadata = {
        "kind": "dagger_student_visited_rollout_aggregate",
        "steps": steps,
        "num_envs": total_envs,
        "sample_count": steps * total_envs,
        "seeds": [int(item["seed"]) for item in metadata],
        "student_rollout_checkpoints": [item["checkpoint"] for item in metadata],
        "teacher_history_is_policy_input": False,
        "teacher_history_role": "privileged label generation only",
        "sources": [str(path.resolve()) for path in args.datasets],
    }
    output["metadata"] = np.asarray(aggregate_metadata, dtype=object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    report = {
        "output": str(args.output.resolve()),
        "steps": steps,
        "environments": total_envs,
        "samples": steps * total_envs,
        "seeds": aggregate_metadata["seeds"],
        "teacher_history_is_policy_input": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
