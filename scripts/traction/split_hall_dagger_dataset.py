#!/usr/bin/env python3
"""Merge and stratify Hall-only DAgger trajectories for reproducible training.

The 641-D privileged Teacher observation and friction value remain offline
labels.  Only ``obs`` is a deployment input; it must match the audited
1864-D Hall Student schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = ("obs", "teacher_obs", "mu", "cmd_vx")
OPTIONAL = (
    "sample_weight",
    "seed",
    "step",
    "fall",
    "recovery",
    "phase",
    "time_since_switch_s",
    "env_id",
)
HALL_SLICE = slice(480, 1830)
TRAILING_FEATURE_MODES = ("sensor_age", "motion_feedback")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        result = {
            key: np.asarray(data[key])
            for key in (*REQUIRED, *OPTIONAL)
            if key in data
        }
    count = len(result["obs"])
    if result["obs"].shape != (count, 1864):
        raise ValueError(f"{path}: obs must be [N,1864], got {result['obs'].shape}")
    if result["teacher_obs"].shape != (count, 641):
        raise ValueError(
            f"{path}: teacher_obs must be [N,641], got {result['teacher_obs'].shape}"
        )
    for key, value in result.items():
        if len(value) != count:
            raise ValueError(f"{path}: {key} has {len(value)} rows, expected {count}")
    if not np.isfinite(result["obs"]).all() or not np.isfinite(
        result["teacher_obs"]
    ).all():
        raise ValueError(f"{path}: non-finite observation")
    if "sample_weight" not in result:
        result["sample_weight"] = np.ones(count, dtype=np.float32)
    return result


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(REQUIRED) | {"sample_weight"}
    keys.update(set.intersection(*(set(part) for part in parts)))
    return {key: np.concatenate([part[key] for part in parts]) for key in sorted(keys)}


def stratified_indices(
    mu: np.ndarray,
    command: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    rng = np.random.default_rng(seed)
    # Rounding protects bins from harmless float32 serialization noise.
    labels = np.stack((np.round(mu, 4), np.round(command, 4)), axis=1)
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for label in np.unique(labels, axis=0):
        indices = np.flatnonzero(np.all(labels == label, axis=1))
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        if len(indices) > 1:
            test_count = min(test_count, len(indices) - 1)
        test.append(indices[:test_count])
        train.append(indices[test_count:])
    train_index = np.concatenate(train)
    test_index = np.concatenate(test)
    rng.shuffle(train_index)
    rng.shuffle(test_index)
    return train_index, test_index


def grouped_indices(
    source_id: np.ndarray,
    seed_value: np.ndarray,
    env_id: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold complete simulator environments out to prevent temporal leakage."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    groups = np.stack((source_id, seed_value, env_id), axis=1)
    unique_groups, inverse = np.unique(groups, axis=0, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("group split requires at least two source/seed/environment groups")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_groups))
    test_groups = max(1, int(round(len(order) * test_fraction)))
    test_groups = min(test_groups, len(order) - 1)
    is_test = np.isin(inverse, order[:test_groups])
    train_index = np.flatnonzero(~is_test)
    test_index = np.flatnonzero(is_test)
    rng.shuffle(train_index)
    rng.shuffle(test_index)
    return train_index, test_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--trailing-feature-mode",
        choices=TRAILING_FEATURE_MODES,
        required=True,
        help=(
            "Semantic meaning of obs[1862:1864]. This is mandatory because "
            "sensor_age and motion_feedback have the same shape."
        ),
    )
    args = parser.parse_args()

    parts = [load(path) for path in args.input]
    for source_index, part in enumerate(parts):
        part["source_id"] = np.full(len(part["obs"]), source_index, dtype=np.int16)
    merged = concatenate(parts)
    if "env_id" in merged and "seed" in merged:
        train_index, test_index = grouped_indices(
            merged["source_id"].reshape(-1),
            merged["seed"].reshape(-1),
            merged["env_id"].reshape(-1),
            args.test_fraction,
            args.seed,
        )
        split_strategy = "complete source/seed/environment holdout"
    else:
        train_index, test_index = stratified_indices(
            merged["mu"].reshape(-1),
            merged["cmd_vx"].reshape(-1),
            args.test_fraction,
            args.seed,
        )
        split_strategy = "row stratified by mu and command (legacy data lacks env_id)"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.npz"
    test_path = args.output_dir / "test.npz"
    np.savez_compressed(train_path, **{key: value[train_index] for key, value in merged.items()})
    np.savez_compressed(test_path, **{key: value[test_index] for key, value in merged.items()})

    hall = merged["obs"][:, HALL_SLICE]
    trailing_values = merged["obs"][:, 1862:1864].mean(axis=0).tolist()
    trailing_summary = (
        {"age_mean_lr": trailing_values}
        if args.trailing_feature_mode == "sensor_age"
        else {
            "body_vy_mean": float(trailing_values[0]),
            "relative_heading_mean": float(trailing_values[1]),
        }
    )
    manifest = {
        "measurement_boundary": "deployment obs contains Hall + proprioception only; Teacher and mu are offline labels",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_strategy": split_strategy,
        "samples": int(len(hall)),
        "train_samples": int(len(train_index)),
        "test_samples": int(len(test_index)),
        "mu_values": sorted(float(value) for value in np.unique(merged["mu"])),
        "cmd_vx_values": sorted(float(value) for value in np.unique(merged["cmd_vx"])),
        "hall_abs_mean": float(np.abs(hall).mean()),
        "hall_std": float(hall.std()),
        "hall_abs_max": float(np.abs(hall).max()),
        "valid_mean_lr": merged["obs"][:, 1860:1862].mean(axis=0).tolist(),
        "trailing_feature_mode": args.trailing_feature_mode,
        "trailing_feature_summary": trailing_summary,
        "sample_weight_range": [
            float(merged["sample_weight"].min()),
            float(merged["sample_weight"].max()),
        ],
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.input
        ],
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256(train_path)},
            "test": {"path": str(test_path.resolve()), "sha256": sha256(test_path)},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
