#!/usr/bin/env python3
"""Build Teacher-labeled DAgger inputs from recorded MuJoCo magnetic rollouts.

The MuJoCo bridge uses the same deterministic 15xXYZ proxy as Isaac.  This
allows the normalized magnetic frames to be inverted back to the compact foot
force history expected by the 641-D oracle Teacher, without exposing oracle
friction to the magnetic student's 1864-D input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_DIM = 480
HISTORY = 15
FEET = 2
SENSORS = 15
AXES = 3
MAGNETIC_DIM = HISTORY * FEET * SENSORS * AXES
INPUT_DIM = 1864
TEACHER_DIM = 641

PROFILE = np.asarray(
    [
        0.70, 0.76, 0.70,
        0.76, 0.82, 0.76,
        0.82, 0.88, 0.82,
        0.88, 0.94, 0.88,
        0.94, 1.00, 0.94,
    ],
    dtype=np.float32,
)
PROFILE /= PROFILE.mean()
MIXING = np.asarray(
    [
        [0.14, 1.00],
        [-0.10, 0.42],
        [1.00, 0.12],
    ],
    dtype=np.float32,
)
MIXING_PINV = np.linalg.pinv(MIXING).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", required=True, type=Path)
    parser.add_argument("--isaac-train", type=Path)
    parser.add_argument("--isaac-test", type=Path)
    parser.add_argument("--isaac-train-limit", type=int, default=12000)
    parser.add_argument("--isaac-test-limit", type=int, default=6000)
    parser.add_argument("--mujoco-repeat", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=8110)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def invert_magnetic(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    magnetic = observation[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
        -1, HISTORY, FEET, SENSORS, AXES
    )
    clipped = np.clip(magnetic / 5.0, -0.999999, 0.999999)
    signal = 5.0 * np.arctanh(clipped)
    per_sensor = signal / PROFILE.reshape(1, 1, 1, SENSORS, 1)
    axis_signal = per_sensor.mean(axis=3)
    forces = np.einsum("ca,ntfa->ntfc", MIXING_PINV, axis_signal, optimize=True)
    forces = np.clip(forces, 0.0, 5.0).astype(np.float32)
    normal = forces[..., 0]
    tangent = forces[..., 1]

    reconstructed_signal = (
        PROFILE.reshape(1, 1, 1, SENSORS, 1)
        * (
            normal[..., None, None] * MIXING[:, 0].reshape(1, 1, 1, 1, AXES)
            + tangent[..., None, None] * MIXING[:, 1].reshape(1, 1, 1, 1, AXES)
        )
    )
    reconstructed = 5.0 * np.tanh(reconstructed_signal / 5.0)
    reconstruction_mae = float(np.mean(np.abs(reconstructed - magnetic)))
    return normal, tangent, reconstruction_mae


def teacher_observation(
    observation: np.ndarray, mu: np.ndarray
) -> tuple[np.ndarray, float]:
    normal, tangent, reconstruction_mae = invert_magnetic(observation)
    force_magnitude_n = 100.0 * np.sqrt(normal**2 + tangent**2)
    contact = 1.0 / (1.0 + np.exp(-np.clip(
        (force_magnitude_n - 5.0) * 2.0, -40.0, 40.0
    )))
    ratio = np.clip(tangent / (normal + 0.05), 0.0, 2.0)
    total_normal = normal.sum(axis=2, keepdims=True)
    load = np.where(
        total_normal > 1.0e-6,
        normal / np.maximum(total_normal, 1.0e-6),
        0.5,
    )

    valid_lr = observation[:, 1860:1862]
    age_lr = observation[:, 1862:1864]
    valid = np.min(valid_lr, axis=1, keepdims=True)
    age = np.max(age_lr, axis=1, keepdims=True)
    teacher = np.concatenate(
        (
            observation[:, :BASE_DIM],
            contact.reshape(-1, HISTORY * FEET),
            normal.reshape(-1, HISTORY * FEET),
            tangent.reshape(-1, HISTORY * FEET),
            ratio.reshape(-1, HISTORY * FEET),
            load.reshape(-1, HISTORY * FEET),
            np.repeat(valid, 5, axis=1),
            np.repeat(age, 5, axis=1),
            mu.reshape(-1, 1),
        ),
        axis=1,
    ).astype(np.float32)
    if teacher.shape[1] != TEACHER_DIM:
        raise RuntimeError(f"constructed Teacher shape is {teacher.shape}")
    return teacher, reconstruction_mae


def load_isaac(path: Path, limit: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        result = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in ("obs", "teacher_obs", "mu", "cmd_vx")
        }
    if limit > 0 and len(result["obs"]) > limit:
        indices = rng.choice(len(result["obs"]), size=limit, replace=False)
        result = {key: value[indices] for key, value in result.items()}
    return result


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("obs", "teacher_obs", "mu", "cmd_vx")
    }


def write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be in (0, 0.5)")
    if args.mujoco_repeat < 1:
        raise ValueError("--mujoco-repeat must be positive")
    rng = np.random.default_rng(args.seed)
    with np.load(args.mujoco) as data:
        observation = np.asarray(data["obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        cmd = np.asarray(data["cmd_vx"], dtype=np.float32).reshape(-1)
    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(f"expected MuJoCo Nx{INPUT_DIM}, got {observation.shape}")
    teacher, reconstruction_mae = teacher_observation(observation, mu)

    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    for friction in np.unique(mu):
        for command in np.unique(cmd[np.isclose(mu, friction)]):
            indices = np.flatnonzero(np.isclose(mu, friction) & np.isclose(cmd, command))
            cut = max(1, int(round(len(indices) * (1.0 - args.test_fraction))))
            train_indices.append(indices[:cut])
            test_indices.append(indices[cut:])
    train_index = np.concatenate(train_indices)
    test_index = np.concatenate(test_indices)
    rng.shuffle(train_index)
    rng.shuffle(test_index)

    mujoco_train = {
        "obs": np.repeat(observation[train_index], args.mujoco_repeat, axis=0),
        "teacher_obs": np.repeat(teacher[train_index], args.mujoco_repeat, axis=0),
        "mu": np.repeat(mu[train_index], args.mujoco_repeat, axis=0),
        "cmd_vx": np.repeat(cmd[train_index], args.mujoco_repeat, axis=0),
    }
    mujoco_test = {
        "obs": observation[test_index],
        "teacher_obs": teacher[test_index],
        "mu": mu[test_index],
        "cmd_vx": cmd[test_index],
    }
    train_parts = [mujoco_train]
    test_parts = [mujoco_test]
    if args.isaac_train:
        train_parts.append(load_isaac(args.isaac_train, args.isaac_train_limit, rng))
    if args.isaac_test:
        test_parts.append(load_isaac(args.isaac_test, args.isaac_test_limit, rng))
    train = concatenate(train_parts)
    test = concatenate(test_parts)
    train_shuffle = rng.permutation(len(train["obs"]))
    test_shuffle = rng.permutation(len(test["obs"]))
    train = {key: value[train_shuffle] for key, value in train.items()}
    test = {key: value[test_shuffle] for key, value in test.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(args.output_dir / "train.npz", train)
    write_npz(args.output_dir / "test.npz", test)
    report = {
        "source": str(args.mujoco.resolve()),
        "isaac_train": str(args.isaac_train.resolve()) if args.isaac_train else None,
        "isaac_test": str(args.isaac_test.resolve()) if args.isaac_test else None,
        "mujoco_raw_samples": len(observation),
        "mujoco_train_raw": len(train_index),
        "mujoco_train_repeat": args.mujoco_repeat,
        "mujoco_test": len(test_index),
        "combined_train": len(train["obs"]),
        "combined_test": len(test["obs"]),
        "proxy_reconstruction_mae": reconstruction_mae,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
