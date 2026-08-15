#!/usr/bin/env python3
"""Train a deployable forward-speed monitor from Hall history + proprioception.

Training labels (``root_lin_vel_b``) are collected by
``scripts/rsl_rl/eval_friction_matrix.py --collect_npz``.  They are never
included in the 1864-D input used at inference time.  The resulting estimator
is therefore suitable as a conservative monitoring or command-governor input
on a robot that has IMU, joint encoders, commands, actions, and the two Hall
arrays, but no external motion-capture speed measurement.

Example:
  python scripts/traction/train_forward_velocity_estimator.py \
      --train artifacts/.../train.npz --test artifacts/.../test.npz \
      --output-dir artifacts/.../forward_velocity_estimator
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.traction.forward_velocity_estimator import (  # noqa: E402
    ForwardVelocityEstimator,
    NormalizedForwardVelocityEstimator,
)


DEPLOY_OBSERVATION_DIM = 1864
HALL_START = 480
HALL_END = 1830
VALID_START = 1860
VALID_END = 1862


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(384, 192, 64))
    parser.add_argument("--output-clip", type=float, default=2.0)
    parser.add_argument("--hall-dropout-prob", type=float, default=0.04)
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Keep post-reset/fall samples; off by default for a clean speed target.",
    )
    return parser.parse_args()


def _load_paths(paths: list[Path], include_invalid: bool) -> dict[str, np.ndarray]:
    records: dict[str, list[np.ndarray]] = {
        "obs": [],
        "target": [],
        "mu": [],
        "cmd_vx": [],
        "contact_slip": [],
    }
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            required = ("obs", "root_lin_vel_b")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path} lacks {missing}; recollect using the current "
                    "eval_friction_matrix.py --collect_npz"
                )
            observation = np.asarray(data["obs"], dtype=np.float32)
            velocity = np.asarray(data["root_lin_vel_b"], dtype=np.float32)
            if observation.ndim != 2 or observation.shape[1] != DEPLOY_OBSERVATION_DIM:
                raise ValueError(
                    f"{path}: expected Nx{DEPLOY_OBSERVATION_DIM} observation, "
                    f"got {observation.shape}"
                )
            if velocity.shape != (len(observation), 3):
                raise ValueError(
                    f"{path}: root_lin_vel_b must be Nx3, got {velocity.shape}"
                )
            finite = np.isfinite(observation).all(axis=1) & np.isfinite(velocity).all(axis=1)
            if not include_invalid and "valid" in data:
                finite &= np.asarray(data["valid"], dtype=bool)
            if not include_invalid and "fall" in data:
                finite &= ~np.asarray(data["fall"], dtype=bool)
            if not finite.any():
                raise ValueError(f"{path}: no valid finite samples")
            records["obs"].append(observation[finite])
            records["target"].append(velocity[finite, 0])
            records["mu"].append(
                np.asarray(data["mu"], dtype=np.float32)[finite]
                if "mu" in data
                else np.full(finite.sum(), np.nan, dtype=np.float32)
            )
            records["cmd_vx"].append(
                np.asarray(data["cmd_vx"], dtype=np.float32)[finite]
                if "cmd_vx" in data
                else np.full(finite.sum(), np.nan, dtype=np.float32)
            )
            records["contact_slip"].append(
                np.asarray(data["contact_slip"], dtype=np.float32)[finite]
                if "contact_slip" in data
                else np.full(finite.sum(), np.nan, dtype=np.float32)
            )
    return {key: np.concatenate(value, axis=0) for key, value in records.items()}


def _normalization(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = observation.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = observation.std(axis=0, dtype=np.float64).astype(np.float32)
    # Constants (for example a command axis in one data collection) should
    # remain usable instead of producing division by zero at deployment.
    scale = np.maximum(scale, 1.0e-3)
    return mean, scale


def _augment(
    normalized: torch.Tensor,
    hall_dropout_prob: float,
    feature_noise_std: float,
) -> torch.Tensor:
    if hall_dropout_prob <= 0.0 and feature_noise_std <= 0.0:
        return normalized
    result = normalized.clone()
    if feature_noise_std > 0.0:
        result.add_(torch.randn_like(result) * feature_noise_std)
    if hall_dropout_prob > 0.0:
        batch = result.shape[0]
        # Drop an entire foot's time history rather than independent scalar
        # entries.  This matches a BLE packet/foot outage more closely.
        foot_keep = (
            torch.rand((batch, 2), device=result.device) >= hall_dropout_prob
        ).to(result.dtype)
        hall = result[:, HALL_START:HALL_END].reshape(batch, 15, 2, 15, 3)
        hall.mul_(foot_keep[:, None, :, None, None])
        # The last four entries are L/R valid and normalized sample age.  Only
        # validity is altered here; age still carries timing information.
        validity = result[:, VALID_START:VALID_END]
        validity.mul_(foot_keep)
    return result


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(np.square(error)))
    variance = float(np.var(target))
    return {
        "mae_mps": float(np.mean(np.abs(error))),
        "rmse_mps": math.sqrt(mse),
        "p95_abs_error_mps": float(np.quantile(np.abs(error), 0.95)),
        "bias_mps": float(np.mean(error)),
        "r2": float(1.0 - mse / max(variance, 1.0e-8)),
    }


def _predict(
    model: nn.Module, observation: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    prediction: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            tensor = torch.from_numpy(observation[start : start + batch_size]).to(device)
            prediction.append(model(tensor).cpu().numpy())
    return np.concatenate(prediction, axis=0)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.hall_dropout_prob < 1.0:
        raise ValueError("hall-dropout-prob must be in [0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train = _load_paths(args.train, args.include_invalid)
    test = _load_paths(args.test, args.include_invalid)
    mean, scale = _normalization(train["obs"])
    train_x = (train["obs"] - mean) / scale
    test_x = (test["obs"] - mean) / scale
    train_y = train["target"]
    test_y = test["target"]

    core = ForwardVelocityEstimator(
        DEPLOY_OBSERVATION_DIM, tuple(args.hidden_dims), args.output_clip
    ).to(device)
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.10)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        core.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_x, batch_y in loader:
            batch_x = _augment(
                batch_x.to(device), args.hall_dropout_prob, args.feature_noise_std
            )
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(core(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_x)
            sample_count += len(batch_x)

        runtime = NormalizedForwardVelocityEstimator(core, mean, scale).to(device)
        test_prediction = _predict(runtime, test["obs"], args.batch_size, device)
        metrics = _metrics(test_prediction, test_y)
        metrics["epoch"] = float(epoch)
        metrics["train_smooth_l1"] = loss_sum / max(sample_count, 1)
        history.append(metrics)
        if metrics["mae_mps"] < best_mae:
            best_mae = metrics["mae_mps"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in core.state_dict().items()
            }
        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_loss={metrics['train_smooth_l1']:.5f} "
                f"test_mae={metrics['mae_mps']:.4f} "
                f"test_p95={metrics['p95_abs_error_mps']:.4f} "
                f"r2={metrics['r2']:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    core.load_state_dict(best_state)
    runtime = NormalizedForwardVelocityEstimator(core, mean, scale).to(device).eval()
    train_metrics = _metrics(_predict(runtime, train["obs"], args.batch_size, device), train_y)
    test_metrics = _metrics(_predict(runtime, test["obs"], args.batch_size, device), test_y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "hall_forward_velocity_estimator/v1",
        "input_dim": DEPLOY_OBSERVATION_DIM,
        "feature_indices": np.arange(DEPLOY_OBSERVATION_DIM, dtype=np.int64),
        "mean": mean,
        "scale": scale,
        "hidden_dims": tuple(int(value) for value in args.hidden_dims),
        "output_clip": float(args.output_clip),
        "model": core.cpu().state_dict(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_files": [str(path) for path in args.train],
        "test_files": [str(path) for path in args.test],
        "seed": args.seed,
        "input_contract": (
            "1864-D deploy observation only; no true friction, contact force, "
            "or simulator root velocity is permitted at inference"
        ),
    }
    checkpoint = args.output_dir / "forward_velocity_estimator.pt"
    torch.save(payload, checkpoint)
    runtime.cpu().eval()
    onnx_path = args.output_dir / "forward_velocity_estimator.onnx"
    torch.onnx.export(
        runtime,
        torch.zeros((2, DEPLOY_OBSERVATION_DIM), dtype=torch.float32),
        onnx_path,
        input_names=["observation"],
        output_names=["forward_velocity_mps"],
        dynamic_axes={"observation": {0: "batch"}, "forward_velocity_mps": {0: "batch"}},
        opset_version=17,
    )
    report = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_validation_mae_mps": best_mae,
        "history": history,
        "quality_gate": {
            "mae_mps_le_0_10": bool(test_metrics["mae_mps"] <= 0.10),
            "p95_abs_error_mps_le_0_25": bool(
                test_metrics["p95_abs_error_mps"] <= 0.25
            ),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
