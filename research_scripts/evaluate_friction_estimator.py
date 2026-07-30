#!/usr/bin/env python3
"""Train, gate and export a deployable friction estimator from actor observations.

The estimator never sees simulator-only friction at inference.  It consumes
the same causal observation vector as the locomotion actor (legacy 640-D or
dual-foot magnetic 1864-D); exact mu is used only as a supervised
training/evaluation label.  A 641-D noisy-Teacher dataset is also accepted:
its final privileged-mu column is removed before training so target leakage is
impossible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FrictionEstimator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Training/randomization support is approximately mu in [0.04, 1.20].
        return 1.30 * torch.sigmoid(self.net(observation)).squeeze(-1)


class NormalizedFrictionEstimator(nn.Module):
    """Raw-observation wrapper exported to ONNX/TorchScript."""

    def __init__(self, estimator: FrictionEstimator, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.estimator = estimator
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        return self.estimator((raw_observation - self.mean) / self.scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployable friction-estimator evaluation")
    parser.add_argument("--train", type=Path, required=True, help="Isaac training NPZ")
    parser.add_argument("--test", type=Path, required=True, help="Unseen-seed/unseen-mu Isaac NPZ")
    parser.add_argument(
        "--augment",
        type=Path,
        action="append",
        default=[],
        help="Additional labeled training NPZ (repeatable; e.g. Oracle MuJoCo)",
    )
    parser.add_argument("--mujoco", type=Path, help="Optional labeled MuJoCo policy-observation NPZ")
    parser.add_argument(
        "--features",
        choices=["all", "foot", "magnetic", "proprio"],
        default="all",
        help=(
            "Feature subset. For 1864-D observations, 'magnetic' uses only "
            "the Hall history and sample-period channels (480:1860), "
            "excluding command/proprioception and motion-feedback channels."
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--strict", action="store_true", help="Exit 3 unless unseen-data acceptance gates pass"
    )
    return parser.parse_args()


def feature_indices(mode: str, dim: int) -> np.ndarray:
    if dim not in (640, 641, 1864):
        raise ValueError(
            "expected a 640-D student, 641-D teacher, or 1864-D magnetic "
            f"observation, got {dim}"
        )
    if dim == 1864:
        if mode == "magnetic":
            return np.arange(480, 1860)
        if mode == "foot":
            return np.arange(480, 1864)
        if mode == "proprio":
            return np.arange(0, 480)
        return np.arange(1864)
    if mode == "magnetic":
        return np.arange(480, 640)
    if mode == "foot":
        return np.arange(480, 640)
    if mode == "proprio":
        return np.arange(0, 480)
    # 641-D Teacher datasets are canonicalized to their 640-D deploy prefix.
    return np.arange(640)


def load_dataset(path: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        observation = np.asarray(data["obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
    if observation.ndim == 2 and observation.shape[1] == 641:
        # The last Teacher column is the exact label.  Never expose it to the
        # estimator, even in `features=all` mode.
        observation = observation[:, :640]
    if observation.ndim != 2 or observation.shape[0] != mu.shape[0]:
        raise ValueError(f"invalid dataset shapes in {path}: obs={observation.shape}, mu={mu.shape}")
    finite = np.isfinite(observation).all(axis=1) & np.isfinite(mu)
    return observation[finite][:, indices], mu[finite]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    prediction = np.clip(prediction, 0.0, 1.30)
    error = prediction - y
    variance = float(np.sum(np.square(y - np.mean(y))))
    result: dict[str, float | int] = {
        "samples": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(1.0 - np.sum(np.square(error)) / max(variance, 1.0e-9)),
    }
    low = y <= 0.25
    high = y >= 0.75
    extremes = low | high
    if np.any(low):
        result["low_mae"] = float(np.mean(np.abs(error[low])))
        result["low_pred_mean"] = float(np.mean(prediction[low]))
    if np.any(high):
        result["high_mae"] = float(np.mean(np.abs(error[high])))
        result["high_pred_mean"] = float(np.mean(prediction[high]))
    if np.any(extremes):
        predicted_high = prediction[extremes] >= 0.55
        actual_high = high[extremes]
        result["extreme_accuracy"] = float(np.mean(predicted_high == actual_high))
    return result


def predict(
    model: nn.Module,
    observation: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    normalized = (observation - mean) / scale
    loader = DataLoader(torch.from_numpy(normalized.astype(np.float32)), batch_size=batch_size)
    output = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def acceptance_gates(result: dict[str, float | int]) -> dict[str, dict[str, object]]:
    """Behavior-oriented first-pass gates for unseen Isaac data."""
    definitions = {
        "mae": (float(result.get("mae", math.inf)) <= 0.15, "<= 0.15"),
        "extreme_accuracy": (
            float(result.get("extreme_accuracy", 0.0)) >= 0.85,
            ">= 0.85",
        ),
        "low_pred_mean": (
            float(result.get("low_pred_mean", math.inf)) <= 0.35,
            "<= 0.35",
        ),
        "high_pred_mean": (
            float(result.get("high_pred_mean", -math.inf)) >= 0.65,
            ">= 0.65",
        ),
    }
    return {
        name: {
            "pass": bool(passed),
            "value": float(result.get(name, math.nan)),
            "target": target,
        }
        for name, (passed, target) in definitions.items()
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with np.load(args.train) as probe:
        indices = feature_indices(args.features, int(probe["obs"].shape[1]))
    train_x, train_y = load_dataset(args.train, indices)
    augmentation_report = []
    for path in args.augment:
        extra_x, extra_y = load_dataset(path, indices)
        augmentation_report.append({"path": str(path), "samples": int(extra_y.size)})
        train_x = np.concatenate((train_x, extra_x), axis=0)
        train_y = np.concatenate((train_y, extra_y), axis=0)
    test_x, test_y = load_dataset(args.test, indices)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-4] = 1.0
    train_normalized = (train_x - mean) / scale

    model = FrictionEstimator(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    loss_fn = nn.SmoothL1Loss(beta=0.08)
    dataset = TensorDataset(
        torch.from_numpy(train_normalized.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for observation, target in loader:
            observation = observation.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(observation), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = sum(losses) / max(len(losses), 1)
        history.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1:03d} loss={epoch_loss:.6f}", flush=True)

    model = model.cpu()
    train_prediction = predict(model, train_x, mean, scale, args.batch_size)
    test_prediction = predict(model, test_x, mean, scale, args.batch_size)
    unseen_metrics = metrics(test_y, test_prediction)
    gates = acceptance_gates(unseen_metrics)
    overall = all(item["pass"] for item in gates.values())
    report: dict[str, object] = {
        "features": args.features,
        "input_dim": int(train_x.shape[1]),
        "augmentation": augmentation_report,
        "train": metrics(train_y, train_prediction),
        "isaac_unseen": unseen_metrics,
        "final_loss": history[-1],
        "acceptance_gates": gates,
        "overall": "PASS" if overall else "FAIL",
    }
    if args.mujoco is not None:
        mujoco_x, mujoco_y = load_dataset(args.mujoco, indices)
        mujoco_prediction = predict(model, mujoco_x, mean, scale, args.batch_size)
        report["mujoco"] = metrics(mujoco_y, mujoco_prediction)
        np.savez_compressed(
            args.output_dir / "mujoco_predictions.npz",
            truth=mujoco_y,
            prediction=mujoco_prediction,
        )

    torch.save(
        {
            "model": model.state_dict(),
            "feature_indices": indices,
            "mean": mean,
            "scale": scale,
            "input_dim": int(train_x.shape[1]),
        },
        args.output_dir / "friction_estimator.pt",
    )

    deploy_model = NormalizedFrictionEstimator(model, mean, scale).eval()
    example = torch.zeros((1, train_x.shape[1]), dtype=torch.float32)
    scripted = torch.jit.trace(deploy_model, example)
    scripted.save(str(args.output_dir / "friction_estimator.ts"))
    torch.onnx.export(
        deploy_model,
        example,
        args.output_dir / "friction_estimator.onnx",
        input_names=["observation"],
        output_names=["estimated_mu"],
        dynamic_axes={"observation": {0: "batch"}, "estimated_mu": {0: "batch"}},
        opset_version=17,
    )
    np.savez_compressed(
        args.output_dir / "isaac_unseen_predictions.npz",
        truth=test_y,
        prediction=test_prediction,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"overall={report['overall']} onnx={args.output_dir / 'friction_estimator.onnx'}")
    return 3 if args.strict and not overall else 0


if __name__ == "__main__":
    raise SystemExit(main())
