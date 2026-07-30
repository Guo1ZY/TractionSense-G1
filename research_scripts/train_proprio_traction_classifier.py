#!/usr/bin/env python3
"""Train a LOW/HIGH traction classifier from the common 480-D G1 history.

Only observation[:, :480] is consumed.  Any foot/Hall/sensor-health suffix is
discarded by construction.  Labels are used only during supervised training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Classifier(nn.Module):
    def __init__(self, input_dim: int = 480):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


class DeployClassifier(nn.Module):
    def __init__(self, model: Classifier, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        normalized = (raw_observation - self.mean) / self.scale
        return torch.sigmoid(self.model(normalized))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="480-D G1 traction classifier")
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-max", type=float, default=0.30)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--min-command", type=float, default=0.18)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. 'auto' selects CUDA when available, otherwise CPU.",
    )
    return parser.parse_args()


def load(paths: list[Path], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metadata: list[dict] = []
    for path in paths:
        with np.load(path) as data:
            observation = np.asarray(data["obs"], dtype=np.float32)
            mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
            command = (
                np.asarray(data["cmd_vx"], dtype=np.float32).reshape(-1)
                if "cmd_vx" in data
                else np.full(mu.shape, 1.0, dtype=np.float32)
            )
        if observation.ndim != 2 or observation.shape[1] < 480:
            raise ValueError(f"{path}: expected observation Nx>=480, got {observation.shape}")
        finite = np.isfinite(observation[:, :480]).all(axis=1) & np.isfinite(mu)
        extreme = (mu <= args.low_max) | (mu >= args.high_min)
        moving = np.abs(command) >= args.min_command
        keep = finite & extreme & moving
        x = observation[keep, :480]
        y = (mu[keep] <= args.low_max).astype(np.float32)
        xs.append(x)
        ys.append(y)
        metadata.append(
            {
                "path": str(path),
                "samples": int(x.shape[0]),
                "low": int(np.sum(y == 1.0)),
                "high": int(np.sum(y == 0.0)),
            }
        )
    return np.concatenate(xs), np.concatenate(ys), metadata


def balanced_indices(y: np.ndarray, seed: int) -> np.ndarray:
    low = np.flatnonzero(y == 1.0)
    high = np.flatnonzero(y == 0.0)
    count = min(low.size, high.size)
    rng = np.random.default_rng(seed)
    return np.concatenate(
        (rng.choice(low, count, replace=False), rng.choice(high, count, replace=False))
    )


def infer(
    model: Classifier,
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    normalized = ((x - mean) / scale).astype(np.float32)
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            logits = model(torch.from_numpy(normalized[start : start + batch_size]))
            output.append(torch.sigmoid(logits).numpy())
    return np.concatenate(output)


def metrics(y: np.ndarray, p_low: np.ndarray) -> dict[str, float | int]:
    prediction = p_low >= 0.5
    actual = y >= 0.5
    low = actual
    high = ~actual
    tp = int(np.sum(prediction & actual))
    tn = int(np.sum((~prediction) & (~actual)))
    fp = int(np.sum(prediction & (~actual)))
    fn = int(np.sum((~prediction) & actual))
    return {
        "samples": int(y.size),
        "accuracy": float(np.mean(prediction == actual)),
        "balanced_accuracy": float(
            0.5
            * (
                np.mean(prediction[low] == actual[low])
                + np.mean(prediction[high] == actual[high])
            )
        ),
        "low_p_mean": float(np.mean(p_low[low])),
        "high_p_mean": float(np.mean(p_low[high])),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_meta = load(args.train, args)
    test_x, test_y, test_meta = load(args.test, args)
    selected = balanced_indices(train_y, args.seed)
    train_x = train_x[selected]
    train_y = train_y[selected]

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-4] = 1.0
    train_normalized = ((train_x - mean) / scale).astype(np.float32)

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, but CUDA is unavailable; "
            "use --device auto or --device cpu"
        )
    device = torch.device(requested_device)
    print(f"device={device}", flush=True)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=2.0e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_normalized),
            torch.from_numpy(train_y),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for observation, target in loader:
            observation = observation.to(device)
            target = target.to(device)
            # Structured sensor gap: noise only on the deployable proprio input.
            noisy = observation + 0.02 * torch.randn_like(observation)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(noisy), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1:03d} loss={history[-1]:.6f}", flush=True)

    model = model.cpu().eval()
    train_probability = infer(model, train_x, mean, scale, args.batch_size)
    test_probability = infer(model, test_x, mean, scale, args.batch_size)
    report = {
        "input_dim": 480,
        "foot_channels": "discarded",
        "low_max": args.low_max,
        "high_min": args.high_min,
        "min_command": args.min_command,
        "train_sources": train_meta,
        "test_sources": test_meta,
        "train": metrics(train_y, train_probability),
        "test": metrics(test_y, test_probability),
        "final_loss": history[-1],
    }
    report["overall"] = (
        "PASS"
        if report["test"]["balanced_accuracy"] >= 0.90
        and report["test"]["low_p_mean"] >= 0.65
        and report["test"]["high_p_mean"] <= 0.35
        else "FAIL"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "input_dim": 480,
        },
        args.output_dir / "traction_classifier.pt",
    )
    deploy = DeployClassifier(model, mean, scale).eval()
    example = torch.zeros((1, 480), dtype=torch.float32)
    torch.onnx.export(
        deploy,
        example,
        args.output_dir / "traction_classifier.onnx",
        input_names=["observation"],
        output_names=["p_low"],
        dynamic_axes={"observation": {0: "batch"}, "p_low": {0: "batch"}},
        opset_version=17,
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        truth=test_y,
        p_low=test_probability,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
