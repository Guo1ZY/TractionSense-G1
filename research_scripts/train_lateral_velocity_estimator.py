#!/usr/bin/env python3
"""Train a deployable body-lateral-velocity estimator from 1864-D rollouts.

The final two policy channels are training labels
``[body_vy, relative_heading]``.  Only the first 1862 causal channels are
provided to the network, so body_vy cannot leak into the estimator input.
Relative heading remains an analytic IMU signal and is not learned here.
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


POLICY_DIM = 1864
INPUT_DIM = 1862
VY_INDEX = 1862


class LateralVelocityEstimator(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return 1.5 * torch.tanh(self.net(observation)).squeeze(-1)


class NormalizedLateralVelocityEstimator(nn.Module):
    def __init__(
        self,
        estimator: LateralVelocityEstimator,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        return self.estimator((raw_observation - self.mean) / self.scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        observation = np.asarray(data["obs"], dtype=np.float32)
    if observation.ndim != 2 or observation.shape[1] != POLICY_DIM:
        raise ValueError(f"{path}: expected Nx{POLICY_DIM}, got {observation.shape}")
    finite = np.isfinite(observation).all(axis=1)
    observation = observation[finite]
    return observation[:, :INPUT_DIM], observation[:, VY_INDEX]


def predict(
    model: nn.Module,
    observation: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(
                ((observation[start : start + batch_size] - mean) / scale).astype(
                    np.float32
                )
            ).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - truth
    variance = float(np.sum(np.square(truth - truth.mean())))
    active = np.abs(truth) >= 0.03
    result: dict[str, float | int] = {
        "samples": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
        "bias": float(np.mean(error)),
        "r2": float(
            1.0 - np.sum(np.square(error)) / max(variance, 1.0e-9)
        ),
    }
    if np.any(active):
        result["active_sign_accuracy"] = float(
            np.mean(np.sign(prediction[active]) == np.sign(truth[active]))
        )
    return result


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_x, train_y = load(args.train)
    augmentation = []
    for path in args.augment:
        extra_x, extra_y = load(path)
        augmentation.append({"path": str(path.resolve()), "samples": len(extra_y)})
        train_x = np.concatenate((train_x, extra_x), axis=0)
        train_y = np.concatenate((train_y, extra_y), axis=0)
    test_x, test_y = load(args.test)

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-4] = 1.0
    normalized = ((train_x - mean) / scale).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(normalized),
        torch.from_numpy(train_y),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    model = LateralVelocityEstimator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for observation, target in loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observation)
            loss = nn.functional.smooth_l1_loss(
                prediction,
                target,
                beta=0.04,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(float(np.mean(losses)))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={history[-1]:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    train_prediction = predict(
        model, train_x, mean, scale, args.batch_size * 2, device
    )
    test_prediction = predict(
        model, test_x, mean, scale, args.batch_size * 2, device
    )
    train_result = metrics(train_y, train_prediction)
    test_result = metrics(test_y, test_prediction)
    gates = {
        "mae": {
            "value": test_result["mae"],
            "target": "<= 0.06 m/s",
            "pass": test_result["mae"] <= 0.06,
        },
        "p95_abs_error": {
            "value": test_result["p95_abs_error"],
            "target": "<= 0.15 m/s",
            "pass": test_result["p95_abs_error"] <= 0.15,
        },
        "active_sign_accuracy": {
            "value": test_result.get("active_sign_accuracy", 0.0),
            "target": ">= 0.70",
            "pass": test_result.get("active_sign_accuracy", 0.0) >= 0.70,
        },
    }
    overall = all(item["pass"] for item in gates.values())
    report = {
        "method": "causal 1862-D body-vy estimator; heading remains analytic IMU",
        "input_dim": INPUT_DIM,
        "target": "policy_obs[1862] body-frame lateral velocity",
        "target_leakage": False,
        "augmentation": augmentation,
        "train": train_result,
        "unseen_seed": test_result,
        "gates": gates,
        "overall": "PASS" if overall else "FAIL",
        "final_loss": history[-1],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu().eval()
    torch.save(
        {
            "model": cpu_model.state_dict(),
            "mean": mean,
            "scale": scale,
            "input_dim": INPUT_DIM,
        },
        args.output_dir / "lateral_velocity_estimator.pt",
    )
    deploy = NormalizedLateralVelocityEstimator(cpu_model, mean, scale).eval()
    example = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        deploy,
        example,
        args.output_dir / "lateral_velocity_estimator.onnx",
        input_names=["observation"],
        output_names=["estimated_body_vy"],
        dynamic_axes={
            "observation": {0: "batch"},
            "estimated_body_vy": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    np.savez_compressed(
        args.output_dir / "unseen_predictions.npz",
        truth=test_y,
        prediction=test_prediction,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 3 if args.strict and not overall else 0


if __name__ == "__main__":
    raise SystemExit(main())
