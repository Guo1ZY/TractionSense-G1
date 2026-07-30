#!/usr/bin/env python3
"""Train the final shared dual-foot magnetic encoder and 29-DoF actor.

The exported model has one flat input for compatibility with g1_ctrl, but its
internal policy trunk consumes only 548 fused features:
  480 proprio + 32 left latent + 32 right latent + 4 status/motion = 548.

External input layout (1864 floats):
  [0:480]       five-frame proprioception/command/action history
  [480:1830]    magnetic history [15, 2, 15, 3]
  [1830:1860]   sample-period history [15, 2] in seconds
  [1860:1862]   current valid [left, right]
  [1862:1864]   task-specific context; the motion Student uses
                [estimated body lateral velocity, relative heading]

The legacy proxy-data path still fills the last pair with normalized sensor
ages. Exact 1864-D motion-feedback distillation is implemented in
``fine_tune_shared_magnetic_dagger.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from distill_traction_student import Actor, action_metrics, load_actor, load_observations
from train_magnetic_traction_student import (
    AGE,
    BASE_DIM,
    FEET,
    HISTORY,
    MAGNETIC_DIM,
    NORMAL,
    OLD_INPUT_DIM,
    SENSORS,
    TANGENT,
    VALID,
    magnetic_observation,
)


AXES = 3
PERIOD_DIM = HISTORY * FEET
HEALTH_DIM = 4
INPUT_DIM = BASE_DIM + MAGNETIC_DIM + PERIOD_DIM + HEALTH_DIM
FUSED_DIM = BASE_DIM + 32 * FEET + HEALTH_DIM
OUTPUT_DIM = 29


class SharedFootEncoder(nn.Module):
    """One set of weights applied independently to left and right feet."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        point_dim = 16
        self.point_mlp = nn.Sequential(
            nn.Linear(AXES, point_dim),
            nn.ELU(),
            nn.Linear(point_dim, point_dim),
            nn.ELU(),
        )
        self.sensor_embedding = nn.Parameter(torch.zeros(SENSORS, point_dim))
        nn.init.normal_(self.sensor_embedding, std=0.02)
        self.frame_mlp = nn.Sequential(
            nn.Linear(SENSORS * point_dim + 1, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.output = nn.Sequential(nn.Linear(32, latent_dim), nn.ELU())

    def forward(self, magnetic: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        # magnetic: [B,T,15,3], period: [B,T]
        batch, history, sensors, axes = magnetic.shape
        if history != HISTORY or sensors != SENSORS or axes != AXES:
            raise ValueError(f"invalid foot history shape {tuple(magnetic.shape)}")
        point = self.point_mlp(magnetic)
        point = point + self.sensor_embedding.view(1, 1, SENSORS, -1)
        frame = point.reshape(batch, history, -1)
        frame = torch.cat((frame, period.unsqueeze(-1)), dim=-1)
        frame = self.frame_mlp(frame)
        temporal = self.temporal(frame.transpose(1, 2))
        return self.output(temporal[:, :, -1])


class SharedMagneticPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foot_encoder = SharedFootEncoder(32)
        self.actor = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, OUTPUT_DIM),
        )
        self.mu_head = nn.Linear(128, 1)
        self.foot_head = nn.Linear(32, 3)

    def split(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected Nx{INPUT_DIM}, got {tuple(observation.shape)}")
        base = observation[:, :BASE_DIM]
        mag_start = BASE_DIM
        mag_end = mag_start + MAGNETIC_DIM
        magnetic = observation[:, mag_start:mag_end].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        period = observation[:, mag_end : mag_end + PERIOD_DIM].reshape(
            -1, HISTORY, FEET
        )
        health = observation[:, -HEALTH_DIM:]
        return base, magnetic, period, health

    def encode(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, magnetic, period, health = self.split(observation)
        left = self.foot_encoder(magnetic[:, :, 0], period[:, :, 0])
        right = self.foot_encoder(magnetic[:, :, 1], period[:, :, 1])
        fused = torch.cat((base, left, right, health), dim=-1)
        return fused, left, right

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.encode(observation)
        return self.actor(fused)

    def auxiliary(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused, left, right = self.encode(observation)
        latent = self.actor[:6](fused)
        mu = 1.30 * torch.sigmoid(self.mu_head(latent)).squeeze(-1)
        # contact logit, normal force, tangent force for each foot.
        feet = torch.stack((self.foot_head(left), self.foot_head(right)), dim=1)
        return mu, feet


def proxy_input(old: torch.Tensor, stochastic: bool) -> torch.Tensor:
    converted = magnetic_observation(old, stochastic=stochastic)
    magnetic = converted[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
        -1, HISTORY, FEET, SENSORS, AXES
    )
    batch = old.shape[0]
    if stochastic:
        foot_period = 0.018 + 0.030 * torch.rand(
            batch, 1, FEET, device=old.device, dtype=old.dtype
        )
        jitter = 0.003 * torch.randn(
            batch, HISTORY, FEET, device=old.device, dtype=old.dtype
        )
        periods = (foot_period + jitter).clamp(0.010, 0.080)
    else:
        periods = torch.full(
            (batch, HISTORY, FEET), 0.020, device=old.device, dtype=old.dtype
        )
    # Old training data has one dual-foot health signal. Duplicate it into the
    # final per-foot schema; live F0M1 carries truly independent L/R health.
    valid = old[:, VALID][:, -1:].repeat(1, FEET).clamp(0.0, 1.0)
    age = old[:, AGE][:, -1:].repeat(1, FEET).clamp(0.0, 1.0)
    if stochastic:
        independent_dropout = (
            torch.rand(batch, FEET, device=old.device) < 0.02
        )
        valid = valid * (~independent_dropout).to(old.dtype)
        fresh_age = 0.08 * torch.rand(
            batch, FEET, device=old.device, dtype=old.dtype
        )
        stale_age = 1.0
        age = torch.maximum(age, fresh_age)
        age = torch.where(independent_dropout, stale_age, age)
        magnetic = magnetic * valid[:, None, :, None, None]
    return torch.cat(
        (
            old[:, :BASE_DIM],
            magnetic.reshape(batch, -1),
            periods.reshape(batch, -1),
            valid,
            age,
        ),
        dim=-1,
    )


def initialize_policy(
    model: SharedMagneticPolicy,
    teacher: Actor,
    source_student: Path | None,
) -> str:
    source = Actor(OLD_INPUT_DIM)
    if source_student is not None:
        payload = torch.load(source_student, map_location="cpu", weights_only=False)
        source.load_state_dict(payload["model"], strict=False)
        description = str(source_student.resolve())
    else:
        with torch.no_grad():
            source.mlp[0].weight.copy_(teacher.mlp[0].weight[:, :OLD_INPUT_DIM])
            source.mlp[0].bias.copy_(
                teacher.mlp[0].bias + 0.5 * teacher.mlp[0].weight[:, OLD_INPUT_DIM]
            )
            for index in (2, 4, 6):
                source.mlp[index].weight.copy_(teacher.mlp[index].weight)
                source.mlp[index].bias.copy_(teacher.mlp[index].bias)
        description = "Teacher proprio/action trunk"
    with torch.no_grad():
        model.actor[0].weight.zero_()
        model.actor[0].weight[:, :BASE_DIM].copy_(
            source.mlp[0].weight[:, :BASE_DIM]
        )
        model.actor[0].bias.copy_(source.mlp[0].bias)
        for destination, source_index in ((2, 2), (4, 4), (6, 6)):
            model.actor[destination].weight.copy_(source.mlp[source_index].weight)
            model.actor[destination].bias.copy_(source.mlp[source_index].bias)
    return description


def make_targets(
    teacher: Actor,
    observation: np.ndarray,
    mu: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            old = torch.from_numpy(observation[start : start + batch_size]).to(device)
            target_mu = torch.from_numpy(mu[start : start + batch_size]).to(device)
            output.append(
                teacher(
                    torch.cat((old, target_mu.clamp(0.0, 1.2).unsqueeze(-1)), dim=-1)
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(output)


def auxiliary_targets(old: torch.Tensor) -> torch.Tensor:
    normal = old[:, NORMAL].reshape(-1, HISTORY, FEET)[:, -1]
    tangent = old[:, TANGENT].reshape(-1, HISTORY, FEET)[:, -1]
    contact = (normal > 0.05).to(old.dtype)
    return torch.stack((contact, normal, tangent), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--source-student", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--augment-repeat", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8030)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def evaluate(
    model: SharedMagneticPolicy,
    old_np: np.ndarray,
    mu_np: np.ndarray,
    target_np: np.ndarray,
    batch_size: int,
    device: torch.device,
    stochastic: bool,
    repeats: int,
) -> tuple[dict, dict, np.ndarray]:
    action_repeats = []
    mu_repeats = []
    model.eval()
    with torch.inference_mode():
        for repeat in range(repeats):
            torch.manual_seed(20000 + repeat)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(20000 + repeat)
            actions, mus = [], []
            for start in range(0, len(old_np), batch_size):
                old = torch.from_numpy(old_np[start : start + batch_size]).to(device)
                observation = proxy_input(old, stochastic)
                actions.append(model(observation).cpu().numpy())
                mus.append(model.auxiliary(observation)[0].cpu().numpy())
            action_repeats.append(np.concatenate(actions))
            mu_repeats.append(np.concatenate(mus))
    action = np.mean(action_repeats, axis=0)
    predicted_mu = np.mean(mu_repeats, axis=0)
    low = mu_np <= 0.25
    high = mu_np >= 0.75
    mu_error = predicted_mu - mu_np
    mu_metrics = {
        "mae": float(np.mean(np.abs(mu_error))),
        "rmse": float(np.sqrt(np.mean(mu_error**2))),
        "low_pred_mean": float(np.mean(predicted_mu[low])),
        "high_pred_mean": float(np.mean(predicted_mu[high])),
        "extreme_accuracy": float(
            np.mean(
                np.concatenate(
                    ((predicted_mu[low] < 0.45), (predicted_mu[high] > 0.60))
                )
            )
        ),
    }
    return action_metrics(target_np, action), mu_metrics, action


def progress(epoch: int, total: int, loss: float, started: float) -> str:
    fraction = epoch / total
    width = 32
    filled = round(width * fraction)
    elapsed = time.monotonic() - started
    eta = elapsed * (1.0 - fraction) / max(fraction, 1.0e-9)
    vram = (
        f" VRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
        if torch.cuda.is_available()
        else ""
    )
    return (
        f"[{'#' * filled}{'-' * (width - filled)}] {100*fraction:6.2f}% "
        f"{epoch:03d}/{total} loss={loss:.6f} ETA={eta/60:.1f}m{vram}"
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_mu, train_cmd = load_observations(args.train)
    augmentation = []
    for path in args.augment:
        extra_x, extra_mu, extra_cmd = load_observations(path)
        augmentation.append({"path": str(path.resolve()), "samples": len(extra_x)})
        for _ in range(max(args.augment_repeat, 1)):
            train_x = np.concatenate((train_x, extra_x))
            train_mu = np.concatenate((train_mu, extra_mu))
            train_cmd = np.concatenate((train_cmd, extra_cmd))
    test_x, test_mu, test_cmd = load_observations(args.test)
    teacher = load_actor(args.teacher_onnx, OLD_INPUT_DIM + 1).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = SharedMagneticPolicy().to(device)
    initialized_from = initialize_policy(model, teacher.cpu(), args.source_student)
    teacher = teacher.to(device)
    print(
        f"device={device} external_input={INPUT_DIM} fused_actor={FUSED_DIM} "
        f"train={len(train_x)} test={len(test_x)}",
        flush=True,
    )
    train_target = make_targets(teacher, train_x, train_mu, args.batch_size * 2, device)
    test_target = make_targets(teacher, test_x, test_mu, args.batch_size * 2, device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_mu),
            torch.from_numpy(train_cmd),
            torch.from_numpy(train_target),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for old, target_mu, command_vx, target_action in loader:
            old = old.to(device, non_blocking=True)
            target_mu = target_mu.to(device, non_blocking=True)
            command_vx = command_vx.to(device, non_blocking=True)
            target_action = target_action.to(device, non_blocking=True)
            observation = proxy_input(old, stochastic=True)
            target_feet = auxiliary_targets(old)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                predicted_action = model(observation)
                predicted_mu, predicted_feet = model.auxiliary(observation)
                error = predicted_action - target_action
                per_sample = 0.7 * error.square().mean(dim=1) + 0.3 * (
                    nn.functional.smooth_l1_loss(
                        predicted_action, target_action, beta=0.05, reduction="none"
                    ).mean(dim=1)
                )
                weight = torch.ones_like(target_mu)
                weight += (target_mu <= 0.25).float()
                weight += (target_mu >= 0.75).float()
                weight += 3.0 * ((target_mu >= 1.10) & (command_vx >= 1.0)).float()
                action_loss = torch.sum(per_sample * weight) / torch.sum(weight)
                mu_loss = nn.functional.smooth_l1_loss(
                    predicted_mu, target_mu, beta=0.08
                )
                contact_loss = nn.functional.binary_cross_entropy_with_logits(
                    predicted_feet[:, :, 0], target_feet[:, :, 0]
                )
                force_loss = nn.functional.smooth_l1_loss(
                    torch.relu(predicted_feet[:, :, 1:]),
                    target_feet[:, :, 1:],
                    beta=0.05,
                )
                loss = action_loss + 0.05 * mu_loss + 0.01 * contact_loss + 0.02 * force_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.detach()))
        scheduler.step()
        epoch_loss = float(np.mean(batch_losses))
        history.append(epoch_loss)
        print(progress(epoch, args.epochs, epoch_loss, started), flush=True)
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "training" if epoch < args.epochs else "evaluating",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "percent": 100.0 * epoch / args.epochs,
                    "loss": epoch_loss,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "input_dim": INPUT_DIM},
            args.output_dir / "checkpoint_latest.pt",
        )

    nominal_action, nominal_mu, _ = evaluate(
        model, test_x, test_mu, test_target, args.batch_size, device, False, 1
    )
    random_action, random_mu, predictions = evaluate(
        model, test_x, test_mu, test_target, args.batch_size, device, True, 3
    )
    gates = {
        "action_mae": {
            "value": random_action["mae"],
            "target": "<= 0.035",
            "pass": random_action["mae"] <= 0.035,
        },
        "action_p95": {
            "value": random_action["p95_abs"],
            "target": "<= 0.090",
            "pass": random_action["p95_abs"] <= 0.090,
        },
        "friction_extreme_accuracy": {
            "value": random_mu["extreme_accuracy"],
            "target": ">= 0.85",
            "pass": random_mu["extreme_accuracy"] >= 0.85,
        },
        "finite": {
            "value": bool(np.isfinite(predictions).all()),
            "target": "true",
            "pass": bool(np.isfinite(predictions).all()),
        },
    }
    overall = all(item["pass"] for item in gates.values())
    report = {
        "method": "shared per-foot spatial-temporal encoder + lateral-guard Teacher distillation",
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "initialized_from": initialized_from,
        "input_dim": INPUT_DIM,
        "internal_actor_dim": FUSED_DIM,
        "output_dim": OUTPUT_DIM,
        "schema": {
            "proprio": BASE_DIM,
            "magnetic_history": [HISTORY, FEET, SENSORS, AXES],
            "period_history": [HISTORY, FEET],
            "status_or_motion": [
                "left_valid",
                "right_valid",
                "context_0",
                "context_1",
            ],
            "shared_foot_encoder": True,
        },
        "training_samples": len(train_x),
        "test_samples": len(test_x),
        "augmentation": augmentation,
        "epochs": args.epochs,
        "final_loss": history[-1],
        "nominal_action": nominal_action,
        "nominal_friction": nominal_mu,
        "randomized_action": random_action,
        "randomized_friction": random_mu,
        "gates": gates,
        "overall": "PASS" if overall else "FAIL",
    }
    model = model.cpu().eval()
    torch.save(
        {"model": model.state_dict(), "metrics": report, "input_dim": INPUT_DIM},
        args.output_dir / "shared_magnetic_policy.pt",
    )
    torch.onnx.export(
        model,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )
    np.savez_compressed(
        args.output_dir / "isaac_unseen_actions.npz",
        truth_mu=test_mu,
        command_vx=test_cmd,
        teacher_action=test_target,
        student_action=predictions,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete" if overall else "evaluation_failed",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "percent": 100.0,
                "overall": report["overall"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if overall or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
