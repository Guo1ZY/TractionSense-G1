#!/usr/bin/env python3
"""Distill the 7989 traction Teacher into a raw dual-foot magnetic-history actor.

Input layout (1840 floats):
  0:480      existing 5-frame proprioception/command/action prefix
  480:1830   15 frames x 2 feet x 15 Hall sensors x XYZ, normalized delta field
  1830:1835  five-frame dual-sensor valid flag
  1835:1840  five-frame normalized sample age

Isaac contact histories are converted online into a randomized magnetic-array
proxy. True friction is used only as a training target for the auxiliary head
and frozen Teacher; it is not present in the exported ONNX input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_traction_student import Actor, action_metrics, load_actor, load_observations


BASE_DIM = 480
OLD_INPUT_DIM = 640
HISTORY = 15
FEET = 2
SENSORS = 15
AXES = 3
MAGNETIC_DIM = HISTORY * FEET * SENSORS * AXES
HEALTH_DIM = 10
INPUT_DIM = BASE_DIM + MAGNETIC_DIM + HEALTH_DIM
OUTPUT_DIM = 29

CONTACT = slice(480, 510)
NORMAL = slice(510, 540)
TANGENT = slice(540, 570)
VALID = slice(630, 635)
AGE = slice(635, 640)


class MagneticActor(Actor):
    def __init__(self) -> None:
        super().__init__(INPUT_DIM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--source-student", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--augment-repeat", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--mu-loss-coef", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7989)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _base_sensor_profile(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 3 columns x 5 rows, heel to toe. Mean response is one so aggregate
    # magnetic magnitude remains in the same numerical range as force obs.
    x = torch.tensor([-1.0, 0.0, 1.0] * 5, device=device, dtype=dtype)
    y = torch.repeat_interleave(
        torch.linspace(-1.0, 1.0, 5, device=device, dtype=dtype), 3
    )
    profile = 0.82 + 0.12 * y + 0.06 * (1.0 - torch.abs(x))
    return profile / profile.mean()


def magnetic_observation(
    old_observation: torch.Tensor,
    *,
    stochastic: bool,
) -> torch.Tensor:
    """Convert old aggregate force history to a normalized 15-point XYZ proxy."""

    if old_observation.ndim != 2 or old_observation.shape[1] != OLD_INPUT_DIM:
        raise ValueError(f"expected Nx{OLD_INPUT_DIM}, got {tuple(old_observation.shape)}")
    batch = old_observation.shape[0]
    device = old_observation.device
    dtype = old_observation.dtype
    normal = old_observation[:, NORMAL].reshape(batch, HISTORY, FEET).clamp(0.0, 5.0)
    tangent = old_observation[:, TANGENT].reshape(batch, HISTORY, FEET).clamp(0.0, 5.0)
    contact = old_observation[:, CONTACT].reshape(batch, HISTORY, FEET).clamp(0.0, 1.0)
    profile = _base_sensor_profile(device, dtype).view(1, 1, 1, SENSORS, 1)

    normal_axis = torch.tensor(
        [0.14, -0.10, 1.00], device=device, dtype=dtype
    ).view(1, 1, 1, 1, AXES)
    tangent_axis = torch.tensor(
        [1.00, 0.42, 0.12], device=device, dtype=dtype
    ).view(1, 1, 1, 1, AXES)
    if stochastic:
        # Episode-correlated mounting, sensor gain, zero residual, polarity,
        # cross-axis response and channel dropout. Parameters are constant
        # across all 15 temporal frames of a sample.
        sensor_gain = 0.72 + 0.56 * torch.rand(
            batch, 1, FEET, SENSORS, 1, device=device, dtype=dtype
        )
        axis_gain = 0.75 + 0.50 * torch.rand(
            batch, 1, FEET, 1, AXES, device=device, dtype=dtype
        )
        tangent_sign = torch.where(
            torch.rand(batch, 1, FEET, 1, 1, device=device) < 0.5,
            -torch.ones((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        )
        normal_mix = normal_axis + 0.08 * torch.randn(
            batch, 1, FEET, 1, AXES, device=device, dtype=dtype
        )
        tangent_mix = tangent_sign * tangent_axis + 0.12 * torch.randn(
            batch, 1, FEET, 1, AXES, device=device, dtype=dtype
        )
        zero_residual = 0.06 * torch.randn(
            batch, 1, FEET, SENSORS, AXES, device=device, dtype=dtype
        )
        channel_keep = (
            torch.rand(batch, 1, FEET, SENSORS, 1, device=device) >= 0.015
        ).to(dtype)
    else:
        sensor_gain = torch.ones(
            batch, 1, FEET, SENSORS, 1, device=device, dtype=dtype
        )
        axis_gain = torch.ones(
            batch, 1, FEET, 1, AXES, device=device, dtype=dtype
        )
        normal_mix = normal_axis
        tangent_mix = tangent_axis
        zero_residual = torch.zeros(
            batch, 1, FEET, SENSORS, AXES, device=device, dtype=dtype
        )
        channel_keep = torch.ones(
            batch, 1, FEET, SENSORS, 1, device=device, dtype=dtype
        )

    fn = normal[..., None, None]
    ft = tangent[..., None, None]
    magnetic = profile * sensor_gain * axis_gain * (
        fn * normal_mix + ft * tangent_mix
    )
    # Hall displacement is mildly nonlinear near the mechanical limit.
    magnetic = 5.0 * torch.tanh(magnetic / 5.0)
    magnetic = magnetic + zero_residual
    if stochastic:
        noise = (0.025 + 0.018 * magnetic.abs()) * torch.randn_like(magnetic)
        magnetic = magnetic + noise
    magnetic = magnetic * channel_keep
    # A fully invalid aggregate packet becomes small baseline residual only.
    magnetic = magnetic * (0.05 + 0.95 * contact[..., None, None])
    magnetic = magnetic.clamp(-6.0, 6.0)
    return torch.cat(
        (
            old_observation[:, :BASE_DIM],
            magnetic.reshape(batch, MAGNETIC_DIM),
            old_observation[:, VALID],
            old_observation[:, AGE],
        ),
        dim=-1,
    )


def initialize_from_source(
    student: MagneticActor,
    teacher: Actor,
    source_path: Path | None,
) -> str:
    source = Actor(OLD_INPUT_DIM)
    description: str
    if source_path is not None:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source.load_state_dict(payload["model"], strict=False)
        description = str(source_path.resolve())
    else:
        with torch.no_grad():
            source.mlp[0].weight.copy_(teacher.mlp[0].weight[:, :OLD_INPUT_DIM])
            source.mlp[0].bias.copy_(
                teacher.mlp[0].bias + 0.5 * teacher.mlp[0].weight[:, OLD_INPUT_DIM]
            )
            for index in (2, 4, 6):
                source.mlp[index].weight.copy_(teacher.mlp[index].weight)
                source.mlp[index].bias.copy_(teacher.mlp[index].bias)
        description = "teacher first 640 columns with nominal mu=0.5"
    with torch.no_grad():
        student.mlp[0].weight.zero_()
        student.mlp[0].weight[:, :BASE_DIM].copy_(
            source.mlp[0].weight[:, :BASE_DIM]
        )
        student.mlp[0].bias.copy_(source.mlp[0].bias)
        for index in (2, 4, 6):
            student.mlp[index].weight.copy_(source.mlp[index].weight)
            student.mlp[index].bias.copy_(source.mlp[index].bias)
    return description


def teacher_targets(
    teacher: Actor,
    observation: np.ndarray,
    mu: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    results: list[np.ndarray] = []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            old = torch.from_numpy(observation[start : start + batch_size]).to(device)
            target_mu = torch.from_numpy(mu[start : start + batch_size]).to(device)
            result = teacher(torch.cat((old, target_mu[:, None].clamp(0.0, 1.2)), dim=-1))
            results.append(result.cpu().numpy())
    return np.concatenate(results)


def evaluate(
    student: MagneticActor,
    old_observation: np.ndarray,
    mu: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
    stochastic: bool,
    repeats: int = 1,
) -> tuple[dict[str, float | int], dict[str, float], np.ndarray]:
    predictions_all = []
    mu_predictions_all = []
    student.eval()
    with torch.inference_mode():
        for repeat in range(repeats):
            predictions = []
            mu_predictions = []
            if stochastic:
                torch.manual_seed(15000 + repeat)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(15000 + repeat)
            for start in range(0, len(old_observation), batch_size):
                old = torch.from_numpy(old_observation[start : start + batch_size]).to(device)
                magnetic = magnetic_observation(old, stochastic=stochastic)
                predictions.append(student(magnetic).cpu().numpy())
                mu_predictions.append(student.predict_mu(magnetic).cpu().numpy())
            predictions_all.append(np.concatenate(predictions))
            mu_predictions_all.append(np.concatenate(mu_predictions))
    prediction = np.mean(predictions_all, axis=0)
    mu_prediction = np.mean(mu_predictions_all, axis=0)
    mu_error = mu_prediction - mu
    low = mu <= 0.25
    high = mu >= 0.75
    mu_metrics = {
        "mae": float(np.mean(np.abs(mu_error))),
        "rmse": float(np.sqrt(np.mean(mu_error**2))),
        "low_pred_mean": float(np.mean(mu_prediction[low])),
        "high_pred_mean": float(np.mean(mu_prediction[high])),
        "extreme_accuracy": float(
            np.mean(
                np.concatenate(
                    ((mu_prediction[low] < 0.45), (mu_prediction[high] > 0.60))
                )
            )
        ),
    }
    return action_metrics(targets, prediction), mu_metrics, prediction


def progress_bar(epoch: int, total: int, loss: float, started: float) -> str:
    fraction = epoch / max(total, 1)
    width = 32
    done = min(width, int(round(width * fraction)))
    elapsed = time.monotonic() - started
    eta = elapsed * (1.0 - fraction) / max(fraction, 1.0e-9)
    gpu = ""
    if torch.cuda.is_available():
        gpu = f" VRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
    return (
        f"[{'#' * done}{'-' * (width - done)}] {100.0 * fraction:6.2f}% "
        f"{epoch:03d}/{total} loss={loss:.6f} elapsed={elapsed / 60:.1f}m "
        f"ETA={eta / 60:.1f}m{gpu}"
    )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
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
        augmentation.append({"path": str(path.resolve()), "samples": int(len(extra_x))})
        for _ in range(max(1, args.augment_repeat)):
            train_x = np.concatenate((train_x, extra_x))
            train_mu = np.concatenate((train_mu, extra_mu))
            train_cmd = np.concatenate((train_cmd, extra_cmd))
    test_x, test_mu, test_cmd = load_observations(args.test)

    teacher = load_actor(args.teacher_onnx, OLD_INPUT_DIM + 1).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = MagneticActor().to(device)
    initialized_from = initialize_from_source(student, teacher.cpu(), args.source_student)
    teacher = teacher.to(device)

    print(
        f"device={device} input={INPUT_DIM} output={OUTPUT_DIM} "
        f"train={len(train_x)} test={len(test_x)} batch={args.batch_size}",
        flush=True,
    )
    print("precomputing frozen Teacher targets...", flush=True)
    train_target = teacher_targets(teacher, train_x, train_mu, args.batch_size * 2, device)
    test_target = teacher_targets(teacher, test_x, test_mu, args.batch_size * 2, device)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_mu),
        torch.from_numpy(train_cmd),
        torch.from_numpy(train_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    started = time.monotonic()
    progress_path = args.output_dir / "progress.json"
    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for old, privileged_mu, command_vx, target in loader:
            old = old.to(device, non_blocking=True)
            privileged_mu = privileged_mu.to(device, non_blocking=True)
            command_vx = command_vx.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            magnetic = magnetic_observation(old, stochastic=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = student(magnetic)
                error = prediction - target
                action_loss = (
                    0.70 * torch.mean(error.square(), dim=1)
                    + 0.30
                    * torch.mean(
                        nn.functional.smooth_l1_loss(
                            prediction, target, beta=0.05, reduction="none"
                        ),
                        dim=1,
                    )
                )
                sample_weight = torch.ones_like(privileged_mu)
                sample_weight += (privileged_mu <= 0.25).float()
                sample_weight += (privileged_mu >= 0.75).float()
                sample_weight += 3.0 * (
                    (privileged_mu >= 1.10) & (command_vx >= 1.0)
                ).float()
                loss = torch.sum(action_loss * sample_weight) / torch.sum(sample_weight)
                predicted_mu = student.predict_mu(magnetic)
                mu_loss = nn.functional.smooth_l1_loss(
                    predicted_mu, privileged_mu, beta=0.08
                )
                loss = loss + args.mu_loss_coef * mu_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        status = progress_bar(epoch, args.epochs, epoch_loss, started)
        print(status, flush=True)
        progress_path.write_text(
            json.dumps(
                {
                    "status": "training" if epoch < args.epochs else "evaluating",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "percent": round(100.0 * epoch / args.epochs, 3),
                    "loss": epoch_loss,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model": student.state_dict(),
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "epoch": epoch,
            },
            args.output_dir / "checkpoint_latest.pt",
        )

    nominal_action, nominal_mu, _ = evaluate(
        student, test_x, test_mu, test_target, args.batch_size, device, False
    )
    randomized_action, randomized_mu, test_prediction = evaluate(
        student, test_x, test_mu, test_target, args.batch_size, device, True, repeats=3
    )
    gates = {
        "randomized_action_mae": {
            "value": randomized_action["mae"],
            "target": "<= 0.035",
            "pass": randomized_action["mae"] <= 0.035,
        },
        "randomized_action_p95": {
            "value": randomized_action["p95_abs"],
            "target": "<= 0.090",
            "pass": randomized_action["p95_abs"] <= 0.090,
        },
        "friction_extreme_accuracy": {
            "value": randomized_mu["extreme_accuracy"],
            "target": ">= 0.85",
            "pass": randomized_mu["extreme_accuracy"] >= 0.85,
        },
        "finite": {
            "value": bool(np.isfinite(test_prediction).all()),
            "target": "true",
            "pass": bool(np.isfinite(test_prediction).all()),
        },
    }
    overall = all(item["pass"] for item in gates.values())
    report = {
        "method": "7989 privileged Teacher distillation with online magnetic-array domain randomization",
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "initialized_from": initialized_from,
        "input_schema": {
            "input_dim": INPUT_DIM,
            "proprio_history": BASE_DIM,
            "magnetic_history": MAGNETIC_DIM,
            "magnetic_shape": [HISTORY, FEET, SENSORS, AXES],
            "sensor_health_history": HEALTH_DIM,
            "friction_in_exported_input": False,
        },
        "output_dim": OUTPUT_DIM,
        "training_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "augmentation": augmentation,
        "augmentation_repeat": int(args.augment_repeat),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_loss": history[-1],
        "nominal_proxy_action": nominal_action,
        "nominal_proxy_friction": nominal_mu,
        "randomized_proxy_action": randomized_action,
        "randomized_proxy_friction": randomized_mu,
        "gates": gates,
        "overall": "PASS" if overall else "FAIL",
        "sim_to_real_limitation": (
            "The proxy must be aligned to real per-channel unloaded baselines and "
            "response scales before hardware deployment."
        ),
    }
    student = student.cpu().eval()
    torch.save(
        {
            "model": student.state_dict(),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "metrics": report,
        },
        args.output_dir / "magnetic_student_actor.pt",
    )
    example = torch.zeros((1, INPUT_DIM), dtype=torch.float32)
    torch.onnx.export(
        student,
        example,
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=None,
        opset_version=17,
    )
    np.savez_compressed(
        args.output_dir / "isaac_unseen_actions.npz",
        truth_mu=test_mu,
        command_vx=test_cmd,
        teacher_action=test_target,
        student_action=test_prediction,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    progress_path.write_text(
        json.dumps(
            {
                "status": "complete" if overall else "evaluation_failed",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "percent": 100.0,
                "overall": report["overall"],
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.strict and not overall:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
