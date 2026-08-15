#!/usr/bin/env python3
"""Distill the friction-conditioned G1 Teacher into a torque-only actor.

The deployable input is strictly causal and onboard: the official 480-D
proprioceptive history followed by 15 frames of 29 motor efforts (915-D total).
True ground friction and the 641-D foot-force Teacher stream are training-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from distill_traction_student import action_metrics, load_actor  # noqa: E402


BASE_DIM = 480
JOINTS = 29
EFFORT_HISTORY = 15
INPUT_DIM = BASE_DIM + JOINTS * EFFORT_HISTORY
OUTPUT_DIM = 29
# Exported Isaac joint order is interleaved left-leg/right-leg/waist before
# upper body.  Motor SDK ids 0..11 are the twelve leg joints.
LEG_ACTION_INDICES = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)
WAIST_ACTION_INDICES = (2, 5, 8)


class ProprioTractionPolicy(nn.Module):
    """Shared torque encoder with action and training/telemetry friction heads."""

    def __init__(self, mean: np.ndarray | torch.Tensor, scale: np.ndarray | torch.Tensor):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.action_head = nn.Linear(128, OUTPUT_DIM)
        self.mu_head = nn.Linear(128, 1)

    def latent(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.input_mean) / self.input_scale
        return self.encoder(normalized)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.latent(observation))

    @torch.jit.export
    def predict_mu(self, observation: torch.Tensor) -> torch.Tensor:
        return 1.30 * torch.sigmoid(self.mu_head(self.latent(observation))).squeeze(-1)


class ProprioFrictionEstimator(nn.Module):
    def __init__(self, policy: ProprioTractionPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy.predict_mu(observation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--augment-repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--mu-loss-coef", type=float, default=0.08)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--deploy-template", type=Path)
    parser.add_argument("--install-slot", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        observation = np.asarray(data["obs"], dtype=np.float32)
        teacher_observation = np.asarray(data["teacher_obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        command = np.asarray(
            data.get("cmd_vx", np.full_like(mu, np.nan)), dtype=np.float32
        ).reshape(-1)
    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(f"expected Nx{INPUT_DIM} proprio observations in {path}, got {observation.shape}")
    if teacher_observation.ndim != 2 or teacher_observation.shape[1] != 641:
        raise ValueError(f"expected Nx641 teacher observations in {path}, got {teacher_observation.shape}")
    if not (len(observation) == len(teacher_observation) == len(mu) == len(command)):
        raise ValueError(f"row-count mismatch in {path}")
    finite = (
        np.isfinite(observation).all(axis=1)
        & np.isfinite(teacher_observation).all(axis=1)
        & np.isfinite(mu)
    )
    return observation[finite], teacher_observation[finite], mu[finite], command[finite]


def make_teacher_targets(
    teacher: nn.Module,
    observation: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(observation[start : start + batch_size]).to(device)
            result.append(teacher(batch).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def mu_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    low = truth <= 0.25
    high = truth >= 0.75
    extremes = low | high
    predicted_high = prediction[extremes] >= 0.55
    actual_high = high[extremes]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "low_pred_mean": float(np.mean(prediction[low])) if np.any(low) else math.nan,
        "high_pred_mean": float(np.mean(prediction[high])) if np.any(high) else math.nan,
        "extreme_accuracy": float(np.mean(predicted_high == actual_high)) if np.any(extremes) else math.nan,
    }


def evaluate(
    model: ProprioTractionPolicy,
    observation: np.ndarray,
    target_action: np.ndarray,
    target_mu: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(observation[start : start + batch_size]).to(device)
            actions.append(model(batch).cpu().numpy())
            mus.append(model.predict_mu(batch).cpu().numpy())
    action = np.concatenate(actions)
    mu = np.concatenate(mus)
    return action_metrics(target_action, action), mu_metrics(target_mu, mu), action, mu


def install_slot(output_dir: Path, template: Path | None, slot: Path | None) -> None:
    if template is None and slot is None:
        return
    if template is None or slot is None:
        raise ValueError("--deploy-template and --install-slot must be supplied together")
    text = template.read_text(encoding="utf-8")
    marker = "\n  foot_contact:"
    if marker not in text:
        raise ValueError(f"teacher template has no foot observation marker: {template}")
    prefix = text.split(marker, 1)[0].rstrip()
    effort_scale = ", ".join(["0.02"] * JOINTS)
    deploy = (
        prefix
        + "\n  joint_effort:\n"
        + "    params: {}\n"
        + "    clip: [-100.0, 100.0]\n"
        + f"    scale: [{effort_scale}]\n"
        + f"    history_length: {EFFORT_HISTORY}\n"
    )
    (slot / "exported").mkdir(parents=True, exist_ok=True)
    (slot / "params").mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "policy.onnx", slot / "exported/policy.onnx")
    shutil.copy2(output_dir / "friction_estimator.onnx", slot / "exported/friction_estimator.onnx")
    (slot / "params/deploy.yaml").write_text(deploy, encoding="utf-8")
    (slot / "checkpoint.txt").write_text(
        str((output_dir / "proprio_policy.pt").resolve()) + "\n", encoding="utf-8"
    )
    shutil.copy2(output_dir / "metrics.json", slot / "distillation_metrics.json")
    manifest = {
        "method": "pure_onboard_tau_est_distillation",
        "input_dim": INPUT_DIM,
        "base_dim": BASE_DIM,
        "effort_history": EFFORT_HISTORY,
        "joint_count": JOINTS,
        "foot_sensor_required": False,
        "magnetic_sensor_required": False,
        "policy": str((slot / "exported/policy.onnx").resolve()),
        "friction_estimator": str((slot / "exported/friction_estimator.onnx").resolve()),
    }
    (slot / "install_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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

    train_x, train_teacher, train_mu, train_cmd = load_pairs(args.train)
    augment_report: list[dict] = []
    for path in args.augment:
        extra_x, extra_teacher, extra_mu, extra_cmd = load_pairs(path)
        augment_report.append({"path": str(path.resolve()), "samples": int(len(extra_x))})
        for _ in range(max(args.augment_repeat, 1)):
            train_x = np.concatenate((train_x, extra_x))
            train_teacher = np.concatenate((train_teacher, extra_teacher))
            train_mu = np.concatenate((train_mu, extra_mu))
            train_cmd = np.concatenate((train_cmd, extra_cmd))
    test_x, test_teacher, test_mu, test_cmd = load_pairs(args.test)

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-4] = 1.0
    teacher = load_actor(args.teacher_onnx, 641).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"precomputing Teacher actions: train={len(train_x)} test={len(test_x)}", flush=True)
    train_target = make_teacher_targets(teacher, train_teacher, args.batch_size * 2, device)
    test_target = make_teacher_targets(teacher, test_teacher, args.batch_size * 2, device)

    model = ProprioTractionPolicy(mean, scale).to(device)
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_target),
            torch.from_numpy(train_mu),
            torch.from_numpy(train_cmd),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    action_weights = torch.ones(OUTPUT_DIM, device=device)
    action_weights[list(LEG_ACTION_INDICES)] = 2.0
    action_weights[list(WAIST_ACTION_INDICES)] = 1.5
    action_weight_sum = action_weights.sum()
    started = time.monotonic()
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for observation, target_action, target_mu, command_vx in loader:
            observation = observation.to(device, non_blocking=True)
            target_action = target_action.to(device, non_blocking=True)
            target_mu = target_mu.to(device, non_blocking=True)
            command_vx = command_vx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                predicted_action = model(observation)
                predicted_mu = model.predict_mu(observation)
                error = predicted_action - target_action
                per_sample = 0.70 * (
                    error.square() * action_weights
                ).sum(dim=1) / action_weight_sum + 0.30 * (
                    nn.functional.smooth_l1_loss(
                        predicted_action, target_action, beta=0.05, reduction="none"
                    ) * action_weights
                ).sum(dim=1) / action_weight_sum
                weight = torch.ones_like(target_mu)
                weight += (target_mu <= 0.25).float()
                weight += (target_mu >= 0.75).float()
                weight += 2.0 * ((target_mu >= 1.0) & (command_vx >= 0.8)).float()
                # Slow walking exposed the largest closed-loop lateral error;
                # oversample these recovery states without changing commands.
                weight += 1.5 * (command_vx <= 0.35).float()
                action_loss = torch.sum(per_sample * weight) / torch.sum(weight)
                friction_loss = nn.functional.smooth_l1_loss(
                    predicted_mu, target_mu, beta=0.08
                )
                loss = action_loss + args.mu_loss_coef * friction_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(float(np.mean(losses)))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.monotonic() - started
            eta = elapsed * (args.epochs - epoch) / max(epoch, 1)
            print(
                f"epoch={epoch:03d}/{args.epochs} loss={history[-1]:.7f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} ETA={eta/60:.1f}m",
                flush=True,
            )

    train_action_metrics, train_mu_metrics, _, _ = evaluate(
        model, train_x, train_target, train_mu, args.batch_size, device
    )
    test_action_metrics, test_mu_metrics, test_action, test_mu_prediction = evaluate(
        model, test_x, test_target, test_mu, args.batch_size, device
    )
    gates = {
        "action_mae": {
            "value": test_action_metrics["mae"],
            "target": "<= 0.045",
            "pass": test_action_metrics["mae"] <= 0.045,
        },
        "action_p95": {
            "value": test_action_metrics["p95_abs"],
            "target": "<= 0.12",
            "pass": test_action_metrics["p95_abs"] <= 0.12,
        },
        "friction_mae": {
            "value": test_mu_metrics["mae"],
            "target": "<= 0.18",
            "pass": test_mu_metrics["mae"] <= 0.18,
        },
        "friction_extreme_accuracy": {
            "value": test_mu_metrics["extreme_accuracy"],
            "target": ">= 0.80",
            "pass": test_mu_metrics["extreme_accuracy"] >= 0.80,
        },
        "finite": {
            "value": bool(np.isfinite(test_action).all() and np.isfinite(test_mu_prediction).all()),
            "target": "true",
            "pass": bool(np.isfinite(test_action).all() and np.isfinite(test_mu_prediction).all()),
        },
    }
    overall = all(item["pass"] for item in gates.values())
    report = {
        "method": "pure onboard 480-D proprioception + 15x29 tau_est history",
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "train_dataset": str(args.train.resolve()),
        "test_dataset": str(args.test.resolve()),
        "training_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "augmentation": augment_report,
        "epochs": int(args.epochs),
        "mu_loss_coef": float(args.mu_loss_coef),
        "final_loss": history[-1],
        "train": {"action": train_action_metrics, "friction": train_mu_metrics},
        "isaac_unseen": {"action": test_action_metrics, "friction": test_mu_metrics},
        "gates": gates,
        "overall": "PASS" if overall else "FAIL",
    }

    model = model.cpu().eval()
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "mean": mean,
            "scale": scale,
            "metrics": report,
        },
        args.output_dir / "proprio_policy.pt",
    )
    example = torch.zeros((1, INPUT_DIM), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save(str(args.output_dir / "policy.ts"))
    torch.onnx.export(
        model,
        example,
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )
    estimator = ProprioFrictionEstimator(model).eval()
    torch.onnx.export(
        estimator,
        example,
        args.output_dir / "friction_estimator.onnx",
        input_names=["observation"],
        output_names=["estimated_mu"],
        opset_version=17,
    )
    np.savez_compressed(
        args.output_dir / "isaac_unseen_predictions.npz",
        truth_mu=test_mu,
        predicted_mu=test_mu_prediction,
        command_vx=test_cmd,
        teacher_action=test_target,
        student_action=test_action,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    install_slot(args.output_dir, args.deploy_template, args.install_slot)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"overall={report['overall']} policy={args.output_dir / 'policy.onnx'}")
    return 3 if args.strict and not overall else 0


if __name__ == "__main__":
    raise SystemExit(main())
