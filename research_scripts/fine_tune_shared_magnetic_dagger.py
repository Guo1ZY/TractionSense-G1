#!/usr/bin/env python3
"""Offline Teacher distillation/DAgger on exact 1864-D Isaac trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from distill_traction_student import action_metrics, load_actor
from train_shared_magnetic_policy import (
    BASE_DIM,
    FEET,
    FUSED_DIM,
    HEALTH_DIM,
    HISTORY,
    INPUT_DIM,
    MAGNETIC_DIM,
    OLD_INPUT_DIM,
    OUTPUT_DIM,
    PERIOD_DIM,
    SENSORS,
    SharedMagneticPolicy,
    auxiliary_targets,
    progress,
)

AXES = 3
JOINTS = 29
JOINT_MIRROR_INDEX = torch.tensor(
    [
        2, 3, 0, 1, 4, 26, 8, 9, 6, 7, 10, 20, 14, 15, 12,
        13, 25, 27, 28, 23, 11, 24, 22, 19, 21, 16, 5, 17, 18,
    ],
    dtype=torch.long,
)
JOINT_MIRROR_SIGN = torch.tensor(
    [
        1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1,
    ],
    dtype=torch.float32,
)
LATERAL_JOINTS = (4, 6, 8, 12, 13, 14, 15, 22)
COMMAND_VX_INDICES = (30, 33, 36, 39, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--augment-repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8110)
    parser.add_argument("--mirror-augmentation", action="store_true")
    parser.add_argument(
        "--motion-feedback",
        action="store_true",
        help=(
            "Interpret obs[1862:1864] as [body_vy, relative_heading]. "
            "Sagittal mirroring negates both values."
        ),
    )
    parser.add_argument("--lateral-joint-weight", type=float, default=1.0)
    parser.add_argument("--symmetry-coef", type=float, default=0.0)
    parser.add_argument("--high-mu-command-scale", type=float, default=1.0)
    parser.add_argument("--high-mu-threshold", type=float, default=0.75)
    parser.add_argument("--high-command-threshold", type=float, default=0.70)
    parser.add_argument("--high-mu-sample-weight", type=float, default=4.0)
    parser.add_argument(
        "--teacher-mix-low",
        type=float,
        default=1.0,
        help="Teacher residual fraction outside the high-mu/high-command region.",
    )
    parser.add_argument(
        "--teacher-mix-high",
        type=float,
        default=1.0,
        help="Teacher residual fraction in the high-mu/high-command region.",
    )
    parser.add_argument(
        "--base-action-coef",
        type=float,
        default=0.0,
        help="Extra behavior-proximal loss that anchors the original Student.",
    )
    parser.add_argument(
        "--freeze-foot-encoder",
        action="store_true",
        help="Preserve the learned magnetic representation during short actor adaptation.",
    )
    parser.add_argument(
        "--actor-head-only",
        action="store_true",
        help="Train only the final 128->29 actor layer for the safest continuation.",
    )
    parser.add_argument(
        "--foot-encoder-only",
        action="store_true",
        help=(
            "Freeze the actor and auxiliary heads; calibrate only the shared "
            "dual-foot encoder against Sim2Sim trajectories."
        ),
    )
    parser.add_argument(
        "--auxiliary-coef-scale",
        type=float,
        default=1.0,
        help="Scale mu/contact/force auxiliary losses; use zero for action-only trust adaptation.",
    )
    parser.add_argument(
        "--mu-loss-coef",
        type=float,
        default=0.05,
        help="Friction-regression coefficient before --auxiliary-coef-scale.",
    )
    parser.add_argument(
        "--contact-loss-coef",
        type=float,
        default=0.01,
        help="Foot-contact auxiliary coefficient before --auxiliary-coef-scale.",
    )
    parser.add_argument(
        "--force-loss-coef",
        type=float,
        default=0.02,
        help="Foot-force auxiliary coefficient before --auxiliary-coef-scale.",
    )
    parser.add_argument("--target-action-clip", type=float, default=4.0)
    parser.add_argument(
        "--dagger-priority-cap",
        type=float,
        default=8.0,
        help="Maximum per-sample failure/recovery priority read from NPZ.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def mirror_joints(value: torch.Tensor) -> torch.Tensor:
    """Reflect Isaac-ordered G1 joint values/actions across the sagittal plane."""

    index = JOINT_MIRROR_INDEX.to(value.device)
    sign = JOINT_MIRROR_SIGN.to(device=value.device, dtype=value.dtype)
    return value.index_select(-1, index) * sign


def mirror_observation(
    observation: torch.Tensor,
    motion_feedback: bool = False,
) -> torch.Tensor:
    """Left/right mirror a flat 1864-D deploy observation."""

    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(f"expected Nx{INPUT_DIM}, got {tuple(observation.shape)}")
    mirrored = observation.clone()
    mirrored[:, 0:15] = (
        observation[:, 0:15].reshape(-1, 5, 3)
        * observation.new_tensor([-1.0, 1.0, -1.0])
    ).reshape(-1, 15)
    mirrored[:, 15:30] = (
        observation[:, 15:30].reshape(-1, 5, 3)
        * observation.new_tensor([1.0, -1.0, 1.0])
    ).reshape(-1, 15)
    mirrored[:, 30:45] = (
        observation[:, 30:45].reshape(-1, 5, 3)
        * observation.new_tensor([1.0, -1.0, -1.0])
    ).reshape(-1, 15)
    for start in (45, 190, 335):
        joint_history = observation[:, start : start + 5 * JOINTS].reshape(
            -1, 5, JOINTS
        )
        mirrored[:, start : start + 5 * JOINTS] = mirror_joints(
            joint_history
        ).reshape(-1, 5 * JOINTS)

    magnetic_start = BASE_DIM
    magnetic_end = magnetic_start + MAGNETIC_DIM
    magnetic = observation[:, magnetic_start:magnetic_end].reshape(
        -1, HISTORY, FEET, SENSORS, AXES
    )
    mirrored[:, magnetic_start:magnetic_end] = magnetic[:, :, [1, 0]].reshape(
        -1, MAGNETIC_DIM
    )
    period = observation[:, magnetic_end : magnetic_end + PERIOD_DIM].reshape(
        -1, HISTORY, FEET
    )
    mirrored[:, magnetic_end : magnetic_end + PERIOD_DIM] = period[:, :, [1, 0]].reshape(
        -1, PERIOD_DIM
    )
    if motion_feedback:
        # Current motion Student schema:
        # [left_valid, right_valid, body_vy, relative_heading].
        mirrored[:, 1860] = observation[:, 1861]
        mirrored[:, 1861] = observation[:, 1860]
        mirrored[:, 1862] = -observation[:, 1862]
        mirrored[:, 1863] = -observation[:, 1863]
    else:
        # Legacy schema: [left_valid, right_valid, left_age, right_age].
        health = observation[:, -HEALTH_DIM:].reshape(-1, 2, FEET)
        mirrored[:, -HEALTH_DIM:] = health[:, :, [1, 0]].reshape(
            -1, HEALTH_DIM
        )
    return mirrored


def weighted_action_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    joint_weight: torch.Tensor,
) -> torch.Tensor:
    """Per-sample squared/Smooth-L1 imitation loss with joint weighting."""

    error = predicted - target
    normalizer = joint_weight.sum()
    squared = (error.square() * joint_weight).sum(dim=1) / normalizer
    robust = (
        nn.functional.smooth_l1_loss(
            predicted,
            target,
            beta=0.05,
            reduction="none",
        )
        * joint_weight
    ).sum(dim=1) / normalizer
    return 0.7 * squared + 0.3 * robust


def load(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        obs = np.asarray(data["obs"], dtype=np.float32)
        teacher_obs = np.asarray(data["teacher_obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        cmd = np.asarray(data["cmd_vx"], dtype=np.float32).reshape(-1)
        priority = (
            np.asarray(data["sample_weight"], dtype=np.float32).reshape(-1)
            if "sample_weight" in data
            else np.ones(len(obs), dtype=np.float32)
        )
    if obs.ndim != 2 or obs.shape[1] != INPUT_DIM:
        raise ValueError(f"{path}: expected Nx{INPUT_DIM}, got {obs.shape}")
    if teacher_obs.ndim != 2 or teacher_obs.shape[1] != OLD_INPUT_DIM + 1:
        raise ValueError(f"{path}: expected Teacher Nx641, got {teacher_obs.shape}")
    finite = (
        np.isfinite(obs).all(axis=1)
        & np.isfinite(teacher_obs).all(axis=1)
        & np.isfinite(mu)
        & np.isfinite(cmd)
        & np.isfinite(priority)
        & (priority > 0.0)
    )
    return (
        obs[finite],
        teacher_obs[finite],
        mu[finite],
        cmd[finite],
        priority[finite],
    )


def teacher_actions(
    teacher: nn.Module,
    observations: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[start : start + batch_size]).to(device)
            outputs.append(teacher(batch).cpu().numpy())
    return np.concatenate(outputs)


def conditioned_teacher_observations(
    observations: np.ndarray,
    mu: np.ndarray,
    command_vx: np.ndarray,
    command_scale: float,
    mu_threshold: float,
    command_threshold: float,
) -> tuple[np.ndarray, int]:
    """Boost only the Teacher's high-grip command while keeping deploy input unchanged."""

    conditioned = observations.copy()
    mask = (mu >= mu_threshold) & (command_vx >= command_threshold)
    if command_scale != 1.0 and np.any(mask):
        conditioned[np.ix_(mask, COMMAND_VX_INDICES)] *= command_scale
        conditioned[np.ix_(mask, COMMAND_VX_INDICES)] = np.clip(
            conditioned[np.ix_(mask, COMMAND_VX_INDICES)], -1.5, 1.5
        )
    return conditioned, int(mask.sum())


def teacher_mix_weights(
    mu: np.ndarray,
    command_vx: np.ndarray,
    low: float,
    high: float,
    mu_threshold: float,
    command_threshold: float,
) -> np.ndarray:
    """Smooth trust-region gate for adding Teacher residuals to base actions."""

    mu_gate = 1.0 / (1.0 + np.exp(-(mu - mu_threshold) / 0.08))
    command_gate = 1.0 / (
        1.0 + np.exp(-(command_vx - command_threshold) / 0.08)
    )
    return (low + (high - low) * mu_gate * command_gate).astype(np.float32)


def evaluate(
    model: SharedMagneticPolicy,
    obs: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(obs), batch_size):
            batch = torch.from_numpy(obs[start : start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions)
    return action_metrics(target, prediction), prediction


def configure_trainable_parameters(
    model: SharedMagneticPolicy,
    *,
    freeze_foot_encoder: bool,
    actor_head_only: bool,
    foot_encoder_only: bool,
) -> None:
    """Apply one explicit trust-region parameter-selection mode."""

    if actor_head_only and foot_encoder_only:
        raise ValueError("--actor-head-only and --foot-encoder-only are mutually exclusive")
    if freeze_foot_encoder:
        for parameter in model.foot_encoder.parameters():
            parameter.requires_grad_(False)
    if actor_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.actor[6].parameters():
            parameter.requires_grad_(True)
    if foot_encoder_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.foot_encoder.parameters():
            parameter.requires_grad_(True)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_obs, train_teacher_obs, train_mu, train_cmd, train_priority = load(
        args.train
    )
    augmentation = []
    for path in args.augment:
        (
            extra_obs,
            extra_teacher_obs,
            extra_mu,
            extra_cmd,
            extra_priority,
        ) = load(path)
        repeats = max(args.augment_repeat, 1)
        augmentation.append(
            {
                "path": str(path.resolve()),
                "samples": len(extra_obs),
                "repeat": repeats,
            }
        )
        for _ in range(repeats):
            train_obs = np.concatenate((train_obs, extra_obs))
            train_teacher_obs = np.concatenate(
                (train_teacher_obs, extra_teacher_obs)
            )
            train_mu = np.concatenate((train_mu, extra_mu))
            train_cmd = np.concatenate((train_cmd, extra_cmd))
            train_priority = np.concatenate(
                (train_priority, extra_priority)
            )
    (
        test_obs,
        test_teacher_obs,
        test_mu,
        test_cmd,
        test_priority,
    ) = load(args.test)
    if args.high_mu_command_scale < 1.0:
        raise ValueError("--high-mu-command-scale must be >= 1")
    if args.high_mu_sample_weight < 1.0:
        raise ValueError("--high-mu-sample-weight must be >= 1")
    if args.symmetry_coef < 0.0:
        raise ValueError("--symmetry-coef must be >= 0")
    if not 0.0 <= args.teacher_mix_low <= 1.0:
        raise ValueError("--teacher-mix-low must be in [0, 1]")
    if not 0.0 <= args.teacher_mix_high <= 1.0:
        raise ValueError("--teacher-mix-high must be in [0, 1]")
    if args.base_action_coef < 0.0:
        raise ValueError("--base-action-coef must be >= 0")
    if args.auxiliary_coef_scale < 0.0:
        raise ValueError("--auxiliary-coef-scale must be >= 0")
    if min(args.mu_loss_coef, args.contact_loss_coef, args.force_loss_coef) < 0.0:
        raise ValueError("auxiliary loss coefficients must be >= 0")
    if args.actor_head_only and args.foot_encoder_only:
        raise ValueError("--actor-head-only and --foot-encoder-only are mutually exclusive")
    train_teacher_target_obs, train_boosted_samples = conditioned_teacher_observations(
        train_teacher_obs,
        train_mu,
        train_cmd,
        args.high_mu_command_scale,
        args.high_mu_threshold,
        args.high_command_threshold,
    )
    test_teacher_target_obs, test_boosted_samples = conditioned_teacher_observations(
        test_teacher_obs,
        test_mu,
        test_cmd,
        args.high_mu_command_scale,
        args.high_mu_threshold,
        args.high_command_threshold,
    )
    teacher = load_actor(args.teacher_onnx, OLD_INPUT_DIM + 1).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_teacher_target = teacher_actions(
        teacher, train_teacher_target_obs, args.batch_size * 2, device
    )
    test_teacher_target = teacher_actions(
        teacher, test_teacher_target_obs, args.batch_size * 2, device
    )
    if args.target_action_clip <= 0.0:
        raise ValueError("--target-action-clip must be positive")
    if args.dagger_priority_cap < 1.0:
        raise ValueError("--dagger-priority-cap must be >= 1")
    train_priority = np.clip(
        train_priority, 1.0, args.dagger_priority_cap
    )
    test_priority = np.clip(
        test_priority, 1.0, args.dagger_priority_cap
    )
    train_teacher_target = np.clip(
        train_teacher_target, -args.target_action_clip, args.target_action_clip
    )
    test_teacher_target = np.clip(
        test_teacher_target, -args.target_action_clip, args.target_action_clip
    )

    payload = torch.load(args.base, map_location="cpu", weights_only=False)
    model = SharedMagneticPolicy().to(device)
    model.load_state_dict(payload["model"], strict=True)
    before_teacher, test_base_action = evaluate(
        model, test_obs, test_teacher_target, args.batch_size, device
    )
    _, train_base_action = evaluate(
        model, train_obs, train_teacher_target, args.batch_size, device
    )
    train_mix = teacher_mix_weights(
        train_mu,
        train_cmd,
        args.teacher_mix_low,
        args.teacher_mix_high,
        args.high_mu_threshold,
        args.high_command_threshold,
    )
    test_mix = teacher_mix_weights(
        test_mu,
        test_cmd,
        args.teacher_mix_low,
        args.teacher_mix_high,
        args.high_mu_threshold,
        args.high_command_threshold,
    )
    train_target = train_base_action + train_mix[:, None] * (
        train_teacher_target - train_base_action
    )
    test_target = test_base_action + test_mix[:, None] * (
        test_teacher_target - test_base_action
    )
    before, _ = evaluate(
        model, test_obs, test_target, args.batch_size, device
    )
    configure_trainable_parameters(
        model,
        freeze_foot_encoder=args.freeze_foot_encoder,
        actor_head_only=args.actor_head_only,
        foot_encoder_only=args.foot_encoder_only,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_obs),
            torch.from_numpy(train_teacher_obs[:, :OLD_INPUT_DIM]),
            torch.from_numpy(train_mu),
            torch.from_numpy(train_cmd),
            torch.from_numpy(train_priority),
            torch.from_numpy(train_base_action),
            torch.from_numpy(train_target),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1.0e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    joint_weight = torch.ones(JOINTS, device=device)
    if args.lateral_joint_weight < 1.0:
        raise ValueError("--lateral-joint-weight must be >= 1")
    joint_weight[list(LATERAL_JOINTS)] = args.lateral_joint_weight
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.monotonic()
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for (
            obs,
            old,
            target_mu,
            command_vx,
            dagger_priority,
            base_action,
            target_action,
        ) in loader:
            obs = obs.to(device, non_blocking=True)
            old = old.to(device, non_blocking=True)
            target_mu = target_mu.to(device, non_blocking=True)
            command_vx = command_vx.to(device, non_blocking=True)
            dagger_priority = dagger_priority.to(
                device, non_blocking=True
            )
            base_action = base_action.to(device, non_blocking=True)
            target_action = target_action.to(device, non_blocking=True)
            target_feet = auxiliary_targets(old)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                predicted_action = model(obs)
                predicted_mu, predicted_feet = model.auxiliary(obs)
                per_sample = weighted_action_error(
                    predicted_action, target_action, joint_weight
                )
                if args.mirror_augmentation:
                    mirrored_obs = mirror_observation(
                        obs, motion_feedback=args.motion_feedback
                    )
                    mirrored_target = mirror_joints(target_action)
                    mirrored_action = model(mirrored_obs)
                    per_sample = 0.5 * (
                        per_sample
                        + weighted_action_error(
                            mirrored_action, mirrored_target, joint_weight
                        )
                    )
                    symmetry_loss = nn.functional.smooth_l1_loss(
                        mirror_joints(mirrored_action),
                        predicted_action,
                        beta=0.03,
                    )
                else:
                    symmetry_loss = predicted_action.new_zeros(())
                weight = torch.ones_like(target_mu)
                weight += (target_mu <= 0.25).float()
                weight += (target_mu >= 0.75).float()
                high_mu_high_command = (
                    (target_mu >= args.high_mu_threshold)
                    & (command_vx >= args.high_command_threshold)
                )
                weight += (args.high_mu_sample_weight - 1.0) * (
                    high_mu_high_command.float()
                )
                weight *= dagger_priority
                action_loss = torch.sum(per_sample * weight) / torch.sum(weight)
                base_action_loss = weighted_action_error(
                    predicted_action, base_action, joint_weight
                ).mean()
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
                loss = (
                    action_loss
                    + args.base_action_coef * base_action_loss
                    + args.symmetry_coef * symmetry_loss
                    + args.auxiliary_coef_scale
                    * (
                        args.mu_loss_coef * mu_loss
                        + args.contact_loss_coef * contact_loss
                        + args.force_loss_coef * force_loss
                    )
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        print(progress(epoch, args.epochs, epoch_loss, started), flush=True)
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": (
                        "training" if epoch < args.epochs else "evaluating"
                    ),
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
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "input_dim": INPUT_DIM,
                "fused_dim": FUSED_DIM,
                "output_dim": OUTPUT_DIM,
            },
            args.output_dir / "checkpoint_latest.pt",
        )

    after, prediction = evaluate(
        model, test_obs, test_target, args.batch_size, device
    )
    after_teacher, _ = evaluate(
        model, test_obs, test_teacher_target, args.batch_size, device
    )
    gates = {
        "action_mae": {
            "value": after["mae"],
            "target": "<= 0.035",
            "pass": after["mae"] <= 0.035,
        },
        "action_p95": {
            "value": after["p95_abs"],
            "target": "<= 0.090",
            "pass": after["p95_abs"] <= 0.090,
        },
        "improved_mae": {
            "value": after["mae"] - before["mae"],
            "target": "< 0",
            "pass": after["mae"] < before["mae"],
        },
        "finite": {
            "value": bool(np.isfinite(prediction).all()),
            "target": "true",
            "pass": bool(np.isfinite(prediction).all()),
        },
    }
    report = {
        "method": "exact 1864-D Isaac offline Teacher distillation/DAgger",
        "architecture": {
            "external_input_dim": INPUT_DIM,
            "shared_per_foot_latent_dim": 32,
            "fused_actor_dim": FUSED_DIM,
            "output_dim": OUTPUT_DIM,
            "shared_foot_encoder": True,
        },
        "observation_tail": (
            [
                "left_valid",
                "right_valid",
                "body_vy",
                "relative_heading",
            ]
            if args.motion_feedback
            else ["left_valid", "right_valid", "left_age", "right_age"]
        ),
        "motion_feedback": args.motion_feedback,
        "base": str(args.base.resolve()),
        "teacher_onnx": str(args.teacher_onnx.resolve()),
        "train": str(args.train.resolve()),
        "test": str(args.test.resolve()),
        "augmentation": augmentation,
        "training_samples": len(train_obs),
        "test_samples": len(test_obs),
        "dagger_priority": {
            "cap": args.dagger_priority_cap,
            "train_mean": float(train_priority.mean()),
            "train_max": float(train_priority.max()),
            "train_prioritized_samples": int(
                np.count_nonzero(train_priority > 1.0)
            ),
            "test_mean": float(test_priority.mean()),
            "test_max": float(test_priority.max()),
            "test_prioritized_samples": int(
                np.count_nonzero(test_priority > 1.0)
            ),
        },
        "epochs": args.epochs,
        "mirror_augmentation": args.mirror_augmentation,
        "lateral_joint_weight": args.lateral_joint_weight,
        "symmetry_coef": args.symmetry_coef,
        "high_mu_command_scale": args.high_mu_command_scale,
        "high_mu_threshold": args.high_mu_threshold,
        "high_command_threshold": args.high_command_threshold,
        "high_mu_sample_weight": args.high_mu_sample_weight,
        "teacher_mix_low": args.teacher_mix_low,
        "teacher_mix_high": args.teacher_mix_high,
        "teacher_mix_mean": {
            "train": float(train_mix.mean()),
            "test": float(test_mix.mean()),
        },
        "base_action_coef": args.base_action_coef,
        "freeze_foot_encoder": args.freeze_foot_encoder,
        "actor_head_only": args.actor_head_only,
        "foot_encoder_only": args.foot_encoder_only,
        "auxiliary_coef_scale": args.auxiliary_coef_scale,
        "mu_loss_coef": args.mu_loss_coef,
        "contact_loss_coef": args.contact_loss_coef,
        "force_loss_coef": args.force_loss_coef,
        "boosted_samples": {
            "train": train_boosted_samples,
            "test": test_boosted_samples,
        },
        "target_action_clip": args.target_action_clip,
        "before": before,
        "after": after,
        "teacher_action_metrics": {
            "before": before_teacher,
            "after": after_teacher,
        },
        "final_loss": losses[-1],
        "gates": gates,
        "overall": "PASS" if all(item["pass"] for item in gates.values()) else "FAIL",
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
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete" if report["overall"] == "PASS" else "evaluation_failed",
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
    return 0 if report["overall"] == "PASS" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
