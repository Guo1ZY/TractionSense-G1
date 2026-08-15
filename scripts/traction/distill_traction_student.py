#!/usr/bin/env python3
"""Offline canonical Teacher-Student distillation with DAgger dataset mixing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.networks import (  # noqa: E402
    DistillationLossCfg,
    GatedTractionPolicy,
    TemporalStudentEncoderCfg,
    teacher_student_loss,
    temporal_history_to_legacy_proprio,
)
from unitree_rl_lab.traction.schema import (  # noqa: E402
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--baseline_checkpoint",
        type=Path,
        default=ROOT / "model" / "rl" / "model_49999.pt",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--slip_positive_weight",
        type=float,
        default=0.0,
        help="Positive slip BCE weight; <=0 selects neg/pos, capped at 20.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_datasets(paths: list[Path]) -> tuple[torch.Tensor, ...]:
    names = (
        "student_history",
        "teacher_action",
        "teacher_latent",
        "slip_label",
        "traction_target",
    )
    merged: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for name in names:
                if name not in data:
                    raise KeyError(f"{path}: missing {name}")
                merged[name].append(np.asarray(data[name], dtype=np.float32))
    arrays = [np.concatenate(merged[name], axis=0) for name in names]
    if arrays[0].shape[1] != TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension:
        raise ValueError(f"Student dimension changed to {arrays[0].shape[1]}")
    if arrays[1].shape[1] != 29 or arrays[2].shape[1] not in (8, 16):
        raise ValueError("Teacher action/latent dimensions are incompatible")
    return tuple(torch.from_numpy(array) for array in arrays)


def _load_baseline(policy: GatedTractionPolicy, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = {
        key: value
        for key, value in checkpoint["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    policy.baseline_actor.load_state_dict(state, strict=True)
    policy.baseline_actor.requires_grad_(False)


def _metrics(
    policy: GatedTractionPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "action_mse": 0.0,
        "latent_mse": 0.0,
        "slip_accuracy": 0.0,
        "traction_mse": 0.0,
        "confidence_mse": 0.0,
    }
    slip_labels: list[np.ndarray] = []
    slip_probabilities: list[np.ndarray] = []
    samples = 0
    policy.eval()
    with torch.inference_mode():
        for history_flat, action, latent, slip, traction in loader:
            history = history_flat.to(device).reshape(
                -1,
                TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
                TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
            )
            output = policy(
                temporal_history_to_legacy_proprio(history),
                history,
                history[:, -1, 93:96],
            )
            batch = history.shape[0]
            totals["action_mse"] += (
                (output.action_mean - action.to(device)).square().mean().item() * batch
            )
            totals["latent_mse"] += (
                (output.latent - latent.to(device)).square().mean().item() * batch
            )
            totals["slip_accuracy"] += (
                ((output.slip_probability > 0.5) == (slip.to(device) > 0.5))
                .float()
                .mean()
                .item()
                * batch
            )
            totals["traction_mse"] += (
                (output.traction_score - traction.to(device)).square().mean().item()
                * batch
            )
            valid = history[:, -1, 102:104]
            age = history[:, -1, 104:106].clamp_min(0.0)
            confidence_target = (
                valid.mean(dim=-1, keepdim=True)
                * torch.exp(-age.mean(dim=-1, keepdim=True) / 0.10)
            )
            totals["confidence_mse"] += (
                (output.sensor_confidence - confidence_target).square().mean().item()
                * batch
            )
            slip_labels.append(slip.numpy())
            slip_probabilities.append(output.slip_probability.cpu().numpy())
            samples += batch
    result = {name: value / max(samples, 1) for name, value in totals.items()}
    label = np.concatenate(slip_labels).reshape(-1) > 0.5
    probability = np.concatenate(slip_probabilities).reshape(-1)
    predicted = probability >= 0.5
    true_positive = int(np.count_nonzero(label & predicted))
    false_positive = int(np.count_nonzero(~label & predicted))
    false_negative = int(np.count_nonzero(label & ~predicted))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    result["slip_precision"] = precision
    result["slip_recall"] = recall
    result["slip_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    positive = int(label.sum())
    negative = len(label) - positive
    if positive and negative:
        order = np.argsort(probability, kind="mergesort")
        sorted_probability = probability[order]
        rank = np.arange(1, len(probability) + 1, dtype=np.float64)
        boundaries = np.r_[
            0,
            1 + np.flatnonzero(np.diff(sorted_probability)),
            len(probability),
        ]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            rank[start:stop] = rank[start:stop].mean()
        unsorted_rank = np.empty_like(rank)
        unsorted_rank[order] = rank
        result["slip_auc"] = float(
            (
                unsorted_rank[label].sum()
                - positive * (positive + 1) / 2
            )
            / (positive * negative)
        )
    else:
        result["slip_auc"] = float("nan")
    return result


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0,0.5)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensors = _load_datasets(args.datasets)
    positive = float(tensors[3].sum().item())
    negative = float(tensors[3].numel() - positive)
    slip_positive_weight = (
        args.slip_positive_weight
        if args.slip_positive_weight > 0.0
        else min(20.0, max(1.0, negative / max(positive, 1.0)))
    )
    loss_cfg = DistillationLossCfg(
        slip_positive_weight=slip_positive_weight
    )
    dataset = TensorDataset(*tensors)
    validation_size = max(1, int(len(dataset) * args.validation_fraction))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(
        dataset,
        (train_size, validation_size),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    policy = GatedTractionPolicy(
        encoder_cfg=TemporalStudentEncoderCfg(latent_dim=tensors[2].shape[1])
    ).to(device)
    _load_baseline(policy, args.baseline_checkpoint)
    optimizer = torch.optim.AdamW(
        (
            parameter
            for name, parameter in policy.named_parameters()
            if not name.startswith("baseline_actor.")
        ),
        lr=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training_metrics.csv"
    best_validation = float("inf")
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        samples = 0
        maximum_gradient = 0.0
        for history_flat, action, latent, slip, traction in train_loader:
            history = history_flat.to(device).reshape(
                -1,
                TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
                TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
            )
            output = policy(
                temporal_history_to_legacy_proprio(history),
                history,
                history[:, -1, 93:96],
            )
            valid = history[:, -1, 102:104]
            age = history[:, -1, 104:106].clamp_min(0.0)
            confidence_target = (
                valid.mean(dim=-1, keepdim=True)
                * torch.exp(-age.mean(dim=-1, keepdim=True) / 0.10)
            )
            losses = teacher_student_loss(
                ppo_loss=torch.zeros((), device=device),
                student=output,
                teacher_latent=latent.to(device),
                teacher_action=action.to(device),
                slip_label=slip.to(device),
                traction_target=traction.to(device),
                confidence_target=confidence_target,
                cfg=loss_cfg,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            optimizer.step()
            maximum_gradient = max(maximum_gradient, float(gradient_norm.item()))
            epoch_loss += losses.total.item() * history.shape[0]
            samples += history.shape[0]

        validation = _metrics(policy, validation_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": epoch_loss / max(samples, 1),
            "maximum_preclip_gradient_norm": maximum_gradient,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history_rows.append(row)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history_rows)
        checkpoint = {
            "student_policy_state_dict": policy.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "datasets": [str(path.resolve()) for path in args.datasets],
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            "metrics": row,
            "schema_version": TEMPORAL_STUDENT_FRAME_SCHEMA.schema_version,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if validation["action_mse"] < best_validation:
            best_validation = validation["action_mse"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(row), flush=True)

    summary = {
        "samples": len(dataset),
        "train_samples": train_size,
        "validation_samples": validation_size,
        "epochs": args.epochs,
        "best_validation_action_mse": best_validation,
        "device": str(device),
        "slip_positive_weight": slip_positive_weight,
        "slip_positive_fraction": positive / max(positive + negative, 1.0),
        "note": "Metrics describe this supplied dataset only; they are not locomotion evaluation.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
