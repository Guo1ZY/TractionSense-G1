#!/usr/bin/env python3
"""Bootstrap a smooth LOW capture residual while preserving speedboost112.

The dataset contains only deployable 1864-D Hall/proprio observations plus
actions.  Its privileged ``low`` bit is a training label and is not an input
to the exported policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureResidual,
    RslActorMean,
    trainable_parameters,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    INPUT_DIM,
    OUTPUT_DIM,
    load_frozen_speedboost_teacher,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations, actions, low = [], [], []
    for path in paths:
        data = np.load(path)
        obs = np.asarray(data["observation"], dtype=np.float32)
        act = np.asarray(data["action"], dtype=np.float32)
        label = np.asarray(data["low"], dtype=np.bool_)
        if obs.ndim != 2 or obs.shape[1] != INPUT_DIM:
            raise ValueError(f"{path}: observation must be [N,{INPUT_DIM}], got {obs.shape}")
        if act.shape != (len(obs), OUTPUT_DIM) or label.shape != (len(obs),):
            raise ValueError(f"{path}: inconsistent action/low shapes")
        finite = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
        observations.append(obs[finite])
        actions.append(act[finite])
        low.append(label[finite])
    return np.concatenate(observations), np.concatenate(actions), np.concatenate(low)


def _load_actor(path: Path, device: torch.device) -> RslActorMean:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError(f"{path} is not an RSL actor checkpoint")
    actor = RslActorMean()
    actor.load_rsl_state_dict(payload["actor_state_dict"])
    actor.eval().to(device)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _batched(module, values: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            outputs.append(module(values[begin : begin + batch_size]).detach().clone())
    return torch.cat(outputs)


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool().cpu()
    scores = scores.float().cpu()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--frozen-teacher", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path)
    parser.add_argument("--spatial-blend", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--residual-limit", type=float, default=1.25)
    parser.add_argument("--gate-power", type=float, default=1.5)
    parser.add_argument("--teacher-trailing-mode", choices=("passthrough", "assume_fresh"), default="passthrough")
    parser.add_argument("--low-weight", type=float, default=4.0)
    parser.add_argument("--gate-weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.spatial_blend <= 1.0:
        parser.error("--spatial-blend must be in [0,1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    obs_np, _, low_np = _load_dataset(args.dataset)
    observation = torch.from_numpy(obs_np).to(device)
    low = torch.from_numpy(low_np).to(device)

    teacher = load_frozen_speedboost_teacher(args.frozen_teacher, device=device)
    model = FastBaseHallCaptureResidual(
        teacher,
        residual_limit=args.residual_limit,
        gate_power=args.gate_power,
        teacher_trailing_mode=args.teacher_trailing_mode,
    ).to(device)
    recovery = _load_actor(args.recovery_checkpoint, device)
    spatial = _load_actor(args.spatial_checkpoint, device) if args.spatial_checkpoint else None

    # Precompute targets once.  Teacher inference is intentionally not repeated
    # inside every optimization minibatch.
    base = _batched(model.base_action, observation, args.batch_size)
    recovery_action = _batched(recovery, observation, args.batch_size).clamp(-3.0, 3.0)
    target_action = recovery_action
    if spatial is not None and args.spatial_blend > 0.0:
        spatial_action = _batched(spatial, observation, args.batch_size).clamp(-3.0, 3.0)
        target_action = torch.lerp(recovery_action, spatial_action, args.spatial_blend)
    target_delta = (target_action - base).clamp(-args.residual_limit, args.residual_limit)
    target_delta = torch.where(low[:, None], target_delta, torch.zeros_like(target_delta))

    permutation = torch.randperm(len(observation), device=device)
    split = max(1, int(0.85 * len(permutation)))
    train_ids, test_ids = permutation[:split], permutation[split:]
    positive_fraction = float(low[train_ids].float().mean())
    pos_weight = torch.tensor((1.0 - positive_fraction) / max(positive_fraction, 1.0e-4), device=device)
    optimizer = torch.optim.AdamW(list(trainable_parameters(model)), lr=args.learning_rate, weight_decay=1.0e-5)
    history = []

    for epoch in range(args.epochs):
        model.train()
        shuffled = train_ids[torch.randperm(len(train_ids), device=device)]
        total = 0.0
        for begin in range(0, len(shuffled), args.batch_size):
            ids = shuffled[begin : begin + args.batch_size]
            predicted = model.capture_delta(observation[ids])
            weights = torch.where(low[ids], args.low_weight, 1.0).unsqueeze(1)
            delta_loss = (weights * (predicted - target_delta[ids]).square()).mean()
            gate_logits = model.gate(observation[ids]).squeeze(1)
            gate_loss = nn.functional.binary_cross_entropy_with_logits(
                gate_logits, low[ids].float(), pos_weight=pos_weight
            )
            loss = delta_loss + args.gate_weight * gate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            predicted = model.capture_delta(observation[test_ids])
            probability = model.capture_probability(observation[test_ids]).squeeze(1)
            low_test = low[test_ids]
            low_mse = (
                (predicted[low_test] - target_delta[test_ids][low_test]).square().mean().item()
                if low_test.any()
                else float("nan")
            )
            high_rms = (
                predicted[~low_test].square().mean().sqrt().item()
                if (~low_test).any()
                else float("nan")
            )
            gate_auc = _auc(low_test, probability)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train_ids),
                "test_low_delta_mse": low_mse,
                "test_high_delta_rms": high_rms,
                "test_gate_auc": gate_auc,
            }
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(json.dumps(history[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "fastbase_capture_residual.pt"
    torch.save(
        {
            "format": "unitree_rl_lab.fastbase_capture_residual",
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "residual_limit": args.residual_limit,
            "gate_power": args.gate_power,
            "teacher_trailing_mode": args.teacher_trailing_mode,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    policy_onnx = args.output_dir / "policy.onnx"
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.zeros(1, INPUT_DIM),
        policy_onnx,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    report = {
        "status": "BOOTSTRAP_ONLY_NOT_ACCEPTED_FOR_HARDWARE",
        "architecture": "frozen_speedboost112_plus_hall_gated_bounded_capture_residual",
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "teacher_frozen": not any(p.requires_grad for p in model.teacher.parameters()),
        "teacher_trailing_mode": args.teacher_trailing_mode,
        "residual_limit": args.residual_limit,
        "gate_power": args.gate_power,
        "dataset_samples": len(observation),
        "low_samples": int(low.sum()),
        "checkpoint": str(checkpoint),
        "onnx": str(policy_onnx),
        "source_sha256": {
            "frozen_teacher": _sha256(args.frozen_teacher),
            "recovery_checkpoint": _sha256(args.recovery_checkpoint),
            "spatial_checkpoint": _sha256(args.spatial_checkpoint) if args.spatial_checkpoint else None,
            "datasets": {str(path): _sha256(path) for path in args.dataset},
        },
        "forbidden_actor_inputs": ["friction", "contact_force", "slip_truth", "course_stage"],
        "final_metrics": history[-1],
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
