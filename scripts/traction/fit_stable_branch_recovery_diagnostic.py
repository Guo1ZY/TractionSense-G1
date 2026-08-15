#!/usr/bin/env python3
"""Fit the bounded stability residual toward the frozen stable branch.

This is an explicitly diagnostic, non-release checkpoint transform.  It uses
only saved deployable 1864-D observations plus evaluator stage labels to test
whether the already embedded ``speedboost112.stable`` branch supplies a useful
recovery direction.  A successful same-rollout test justifies an online
multi-seed curriculum; it is never sufficient for deployment acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureStabilityResidual,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    load_frozen_speedboost_teacher,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--target-cap", type=float, default=0.20)
    parser.add_argument("--authority-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    for path in (args.checkpoint, args.dataset, args.teacher):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch-size must be positive")
    if not 0.0 < args.learning_rate <= 1.0e-2:
        raise ValueError("learning-rate must be in (0, 1e-2]")
    if not 0.0 < args.target_cap <= 0.25:
        raise ValueError("target-cap must be in (0, 0.25]")
    if not 0.0 <= args.authority_min < 1.0:
        raise ValueError("authority-min must be in [0, 1)")

    torch.manual_seed(args.seed)
    source_path = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    teacher_path = args.teacher.expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise RuntimeError("checkpoint is missing actor_state_dict")

    teacher = load_frozen_speedboost_teacher(teacher_path, device="cpu")
    model = FastBaseHallCaptureStabilityResidual(
        teacher,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        stability_limit=0.25,
        stability_heading_start=0.25,
        stability_heading_full=0.55,
        stability_tilt_start=0.08,
        stability_tilt_full=0.25,
        stability_omega_start=0.60,
        stability_omega_full=1.80,
        stability_turning_yaw_threshold=0.05,
    )
    prefix = "mlp."
    mean_state = {
        key[len(prefix) :]: value
        for key, value in actor_state.items()
        if key.startswith(prefix)
    }
    result = model.load_state_dict(mean_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict actor load failed: {result}")
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.stability_residual.parameters():
        parameter.requires_grad_(True)

    with np.load(dataset_path, allow_pickle=False) as data:
        observation = torch.from_numpy(data["observation"]).float()
        stage = torch.from_numpy(data["fastbase_course_stage"]).long()
    if observation.ndim != 2 or observation.shape[1] != 1864:
        raise RuntimeError(f"expected observation [N,1864], got {tuple(observation.shape)}")
    if stage.shape != (observation.shape[0],):
        raise RuntimeError("course stage is not row-aligned with observations")
    if not torch.isfinite(observation).all():
        raise FloatingPointError("dataset observation contains NaN/Inf")

    with torch.inference_mode():
        authority = model.stability_authority(observation)
        teacher_observation = model.teacher_observation(observation)
        teacher_model = model.teacher
        boost = (
            (teacher_model.config.boost_factor - 1.0)
            * teacher_model.boost_gate(teacher_observation)
        ).unsqueeze(1)
        boosted = teacher_observation * (
            1.0 + boost * teacher_model.command_mask.unsqueeze(0)
        )
        stable_action = teacher_model.stable(boosted)
        anchor_action = model.anchor_action_without_stability(observation)
        target = (stable_action - anchor_action).clamp(
            -float(args.target_cap), float(args.target_cap)
        )
        effective_target = authority * target
        features = model.stability_features(observation)
        mask = (stage != 1) & (authority[:, 0] >= float(args.authority_min))
    if int(mask.sum()) < args.batch_size:
        raise RuntimeError(
            f"only {int(mask.sum())} eligible HIGH-risk rows for batch {args.batch_size}"
        )
    features = features[mask]
    authority = authority[mask]
    effective_target = effective_target[mask]

    optimizer = torch.optim.Adam(
        model.stability_residual.parameters(), lr=float(args.learning_rate)
    )
    with torch.inference_mode():
        initial = authority * model.stability_limit * torch.tanh(
            model.stability_residual(features)
        )
        initial_loss = float(
            F.smooth_l1_loss(initial, effective_target, beta=0.05).item()
        )
    for _ in range(args.updates):
        index = torch.randint(0, features.shape[0], (args.batch_size,))
        prediction = authority[index] * model.stability_limit * torch.tanh(
            model.stability_residual(features[index])
        )
        loss = F.smooth_l1_loss(
            prediction, effective_target[index], beta=0.05
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stability_residual.parameters(), 1.0)
        optimizer.step()
    with torch.inference_mode():
        final = authority * model.stability_limit * torch.tanh(
            model.stability_residual(features)
        )
        final_loss = float(
            F.smooth_l1_loss(final, effective_target, beta=0.05).item()
        )
        max_abs = float(final.abs().max().item())
    if not np.isfinite((initial_loss, final_loss, max_abs)).all():
        raise FloatingPointError("diagnostic fit produced NaN/Inf")

    trained_state = model.state_dict()
    for name, value in trained_state.items():
        if name.startswith("stability_residual."):
            actor_state[prefix + name] = value.detach().clone()
    report = {
        "format": "unitree_rl_lab.stable_branch_recovery_diagnostic",
        "release_candidate": False,
        "same_rollout_mechanism_test_only": True,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "teacher": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "eligible_rows": int(features.shape[0]),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "target_cap": float(args.target_cap),
        "authority_min": float(args.authority_min),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": final_loss / max(initial_loss, 1.0e-12),
        "maximum_effective_action_delta": max_abs,
        "actor_input": "policy[1864] Hall+proprio only",
        "privileged_stage_use": "training mask only; absent from actor/export",
    }
    infos = dict(payload.get("infos") or {})
    infos["stable_branch_recovery_diagnostic"] = report
    payload["infos"] = infos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report["output_checkpoint"] = str(args.output.resolve())
    report["output_checkpoint_sha256"] = _sha256(args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
