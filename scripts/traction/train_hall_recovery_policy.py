#!/usr/bin/env python3
"""Train a bounded Hall-only recovery residual on frozen locomotion networks.

Privileged Teacher observations and true friction are offline labels only.  The
exported policy receives exactly the deployment Hall/proprioceptive observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_layout_magnetic_student import (  # noqa: E402
    FrozenTeacher,
    apply_foot_dropout_metadata,
    randomize_hall,
)
from unitree_rl_lab.traction.hall_recovery_policy import HallRecoveryPolicy  # noqa: E402
from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    build_hall_risk_estimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    ACTION_OUTPUT_LIMIT,
    INPUT_DIM,
    LayoutMagneticStudent,
    normalize_trailing_feature_mode,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_part(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("obs", "teacher_obs", "mu")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        obs = np.asarray(data["obs"], dtype=np.float32)
        teacher_obs = np.asarray(data["teacher_obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        count = len(obs)
        weight = np.asarray(
            data["sample_weight"] if "sample_weight" in data else np.ones(count),
            dtype=np.float32,
        ).reshape(-1)
        if "time_since_switch_s" in data:
            transition = (
                np.asarray(data["time_since_switch_s"], dtype=np.float32).reshape(-1)
                <= 1.5
            ).astype(np.float32)
        else:
            transition = np.zeros(count, dtype=np.float32)
    if obs.shape != (len(obs), INPUT_DIM):
        raise ValueError(f"{path}: expected Nx{INPUT_DIM} obs, got {obs.shape}")
    if teacher_obs.shape != (len(obs), 641):
        raise ValueError(f"{path}: expected Nx641 teacher_obs, got {teacher_obs.shape}")
    finite = (
        np.isfinite(obs).all(axis=1)
        & np.isfinite(teacher_obs).all(axis=1)
        & np.isfinite(mu)
        & np.isfinite(weight)
    )
    return {
        "obs": obs[finite],
        "teacher_obs": teacher_obs[finite],
        "mu": mu[finite],
        "weight": np.maximum(weight[finite], 0.0),
        "transition": transition[finite],
    }


def load_many(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [load_part(path) for path in paths]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def infer_teacher(
    teacher: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            output.append(teacher(batch).clamp(
                -ACTION_OUTPUT_LIMIT, ACTION_OUTPUT_LIMIT
            ).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def build_model(
    policy_path: Path,
    risk_path: Path,
    correction_limit: float,
    risk_gate_start: float,
    risk_gate_full: float,
    device: torch.device,
) -> HallRecoveryPolicy:
    policy_payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    base = LayoutMagneticStudent(
        float(policy_payload.get("residual_limit", 1.0)),
        trailing_feature_mode=normalize_trailing_feature_mode(
            str(policy_payload.get("trailing_feature_mode", "sensor_age"))
        ),
    )
    base.load_state_dict(policy_payload["model"], strict=True)
    risk_payload = torch.load(risk_path, map_location="cpu", weights_only=False)
    risk = build_hall_risk_estimator(risk_payload)
    return HallRecoveryPolicy(
        base,
        risk,
        correction_limit=correction_limit,
        risk_gate_start=risk_gate_start,
        risk_gate_full=risk_gate_full,
    ).to(device)


def metrics(
    model: HallRecoveryPolicy,
    data: dict[str, np.ndarray],
    target: np.ndarray,
    device: torch.device,
    batch_size: int,
    randomized: bool,
    seed: int,
) -> dict[str, float | int | bool]:
    torch.manual_seed(seed)
    actions: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["obs"]), batch_size):
            obs = torch.from_numpy(data["obs"][start : start + batch_size]).to(device)
            if randomized:
                obs = randomize_hall(
                    obs, model.base_policy.trailing_feature_mode
                )
            action, base, risk, correction = model.recovery_outputs(obs)
            actions.append(action.cpu().numpy())
            bases.append(base.cpu().numpy())
            risks.append(risk.cpu().numpy())
            corrections.append(correction.cpu().numpy())
    action = np.concatenate(actions)
    base = np.concatenate(bases)
    risk = np.concatenate(risks).reshape(-1)
    correction = np.concatenate(corrections)
    error = np.abs(action - target)
    base_error = np.abs(base - target)
    active = risk >= float(model.risk_gate_start)
    clear = risk <= float(model.risk_gate_start)
    low = data["mu"] <= 0.25

    def selected_mean(value: np.ndarray, mask: np.ndarray) -> float:
        return float(value[mask].mean()) if np.any(mask) else float("nan")

    active_mae = selected_mean(error, active)
    active_base_mae = selected_mean(base_error, active)
    return {
        "samples": int(len(action)),
        "mae": float(error.mean()),
        "base_mae": float(base_error.mean()),
        "active_samples": int(active.sum()),
        "active_mae": active_mae,
        "active_base_mae": active_base_mae,
        "active_improvement_fraction": float(
            (active_base_mae - active_mae) / max(active_base_mae, 1.0e-8)
        ),
        "low_mu_mae": selected_mean(error, low),
        "low_mu_base_mae": selected_mean(base_error, low),
        "clear_base_difference_max": float(
            np.abs(action[clear] - base[clear]).max() if np.any(clear) else 0.0
        ),
        "correction_abs_mean": float(np.abs(correction).mean()),
        "correction_abs_max": float(np.abs(correction).max()),
        "risk_mean": float(risk.mean()),
        "finite": bool(
            np.isfinite(action).all()
            and np.isfinite(risk).all()
            and np.isfinite(correction).all()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--correction-limit", type=float, default=0.25)
    parser.add_argument("--risk-gate-start", type=float, default=0.35)
    parser.add_argument("--risk-gate-full", type=float, default=0.75)
    parser.add_argument(
        "--activation-mode",
        choices=("risk_gated", "confidence_gated"),
        default="risk_gated",
        help=(
            "risk_gated preserves the original low-traction-only residual; "
            "confidence_gated learns terrain posture/clearance corrections on "
            "all healthy Hall packets while retaining exact sensor-loss fallback"
        ),
    )
    parser.add_argument("--randomization-probability", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = load_many(args.train)
    testing = load_many(args.test)
    teacher = FrozenTeacher(args.teacher_onnx).to(device).eval()
    train_target = infer_teacher(teacher, training["teacher_obs"], device, args.batch_size)
    test_target = infer_teacher(teacher, testing["teacher_obs"], device, args.batch_size)
    risk_gate_start = (
        0.0 if args.activation_mode == "confidence_gated" else args.risk_gate_start
    )
    risk_gate_full = (
        1.0e-3 if args.activation_mode == "confidence_gated" else args.risk_gate_full
    )
    model = build_model(
        args.base_policy,
        args.risk_checkpoint,
        args.correction_limit,
        risk_gate_start,
        risk_gate_full,
        device,
    )
    risk_model_variant = (
        "baseline_invariant"
        if model.risk_estimator.__class__.__name__.startswith("BaselineInvariant")
        else "layout_encoder"
    )

    dataset = TensorDataset(
        torch.from_numpy(training["obs"]),
        torch.from_numpy(train_target),
        torch.from_numpy(training["mu"][:, None]),
        torch.from_numpy(training["weight"][:, None]),
        torch.from_numpy(training["transition"][:, None]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.recovery_head.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for obs, target, mu, sample_weight, transition in loader:
            obs = obs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mu = mu.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            transition = transition.to(device, non_blocking=True)
            if torch.rand((), device=device) < args.randomization_probability:
                obs = randomize_hall(
                    obs, model.base_policy.trailing_feature_mode
                )
            action, _, risk, correction = model.recovery_outputs(obs)
            active_weight = 0.10 + 3.0 * torch.clamp(
                (risk - args.risk_gate_start)
                / max(args.risk_gate_full - args.risk_gate_start, 1.0e-6),
                0.0,
                1.0,
            )
            low_weight = torch.where(mu <= 0.25, 1.8, 1.0)
            transition_weight = 1.0 + 1.5 * transition
            weight = sample_weight * active_weight * low_weight * transition_weight
            point_loss = nn.functional.smooth_l1_loss(
                action, target, reduction="none", beta=0.10
            ).mean(dim=-1, keepdim=True)
            imitation = (weight * point_loss).sum() / weight.sum().clamp_min(1.0)
            regularization = 0.01 * torch.mean(torch.square(correction))
            loss = imitation + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.recovery_head.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(obs)
            total_count += len(obs)
        row = {"epoch": epoch + 1, "loss": total_loss / max(total_count, 1)}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(row), flush=True)

    nominal = metrics(model, testing, test_target, device, args.batch_size, False, args.seed)
    randomized = metrics(
        model, testing, test_target, device, args.batch_size, True, args.seed + 1
    )

    # Physical Hall loss must remove learned action authority exactly.
    fault_obs = torch.from_numpy(testing["obs"][: min(64, len(testing["obs"]))]).to(device)
    fault_obs = fault_obs.clone()
    apply_foot_dropout_metadata(
        fault_obs,
        torch.ones((len(fault_obs), 2), dtype=torch.bool, device=device),
        model.base_policy.trailing_feature_mode,
    )
    with torch.inference_mode():
        fault_action, fault_base, fault_risk, fault_correction = model.recovery_outputs(
            fault_obs
        )
    fault_fallback_max = float(torch.max(torch.abs(fault_action - fault_base)).cpu())
    fault_correction_max = float(torch.max(torch.abs(fault_correction)).cpu())
    fault_risk_min = float(torch.min(fault_risk).cpu())

    checkpoint = args.output_dir / "hall_recovery_policy.pt"
    torch.save(
        {
            "policy_type": "hall_recovery_policy",
            "model": model.state_dict(),
            "input_dim": INPUT_DIM,
            "correction_limit": args.correction_limit,
            "risk_gate_start": args.risk_gate_start,
            "risk_gate_full": args.risk_gate_full,
            "activation_mode": args.activation_mode,
            "base_residual_limit": float(model.base_policy.residual_limit),
            "trailing_feature_mode": model.base_policy.trailing_feature_mode,
            "risk_model_variant": risk_model_variant,
            "measurement_boundary": (
                "deployment input is Hall Bx/By/Bz history + proprioception only; "
                "Teacher/contact/friction are offline labels"
            ),
        },
        checkpoint,
    )
    model.eval()
    example = torch.from_numpy(testing["obs"][:4]).to(device)
    onnx_path = args.output_dir / "policy.onnx"
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["observation"],
        output_names=["actions"],
        dynamic_axes={"observation": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"observation": example.cpu().numpy()})[0]
    with torch.inference_mode():
        torch_output = model(example).cpu().numpy()
    onnx_max_difference = float(np.max(np.abs(onnx_output - torch_output)))

    gates = {
        "nominal_active_improvement": nominal["active_improvement_fraction"] >= 0.04,
        "randomized_active_improvement": randomized["active_improvement_fraction"] >= 0.02,
        "clear_state_exact_preservation": nominal["clear_base_difference_max"] <= 1.0e-7,
        "bounded_correction": nominal["correction_abs_max"] <= args.correction_limit + 1.0e-6,
        "fault_exact_fallback": fault_fallback_max == 0.0 and fault_correction_max == 0.0,
        "fault_is_conservative": fault_risk_min >= 0.999999,
        "onnx_parity": onnx_max_difference <= 5.0e-5,
        "finite": bool(nominal["finite"] and randomized["finite"]),
    }
    summary = {
        "status": "PASS" if all(gates.values()) else "NEEDS_TRAINING",
        "measurement_boundary": (
            "Hall Bx/By/Bz and proprioception are the only runtime observations; "
            "there is no Hall-to-force/friction inverse"
        ),
        "train_samples": int(len(training["obs"])),
        "test_samples": int(len(testing["obs"])),
        "nominal": nominal,
        "randomized": randomized,
        "fault_fallback_max": fault_fallback_max,
        "fault_correction_max": fault_correction_max,
        "fault_risk_min": fault_risk_min,
        "onnx_max_difference": onnx_max_difference,
        "gates": gates,
        "configuration": {
            "correction_limit": args.correction_limit,
            "risk_gate_start": args.risk_gate_start,
            "risk_gate_full": args.risk_gate_full,
            "effective_risk_gate_start": float(model.risk_gate_start),
            "effective_risk_gate_full": float(model.risk_gate_full),
            "activation_mode": args.activation_mode,
            "epochs": args.epochs,
            "seed": args.seed,
            "risk_model_variant": risk_model_variant,
        },
        "sources": {
            "train": [str(path.resolve()) for path in args.train],
            "test": [str(path.resolve()) for path in args.test],
            "base_policy": str(args.base_policy.resolve()),
            "risk_checkpoint": str(args.risk_checkpoint.resolve()),
            "teacher_onnx": str(args.teacher_onnx.resolve()),
        },
        "artifacts": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "onnx": str(onnx_path.resolve()),
            "onnx_sha256": sha256(onnx_path),
        },
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
