#!/usr/bin/env python3
"""Fit a bounded Hall recovery residual around the audited original G1 actor.

The only deployment input is the standard 1864-D Hall/proprioceptive
observation.  Privileged Teacher action, simulator friction, contact and slip
are used as offline supervision only.  The exported ONNX is intended for the
``--hall_recovery_onnx`` path in ``eval_friction_matrix.py``: it is selected
only after the independent Hall-risk governor enters a valid LOW/probe state.
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
from unitree_rl_lab.traction.anchored_hall_recovery import (  # noqa: E402
    AnchoredHallRecoveryPolicy,
    load_baseline_actor,
)
from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    build_hall_risk_estimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    INPUT_DIM,
    VALID_SLICE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_part(path: Path) -> dict[str, np.ndarray]:
    """Load a causal DAgger sequence without exposing labels at runtime."""

    with np.load(path, allow_pickle=False) as data:
        required = {"obs", "teacher_obs", "mu"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        obs = np.asarray(data["obs"], dtype=np.float32)
        teacher_obs = np.asarray(data["teacher_obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
        count = len(obs)
        sample_weight = np.asarray(
            data["sample_weight"] if "sample_weight" in data else np.ones(count),
            dtype=np.float32,
        ).reshape(-1)
        if "time_since_switch_s" in data:
            transition = (
                np.asarray(data["time_since_switch_s"], dtype=np.float32).reshape(-1)
                <= 1.25
            ).astype(np.float32)
        else:
            transition = np.full(
                count, 1.0 if "switch" in path.name.casefold() else 0.0, dtype=np.float32
            )
    if obs.shape != (count, INPUT_DIM):
        raise ValueError(f"{path}: expected Nx{INPUT_DIM} obs, got {obs.shape}")
    if teacher_obs.shape != (count, 641):
        raise ValueError(f"{path}: expected Nx641 Teacher obs, got {teacher_obs.shape}")
    # Train the action residual on healthy packets only.  Sensor-loss behavior
    # is a deterministic exact baseline fallback tested separately below.
    healthy = (
        np.isfinite(obs).all(axis=1)
        & np.isfinite(teacher_obs).all(axis=1)
        & np.isfinite(mu)
        & np.isfinite(sample_weight)
        & (obs[:, VALID_SLICE] > 0.5).all(axis=1)
    )
    return {
        "obs": obs[healthy],
        "teacher_obs": teacher_obs[healthy],
        "mu": mu[healthy],
        "sample_weight": np.maximum(sample_weight[healthy], 0.0),
        "transition": transition[healthy],
    }


def load_many(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [load_part(path) for path in paths]
    if not parts:
        raise ValueError("at least one dataset is required")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def infer_teacher(
    teacher: FrozenTeacher,
    observation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result: list[np.ndarray] = []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            result.append(
                teacher(torch.from_numpy(observation[start : start + batch_size]).to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(result).astype(np.float32)


def infer_baseline(
    baseline: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate the fixed original actor for data-quality filtering only."""

    result: list[np.ndarray] = []
    baseline.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            result.append(
                baseline(
                    torch.nan_to_num(
                        torch.from_numpy(observation[start : start + batch_size, :480])
                        .to(device)
                    )
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(result).astype(np.float32)


def filter_stable_action_samples(
    data: dict[str, np.ndarray],
    teacher_action: np.ndarray,
    baseline_action: np.ndarray,
    max_abs_action: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    """Discard post-fall/reset outliers before they enter imitation loss.

    Isaac automatically resets terminated environments.  Its first few stale
    history frames can make either the raw Teacher or legacy actor extrapolate
    to enormous values even though the action manager clips the command.  Such
    rows are neither a recoverable walking state nor a valid supervision
    target.  The filter is based only on policy outputs, never on force/slip
    labels, and is applied identically to train and held-out data.
    """

    if teacher_action.shape != baseline_action.shape or len(teacher_action) != len(data["obs"]):
        raise ValueError("action outputs and dataset length must agree")
    stable = (
        np.isfinite(teacher_action).all(axis=1)
        & np.isfinite(baseline_action).all(axis=1)
        & (np.max(np.abs(teacher_action), axis=1) <= max_abs_action)
        & (np.max(np.abs(baseline_action), axis=1) <= max_abs_action)
    )
    if not np.any(stable):
        raise RuntimeError("stable-action filter removed every sample")
    filtered = {key: value[stable] for key, value in data.items()}
    report = {
        "input_samples": int(len(stable)),
        "kept_samples": int(stable.sum()),
        "dropped_post_reset_or_outlier_samples": int((~stable).sum()),
    }
    return filtered, teacher_action[stable], report


def bounded_target(
    teacher_action: torch.Tensor,
    baseline_action: torch.Tensor,
    correction_limit: float,
) -> torch.Tensor:
    """Project an oracle action into the explicitly safe residual envelope."""

    return baseline_action + (teacher_action - baseline_action).clamp(
        -correction_limit, correction_limit
    )


def offline_supervision_gate(
    deploy_gate: torch.Tensor,
    mu: torch.Tensor,
    low_mu_gate_floor: float,
) -> torch.Tensor:
    """Keep stable low-traction Teacher trajectories in the imitation loss.

    ``mu`` is an offline Isaac label and is deliberately used *only* here,
    never by :class:`AnchoredHallRecoveryPolicy` or exported ONNX inference.
    A competent Teacher can suppress slip quickly enough that the frozen risk
    head no longer emits a large score on its stable low-traction trajectory.
    Without this training-only floor, precisely those desirable recovery
    examples would have almost no gradient for the residual head.
    """

    if not 0.0 <= low_mu_gate_floor <= 1.0:
        raise ValueError("low_mu_gate_floor must be in [0,1]")
    if low_mu_gate_floor == 0.0:
        return deploy_gate
    floor = deploy_gate.new_full(deploy_gate.shape, low_mu_gate_floor)
    return torch.where(mu.reshape(-1) <= 0.25, torch.maximum(deploy_gate, floor), deploy_gate)


def _infer(
    model: AnchoredHallRecoveryPolicy,
    data: dict[str, np.ndarray],
    teacher_target: np.ndarray,
    device: torch.device,
    batch_size: int,
    randomized: bool,
    seed: int,
) -> dict[str, np.ndarray]:
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
                    obs, model.risk_estimator.trailing_feature_mode
                )
            action, base, risk, correction = model.recovery_outputs(obs)
            actions.append(action.cpu().numpy())
            bases.append(base.cpu().numpy())
            risks.append(risk.cpu().numpy())
            corrections.append(correction.cpu().numpy())
    return {
        "action": np.concatenate(actions),
        "base": np.concatenate(bases),
        "risk": np.concatenate(risks).reshape(-1),
        "correction": np.concatenate(corrections),
        "teacher": teacher_target,
    }


def metrics(result: dict[str, np.ndarray], data: dict[str, np.ndarray], gate_start: float) -> dict[str, float | int | bool]:
    action = result["action"]
    base = result["base"]
    target = result["teacher"]
    correction = result["correction"]
    risk = result["risk"]
    active = risk >= gate_start
    low = data["mu"] <= 0.25

    def mean(value: np.ndarray, mask: np.ndarray) -> float:
        return float(value[mask].mean()) if np.any(mask) else float("nan")

    error = np.abs(action - target)
    base_error = np.abs(base - target)
    active_error = mean(error, active)
    active_base_error = mean(base_error, active)
    clear = risk < gate_start
    return {
        "samples": int(len(action)),
        "mae_to_teacher": float(error.mean()),
        "base_mae_to_teacher": float(base_error.mean()),
        "active_samples": int(active.sum()),
        "active_mae_to_teacher": active_error,
        "active_base_mae_to_teacher": active_base_error,
        "active_improvement_fraction": float(
            (active_base_error - active_error) / max(active_base_error, 1.0e-8)
        )
        if np.isfinite(active_error) and np.isfinite(active_base_error)
        else float("nan"),
        "low_mu_mae_to_teacher": mean(error, low),
        "low_mu_base_mae_to_teacher": mean(base_error, low),
        "clear_state_exact_preservation_max": float(
            np.abs(action[clear] - base[clear]).max() if np.any(clear) else 0.0
        ),
        "correction_abs_mean": float(np.abs(correction).mean()),
        "correction_abs_max": float(np.abs(correction).max()),
        "risk_mean": float(risk.mean()),
        "finite": bool(
            np.isfinite(action).all()
            and np.isfinite(base).all()
            and np.isfinite(correction).all()
            and np.isfinite(risk).all()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("model/rl/model_49999.pt"))
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=6.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--correction-limit", type=float, default=0.12)
    parser.add_argument("--risk-gate-start", type=float, default=0.50)
    parser.add_argument("--risk-gate-full", type=float, default=0.75)
    parser.add_argument("--low-traction-weight", type=float, default=2.5)
    parser.add_argument("--transition-weight", type=float, default=2.0)
    parser.add_argument(
        "--low-mu-training-gate-floor",
        type=float,
        default=0.0,
        help=(
            "Offline-only minimum residual supervision gate for mu <= 0.25. "
            "It never appears in the deployable observation or exported ONNX."
        ),
    )
    parser.add_argument(
        "--max-stable-action",
        type=float,
        default=3.0,
        help=(
            "Reject causal rows whose raw original or Teacher action exceeds this "
            "magnitude; they are post-fall/reset outliers, not recovery targets."
        ),
    )
    parser.add_argument("--randomization-probability", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.correction_limit <= 0.0:
        raise ValueError("correction limit must be positive")
    if args.low_traction_weight < 1.0 or args.transition_weight < 1.0:
        raise ValueError("extra training weights must be >= 1")
    if not 0.0 <= args.low_mu_training_gate_floor <= 1.0:
        raise ValueError("--low-mu-training-gate-floor must be in [0,1]")
    if args.max_stable_action <= 0.0:
        raise ValueError("--max-stable-action must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    training = load_many(args.train)
    testing = load_many(args.test)
    teacher = FrozenTeacher(args.teacher_onnx).to(device).eval()

    risk_payload = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(risk_payload, dict):
        raise ValueError("risk checkpoint payload must be a dictionary")
    risk = build_hall_risk_estimator(risk_payload)
    baseline = load_baseline_actor(args.baseline_checkpoint).to(device)
    train_teacher = infer_teacher(teacher, training["teacher_obs"], device, args.batch_size)
    test_teacher = infer_teacher(teacher, testing["teacher_obs"], device, args.batch_size)
    train_baseline = infer_baseline(baseline, training["obs"], device, args.batch_size)
    test_baseline = infer_baseline(baseline, testing["obs"], device, args.batch_size)
    training, train_teacher, train_filter = filter_stable_action_samples(
        training, train_teacher, train_baseline, args.max_stable_action
    )
    testing, test_teacher, test_filter = filter_stable_action_samples(
        testing, test_teacher, test_baseline, args.max_stable_action
    )
    model = AnchoredHallRecoveryPolicy(
        baseline,
        risk,
        correction_limit=args.correction_limit,
        risk_gate_start=args.risk_gate_start,
        risk_gate_full=args.risk_gate_full,
    ).to(device)
    model.freeze_upstream()

    dataset = TensorDataset(
        torch.from_numpy(training["obs"]),
        torch.from_numpy(train_teacher),
        torch.from_numpy(training["mu"]),
        torch.from_numpy(training["sample_weight"]),
        torch.from_numpy(training["transition"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.recovery_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.10
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for obs, teacher_action, mu, sample_weight, transition in loader:
            obs = obs.to(device)
            teacher_action = teacher_action.to(device)
            mu = mu.to(device)
            sample_weight = sample_weight.to(device)
            transition = transition.to(device)
            if random.random() < args.randomization_probability:
                obs = randomize_hall(
                    obs, model.risk_estimator.trailing_feature_mode
                )
            action, base, risk_value, correction = model.recovery_outputs(obs)
            target = bounded_target(teacher_action, base, args.correction_limit)
            gate = torch.clamp(
                (risk_value.reshape(-1) - args.risk_gate_start)
                / max(args.risk_gate_full - args.risk_gate_start, 1.0e-6),
                0.0,
                1.0,
            )
            supervision_gate = offline_supervision_gate(
                gate, mu, args.low_mu_training_gate_floor
            )
            active_weight = 0.05 + 2.5 * supervision_gate
            low_weight = torch.where(mu <= 0.25, args.low_traction_weight, 1.0)
            transition_multiplier = 1.0 + (args.transition_weight - 1.0) * transition
            weight = sample_weight * active_weight * low_weight * transition_multiplier
            point = nn.functional.smooth_l1_loss(action, target, reduction="none", beta=0.08).mean(dim=1)
            imitation = (weight * point).sum() / weight.sum().clamp_min(1.0)
            regularization = 0.02 * torch.square(correction).mean()
            loss = imitation + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.recovery_head.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(obs)
            count += len(obs)
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": loss_sum / max(count, 1)}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(row), flush=True)

    nominal_result = _infer(model, testing, test_teacher, device, args.batch_size, False, args.seed)
    randomized_result = _infer(model, testing, test_teacher, device, args.batch_size, True, args.seed + 1)
    nominal = metrics(nominal_result, testing, args.risk_gate_start)
    randomized = metrics(randomized_result, testing, args.risk_gate_start)

    fault_obs = torch.from_numpy(testing["obs"][: min(len(testing["obs"]), 64)]).to(device).clone()
    apply_foot_dropout_metadata(
        fault_obs,
        torch.ones((len(fault_obs), 2), dtype=torch.bool, device=device),
        model.risk_estimator.trailing_feature_mode,
    )
    with torch.inference_mode():
        fault_action, fault_base, fault_risk, fault_correction = model.recovery_outputs(fault_obs)
    fault_action_error = float(torch.max(torch.abs(fault_action - fault_base)).cpu())
    fault_correction_error = float(torch.max(torch.abs(fault_correction)).cpu())
    fault_risk_min = float(torch.min(fault_risk).cpu())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "anchored_hall_recovery.pt"
    torch.save(
        {
            "policy_type": "anchored_hall_recovery_policy",
            "model": model.state_dict(),
            "input_dim": INPUT_DIM,
            "correction_limit": args.correction_limit,
            "risk_gate_start": args.risk_gate_start,
            "risk_gate_full": args.risk_gate_full,
            "baseline_actor_state": baseline.state_dict(),
            "risk_payload": risk_payload,
            "measurement_boundary": "Hall Bx/By/Bz history + proprioception only; Teacher/friction/contact are offline labels",
        },
        checkpoint,
    )
    model.eval()
    example = torch.from_numpy(testing["obs"][: min(len(testing["obs"]), 4)]).to(device)
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
    onnx_action = session.run(None, {session.get_inputs()[0].name: example.cpu().numpy()})[0]
    with torch.inference_mode():
        torch_action = model(example).cpu().numpy()
    onnx_error = float(np.max(np.abs(onnx_action - torch_action)))

    gates = {
        "nominal_active_improvement": nominal["active_improvement_fraction"] >= 0.03,
        "randomized_active_improvement": randomized["active_improvement_fraction"] >= 0.015,
        "clear_state_exact_preservation": nominal["clear_state_exact_preservation_max"] <= 1.0e-7,
        "bounded_correction": nominal["correction_abs_max"] <= args.correction_limit + 1.0e-6,
        "fault_exact_baseline": fault_action_error == 0.0 and fault_correction_error == 0.0,
        "fault_conservative_risk": fault_risk_min >= 0.999999,
        "onnx_parity": onnx_error <= 5.0e-5,
        "finite": bool(nominal["finite"] and randomized["finite"]),
    }
    summary = {
        "status": "PASS" if all(gates.values()) else "NEEDS_TRAINING",
        "measurement_boundary": "Hall Bx/By/Bz and proprioception are the only runtime observations; there is no Hall-to-force/friction inverse",
        "train_samples": int(len(training["obs"])),
        "test_samples": int(len(testing["obs"])),
        "nominal": nominal,
        "randomized": randomized,
        "fault_action_difference_max": fault_action_error,
        "fault_correction_max": fault_correction_error,
        "fault_risk_min": fault_risk_min,
        "onnx_parity_max_abs": onnx_error,
        "gates": gates,
        "configuration": {
            "correction_limit": args.correction_limit,
            "risk_gate_start": args.risk_gate_start,
            "risk_gate_full": args.risk_gate_full,
            "epochs": args.epochs,
            "seed": args.seed,
            "max_stable_action": args.max_stable_action,
            "low_mu_training_gate_floor_offline_only": (
                args.low_mu_training_gate_floor
            ),
        },
        "stable_action_filter": {
            "train": train_filter,
            "test": test_filter,
        },
        "sources": {
            "train": [str(path.resolve()) for path in args.train],
            "test": [str(path.resolve()) for path in args.test],
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
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
    if args.strict and not all(gates.values()):
        raise RuntimeError("anchored Hall recovery did not pass offline gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
