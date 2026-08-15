#!/usr/bin/env python3
"""Behavior-clone the exact RSL-RL Hall actor before safety PPO.

The exported actor has the deployed 1864-D interface only:
``proprioception + dual-foot Hall Bx/By/Bz history + link health``.  A frozen
641-D Oracle Teacher is used *offline* to label recorded simulator states;
contact force, slip truth and ground friction never enter the Hall actor.

This pretraining step prevents the first PPO updates from spending most of
their samples rediscovering the nominal gait.  It also injects explicit
all-Hall-loss examples whose target is the audited 480-D base actor, so a
missing Hall stream has a conservative learned fallback before PPO sees
fault-randomized rollouts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    ACTION_DIM,
    BASE_DIM,
    INPUT_DIM,
    MAGNETIC_SLICE,
    PERIOD_SLICE,
    VALID_SLICE,
)


HALL_END = PERIOD_SLICE.stop
TEACHER_DIM = 641
ACTION_LIMIT = 3.0


class MlpActor(nn.Module):
    """The exact ELU MLP shape used by the RSL-RL G1 actor."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


def _mlp_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in state.items() if key.startswith("mlp.")}
    required = {
        f"mlp.{index}.{field}"
        for index in (0, 2, 4, 6)
        for field in ("weight", "bias")
    }
    missing = sorted(required - set(result))
    if missing:
        raise KeyError(f"actor checkpoint is missing {missing}")
    return result


def _load_actor_from_rsl(state: dict[str, torch.Tensor], input_dim: int) -> MlpActor:
    actor = MlpActor(input_dim)
    actor.load_state_dict(_mlp_state(state), strict=True)
    return actor


def _frozen_onnx_mlp(path: Path, input_dim: int) -> MlpActor:
    graph = onnx.load(path).graph
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    actor = MlpActor(input_dim)
    state = actor.state_dict()
    for index in (0, 2, 4, 6):
        for field in ("weight", "bias"):
            key = f"mlp.{index}.{field}"
            if key not in tensors:
                raise KeyError(f"{path}: missing ONNX initializer {key}")
            value = torch.from_numpy(np.array(tensors[key], copy=True))
            if state[key].shape != value.shape:
                raise ValueError(
                    f"{path}: {key} shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
            state[key] = value
    actor.load_state_dict(state, strict=True)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor.eval()


def _batched(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("obs", "teacher_obs")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        observation = np.asarray(data["obs"], dtype=np.float32)
        teacher_observation = np.asarray(data["teacher_obs"], dtype=np.float32)
        count = len(observation)
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"{path}: expected obs shape [N,{INPUT_DIM}]")
        if teacher_observation.ndim != 2 or teacher_observation.shape[1] != TEACHER_DIM:
            raise ValueError(f"{path}: expected teacher_obs shape [N,{TEACHER_DIM}]")
        if len(teacher_observation) != count:
            raise ValueError(f"{path}: obs / teacher_obs row count mismatch")
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(count) if "mu" in data else np.full(count, np.nan, np.float32)
        transition = (
            np.asarray(data["time_since_switch_s"], dtype=np.float32).reshape(count)
            if "time_since_switch_s" in data
            else np.full(count, np.inf, np.float32)
        )
        # ``eval_friction_matrix.py`` assigns a higher collection weight to
        # the first causal frames after a material switch.  Those are the
        # only frames in which the Hall actor can learn a rapid response;
        # retaining the weight here prevents a long steady-state rollout from
        # drowning them out.  It is a training priority, not an actor input.
        collection_weight = (
            np.asarray(data["sample_weight"], dtype=np.float32).reshape(count)
            if "sample_weight" in data
            else np.ones(count, dtype=np.float32)
        )
    finite = (
        np.isfinite(observation).all(axis=1)
        & np.isfinite(teacher_observation).all(axis=1)
        & np.isfinite(collection_weight)
        & (collection_weight > 0.0)
    )
    if not np.any(finite):
        raise RuntimeError(f"{path}: no finite rows")
    return {
        "obs": observation[finite],
        "teacher_obs": teacher_observation[finite],
        "mu": mu[finite],
        "time_since_switch_s": transition[finite],
        "sample_weight": collection_weight[finite],
    }


def _concat(parts: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    values = list(parts)
    if not values:
        raise ValueError("at least one dataset is required")
    return {name: np.concatenate([part[name] for part in values], axis=0) for name in values[0]}


def select_deploy_targets(
    teacher_action: np.ndarray,
    baseline_action: np.ndarray,
    valid_lr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use Teacher labels only when both Hall feet are valid.

    The validity bit itself is a real deployable input.  On a missing foot or
    total loss, the target is the audited proprioceptive actor, not a force or
    friction estimate.  This makes the fail-safe rule explicit and testable.
    """

    if teacher_action.shape != baseline_action.shape or teacher_action.ndim != 2:
        raise ValueError("teacher/baseline action shapes must match")
    valid = np.asarray(valid_lr, dtype=np.float32)
    if valid.shape != (len(teacher_action), 2):
        raise ValueError("valid_lr must be [N,2]")
    use_teacher = np.all(valid >= 0.5, axis=1)
    target = np.where(use_teacher[:, None], teacher_action, baseline_action)
    return np.clip(target, -ACTION_LIMIT, ACTION_LIMIT).astype(np.float32), use_teacher


def _randomize_hall_or_drop(
    observation: torch.Tensor,
    *,
    probability: float,
    fallback_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb Hall data and synthesize all-Hall-loss fallback examples."""

    result = observation.clone()
    batch = len(result)
    randomize = torch.rand(batch, device=result.device) < probability
    if torch.any(randomize):
        hall = result[randomize, MAGNETIC_SLICE]
        gain = 0.75 + 0.50 * torch.rand((len(hall), 1), device=hall.device, dtype=hall.dtype)
        noise = (0.015 + 0.020 * hall.abs()) * torch.randn_like(hall)
        result[randomize, MAGNETIC_SLICE] = torch.clamp(hall * gain + noise, -6.0, 6.0)
        period = result[randomize, PERIOD_SLICE]
        result[randomize, PERIOD_SLICE] = torch.clamp(
            period * (0.90 + 0.20 * torch.rand((len(period), 1), device=period.device, dtype=period.dtype)),
            0.001,
            0.25,
        )
    forced_fallback = torch.rand(batch, device=result.device) < fallback_probability
    if torch.any(forced_fallback):
        result[forced_fallback, BASE_DIM:HALL_END] = 0.0
        result[forced_fallback, VALID_SLICE] = 0.0
    return result, forced_fallback


def _metrics(prediction: np.ndarray, target: np.ndarray, fallback_mask: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - target)
    result = {
        "mae": float(error.mean()),
        "p95_abs": float(np.quantile(error, 0.95)),
        "max_abs": float(error.max()),
    }
    if np.any(fallback_mask):
        result["fallback_mae"] = float(error[fallback_mask].mean())
        result["fallback_max_abs"] = float(error[fallback_mask].max())
    else:
        result["fallback_mae"] = float("nan")
        result["fallback_max_abs"] = float("nan")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_datasets() -> tuple[list[Path], list[Path]]:
    root = ROOT / "artifacts" / "hall_speed_demo"
    return (
        [
            root / "teacher_switch_dagger_seed229.npz",
            root / "teacher_switch_dagger_seed230.npz",
        ],
        [root / "teacher_switch_dagger_seed231.npz"],
    )


def parse_args() -> argparse.Namespace:
    default_train, default_valid = _default_datasets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", default=None)
    parser.add_argument("--validation", type=Path, action="append", default=None)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=(
            ROOT
            / "logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_motion_switch"
            / "2026-07-29_15-57-53_two_surface_switch_two_surface_prod_20260729/exported/policy.onnx"
        ),
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "model/rl/model_49999.pt")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hall-randomization-probability", type=float, default=0.35)
    parser.add_argument("--fallback-augmentation-probability", type=float, default=0.20)
    parser.add_argument(
        "--low-mu-transition-weight",
        type=float,
        default=2.0,
        help="Additional weight for the first low-mu switch frames (offline only).",
    )
    parser.add_argument(
        "--max-sample-weight",
        type=float,
        default=8.0,
        help="Safety cap for recorded offline priorities.",
    )
    parser.add_argument("--max-label-action", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=322)
    args = parser.parse_args()
    args.train = args.train or default_train
    args.validation = args.validation or default_valid
    return args


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if args.low_mu_transition_weight <= 0.0 or args.max_sample_weight <= 0.0:
        raise ValueError("offline sample weights must be positive")
    for name, value in (
        ("hall-randomization-probability", args.hall_randomization_probability),
        ("fallback-augmentation-probability", args.fallback_augmentation_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0,1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train = _concat(_load_npz(path) for path in args.train)
    validation = _concat(_load_npz(path) for path in args.validation)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    baseline_payload = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in source or "actor_state_dict" not in baseline_payload:
        raise KeyError("source and baseline checkpoint must contain actor_state_dict")
    actor = _load_actor_from_rsl(source["actor_state_dict"], INPUT_DIM).to(device)
    baseline = _load_actor_from_rsl(baseline_payload["actor_state_dict"], BASE_DIM).to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    teacher = _frozen_onnx_mlp(args.teacher, TEACHER_DIM).to(device)

    def prepare(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        teacher_action = _batched(teacher, data["teacher_obs"], device, args.batch_size)
        baseline_action = _batched(baseline, data["obs"][:, :BASE_DIM], device, args.batch_size)
        stable = (
            np.isfinite(teacher_action).all(axis=1)
            & np.isfinite(baseline_action).all(axis=1)
            & (np.max(np.abs(teacher_action), axis=1) <= args.max_label_action)
            & (np.max(np.abs(baseline_action), axis=1) <= args.max_label_action)
        )
        if not np.any(stable):
            raise RuntimeError("stable-label filter removed every sample")
        valid = data["obs"][:, VALID_SLICE][stable]
        target, use_teacher = select_deploy_targets(
            teacher_action[stable], baseline_action[stable], valid
        )
        early_low_mu_transition = (
            (data["time_since_switch_s"][stable] <= 0.8)
            & np.isfinite(data["mu"][stable])
            & (data["mu"][stable] <= 0.25)
        )
        transition_weight = np.where(
            early_low_mu_transition,
            args.low_mu_transition_weight,
            1.0,
        ).astype(np.float32)
        weight = np.clip(
            data["sample_weight"][stable] * transition_weight,
            1.0e-3,
            args.max_sample_weight,
        ).astype(np.float32)
        return {
            "obs": data["obs"][stable],
            "target": target,
            "baseline": baseline_action[stable],
            "weight": weight,
            "use_teacher": use_teacher.astype(np.bool_),
            "input_rows": np.asarray([len(data["obs"])], dtype=np.int64),
            "mean_weight": np.asarray([weight.mean()], dtype=np.float32),
        }

    train = prepare(train)
    validation = prepare(validation)
    print(
        f"device={device} train={len(train['obs'])} validation={len(validation['obs'])} "
        f"teacher_labels={int(train['use_teacher'].sum())}/{len(train['obs'])}",
        flush=True,
    )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["obs"]),
            torch.from_numpy(train["target"]),
            torch.from_numpy(train["baseline"]),
            torch.from_numpy(train["weight"]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.10
    )
    joint_weight = torch.ones(ACTION_DIM, device=device)
    joint_weight[[0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]] = 2.0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        actor.train()
        totals = {"loss": 0.0, "teacher": 0.0, "fallback": 0.0}
        batches = 0
        for observation, target, base_action, sample_weight in loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            base_action = base_action.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            randomized, forced_fallback = _randomize_hall_or_drop(
                observation,
                probability=args.hall_randomization_probability,
                fallback_probability=args.fallback_augmentation_probability,
            )
            expected = torch.where(forced_fallback[:, None], base_action, target)
            prediction = actor(randomized)
            per_joint = torch.nn.functional.smooth_l1_loss(prediction, expected, reduction="none")
            weighted = (per_joint * joint_weight).mean(dim=1) * sample_weight
            loss = weighted.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.25)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["teacher"] += float(
                torch.nn.functional.l1_loss(prediction[~forced_fallback], target[~forced_fallback])
                if torch.any(~forced_fallback)
                else torch.zeros((), device=device)
            )
            totals["fallback"] += float(
                torch.nn.functional.l1_loss(prediction[forced_fallback], base_action[forced_fallback])
                if torch.any(forced_fallback)
                else torch.zeros((), device=device)
            )
            batches += 1
        scheduler.step()
        row = {name: value / max(batches, 1) for name, value in totals.items()}
        row["epoch"] = float(epoch)
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        if row["loss"] < best_loss and np.isfinite(row["loss"]):
            best_loss = row["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={row['loss']:.5f} teacher_l1={row['teacher']:.5f} "
                f"fallback_l1={row['fallback']:.5f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("pretraining produced no finite state")
    actor.load_state_dict(best_state, strict=True)

    actor.eval()
    valid_prediction = _batched(actor, validation["obs"], device, args.batch_size)
    # Explicit all-Hall-loss validation.  The proprioceptive prefix remains
    # real; Hall and timing content are removed and both valid bits are zero.
    fallback_observation = validation["obs"].copy()
    fallback_observation[:, BASE_DIM:HALL_END] = 0.0
    fallback_observation[:, VALID_SLICE] = 0.0
    fallback_prediction = _batched(actor, fallback_observation, device, args.batch_size)
    fallback_target = validation["baseline"]
    validation_metrics = _metrics(
        valid_prediction,
        validation["target"],
        ~validation["use_teacher"],
    )
    fallback_error = np.abs(fallback_prediction - fallback_target)
    validation_metrics["forced_all_hall_loss_mae"] = float(fallback_error.mean())
    validation_metrics["forced_all_hall_loss_max_abs"] = float(fallback_error.max())

    output = copy.deepcopy(source)
    output_actor = dict(output["actor_state_dict"])
    for key, value in actor.state_dict().items():
        output_actor[key] = value.detach().cpu()
    output["actor_state_dict"] = output_actor
    output["iter"] = 0
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_checkpoint)
    summary = {
        "format": "g1-rsl-hall-actor-pretrain-v1",
        "status": "offline_pretrained_not_a_deployment_candidate",
        "measurement_boundary": "actor uses Hall Bx/By/Bz history plus proprioception only",
        "teacher_usage": "offline action labels only; 641-D privileged input is never exported to actor",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "teacher": str(args.teacher.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_sha256": _sha256(args.output_checkpoint),
        "datasets": {
            "train": [str(path.resolve()) for path in args.train],
            "validation": [str(path.resolve()) for path in args.validation],
            "train_rows": int(len(train["obs"])),
            "validation_rows": int(len(validation["obs"])),
            "train_teacher_labels": int(train["use_teacher"].sum()),
            "validation_teacher_labels": int(validation["use_teacher"].sum()),
            "train_mean_offline_weight": float(train["mean_weight"][0]),
            "validation_mean_offline_weight": float(validation["mean_weight"][0]),
        },
        "validation": validation_metrics,
        "history": history,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["validation"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
