#!/usr/bin/env python3
"""Distill local privileged Hall-force and remote torque policies into one Student.

The local datasets provide paired ideal-contact Teacher labels and randomized
three-axis foot-force observations.  The remote dataset provides the later
641-D friction-conditioned Teacher together with its 915-D torque-only Student
domain.  Simulator-only fields are used as losses/labels and are never copied
into the final 1011-D deployment observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.final_student import (  # noqa: E402
    ACTION_DIM,
    BASE_PROPRIO_DIM,
    FINAL_STUDENT_INPUT_DIM,
    FINAL_STUDENT_SCHEMA,
    FOOT_AGE_SLICE,
    FOOT_FORCE_DIM,
    FOOT_FORCE_SLICE,
    FOOT_PERIOD_SLICE,
    FOOT_VALID_SLICE,
    HISTORY_FRAMES,
    JOINT_EFFORT_DIM,
    JOINT_EFFORT_SLICE,
    FinalHallForceStudent,
)
from unitree_rl_lab.traction.networks import LegacyLocomotionActor  # noqa: E402


LEG_ACTION_INDICES = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)
WAIST_ACTION_INDICES = (2, 5, 8)
LOCAL_FRAME_DIM = 106


class RemoteTeacherActor(nn.Module):
    def __init__(self, path: Path) -> None:
        super().__init__()
        graph = onnx.load(path).graph
        tensors = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
        layers: list[nn.Module] = []
        for index, name in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
            weight = torch.from_numpy(np.array(tensors[f"{name}.weight"], copy=True))
            bias = torch.from_numpy(np.array(tensors[f"{name}.bias"], copy=True))
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            layers.append(layer)
            if index != 3:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


class RemoteTorquePolicy(nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"]
        self.register_buffer("input_mean", torch.as_tensor(payload["mean"], dtype=torch.float32))
        self.register_buffer("input_scale", torch.as_tensor(payload["scale"], dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(915, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.action_head = nn.Linear(128, ACTION_DIM)
        self.mu_head = nn.Linear(128, 1)
        compatible = {
            key: value
            for key, value in state.items()
            if key not in {"input_mean", "input_scale"}
        }
        self.load_state_dict(compatible, strict=False)

    def latent(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder((observation - self.input_mean) / self.input_scale)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.latent(observation)
        return self.action_head(latent), 1.30 * torch.sigmoid(self.mu_head(latent))


class RemoteTorqueSelector(nn.Module):
    """Exact remote selector028 semantics, rebuilt from its checkpoint."""

    def __init__(self, torque_policy: RemoteTorquePolicy, official: LegacyLocomotionActor) -> None:
        super().__init__()
        self.torque_policy = torque_policy
        self.official = official

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_action, estimated_mu = self.torque_policy(observation)
        fast_action = self.official(observation[:, :BASE_PROPRIO_DIM])
        traction_gate = torch.sigmoid(20.0 * (estimated_mu - 0.28))
        command_gate = torch.sigmoid(18.0 * (observation[:, 42:43] - 0.60))
        blend = traction_gate * command_gate
        return safe_action + blend * (fast_action - safe_action), estimated_mu


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_actor(checkpoint_path: Path) -> tuple[LegacyLocomotionActor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = LegacyLocomotionActor(BASE_PROPRIO_DIM)
    actor.load_state_dict(
        {
            key: value
            for key, value in checkpoint["actor_state_dict"].items()
            if key.startswith("mlp.")
        },
        strict=True,
    )
    return actor.eval(), checkpoint


def _legacy_proprio(history_flat: np.ndarray) -> np.ndarray:
    history = history_flat.reshape(-1, HISTORY_FRAMES, LOCAL_FRAME_DIM)
    recent = history[:, -5:]
    return np.concatenate(
        [
            recent[..., term].reshape(len(history), -1)
            for term in (
                slice(0, 3),
                slice(3, 6),
                slice(93, 96),
                slice(6, 35),
                slice(35, 64),
                slice(64, 93),
            )
        ],
        axis=-1,
    ).astype(np.float32)


def _torque_history(
    torque: np.ndarray,
    environment_id: np.ndarray,
    timestamp_s: np.ndarray,
) -> np.ndarray:
    ids = environment_id.reshape(-1).astype(np.int64)
    stamps = timestamp_s.reshape(-1)
    unique_ids = np.unique(ids)
    result = np.empty((len(torque), HISTORY_FRAMES, JOINT_EFFORT_DIM), dtype=np.float32)
    for environment in unique_ids:
        indices = np.flatnonzero(ids == environment)
        indices = indices[np.argsort(stamps[indices], kind="stable")]
        values = np.clip(torque[indices], -100.0, 100.0).astype(np.float32) * 0.02
        for position, row in enumerate(indices):
            start = max(0, position - HISTORY_FRAMES + 1)
            window = values[start : position + 1]
            if len(window) < HISTORY_FRAMES:
                padding = np.repeat(window[0:1], HISTORY_FRAMES - len(window), axis=0)
                window = np.concatenate((padding, window), axis=0)
            result[row] = window
    return result


def load_local(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = (
            "student_history", "teacher_action", "joint_torque",
            "environment_id", "timestamp_s", "ground_friction_mu",
            "slip_label", "sensor_valid", "sensor_age_s",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        raw = {name: np.asarray(data[name]) for name in required}
    history = raw["student_history"].astype(np.float32)
    if history.shape[1] != HISTORY_FRAMES * LOCAL_FRAME_DIM:
        raise ValueError(f"{path}: unexpected history shape {history.shape}")
    history_frames = history.reshape(-1, HISTORY_FRAMES, LOCAL_FRAME_DIM)
    baseline = _legacy_proprio(history)
    torque = _torque_history(
        raw["joint_torque"], raw["environment_id"], raw["timestamp_s"]
    ).reshape(-1, HISTORY_FRAMES * JOINT_EFFORT_DIM)
    force = history_frames[..., 96:102].reshape(-1, HISTORY_FRAMES * FOOT_FORCE_DIM)
    valid = history_frames[:, -1, 102:104]
    age = history_frames[:, -1, 104:106]
    period = np.full((len(history), 2), 0.02, dtype=np.float32)
    observation = np.concatenate((baseline, torque, force, valid, age, period), axis=1)
    finite = (
        np.isfinite(observation).all(axis=1)
        & np.isfinite(raw["teacher_action"]).all(axis=1)
    )
    return {
        "observation": observation[finite].astype(np.float32),
        "teacher_action": raw["teacher_action"][finite].astype(np.float32),
        "mu": np.min(raw["ground_friction_mu"][finite], axis=1).astype(np.float32),
        "slip": raw["slip_label"][finite].astype(np.float32),
        "slip_mask": np.ones((int(finite.sum()), 1), dtype=np.float32),
        "environment_id": raw["environment_id"][finite].reshape(-1).astype(np.int64),
        "domain": np.zeros((int(finite.sum()),), dtype=np.int64),
    }


def load_remote(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        observation = np.asarray(data["obs"], dtype=np.float32)
        teacher_observation = np.asarray(data["teacher_obs"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
    if observation.shape[1] != 915 or teacher_observation.shape[1] != 641:
        raise ValueError(f"{path}: remote shapes changed")
    force = np.zeros((len(observation), HISTORY_FRAMES * FOOT_FORCE_DIM), dtype=np.float32)
    valid = np.zeros((len(observation), 2), dtype=np.float32)
    age = np.full((len(observation), 2), 1.0, dtype=np.float32)
    period = np.full((len(observation), 2), 0.02, dtype=np.float32)
    final_observation = np.concatenate((observation, force, valid, age, period), axis=1)
    finite = (
        np.isfinite(final_observation).all(axis=1)
        & np.isfinite(teacher_observation).all(axis=1)
        & np.isfinite(mu)
    )
    count = int(finite.sum())
    return {
        "observation": final_observation[finite].astype(np.float32),
        "teacher_observation": teacher_observation[finite].astype(np.float32),
        "mu": mu[finite],
        "slip": np.zeros((count, 2), dtype=np.float32),
        "slip_mask": np.zeros((count, 1), dtype=np.float32),
        "environment_id": np.full((count,), -1, dtype=np.int64),
        "domain": np.ones((count,), dtype=np.int64),
    }


def infer(
    model: nn.Module,
    value: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    tuple_index: int | None = None,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(value), batch_size):
            result = model(torch.from_numpy(value[start : start + batch_size]).to(device))
            if tuple_index is not None:
                result = result[tuple_index]
            output.append(result.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def bounded_target(
    raw_target: np.ndarray,
    baseline_action: np.ndarray,
    limit: float,
) -> np.ndarray:
    return baseline_action + np.clip(raw_target - baseline_action, -limit, limit)


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = actual - reference
    absolute = np.abs(error)
    return {
        "samples": int(len(reference)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p95_l2": float(np.quantile(np.linalg.norm(error, axis=1), 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def evaluate(
    model: FinalHallForceStudent,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    outputs: list[list[np.ndarray]] = [[] for _ in range(5)]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset["observation"]), batch_size):
            batch = torch.from_numpy(dataset["observation"][start : start + batch_size]).to(device)
            for bucket, value in zip(outputs, model(batch), strict=True):
                bucket.append(value.cpu().numpy())
    action, mu, slip, confidence, residual = [np.concatenate(items) for items in outputs]
    mu_error = mu[:, 0] - dataset["mu"]
    result: dict[str, object] = {
        "action_bounded_target": metrics(dataset["target_action"], action),
        "action_raw_teacher": metrics(dataset["raw_teacher_action"], action),
        "mu_mae": float(np.mean(np.abs(mu_error))),
        "mu_rmse": float(np.sqrt(np.mean(np.square(mu_error)))),
        "confidence_mean": float(np.mean(confidence)),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "finite": bool(
            np.isfinite(action).all()
            and np.isfinite(mu).all()
            and np.isfinite(slip).all()
            and np.isfinite(confidence).all()
        ),
    }
    local = dataset["slip_mask"][:, 0] > 0.5
    if np.any(local):
        truth = dataset["slip"][local] > 0.5
        predicted = slip[local] >= 0.5
        tp = np.sum(truth & predicted)
        fp = np.sum(~truth & predicted)
        fn = np.sum(truth & ~predicted)
        result["slip_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    for domain, label in ((0, "hall_force"), (1, "remote_tau"), (2, "forced_fallback")):
        selected = dataset["domain"] == domain
        if np.any(selected):
            result[f"action_{label}"] = metrics(
                dataset["target_action"][selected], action[selected]
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_artifacts = ROOT / "artifacts" / "final_hall_policy_20260806"
    canonical = ROOT / "artifacts" / "canonical_traction_20260731"
    remote = default_artifacts / "remote"
    parser.add_argument(
        "--local-dataset", type=Path, action="append",
        default=[
            canonical / "warmstart_teacher_transition_dataset_16000.npz",
            canonical / "warmstart_dagger_transition_dataset_16000.npz",
        ],
    )
    parser.add_argument("--remote-train", type=Path, default=remote / "tau_dataset/train.npz")
    parser.add_argument("--remote-test", type=Path, default=remote / "tau_dataset/test_unseen.npz")
    parser.add_argument(
        "--remote-teacher", type=Path,
        default=remote / "traction_teacher_7989/exported/policy.onnx",
    )
    parser.add_argument(
        "--remote-tau-checkpoint", type=Path,
        default=remote / "traction_proprio_tau_selector028/proprio_policy.pt",
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "model/rl/model_49999.pt")
    parser.add_argument("--output-dir", type=Path, default=default_artifacts / "student")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--residual-limit", type=float, default=1.0)
    parser.add_argument("--fallback-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    official, baseline_checkpoint = _baseline_actor(args.baseline_checkpoint)
    remote_teacher = RemoteTeacherActor(args.remote_teacher).to(device).eval()
    remote_tau = RemoteTorquePolicy(args.remote_tau_checkpoint)
    remote_selector = RemoteTorqueSelector(remote_tau, official).to(device).eval()
    official = official.to(device).eval()
    for frozen in (remote_teacher, remote_selector, official):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    local_parts = [load_local(path) for path in args.local_dataset]
    local = concatenate(local_parts)
    local_train_mask = local["environment_id"] < 56
    local_test_mask = ~local_train_mask
    remote_train = load_remote(args.remote_train)
    remote_test = load_remote(args.remote_test)

    print("precomputing privileged Teacher and remote fallback actions", flush=True)
    remote_train_teacher = infer(
        remote_teacher, remote_train.pop("teacher_observation"), device, args.batch_size * 2
    )
    remote_test_teacher = infer(
        remote_teacher, remote_test.pop("teacher_observation"), device, args.batch_size * 2
    )
    remote_train_fallback = infer(
        remote_selector,
        remote_train["observation"][:, :915],
        device,
        args.batch_size * 2,
        tuple_index=0,
    )
    remote_test_fallback = infer(
        remote_selector,
        remote_test["observation"][:, :915],
        device,
        args.batch_size * 2,
        tuple_index=0,
    )

    def finish(
        data: dict[str, np.ndarray],
        raw_teacher: np.ndarray,
        fallback: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        baseline = infer(official, data["observation"][:, :480], device, args.batch_size * 2)
        if fallback is not None:
            raw_target = 0.70 * raw_teacher + 0.30 * fallback
        else:
            raw_target = raw_teacher
        data["raw_teacher_action"] = raw_teacher.astype(np.float32)
        data["target_action"] = bounded_target(
            raw_target, baseline, args.residual_limit
        ).astype(np.float32)
        return data

    remote_train = finish(remote_train, remote_train_teacher, remote_train_fallback)
    remote_test = finish(remote_test, remote_test_teacher, remote_test_fallback)
    local["raw_teacher_action"] = local["teacher_action"]
    local.pop("teacher_action")
    local = finish(local, local["raw_teacher_action"])

    def select(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
        return {key: value[mask] for key, value in data.items()}

    local_train = select(local, local_train_mask)
    local_test = select(local, local_test_mask)

    # Explicit sensor-loss copies teach a smooth handoff to the remote torque
    # policy instead of allowing an untrained invalid-Hall corner.
    fallback_count = int(round(len(local_train["observation"]) * args.fallback_fraction))
    fallback_indices = np.random.default_rng(args.seed).choice(
        len(local_train["observation"]), fallback_count, replace=False
    )
    local_fallback = {key: value[fallback_indices].copy() for key, value in local_train.items()}
    local_fallback["observation"][:, FOOT_FORCE_SLICE] = 0.0
    local_fallback["observation"][:, FOOT_VALID_SLICE] = 0.0
    local_fallback["observation"][:, FOOT_AGE_SLICE] = 1.0
    fallback_action = infer(
        remote_selector,
        local_fallback["observation"][:, :915],
        device,
        args.batch_size * 2,
        tuple_index=0,
    )
    fallback_baseline = infer(
        official,
        local_fallback["observation"][:, :480],
        device,
        args.batch_size * 2,
    )
    local_fallback["target_action"] = bounded_target(
        fallback_action, fallback_baseline, args.residual_limit
    )
    local_fallback["raw_teacher_action"] = fallback_action
    local_fallback["slip_mask"][:] = 0.0
    local_fallback["domain"][:] = 2

    training = concatenate([local_train, remote_train, local_fallback])
    validation = concatenate([local_test, remote_test])
    signal = np.concatenate(
        (
            training["observation"][:, JOINT_EFFORT_SLICE].reshape(-1, JOINT_EFFORT_DIM),
            training["observation"][:, FOOT_FORCE_SLICE].reshape(-1, FOOT_FORCE_DIM),
        ),
        axis=1,
    )
    signal_mean = signal.mean(axis=0, dtype=np.float64).astype(np.float32)
    signal_scale = signal.std(axis=0, dtype=np.float64).astype(np.float32)
    signal_scale[signal_scale < 1.0e-3] = 1.0

    model = FinalHallForceStudent(
        signal_mean=torch.from_numpy(signal_mean),
        signal_scale=torch.from_numpy(signal_scale),
        residual_limit=args.residual_limit,
    )
    model.load_baseline_checkpoint(baseline_checkpoint)
    model.to(device)
    # Preserve the audited gait; only the temporal fusion/residual/auxiliary
    # path is distilled.
    for parameter in model.baseline_actor.parameters():
        parameter.requires_grad_(False)

    confidence_target = (
        training["observation"][:, FOOT_VALID_SLICE].mean(axis=1, keepdims=True)
        * np.exp(-training["observation"][:, FOOT_AGE_SLICE].max(axis=1, keepdims=True) / 0.10)
    ).astype(np.float32)
    sample_weight = np.ones(len(training["observation"]), dtype=np.float32)
    sample_weight[training["domain"] == 0] = 1.5
    sample_weight[training["domain"] == 2] = 1.25
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training["observation"]),
            torch.from_numpy(training["target_action"]),
            torch.from_numpy(training["mu"][:, None]),
            torch.from_numpy(training["slip"]),
            torch.from_numpy(training["slip_mask"]),
            torch.from_numpy(confidence_target),
            torch.from_numpy(sample_weight[:, None]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    action_weights = torch.ones(ACTION_DIM, device=device)
    action_weights[list(LEG_ACTION_INDICES)] = 2.0
    action_weights[list(WAIST_ACTION_INDICES)] = 1.5
    action_weight_sum = action_weights.sum()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"total": 0.0, "action": 0.0, "mu": 0.0, "slip": 0.0, "confidence": 0.0}
        batches = 0
        for observation, target, mu, slip, slip_mask, confidence, weight in loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mu = mu.to(device, non_blocking=True)
            slip = slip.to(device, non_blocking=True)
            slip_mask = slip_mask.to(device, non_blocking=True)
            confidence = confidence.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                action, predicted_mu, predicted_slip, predicted_confidence, residual = model(observation)
                component = nn.functional.smooth_l1_loss(
                    action, target, beta=0.05, reduction="none"
                ) * action_weights
                action_per_sample = component.sum(dim=1, keepdim=True) / action_weight_sum
                action_loss = (action_per_sample * weight).sum() / weight.sum()
                mu_loss = nn.functional.smooth_l1_loss(predicted_mu, mu, beta=0.08)
                slip_component = nn.functional.binary_cross_entropy(
                    predicted_slip, slip, reduction="none"
                ).mean(dim=1, keepdim=True)
                slip_loss = (slip_component * slip_mask).sum() / slip_mask.sum().clamp_min(1.0)
                confidence_loss = nn.functional.mse_loss(predicted_confidence, confidence)
                residual_loss = residual.square().mean()
                loss = (
                    action_loss + 0.05 * mu_loss + 0.08 * slip_loss
                    + 0.05 * confidence_loss + 0.001 * residual_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["total"] += float(loss.detach())
            totals["action"] += float(action_loss.detach())
            totals["mu"] += float(mu_loss.detach())
            totals["slip"] += float(slip_loss.detach())
            totals["confidence"] += float(confidence_loss.detach())
            batches += 1
        scheduler.step()
        record = {key: value / max(batches, 1) for key, value in totals.items()}
        record["epoch"] = float(epoch)
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.monotonic() - started
            eta = elapsed * (args.epochs - epoch) / epoch
            print(
                f"epoch={epoch:03d}/{args.epochs} loss={record['total']:.6f} "
                f"action={record['action']:.6f} ETA={eta / 60:.1f}m",
                flush=True,
            )

    train_metrics = evaluate(model, training, device, args.batch_size * 2)
    validation_metrics = evaluate(model, validation, device, args.batch_size * 2)
    checkpoint_path = args.output_dir / "final_hall_force_student.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "schema": FINAL_STUDENT_SCHEMA.to_dict(),
            "signal_mean": signal_mean,
            "signal_scale": signal_scale,
            "residual_limit": args.residual_limit,
            "seed": args.seed,
            "epochs": args.epochs,
            "training_history": history,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "sources": {
                "local_datasets": [str(path.resolve()) for path in args.local_dataset],
                "remote_train": str(args.remote_train.resolve()),
                "remote_test": str(args.remote_test.resolve()),
                "remote_teacher": str(args.remote_teacher.resolve()),
                "remote_tau_checkpoint": str(args.remote_tau_checkpoint.resolve()),
                "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            },
        },
        checkpoint_path,
    )
    summary = {
        "status": "trained_candidate",
        "method": "privileged local Hall-force Teacher + remote 641-D Teacher + torque fallback distillation",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_samples": int(len(training["observation"])),
        "validation_samples": int(len(validation["observation"])),
        "train": train_metrics,
        "validation": validation_metrics,
        "gates": {
            "finite": bool(validation_metrics["finite"]),
            "bounded_action_mae": validation_metrics["action_bounded_target"]["mae"] <= 0.08,
            "bounded_action_p95": validation_metrics["action_bounded_target"]["p95_abs"] <= 0.25,
            "residual_bound": validation_metrics["residual_max_abs"] <= args.residual_limit + 1.0e-6,
        },
    }
    summary["overall"] = "PASS" if all(summary["gates"].values()) else "FAIL"
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

