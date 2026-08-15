#!/usr/bin/env python3
"""Train a causal Hall/proprioceptive *future-slip* risk estimator.

This is intentionally different from regressing a friction coefficient or a
normal/tangential force.  ``contact_slip`` and ``fall`` in the input NPZ are
simulator-only labels.  At runtime the exported network receives only the
1864-D deployable observation (proprioception, Bx/By/Bz histories, timing and
packet health) and returns a bounded risk probability for the command
governor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

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
    apply_foot_dropout_metadata,
    randomize_hall,
)
from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    SlipAwareHallRiskEstimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    INPUT_DIM,
    TRAILING_FEATURE_MODES,
    VALID_SLICE,
    normalize_trailing_feature_mode,
)


ONNX_PARITY_TOLERANCE = 3.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prospective_risk_target(
    contact_slip: np.ndarray,
    fall: np.ndarray,
    trajectory_id: np.ndarray,
    step: np.ndarray,
    horizon_steps: int,
    pre_fall_steps: int,
    slip_threshold: float,
    slip_quantile: float,
) -> np.ndarray:
    """Create a causal label from *future* simulator outcomes per environment.

    ``target[t]`` uses a high quantile of contact-point slip in the next short
    horizon, rather than one isolated contact-solver spike.  It is set to one
    when a fall will occur within the longer pre-fall horizon.  The robot
    cannot access either quantity at runtime; they are only supervision while
    fitting the Hall time series.
    """

    result = np.zeros(len(contact_slip), dtype=np.float32)
    for identifier in np.unique(trajectory_id):
        indices = np.flatnonzero(trajectory_id == identifier)
        indices = indices[np.argsort(step[indices], kind="stable")]
        slip = np.nan_to_num(contact_slip[indices], nan=0.0, posinf=10.0)
        falls = fall[indices].astype(bool)
        for local_index, original_index in enumerate(indices):
            slip_end = min(len(indices), local_index + horizon_steps + 1)
            fall_end = min(len(indices), local_index + pre_fall_steps + 1)
            sustained_slip = float(
                np.quantile(slip[local_index:slip_end], slip_quantile)
            )
            slip_risk = min(1.0, sustained_slip / slip_threshold)
            result[original_index] = 1.0 if np.any(falls[local_index:fall_end]) else slip_risk
    return result


def trajectory_ids(
    env_id: np.ndarray,
    step: np.ndarray,
    rollout_id: np.ndarray | None,
) -> np.ndarray:
    """Create reset-safe causal trajectory groups for rollout supervision.

    Evaluation matrices reuse the same vector-environment indices for every
    μ/command cell.  Joining solely on ``env_id`` crosses a hard reset and
    leaks outcomes from another material into a label.  New collectors write
    ``rollout_id`` explicitly.  Legacy files are repaired conservatively by
    splitting each environment whenever its recorded step counter restarts.
    """

    env_id = np.asarray(env_id, dtype=np.int64).reshape(-1)
    step = np.asarray(step, dtype=np.int64).reshape(-1)
    if len(env_id) != len(step):
        raise ValueError("env_id and step lengths must agree")
    if rollout_id is not None:
        rollout = np.asarray(rollout_id, dtype=np.int64).reshape(-1)
        if len(rollout) != len(env_id):
            raise ValueError("rollout_id length must agree with env_id")
        # Use arithmetic rather than a bitwise cast so negative/large ids
        # remain readable in saved diagnostics.
        return rollout * 1_000_000 + env_id

    result = np.empty(len(env_id), dtype=np.int64)
    next_segment = 0
    for identifier in np.unique(env_id):
        indices = np.flatnonzero(env_id == identifier)
        segment = next_segment
        previous_step: int | None = None
        for index in indices:
            current_step = int(step[index])
            if previous_step is not None and current_step <= previous_step:
                segment += 1
            result[index] = segment
            previous_step = current_step
        next_segment = segment + 1
    return result


def load_rollout(
    path: Path,
    horizon_steps: int,
    pre_fall_steps: int,
    slip_threshold: float,
    slip_quantile: float,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"obs", "contact_slip", "fall", "env_id", "step"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"{path}: future-slip training requires {missing}; "
                "collect with eval_friction_matrix.py --collect_npz"
            )
        obs = np.asarray(data["obs"], dtype=np.float32)
        contact_slip = np.asarray(data["contact_slip"], dtype=np.float32).reshape(-1)
        fall = np.asarray(data["fall"], dtype=bool).reshape(-1)
        env_id = np.asarray(data["env_id"], dtype=np.int32).reshape(-1)
        step = np.asarray(data["step"], dtype=np.int32).reshape(-1)
        rollout_id = (
            np.asarray(data["rollout_id"], dtype=np.int32).reshape(-1)
            if "rollout_id" in data
            else None
        )
        valid = np.asarray(
            data["valid"] if "valid" in data else np.ones(len(obs), dtype=bool),
            dtype=bool,
        ).reshape(-1)
    if obs.ndim != 2 or obs.shape[1] != INPUT_DIM:
        raise ValueError(f"{path}: expected [N,{INPUT_DIM}] deployable observation")
    count = len(obs)
    if not all(len(value) == count for value in (contact_slip, fall, env_id, step, valid)):
        raise ValueError(f"{path}: rollout array lengths do not agree")
    if rollout_id is not None and len(rollout_id) != count:
        raise ValueError(f"{path}: rollout_id length does not agree")
    causal_trajectory_id = trajectory_ids(env_id, step, rollout_id)
    target = prospective_risk_target(
        contact_slip,
        fall,
        causal_trajectory_id,
        step,
        horizon_steps,
        pre_fall_steps,
        slip_threshold,
        slip_quantile,
    )
    finite = np.isfinite(obs).all(axis=1) & np.isfinite(contact_slip)
    # The terminal sample is an already-reset simulator state.  Its preceding
    # sample remains in the data and receives the prospective fall label.
    keep = finite & valid
    return {
        "obs": obs[keep],
        "target": target[keep],
        "contact_slip": contact_slip[keep],
        "fall": fall[keep],
        "source": np.full(int(keep.sum()), path.name, dtype="U128"),
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("at least one rollout is required")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def raw_feature_statistics(
    observation: np.ndarray, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            raw, _ = SlipAwareHallRiskEstimator.raw_features(
                torch.from_numpy(observation[start : start + batch_size])
            )
            features.append(raw)
    values = torch.cat(features, dim=0)
    return values.mean(dim=0), values.std(dim=0).clamp_min(0.05)


def infer(
    model: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    batch_size: int,
    randomized: bool,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(observation[start : start + batch_size]).to(device)
            if randomized:
                batch = randomize_hall(batch, model.trailing_feature_mode)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).reshape(-1)


def roc_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positive = target >= 0.5
    negative = target < 0.5
    if not np.any(positive) or not np.any(negative):
        return None
    # The equivalent Mann--Whitney statistic avoids an extra sklearn runtime
    # dependency in the Isaac environment.
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average ranks for ties, which are common at clipped conservative risk.
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    del values
    for group, count in enumerate(counts):
        if count > 1:
            selected = inverse == group
            ranks[selected] = ranks[selected].mean()
    n_pos = float(positive.sum())
    n_neg = float(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def metrics(target: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, object]:
    positive = target >= 0.5
    negative = target < 0.05
    predicted = prediction >= threshold
    return {
        "samples": int(len(target)),
        "positive_fraction": float(positive.mean()),
        "mae": float(np.mean(np.abs(target - prediction))),
        "brier": float(np.mean(np.square(target - prediction))),
        "auc": roc_auc(target, prediction),
        "prospective_risk_recall": (
            float(np.mean(predicted[positive])) if np.any(positive) else None
        ),
        "safe_false_alarm_rate": (
            float(np.mean(predicted[negative])) if np.any(negative) else None
        ),
        "mean_risk_positive": (
            float(prediction[positive].mean()) if np.any(positive) else None
        ),
        "mean_risk_safe": (
            float(prediction[negative].mean()) if np.any(negative) else None
        ),
        "finite": bool(np.isfinite(prediction).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--pre-fall-steps", type=int, default=25)
    parser.add_argument("--slip-threshold", type=float, default=0.25)
    parser.add_argument(
        "--slip-quantile",
        type=float,
        default=0.75,
        help="Future-window contact-slip quantile used as the sustained-slip label.",
    )
    parser.add_argument("--operating-threshold", type=float, default=0.65)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hall-randomization-prob", type=float, default=0.35)
    parser.add_argument(
        "--trailing-feature-mode",
        choices=TRAILING_FEATURE_MODES,
        required=True,
        help="Meaning of observation channels 1862:1864 for packet confidence.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.trailing_feature_mode = normalize_trailing_feature_mode(
        args.trailing_feature_mode
    )
    if args.horizon_steps <= 0 or args.pre_fall_steps <= 0:
        raise ValueError("risk horizons must be positive")
    if args.slip_threshold <= 0.0:
        raise ValueError("--slip-threshold must be positive")
    if not 0.0 <= args.slip_quantile <= 1.0:
        raise ValueError("--slip-quantile must be in [0,1]")
    if not 0.0 < args.operating_threshold < 1.0:
        raise ValueError("--operating-threshold must be in (0,1)")
    if args.positive_weight <= 0.0:
        raise ValueError("--positive-weight must be positive")
    if not 0.0 <= args.hall_randomization_prob <= 1.0:
        raise ValueError("--hall-randomization-prob must be in [0,1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train = concatenate(
        [
            load_rollout(
                path,
                args.horizon_steps,
                args.pre_fall_steps,
                args.slip_threshold,
                args.slip_quantile,
            )
            for path in args.train
        ]
    )
    test = concatenate(
        [
            load_rollout(
                path,
                args.horizon_steps,
                args.pre_fall_steps,
                args.slip_threshold,
                args.slip_quantile,
            )
            for path in args.test
        ]
    )
    feature_mean, feature_scale = raw_feature_statistics(train["obs"], args.batch_size)
    model = SlipAwareHallRiskEstimator(
        feature_mean,
        feature_scale,
        trailing_feature_mode=args.trailing_feature_mode,
    ).to(device)

    sample_weight = 1.0 + (args.positive_weight - 1.0) * train["target"]
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["obs"]),
            torch.from_numpy(train["target"]),
            torch.from_numpy(sample_weight.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.10
    )
    history: list[dict[str, float]] = []
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for observation, target, weight in loader:
            observation = observation.to(device)
            target = target.to(device)
            weight = weight.to(device)
            if random.random() < args.hall_randomization_prob:
                observation = randomize_hall(
                    observation, model.trailing_feature_mode
                )
            logit, health = model.learned_logit(observation)
            confidence = model.physical_confidence(
                health, model.trailing_feature_mode
            ).reshape(-1)
            effective_target = confidence * target + (1.0 - confidence)
            per_sample = nn.functional.binary_cross_entropy_with_logits(
                logit.reshape(-1).float(), effective_target.float(), reduction="none"
            )
            loss = (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(observation)
            count += len(observation)
        scheduler.step()
        epoch_loss = loss_sum / max(count, 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
            print(
                f"epoch={epoch + 1:03d}/{args.epochs} loss={epoch_loss:.6f} "
                f"ETA={eta / 60.0:.1f}m",
                flush=True,
            )

    nominal_prediction = infer(
        model, test["obs"], device, args.batch_size, False, args.seed + 1
    )
    randomized_prediction = infer(
        model, test["obs"], device, args.batch_size, True, args.seed + 2
    )
    nominal = metrics(test["target"], nominal_prediction, args.operating_threshold)
    randomized = metrics(test["target"], randomized_prediction, args.operating_threshold)

    fault_obs = torch.from_numpy(test["obs"][: min(128, len(test["obs"]))]).to(device)
    fault_obs = fault_obs.clone()
    apply_foot_dropout_metadata(
        fault_obs,
        torch.ones((len(fault_obs), 2), dtype=torch.bool, device=device),
        model.trailing_feature_mode,
    )
    with torch.inference_mode():
        fault_risk_min = float(model(fault_obs).min().item())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "hall_slip_risk_estimator.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_variant": "slip_aware_invariant",
            "input_dim": INPUT_DIM,
            "trailing_feature_mode": args.trailing_feature_mode,
            "measurement_boundary": (
                "runtime input is Hall Bx/By/Bz history + proprioception only; "
                "contact slip/falls are offline simulator labels, not inputs"
            ),
            "risk_target": "prospective contact-point slip/fall",
        },
        checkpoint,
    )
    onnx_path = args.output_dir / "hall_slip_risk.onnx"
    model.eval()
    example = torch.zeros((1, INPUT_DIM), device=device)
    example[:, VALID_SLICE] = 1.0
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["observation"],
        output_names=["low_traction_probability"],
        dynamic_axes={"observation": {0: "batch"}, "low_traction_probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    parity_obs = test["obs"][: min(256, len(test["obs"]))]
    onnx = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_prediction = onnx.run(None, {"observation": parity_obs})[0].reshape(-1)
    parity = float(np.max(np.abs(onnx_prediction - nominal_prediction[: len(onnx_prediction)])))

    gates = {
        "finite": bool(nominal["finite"] and randomized["finite"]),
        "nominal_auc_ge_0p85": bool(nominal["auc"] is not None and nominal["auc"] >= 0.85),
        "nominal_risk_recall_ge_0p85": bool(
            nominal["prospective_risk_recall"] is not None
            and nominal["prospective_risk_recall"] >= 0.85
        ),
        "randomized_risk_recall_ge_0p75": bool(
            randomized["prospective_risk_recall"] is not None
            and randomized["prospective_risk_recall"] >= 0.75
        ),
        "sensor_loss_maps_to_risk_one": fault_risk_min >= 0.999999,
        "onnx_parity": parity <= ONNX_PARITY_TOLERANCE,
    }
    summary = {
        "status": "trained_sim_hall_slip_risk_estimator",
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "model_variant": "slip_aware_invariant",
        "trailing_feature_mode": args.trailing_feature_mode,
        "measurement_boundary": (
            "Hall Bx/By/Bz and proprioception are the only runtime inputs; "
            "there is no Hall-to-force or Hall-to-friction inverse"
        ),
        "offline_label": {
            "definition": "future sustained contact-point slip or fall",
            "horizon_steps": args.horizon_steps,
            "pre_fall_steps": args.pre_fall_steps,
            "slip_threshold_m_s": args.slip_threshold,
            "slip_quantile": args.slip_quantile,
            "operating_threshold": args.operating_threshold,
        },
        "train_samples": int(len(train["obs"])),
        "test_samples": int(len(test["obs"])),
        "nominal": nominal,
        "randomized": randomized,
        "fault_risk_min": fault_risk_min,
        "onnx_parity_max_abs": parity,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "gates": gates,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256(onnx_path),
        "sources": {
            "train": [str(path.resolve()) for path in args.train],
            "test": [str(path.resolve()) for path in args.test],
        },
        "history": history,
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(gates.values()) or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
