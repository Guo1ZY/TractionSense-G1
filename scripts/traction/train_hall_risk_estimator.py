#!/usr/bin/env python3
"""Train an independent Hall-only traction-risk head.

The estimator consumes only the deployable Hall + proprioceptive observation.
Simulator friction is an offline label.  Keeping this network independent from
the action Student lets transition recovery improve without changing gait
actions that already passed their acceptance gates.
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
ONNX_PARITY_TOLERANCE = 3.0e-5
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_layout_magnetic_student import (  # noqa: E402
    apply_foot_dropout_metadata,
    load_dataset,
    randomize_hall,
)
from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    BaselineInvariantHallTractionRiskEstimator,
    HallTractionRiskEstimator,
    build_hall_risk_estimator,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    INPUT_DIM,
    TRAILING_FEATURE_MODES,
    VALID_SLICE,
    normalize_trailing_feature_mode,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_parts(
    paths: list[Path], crosssim_weight: float = 1.0
) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for path in paths:
        try:
            part = load_dataset(path)
        except ValueError as error:
            # A MuJoCo Hall-only rollout has no privileged 641-D Teacher
            # observation because the risk head needs only obs and offline μ.
            # Keep this narrow fallback explicit so malformed Isaac DAgger
            # datasets still fail validation.
            with np.load(path, allow_pickle=False) as data:
                if "teacher_obs" in data or "obs" not in data or "mu" not in data:
                    raise error
                obs = np.asarray(data["obs"], dtype=np.float32)
                mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
                count = len(obs)
                sample_weight = np.asarray(
                    data["sample_weight"] if "sample_weight" in data else np.ones(count),
                    dtype=np.float32,
                ).reshape(-1)
                if "time_since_switch_s" in data:
                    switch_transition = (
                        np.asarray(data["time_since_switch_s"], dtype=np.float32)
                        .reshape(-1)
                        <= 1.0
                    ).astype(np.float32)
                else:
                    switch_transition = np.full(
                        count,
                        1.0 if "switch" in str(path).casefold() else 0.0,
                        dtype=np.float32,
                    )
            if obs.shape != (count, INPUT_DIM):
                raise ValueError(f"{path}: expected Nx{INPUT_DIM} Hall observation")
            finite = (
                np.isfinite(obs).all(axis=1)
                & np.isfinite(mu)
                & np.isfinite(sample_weight)
                & np.isfinite(switch_transition)
            )
            part = {
                "obs": obs[finite],
                "mu": mu[finite],
                "sample_weight": np.maximum(sample_weight[finite], 0.0),
                "switch_transition": switch_transition[finite],
            }
        if "unitree_mujoco" in str(path.resolve()).casefold():
            part["sample_weight"] = part["sample_weight"] * crosssim_weight
        count = len(part["obs"])
        # Prefer the causal per-sample marker produced by collection.  The
        # previous filename heuristic silently treated active-probe datasets
        # as steady state when their filename did not contain ``switch`` and
        # overweighted every sample when it did.  Keep the heuristic only for
        # legacy datasets that predate the marker.
        if "switch_transition" in part:
            part["is_switch"] = np.asarray(
                part["switch_transition"], dtype=np.float32
            ).reshape(count)
        else:
            part["is_switch"] = np.full(
                count, "switch" in str(path).casefold(), dtype=np.float32
            )
        parts.append(part)
    keys = ("obs", "mu", "sample_weight", "is_switch")
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def target_from_mu(
    mu: np.ndarray | torch.Tensor,
    center: float = 0.25,
    temperature: float = 0.05,
):
    if temperature <= 0.0:
        raise ValueError("risk target temperature must be positive")
    if isinstance(mu, torch.Tensor):
        return torch.sigmoid((center - mu) / temperature)
    return 1.0 / (1.0 + np.exp((mu - center) / temperature))


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


def metrics(
    prediction: np.ndarray,
    mu: np.ndarray,
    is_switch: np.ndarray,
    risk_center: float = 0.25,
    risk_temperature: float = 0.05,
) -> dict[str, object]:
    low = mu <= 0.25
    high = mu >= 0.75
    switch_high = high & (is_switch > 0.5)
    target = target_from_mu(mu, risk_center, risk_temperature)
    low_detect = float(np.mean(prediction[low] >= 0.65)) if np.any(low) else None
    high_clear = float(np.mean(prediction[high] <= 0.55)) if np.any(high) else None
    balanced = (
        0.5 * (low_detect + high_clear)
        if low_detect is not None and high_clear is not None
        else None
    )
    return {
        "samples": int(len(mu)),
        "mae_soft_target": float(np.mean(np.abs(prediction - target))),
        "low_mean": float(prediction[low].mean()) if np.any(low) else None,
        "high_mean": float(prediction[high].mean()) if np.any(high) else None,
        "switch_high_mean": (
            float(prediction[switch_high].mean()) if np.any(switch_high) else None
        ),
        "low_detection_rate_p_ge_0p65": low_detect,
        "high_clear_rate_p_le_0p55": high_clear,
        "balanced_accuracy": balanced,
        "high_false_critical_rate": (
            float(np.mean(prediction[high] >= 0.85)) if np.any(high) else None
        ),
        "finite": bool(np.isfinite(prediction).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--initialize-policy", type=Path)
    parser.add_argument(
        "--initialize-risk",
        type=Path,
        help="Existing hall_risk_estimator.pt for evaluation/export without retraining.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--switch-weight", type=float, default=3.0)
    parser.add_argument(
        "--low-traction-weight",
        type=float,
        default=1.0,
        help="Additional loss weight for samples with simulator-only mu <= 0.25.",
    )
    parser.add_argument(
        "--crosssim-weight",
        type=float,
        default=4.0,
        help="Offline loss multiplier for MuJoCo Hall-forward-model trajectories.",
    )
    parser.add_argument("--hall-randomization-prob", type=float, default=0.35)
    parser.add_argument(
        "--trailing-feature-mode",
        choices=TRAILING_FEATURE_MODES,
        required=True,
        help="Meaning of channels 1862:1864 when calculating link confidence.",
    )
    parser.add_argument("--risk-center", type=float, default=0.25)
    parser.add_argument("--risk-temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--model-variant",
        choices=("layout_encoder", "baseline_invariant"),
        default="layout_encoder",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.trailing_feature_mode = normalize_trailing_feature_mode(
        args.trailing_feature_mode
    )
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.crosssim_weight <= 0.0:
        raise ValueError("--crosssim-weight must be positive")
    if args.risk_temperature <= 0.0:
        raise ValueError("--risk-temperature must be positive")
    if args.low_traction_weight <= 0.0:
        raise ValueError("--low-traction-weight must be positive")
    train = load_parts(args.train, args.crosssim_weight)
    test = load_parts(args.test, 1.0)
    if args.model_variant == "baseline_invariant":
        if args.initialize_policy is not None:
            raise ValueError(
                "--initialize-policy is unavailable for baseline_invariant; "
                "train its compact feature network directly"
            )
        feature_batches: list[torch.Tensor] = []
        for start in range(0, len(train["obs"]), args.batch_size):
            raw, _ = BaselineInvariantHallTractionRiskEstimator.raw_features(
                torch.from_numpy(train["obs"][start : start + args.batch_size])
            )
            feature_batches.append(raw)
        feature_values = torch.cat(feature_batches, dim=0)
        feature_mean = feature_values.mean(dim=0)
        feature_scale = feature_values.std(dim=0).clamp_min(0.05)
        model = BaselineInvariantHallTractionRiskEstimator(
            feature_mean,
            feature_scale,
            trailing_feature_mode=args.trailing_feature_mode,
        ).to(device)
    else:
        model = HallTractionRiskEstimator(
            trailing_feature_mode=args.trailing_feature_mode
        ).to(device)
    if args.initialize_risk is not None:
        payload = torch.load(args.initialize_risk, map_location="cpu", weights_only=False)
        initialized = build_hall_risk_estimator(payload)
        if (
            args.model_variant == "baseline_invariant"
        ) != isinstance(initialized, BaselineInvariantHallTractionRiskEstimator):
            raise ValueError("--initialize-risk model_variant does not match")
        if initialized.trailing_feature_mode != args.trailing_feature_mode:
            raise ValueError(
                "--initialize-risk trailing_feature_mode does not match: "
                f"{initialized.trailing_feature_mode!r} != "
                f"{args.trailing_feature_mode!r}"
            )
        model.load_state_dict(initialized.state_dict(), strict=True)
    elif args.initialize_policy is not None:
        payload = torch.load(args.initialize_policy, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        encoder_state = {
            key.removeprefix("foot_encoder."): value
            for key, value in state.items()
            if key.startswith("foot_encoder.")
        }
        model.foot_encoder.load_state_dict(encoder_state, strict=True)

    target = target_from_mu(
        train["mu"], args.risk_center, args.risk_temperature
    ).astype(np.float32)
    weight = train["sample_weight"].astype(np.float32) * np.where(
        train["is_switch"] > 0.5, args.switch_weight, 1.0
    ).astype(np.float32)
    weight *= np.where(
        train["mu"] <= 0.25, args.low_traction_weight, 1.0
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train["obs"]),
        torch.from_numpy(target),
        torch.from_numpy(weight),
    )
    loader = DataLoader(
        dataset,
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
    start_time = time.time()
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for observation, target_batch, weight_batch in loader:
            observation = observation.to(device)
            target_batch = target_batch.to(device)
            weight_batch = weight_batch.to(device)
            if random.random() < args.hall_randomization_prob:
                observation = randomize_hall(
                    observation, model.trailing_feature_mode
                )
            logit, health = model.learned_logit(observation)
            target_batch = target_batch[:, None]
            # Physical loss of confidence is always treated as maximum risk.
            confidence = model.physical_confidence(
                health, model.trailing_feature_mode
            )
            effective_target = confidence * target_batch + (1.0 - confidence)
            per_sample = nn.functional.binary_cross_entropy_with_logits(
                logit.float(), effective_target.float(), reduction="none"
            ).reshape(-1)
            loss = torch.sum(per_sample * weight_batch) / weight_batch.sum().clamp_min(1.0)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "hall_risk_estimator.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_variant": args.model_variant,
            "input_dim": INPUT_DIM,
            "trailing_feature_mode": args.trailing_feature_mode,
            "measurement_boundary": "Hall + proprioception only; no Hall-to-force inverse",
        },
        checkpoint,
    )
    nominal_prediction = infer(
        model, test["obs"], device, args.batch_size, False, args.seed + 1
    )
    randomized_prediction = infer(
        model, test["obs"], device, args.batch_size, True, args.seed + 2
    )
    nominal = metrics(
        nominal_prediction,
        test["mu"],
        test["is_switch"],
        args.risk_center,
        args.risk_temperature,
    )
    randomized = metrics(
        randomized_prediction,
        test["mu"],
        test["is_switch"],
        args.risk_center,
        args.risk_temperature,
    )

    fault_obs = torch.from_numpy(test["obs"][: min(128, len(test["obs"]))]).to(device)
    fault_obs = fault_obs.clone()
    apply_foot_dropout_metadata(
        fault_obs,
        torch.ones((len(fault_obs), 2), dtype=torch.bool, device=device),
        model.trailing_feature_mode,
    )
    with torch.inference_mode():
        fault_risk = model(fault_obs).cpu().numpy()
    fault_min = float(fault_risk.min())

    onnx_path = args.output_dir / "hall_risk.onnx"
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
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_prediction = session.run(None, {"observation": parity_obs})[0]
    torch_prediction = infer(model, parity_obs, device, args.batch_size, False, args.seed + 3)
    parity = float(np.max(np.abs(onnx_prediction.reshape(-1) - torch_prediction)))

    gates = {
        "finite": bool(nominal["finite"] and randomized["finite"]),
        "nominal_balanced_accuracy_ge_0p85": bool(
            nominal["balanced_accuracy"] is not None
            and nominal["balanced_accuracy"] >= 0.85
        ),
        "randomized_balanced_accuracy_ge_0p80": bool(
            randomized["balanced_accuracy"] is not None
            and randomized["balanced_accuracy"] >= 0.80
        ),
        "switch_high_mean_le_0p45": bool(
            nominal["switch_high_mean"] is not None
            and nominal["switch_high_mean"] <= 0.45
        ),
        "sensor_loss_maps_to_risk_one": fault_min >= 0.999999,
        # CUDA/PyTorch and CPU/ONNX Runtime use slightly different reduction
        # orders in the temporal Hall encoder.  3e-5 is still far below Hall
        # quantization/noise and preserves a strict deploy-export equivalence
        # gate without making it backend-specific.
        "onnx_parity": parity <= ONNX_PARITY_TOLERANCE,
    }
    summary = {
        "status": "trained_sim_risk_estimator",
        "model_variant": args.model_variant,
        "trailing_feature_mode": args.trailing_feature_mode,
        "risk_target": {
            "center_mu": args.risk_center,
            "temperature": args.risk_temperature,
            "low_traction_weight": args.low_traction_weight,
        },
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "student_inputs": "proprio + dual-foot 15x3 Hall history + timing/health",
        "forbidden_inputs": [
            "normal_force", "tangential_force", "friction_truth", "contact_truth"
        ],
        "train_samples": int(len(train["obs"])),
        "test_samples": int(len(test["obs"])),
        "nominal": nominal,
        "randomized": randomized,
        "fault_risk_min": fault_min,
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
            "initialize_policy": (
                str(args.initialize_policy.resolve()) if args.initialize_policy else None
            ),
            "initialize_risk": (
                str(args.initialize_risk.resolve()) if args.initialize_risk else None
            ),
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
