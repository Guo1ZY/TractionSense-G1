#!/usr/bin/env python3
"""Distill a privileged traction Teacher into the layout-aware Hall Student.

Teacher observations and true friction are labels only.  The Student receives
exactly the 1864-D deployment observation and never receives contact force,
normal/tangential force, or ground friction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    ACTION_DIM,
    ACTION_OUTPUT_LIMIT,
    AGE_SLICE,
    AXES,
    BASE_DIM,
    FEET,
    HEALTH_DIM,
    HISTORY,
    INPUT_DIM,
    MAGNETIC_DIM,
    MAGNETIC_SLICE,
    PERIOD_DIM,
    PERIOD_SLICE,
    SCHEMA,
    SENSORS,
    TRAILING_FEATURE_MODE_SENSOR_AGE,
    TRAILING_FEATURE_MODES,
    VALID_SLICE,
    LayoutMagneticStudent,
    normalize_trailing_feature_mode,
    schema_for_trailing_feature_mode,
)
from unitree_rl_lab.traction.networks import LegacyLocomotionActor  # noqa: E402


LEG_JOINTS = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)
WAIST_JOINTS = (2, 5, 8)


class FrozenTeacher(nn.Module):
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


class HallPolicyDiagnostics(nn.Module):
    """Optional monitor export; none of these outputs are measured force."""

    def __init__(self, policy: LayoutMagneticStudent) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, estimated_mu, slip_risk_lr, confidence, _ = self.policy.all_outputs(
            observation
        )
        traction_score = torch.clamp(estimated_mu / 1.30, 0.0, 1.0)
        return action, traction_score, slip_risk_lr, confidence


class HallRiskOutput(nn.Module):
    """Single-output wrapper for the deployment command governor.

    The scalar is a learned risk probability computed from the same causal
    Hall/proprioceptive observation as the action policy.  It is deliberately
    not described as force or friction: no Hall-to-force inverse is present.
    """

    def __init__(self, policy: LayoutMagneticStudent) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        _, estimated_mu, _, confidence, _ = self.policy.all_outputs(
            observation
        )
        risk = torch.sigmoid((0.25 - estimated_mu) / 0.05)
        # A stale/missing Hall stream must remain conservative.  Confidence
        # zero therefore maps to maximum risk instead of a false clear state.
        return confidence * risk + (1.0 - confidence)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("obs", "teacher_obs", "mu", "cmd_vx")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        result = {key: np.asarray(data[key], dtype=np.float32) for key in required}
        count = len(result["obs"])
        sample_weight = np.asarray(
            data["sample_weight"] if "sample_weight" in data else np.ones(count),
            dtype=np.float32,
        ).reshape(count)
        if "time_since_switch_s" in data:
            switch_transition = (
                np.asarray(data["time_since_switch_s"], dtype=np.float32).reshape(count)
                <= 1.0
            ).astype(np.float32)
        else:
            switch_transition = np.full(
                count,
                1.0 if "switch" in str(path).casefold() else 0.0,
                dtype=np.float32,
            )
    if result["obs"].ndim != 2 or result["obs"].shape[1] != INPUT_DIM:
        raise ValueError(f"{path}: expected Nx{INPUT_DIM} Student obs")
    if result["teacher_obs"].ndim != 2 or result["teacher_obs"].shape[1] != 641:
        raise ValueError(f"{path}: expected Nx641 Teacher obs")
    finite = (
        np.isfinite(result["obs"]).all(axis=1)
        & np.isfinite(result["teacher_obs"]).all(axis=1)
        & np.isfinite(result["mu"].reshape(count))
        & np.isfinite(result["cmd_vx"].reshape(count))
        & np.isfinite(sample_weight)
    )
    return {
        "obs": result["obs"][finite],
        "teacher_obs": result["teacher_obs"][finite],
        "mu": result["mu"].reshape(count)[finite],
        "cmd_vx": result["cmd_vx"].reshape(count)[finite],
        "sample_weight": np.maximum(sample_weight[finite], 0.0),
        "switch_transition": switch_transition[finite],
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def batched_infer(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            result = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            output.append(result.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def filter_stable_action_samples(
    data: dict[str, np.ndarray],
    raw_teacher: np.ndarray,
    baseline_action: np.ndarray,
    max_abs_action: float,
) -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, int]
]:
    """Remove post-fall/reset action outliers before imitation training.

    Isaac's managed environments reset terminated rows automatically.  The
    first stale history frames after a reset can make a privileged Teacher or
    frozen legacy actor emit huge actions, even though the action manager
    later clips them.  Such rows are neither deployable walking states nor
    valid recovery targets.  Leaving them in a bounded imitation loss creates
    contradictory labels and degrades low-traction recovery.

    This is an offline supervision-quality filter only.  It neither inspects
    forces nor reconstructs them from Hall measurements, and it is applied to
    both train and held-out validation data.
    """

    if max_abs_action <= 0.0:
        raise ValueError("max_abs_action must be positive")
    count = len(data["obs"])
    if (
        raw_teacher.shape != baseline_action.shape
        or raw_teacher.ndim != 2
        or raw_teacher.shape[0] != count
        or raw_teacher.shape[1] != ACTION_DIM
    ):
        raise ValueError("action predictions and dataset rows must agree")
    stable = (
        np.isfinite(raw_teacher).all(axis=1)
        & np.isfinite(baseline_action).all(axis=1)
        & (np.max(np.abs(raw_teacher), axis=1) <= max_abs_action)
        & (np.max(np.abs(baseline_action), axis=1) <= max_abs_action)
    )
    if not np.any(stable):
        raise RuntimeError("stable-action filter removed every training sample")
    filtered = {key: value[stable] for key, value in data.items()}
    report = {
        "input_samples": int(count),
        "kept_samples": int(stable.sum()),
        "dropped_post_reset_or_outlier_samples": int((~stable).sum()),
    }
    return filtered, raw_teacher[stable], baseline_action[stable], report


def apply_foot_dropout_metadata(
    observation: torch.Tensor,
    foot_dropout: torch.Tensor,
    trailing_feature_mode: str,
) -> torch.Tensor:
    """Apply link validity without overwriting Motion feedback as packet age."""

    mode = normalize_trailing_feature_mode(trailing_feature_mode)
    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}")
    dropout = torch.as_tensor(
        foot_dropout, device=observation.device, dtype=torch.bool
    )
    if dropout.shape != (observation.shape[0], FEET):
        raise ValueError(
            f"foot_dropout must have shape {(observation.shape[0], FEET)}, "
            f"got {tuple(dropout.shape)}"
        )
    observation[:, VALID_SLICE] *= (~dropout).to(observation.dtype)
    if mode == TRAILING_FEATURE_MODE_SENSOR_AGE:
        observation[:, AGE_SLICE] = torch.where(
            dropout,
            torch.ones_like(observation[:, AGE_SLICE]),
            observation[:, AGE_SLICE],
        )
    return observation


def randomize_hall(
    observation: torch.Tensor,
    trailing_feature_mode: str,
) -> torch.Tensor:
    """Direct Hall-domain randomization; no force reconstruction is used.

    The trailing-feature mode is intentionally mandatory.  Motion policies use
    channels 1862:1864 for ``body_vy`` and ``relative_heading`` whereas legacy
    policies use them for packet age.  A silent sensor-age default can
    therefore produce a shape-correct but semantically corrupt training row.
    """

    result = observation.clone()
    batch = len(result)
    hall = result[:, MAGNETIC_SLICE].reshape(
        batch, HISTORY, FEET, SENSORS, AXES
    )

    sensor_gain = 0.72 + 0.56 * torch.rand(
        batch, 1, FEET, SENSORS, 1, device=hall.device, dtype=hall.dtype
    )
    axis_gain = 0.78 + 0.44 * torch.rand(
        batch, 1, FEET, 1, AXES, device=hall.device, dtype=hall.dtype
    )
    hall = hall * sensor_gain * axis_gain

    identity = torch.eye(AXES, device=hall.device, dtype=hall.dtype).view(
        1, 1, 1, AXES, AXES
    )
    coupling = identity + 0.05 * torch.randn(
        batch, 1, FEET, AXES, AXES, device=hall.device, dtype=hall.dtype
    )
    hall = torch.einsum("ntfsa,ntfac->ntfsc", hall, coupling)

    static_bias = 0.10 * torch.randn(
        batch, 1, FEET, SENSORS, AXES, device=hall.device, dtype=hall.dtype
    )
    drift = 0.08 * torch.randn_like(static_bias)
    ramp = torch.linspace(-0.5, 0.5, HISTORY, device=hall.device, dtype=hall.dtype)
    hall = hall + static_bias + drift * ramp.view(1, HISTORY, 1, 1, 1)
    noise = (0.025 + 0.018 * hall.abs()) * torch.randn_like(hall)
    hall = hall + noise

    point_keep = (
        torch.rand(batch, 1, FEET, SENSORS, 1, device=hall.device) >= 0.02
    ).to(hall.dtype)
    hall = hall * point_keep
    saturation = 4.5 + 1.5 * torch.rand(
        batch, 1, FEET, 1, 1, device=hall.device, dtype=hall.dtype
    )
    hall = torch.maximum(torch.minimum(hall, saturation), -saturation)

    # Independent causal packet delay for each foot, constant within a sample.
    delayed = torch.empty_like(hall)
    delay = torch.multinomial(
        hall.new_tensor([0.60, 0.25, 0.10, 0.05]).expand(batch * FEET, -1),
        1,
    ).reshape(batch, FEET)
    time_index = torch.arange(HISTORY, device=hall.device)
    for foot in range(FEET):
        index = (time_index.view(1, -1) - delay[:, foot : foot + 1]).clamp_min(0)
        gather_index = index[:, :, None, None].expand(-1, -1, SENSORS, AXES)
        delayed[:, :, foot] = torch.gather(hall[:, :, foot], 1, gather_index)
    hall = delayed

    foot_dropout = torch.rand(batch, FEET, device=hall.device) < 0.025
    if torch.any(foot_dropout):
        hall = hall * (~foot_dropout)[:, None, :, None, None].to(hall.dtype)
        apply_foot_dropout_metadata(result, foot_dropout, trailing_feature_mode)

    result[:, MAGNETIC_SLICE] = hall.reshape(batch, MAGNETIC_DIM)
    period = result[:, PERIOD_SLICE].reshape(batch, HISTORY, FEET)
    period = (period * (0.90 + 0.20 * torch.rand(
        batch, 1, FEET, device=period.device, dtype=period.dtype
    )) + 0.002 * torch.randn_like(period)).clamp(0.001, 0.25)
    result[:, PERIOD_SLICE] = period.reshape(batch, PERIOD_DIM)
    return result


def action_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    absolute = np.abs(prediction - reference)
    return {
        "samples": int(len(reference)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - reference)))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(absolute.max()),
    }


def evaluate(
    model: LayoutMagneticStudent,
    observation: np.ndarray,
    target: np.ndarray,
    raw_teacher: np.ndarray,
    mu: np.ndarray,
    device: torch.device,
    batch_size: int,
    randomized: bool,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    actions: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(observation[start : start + batch_size]).to(device)
            if randomized:
                batch = randomize_hall(batch, model.trailing_feature_mode)
            action, predicted_mu, risk, confidence, residual = model.all_outputs(batch)
            actions.append(action.cpu().numpy())
            mus.append(predicted_mu.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
            residuals.append(residual.cpu().numpy())
            risks.append(risk.cpu().numpy())
    action = np.concatenate(actions)
    predicted_mu = np.concatenate(mus)[:, 0]
    confidence = np.concatenate(confidences)[:, 0]
    residual = np.concatenate(residuals)
    risk = np.concatenate(risks)
    risk_target = (
        1.0 / (1.0 + np.exp((mu - 0.25) / 0.05))
    ) * np.clip(np.abs(observation[:, 30]) / 0.60, 0.0, 1.0)
    low = mu <= 0.25
    high = mu >= 0.75
    def selected_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(values[mask].mean()) if np.any(mask) else None

    extreme_parts = []
    if np.any(low):
        extreme_parts.append(predicted_mu[low] < 0.45)
    if np.any(high):
        extreme_parts.append(predicted_mu[high] > 0.60)
    result: dict[str, object] = {
        "bounded_target": action_metrics(target, action),
        "raw_teacher": action_metrics(raw_teacher, action),
        "mu_mae": float(np.mean(np.abs(predicted_mu - mu))),
        "mu_low_mean": selected_mean(predicted_mu, low),
        "mu_high_mean": selected_mean(predicted_mu, high),
        "mu_extreme_accuracy": (
            float(np.mean(np.concatenate(extreme_parts))) if extreme_parts else None
        ),
        "confidence_mean": float(confidence.mean()),
        "slip_risk_mae": float(
            np.mean(np.abs(risk - risk_target[:, None]))
        ),
        "slip_risk_mean": float(risk.mean()),
        "residual_max_abs": float(np.abs(residual).max()),
        "finite": bool(
            np.isfinite(action).all()
            and np.isfinite(predicted_mu).all()
            and np.isfinite(risk).all()
            and np.isfinite(confidence).all()
        ),
    }
    return result, action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    shared = ROOT / "logs/evaluations/traction_shared_magnetic"
    parser.add_argument(
        "--train", type=Path, action="append",
        default=None,
        help="Training NPZ (repeatable). Defaults to the two legacy datasets only when omitted.",
    )
    parser.add_argument(
        "--test", type=Path, action="append",
        default=None,
        help="Validation NPZ (repeatable). Defaults to the two legacy datasets only when omitted.",
    )
    parser.add_argument(
        "--teacher", type=Path,
        default=ROOT / "logs/imported_remote_20260728/teacher_8110.onnx",
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "model/rl/model_49999.pt"
    )
    parser.add_argument(
        "--initialize",
        type=Path,
        help="Optional layout_magnetic_student.pt used for DAgger continuation",
    )
    parser.add_argument(
        "--low-behavior-policy",
        type=Path,
        help="Optional safe Hall policy providing low-friction consolidation labels",
    )
    parser.add_argument(
        "--high-behavior-policy",
        type=Path,
        help="Optional fast Hall policy providing higher-friction consolidation labels",
    )
    parser.add_argument("--behavior-switch-mu", type=float, default=0.15)
    parser.add_argument("--behavior-switch-width", type=float, default=0.025)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/layout_magnetic_student_20260806",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help=(
            "Stop after this many non-improving epochs and restore the best "
            "finite training state before exporting.  Zero disables early stop "
            "but still restores the best state."
        ),
    )
    parser.add_argument("--residual-limit", type=float, default=1.0)
    parser.add_argument(
        "--trailing-feature-mode",
        choices=TRAILING_FEATURE_MODES,
        required=True,
        help=(
            "Meaning of policy channels 1862:1864.  Use motion_feedback for "
            "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotion* datasets; "
            "use sensor_age for the ordinary Hall task."
        ),
    )
    parser.add_argument(
        "--mu-loss-weight",
        type=float,
        default=0.08,
        help="Auxiliary usable-traction regression weight (offline label only).",
    )
    parser.add_argument(
        "--risk-loss-weight",
        type=float,
        default=0.04,
        help="Auxiliary causal slip-risk probability weight.",
    )
    parser.add_argument(
        "--hall-randomization-prob",
        type=float,
        default=0.35,
        help="Probability of an additional offline Hall perturbation (Isaac data is already randomized).",
    )
    parser.add_argument(
        "--max-stable-action",
        type=float,
        default=3.0,
        help=(
            "Discard DAgger rows whose raw Teacher or frozen baseline action "
            "exceeds this magnitude; they are post-fall/reset outliers, not "
            "valid walking supervision."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.trailing_feature_mode = normalize_trailing_feature_mode(
        args.trailing_feature_mode
    )
    if not 0.0 <= args.hall_randomization_prob <= 1.0:
        raise ValueError("--hall-randomization-prob must be in [0,1]")
    if args.mu_loss_weight < 0.0 or args.risk_loss_weight < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.max_stable_action <= 0.0:
        raise ValueError("--max-stable-action must be positive")
    if args.train is None:
        shared = ROOT / "logs/evaluations/traction_shared_magnetic"
        args.train = [
            shared / "20260728_shared15x3_lateral8110_dagger_data/train.npz",
            shared / "20260728_dagger1_mujoco_dagger_data/train.npz",
        ]
    if args.test is None:
        shared = ROOT / "logs/evaluations/traction_shared_magnetic"
        args.test = [
            shared / "20260728_shared15x3_lateral8110_dagger_data/test.npz",
            shared / "20260728_dagger1_mujoco_dagger_data/test.npz",
        ]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = concatenate([load_dataset(path) for path in args.train])
    validation = concatenate([load_dataset(path) for path in args.test])
    teacher = FrozenTeacher(args.teacher).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    baseline_checkpoint = torch.load(
        args.baseline, map_location="cpu", weights_only=False
    )
    baseline = LegacyLocomotionActor(BASE_DIM)
    baseline.load_state_dict(
        {
            key: value
            for key, value in baseline_checkpoint["actor_state_dict"].items()
            if key.startswith("mlp.")
        },
        strict=True,
    )
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    print(
        f"device={device} train={len(training['obs'])} "
        f"test={len(validation['obs'])} input={INPUT_DIM}",
        flush=True,
    )
    print("precomputing privileged Teacher labels (labels only)", flush=True)
    prepared_data: list[dict[str, np.ndarray]] = []
    stable_action_filter: dict[str, dict[str, int]] = {}
    for label, data in (("train", training), ("validation", validation)):
        raw_teacher = batched_infer(
            teacher, data.pop("teacher_obs"), device, args.batch_size * 2
        )
        baseline_action = batched_infer(
            baseline, data["obs"][:, :BASE_DIM], device, args.batch_size * 2
        )
        data, raw_teacher, baseline_action, report = filter_stable_action_samples(
            data, raw_teacher, baseline_action, args.max_stable_action
        )
        stable_action_filter[label] = report
        data["raw_teacher"] = raw_teacher
        data["baseline"] = np.clip(
            baseline_action, -ACTION_OUTPUT_LIMIT, ACTION_OUTPUT_LIMIT
        )
        data["target"] = np.clip(
            data["baseline"] + np.clip(
                raw_teacher - data["baseline"],
                -args.residual_limit,
                args.residual_limit,
            ),
            -ACTION_OUTPUT_LIMIT,
            ACTION_OUTPUT_LIMIT,
        )
        prepared_data.append(data)
    training, validation = prepared_data
    print(
        "stable action filter: "
        f"train={stable_action_filter['train']['kept_samples']}/"
        f"{stable_action_filter['train']['input_samples']}, "
        f"validation={stable_action_filter['validation']['kept_samples']}/"
        f"{stable_action_filter['validation']['input_samples']}",
        flush=True,
    )

    behavior_sources: dict[str, str] | None = None
    if (args.low_behavior_policy is None) != (args.high_behavior_policy is None):
        raise ValueError("both low/high behavior policies are required together")
    if args.low_behavior_policy is not None and args.high_behavior_policy is not None:
        if args.behavior_switch_width <= 0.0:
            raise ValueError("--behavior-switch-width must be positive")

        def load_behavior(path: Path) -> LayoutMagneticStudent:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model")
            if not isinstance(state, dict):
                raise ValueError(f"{path}: checkpoint has no model state")
            policy = LayoutMagneticStudent(
                float(payload.get("residual_limit", args.residual_limit)),
                trailing_feature_mode=normalize_trailing_feature_mode(
                    str(
                        payload.get(
                            "trailing_feature_mode",
                            TRAILING_FEATURE_MODES[0],
                        )
                    )
                ),
            )
            policy.load_state_dict(state, strict=True)
            policy = policy.to(device).eval()
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
            return policy

        low_behavior = load_behavior(args.low_behavior_policy)
        high_behavior = load_behavior(args.high_behavior_policy)
        print(
            "consolidating low-friction safe and higher-friction fast Hall behaviors",
            flush=True,
        )
        for data in (training, validation):
            low_action = batched_infer(
                low_behavior, data["obs"], device, args.batch_size * 2
            )
            high_action = batched_infer(
                high_behavior, data["obs"], device, args.batch_size * 2
            )
            gate = 1.0 / (
                1.0
                + np.exp(
                    -(
                        data["mu"] - args.behavior_switch_mu
                    ) / args.behavior_switch_width
                )
            )
            consolidated = (
                (1.0 - gate[:, None]) * low_action + gate[:, None] * high_action
            )
            bounded_consolidated = np.clip(
                data["baseline"] + np.clip(
                    consolidated - data["baseline"],
                    -args.residual_limit,
                    args.residual_limit,
                ),
                -ACTION_OUTPUT_LIMIT,
                ACTION_OUTPUT_LIMIT,
            )
            # Only transition rollouts on the low-friction phase need the
            # privileged Teacher's recovery action.  Steady low-friction data
            # retains the safe Hall-policy label and higher friction retains
            # the fast Hall-policy label.
            transition_recovery = (
                (data["switch_transition"] > 0.5)
                & (data["mu"] <= args.behavior_switch_mu)
            )
            teacher_recovery = np.clip(
                data["baseline"] + np.clip(
                    data["raw_teacher"] - data["baseline"],
                    -args.residual_limit,
                    args.residual_limit,
                ),
                -ACTION_OUTPUT_LIMIT,
                ACTION_OUTPUT_LIMIT,
            )
            data["target"] = np.where(
                transition_recovery[:, None], teacher_recovery, bounded_consolidated
            ).astype(np.float32)
        behavior_sources = {
            "low": str(args.low_behavior_policy.resolve()),
            "high": str(args.high_behavior_policy.resolve()),
            "switch_mu": str(args.behavior_switch_mu),
            "switch_width": str(args.behavior_switch_width),
        }

    model = LayoutMagneticStudent(
        args.residual_limit,
        trailing_feature_mode=args.trailing_feature_mode,
    )
    model.load_baseline_checkpoint(baseline_checkpoint)
    if args.initialize is not None:
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        state = initial.get("model")
        if not isinstance(state, dict):
            raise ValueError(f"{args.initialize}: checkpoint has no model state")
        model.load_state_dict(state, strict=True)
    model = model.to(device)
    for parameter in model.baseline_actor.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training["obs"]),
            torch.from_numpy(training["target"]),
            torch.from_numpy(training["baseline"]),
            torch.from_numpy(training["mu"][:, None]),
            torch.from_numpy(training["cmd_vx"][:, None]),
            torch.from_numpy(training["sample_weight"][:, None]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    joint_weight = torch.ones(ACTION_DIM, device=device)
    joint_weight[list(LEG_JOINTS)] = 2.0
    joint_weight[list(WAIST_JOINTS)] = 1.5
    joint_weight_sum = joint_weight.sum()
    history: list[dict[str, float]] = []
    started = time.monotonic()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "action": 0.0,
            "mu": 0.0,
            "risk": 0.0,
            "fallback": 0.0,
        }
        batches = 0
        for observation, target, base_action, mu, command, sample_weight in loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            base_action = base_action.to(device, non_blocking=True)
            mu = mu.to(device, non_blocking=True)
            command = command.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            randomized_all = randomize_hall(
                observation, args.trailing_feature_mode
            )
            domain_randomized = (
                torch.rand(len(observation), device=device)
                < args.hall_randomization_prob
            )
            randomized = torch.where(
                domain_randomized[:, None], randomized_all, observation
            )

            # Full link-loss examples explicitly teach exact baseline fallback.
            loss_mask = torch.rand(len(randomized), device=device) < 0.08
            if torch.any(loss_mask):
                randomized[loss_mask, MAGNETIC_SLICE] = 0.0
                randomized[loss_mask, VALID_SLICE] = 0.0
                if args.trailing_feature_mode == TRAILING_FEATURE_MODE_SENSOR_AGE:
                    randomized[loss_mask, AGE_SLICE] = 1.0
            effective_target = torch.where(loss_mask[:, None], base_action, target)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                action, estimated_mu, slip_risk, confidence, residual = model.all_outputs(
                    randomized
                )
                component = nn.functional.smooth_l1_loss(
                    action, effective_target, beta=0.05, reduction="none"
                ) * joint_weight
                per_sample = component.sum(dim=1, keepdim=True) / joint_weight_sum
                weight = torch.ones_like(mu)
                weight += (mu <= 0.25).to(mu.dtype)
                weight += (mu >= 0.75).to(mu.dtype)
                weight += 2.0 * ((mu >= 0.75) & (command >= 0.70)).to(mu.dtype)
                # Full MuJoCo matrix DAgger exposed a narrow ice/mid-command
                # recovery hole that endpoint-only training could not see.
                weight += 3.0 * (
                    (mu <= 0.12) & (command >= 0.30) & (command <= 0.70)
                ).to(mu.dtype)
                weight += 3.0 * ((mu <= 0.12) & (command >= 0.70)).to(mu.dtype)
                weight *= sample_weight
                action_loss = (per_sample * weight).sum() / weight.sum().clamp_min(1.0e-6)
                mu_loss = nn.functional.smooth_l1_loss(
                    estimated_mu, mu, beta=0.08
                )
                risk_target = (
                    torch.sigmoid((0.25 - mu) / 0.05)
                    * torch.clamp(torch.abs(command) / 0.60, 0.0, 1.0)
                ).expand(-1, FEET)
                # BCE on post-sigmoid probabilities is evaluated explicitly
                # in float32: fp16 rounds 1-1e-5 back to one, which makes the
                # logarithmic form infinite, and PyTorch deliberately rejects
                # probability BCE while autocast is active.
                with torch.autocast(device_type=device.type, enabled=False):
                    risk_loss = nn.functional.binary_cross_entropy(
                        slip_risk.float(), risk_target.float()
                    )
                fallback_loss = (
                    (action[loss_mask] - base_action[loss_mask]).square().mean()
                    if torch.any(loss_mask)
                    else action.sum() * 0.0
                )
                confidence_target = (~loss_mask).to(confidence.dtype)[:, None]
                confidence_loss = nn.functional.mse_loss(
                    confidence, confidence_target
                )
                loss = (
                    action_loss + args.mu_loss_weight * mu_loss
                    + 0.10 * fallback_loss
                    + args.risk_loss_weight * risk_loss
                    + 0.02 * confidence_loss
                    + 0.001 * residual.square().mean()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["action"] += float(action_loss.detach())
            totals["mu"] += float(mu_loss.detach())
            totals["risk"] += float(risk_loss.detach())
            totals["fallback"] += float(fallback_loss.detach())
            batches += 1
        scheduler.step()
        record = {key: value / max(batches, 1) for key, value in totals.items()}
        record["epoch"] = float(epoch)
        history.append(record)
        if np.isfinite(record["loss"]) and record["loss"] < best_loss:
            best_loss = record["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.monotonic() - started
            eta = elapsed * (args.epochs - epoch) / epoch
            print(
                f"epoch={epoch:03d}/{args.epochs} loss={record['loss']:.6f} "
                f"action={record['action']:.6f} ETA={eta / 60:.1f}m",
                flush=True,
            )
        if (
            args.early_stop_patience > 0
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                f"early-stop epoch={epoch:03d}; restoring best epoch={best_epoch:03d} "
                f"loss={best_loss:.6f}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("no finite Hall Student training state was produced")
    model.load_state_dict(best_state, strict=True)

    nominal, nominal_action = evaluate(
        model,
        validation["obs"],
        validation["target"],
        validation["raw_teacher"],
        validation["mu"],
        device,
        args.batch_size * 2,
        False,
        args.seed + 1,
    )
    randomized, randomized_action = evaluate(
        model,
        validation["obs"],
        validation["target"],
        validation["raw_teacher"],
        validation["mu"],
        device,
        args.batch_size * 2,
        True,
        args.seed + 2,
    )

    checkpoint = args.output_dir / "layout_magnetic_student.pt"
    model = model.cpu().eval()
    torch.save(
        {
            "model": model.state_dict(),
        "schema": schema_for_trailing_feature_mode(
            args.trailing_feature_mode,
            residual_limit=args.residual_limit,
        ).to_dict(),
        "residual_limit": args.residual_limit,
        "trailing_feature_mode": args.trailing_feature_mode,
            "history": history,
            "nominal": nominal,
            "randomized": randomized,
            "seed": args.seed,
        "epochs": args.epochs,
        "hall_randomization_probability": args.hall_randomization_prob,
        },
        checkpoint,
    )
    schema_for_trailing_feature_mode(
        args.trailing_feature_mode,
        residual_limit=args.residual_limit,
    ).write_json(args.output_dir / "observation_schema.json")
    torch.onnx.export(
        model,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
    )
    diagnostics = HallPolicyDiagnostics(model).eval()
    torch.onnx.export(
        diagnostics,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "policy_diagnostics.onnx",
        input_names=["obs"],
        output_names=["actions", "traction_score", "slip_risk_lr", "confidence"],
        opset_version=17,
    )
    hall_risk = HallRiskOutput(model).eval()
    torch.onnx.export(
        hall_risk,
        torch.zeros(1, INPUT_DIM),
        args.output_dir / "hall_risk.onnx",
        input_names=["obs"],
        output_names=["risk_probability"],
        opset_version=17,
    )
    scripted = torch.jit.trace(model, torch.zeros(1, INPUT_DIM))
    scripted.save(str(args.output_dir / "policy.ts"))

    session = ort.InferenceSession(
        str(args.output_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    parity_input = validation["obs"][:1]
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(parity_input)).numpy()
    onnx_output = session.run(None, {session.get_inputs()[0].name: parity_input})[0]
    parity_max_abs = float(np.max(np.abs(torch_output - onnx_output)))

    failure_input = validation["obs"][:1024].copy()
    failure_input[:, MAGNETIC_SLICE] = 0.0
    failure_input[:, VALID_SLICE] = 0.0
    if args.trailing_feature_mode == TRAILING_FEATURE_MODE_SENSOR_AGE:
        failure_input[:, AGE_SLICE] = 1.0
    with torch.inference_mode():
        failure_tensor = torch.from_numpy(failure_input)
        failure_action = model(failure_tensor).numpy()
        fallback_reference = torch.clamp(
            model.baseline_actor(failure_tensor[:, :BASE_DIM]),
            -ACTION_OUTPUT_LIMIT,
            ACTION_OUTPUT_LIMIT,
        ).numpy()
    fallback_error = float(
        np.max(np.abs(failure_action - fallback_reference))
    )
    gates = {
        "finite": bool(nominal["finite"] and randomized["finite"]),
        "nominal_action_mae_le_0p08": nominal["bounded_target"]["mae"] <= 0.08,
        "randomized_action_mae_le_0p10": randomized["bounded_target"]["mae"] <= 0.10,
        "randomized_p95_within_residual_budget": (
            randomized["bounded_target"]["p95_abs"]
            <= max(0.30, args.residual_limit + 1.0e-6)
        ),
        "residual_bound": randomized["residual_max_abs"] <= args.residual_limit + 1.0e-6,
        # CPU/ONNX and CUDA may differ by a few float32 ULPs even though the
        # confidence gate is mathematically zero.  This matches the audited
        # deployment parity tolerance instead of treating 1e-6 as exact math.
        "sensor_loss_exact_fallback": fallback_error <= 1.0e-5,
        "onnx_parity": parity_max_abs <= 1.0e-4,
    }
    summary = {
        "status": "trained_sim_candidate",
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "method": "layout-aware direct Hall-domain privileged Teacher distillation",
        "student_inputs": "proprio + dual-foot 15x3 normalized Hall history + timing/health",
        "trailing_feature_mode": args.trailing_feature_mode,
        "student_forbidden_inputs": SCHEMA.to_dict()["forbidden_student_inputs"],
        "training_samples": int(len(training["obs"])),
        "validation_samples": int(len(validation["obs"])),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(args.output_dir / "policy.onnx"),
        "diagnostics_onnx_sha256": sha256(
            args.output_dir / "policy_diagnostics.onnx"
        ),
        "hall_risk_onnx_sha256": sha256(args.output_dir / "hall_risk.onnx"),
        "nominal": nominal,
        "randomized": randomized,
        "fallback_max_abs": fallback_error,
        "onnx_parity_max_abs": parity_max_abs,
        "gates": gates,
        "loss_weights": {
            "usable_traction": args.mu_loss_weight,
            "slip_risk": args.risk_loss_weight,
        },
        "selection": {
            "best_training_epoch": best_epoch,
            "best_training_loss": best_loss,
            "early_stop_patience": args.early_stop_patience,
            "completed_epochs": len(history),
        },
        "stable_action_filter": stable_action_filter,
        "sources": {
            "train": [str(path.resolve()) for path in args.train],
            "test": [str(path.resolve()) for path in args.test],
            "teacher": str(args.teacher.resolve()),
            "baseline": str(args.baseline.resolve()),
            "initialize": (
                str(args.initialize.resolve()) if args.initialize is not None else None
            ),
            "behavior_consolidation": behavior_sources,
        },
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        mu=validation["mu"],
        cmd_vx=validation["cmd_vx"],
        bounded_target=validation["target"],
        raw_teacher=validation["raw_teacher"],
        nominal_action=nominal_action,
        randomized_action=randomized_action,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["overall"] == "PASS" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
