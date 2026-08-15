"""RSL-RL 5 PPO with a frozen high-friction Hall Teacher anchor.

This module is deliberately project-local.  It subclasses the public RSL-RL
5 interfaces without patching ``site-packages`` and keeps all privileged
course state outside the deployable Actor observation:

* the Actor still consumes only the configured ``policy`` observation group;
* the frozen speedboost Teacher consumes a private copy whose final two
  motion-feedback channels are replaced by the real, training-only Hall ages;
* the physical HIGH_START/HIGH_END stage only gates the auxiliary loss;
* Teacher actions and masks live in a dedicated rollout cache, not in the
  environment observation TensorDict.

The implementation intentionally rejects recurrent policies and symmetry
augmentation.  Neither is used by the 1864-D Hall spatial task, and silently
transforming cached Teacher targets under either feature would be unsafe.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .frozen_speedboost_teacher import (
    INPUT_DIM,
    KNOWN_SPEEDBOOST112_SHA256,
    OUTPUT_DIM,
    VALID_SLICE,
    adapt_teacher_observation,
    load_frozen_speedboost_teacher,
)
from .frozen_low_expert import (
    LOW_EXPERT_COMMAND,
    load_frozen_low_recovery_expert,
    rewrite_term_major_velocity_command,
)


ANCHOR_FORMAT = "unitree_rl_lab.high_friction_teacher_anchor"
ANCHOR_FORMAT_VERSION = 3
SPATIAL_HIGH_START = 0
SPATIAL_LOW = 1
SPATIAL_HIGH_END = 2
HIGH_STAGE_IDS = (SPATIAL_HIGH_START, SPATIAL_HIGH_END)
OPTIMIZER_ROLE_KEY = "unitree_optimizer_role"
PPO_OPTIMIZER_ROLE = "ppo"
STAGE_AUX_OPTIMIZER_ROLE = "stage_auxiliary"
CAPTURE_GATE_OPTIMIZER_ROLE = "capture_gate"
CAPTURE_RESIDUAL_OPTIMIZER_ROLE = "capture_residual"
STABILITY_RESIDUAL_OPTIMIZER_ROLE = "stability_residual"
TRAINING_PROVENANCE_FORMAT = "unitree_rl_lab.training_provenance"
TRAINING_PROVENANCE_FORMAT_VERSION = 1
STRICT_ACTOR_CRITIC_RESUME_CFG = {
    "actor": True,
    "critic": True,
    "optimizer": False,
    "iteration": True,
    "rnd": False,
}


@dataclass(frozen=True)
class StageAuxiliaryTargets:
    """Private binary friction-stage supervision for one rollout step.

    ``label`` is one for LOW and zero for either physical HIGH patch.  ``mask``
    rejects reset frames and unknown stage IDs.  Neither Tensor is part of the
    environment observation TensorDict.
    """

    label: torch.Tensor
    mask: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class StageAuxiliaryLoss:
    """Balanced binary classification loss and detached diagnostics."""

    total: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor
    accuracy: torch.Tensor
    valid_fraction: torch.Tensor
    low_fraction: torch.Tensor


def _require_rsl_rl_v5() -> str:
    """Return the installed version and fail closed on an unaudited API."""

    try:
        installed = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        installed = importlib.metadata.version("rsl_rl")
    major = int(installed.split(".", 1)[0])
    if major != 5:
        raise RuntimeError(
            "AnchoredPPO was audited against RSL-RL 5.x; "
            f"found {installed}. Refuse to guess across storage/update APIs."
        )
    return installed


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without accepting directories or missing inputs."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return _sha256_file(checkpoint)


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Return a stable hash for an observation schema/provenance dictionary."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_bounded_new_updates(requested: int, maximum: int) -> int:
    """Validate a short continuation budget without treating zero as a smoke."""

    requested_updates = int(requested)
    maximum_updates = int(maximum)
    if maximum_updates <= 0:
        raise ValueError("maximum allowed new updates must be positive")
    if requested_updates < 1 or requested_updates > maximum_updates:
        raise RuntimeError(
            "fail-closed continuation update count must satisfy "
            f"1 <= updates <= {maximum_updates}, got {requested_updates}"
        )
    return requested_updates


def validate_hall_randomization_seed(
    base_env: object,
    expected_seed: int,
    *,
    observation_reader=None,
) -> dict[str, Any]:
    """Prove Hall domain randomization uses the effective environment seed.

    Hall construction is observation-lazy in manager environments.  The
    optional reader may therefore perform one public, read-only observation
    request before the check; no physics step or reset is issued here.
    """

    expected = int(expected_seed)
    actual = getattr(base_env, "_hall_foot_sensor_seed", None)
    initialized_on_demand = False
    if actual is None and observation_reader is not None:
        observation_reader()
        initialized_on_demand = True
        actual = getattr(base_env, "_hall_foot_sensor_seed", None)
    if actual is None:
        raise RuntimeError(
            "Hall sensor seed audit unavailable after observation initialization"
        )
    actual_seed = int(actual)
    if actual_seed != expected:
        raise RuntimeError(
            "Hall sensor domain-randomization seed mismatch: "
            f"environment={expected}, sensor={actual_seed}"
        )
    return {
        "environment_seed": expected,
        "hall_foot_sensor_seed": actual_seed,
        "match": True,
        "initialized_by_read_only_observation": initialized_on_demand,
    }


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _finite_scalar(value: object, *, name: str) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() != 1:
            raise RuntimeError(f"{name} must be scalar, got {tuple(tensor.shape)}")
        result = float(tensor.item())
    else:
        result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite, got {result}")
    return result


def actor_exploration_std_manifest(actor: MLPModel) -> dict[str, Any]:
    """Read the state-independent RSL Gaussian std without sampling actions."""

    distribution = getattr(actor, "distribution", None)
    if distribution is None:
        raise RuntimeError("actor has no exploration distribution")
    std_type = str(getattr(distribution, "std_type", ""))
    if std_type == "scalar" and torch.is_tensor(
        getattr(distribution, "std_param", None)
    ):
        values = distribution.std_param.detach().cpu()
    elif std_type == "log" and torch.is_tensor(
        getattr(distribution, "log_std_param", None)
    ):
        values = torch.exp(distribution.log_std_param.detach().cpu())
    else:
        raise RuntimeError(f"unsupported actor exploration distribution {std_type!r}")
    if values.ndim != 1 or not torch.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("actor exploration std is non-finite, non-positive, or not 1-D")
    return {
        "parameterization": std_type,
        "dimension": int(values.numel()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "values": [float(value) for value in values.tolist()],
    }


def optimizer_roles_manifest(algorithm: object) -> list[dict[str, Any]]:
    """Describe live optimizer ownership using stable role names."""

    optimizer = getattr(algorithm, "optimizer", None)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise RuntimeError("algorithm optimizer parameter groups are unavailable")
    result: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        role = group.get(OPTIMIZER_ROLE_KEY)
        parameters = list(group.get("params", []))
        learning_rate = _finite_scalar(
            group.get("lr", float("nan")), name=f"optimizer group {index} lr"
        )
        result.append(
            {
                "index": index,
                "role": role,
                "learning_rate": learning_rate,
                "parameter_tensor_count": len(parameters),
                "trainable_parameter_tensor_count": sum(
                    bool(getattr(parameter, "requires_grad", False))
                    for parameter in parameters
                ),
                "parameter_count": sum(
                    int(parameter.numel())
                    for parameter in parameters
                    if torch.is_tensor(parameter)
                ),
            }
        )
    return result


def validate_fail_closed_gate_training_start(
    runner: object,
    *,
    load_mode: str,
    source_checkpoint_sha256: str,
    required_checkpoint_sha256: str,
    required_checkpoint_iteration: int,
    required_capture_gate_completed_updates: int,
    required_actor_observation_dim: int,
    required_critic_observation_dim: int,
    required_action_dim: int,
    required_gate_logit_scale: float,
    required_gate_logit_bias: float,
) -> dict[str, Any]:
    """Prove GateBceOnly is released and correctly wired before rollout.

    This deliberately checks live tensors and optimizer ownership in addition
    to configuration strings.  A dimension-compatible but semantically wrong
    checkpoint, a fresh 12-update run, or a residual left at ``lr=0`` therefore
    cannot silently consume simulation time.
    """

    if load_mode != "strict_resume":
        raise RuntimeError(
            "fail-closed GateBceOnly training requires --resume_checkpoint"
        )
    if source_checkpoint_sha256 != required_checkpoint_sha256:
        raise RuntimeError(
            "GateBceOnly source checkpoint SHA-256 mismatch: "
            f"expected {required_checkpoint_sha256}, got {source_checkpoint_sha256}"
        )
    load_audit = getattr(runner, "checkpoint_load_audit", None)
    if not isinstance(load_audit, dict):
        raise RuntimeError("strict checkpoint load audit is unavailable")
    if load_audit.get("load_cfg") != STRICT_ACTOR_CRITIC_RESUME_CFG:
        raise RuntimeError(
            "GateBceOnly requires strict actor+critic+iteration loading with "
            "optimizer=false"
        )
    if load_audit.get("strict") is not True:
        raise RuntimeError("GateBceOnly checkpoint tensors must load with strict=True")
    source_iteration = int(load_audit.get("saved_completed_iteration", -1))
    if source_iteration != int(required_checkpoint_iteration):
        raise RuntimeError(
            "GateBceOnly source iteration mismatch: "
            f"expected {required_checkpoint_iteration}, got {source_iteration}"
        )
    expected_next = source_iteration + 1
    if int(getattr(runner, "current_learning_iteration", -1)) != expected_next:
        raise RuntimeError(
            "resume iteration semantics are inconsistent: expected next iteration "
            f"{expected_next}, got {getattr(runner, 'current_learning_iteration', None)}"
        )

    algorithm = getattr(runner, "alg", None)
    actor = getattr(algorithm, "actor", None)
    critic = getattr(algorithm, "critic", None)
    actor_dim = int(getattr(actor, "obs_dim", -1))
    critic_dim = int(getattr(critic, "obs_dim", -1))
    action_dim = int(getattr(getattr(runner, "env", None), "num_actions", -1))
    if list(getattr(actor, "obs_groups", [])) != ["policy"]:
        raise RuntimeError("GateBceOnly actor must consume only the policy group")
    if list(getattr(critic, "obs_groups", [])) != ["critic"]:
        raise RuntimeError("GateBceOnly critic must consume only the critic group")
    if actor_dim != int(required_actor_observation_dim):
        raise RuntimeError(
            f"GateBceOnly actor observation dimension must be {required_actor_observation_dim}, got {actor_dim}"
        )
    if critic_dim != int(required_critic_observation_dim):
        raise RuntimeError(
            f"GateBceOnly critic observation dimension must be {required_critic_observation_dim}, got {critic_dim}"
        )
    if action_dim != int(required_action_dim):
        raise RuntimeError(
            f"GateBceOnly action dimension must be {required_action_dim}, got {action_dim}"
        )
    actor_output_dim = int(
        getattr(getattr(actor, "distribution", None), "output_dim", -1)
    )
    if actor_output_dim != action_dim:
        raise RuntimeError(
            "actor distribution/action-manager dimension mismatch: "
            f"actor={actor_output_dim}, environment={action_dim}"
        )

    completed = int(getattr(algorithm, "capture_gate_updates_completed", -1))
    warmup_updates = int(getattr(algorithm, "capture_gate_warmup_updates", -1))
    minimum_completed = max(
        int(required_capture_gate_completed_updates), warmup_updates
    )
    if completed < minimum_completed:
        raise RuntimeError(
            "GateBceOnly residual is not released: "
            f"completed={completed}, required>={minimum_completed}"
        )
    if bool(getattr(algorithm, "capture_gate_warmup_active", True)):
        raise RuntimeError("GateBceOnly capture gate warm-up is still active")
    if str(getattr(algorithm, "capture_gate_gradient_mode", "")) != "stage_bce_only":
        raise RuntimeError("GateBceOnly gate must use stage_bce_only gradients")
    if str(getattr(algorithm, "low_expert_residual_gradient_mode", "")) != "supervised_only":
        raise RuntimeError("GateBceOnly residual must use supervised_only gradients")

    residual = getattr(algorithm, "capture_residual", None)
    residual_parameters = list(
        getattr(algorithm, "capture_residual_parameters", [])
    )
    if residual is None or not residual_parameters:
        raise RuntimeError("GateBceOnly capture residual branch is unavailable")
    if not all(parameter.requires_grad for parameter in residual_parameters):
        raise RuntimeError("GateBceOnly residual parameters are not all trainable")
    residual_lr = _finite_scalar(
        getattr(algorithm, "capture_residual_current_learning_rate", 0.0),
        name="capture residual learning rate",
    )
    if residual_lr <= 0.0:
        raise RuntimeError("GateBceOnly residual learning rate must be positive")
    residual_groups = [
        group
        for group in getattr(algorithm.optimizer, "param_groups", [])
        if group.get(OPTIMIZER_ROLE_KEY) == CAPTURE_RESIDUAL_OPTIMIZER_ROLE
    ]
    if len(residual_groups) != 1:
        raise RuntimeError("GateBceOnly requires exactly one residual optimizer role")
    group = residual_groups[0]
    if {id(parameter) for parameter in group["params"]} != {
        id(parameter) for parameter in residual_parameters
    }:
        raise RuntimeError("residual optimizer role owns the wrong parameters")
    group_lr = _finite_scalar(group.get("lr", 0.0), name="residual optimizer lr")
    if group_lr <= 0.0 or not math.isclose(group_lr, residual_lr, rel_tol=1.0e-9):
        raise RuntimeError(
            "residual optimizer learning rate is not released consistently: "
            f"group={group_lr}, algorithm={residual_lr}"
        )

    mean_module = getattr(actor, "mlp", None)
    gate_logit_scale = _finite_scalar(
        getattr(mean_module, "gate_logit_scale", float("nan")),
        name="gate logit scale",
    )
    gate_logit_bias = _finite_scalar(
        getattr(mean_module, "gate_logit_bias", float("nan")),
        name="gate logit bias",
    )
    if not math.isclose(
        gate_logit_scale, float(required_gate_logit_scale), rel_tol=0.0, abs_tol=1.0e-6
    ) or not math.isclose(
        gate_logit_bias, float(required_gate_logit_bias), rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise RuntimeError(
            "GateBceOnly calibration mismatch: "
            f"expected ({required_gate_logit_scale}, {required_gate_logit_bias}), "
            f"got ({gate_logit_scale}, {gate_logit_bias})"
        )

    return {
        "source_checkpoint_iteration": source_iteration,
        "source_last_completed_iteration": source_iteration,
        "next_learning_iteration": expected_next,
        "capture_gate_updates_completed": completed,
        "capture_gate_warmup_updates": warmup_updates,
        "capture_residual_learning_rate": residual_lr,
        "dimensions": {
            "actor_observation": actor_dim,
            "critic_observation": critic_dim,
            "action": action_dim,
        },
        "gate_calibration": {
            "logit_scale": gate_logit_scale,
            "logit_bias": gate_logit_bias,
            "legacy_checkpoint_values_injected_from_audited_runner_config": bool(
                getattr(mean_module, "loaded_legacy_calibration", False)
            ),
        },
        "optimizer_roles": optimizer_roles_manifest(algorithm),
    }


def high_friction_anchor_mask(
    stage: torch.Tensor,
    *,
    finite_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[N,1]`` mask for HIGH_START/HIGH_END only.

    ``SPATIAL_LOW`` and unknown/completed states never receive an anchor.  A
    per-environment finite mask can additionally fail closed on malformed
    Teacher inputs or outputs.
    """

    if stage.ndim == 2 and stage.shape[1] == 1:
        stage = stage[:, 0]
    if stage.ndim != 1:
        raise ValueError(f"stage must have shape [N] or [N,1], got {tuple(stage.shape)}")
    mask = (stage == SPATIAL_HIGH_START) | (stage == SPATIAL_HIGH_END)
    if finite_mask is not None:
        if finite_mask.ndim == 2 and finite_mask.shape[1] == 1:
            finite_mask = finite_mask[:, 0]
        if finite_mask.shape != stage.shape:
            raise ValueError(
                f"finite_mask must have shape {tuple(stage.shape)}, got {tuple(finite_mask.shape)}"
            )
        mask &= finite_mask.to(device=mask.device, dtype=torch.bool)
    return mask.unsqueeze(1)


def stage_auxiliary_targets(
    stage: torch.Tensor,
    episode_length: torch.Tensor,
    *,
    reset_mask_steps: int = 1,
    high_end_weight: float = 1.0,
) -> StageAuxiliaryTargets:
    """Convert private H--L--H state into leak-free binary targets.

    HIGH_START (0) and HIGH_END (2) are both labelled HIGH (zero); LOW (1) is
    labelled one.  Completed/corrupt stage values are unknown and therefore
    masked.  The first ``reset_mask_steps`` observations of each episode are
    also masked because the event manager and Hall history are still being
    initialized then.
    """

    if stage.ndim == 2 and stage.shape[1] == 1:
        stage = stage[:, 0]
    if episode_length.ndim == 2 and episode_length.shape[1] == 1:
        episode_length = episode_length[:, 0]
    if stage.ndim != 1 or episode_length.shape != stage.shape:
        raise ValueError(
            "stage and episode_length must share shape [N] (or [N,1]), got "
            f"{tuple(stage.shape)} and {tuple(episode_length.shape)}"
        )
    if reset_mask_steps < 0:
        raise ValueError("reset_mask_steps must be non-negative")
    if high_end_weight < 1.0:
        raise ValueError("high_end_weight must be at least 1.0")

    known = (
        (stage == SPATIAL_HIGH_START)
        | (stage == SPATIAL_LOW)
        | (stage == SPATIAL_HIGH_END)
    )
    outside_reset = episode_length.to(device=stage.device) > int(reset_mask_steps)
    label = (stage == SPATIAL_LOW).to(dtype=torch.float32).unsqueeze(1)
    mask = (known & outside_reset).unsqueeze(1)
    weight = torch.ones_like(label)
    weight = torch.where(
        (stage == SPATIAL_HIGH_END).unsqueeze(1),
        torch.full_like(weight, float(high_end_weight)),
        weight,
    )
    return StageAuxiliaryTargets(label=label, mask=mask, weight=weight)


def balanced_masked_stage_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> StageAuxiliaryLoss:
    """Balanced HIGH/LOW BCE which is exactly zero for an empty class.

    Each present class contributes half of the loss regardless of its rollout
    duration.  Optional positive sample weights operate *inside* each class;
    the FastBase configuration uses this to keep HIGH_END return frames from
    being diluted by the longer HIGH_START segment.  When a mini-batch contains
    only one class, that class receives full weight; an all-masked batch has a
    finite zero loss with a valid zero gradient.
    """

    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [N,1], got {tuple(logits.shape)}")
    if labels.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError(
            "labels and mask must match logits [N,1], got "
            f"{tuple(labels.shape)} and {tuple(mask.shape)}"
        )
    if sample_weight is None:
        sample_weight = torch.ones_like(logits)
    elif sample_weight.ndim == 1:
        sample_weight = sample_weight.unsqueeze(1)
    if sample_weight.shape != logits.shape:
        raise ValueError(
            f"sample_weight must match logits [N,1], got {tuple(sample_weight.shape)}"
        )

    raw_labels = labels.to(device=logits.device, dtype=logits.dtype)
    valid = mask.to(device=logits.device, dtype=torch.bool)
    raw_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)
    finite = torch.isfinite(logits) & torch.isfinite(raw_labels) & torch.isfinite(raw_weight)
    valid &= finite
    safe_weight = torch.nan_to_num(raw_weight, nan=0.0, posinf=0.0, neginf=0.0)
    if bool((safe_weight < 0.0).any().item()):
        raise ValueError("sample_weight must be non-negative")
    safe_labels = torch.nan_to_num(raw_labels, nan=0.0, posinf=1.0, neginf=0.0).clamp(
        0.0, 1.0
    )
    high = valid & (safe_labels < 0.5)
    low = valid & ~high
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0),
        safe_labels,
        reduction="none",
    )

    def _masked_mean(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = selected.to(dtype=logits.dtype).sum()
        weighted = selected.to(dtype=logits.dtype) * safe_weight
        value = (per_row * weighted).sum() / weighted.sum().clamp_min(1.0)
        return value, count

    high_loss, high_count = _masked_mean(high)
    low_loss, low_count = _masked_mean(low)
    present_classes = (high_count > 0).to(logits.dtype) + (low_count > 0).to(logits.dtype)
    total = (high_loss + low_loss) / present_classes.clamp_min(1.0)

    valid_count = valid.to(dtype=logits.dtype).sum()
    predicted_low = logits >= 0.0
    high_accuracy = (
        ((~predicted_low) & high).to(dtype=logits.dtype).sum()
        / high_count.clamp_min(1.0)
    )
    low_accuracy = (
        (predicted_low & low).to(dtype=logits.dtype).sum()
        / low_count.clamp_min(1.0)
    )
    accuracy = (high_accuracy + low_accuracy) / present_classes.clamp_min(1.0)
    valid_fraction = valid.to(dtype=logits.dtype).mean()
    low_fraction = low_count / valid_count.clamp_min(1.0)
    return StageAuxiliaryLoss(
        total=total,
        high=high_loss.detach(),
        low=low_loss.detach(),
        accuracy=accuracy.detach(),
        valid_fraction=valid_fraction.detach(),
        low_fraction=low_fraction.detach(),
    )


def actor_shared_trunk_latent(actor: MLPModel, observations: TensorDict) -> torch.Tensor:
    """Evaluate the deployable actor MLP up to (but not including) its action head."""

    layers = list(actor.mlp.children())
    if len(layers) < 2 or not isinstance(layers[-1], nn.Linear):
        raise TypeError(
            "stage auxiliary head requires the audited RSL MLP layout ending in nn.Linear"
        )
    latent = actor.get_latent(observations)
    for layer in layers[:-1]:
        latent = layer(latent)
    if latent.ndim != 2 or latent.shape[1] != layers[-1].in_features:
        raise RuntimeError(
            "actor shared latent has unexpected shape "
            f"{tuple(latent.shape)}; expected [N,{layers[-1].in_features}]"
        )
    return latent


def actor_shared_latent_dim(actor: MLPModel) -> int:
    """Return the feature dimension immediately before the actor action head."""

    layers = list(actor.mlp.children())
    if len(layers) < 2 or not isinstance(layers[-1], nn.Linear):
        raise TypeError(
            "stage auxiliary head requires the audited RSL MLP layout ending in nn.Linear"
        )
    return int(layers[-1].in_features)


class StageAuxiliaryHead(nn.Module):
    """Training-only LOW-vs-HIGH classifier attached to the actor shared latent."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.zeros_(self.net[0].bias)
        nn.init.orthogonal_(self.net[2].weight, gain=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def stage_auxiliary_logits(
    actor: MLPModel,
    observations: TensorDict,
    fallback_head: StageAuxiliaryHead | None,
) -> tuple[torch.Tensor, str]:
    """Return logits from an uncalibrated native gate or training-only head.

    Calibration is a deployment authority transform.  Keeping the LOW/HIGH
    BCE on the raw gate preserves its probabilistic learning target and makes
    post-training monotone calibration independent of optimization.
    """

    capture = getattr(actor, "raw_capture_probability", None)
    if callable(capture):
        probability = capture(observations)
        expected_batch = int(observations.batch_size[0])
        if probability.shape != (expected_batch, 1):
            raise RuntimeError(
                "actor.raw_capture_probability must return [N,1], got "
                f"{tuple(probability.shape)}"
            )
        probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        return (
            torch.log(probability) - torch.log1p(-probability),
            "actor_raw_capture_gate",
        )
    # Backward-compatible native actor protocol predating explicit raw versus
    # calibrated diagnostics.  New FastBase actors always take the branch
    # above; this fallback does not change old custom actor behavior.
    capture = getattr(actor, "capture_probability", None)
    if callable(capture):
        probability = capture(observations)
        expected_batch = int(observations.batch_size[0])
        if probability.shape != (expected_batch, 1):
            raise RuntimeError(
                "actor.capture_probability must return [N,1], got "
                f"{tuple(probability.shape)}"
            )
        probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        return torch.log(probability) - torch.log1p(-probability), "actor_capture_gate"
    if fallback_head is None:
        raise RuntimeError("stock MLP actor requires a fallback stage auxiliary head")
    return (
        fallback_head(actor_shared_trunk_latent(actor, observations)),
        "training_only_shared_latent_head",
    )


def bounded_teacher_targets(
    teacher_action: torch.Tensor,
    student_mean: torch.Tensor,
    *,
    action_clamp: float,
    delta_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sanitize and cap cached Teacher targets.

    Non-finite rows are rejected before replacement.  For valid rows the raw
    Teacher action is clamped to ``[-action_clamp, action_clamp]`` and its
    per-joint delta from the rollout Actor mean is capped to ``delta_cap``.
    The returned validity mask has shape ``[N]``.
    """

    if teacher_action.shape != student_mean.shape or teacher_action.ndim != 2:
        raise ValueError(
            "teacher_action and student_mean must share shape [N,A], got "
            f"{tuple(teacher_action.shape)} and {tuple(student_mean.shape)}"
        )
    if action_clamp <= 0.0:
        raise ValueError("action_clamp must be positive")
    if delta_cap <= 0.0:
        raise ValueError("delta_cap must be positive")

    finite = torch.isfinite(teacher_action).all(dim=1) & torch.isfinite(student_mean).all(dim=1)
    safe_student = torch.nan_to_num(
        student_mean.detach(), nan=0.0, posinf=action_clamp, neginf=-action_clamp
    ).clamp(-action_clamp, action_clamp)
    safe_teacher = torch.nan_to_num(
        teacher_action.detach(), nan=0.0, posinf=action_clamp, neginf=-action_clamp
    ).clamp(-action_clamp, action_clamp)
    delta = (safe_teacher - safe_student).clamp(-delta_cap, delta_cap)
    target = (safe_student + delta).clamp(-action_clamp, action_clamp)
    # Rejected rows retain a finite no-op target even though their loss mask is
    # zero.  This prevents NaNs from entering checkpoints or diagnostics.
    target = torch.where(finite.unsqueeze(1), target, safe_student)
    return target, finite


def masked_anchor_mse(
    actor_mean: torch.Tensor,
    teacher_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-action squared error over active HIGH rows only."""

    if actor_mean.shape != teacher_target.shape or actor_mean.ndim != 2:
        raise ValueError("actor_mean and teacher_target must share shape [N,A]")
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    if mask.shape != (actor_mean.shape[0], 1):
        raise ValueError(f"mask must have shape {(actor_mean.shape[0], 1)}, got {tuple(mask.shape)}")
    weight = mask.to(device=actor_mean.device, dtype=actor_mean.dtype)
    per_row = (actor_mean - teacher_target.detach()).square().mean(dim=1, keepdim=True)
    active = weight.sum()
    # ``clamp_min`` yields exactly zero (with a valid zero gradient) when no
    # HIGH samples appear in a minibatch.
    return (per_row * weight).sum() / active.clamp_min(1.0)


def _base_env(env: VecEnv) -> Any:
    """Resolve Isaac Lab's base env through project and RSL wrappers."""

    candidate = env
    seen: set[int] = set()
    for _ in range(8):
        if id(candidate) in seen:
            break
        seen.add(id(candidate))
        try:
            unwrapped = getattr(candidate, "unwrapped")
        except (AttributeError, RuntimeError):
            unwrapped = None
        if unwrapped is not None and unwrapped is not candidate:
            candidate = unwrapped
            continue
        nested = getattr(candidate, "_base", None)
        if nested is not None and nested is not candidate:
            candidate = nested
            continue
        nested = getattr(candidate, "env", None)
        if nested is not None and nested is not candidate:
            candidate = nested
            continue
        break
    return candidate


def training_anchor_context(
    env: VecEnv,
    *,
    num_envs: int,
    device: str | torch.device,
    sensor_age_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read private stage and Hall-age state without adding observation keys."""

    if sensor_age_scale <= 0.0:
        raise ValueError("sensor_age_scale must be positive")
    base = _base_env(env)
    stage = getattr(base, "spatial_course_stage_buf", None)
    if not torch.is_tensor(stage) or stage.shape != (num_envs,):
        raise RuntimeError(
            "AnchoredPPO requires private spatial_course_stage_buf [num_envs]; "
            "it is intentionally not synthesized or read from Actor observations"
        )
    packet = getattr(base, "_hall_foot_packet_cache", None)
    age = packet.get("age") if isinstance(packet, dict) else None
    if not torch.is_tensor(age) or age.shape != (num_envs, 2):
        raise RuntimeError(
            "AnchoredPPO requires the current private Hall packet age [num_envs,2]. "
            "The policy observation must be computed before act()."
        )
    return (
        stage.detach().to(device=device, dtype=torch.long),
        torch.nan_to_num(age.detach().to(device=device, dtype=torch.float32) / sensor_age_scale)
        .clamp(0.0, 1.0),
    )


def training_stage_auxiliary_context(
    env: VecEnv,
    *,
    stage: torch.Tensor,
    num_envs: int,
    device: str | torch.device,
    reset_mask_steps: int,
    high_end_weight: float,
) -> StageAuxiliaryTargets:
    """Read only episode age and form private stage targets outside observations."""

    base = _base_env(env)
    episode_length = getattr(base, "episode_length_buf", None)
    if not torch.is_tensor(episode_length) or episode_length.shape != (num_envs,):
        raise RuntimeError(
            "stage auxiliary loss requires private episode_length_buf [num_envs] "
            "to mask reset observations"
        )
    return stage_auxiliary_targets(
        stage.detach().to(device=device, dtype=torch.long),
        episode_length.detach().to(device=device, dtype=torch.long),
        reset_mask_steps=reset_mask_steps,
        high_end_weight=high_end_weight,
    )


def validate_actor_observation_contract(
    actor: MLPModel,
    *,
    policy_group: str = "policy",
    expected_dim: int = INPUT_DIM,
) -> None:
    """Reject privileged groups or schema drift in the deployable Actor."""

    groups = list(getattr(actor, "obs_groups", []))
    if groups != [policy_group]:
        raise ValueError(
            "Actor truth-leakage guard requires exactly one deployable observation group "
            f"[{policy_group!r}], got {groups!r}"
        )
    dimension = int(getattr(actor, "obs_dim", -1))
    if dimension != expected_dim:
        raise ValueError(
            f"Actor observation dimension must remain {expected_dim}, got {dimension}"
        )


def _native_capture_residual(actor: MLPModel) -> nn.Sequential | None:
    """Return the FastBase residual branch without depending on its class.

    Stock anchored actors deliberately return ``None``.  The structural check
    keeps the algorithm usable across checkpoint imports while still failing
    closed when gate-only warm-up is requested for an incompatible actor.
    """

    mean_module = getattr(actor, "mlp", None)
    residual = getattr(mean_module, "residual", None)
    return residual if isinstance(residual, nn.Sequential) else None


def capture_residual_has_zero_output(residual: nn.Sequential) -> bool:
    """Prove that a residual branch currently emits exact zero for finite input.

    The hidden feature extractor remains initialized for later training; an
    all-zero final affine layer is sufficient to make ``tanh(residual(.))``
    identically zero during gate-only warm-up.
    """

    if len(residual) == 0 or not isinstance(residual[-1], nn.Linear):
        raise TypeError("capture residual must end in nn.Linear")
    output = residual[-1]
    return bool(
        torch.count_nonzero(output.weight.detach()).item() == 0
        and (
            output.bias is None
            or torch.count_nonzero(output.bias.detach()).item() == 0
        )
    )


@dataclass(frozen=True)
class AnchorBatch:
    """One rollout step of cached supervision."""

    target: torch.Tensor
    mask: torch.Tensor
    finite: torch.Tensor


@dataclass(frozen=True)
class LowExpertResidualBatch:
    """Cached counterfactual residual target for one rollout step."""

    target: torch.Tensor
    mask: torch.Tensor
    finite: torch.Tensor


@dataclass(frozen=True)
class LowExpertDistillationLoss:
    """Masked SmoothL1 loss and detached rollout diagnostics."""

    total: torch.Tensor
    valid_fraction: torch.Tensor
    mean_abs_target: torch.Tensor


def masked_low_expert_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> LowExpertDistillationLoss:
    """SmoothL1 over selected rows, normalized across rows and 29 actions."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "LOW expert prediction/target must share [N,A], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if mask.shape != (prediction.shape[0], 1) or mask.dtype != torch.bool:
        raise ValueError("LOW expert mask must be bool [N,1]")
    if beta <= 0.0:
        raise ValueError("LOW expert SmoothL1 beta must be positive")
    finite_rows = (
        torch.isfinite(prediction).all(dim=1, keepdim=True)
        & torch.isfinite(target).all(dim=1, keepdim=True)
    )
    selected = mask & finite_rows
    safe_prediction = torch.nan_to_num(prediction, nan=0.0, posinf=3.0, neginf=-3.0)
    safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.2, neginf=-0.2)
    element = torch.nn.functional.smooth_l1_loss(
        safe_prediction, safe_target, reduction="none", beta=float(beta)
    )
    weights = selected.to(dtype=element.dtype)
    denominator = (weights.sum() * prediction.shape[1]).clamp_min(1.0)
    total = (element * weights).sum() / denominator
    target_denominator = denominator.detach()
    mean_abs_target = (
        safe_target.detach().abs() * weights.detach()
    ).sum() / target_denominator
    return LowExpertDistillationLoss(
        total=total,
        valid_fraction=selected.float().mean().detach(),
        mean_abs_target=mean_abs_target.detach(),
    )


class FrozenLowExpertResidualTargetBuilder:
    """Build LOW-only ``expert(cmd=.16) - frozen_base(original_cmd)`` targets."""

    def __init__(
        self,
        expert: nn.Module,
        *,
        frozen_base_action: Any,
        command: tuple[float, float, float] = LOW_EXPERT_COMMAND,
        target_cap: float = 0.20,
        valid_threshold: float = 0.5,
    ) -> None:
        if not callable(frozen_base_action):
            raise TypeError("frozen_base_action must be callable")
        if target_cap <= 0.0:
            raise ValueError("LOW expert target_cap must be positive")
        if not 0.0 <= valid_threshold < 1.0:
            raise ValueError("LOW expert valid_threshold must be in [0,1)")
        self.expert = expert.eval()
        for parameter in self.expert.parameters():
            parameter.requires_grad_(False)
        self.frozen_base_action = frozen_base_action
        self.command = tuple(float(value) for value in command)
        self.target_cap = float(target_cap)
        self.valid_threshold = float(valid_threshold)
        self.inference_calls = 0
        self.cache_writes = 0
        self.rows_inferred = 0

    def build(
        self,
        policy_observation: torch.Tensor,
        *,
        stage: torch.Tensor,
    ) -> LowExpertResidualBatch:
        """Invoke the expert at most once, and only on eligible LOW rows."""

        if policy_observation.ndim != 2 or policy_observation.shape[1] != INPUT_DIM:
            raise ValueError(
                f"policy observation must be [N,{INPUT_DIM}], got "
                f"{tuple(policy_observation.shape)}"
            )
        if stage.shape != (policy_observation.shape[0],):
            raise ValueError("LOW expert stage must be [N]")
        detached = policy_observation.detach()
        input_finite = torch.isfinite(detached).all(dim=1)
        both_feet_valid = (
            detached[:, VALID_SLICE].amin(dim=1) > self.valid_threshold
        )
        eligible = (
            stage.detach().to(device=detached.device, dtype=torch.long) == SPATIAL_LOW
        ) & input_finite & both_feet_valid
        target = torch.zeros(
            detached.shape[0], OUTPUT_DIM, device=detached.device, dtype=detached.dtype
        )
        finite = input_finite.clone()
        mask = torch.zeros(
            detached.shape[0], 1, device=detached.device, dtype=torch.bool
        )
        if bool(eligible.any().item()):
            safe_original = torch.nan_to_num(
                detached[eligible], nan=0.0, posinf=6.0, neginf=-6.0
            )
            expert_observation = rewrite_term_major_velocity_command(
                safe_original, self.command
            )
            with torch.inference_mode():
                expert_action = self.expert(expert_observation)
                # Deployment keeps the original rollout command in the frozen
                # speedboost base.  Only the recovery expert sees cmd=.16.
                frozen_action = self.frozen_base_action(safe_original)
            self.inference_calls += 1
            self.rows_inferred += int(safe_original.shape[0])
            expected_shape = (safe_original.shape[0], OUTPUT_DIM)
            if expert_action.shape != expected_shape or frozen_action.shape != expected_shape:
                raise RuntimeError(
                    "LOW expert/base must return matching [eligible,29] actions"
                )
            output_finite = torch.isfinite(expert_action).all(dim=1) & torch.isfinite(
                frozen_action
            ).all(dim=1)
            difference = torch.nan_to_num(
                expert_action - frozen_action,
                nan=0.0,
                posinf=self.target_cap,
                neginf=-self.target_cap,
            ).clamp(-self.target_cap, self.target_cap)
            eligible_ids = torch.nonzero(eligible, as_tuple=False).squeeze(1)
            target[eligible_ids] = difference
            finite[eligible_ids] &= output_finite
            mask[eligible_ids, 0] = output_finite
        self.cache_writes += 1
        return LowExpertResidualBatch(
            target=target.detach(), mask=mask.detach(), finite=finite.detach()
        )


class FrozenTeacherAnchorTargetBuilder:
    """Single-call frozen Teacher adapter used by :class:`AnchoredPPO`."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        action_clamp: float = 3.0,
        delta_cap: float = 0.25,
    ) -> None:
        if action_clamp <= 0.0 or delta_cap <= 0.0:
            raise ValueError("action_clamp and delta_cap must be positive")
        self.teacher = teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.action_clamp = float(action_clamp)
        self.delta_cap = float(delta_cap)
        self.inference_calls = 0
        self.cache_writes = 0

    def build(
        self,
        policy_observation: torch.Tensor,
        *,
        sensor_age_lr: torch.Tensor,
        stage: torch.Tensor,
        student_mean: torch.Tensor,
    ) -> AnchorBatch:
        """Run the Teacher exactly once and return a detached cache entry."""

        if policy_observation.shape != (student_mean.shape[0], INPUT_DIM):
            raise ValueError(
                f"policy observation must be [N,{INPUT_DIM}], got {tuple(policy_observation.shape)}"
            )
        input_finite = torch.isfinite(policy_observation).all(dim=1)
        safe_policy = torch.nan_to_num(policy_observation.detach(), nan=0.0, posinf=6.0, neginf=-6.0)
        teacher_observation = adapt_teacher_observation(
            safe_policy,
            policy_trailing_feature_mode="motion_feedback",
            sensor_age_lr=sensor_age_lr,
        )
        with torch.inference_mode():
            teacher_action = self.teacher(teacher_observation)
        self.inference_calls += 1
        if teacher_action.shape != (student_mean.shape[0], OUTPUT_DIM):
            raise RuntimeError(
                f"Teacher must return [N,{OUTPUT_DIM}], got {tuple(teacher_action.shape)}"
            )
        target, output_finite = bounded_teacher_targets(
            teacher_action,
            student_mean,
            action_clamp=self.action_clamp,
            delta_cap=self.delta_cap,
        )
        finite = input_finite & output_finite
        mask = high_friction_anchor_mask(stage, finite_mask=finite)
        self.cache_writes += 1
        return AnchorBatch(target=target.detach(), mask=mask.detach(), finite=finite.detach())


class AnchoredRolloutStorage(RolloutStorage):
    """RSL-RL 5 feed-forward storage plus cached Teacher targets/masks."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device,
        )
        if training_type != "rl":
            raise ValueError("AnchoredRolloutStorage supports RL rollouts only")
        self.anchor_targets = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.anchor_masks = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.bool
        )
        self.stage_aux_labels = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.float32
        )
        self.stage_aux_masks = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.bool
        )
        self.stage_aux_weights = torch.ones(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.float32
        )
        self.low_expert_residual_targets = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.low_expert_residual_masks = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.bool
        )

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        target = getattr(transition, "anchor_target", None)
        mask = getattr(transition, "anchor_mask", None)
        stage_label = getattr(transition, "stage_aux_label", None)
        stage_mask = getattr(transition, "stage_aux_mask", None)
        stage_weight = getattr(transition, "stage_aux_weight", None)
        low_expert_target = getattr(
            transition, "low_expert_residual_target", None
        )
        low_expert_mask = getattr(transition, "low_expert_residual_mask", None)
        if not torch.is_tensor(target) or target.shape != self.anchor_targets[self.step].shape:
            raise RuntimeError("transition is missing a correctly shaped cached anchor_target")
        if not torch.is_tensor(mask) or mask.shape != self.anchor_masks[self.step].shape:
            raise RuntimeError("transition is missing a correctly shaped cached anchor_mask")
        if (
            not torch.is_tensor(stage_label)
            or stage_label.shape != self.stage_aux_labels[self.step].shape
        ):
            raise RuntimeError(
                "transition is missing a correctly shaped cached stage_aux_label"
            )
        if (
            not torch.is_tensor(stage_mask)
            or stage_mask.shape != self.stage_aux_masks[self.step].shape
        ):
            raise RuntimeError(
                "transition is missing a correctly shaped cached stage_aux_mask"
            )
        if (
            not torch.is_tensor(stage_weight)
            or stage_weight.shape != self.stage_aux_weights[self.step].shape
        ):
            raise RuntimeError(
                "transition is missing a correctly shaped cached stage_aux_weight"
            )
        self.anchor_targets[self.step].copy_(target)
        self.anchor_masks[self.step].copy_(mask.to(dtype=torch.bool))
        self.stage_aux_labels[self.step].copy_(stage_label.to(dtype=torch.float32))
        self.stage_aux_masks[self.step].copy_(stage_mask.to(dtype=torch.bool))
        self.stage_aux_weights[self.step].copy_(stage_weight.to(dtype=torch.float32))
        if low_expert_target is None and low_expert_mask is None:
            # Backward-compatible direct storage users and disabled configs.
            self.low_expert_residual_targets[self.step].zero_()
            self.low_expert_residual_masks[self.step].zero_()
        else:
            if (
                not torch.is_tensor(low_expert_target)
                or low_expert_target.shape
                != self.low_expert_residual_targets[self.step].shape
            ):
                raise RuntimeError(
                    "transition is missing a correctly shaped cached "
                    "low_expert_residual_target"
                )
            if (
                not torch.is_tensor(low_expert_mask)
                or low_expert_mask.shape
                != self.low_expert_residual_masks[self.step].shape
            ):
                raise RuntimeError(
                    "transition is missing a correctly shaped cached "
                    "low_expert_residual_mask"
                )
            self.low_expert_residual_targets[self.step].copy_(low_expert_target)
            self.low_expert_residual_masks[self.step].copy_(
                low_expert_mask.to(dtype=torch.bool)
            )
        super().add_transition(transition)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Yield aligned PPO and anchor batches using one shared permutation."""

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size <= 0:
            raise ValueError("num_mini_batches exceeds rollout batch size")
        usable = num_mini_batches * mini_batch_size
        indices = torch.randperm(usable, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        anchor_targets = self.anchor_targets.flatten(0, 1)
        anchor_masks = self.anchor_masks.flatten(0, 1)
        stage_aux_labels = self.stage_aux_labels.flatten(0, 1)
        stage_aux_masks = self.stage_aux_masks.flatten(0, 1)
        stage_aux_weights = self.stage_aux_weights.flatten(0, 1)
        low_expert_residual_targets = self.low_expert_residual_targets.flatten(0, 1)
        low_expert_residual_masks = self.low_expert_residual_masks.flatten(0, 1)

        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                stop = (index + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
                batch.anchor_targets = anchor_targets[batch_idx]
                batch.anchor_masks = anchor_masks[batch_idx]
                batch.stage_aux_labels = stage_aux_labels[batch_idx]
                batch.stage_aux_masks = stage_aux_masks[batch_idx]
                batch.stage_aux_weights = stage_aux_weights[batch_idx]
                batch.low_expert_residual_targets = low_expert_residual_targets[
                    batch_idx
                ]
                batch.low_expert_residual_masks = low_expert_residual_masks[
                    batch_idx
                ]
                yield batch


class AnchoredPPO(PPO):
    """PPO with a privileged HIGH-only frozen-Teacher auxiliary loss."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: AnchoredRolloutStorage,
        *,
        env: VecEnv,
        anchor_teacher_checkpoint: str,
        anchor_loss_coef: float = 1.0,
        anchor_delta_cap: float = 0.25,
        anchor_teacher_action_clamp: float = 3.0,
        anchor_policy_observation_group: str = "policy",
        anchor_sensor_age_scale: float = 0.25,
        anchor_expected_teacher_source_sha256: str = KNOWN_SPEEDBOOST112_SHA256,
        anchor_min_learning_rate: float = 1.0e-7,
        anchor_max_learning_rate: float = 1.0e-5,
        stage_aux_loss_coef: float = 0.10,
        stage_aux_hidden_dim: int = 64,
        stage_aux_reset_mask_steps: int = 1,
        stage_aux_high_end_weight: float = 1.0,
        capture_gate_warmup_updates: int = 0,
        capture_gate_warmup_learning_rate: float = 1.0e-4,
        capture_gate_learning_rate: float = 1.0e-5,
        capture_gate_max_grad_norm: float = 1.0,
        capture_gate_gradient_mode: str = "joint",
        capture_residual_learning_rate: float = 5.0e-5,
        capture_residual_max_grad_norm: float = 0.5,
        stability_residual_learning_rate: float = 2.0e-5,
        stability_residual_max_grad_norm: float = 0.20,
        low_expert_checkpoint: str = "",
        low_expert_expected_sha256: str = "",
        low_expert_distillation_loss_coef: float = 0.0,
        low_expert_target_cap: float = 0.20,
        low_expert_smooth_l1_beta: float = 0.05,
        low_expert_command: tuple[float, float, float] = LOW_EXPERT_COMMAND,
        low_expert_residual_gradient_mode: str = "joint",
        **ppo_kwargs: Any,
    ) -> None:
        _require_rsl_rl_v5()
        if anchor_loss_coef < 0.0:
            raise ValueError("anchor_loss_coef must be non-negative")
        if anchor_delta_cap <= 0.0:
            raise ValueError("anchor_delta_cap must be positive")
        if anchor_teacher_action_clamp <= 0.0:
            raise ValueError("anchor_teacher_action_clamp must be positive")
        if anchor_sensor_age_scale <= 0.0:
            raise ValueError("anchor_sensor_age_scale must be positive")
        if not 0.0 < anchor_min_learning_rate <= anchor_max_learning_rate:
            raise ValueError(
                "anchor learning-rate bounds must satisfy 0 < min <= max"
            )
        if stage_aux_loss_coef < 0.0:
            raise ValueError("stage_aux_loss_coef must be non-negative")
        if stage_aux_hidden_dim <= 0:
            raise ValueError("stage_aux_hidden_dim must be positive")
        if stage_aux_reset_mask_steps < 0:
            raise ValueError("stage_aux_reset_mask_steps must be non-negative")
        if stage_aux_high_end_weight < 1.0:
            raise ValueError("stage_aux_high_end_weight must be at least 1.0")
        if capture_gate_warmup_updates < 0:
            raise ValueError("capture_gate_warmup_updates must be non-negative")
        if capture_gate_warmup_learning_rate <= 0.0:
            raise ValueError("capture_gate_warmup_learning_rate must be positive")
        if capture_gate_learning_rate <= 0.0:
            raise ValueError("capture_gate_learning_rate must be positive")
        if capture_gate_max_grad_norm <= 0.0:
            raise ValueError("capture_gate_max_grad_norm must be positive")
        if capture_gate_gradient_mode not in ("joint", "stage_bce_only"):
            raise ValueError(
                "capture_gate_gradient_mode must be 'joint' or 'stage_bce_only'"
            )
        if capture_residual_learning_rate <= 0.0:
            raise ValueError("capture_residual_learning_rate must be positive")
        if capture_residual_max_grad_norm <= 0.0:
            raise ValueError("capture_residual_max_grad_norm must be positive")
        if stability_residual_learning_rate <= 0.0:
            raise ValueError("stability_residual_learning_rate must be positive")
        if stability_residual_max_grad_norm <= 0.0:
            raise ValueError("stability_residual_max_grad_norm must be positive")
        if low_expert_distillation_loss_coef < 0.0:
            raise ValueError("low_expert_distillation_loss_coef must be non-negative")
        if low_expert_target_cap <= 0.0:
            raise ValueError("low_expert_target_cap must be positive")
        if low_expert_smooth_l1_beta <= 0.0:
            raise ValueError("low_expert_smooth_l1_beta must be positive")
        if len(low_expert_command) != 3 or any(
            not math.isfinite(float(value)) for value in low_expert_command
        ):
            raise ValueError("low_expert_command must contain finite vx/vy/yaw")
        if low_expert_residual_gradient_mode not in ("joint", "supervised_only"):
            raise ValueError(
                "low_expert_residual_gradient_mode must be 'joint' or 'supervised_only'"
            )
        if (
            low_expert_residual_gradient_mode == "supervised_only"
            and low_expert_distillation_loss_coef <= 0.0
        ):
            raise ValueError(
                "supervised_only residual gradients require positive LOW expert supervision"
            )
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("AnchoredPPO currently supports feed-forward actor/critic only")

        self.anchor_env = env
        self.anchor_teacher_checkpoint = str(Path(anchor_teacher_checkpoint).expanduser().resolve())
        self.anchor_loss_coef = float(anchor_loss_coef)
        self.anchor_delta_cap = float(anchor_delta_cap)
        self.anchor_teacher_action_clamp = float(anchor_teacher_action_clamp)
        self.anchor_policy_observation_group = str(anchor_policy_observation_group)
        self.anchor_sensor_age_scale = float(anchor_sensor_age_scale)
        self.anchor_expected_teacher_source_sha256 = str(anchor_expected_teacher_source_sha256)
        self.anchor_min_learning_rate = float(anchor_min_learning_rate)
        self.anchor_max_learning_rate = float(anchor_max_learning_rate)
        self.stage_aux_loss_coef = float(stage_aux_loss_coef)
        self.stage_aux_hidden_dim = int(stage_aux_hidden_dim)
        self.stage_aux_reset_mask_steps = int(stage_aux_reset_mask_steps)
        self.stage_aux_high_end_weight = float(stage_aux_high_end_weight)
        self.capture_gate_warmup_updates = int(capture_gate_warmup_updates)
        self.capture_gate_updates_completed = 0
        self.capture_gate_warmup_learning_rate = float(
            capture_gate_warmup_learning_rate
        )
        self.capture_gate_learning_rate = float(capture_gate_learning_rate)
        self.capture_gate_max_grad_norm = float(capture_gate_max_grad_norm)
        self.capture_gate_gradient_mode = str(capture_gate_gradient_mode)
        self.capture_residual_learning_rate = float(
            capture_residual_learning_rate
        )
        self.capture_residual_max_grad_norm = float(
            capture_residual_max_grad_norm
        )
        self.stability_residual_learning_rate = float(
            stability_residual_learning_rate
        )
        self.stability_residual_max_grad_norm = float(
            stability_residual_max_grad_norm
        )
        self.low_expert_distillation_loss_coef = float(
            low_expert_distillation_loss_coef
        )
        self.low_expert_target_cap = float(low_expert_target_cap)
        self.low_expert_smooth_l1_beta = float(low_expert_smooth_l1_beta)
        self.low_expert_command = tuple(float(value) for value in low_expert_command)
        self.low_expert_residual_gradient_mode = str(
            low_expert_residual_gradient_mode
        )
        self.low_expert_checkpoint = ""
        self.low_expert_checkpoint_sha256 = ""
        self.low_expert_expected_sha256 = str(low_expert_expected_sha256).lower()
        if self.low_expert_distillation_loss_coef > 0.0:
            if not low_expert_checkpoint:
                raise ValueError(
                    "positive LOW expert distillation coef requires a checkpoint"
                )
            self.low_expert_checkpoint = str(
                Path(low_expert_checkpoint).expanduser().resolve()
            )
            self.low_expert_checkpoint_sha256 = _sha256_file(
                self.low_expert_checkpoint
            )
        teacher_payload = torch.load(
            self.anchor_teacher_checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(teacher_payload, dict):
            raise TypeError("frozen Teacher artifact must contain an audit dictionary")
        self.anchor_teacher_source_graph = dict(
            teacher_payload.get("source_graph", {})
        )
        self.anchor_teacher_parity = dict(teacher_payload.get("parity", {}))
        self.anchor_teacher_provenance = dict(
            teacher_payload.get("provenance", {})
        )
        validate_actor_observation_contract(
            actor,
            policy_group=self.anchor_policy_observation_group,
            expected_dim=INPUT_DIM,
        )

        symmetry_cfg = ppo_kwargs.get("symmetry_cfg")
        if symmetry_cfg and (
            symmetry_cfg.get("use_data_augmentation", False)
            or symmetry_cfg.get("use_mirror_loss", False)
        ):
            raise ValueError(
                "AnchoredPPO rejects symmetry augmentation because cached Teacher targets "
                "do not have an audited action transform"
            )

        teacher = load_frozen_speedboost_teacher(
            self.anchor_teacher_checkpoint,
            device=ppo_kwargs.get("device", "cpu"),
            expected_source_sha256=self.anchor_expected_teacher_source_sha256,
        )
        self.anchor_builder = FrozenTeacherAnchorTargetBuilder(
            teacher,
            action_clamp=self.anchor_teacher_action_clamp,
            delta_cap=self.anchor_delta_cap,
        )
        self.anchor_invalid_rows_total = 0
        self.anchor_rows_total = 0
        super().__init__(actor, critic, storage, **ppo_kwargs)
        # A native FastBase Hall actor exposes both its learned raw gate and a
        # calibrated deployable probability.  Supervise only the raw gate;
        # calibration is a monotone deployment transform, never a privileged
        # target or actor input.  Older native actors retain their legacy
        # capture_probability fallback.
        if callable(getattr(self.actor, "raw_capture_probability", None)):
            self.stage_aux_source = "actor_raw_capture_gate"
        elif callable(getattr(self.actor, "capture_probability", None)):
            self.stage_aux_source = "actor_capture_gate"
        else:
            self.stage_aux_source = "training_only_shared_latent_head"
        self.stage_aux_uses_actor_capture_gate = self.stage_aux_source != (
            "training_only_shared_latent_head"
        )
        self.stage_aux_head: StageAuxiliaryHead | None = None
        if not self.stage_aux_uses_actor_capture_gate:
            self.stage_aux_head = StageAuxiliaryHead(
                actor_shared_latent_dim(self.actor), self.stage_aux_hidden_dim
            ).to(self.device)
            # This extra optimizer group is absent for native capture actors,
            # whose gate already belongs to actor.parameters().
            self.optimizer.add_param_group(
                {
                    "params": list(self.stage_aux_head.parameters()),
                    "lr": self.learning_rate,
                    OPTIMIZER_ROLE_KEY: STAGE_AUX_OPTIMIZER_ROLE,
                }
            )
        self.capture_residual = _native_capture_residual(self.actor)
        if self.capture_gate_warmup_updates > 0 and self.capture_residual is None:
            raise ValueError(
                "capture_gate_warmup_updates requires a native FastBase capture residual actor"
            )
        self.capture_gate_parameters = self._capture_gate_parameter_list()
        self.capture_residual_parameters = (
            list(self.capture_residual.parameters())
            if self.capture_residual is not None
            else []
        )
        mean_module = getattr(self.actor, "mlp", None)
        self.capture_branches_frozen = bool(
            getattr(mean_module, "freeze_capture_branches", False)
        )
        self.stability_residual = getattr(
            mean_module, "stability_residual", None
        )
        if self.stability_residual is not None and not isinstance(
            self.stability_residual, nn.Module
        ):
            raise TypeError("actor.mlp.stability_residual must be an nn.Module")
        self.stability_residual_parameters = (
            list(self.stability_residual.parameters())
            if isinstance(self.stability_residual, nn.Module)
            else []
        )
        if self.capture_gate_gradient_mode == "stage_bce_only":
            if self.stage_aux_source != "actor_raw_capture_gate":
                raise ValueError(
                    "stage_bce_only requires the native uncalibrated raw capture gate"
                )
            if not self.capture_gate_parameters or self.stage_aux_loss_coef <= 0.0:
                raise ValueError(
                    "stage_bce_only requires trainable gate parameters and positive BCE coef"
                )
        self._legacy_gate_only_ppo_parameters: list[nn.Parameter] = []
        self.low_expert_builder: FrozenLowExpertResidualTargetBuilder | None = None
        if self.low_expert_distillation_loss_coef > 0.0:
            mean_module = getattr(self.actor, "mlp", None)
            base_action = getattr(mean_module, "base_action", None)
            capture_features = getattr(mean_module, "capture_features", None)
            residual_limit = float(getattr(mean_module, "residual_limit", 0.0))
            if (
                self.capture_residual is None
                or not callable(base_action)
                or not callable(capture_features)
            ):
                raise ValueError(
                    "LOW expert distillation requires a native FastBase residual actor"
                )
            if abs(residual_limit - 0.55) > 1.0e-12:
                raise ValueError(
                    "LOW expert distillation is audited for residual_limit=0.55, "
                    f"got {residual_limit}"
                )
            expert = load_frozen_low_recovery_expert(
                self.low_expert_checkpoint,
                device=self.device,
                expected_sha256=(
                    self.low_expert_expected_sha256 or None
                ),
            )
            self.low_expert_builder = FrozenLowExpertResidualTargetBuilder(
                expert,
                frozen_base_action=base_action,
                command=self.low_expert_command,
                target_cap=self.low_expert_target_cap,
            )
        self._configure_optimizer_roles()
        self._apply_capture_gate_warmup_state()
        # ``super().__init__`` may put peer modules into a new state, so assert
        # the Teacher safety contract after construction as well.
        self.anchor_builder.teacher.eval()
        for parameter in self.anchor_builder.teacher.parameters():
            parameter.requires_grad_(False)

    @property
    def capture_gate_warmup_active(self) -> bool:
        return self.capture_gate_updates_completed < self.capture_gate_warmup_updates

    def _capture_gate_parameter_list(self) -> list[nn.Parameter]:
        if not self.stage_aux_uses_actor_capture_gate:
            return []
        mean_module = getattr(self.actor, "mlp", None)
        gate = getattr(mean_module, "gate", None)
        if not isinstance(gate, nn.Module):
            raise TypeError(
                "native capture_probability actor must expose its gate as actor.mlp.gate"
            )
        parameters = list(gate.parameters())
        if not parameters:
            raise ValueError("native capture gate has no trainable parameters")
        return parameters

    def _configure_optimizer_roles(self) -> None:
        """Split FastBase gate and residual into independent optimizer groups."""

        for group in self.optimizer.param_groups:
            group.setdefault(OPTIMIZER_ROLE_KEY, PPO_OPTIMIZER_ROLE)
        if (
            not self.capture_gate_parameters
            and not self.capture_residual_parameters
            and not self.stability_residual_parameters
        ):
            return

        gate_ids = {id(parameter) for parameter in self.capture_gate_parameters}
        residual_ids = {
            id(parameter) for parameter in self.capture_residual_parameters
        }
        stability_ids = {
            id(parameter) for parameter in self.stability_residual_parameters
        }
        if gate_ids & residual_ids or gate_ids & stability_ids or residual_ids & stability_ids:
            raise RuntimeError("private actor optimizer parameter sets overlap")
        # This reproduces the previous gate-only optimizer's PPO parameter
        # order and permits an exact, identity-aware optimizer migration from
        # model49-era checkpoints where residual parameters still lived in the
        # PPO group.
        self._legacy_gate_only_ppo_parameters = [
            parameter
            for group in self.optimizer.param_groups
            if group[OPTIMIZER_ROLE_KEY] == PPO_OPTIMIZER_ROLE
            for parameter in group["params"]
            if id(parameter) not in gate_ids
        ]
        removed_gate: list[nn.Parameter] = []
        removed_residual: list[nn.Parameter] = []
        removed_stability: list[nn.Parameter] = []
        for group in self.optimizer.param_groups:
            if group[OPTIMIZER_ROLE_KEY] in (
                CAPTURE_GATE_OPTIMIZER_ROLE,
                CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
                STABILITY_RESIDUAL_OPTIMIZER_ROLE,
            ):
                raise RuntimeError("private actor optimizer group already exists")
            retained = []
            for parameter in group["params"]:
                if id(parameter) in gate_ids:
                    removed_gate.append(parameter)
                elif id(parameter) in residual_ids:
                    removed_residual.append(parameter)
                elif id(parameter) in stability_ids:
                    removed_stability.append(parameter)
                else:
                    retained.append(parameter)
            group["params"] = retained
        if (
            {id(parameter) for parameter in removed_gate} != gate_ids
            or len(removed_gate) != len(gate_ids)
        ):
            raise RuntimeError(
                "failed to extract every capture gate parameter exactly once from PPO optimizer"
            )
        if (
            {id(parameter) for parameter in removed_residual} != residual_ids
            or len(removed_residual) != len(residual_ids)
        ):
            raise RuntimeError(
                "failed to extract every capture residual parameter exactly once "
                "from PPO optimizer"
            )
        if (
            {id(parameter) for parameter in removed_stability} != stability_ids
            or len(removed_stability) != len(stability_ids)
        ):
            raise RuntimeError(
                "failed to extract every stability residual parameter exactly once "
                "from PPO optimizer"
            )
        if self.capture_gate_parameters:
            self.optimizer.add_param_group(
                {
                    "params": self.capture_gate_parameters,
                    "lr": (
                        0.0
                        if self.capture_branches_frozen
                        else self.capture_gate_warmup_learning_rate
                    ),
                    OPTIMIZER_ROLE_KEY: CAPTURE_GATE_OPTIMIZER_ROLE,
                }
            )
        if self.capture_residual_parameters:
            self.optimizer.add_param_group(
                {
                    "params": self.capture_residual_parameters,
                    "lr": (
                        0.0
                        if (
                            self.capture_branches_frozen
                            or self.capture_gate_warmup_active
                        )
                        else self.capture_residual_learning_rate
                    ),
                    OPTIMIZER_ROLE_KEY: CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
                }
            )
        if self.stability_residual_parameters:
            self.optimizer.add_param_group(
                {
                    "params": self.stability_residual_parameters,
                    "lr": self.stability_residual_learning_rate,
                    OPTIMIZER_ROLE_KEY: STABILITY_RESIDUAL_OPTIMIZER_ROLE,
                }
            )
        self._validate_optimizer_roles()

    def _validate_optimizer_roles(self) -> None:
        roles = [group.get(OPTIMIZER_ROLE_KEY) for group in self.optimizer.param_groups]
        gate_group_count = roles.count(CAPTURE_GATE_OPTIMIZER_ROLE)
        expected_gate_groups = 1 if self.capture_gate_parameters else 0
        if gate_group_count != expected_gate_groups:
            raise RuntimeError(
                "optimizer must contain exactly one capture-gate group for FastBase actor"
            )
        residual_group_count = roles.count(CAPTURE_RESIDUAL_OPTIMIZER_ROLE)
        expected_residual_groups = 1 if self.capture_residual_parameters else 0
        if residual_group_count != expected_residual_groups:
            raise RuntimeError(
                "optimizer must contain exactly one capture-residual group for FastBase actor"
            )
        stability_group_count = roles.count(STABILITY_RESIDUAL_OPTIMIZER_ROLE)
        expected_stability_groups = 1 if self.stability_residual_parameters else 0
        if stability_group_count != expected_stability_groups:
            raise RuntimeError(
                "optimizer must contain exactly one stability-residual group when enabled"
            )
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                identifier = id(parameter)
                if identifier in seen:
                    raise RuntimeError("optimizer parameter appears in more than one group")
                seen.add(identifier)
        gate_ids = {id(parameter) for parameter in self.capture_gate_parameters}
        grouped_gate_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == CAPTURE_GATE_OPTIMIZER_ROLE
            for parameter in group["params"]
        }
        if grouped_gate_ids != gate_ids:
            raise RuntimeError("capture-gate optimizer group parameter identity drifted")
        residual_ids = {
            id(parameter) for parameter in self.capture_residual_parameters
        }
        grouped_residual_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == CAPTURE_RESIDUAL_OPTIMIZER_ROLE
            for parameter in group["params"]
        }
        if grouped_residual_ids != residual_ids:
            raise RuntimeError(
                "capture-residual optimizer group parameter identity drifted"
            )
        stability_ids = {
            id(parameter) for parameter in self.stability_residual_parameters
        }
        grouped_stability_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == STABILITY_RESIDUAL_OPTIMIZER_ROLE
            for parameter in group["params"]
        }
        if grouped_stability_ids != stability_ids:
            raise RuntimeError(
                "stability-residual optimizer group parameter identity drifted"
            )

    def _capture_gate_group(self) -> dict[str, Any] | None:
        groups = [
            group
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == CAPTURE_GATE_OPTIMIZER_ROLE
        ]
        if not groups:
            return None
        if len(groups) != 1:
            raise RuntimeError("capture gate optimizer role is ambiguous")
        return groups[0]

    def _capture_residual_group(self) -> dict[str, Any] | None:
        groups = [
            group
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == CAPTURE_RESIDUAL_OPTIMIZER_ROLE
        ]
        if not groups:
            return None
        if len(groups) != 1:
            raise RuntimeError("capture residual optimizer role is ambiguous")
        return groups[0]

    def _stability_residual_group(self) -> dict[str, Any] | None:
        groups = [
            group
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == STABILITY_RESIDUAL_OPTIMIZER_ROLE
        ]
        if not groups:
            return None
        if len(groups) != 1:
            raise RuntimeError("stability residual optimizer role is ambiguous")
        return groups[0]

    @property
    def capture_gate_current_learning_rate(self) -> float | None:
        group = self._capture_gate_group()
        return None if group is None else float(group["lr"])

    @property
    def capture_residual_current_learning_rate(self) -> float | None:
        group = self._capture_residual_group()
        return None if group is None else float(group["lr"])

    @property
    def stability_residual_current_learning_rate(self) -> float | None:
        group = self._stability_residual_group()
        return None if group is None else float(group["lr"])

    def _sync_optimizer_learning_rates(self) -> None:
        """Apply KL-adaptive LR only to PPO groups; keep capture LRs private."""

        for group in self.optimizer.param_groups:
            if group.get(OPTIMIZER_ROLE_KEY) not in (
                CAPTURE_GATE_OPTIMIZER_ROLE,
                CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
                STABILITY_RESIDUAL_OPTIMIZER_ROLE,
            ):
                group["lr"] = self.learning_rate
        gate_group = self._capture_gate_group()
        if gate_group is not None:
            gate_group["lr"] = (
                0.0
                if self.capture_branches_frozen
                else self.capture_gate_warmup_learning_rate
                if self.capture_gate_warmup_active
                else self.capture_gate_learning_rate
            )
        residual_group = self._capture_residual_group()
        if residual_group is not None:
            residual_group["lr"] = (
                0.0
                if (
                    self.capture_branches_frozen
                    or self.capture_gate_warmup_active
                )
                else self.capture_residual_learning_rate
            )
        stability_group = self._stability_residual_group()
        if stability_group is not None:
            stability_group["lr"] = self.stability_residual_learning_rate

    def _apply_capture_gate_warmup_state(self) -> None:
        """Freeze/unfreeze the residual branch from the checkpointed counter."""

        for parameter in self.capture_gate_parameters:
            parameter.requires_grad_(not self.capture_branches_frozen)
        if self.capture_residual is None:
            self._sync_optimizer_learning_rates()
            return
        active = self.capture_gate_warmup_active
        if active and not capture_residual_has_zero_output(self.capture_residual):
            raise RuntimeError(
                "gate-only warm-up requires an exact-zero residual output head; "
                "start from FastBase model_0 or set capture_gate_warmup_updates=0"
            )
        for parameter in self.capture_residual.parameters():
            parameter.requires_grad_(
                not active and not self.capture_branches_frozen
            )
        self._sync_optimizer_learning_rates()

    def _advance_capture_gate_warmup(self) -> None:
        """Advance once per completed PPO update and release residual training."""

        if self.capture_gate_warmup_updates == 0:
            return
        self.capture_gate_updates_completed += 1
        self._apply_capture_gate_warmup_state()

    def _clip_training_gradients(self) -> tuple[float, float, float]:
        """Clip PPO, gate and residual gradients under independent limits."""

        private_parameter_ids = {
            id(parameter)
            for parameter in (
                self.capture_gate_parameters
                + self.capture_residual_parameters
                + self.stability_residual_parameters
            )
        }
        actor_and_aux_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if id(parameter) not in private_parameter_ids
        ]
        if self.stage_aux_head is not None:
            actor_and_aux_parameters.extend(self.stage_aux_head.parameters())
        nn.utils.clip_grad_norm_(actor_and_aux_parameters, self.max_grad_norm)
        gate_grad_norm = 0.0
        if self.capture_gate_parameters:
            gate_grad_norm = float(
                nn.utils.clip_grad_norm_(
                    self.capture_gate_parameters,
                    self.capture_gate_max_grad_norm,
                ).item()
            )
        residual_grad_norm = 0.0
        if self.capture_residual_parameters:
            residual_grad_norm = float(
                nn.utils.clip_grad_norm_(
                    self.capture_residual_parameters,
                    self.capture_residual_max_grad_norm,
                ).item()
            )
        stability_grad_norm = 0.0
        if self.stability_residual_parameters:
            stability_grad_norm = float(
                nn.utils.clip_grad_norm_(
                    self.stability_residual_parameters,
                    self.stability_residual_max_grad_norm,
                ).item()
            )
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        return gate_grad_norm, residual_grad_norm, stability_grad_norm

    def _ungated_capture_residual(self, observations: TensorDict) -> torch.Tensor:
        """Return audited ``0.55*tanh(residual)`` without gate/confidence."""

        if self.capture_residual is None:
            raise RuntimeError("ungated residual requires a FastBase actor")
        policy_observation = observations[self.anchor_policy_observation_group]
        mean_module = getattr(self.actor, "mlp", None)
        capture_features = getattr(mean_module, "capture_features", None)
        if not callable(capture_features):
            raise TypeError("FastBase actor has no capture_features interface")
        features = capture_features(policy_observation)
        prediction = 0.55 * torch.tanh(self.capture_residual(features))
        if prediction.shape != (policy_observation.shape[0], OUTPUT_DIM):
            raise RuntimeError("ungated FastBase residual must return [N,29]")
        return prediction

    def _anchor_student_mean(
        self, policy_observation: torch.Tensor, composite_mean: torch.Tensor
    ) -> torch.Tensor:
        """Exclude the optional stability branch from frozen-teacher anchoring."""

        mean_module = getattr(self.actor, "mlp", None)
        anchor_action = getattr(mean_module, "anchor_action_without_stability", None)
        if not callable(anchor_action):
            return composite_mean
        prediction = anchor_action(policy_observation)
        if prediction.shape != composite_mean.shape:
            raise RuntimeError(
                "anchor-only actor mean must match the composite action shape"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError("anchor-only actor mean contains non-finite values")
        return prediction

    def _backward_policy_and_low_expert(
        self,
        primary_loss: torch.Tensor,
        anchor_loss: torch.Tensor,
        stage_aux_loss: torch.Tensor,
        low_expert_loss: torch.Tensor,
        *,
        gate_supervision_has_rows: bool,
        residual_supervision_has_rows: bool,
    ) -> None:
        """Backpropagate joint or private supervised gate/residual gradients.

        Existing runners keep exact ``joint`` behavior.  The fail-closed branch
        can instead replace gate gradients with raw-gate stage BCE only and
        residual gradients with HIGH-anchor plus LOW-expert supervision.  PPO
        still trains the critic/distribution, but cannot open the gate or bend
        the residual against those labels.  Global empty masks leave the
        corresponding grads as ``None`` so Adam momentum cannot move private
        parameters on unlabeled data.
        """

        weighted_expert_loss = self.low_expert_distillation_loss_coef * low_expert_loss
        self.optimizer.zero_grad(set_to_none=True)
        globally_gate_supervised = bool(gate_supervision_has_rows)
        globally_residual_supervised = bool(residual_supervision_has_rows)
        if self.is_multi_gpu and (
            self.capture_gate_gradient_mode != "joint"
            or self.low_expert_residual_gradient_mode != "joint"
        ):
            supervised_counts = torch.tensor(
                (
                    float(globally_gate_supervised),
                    float(globally_residual_supervised),
                ),
                device=primary_loss.device,
                dtype=torch.float32,
            )
            torch.distributed.all_reduce(
                supervised_counts, op=torch.distributed.ReduceOp.SUM
            )
            globally_gate_supervised = bool(supervised_counts[0].item() > 0.0)
            globally_residual_supervised = bool(
                supervised_counts[1].item() > 0.0
            )

        gate_supervised_gradients: tuple[torch.Tensor | None, ...] | None = None
        if (
            self.capture_gate_gradient_mode == "stage_bce_only"
            and globally_gate_supervised
            and not self.capture_branches_frozen
        ):
            gate_supervised_gradients = torch.autograd.grad(
                self.stage_aux_loss_coef * stage_aux_loss,
                self.capture_gate_parameters,
                retain_graph=True,
                allow_unused=True,
            )

        residual_supervised_gradients: tuple[torch.Tensor | None, ...] | None = None
        if (
            self.low_expert_residual_gradient_mode == "supervised_only"
            and globally_residual_supervised
            and not self.capture_gate_warmup_active
            and not self.capture_branches_frozen
        ):
            residual_supervision = (
                self.anchor_loss_coef * anchor_loss + weighted_expert_loss
            )
            residual_supervised_gradients = torch.autograd.grad(
                residual_supervision,
                self.capture_residual_parameters,
                retain_graph=True,
                allow_unused=True,
            )

        backward_loss = primary_loss
        if self.low_expert_residual_gradient_mode == "joint":
            backward_loss = backward_loss + weighted_expert_loss
        backward_loss.backward()

        if (
            self.capture_gate_gradient_mode == "stage_bce_only"
            and not self.capture_branches_frozen
        ):
            for index, parameter in enumerate(self.capture_gate_parameters):
                gradient = (
                    None
                    if gate_supervised_gradients is None
                    else gate_supervised_gradients[index]
                )
                parameter.grad = None if gradient is None else gradient.detach()
        if (
            self.low_expert_residual_gradient_mode == "supervised_only"
            and not self.capture_branches_frozen
        ):
            for index, parameter in enumerate(self.capture_residual_parameters):
                gradient = (
                    None
                    if residual_supervised_gradients is None
                    else residual_supervised_gradients[index]
                )
                parameter.grad = None if gradient is None else gradient.detach()

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        policy_observation = obs[self.anchor_policy_observation_group]
        if policy_observation.shape != (actions.shape[0], INPUT_DIM):
            raise RuntimeError(
                f"deployable policy group changed shape to {tuple(policy_observation.shape)}"
            )
        stage, sensor_age_lr = training_anchor_context(
            self.anchor_env,
            num_envs=actions.shape[0],
            device=self.device,
            sensor_age_scale=self.anchor_sensor_age_scale,
        )
        cache = self.anchor_builder.build(
            policy_observation,
            sensor_age_lr=sensor_age_lr,
            stage=stage,
            student_mean=self._anchor_student_mean(
                policy_observation, self.actor.output_mean
            ).detach(),
        )
        stage_aux = training_stage_auxiliary_context(
            self.anchor_env,
            stage=stage,
            num_envs=actions.shape[0],
            device=self.device,
            reset_mask_steps=self.stage_aux_reset_mask_steps,
            high_end_weight=self.stage_aux_high_end_weight,
        )
        if self.low_expert_builder is None:
            low_expert_target = torch.zeros_like(actions)
            low_expert_mask = torch.zeros(
                actions.shape[0], 1, device=actions.device, dtype=torch.bool
            )
        else:
            low_expert = self.low_expert_builder.build(
                policy_observation, stage=stage
            )
            low_expert_target = low_expert.target
            low_expert_mask = low_expert.mask
        self.transition.anchor_target = cache.target
        self.transition.anchor_mask = cache.mask
        self.transition.stage_aux_label = stage_aux.label.detach()
        self.transition.stage_aux_mask = stage_aux.mask.detach()
        self.transition.stage_aux_weight = stage_aux.weight.detach()
        self.transition.low_expert_residual_target = low_expert_target.detach()
        self.transition.low_expert_residual_mask = low_expert_mask.detach()
        self.anchor_invalid_rows_total += int((~cache.finite).sum().item())
        self.anchor_rows_total += int(cache.finite.numel())
        return actions

    def update(self) -> dict[str, float]:
        """RSL-RL 5.0.1 PPO update plus the cached HIGH-only MSE."""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_anchor_loss = 0.0
        mean_anchor_fraction = 0.0
        mean_stage_aux_loss = 0.0
        mean_stage_aux_high_loss = 0.0
        mean_stage_aux_low_loss = 0.0
        mean_stage_aux_accuracy = 0.0
        mean_stage_aux_valid_fraction = 0.0
        mean_stage_aux_low_fraction = 0.0
        mean_gate_grad_norm = 0.0
        mean_residual_grad_norm = 0.0
        mean_stability_grad_norm = 0.0
        mean_low_expert_distillation_loss = 0.0
        mean_low_expert_valid_fraction = 0.0
        mean_low_expert_abs_target = 0.0
        mean_low_expert_raw_gate = 0.0
        mean_low_expert_calibrated_gate = 0.0
        mean_low_expert_effective_gate = 0.0
        mean_low_expert_ungated_residual_abs = 0.0
        mean_low_expert_effective_delta_abs = 0.0
        mean_high_effective_delta_abs = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1.0e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = tuple(p for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(
                                self.anchor_min_learning_rate,
                                self.learning_rate / 1.5,
                            )
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(
                                self.anchor_max_learning_rate,
                                self.learning_rate * 1.5,
                            )
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    self._sync_optimizer_learning_rates()

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            policy_observation = batch.observations[
                self.anchor_policy_observation_group
            ]
            anchor_loss = masked_anchor_mse(
                self._anchor_student_mean(
                    policy_observation, self.actor.output_mean
                ),
                batch.anchor_targets,
                batch.anchor_masks,
            )
            stage_aux_logits, stage_aux_source = stage_auxiliary_logits(
                self.actor, batch.observations, self.stage_aux_head
            )
            expected_stage_aux_source = self.stage_aux_source
            if stage_aux_source != expected_stage_aux_source:
                raise RuntimeError("stage auxiliary source changed after construction")
            stage_aux = balanced_masked_stage_bce(
                stage_aux_logits,
                batch.stage_aux_labels,
                batch.stage_aux_masks,
                batch.stage_aux_weights,
            )
            if self.low_expert_builder is None:
                zero = surrogate_loss.new_zeros(())
                ungated_residual = None
                low_expert_distillation = LowExpertDistillationLoss(
                    total=zero,
                    valid_fraction=zero.detach(),
                    mean_abs_target=zero.detach(),
                )
            else:
                ungated_residual = self._ungated_capture_residual(
                    batch.observations
                )
                low_expert_distillation = masked_low_expert_smooth_l1(
                    ungated_residual,
                    batch.low_expert_residual_targets,
                    batch.low_expert_residual_masks,
                    beta=self.low_expert_smooth_l1_beta,
                )
                with torch.no_grad():
                    policy_observation = batch.observations[
                        self.anchor_policy_observation_group
                    ]
                    mean_module = getattr(self.actor, "mlp", None)
                    calibrate = getattr(
                        mean_module, "calibrate_capture_probability", None
                    )
                    if not callable(calibrate):
                        raise TypeError(
                            "LOW expert diagnostics require FastBase gate calibration"
                        )
                    raw_gate = torch.sigmoid(stage_aux_logits.detach())
                    calibrated_gate = calibrate(raw_gate)
                    confidence = (
                        policy_observation[:, VALID_SLICE]
                        .amin(dim=1, keepdim=True)
                        .clamp(0.0, 1.0)
                    )
                    effective_gate = confidence * calibrated_gate
                    effective_delta = effective_gate * ungated_residual.detach()
                    low_weight = batch.low_expert_residual_masks.float()
                    high_weight = batch.anchor_masks.float()

                    def _masked_row_mean(
                        value: torch.Tensor, weight: torch.Tensor
                    ) -> float:
                        if value.ndim == 2 and value.shape[1] != 1:
                            value = value.abs().mean(dim=1, keepdim=True)
                        denominator = weight.sum().clamp_min(1.0)
                        return float((value * weight).sum().item() / denominator.item())

                    mean_low_expert_raw_gate += _masked_row_mean(
                        raw_gate, low_weight
                    )
                    mean_low_expert_calibrated_gate += _masked_row_mean(
                        calibrated_gate, low_weight
                    )
                    mean_low_expert_effective_gate += _masked_row_mean(
                        effective_gate, low_weight
                    )
                    mean_low_expert_ungated_residual_abs += _masked_row_mean(
                        ungated_residual.detach(), low_weight
                    )
                    mean_low_expert_effective_delta_abs += _masked_row_mean(
                        effective_delta, low_weight
                    )
                    mean_high_effective_delta_abs += _masked_row_mean(
                        effective_delta, high_weight
                    )
            primary_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.anchor_loss_coef * anchor_loss
                + self.stage_aux_loss_coef * stage_aux.total
            )

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations)
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.functional.mse_loss(predicted_embedding, target_embedding)

            self._backward_policy_and_low_expert(
                primary_loss,
                anchor_loss,
                stage_aux.total,
                low_expert_distillation.total,
                gate_supervision_has_rows=bool(
                    batch.stage_aux_masks.any().item()
                ),
                residual_supervision_has_rows=bool(
                    (
                        batch.anchor_masks.any()
                        | batch.low_expert_residual_masks.any()
                    ).item()
                ),
            )
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            (
                gate_grad_norm,
                residual_grad_norm,
                stability_grad_norm,
            ) = self._clip_training_gradients()
            mean_gate_grad_norm += gate_grad_norm
            mean_residual_grad_norm += residual_grad_norm
            mean_stability_grad_norm += stability_grad_norm
            mean_low_expert_distillation_loss += (
                low_expert_distillation.total.item()
            )
            mean_low_expert_valid_fraction += (
                low_expert_distillation.valid_fraction.item()
            )
            mean_low_expert_abs_target += (
                low_expert_distillation.mean_abs_target.item()
            )
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_anchor_loss += anchor_loss.item()
            mean_anchor_fraction += batch.anchor_masks.float().mean().item()
            mean_stage_aux_loss += stage_aux.total.item()
            mean_stage_aux_high_loss += stage_aux.high.item()
            mean_stage_aux_low_loss += stage_aux.low.item()
            mean_stage_aux_accuracy += stage_aux.accuracy.item()
            mean_stage_aux_valid_fraction += stage_aux.valid_fraction.item()
            mean_stage_aux_low_fraction += stage_aux.low_fraction.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_anchor_loss /= num_updates
        mean_anchor_fraction /= num_updates
        mean_stage_aux_loss /= num_updates
        mean_stage_aux_high_loss /= num_updates
        mean_stage_aux_low_loss /= num_updates
        mean_stage_aux_accuracy /= num_updates
        mean_stage_aux_valid_fraction /= num_updates
        mean_stage_aux_low_fraction /= num_updates
        mean_gate_grad_norm /= num_updates
        mean_residual_grad_norm /= num_updates
        mean_stability_grad_norm /= num_updates
        mean_low_expert_distillation_loss /= num_updates
        mean_low_expert_valid_fraction /= num_updates
        mean_low_expert_abs_target /= num_updates
        mean_low_expert_raw_gate /= num_updates
        mean_low_expert_calibrated_gate /= num_updates
        mean_low_expert_effective_gate /= num_updates
        mean_low_expert_ungated_residual_abs /= num_updates
        mean_low_expert_effective_delta_abs /= num_updates
        mean_high_effective_delta_abs /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        self.storage.clear()
        warmup_was_active = self.capture_gate_warmup_active
        self._advance_capture_gate_warmup()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "high_friction_anchor": mean_anchor_loss,
            "high_friction_anchor_fraction": mean_anchor_fraction,
            "stage_auxiliary": mean_stage_aux_loss,
            "stage_auxiliary_high": mean_stage_aux_high_loss,
            "stage_auxiliary_low": mean_stage_aux_low_loss,
            "stage_auxiliary_accuracy": mean_stage_aux_accuracy,
            "stage_auxiliary_valid_fraction": mean_stage_aux_valid_fraction,
            "stage_auxiliary_low_fraction": mean_stage_aux_low_fraction,
            "capture_gate_warmup_active": float(warmup_was_active),
            "capture_gate_updates_completed": float(
                self.capture_gate_updates_completed
            ),
            "gate_lr": float(self.capture_gate_current_learning_rate or 0.0),
            "gate_grad_norm": mean_gate_grad_norm,
            "capture_gate_stage_bce_only": float(
                self.capture_gate_gradient_mode == "stage_bce_only"
            ),
            "residual_lr": float(
                self.capture_residual_current_learning_rate or 0.0
            ),
            "residual_grad_norm": mean_residual_grad_norm,
            "stability_residual_lr": float(
                self.stability_residual_current_learning_rate or 0.0
            ),
            "stability_residual_grad_norm": mean_stability_grad_norm,
            "low_expert_distillation": mean_low_expert_distillation_loss,
            "low_expert_valid_fraction": mean_low_expert_valid_fraction,
            "low_expert_abs_target": mean_low_expert_abs_target,
            "low_expert_raw_gate": mean_low_expert_raw_gate,
            "low_expert_calibrated_gate": mean_low_expert_calibrated_gate,
            "low_expert_effective_gate": mean_low_expert_effective_gate,
            "low_expert_ungated_residual_abs": (
                mean_low_expert_ungated_residual_abs
            ),
            "low_expert_effective_delta_abs": (
                mean_low_expert_effective_delta_abs
            ),
            "high_effective_delta_abs": mean_high_effective_delta_abs,
            "low_expert_supervised_only": float(
                self.low_expert_residual_gradient_mode == "supervised_only"
            ),
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.anchor_builder.teacher.eval()
        if self.stage_aux_head is not None:
            self.stage_aux_head.train()
        if self.low_expert_builder is not None:
            self.low_expert_builder.expert.eval()
            for parameter in self.low_expert_builder.expert.parameters():
                parameter.requires_grad_(False)

    def eval_mode(self) -> None:
        super().eval_mode()
        self.anchor_builder.teacher.eval()
        if self.stage_aux_head is not None:
            self.stage_aux_head.eval()
        if self.low_expert_builder is not None:
            self.low_expert_builder.expert.eval()

    def save(self) -> dict:
        payload = super().save()
        if self.stage_aux_head is not None:
            payload["stage_auxiliary_state_dict"] = self.stage_aux_head.state_dict()
        payload["high_friction_anchor"] = self.anchor_manifest()
        payload["capture_gate_warmup"] = {
            "configured_updates": self.capture_gate_warmup_updates,
            "completed_updates": self.capture_gate_updates_completed,
            "active": self.capture_gate_warmup_active,
            "warmup_learning_rate": self.capture_gate_warmup_learning_rate,
            "released_learning_rate": self.capture_gate_learning_rate,
            "current_learning_rate": self.capture_gate_current_learning_rate,
            "max_grad_norm": self.capture_gate_max_grad_norm,
            "gradient_mode": self.capture_gate_gradient_mode,
            "residual_warmup_learning_rate": 0.0,
            "residual_released_learning_rate": self.capture_residual_learning_rate,
            "residual_current_learning_rate": (
                self.capture_residual_current_learning_rate
            ),
            "residual_max_grad_norm": self.capture_residual_max_grad_norm,
            "capture_branches_frozen": self.capture_branches_frozen,
        }
        return payload

    def _migrate_legacy_gate_only_optimizer_state(
        self, optimizer_state: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Split a legacy PPO+gate checkpoint into PPO+gate+residual roles.

        The prior optimizer removed only gate parameters from PPO.  Parameter
        order was otherwise preserved, so the construction-time identity list
        maps every saved Adam state—including any already-created residual
        moments—to the new role without relying on tensor shape heuristics.
        """

        if not self.capture_residual_parameters:
            return None
        saved_groups = optimizer_state.get("param_groups", [])
        saved_roles = [group.get(OPTIMIZER_ROLE_KEY) for group in saved_groups]
        if (
            saved_roles.count(PPO_OPTIMIZER_ROLE) != 1
            or saved_roles.count(CAPTURE_GATE_OPTIMIZER_ROLE) != 1
            or CAPTURE_RESIDUAL_OPTIMIZER_ROLE in saved_roles
            or len(saved_groups) != 2
        ):
            return None

        current_state = self.optimizer.state_dict()
        current_groups = current_state.get("param_groups", [])
        current_roles = [
            group.get(OPTIMIZER_ROLE_KEY) for group in current_groups
        ]
        if (
            current_roles.count(PPO_OPTIMIZER_ROLE) != 1
            or current_roles.count(CAPTURE_GATE_OPTIMIZER_ROLE) != 1
            or current_roles.count(CAPTURE_RESIDUAL_OPTIMIZER_ROLE) != 1
            or len(current_groups) != 3
        ):
            return None

        saved_by_role = {
            group[OPTIMIZER_ROLE_KEY]: group for group in saved_groups
        }
        saved_ppo_ids = list(saved_by_role[PPO_OPTIMIZER_ROLE]["params"])
        saved_gate_ids = list(
            saved_by_role[CAPTURE_GATE_OPTIMIZER_ROLE]["params"]
        )
        if len(saved_ppo_ids) != len(self._legacy_gate_only_ppo_parameters):
            raise ValueError(
                "legacy capture optimizer PPO parameter count changed: "
                f"checkpoint={len(saved_ppo_ids)}, "
                f"expected={len(self._legacy_gate_only_ppo_parameters)}"
            )
        if len(saved_gate_ids) != len(self.capture_gate_parameters):
            raise ValueError(
                "legacy capture optimizer gate parameter count changed: "
                f"checkpoint={len(saved_gate_ids)}, "
                f"expected={len(self.capture_gate_parameters)}"
            )

        current_id_by_identity: dict[int, int] = {}
        for live_group, serialized_group in zip(
            self.optimizer.param_groups, current_groups
        ):
            if len(live_group["params"]) != len(serialized_group["params"]):
                raise RuntimeError("current optimizer serialization order drifted")
            for parameter, serialized_id in zip(
                live_group["params"], serialized_group["params"]
            ):
                current_id_by_identity[id(parameter)] = int(serialized_id)

        saved_to_current: dict[int, int] = {}
        for saved_id, parameter in zip(
            saved_ppo_ids, self._legacy_gate_only_ppo_parameters
        ):
            saved_to_current[int(saved_id)] = current_id_by_identity[id(parameter)]
        for saved_id, parameter in zip(
            saved_gate_ids, self.capture_gate_parameters
        ):
            saved_to_current[int(saved_id)] = current_id_by_identity[id(parameter)]

        saved_states = optimizer_state.get("state", {})
        unmapped_state_ids = set(saved_states) - set(saved_to_current)
        if unmapped_state_ids:
            raise ValueError(
                "legacy capture optimizer contains unmapped Adam state ids: "
                f"{sorted(unmapped_state_ids)}"
            )
        migrated = copy.deepcopy(current_state)
        migrated["state"] = {
            saved_to_current[int(saved_id)]: copy.deepcopy(value)
            for saved_id, value in saved_states.items()
        }
        migrated_by_role = {
            group[OPTIMIZER_ROLE_KEY]: group
            for group in migrated["param_groups"]
        }
        for role, source_role in (
            (PPO_OPTIMIZER_ROLE, PPO_OPTIMIZER_ROLE),
            (CAPTURE_GATE_OPTIMIZER_ROLE, CAPTURE_GATE_OPTIMIZER_ROLE),
            (CAPTURE_RESIDUAL_OPTIMIZER_ROLE, PPO_OPTIMIZER_ROLE),
        ):
            target = migrated_by_role[role]
            source = saved_by_role[source_role]
            target_params = target["params"]
            for name, value in source.items():
                if name not in ("params", OPTIMIZER_ROLE_KEY):
                    target[name] = copy.deepcopy(value)
            target["params"] = target_params
            target[OPTIMIZER_ROLE_KEY] = role
        return migrated

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load old anchor checkpoints and new auxiliary-head checkpoints safely."""

        effective_cfg = (
            {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
            if load_cfg is None
            else dict(load_cfg)
        )
        base_cfg = dict(effective_cfg)
        base_cfg["optimizer"] = False
        super().load(loaded_dict, base_cfg, strict)

        warmup_state = loaded_dict.get("capture_gate_warmup")
        restore_training_phase = bool(
            effective_cfg.get("optimizer", False)
            or effective_cfg.get("iteration", False)
        )
        if warmup_state is not None and restore_training_phase:
            if not isinstance(warmup_state, dict):
                raise TypeError("capture_gate_warmup checkpoint entry must be a dictionary")
            saved_total = int(warmup_state.get("configured_updates", -1))
            if saved_total != self.capture_gate_warmup_updates:
                raise ValueError(
                    "capture gate warm-up configuration changed across resume: "
                    f"checkpoint={saved_total}, current={self.capture_gate_warmup_updates}"
                )
            self.capture_gate_updates_completed = int(
                warmup_state.get("completed_updates", -1)
            )
        elif warmup_state is None and self.capture_gate_warmup_updates > 0 and restore_training_phase:
            # Compatibility path for the original FastBase model_0 artifact.
            # A legacy trained residual is rejected below because its output
            # head is no longer exact zero.
            self.capture_gate_updates_completed = (
                int(loaded_dict.get("iter", 0))
                if effective_cfg.get("iteration", False)
                else 0
            )
        if self.capture_gate_updates_completed < 0:
            raise ValueError("capture gate warm-up completed counter must be non-negative")
        self._apply_capture_gate_warmup_state()

        auxiliary_state = loaded_dict.get("stage_auxiliary_state_dict")
        if auxiliary_state is not None:
            if self.stage_aux_head is None:
                raise ValueError(
                    "checkpoint contains a fallback auxiliary head but current actor "
                    "uses its native capture gate"
                )
            self.stage_aux_head.load_state_dict(auxiliary_state, strict=strict)

        if effective_cfg.get("optimizer"):
            saved_manifest = loaded_dict.get("high_friction_anchor", {})
            saved_gate_training = (
                saved_manifest.get("capture_gate_warmup", {})
                if isinstance(saved_manifest, dict)
                else {}
            )
            saved_gate_gradient_mode = (
                saved_gate_training.get("gradient_mode", "joint")
                if isinstance(saved_gate_training, dict)
                else "joint"
            )
            if saved_gate_gradient_mode != self.capture_gate_gradient_mode:
                raise ValueError(
                    "capture gate gradient mode changed across optimizer resume; "
                    "restart with a fresh optimizer (checkpoint="
                    f"{saved_gate_gradient_mode!r}, current="
                    f"{self.capture_gate_gradient_mode!r})"
                )
            saved_capture_frozen = bool(
                saved_gate_training.get("capture_branches_frozen", False)
            )
            if saved_capture_frozen != self.capture_branches_frozen:
                raise ValueError(
                    "capture branch freeze contract changed across optimizer "
                    "resume; restart with a fresh optimizer (checkpoint="
                    f"{saved_capture_frozen}, current="
                    f"{self.capture_branches_frozen})"
                )
            saved_low_expert = (
                saved_manifest.get("low_expert_distillation", {})
                if isinstance(saved_manifest, dict)
                else {}
            )
            saved_gradient_mode = (
                saved_low_expert.get("residual_gradient_mode", "joint")
                if isinstance(saved_low_expert, dict)
                else "joint"
            )
            if saved_gradient_mode != self.low_expert_residual_gradient_mode:
                raise ValueError(
                    "LOW expert residual gradient mode changed across optimizer "
                    "resume; restart with a fresh optimizer (checkpoint="
                    f"{saved_gradient_mode!r}, current="
                    f"{self.low_expert_residual_gradient_mode!r})"
                )
            optimizer_state = loaded_dict.get("optimizer_state_dict")
            if optimizer_state is None:
                raise KeyError("checkpoint has no optimizer_state_dict")
            saved_groups = optimizer_state.get("param_groups", [])
            current_state = self.optimizer.state_dict()
            current_groups = current_state.get("param_groups", [])
            if len(saved_groups) == len(current_groups):
                self.optimizer.load_state_dict(optimizer_state)
            elif (
                migrated_capture := self._migrate_legacy_gate_only_optimizer_state(
                    optimizer_state
                )
            ) is not None:
                self.optimizer.load_state_dict(migrated_capture)
            elif (
                self.stage_aux_head is not None
                and auxiliary_state is None
                and len(saved_groups) + 1 == len(current_groups)
            ):
                # Legacy AnchoredPPO checkpoints had one actor+critic optimizer
                # group.  Preserve its Adam moments and append the fresh,
                # empty auxiliary group without guessing parameter identities.
                migrated = copy.deepcopy(optimizer_state)
                migrated["param_groups"].append(copy.deepcopy(current_groups[-1]))
                self.optimizer.load_state_dict(migrated)
            else:
                raise ValueError(
                    "optimizer parameter-group layout is incompatible with the "
                    "stage auxiliary/capture private roles"
                )
            self._validate_optimizer_roles()
            self._sync_optimizer_learning_rates()
        return bool(effective_cfg.get("iteration", False))

    def broadcast_parameters(self) -> None:
        """Synchronize the training-only head in addition to actor/critic."""

        super().broadcast_parameters()
        if self.stage_aux_head is not None:
            auxiliary_state = [self.stage_aux_head.state_dict()]
            torch.distributed.broadcast_object_list(auxiliary_state, src=0)
            self.stage_aux_head.load_state_dict(auxiliary_state[0])

    def reduce_parameters(self) -> None:
        """Average actor/critic and auxiliary-head gradients across workers."""

        super().reduce_parameters()
        if self.stage_aux_head is None:
            return
        parameters = [p for p in self.stage_aux_head.parameters() if p.grad is not None]
        if not parameters:
            return
        flat = torch.cat([parameter.grad.view(-1) for parameter in parameters])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.data.copy_(flat[offset : offset + count].view_as(parameter.grad))
            offset += count

    def anchor_manifest(self) -> dict[str, Any]:
        invalid_fraction = (
            self.anchor_invalid_rows_total / self.anchor_rows_total
            if self.anchor_rows_total
            else 0.0
        )
        actor_mean = getattr(self.actor, "mlp", None)
        stability_residual = getattr(actor_mean, "stability_residual", None)
        stability_parameters = (
            list(stability_residual.parameters())
            if isinstance(stability_residual, nn.Module)
            else []
        )
        stability_parameter_ids = {id(parameter) for parameter in stability_parameters}
        ppo_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == PPO_OPTIMIZER_ROLE
            for parameter in group["params"]
        }
        stability_role_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            if group.get(OPTIMIZER_ROLE_KEY) == STABILITY_RESIDUAL_OPTIMIZER_ROLE
            for parameter in group["params"]
        }
        return {
            "format": ANCHOR_FORMAT,
            "format_version": ANCHOR_FORMAT_VERSION,
            "rsl_rl_version": _require_rsl_rl_v5(),
            "algorithm_class": f"{type(self).__module__}:{type(self).__name__}",
            "teacher_checkpoint": self.anchor_teacher_checkpoint,
            "teacher_checkpoint_sha256": _sha256_file(self.anchor_teacher_checkpoint),
            "teacher_source_onnx_sha256": self.anchor_expected_teacher_source_sha256,
            "teacher_source_graph": self.anchor_teacher_source_graph,
            "teacher_conversion_parity": self.anchor_teacher_parity,
            "teacher_provenance": self.anchor_teacher_provenance,
            "teacher_frozen": all(
                not parameter.requires_grad
                for parameter in self.anchor_builder.teacher.parameters()
            ),
            "teacher_eval": not self.anchor_builder.teacher.training,
            "teacher_inference_calls": self.anchor_builder.inference_calls,
            "teacher_cache_writes": self.anchor_builder.cache_writes,
            "teacher_invalid_fraction": invalid_fraction,
            "actor_observation_group": self.anchor_policy_observation_group,
            "actor_observation_dim": INPUT_DIM,
            "actor_trailing_feature_mode": "motion_feedback",
            "teacher_trailing_feature_mode": "sensor_age",
            "teacher_signal_in_actor_observation": False,
            "privileged_stage_in_actor_observation": False,
            "proprioceptive_stability_residual": {
                "enabled": bool(stability_parameters),
                "input_dim": 482 if stability_parameters else None,
                "input_slices": (
                    [[0, 480], [1862, 1864]] if stability_parameters else []
                ),
                "uses_force_contact_mu_or_stage": False,
                "limit_actor_units": (
                    float(getattr(actor_mean, "stability_limit"))
                    if stability_parameters
                    else None
                ),
                "joint_position_action_scale_rad": 0.25,
                "maximum_joint_correction_rad": (
                    0.25 * float(getattr(actor_mean, "stability_limit"))
                    if stability_parameters
                    else None
                ),
                "heading_thresholds_disabled_during_commanded_turn": True,
                "legacy_capture_checkpoint_imported_with_zero_output": bool(
                    getattr(actor_mean, "loaded_legacy_stability", False)
                ),
                "output_layer_exact_zero": (
                    capture_residual_has_zero_output(stability_residual)
                    if isinstance(stability_residual, nn.Sequential)
                    else None
                ),
                "parameters_in_ppo_role": bool(stability_parameters)
                and stability_parameter_ids <= ppo_parameter_ids,
                "parameters_in_stability_private_role": bool(stability_parameters)
                and stability_parameter_ids == stability_role_parameter_ids,
                "parameters_in_capture_private_roles": bool(
                    stability_parameter_ids
                    & {
                        id(parameter)
                        for group in self.optimizer.param_groups
                        if group.get(OPTIMIZER_ROLE_KEY)
                        in (
                            CAPTURE_GATE_OPTIMIZER_ROLE,
                            CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
                        )
                        for parameter in group["params"]
                    }
                ),
                "excluded_from_frozen_teacher_anchor": bool(
                    stability_parameters
                ),
                "optimizer_learning_rate": (
                    self.stability_residual_current_learning_rate
                    if stability_parameters
                    else None
                ),
                "optimizer_max_grad_norm": (
                    self.stability_residual_max_grad_norm
                    if stability_parameters
                    else None
                ),
                "adaptive_ppo_lr_excluded": bool(stability_parameters),
                "optimizer_role": (
                    STABILITY_RESIDUAL_OPTIMIZER_ROLE
                    if stability_parameters
                    else None
                ),
                "checkpointed_and_exported": bool(stability_parameters),
            },
            "stage_auxiliary": {
                "enabled": self.stage_aux_loss_coef > 0.0,
                "loss_coef": self.stage_aux_loss_coef,
                "supervision_target": (
                    "raw_actor_capture_gate_pre_calibration"
                    if self.stage_aux_source == "actor_raw_capture_gate"
                    else "legacy_actor_capture_gate"
                    if self.stage_aux_source == "actor_capture_gate"
                    else "training_only_shared_latent_head"
                ),
                "source": self.stage_aux_source,
                "calibration_in_bce": False,
                "head_hidden_dim": (
                    None if self.stage_aux_head is None else self.stage_aux_hidden_dim
                ),
                "shared_actor_latent_dim": (
                    None
                    if self.stage_aux_head is None
                    else actor_shared_latent_dim(self.actor)
                ),
                "labels": {"HIGH_START": 0, "LOW": 1, "HIGH_END": 0},
                "balanced_per_class_bce": True,
                "high_end_sample_weight": self.stage_aux_high_end_weight,
                "reset_mask_steps": self.stage_aux_reset_mask_steps,
                "unknown_stage_masked": True,
                "label_in_actor_observation": False,
                "fallback_head_in_actor_state_dict": False,
                "fallback_head_in_policy_export": False,
                "native_capture_gate_in_policy_export": self.stage_aux_uses_actor_capture_gate,
            },
            "capture_gate_calibration": {
                "logit_scale": float(
                    getattr(getattr(self.actor, "mlp", None), "gate_logit_scale", 1.0)
                ),
                "logit_bias": float(
                    getattr(getattr(self.actor, "mlp", None), "gate_logit_bias", 0.0)
                ),
                "state_dict_entries": bool(
                    torch.is_tensor(
                        getattr(
                            getattr(self.actor, "mlp", None),
                            "gate_logit_scale",
                            None,
                        )
                    )
                    and torch.is_tensor(
                        getattr(
                            getattr(self.actor, "mlp", None),
                            "gate_logit_bias",
                            None,
                        )
                    )
                ),
                "legacy_checkpoint_values_injected_from_config": bool(
                    getattr(
                        getattr(self.actor, "mlp", None),
                        "loaded_legacy_calibration",
                        False,
                    )
                ),
                "monotone": True,
                "affects": "deployable_residual_authority_only",
            },
            "capture_gate_warmup": {
                "configured_updates": self.capture_gate_warmup_updates,
                "completed_updates": self.capture_gate_updates_completed,
                "active": self.capture_gate_warmup_active,
                "warmup_learning_rate": self.capture_gate_warmup_learning_rate,
                "released_learning_rate": self.capture_gate_learning_rate,
                "current_learning_rate": self.capture_gate_current_learning_rate,
                "max_grad_norm": self.capture_gate_max_grad_norm,
                "gradient_mode": self.capture_gate_gradient_mode,
                "ppo_gradient_into_gate": (
                    self.capture_gate_gradient_mode == "joint"
                ),
                "stage_bce_gradient_into_gate": True,
                "independent_optimizer_group": bool(
                    self.capture_gate_parameters
                ),
                "adaptive_ppo_lr_excluded": bool(
                    self.capture_gate_parameters
                ),
                "residual_exact_zero_while_active": (
                    None
                    if self.capture_residual is None
                    else (
                        capture_residual_has_zero_output(self.capture_residual)
                        if self.capture_gate_warmup_active
                        else None
                    )
                ),
                "counter_checkpointed": True,
                "frozen_by_actor_contract": self.capture_branches_frozen,
                "all_parameters_require_grad_false": bool(
                    self.capture_gate_parameters
                )
                and all(
                    not parameter.requires_grad
                    for parameter in self.capture_gate_parameters
                ),
            },
            "capture_residual_optimizer": {
                "warmup_learning_rate": 0.0,
                "released_learning_rate": self.capture_residual_learning_rate,
                "current_learning_rate": (
                    self.capture_residual_current_learning_rate
                ),
                "max_grad_norm": self.capture_residual_max_grad_norm,
                "independent_optimizer_group": bool(
                    self.capture_residual_parameters
                ),
                "adaptive_ppo_lr_excluded": bool(
                    self.capture_residual_parameters
                ),
                "release_counter_shared_with_gate_warmup": True,
                "optimizer_role": CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
                "frozen_by_actor_contract": self.capture_branches_frozen,
                "all_parameters_require_grad_false": bool(
                    self.capture_residual_parameters
                )
                and all(
                    not parameter.requires_grad
                    for parameter in self.capture_residual_parameters
                ),
            },
            "low_expert_distillation": {
                "enabled": self.low_expert_builder is not None,
                "loss_coef": self.low_expert_distillation_loss_coef,
                "residual_gradient_mode": (
                    self.low_expert_residual_gradient_mode
                ),
                "ppo_gradient_into_residual": (
                    self.low_expert_residual_gradient_mode == "joint"
                ),
                "high_anchor_gradient_into_residual": True,
                "smooth_l1_beta": self.low_expert_smooth_l1_beta,
                "target_cap": self.low_expert_target_cap,
                "expert_checkpoint": self.low_expert_checkpoint or None,
                "expert_checkpoint_sha256": (
                    self.low_expert_checkpoint_sha256 or None
                ),
                "counterfactual_expert_command": list(self.low_expert_command),
                "command_flat_slice": [30, 45],
                "command_history_order": "term-major_oldest-to-newest",
                "target_definition": (
                    "clamp(model6149(obs_cmd_0.16)-"
                    "frozen_speedboost(original_obs),-0.20,+0.20)"
                ),
                "prediction_definition": "0.55*tanh(ungated_residual)",
                "mask": "LOW_and_both_foot_valid_gt_0.5_and_finite",
                "expert_inference_calls": (
                    0
                    if self.low_expert_builder is None
                    else self.low_expert_builder.inference_calls
                ),
                "expert_cache_writes": (
                    0
                    if self.low_expert_builder is None
                    else self.low_expert_builder.cache_writes
                ),
                "expert_rows_inferred": (
                    0
                    if self.low_expert_builder is None
                    else self.low_expert_builder.rows_inferred
                ),
                "expert_in_actor_state_dict": False,
                "expert_in_optimizer": False,
                "expert_in_policy_export": False,
                "targets_cached_once_per_rollout_step": True,
                "privileged_stage_in_actor_observation": False,
            },
            "anchor_stage_ids": list(HIGH_STAGE_IDS),
            "anchor_stage_names": ["HIGH_START", "HIGH_END"],
            "excluded_stage_ids": [SPATIAL_LOW],
            "anchor_loss_coef": self.anchor_loss_coef,
            "anchor_delta_cap": self.anchor_delta_cap,
            "teacher_action_clamp": [
                -self.anchor_teacher_action_clamp,
                self.anchor_teacher_action_clamp,
            ],
            "sensor_age_scale_s": self.anchor_sensor_age_scale,
            "adaptive_learning_rate_bounds": [
                self.anchor_min_learning_rate,
                self.anchor_max_learning_rate,
            ],
            "finite_row_rejection": True,
            "rollout_teacher_calls_per_step": 1,
            "cached_target_shape": [OUTPUT_DIM],
        }

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AnchoredPPO":
        """Construct against the audited RSL-RL 5 model/storage API."""

        _require_rsl_rl_v5()
        alg_class: type[AnchoredPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"], obs, cfg["obs_groups"], env
        )
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")
        storage = AnchoredRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        return alg_class(
            actor,
            critic,
            storage,
            env=env,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )


class AnchoredOnPolicyRunner(OnPolicyRunner):
    """Standard RSL-RL runner with an atomic anchor audit manifest."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if not isinstance(self.alg, AnchoredPPO):
            raise TypeError("AnchoredOnPolicyRunner requires AnchoredPPO")
        self._checkpoint_load_audit: dict[str, Any] | None = None
        self._training_provenance: dict[str, Any] | None = None
        self._training_provenance_path: Path | None = None
        self._last_completed_iteration: int | None = None
        self._anchor_manifest_path: Path | None = None
        if log_dir is not None:
            self._training_provenance_path = (
                Path(log_dir) / "params" / "training_provenance.json"
            )
            self._anchor_manifest_path = (
                Path(log_dir) / "params" / "high_friction_anchor_manifest.json"
            )
            self.write_anchor_manifest(self._anchor_manifest_path)

    @property
    def checkpoint_load_audit(self) -> dict[str, Any] | None:
        """Return an isolated copy of the most recent checkpoint load facts."""

        return copy.deepcopy(self._checkpoint_load_audit)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load a completed checkpoint and resume at the *next* iteration.

        RSL-RL 5 saves ``iter`` after the corresponding update has completed,
        but its stock loader restores that value as the next loop index.  That
        repeats the last update number and made model49 + 12 updates finish as
        model60.  The local runner records the saved/next meanings explicitly
        and resumes model49 at iteration 50, yielding model61 after 12 updates.
        """

        checkpoint = Path(path).expanduser().resolve()
        loaded_dict = torch.load(
            checkpoint, weights_only=False, map_location=map_location
        )
        if not isinstance(loaded_dict, dict):
            raise TypeError("checkpoint payload must be a dictionary")
        effective_load_cfg = (
            {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
            if load_cfg is None
            else dict(load_cfg)
        )
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        saved_iteration: int | None = None
        next_iteration: int | None = None
        if load_iteration:
            if "iter" not in loaded_dict:
                raise KeyError("checkpoint has no completed iteration field 'iter'")
            saved_iteration = int(loaded_dict["iter"])
            if saved_iteration < 0:
                raise ValueError("checkpoint completed iteration must be non-negative")
            next_iteration = saved_iteration + 1
            self.current_learning_iteration = next_iteration
            self._last_completed_iteration = saved_iteration
        warmup = loaded_dict.get("capture_gate_warmup")
        self._checkpoint_load_audit = {
            "path": str(checkpoint),
            "strict": bool(strict),
            "load_cfg": effective_load_cfg,
            "saved_completed_iteration": saved_iteration,
            "next_learning_iteration": next_iteration,
            "capture_gate_updates_completed": (
                int(warmup.get("completed_updates", -1))
                if isinstance(warmup, dict)
                else None
            ),
        }
        return loaded_dict["infos"]

    def attach_training_provenance(
        self, payload: dict[str, Any], path: str | Path | None = None
    ) -> None:
        """Persist immutable start facts and embed them into future checkpoints."""

        if not isinstance(payload, dict):
            raise TypeError("training provenance must be a dictionary")
        provenance = copy.deepcopy(payload)
        if provenance.get("format") != TRAINING_PROVENANCE_FORMAT:
            raise ValueError("unexpected training provenance format")
        if int(provenance.get("format_version", -1)) != TRAINING_PROVENANCE_FORMAT_VERSION:
            raise ValueError("unsupported training provenance version")
        # Validate JSON finiteness/serializability before retaining state.
        json.dumps(provenance, allow_nan=False)
        self._training_provenance = provenance
        destination = Path(path) if path is not None else self._training_provenance_path
        if destination is None:
            raise RuntimeError("training provenance destination is unavailable")
        self._training_provenance_path = destination
        _atomic_write_json(destination, provenance)
        if self._anchor_manifest_path is not None:
            self.write_anchor_manifest(self._anchor_manifest_path)

    def _checkpoint_training_provenance(self) -> dict[str, Any] | None:
        if self._training_provenance is None:
            return None
        payload = copy.deepcopy(self._training_provenance)
        payload["checkpoint_state"] = {
            "completed_iteration": self._last_completed_iteration,
            "capture_gate_updates_completed": int(
                getattr(self.alg, "capture_gate_updates_completed", -1)
            ),
        }
        return payload

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the policy and refresh the standalone runtime audit."""

        # ``OnPolicyRunner.learn`` assigns the just-completed loop index before
        # every save call.  Keep that separate from the pre-loop resume value,
        # which denotes the next iteration rather than a completed one.
        self._last_completed_iteration = int(self.current_learning_iteration)
        provenance = self._checkpoint_training_provenance()
        if provenance is not None:
            if infos is None:
                checkpoint_infos: dict[str, Any] = {}
            elif isinstance(infos, dict):
                checkpoint_infos = copy.deepcopy(infos)
            else:
                raise TypeError(
                    "checkpoint infos must be a dictionary when provenance is enabled"
                )
            checkpoint_infos["training_provenance"] = provenance
        else:
            checkpoint_infos = infos
        super().save(path, infos=checkpoint_infos)
        if self._anchor_manifest_path is not None:
            self.write_anchor_manifest(self._anchor_manifest_path)

    def write_anchor_manifest(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.alg.anchor_manifest()
        payload["written_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["runner_class"] = f"{type(self).__module__}:{type(self).__name__}"
        training_provenance = self._checkpoint_training_provenance()
        if training_provenance is not None:
            payload["training_provenance"] = training_provenance
        _atomic_write_json(destination, payload)


__all__ = [
    "ANCHOR_FORMAT",
    "ANCHOR_FORMAT_VERSION",
    "CAPTURE_GATE_OPTIMIZER_ROLE",
    "CAPTURE_RESIDUAL_OPTIMIZER_ROLE",
    "STABILITY_RESIDUAL_OPTIMIZER_ROLE",
    "OPTIMIZER_ROLE_KEY",
    "PPO_OPTIMIZER_ROLE",
    "STRICT_ACTOR_CRITIC_RESUME_CFG",
    "TRAINING_PROVENANCE_FORMAT",
    "TRAINING_PROVENANCE_FORMAT_VERSION",
    "AnchoredOnPolicyRunner",
    "AnchoredPPO",
    "AnchoredRolloutStorage",
    "AnchorBatch",
    "FrozenTeacherAnchorTargetBuilder",
    "FrozenLowExpertResidualTargetBuilder",
    "HIGH_STAGE_IDS",
    "StageAuxiliaryHead",
    "StageAuxiliaryLoss",
    "StageAuxiliaryTargets",
    "LowExpertDistillationLoss",
    "LowExpertResidualBatch",
    "actor_shared_latent_dim",
    "actor_shared_trunk_latent",
    "actor_exploration_std_manifest",
    "balanced_masked_stage_bce",
    "bounded_teacher_targets",
    "canonical_json_sha256",
    "checkpoint_sha256",
    "capture_residual_has_zero_output",
    "high_friction_anchor_mask",
    "masked_anchor_mse",
    "masked_low_expert_smooth_l1",
    "optimizer_roles_manifest",
    "stage_auxiliary_logits",
    "stage_auxiliary_targets",
    "training_anchor_context",
    "training_stage_auxiliary_context",
    "validate_actor_observation_contract",
    "validate_bounded_new_updates",
    "validate_fail_closed_gate_training_start",
    "validate_hall_randomization_seed",
]
