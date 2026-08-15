"""Deployable Hall/proprioceptive low-traction risk estimator.

This module predicts a conservative risk probability directly from the same
causal 1864-D observation used by the Hall locomotion Student.  It never
reconstructs normal force, tangential force, friction, or contact truth.
"""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from .layout_magnetic_student import (
    AGE_SLICE,
    AXES,
    BASE_DIM,
    FEET,
    HEALTH_DIM,
    HISTORY,
    INPUT_DIM,
    MAGNETIC_SLICE,
    PERIOD_SLICE,
    SENSORS,
    TRAILING_FEATURE_MODE_MOTION_FEEDBACK,
    TRAILING_FEATURE_MODE_SENSOR_AGE,
    VALID_SLICE,
    LayoutHallEncoder,
    normalize_trailing_feature_mode,
    schema_for_trailing_feature_mode,
)


INVARIANT_HALL_STATISTICS = 4
INVARIANT_FEATURE_DIM = (
    BASE_DIM
    + FEET * SENSORS * AXES * INVARIANT_HALL_STATISTICS
    + 2 * FEET
    + HEALTH_DIM
)

# Five frames of [vx, vy, yaw] occupy these term-major legacy actor columns.
# The command-invariant risk head accepts the canonical 1864-D packet for ABI
# compatibility, but masks this slice *inside the module*.  Callers therefore
# cannot accidentally create the command -> risk -> command self-locking loop
# that this head is intended to avoid.
COMMAND_HISTORY_SLICE = slice(30, 45)
COMMAND_MASKED_MODEL_VARIANT = "command_masked_invariant_v1"
COMMAND_MASKED_TRAILING_FEATURE_MODE = TRAILING_FEATURE_MODE_MOTION_FEEDBACK


def command_masked_risk_schema() -> dict[str, object]:
    """Return the canonical, hashable runtime contract for the masked head."""

    return {
        "version": "g1_dual_15x3_hall_command_masked_risk_v1",
        "input_dimension": INPUT_DIM,
        "runtime_input": "canonical Hall-motion actor observation",
        "actor_observation_schema": schema_for_trailing_feature_mode(
            COMMAND_MASKED_TRAILING_FEATURE_MODE
        ).to_dict(),
        "masked_input_slices": {
            "command_history": [
                COMMAND_HISTORY_SLICE.start,
                COMMAND_HISTORY_SLICE.stop,
            ]
        },
        "mask_value": 0.0,
        "trailing_feature_mode": COMMAND_MASKED_TRAILING_FEATURE_MODE,
        "offline_labels_are_not_runtime_inputs": [
            "ground_friction_mu",
            "fastbase_course_stage",
            "contact",
            "force",
            "slip_truth",
        ],
    }


def command_masked_risk_schema_sha256() -> str:
    encoded = json.dumps(
        command_masked_risk_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HallTractionRiskEstimator(nn.Module):
    """Independent risk head so risk adaptation cannot degrade gait actions."""

    feature_dim = 64

    def __init__(self, trailing_feature_mode: str = TRAILING_FEATURE_MODE_SENSOR_AGE) -> None:
        super().__init__()
        self.trailing_feature_mode = normalize_trailing_feature_mode(
            trailing_feature_mode
        )
        self.foot_encoder = LayoutHallEncoder(64)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(BASE_DIM, 96),
            nn.ELU(),
            nn.Linear(96, 64),
            nn.ELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 2 * 64 + HEALTH_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.risk_head = nn.Linear(64, 1)

    @staticmethod
    def physical_confidence(
        health: torch.Tensor,
        trailing_feature_mode: str = TRAILING_FEATURE_MODE_SENSOR_AGE,
    ) -> torch.Tensor:
        """Return packet confidence without confusing motion feedback for age.

        The static default preserves old checkpoint/test behavior.  New
        Motion-task artifacts pass ``motion_feedback`` explicitly, in which
        case the final two values are body-vy/heading and cannot be used as a
        stale-packet decay signal.
        """

        mode = normalize_trailing_feature_mode(trailing_feature_mode)
        valid = health[:, :FEET].clamp(0.0, 1.0).mean(dim=-1, keepdim=True)
        if mode == TRAILING_FEATURE_MODE_MOTION_FEEDBACK:
            return valid
        age = health[:, FEET:].clamp(0.0, 1.0).amax(dim=-1, keepdim=True)
        # Age is normalized as age_s / 0.25.  A healthy 40--50 Hz BLE stream
        # naturally has non-zero packet age, so do not penalize the first
        # 0.10 s (normalized 0.40).  Beyond that grace interval confidence
        # decays with a 0.05 s time constant.  Explicit invalid feet still
        # map to zero confidence immediately through ``valid``.
        stale_age = torch.relu(age - 0.40)
        return valid * torch.exp(-stale_age / 0.20)

    def features(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return causal Hall/proprioceptive features and packet health.

        The latent is deliberately exposed for downstream *forward* decisions
        such as a bounded recovery-action head.  It is not a force, contact or
        friction reconstruction and it never consumes simulator truth.
        """
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}")
        observation = torch.nan_to_num(observation)
        hall = observation[:, MAGNETIC_SLICE].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        period = observation[:, PERIOD_SLICE].reshape(-1, HISTORY, FEET)
        health = observation[:, VALID_SLICE.start : AGE_SLICE.stop]
        proprio = self.proprio_encoder(observation[:, :BASE_DIM])
        left = self.foot_encoder(hall[:, :, 0], period[:, :, 0])
        right = self.foot_encoder(hall[:, :, 1], period[:, :, 1])
        latent = self.fusion(torch.cat((proprio, left, right, health), dim=-1))
        return latent, health

    def learned_logit(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, health = self.features(observation)
        return self.risk_head(latent), health

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logit, health = self.learned_logit(observation)
        learned = torch.sigmoid(logit)
        confidence = self.physical_confidence(health, self.trailing_feature_mode)
        # Missing/stale Hall packets are uncertainty, so deployment risk is 1.
        return confidence * learned + (1.0 - confidence)


class BaselineInvariantHallTractionRiskEstimator(nn.Module):
    """Risk head using explicit temporal Hall statistics.

    Per-channel mean preserves load level, while standard deviation, range and
    first-to-last change describe compression/bending/shear dynamics without
    depending on one ideal absolute magnetic baseline.  These remain direct
    Bx/By/Bz-derived features; no force or friction reconstruction is used.
    """

    feature_dim = INVARIANT_FEATURE_DIM

    def __init__(
        self,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
        trailing_feature_mode: str = TRAILING_FEATURE_MODE_SENSOR_AGE,
    ) -> None:
        super().__init__()
        self.trailing_feature_mode = normalize_trailing_feature_mode(
            trailing_feature_mode
        )
        if feature_mean is None:
            feature_mean = torch.zeros(INVARIANT_FEATURE_DIM)
        if feature_scale is None:
            feature_scale = torch.ones(INVARIANT_FEATURE_DIM)
        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
        feature_scale = torch.as_tensor(feature_scale, dtype=torch.float32).reshape(-1)
        if feature_mean.numel() != INVARIANT_FEATURE_DIM:
            raise ValueError(
                f"feature_mean must contain {INVARIANT_FEATURE_DIM} values"
            )
        if feature_scale.numel() != INVARIANT_FEATURE_DIM:
            raise ValueError(
                f"feature_scale must contain {INVARIANT_FEATURE_DIM} values"
            )
        self.register_buffer("feature_mean", feature_mean.clone())
        self.register_buffer("feature_scale", feature_scale.clamp_min(0.05).clone())
        self.network = nn.Sequential(
            nn.Linear(INVARIANT_FEATURE_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def physical_confidence(
        health: torch.Tensor,
        trailing_feature_mode: str = TRAILING_FEATURE_MODE_SENSOR_AGE,
    ) -> torch.Tensor:
        return HallTractionRiskEstimator.physical_confidence(
            health, trailing_feature_mode
        )

    @staticmethod
    def raw_features(observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}")
        observation = torch.nan_to_num(observation)
        hall = observation[:, MAGNETIC_SLICE].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        hall_statistics = torch.cat(
            (
                hall.mean(dim=1),
                hall.std(dim=1),
                hall.amax(dim=1) - hall.amin(dim=1),
                hall[:, -1] - hall[:, 0],
            ),
            dim=-1,
        ).flatten(start_dim=1)
        period = observation[:, PERIOD_SLICE].reshape(-1, HISTORY, FEET)
        health = observation[:, VALID_SLICE.start : AGE_SLICE.stop]
        raw = torch.cat(
            (
                observation[:, :BASE_DIM],
                hall_statistics,
                period.mean(dim=1),
                period.std(dim=1),
                health,
            ),
            dim=-1,
        )
        return raw, health

    def features(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw, health = self.raw_features(observation)
        normalized = (raw - self.feature_mean) / self.feature_scale
        return normalized, health

    def learned_logit(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, health = self.features(observation)
        return self.network(features), health

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logit, health = self.learned_logit(observation)
        learned = torch.sigmoid(logit)
        confidence = self.physical_confidence(health, self.trailing_feature_mode)
        return confidence * learned + (1.0 - confidence)


class SlipAwareHallRiskEstimator(BaselineInvariantHallTractionRiskEstimator):
    """Invariant Hall risk head trained against *future slip* labels.

    It has the same causal, deployable input as the other Hall-risk heads:
    proprioception, the two 15-site ``Bx/By/Bz`` histories, sample timing and
    packet health.  Contact slip and falls are used only as offline simulator
    supervision while training; they are never an input to this module.

    Clipping is part of the exported model rather than a preprocessing detail.
    This prevents a post-impact proprioceptive outlier from producing an
    arbitrary neural-network extrapolation and keeps PyTorch/ONNX inference
    identical.  Missing/stale Hall data still maps to risk one through the
    inherited physical-confidence rule.
    """

    observation_clip: float = 6.0

    @classmethod
    def raw_features(cls, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = torch.nan_to_num(
            observation,
            nan=0.0,
            posinf=cls.observation_clip,
            neginf=-cls.observation_clip,
        ).clamp(-cls.observation_clip, cls.observation_clip)
        return BaselineInvariantHallTractionRiskEstimator.raw_features(clipped)


class CommandMaskedHallRiskEstimator(BaselineInvariantHallTractionRiskEstimator):
    """Hall/proprio risk head that is exactly invariant to commanded motion.

    The canonical actor ABI contains five command-history frames at
    :data:`COMMAND_HISTORY_SLICE`.  Merely asking a caller to zero those
    values is unsafe: one missed call site would let the risk output depend on
    the command that the governor is about to modify.  This implementation
    removes those values in its own graph before feature extraction, so the
    invariant also survives checkpoint loading and ONNX export.

    Friction/stage/contact/force values are intentionally absent from the
    signature.  They may supervise this model offline but can never be used
    during runtime inference.
    """

    observation_clip: float = 6.0

    def __init__(
        self,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
        trailing_feature_mode: str = COMMAND_MASKED_TRAILING_FEATURE_MODE,
    ) -> None:
        mode = normalize_trailing_feature_mode(trailing_feature_mode)
        if mode != COMMAND_MASKED_TRAILING_FEATURE_MODE:
            raise ValueError(
                f"{COMMAND_MASKED_MODEL_VARIANT} requires "
                f"trailing_feature_mode={COMMAND_MASKED_TRAILING_FEATURE_MODE!r}, "
                f"got {mode!r}"
            )
        super().__init__(feature_mean, feature_scale, trailing_feature_mode=mode)

    @classmethod
    def mask_runtime_observation(cls, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}")
        clipped = torch.nan_to_num(
            observation,
            nan=0.0,
            posinf=cls.observation_clip,
            neginf=-cls.observation_clip,
        ).clamp(-cls.observation_clip, cls.observation_clip)
        # Concatenation avoids an in-place mutation of a caller-owned tensor
        # and makes the zeroing operation explicit in exported graphs.
        return torch.cat(
            (
                clipped[:, : COMMAND_HISTORY_SLICE.start],
                torch.zeros_like(clipped[:, COMMAND_HISTORY_SLICE]),
                clipped[:, COMMAND_HISTORY_SLICE.stop :],
            ),
            dim=1,
        )

    @classmethod
    def raw_features(cls, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return BaselineInvariantHallTractionRiskEstimator.raw_features(
            cls.mask_runtime_observation(observation)
        )


def _validate_command_masked_checkpoint(payload: dict[str, object]) -> None:
    """Fail closed on every semantic field unique to the masked risk head."""

    if payload.get("input_dim") != INPUT_DIM:
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint input_dim must be {INPUT_DIM}"
        )
    if payload.get("trailing_feature_mode") != COMMAND_MASKED_TRAILING_FEATURE_MODE:
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint requires "
            f"trailing_feature_mode={COMMAND_MASKED_TRAILING_FEATURE_MODE!r}"
        )
    expected_slices = {
        "command_history": [COMMAND_HISTORY_SLICE.start, COMMAND_HISTORY_SLICE.stop]
    }
    if payload.get("masked_input_slices") != expected_slices:
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint masked_input_slices "
            f"must equal {expected_slices!r}"
        )
    expected_sha = command_masked_risk_schema_sha256()
    if payload.get("observation_schema_sha256") != expected_sha:
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint observation schema SHA mismatch"
        )
    if payload.get("schema_sha256") != expected_sha:
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint schema_sha256 mismatch"
        )
    if payload.get("observation_schema") != command_masked_risk_schema():
        raise ValueError(
            f"{COMMAND_MASKED_MODEL_VARIANT} checkpoint observation schema mismatch"
        )


def build_hall_risk_estimator(
    payload: dict[str, object] | dict[str, torch.Tensor],
) -> nn.Module:
    """Construct a compatible risk estimator from a saved checkpoint payload."""
    variant = payload.get("model_variant", "layout_encoder")
    if variant == "layout_encoder":
        model: nn.Module = HallTractionRiskEstimator(
            str(payload.get("trailing_feature_mode", TRAILING_FEATURE_MODE_SENSOR_AGE))
        )
    elif variant == "baseline_invariant":
        model = BaselineInvariantHallTractionRiskEstimator(
            trailing_feature_mode=str(
                payload.get("trailing_feature_mode", TRAILING_FEATURE_MODE_SENSOR_AGE)
            )
        )
    elif variant == "slip_aware_invariant":
        model = SlipAwareHallRiskEstimator(
            trailing_feature_mode=str(
                payload.get("trailing_feature_mode", TRAILING_FEATURE_MODE_SENSOR_AGE)
            )
        )
    elif variant == COMMAND_MASKED_MODEL_VARIANT:
        _validate_command_masked_checkpoint(payload)
        model = CommandMaskedHallRiskEstimator(
            trailing_feature_mode=str(payload["trailing_feature_mode"])
        )
    else:
        raise ValueError(f"unsupported Hall risk model_variant: {variant!r}")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError("Hall risk checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    return model
