"""Frozen PyTorch reconstruction of the validated 1864-D speedboost Teacher.

The original policy was exported as a composite ONNX graph and its referenced
``fast`` and ``stable`` training checkpoints are no longer available.  This
module mirrors the source architecture that produced that graph so an offline
converter can recover a portable PyTorch state dict from the ONNX initializers.

The Teacher consumes the historical ``sensor_age`` interpretation of the last
two observation channels.  Current spatial Hall policies use those channels
for motion feedback, so callers must use :func:`adapt_teacher_observation`
before invoking the Teacher.  That adapter never changes the deployable Actor
observation in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn


BASE_DIM = 480
HISTORY = 15
FEET = 2
SENSORS = 15
AXES = 3
MAGNETIC_DIM = HISTORY * FEET * SENSORS * AXES
PERIOD_DIM = HISTORY * FEET
HEALTH_DIM = 4
INPUT_DIM = BASE_DIM + MAGNETIC_DIM + PERIOD_DIM + HEALTH_DIM
FUSED_DIM = BASE_DIM + 32 * FEET + HEALTH_DIM
OUTPUT_DIM = 29

VALID_SLICE = slice(1860, 1862)
TRAILING_SLICE = slice(1862, 1864)
COMMAND_VX_INDICES = (30, 33, 36, 39, 42)
LATERAL_JOINTS = (4, 6, 8, 12, 13, 14, 15, 22)
LATERAL_ARM_JOINTS = (5, 17, 18, 19, 23, 26, 27, 28)

CHECKPOINT_FORMAT = "unitree_rl_lab.frozen_speedboost_teacher"
CHECKPOINT_VERSION = 1
KNOWN_SPEEDBOOST112_SHA256 = "9a26808aef32a1aa6476e5df36fffe78539111cad3b7a389a896ec8a54d57ba1"

TrailingFeatureMode = Literal["sensor_age", "motion_feedback"]


@dataclass(frozen=True)
class SpeedBoostTeacherConfig:
    """Scalar configuration recovered from the speedboost112 ONNX graph."""

    residual_center: float = 0.06
    residual_sharpness: float = 150.0
    evidence_center: float = 0.15
    evidence_sharpness: float = 50.0
    boost_factor: float = 1.12
    traction_center: float = 0.65
    traction_sharpness: float = 10.0
    command_center: float = 0.70
    command_sharpness: float = 15.0
    mu_max: float = 1.30
    stable_uses_boosted_command: bool = True

    def validate(self) -> None:
        if not 1.0 <= self.boost_factor <= 1.25:
            raise ValueError(f"boost_factor must be in [1.0, 1.25], got {self.boost_factor}")
        positive = {
            "residual_sharpness": self.residual_sharpness,
            "evidence_sharpness": self.evidence_sharpness,
            "traction_sharpness": self.traction_sharpness,
            "command_sharpness": self.command_sharpness,
            "mu_max": self.mu_max,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")


class SharedFootEncoder(nn.Module):
    """Original shared 15-site temporal Hall encoder."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        point_dim = 16
        self.point_mlp = nn.Sequential(
            nn.Linear(AXES, point_dim),
            nn.ELU(),
            nn.Linear(point_dim, point_dim),
            nn.ELU(),
        )
        self.sensor_embedding = nn.Parameter(torch.zeros(SENSORS, point_dim))
        self.frame_mlp = nn.Sequential(
            nn.Linear(SENSORS * point_dim + 1, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.output = nn.Sequential(nn.Linear(32, latent_dim), nn.ELU())

    def forward(self, magnetic: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        if magnetic.ndim != 4 or tuple(magnetic.shape[1:]) != (HISTORY, SENSORS, AXES):
            raise ValueError(
                f"magnetic history must be [batch,{HISTORY},{SENSORS},{AXES}], "
                f"got {tuple(magnetic.shape)}"
            )
        if period.ndim != 2 or tuple(period.shape[1:]) != (HISTORY,):
            raise ValueError(f"period history must be [batch,{HISTORY}], got {tuple(period.shape)}")
        point = self.point_mlp(magnetic)
        point = point + self.sensor_embedding.view(1, 1, SENSORS, -1)
        frame = point.reshape(magnetic.shape[0], HISTORY, -1)
        frame = self.frame_mlp(torch.cat((frame, period.unsqueeze(-1)), dim=-1))
        temporal = self.temporal(frame.transpose(1, 2))
        return self.output(temporal[:, :, -1])


class SharedMagneticPolicy(nn.Module):
    """One branch of the original composite policy.

    Only the fast branch needs ``mu_head`` in the exported graph.  The original
    source also had per-foot auxiliary heads, but ONNX pruned them because the
    composite policy never consumed those values.
    """

    def __init__(self, *, with_mu_head: bool = False, mu_max: float = 1.30) -> None:
        super().__init__()
        self.foot_encoder = SharedFootEncoder(32)
        self.actor = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, OUTPUT_DIM),
        )
        self.mu_head = nn.Linear(128, 1) if with_mu_head else None
        self.mu_max = float(mu_max)

    def split(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [batch,{INPUT_DIM}], got {tuple(observation.shape)}")
        base = observation[:, :BASE_DIM]
        magnetic = observation[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        period_start = BASE_DIM + MAGNETIC_DIM
        period = observation[:, period_start : period_start + PERIOD_DIM].reshape(-1, HISTORY, FEET)
        health = observation[:, -HEALTH_DIM:]
        return base, magnetic, period, health

    def encode(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, magnetic, period, health = self.split(observation)
        left = self.foot_encoder(magnetic[:, :, 0], period[:, :, 0])
        right = self.foot_encoder(magnetic[:, :, 1], period[:, :, 1])
        return torch.cat((base, left, right, health), dim=-1), left, right

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.encode(observation)
        return self.actor(fused)

    def predict_mu(self, observation: torch.Tensor) -> torch.Tensor:
        if self.mu_head is None:
            raise RuntimeError("this SharedMagneticPolicy branch has no mu_head")
        fused, _, _ = self.encode(observation)
        latent = self.actor[:6](fused)
        return self.mu_max * torch.sigmoid(self.mu_head(latent)).squeeze(-1)


def _default_profile() -> torch.Tensor:
    profile = torch.tensor(
        (
            0.70,
            0.76,
            0.70,
            0.76,
            0.82,
            0.76,
            0.82,
            0.88,
            0.82,
            0.88,
            0.94,
            0.88,
            0.94,
            1.00,
            0.94,
        ),
        dtype=torch.float32,
    )
    return (profile / profile.mean()).reshape(1, 1, 1, SENSORS, 1)


def _default_stable_weight() -> torch.Tensor:
    weight = torch.zeros(OUTPUT_DIM, dtype=torch.float32)
    weight[list(LATERAL_JOINTS)] = 0.75
    weight[list(LATERAL_ARM_JOINTS)] = 1.0
    return weight


def _default_command_mask() -> torch.Tensor:
    mask = torch.zeros(INPUT_DIM, dtype=torch.float32)
    mask[list(COMMAND_VX_INDICES)] = 1.0
    return mask


class FrozenSpeedBoostTeacher(nn.Module):
    """Exact PyTorch topology of the speedboost112 composite ONNX policy."""

    def __init__(self, config: SpeedBoostTeacherConfig = SpeedBoostTeacherConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.safe = SharedMagneticPolicy(mu_max=config.mu_max)
        self.fast = SharedMagneticPolicy(with_mu_head=True, mu_max=config.mu_max)
        self.stable = SharedMagneticPolicy(mu_max=config.mu_max)
        self.register_buffer("profile", _default_profile())
        self.register_buffer("stable_weight", _default_stable_weight())
        self.register_buffer("command_mask", _default_command_mask())
        self.freeze()

    def freeze(self) -> FrozenSpeedBoostTeacher:
        super().train(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> FrozenSpeedBoostTeacher:
        """Keep the Teacher deterministic even when its parent algorithm trains."""

        del mode
        return self.freeze()

    def confidence(self, observation: torch.Tensor) -> torch.Tensor:
        magnetic = observation[:, BASE_DIM : BASE_DIM + MAGNETIC_DIM].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        normalized = magnetic / self.profile
        sensor_mean = normalized.mean(dim=3, keepdim=True)
        residual = torch.abs(normalized - sensor_mean).mean(dim=(1, 2, 3, 4))
        evidence = torch.abs(sensor_mean).mean(dim=(1, 2, 3, 4))
        score = residual / (evidence + 0.05)
        calibration = torch.sigmoid(
            (self.config.residual_center - score) * self.config.residual_sharpness
        )
        has_evidence = torch.sigmoid(
            (evidence - self.config.evidence_center) * self.config.evidence_sharpness
        )
        valid = observation[:, VALID_SLICE].amin(dim=1)
        normalized_age = observation[:, TRAILING_SLICE].amax(dim=1)
        fresh = 1.0 - normalized_age.square()
        return (calibration * has_evidence * valid * fresh).clamp(0.0, 1.0)

    def boost_gate(self, observation: torch.Tensor) -> torch.Tensor:
        predicted_mu = self.fast.predict_mu(observation)
        traction = torch.sigmoid(
            (predicted_mu - self.config.traction_center) * self.config.traction_sharpness
        )
        command_vx = observation[:, list(COMMAND_VX_INDICES)].mean(dim=1)
        high_command = torch.sigmoid(
            (command_vx - self.config.command_center) * self.config.command_sharpness
        )
        return (self.confidence(observation) * traction * high_command).clamp(0.0, 1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [batch,{INPUT_DIM}], got {tuple(observation.shape)}")
        calibration_confidence = self.confidence(observation)
        boost = ((self.config.boost_factor - 1.0) * self.boost_gate(observation)).unsqueeze(1)
        boosted_observation = observation * (1.0 + boost * self.command_mask.unsqueeze(0))
        fast_action = self.fast(boosted_observation)
        stable_observation = boosted_observation if self.config.stable_uses_boosted_command else observation
        stable_action = self.stable(stable_observation)
        corrected_action = torch.lerp(fast_action, stable_action, self.stable_weight)
        safe_action = self.safe(observation)
        return torch.lerp(safe_action, corrected_action, calibration_confidence.unsqueeze(1))


def adapt_teacher_observation(
    policy_observation: torch.Tensor,
    *,
    policy_trailing_feature_mode: TrailingFeatureMode,
    sensor_age_lr: torch.Tensor | None = None,
    assume_fresh_if_motion_feedback: bool = False,
) -> torch.Tensor:
    """Return a Teacher-compatible copy without mutating the Actor observation.

    The speedboost112 Teacher interprets columns 1862:1864 as normalized Hall
    packet ages.  For a motion-feedback Actor, callers must supply actual ages
    from a training-only observation group or explicitly opt into the safer
    ``age=0`` compatibility assumption.
    """

    if policy_observation.ndim != 2 or policy_observation.shape[1] != INPUT_DIM:
        raise ValueError(f"expected [batch,{INPUT_DIM}], got {tuple(policy_observation.shape)}")
    if policy_trailing_feature_mode not in ("sensor_age", "motion_feedback"):
        raise ValueError(f"unsupported trailing feature mode {policy_trailing_feature_mode!r}")
    if policy_trailing_feature_mode == "sensor_age":
        if sensor_age_lr is not None:
            raise ValueError("sensor_age_lr must be omitted when the policy observation already contains sensor ages")
        return policy_observation
    if sensor_age_lr is None:
        if not assume_fresh_if_motion_feedback:
            raise ValueError(
                "motion_feedback observations require training-only sensor_age_lr; "
                "set assume_fresh_if_motion_feedback=True only for explicit compatibility testing"
            )
        sensor_age_lr = torch.zeros(
            policy_observation.shape[0], FEET, device=policy_observation.device, dtype=policy_observation.dtype
        )
    if sensor_age_lr.shape != (policy_observation.shape[0], FEET):
        raise ValueError(
            f"sensor_age_lr must have shape {(policy_observation.shape[0], FEET)}, got {tuple(sensor_age_lr.shape)}"
        )
    adapted = policy_observation.clone()
    adapted[:, TRAILING_SLICE] = sensor_age_lr.to(device=adapted.device, dtype=adapted.dtype).clamp(0.0, 1.0)
    return adapted


def make_frozen_teacher_payload(
    model: FrozenSpeedBoostTeacher,
    *,
    source_onnx_sha256: str,
    source_graph: Mapping[str, Any],
    parity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the tensor-only, ``weights_only=True`` compatible artifact."""

    if len(source_onnx_sha256) != 64:
        raise ValueError("source_onnx_sha256 must contain 64 hexadecimal characters")
    int(source_onnx_sha256, 16)
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "teacher_trailing_feature_mode": "sensor_age",
        "source_onnx_sha256": source_onnx_sha256.lower(),
        "source_graph": dict(source_graph),
        "parity": dict(parity),
        "provenance": dict(provenance or {}),
        "config": asdict(model.config),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


def save_frozen_speedboost_teacher(
    path: str | Path,
    model: FrozenSpeedBoostTeacher,
    *,
    source_onnx_sha256: str,
    source_graph: Mapping[str, Any],
    parity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        make_frozen_teacher_payload(
            model,
            source_onnx_sha256=source_onnx_sha256,
            source_graph=source_graph,
            parity=parity,
            provenance=provenance,
        ),
        destination,
    )


def load_frozen_speedboost_teacher(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_source_sha256: str | None = KNOWN_SPEEDBOOST112_SHA256,
) -> FrozenSpeedBoostTeacher:
    """Load and freeze a converted Teacher, rejecting schema/hash drift."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("frozen Teacher checkpoint must contain a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported frozen Teacher format {payload.get('format')!r} version {payload.get('format_version')!r}"
        )
    if payload.get("input_dim") != INPUT_DIM or payload.get("output_dim") != OUTPUT_DIM:
        raise ValueError(
            f"frozen Teacher schema must be {INPUT_DIM}->{OUTPUT_DIM}, got "
            f"{payload.get('input_dim')}->{payload.get('output_dim')}"
        )
    if payload.get("teacher_trailing_feature_mode") != "sensor_age":
        raise ValueError("speedboost Teacher artifact must declare sensor_age trailing channels")
    source_sha256 = str(payload.get("source_onnx_sha256", "")).lower()
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256.lower():
        raise ValueError(
            f"source ONNX SHA256 mismatch: expected {expected_source_sha256.lower()}, got {source_sha256}"
        )
    config_value = payload.get("config")
    if not isinstance(config_value, dict):
        raise TypeError("frozen Teacher checkpoint is missing its config dictionary")
    model = FrozenSpeedBoostTeacher(SpeedBoostTeacherConfig(**config_value))
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("frozen Teacher checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite tensor in frozen Teacher state: {name}")
    return model.to(device).freeze()


assert INPUT_DIM == 1864
assert OUTPUT_DIM == 29

