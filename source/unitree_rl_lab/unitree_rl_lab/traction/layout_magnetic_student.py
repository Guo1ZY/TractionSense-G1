"""Geometry-aware dual-foot Hall Student for friction-adaptive G1 walking.

The privileged Teacher is intentionally absent from this module.  The policy
input contains only deployable proprioception/commands, normalized Hall
signals from two 15-site flexible soles, and sensor timing/health metadata.
No force, contact, slip truth, terrain label, or ground-friction value is an
input to the Student.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from unitree_rl_lab.sensors.hall_sensor_config import (
    DEFAULT_HALL_AXIS_YAW_DEG,
    HallFootSensorCfg,
)

from .networks import LegacyLocomotionActor
from .sensor_layout import PROVISIONAL_NORMALIZED_LAYOUT


BASE_DIM = 480
HISTORY = 15
FEET = 2
SENSORS = 15
AXES = 3
MAGNETIC_DIM = HISTORY * FEET * SENSORS * AXES
PERIOD_DIM = HISTORY * FEET
HEALTH_DIM = 4
INPUT_DIM = BASE_DIM + MAGNETIC_DIM + PERIOD_DIM + HEALTH_DIM
ACTION_DIM = 29
ACTION_OUTPUT_LIMIT = 3.0

MAGNETIC_SLICE = slice(BASE_DIM, BASE_DIM + MAGNETIC_DIM)
PERIOD_SLICE = slice(MAGNETIC_SLICE.stop, MAGNETIC_SLICE.stop + PERIOD_DIM)
VALID_SLICE = slice(PERIOD_SLICE.stop, PERIOD_SLICE.stop + FEET)
AGE_SLICE = slice(VALID_SLICE.stop, VALID_SLICE.stop + FEET)

# The last two policy channels deliberately have two supported meanings.  The
# ordinary Hall task reports source age for each foot.  The closed-loop motion
# task keeps the same 1864-D contract but replaces those two slots with body
# lateral velocity and relative-heading feedback.  Treating one as the other
# silently attenuates the Hall residual and was the source of a real
# train/evaluation semantic mismatch; keep the choice explicit in artifacts.
TrailingFeatureMode = Literal["sensor_age", "motion_feedback"]
TRAILING_FEATURE_MODE_SENSOR_AGE: TrailingFeatureMode = "sensor_age"
TRAILING_FEATURE_MODE_MOTION_FEEDBACK: TrailingFeatureMode = "motion_feedback"
TRAILING_FEATURE_MODES: tuple[TrailingFeatureMode, ...] = (
    TRAILING_FEATURE_MODE_SENSOR_AGE,
    TRAILING_FEATURE_MODE_MOTION_FEEDBACK,
)

assert INPUT_DIM == 1864
assert AGE_SLICE.stop == INPUT_DIM


@dataclass(frozen=True)
class LayoutMagneticSchema:
    version: str = "g1_dual_15x3_layout_hall_student_v1"
    input_dimension: int = INPUT_DIM
    output_dimension: int = ACTION_DIM
    policy_rate_hz: float = 50.0
    history_frames: int = HISTORY
    foot_order: tuple[str, ...] = ("left", "right")
    sensor_order: tuple[str, ...] = tuple(f"P{i:02d}" for i in range(SENSORS))
    hall_axis_order: tuple[str, ...] = ("x", "y", "z")
    hall_frame: str = (
        "per_site_hall_ic_local_xyz; P00..P14 mounting yaw is explicit and "
        "right-foot mirror sign is part of the source convention"
    )
    hall_units: str = "per-channel baseline/temperature compensated normalized Hall"
    hall_clip: tuple[float, float] = (-6.0, 6.0)
    sensor_age_units: str = "normalized source age: clip(age_s / 0.25, 0, 1)"
    flatten_order: str = (
        "legacy_proprio_480_term_major, hall_[time,left/right,P00..P14,XYZ], "
        "sample_period_[time,left/right], valid_left_right, age_left_right"
    )
    residual_limit: float = 1.0
    trailing_feature_mode: TrailingFeatureMode = TRAILING_FEATURE_MODE_SENSOR_AGE

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.trailing_feature_mode == TRAILING_FEATURE_MODE_SENSOR_AGE:
            result["trailing_feature_names"] = ["sensor_age_left", "sensor_age_right"]
            result["trailing_feature_units"] = "clip(source_age_s / 0.25, 0, 1)"
            trailing_slice_name = "age_lr"
            trailing_flatten_name = "age_left_right"
        elif self.trailing_feature_mode == TRAILING_FEATURE_MODE_MOTION_FEEDBACK:
            result["trailing_feature_names"] = [
                "body_lateral_velocity",
                "relative_heading_error",
            ]
            result["trailing_feature_units"] = "m/s, rad (both clipped by task observation)"
            trailing_slice_name = "motion_feedback"
            trailing_flatten_name = "motion_feedback_[body_vy,relative_heading]"
        else:  # Guard against a hand-edited/invalid serialized artifact.
            raise ValueError(
                f"unsupported trailing_feature_mode={self.trailing_feature_mode!r}"
            )
        # ``flatten_order`` and the final slice name used to retain their
        # sensor-age wording even for Motion actors.  That ambiguity allowed a
        # deployment YAML to feed packet age into a checkpoint trained with
        # body-vy/heading.  Serialize the actual selected semantics instead.
        result["flatten_order"] = (
            "legacy_proprio_480_term_major, "
            "hall_[time,left/right,P00..P14,XYZ], "
            "sample_period_[time,left/right], valid_left_right, "
            f"{trailing_flatten_name}"
        )
        result["sensor_positions_xy"] = [
            list(value) for value in PROVISIONAL_NORMALIZED_LAYOUT.sensor_positions_xy
        ]
        result["region_ids"] = list(PROVISIONAL_NORMALIZED_LAYOUT.region_ids)
        result["hall_axis_yaw_deg"] = list(DEFAULT_HALL_AXIS_YAW_DEG)
        result["right_foot_axis_sign"] = list(
            HallFootSensorCfg().right_hall_axis_sign
        )
        result["ble_channel_to_sensor"] = list(
            PROVISIONAL_NORMALIZED_LAYOUT.ble_channel_to_sensor_index
        )
        result["position_unit"] = PROVISIONAL_NORMALIZED_LAYOUT.position_unit
        result["layout_provisional"] = PROVISIONAL_NORMALIZED_LAYOUT.is_provisional
        result["slices"] = {
            "proprio": [0, BASE_DIM],
            "magnetic_history": [MAGNETIC_SLICE.start, MAGNETIC_SLICE.stop],
            "sample_period_history": [PERIOD_SLICE.start, PERIOD_SLICE.stop],
            "valid_lr": [VALID_SLICE.start, VALID_SLICE.stop],
            "trailing_features": [AGE_SLICE.start, AGE_SLICE.stop],
            trailing_slice_name: [AGE_SLICE.start, AGE_SLICE.stop],
        }
        result["forbidden_student_inputs"] = [
            "normal_force",
            "tangential_force",
            "calibrated_force_xyz",
            "ideal_contact_force",
            "ground_friction_mu",
            "contact_or_slip_truth",
            "terrain_label",
        ]
        return result

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


SCHEMA = LayoutMagneticSchema()


def normalize_trailing_feature_mode(value: str) -> TrailingFeatureMode:
    """Validate serialized task semantics before loading a Hall policy."""

    if value not in TRAILING_FEATURE_MODES:
        raise ValueError(
            "trailing_feature_mode must be one of "
            f"{TRAILING_FEATURE_MODES}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def schema_for_trailing_feature_mode(
    trailing_feature_mode: str,
    *,
    residual_limit: float = 1.0,
) -> LayoutMagneticSchema:
    """Return artifact metadata for the exact final-four-channel contract."""

    return LayoutMagneticSchema(
        residual_limit=float(residual_limit),
        trailing_feature_mode=normalize_trailing_feature_mode(trailing_feature_mode),
    )


def _layout_features() -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        PROVISIONAL_NORMALIZED_LAYOUT.sensor_positions_xy, dtype=torch.float32
    )
    regions = torch.nn.functional.one_hot(
        torch.tensor(PROVISIONAL_NORMALIZED_LAYOUT.region_ids), num_classes=3
    ).to(torch.float32)
    geometry = torch.cat((positions, regions), dim=-1)

    # Fixed distance graph.  It is an encoding of the audited P00..P14 layout,
    # not a learnable channel permutation.
    distance = torch.cdist(positions, positions)
    nonzero = distance[distance > 0.0]
    length_scale = torch.median(nonzero)
    adjacency = torch.exp(-torch.square(distance / length_scale.clamp_min(1.0e-6)))
    adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
    return geometry, adjacency


class LayoutHallEncoder(nn.Module):
    """Shared spatial/temporal encoder applied independently to both feet."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        geometry, adjacency = _layout_features()
        self.register_buffer("sensor_geometry", geometry)
        self.register_buffer("layout_adjacency", adjacency)
        self.point_mlp = nn.Sequential(
            nn.Linear(AXES + geometry.shape[1], 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(SENSORS * 64 + 1, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.output = nn.Sequential(nn.Linear(128, latent_dim), nn.ELU())

    def forward(self, hall: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        if hall.ndim != 4 or hall.shape[1:] != (HISTORY, SENSORS, AXES):
            raise ValueError(f"Hall history must be [B,15,15,3], got {tuple(hall.shape)}")
        geometry = self.sensor_geometry.view(1, 1, SENSORS, -1).expand(
            hall.shape[0], hall.shape[1], -1, -1
        )
        point = self.point_mlp(torch.cat((hall, geometry), dim=-1))
        neighbor = torch.einsum("ij,btjd->btid", self.layout_adjacency, point)
        frame = torch.cat((point, neighbor), dim=-1).reshape(
            hall.shape[0], HISTORY, SENSORS * 64
        )
        frame = self.frame_mlp(torch.cat((frame, period.unsqueeze(-1)), dim=-1))
        temporal = self.temporal(frame.transpose(1, 2))
        summary = torch.cat((temporal[:, :, -1], temporal.mean(dim=-1)), dim=-1)
        return self.output(summary)


class LayoutMagneticStudent(nn.Module):
    """Audited proprioceptive gait plus a confidence-gated Hall residual."""

    def __init__(
        self,
        residual_limit: float = 1.0,
        *,
        trailing_feature_mode: TrailingFeatureMode = TRAILING_FEATURE_MODE_SENSOR_AGE,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        self.trailing_feature_mode = normalize_trailing_feature_mode(
            trailing_feature_mode
        )
        self.baseline_actor = LegacyLocomotionActor(BASE_DIM)
        self.foot_encoder = LayoutHallEncoder(64)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 2 * 64 + HEALTH_DIM, 192),
            nn.ELU(),
            nn.Linear(192, 128),
            nn.ELU(),
        )
        self.residual_head = nn.Linear(128, ACTION_DIM)
        self.mu_head = nn.Linear(128, 1)
        self.slip_head = nn.Linear(128, FEET)
        self.learned_confidence_head = nn.Linear(128, 1)
        self.register_buffer(
            "residual_limit", torch.tensor(float(residual_limit), dtype=torch.float32)
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def load_baseline_checkpoint(self, checkpoint: dict[str, object]) -> None:
        state = checkpoint.get("actor_state_dict")
        if not isinstance(state, dict):
            raise ValueError("baseline checkpoint has no actor_state_dict")
        self.baseline_actor.load_state_dict(
            {key: value for key, value in state.items() if key.startswith("mlp.")},
            strict=True,
        )

    def split(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(observation.shape)}")
        observation = torch.nan_to_num(observation)
        base = observation[:, :BASE_DIM]
        hall = observation[:, MAGNETIC_SLICE].reshape(
            -1, HISTORY, FEET, SENSORS, AXES
        )
        period = observation[:, PERIOD_SLICE].reshape(-1, HISTORY, FEET)
        health = observation[:, VALID_SLICE.start : AGE_SLICE.stop]
        return base, hall, period, health

    def latent(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, hall, period, health = self.split(observation)
        base_latent = self.baseline_actor.mlp[:6](base)
        left = self.foot_encoder(hall[:, :, 0], period[:, :, 0])
        right = self.foot_encoder(hall[:, :, 1], period[:, :, 1])
        fused = self.fusion(torch.cat((base_latent, left, right, health), dim=-1))
        return fused, base_latent, health

    def physical_confidence(self, health: torch.Tensor) -> torch.Tensor:
        """Hardware-link confidence, with task-specific trailing semantics.

        Motion feedback is ordinary proprioception, not a transport-age value.
        It therefore cannot suppress the Hall residual.  In sensor-age mode,
        stale packets continue to produce a conservative decay exactly as
        before.  Invalid feet always reduce confidence in both modes.
        """

        valid = health[:, :FEET].clamp(0.0, 1.0).mean(dim=-1, keepdim=True)
        if self.trailing_feature_mode == TRAILING_FEATURE_MODE_MOTION_FEEDBACK:
            return valid
        age_normalized = (
            health[:, FEET:].clamp(0.0, 1.0).max(dim=-1, keepdim=True).values
        )
        # Deployment foot_sensor_age_lr is age_s / 0.25.  A scale of 0.40 in
        # this normalized domain is a 0.10 s physical confidence time constant.
        return valid * torch.exp(-age_normalized / 0.40)

    def all_outputs(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, base_latent, health = self.latent(observation)
        baseline = self.baseline_actor.mlp[6](base_latent)
        confidence = self.physical_confidence(health)
        raw_residual = self.residual_limit * torch.tanh(self.residual_head(fused))
        residual = confidence * raw_residual
        # The legacy actor is well behaved on the walking manifold but can
        # emit very large values after a fall.  Keep the exported policy
        # contract bounded independently of the simulator/action-manager clip.
        action = torch.clamp(
            baseline + residual, -ACTION_OUTPUT_LIMIT, ACTION_OUTPUT_LIMIT
        )
        estimated_mu = 1.30 * torch.sigmoid(self.mu_head(fused))
        slip_probability = torch.sigmoid(self.slip_head(fused))
        learned_confidence = torch.sigmoid(self.learned_confidence_head(fused))
        return (
            action,
            estimated_mu,
            slip_probability,
            confidence * learned_confidence,
            residual,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Keep deployment ONNX compatible with the existing g1_ctrl contract.
        return self.all_outputs(observation)[0]
