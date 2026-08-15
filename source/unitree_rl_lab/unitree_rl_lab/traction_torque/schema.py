"""Versioned deployment schema for torque-based traction policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from unitree_rl_lab.traction.schema import (
    ACTION_DIM,
    FORCE_FRAME,
    FORCE_ORDER,
    FORCE_UNIT,
    G1_29DOF_JOINT_ORDER,
    POLICY_DT_S,
    ObservationTermSpec,
)


TORQUE_TRACTION_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
TORQUE_TRACTION_JOINT_INDICES = tuple(
    G1_29DOF_JOINT_ORDER.index(name) for name in TORQUE_TRACTION_JOINT_ORDER
)
LEFT_LEG_ACTION_INDICES = TORQUE_TRACTION_JOINT_INDICES[:6]
RIGHT_LEG_ACTION_INDICES = TORQUE_TRACTION_JOINT_INDICES[6:]


@dataclass(frozen=True)
class EstimatedDualFootForce:
    """Common Isaac/MuJoCo/offline/deployment analytical-force result."""

    timestamp: float
    left_force_xyz: np.ndarray
    right_force_xyz: np.ndarray
    left_contact_probability: float
    right_contact_probability: float
    left_confidence: float
    right_confidence: float
    left_residual_norm: float
    right_residual_norm: float
    left_condition_score: float
    right_condition_score: float

    def __post_init__(self) -> None:
        for name in ("left_force_xyz", "right_force_xyz"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite shape-(3,) vector")
            object.__setattr__(self, name, value)
        for name in (
            "left_contact_probability",
            "right_contact_probability",
            "left_confidence",
            "right_confidence",
            "left_condition_score",
            "right_condition_score",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0,1]")
        for name in ("left_residual_norm", "right_residual_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def force_vector(self) -> np.ndarray:
        return np.concatenate((self.left_force_xyz, self.right_force_xyz))


@dataclass(frozen=True)
class TorqueTractionHistorySchema:
    terms: tuple[ObservationTermSpec, ...]
    policy_dt_s: float = POLICY_DT_S
    history_seconds: float = 0.30
    flatten_order: str = "time_major_oldest_to_newest_then_term_order"
    schema_version: str = "torque_traction_student_v1"

    def __post_init__(self) -> None:
        if self.policy_dt_s <= 0.0 or self.history_seconds <= 0.0:
            raise ValueError("policy dt and history must be positive")
        if len({term.name for term in self.terms}) != len(self.terms):
            raise ValueError("duplicate observation term")

    @property
    def history_frames(self) -> int:
        return int(round(self.history_seconds / self.policy_dt_s))

    @property
    def frame_dimension(self) -> int:
        return sum(term.dimension for term in self.terms)

    @property
    def flat_dimension(self) -> int:
        return self.history_frames * self.frame_dimension

    def term_slice(self, name: str) -> slice:
        offset = 0
        for term in self.terms:
            if term.name == name:
                return slice(offset, offset + term.dimension)
            offset += term.dimension
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            history_frames=self.history_frames,
            frame_dimension=self.frame_dimension,
            flat_dimension=self.flat_dimension,
            action_dimension=ACTION_DIM,
            action_joint_order=G1_29DOF_JOINT_ORDER,
            leg_torque_order=TORQUE_TRACTION_JOINT_ORDER,
            force_order=FORCE_ORDER,
            force_frame=FORCE_FRAME,
            force_unit=FORCE_UNIT,
            privileged_terms_forbidden=(
                "contact_sensor_force",
                "ground_truth_contact_force",
                "ground_friction_mu",
                "terrain_friction_label",
                "privileged_slip_label",
                "future_friction",
            ),
        )
        return result

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


TORQUE_TRACTION_FRAME_SCHEMA = TorqueTractionHistorySchema(
    terms=(
        ObservationTermSpec("base_ang_vel", 3, "rad/s_scaled_0.2", "base"),
        ObservationTermSpec("projected_gravity", 3, "dimensionless", "base"),
        ObservationTermSpec(
            "command",
            3,
            "m/s,m/s,rad/s",
            "base_yaw",
            ("vx", "vy", "yaw_rate"),
        ),
        ObservationTermSpec(
            "joint_pos_rel", 29, "rad", order=G1_29DOF_JOINT_ORDER
        ),
        ObservationTermSpec(
            "joint_vel", 29, "rad/s_scaled_0.05", order=G1_29DOF_JOINT_ORDER
        ),
        ObservationTermSpec(
            "previous_action", 29, "policy_action", order=G1_29DOF_JOINT_ORDER
        ),
        ObservationTermSpec(
            "leg_joint_tau_est",
            12,
            "Nm_scaled_by_effort_limit",
            order=TORQUE_TRACTION_JOINT_ORDER,
        ),
        ObservationTermSpec(
            "estimated_foot_force",
            6,
            "F_hat_N/(robot_mass_kg*9.81)",
            FORCE_FRAME,
            FORCE_ORDER,
        ),
        ObservationTermSpec(
            "contact_probability",
            2,
            "probability",
            order=("left", "right"),
        ),
        ObservationTermSpec(
            "force_estimator_confidence",
            2,
            "probability",
            order=("left", "right"),
        ),
        ObservationTermSpec(
            "foot_planar_velocity",
            4,
            "m/s",
            "world_ground_tangent",
            ("L_vx", "L_vy", "R_vx", "R_vy"),
        ),
        ObservationTermSpec(
            "imu_linear_acceleration",
            3,
            "m/s2_scaled_by_9.81",
            "base",
        ),
    )
)

if TORQUE_TRACTION_FRAME_SCHEMA.frame_dimension != 125:
    raise RuntimeError("torque traction frame must remain 125-D")
if TORQUE_TRACTION_FRAME_SCHEMA.flat_dimension != 1875:
    raise RuntimeError("torque traction history must remain 1875-D")


def concatenate_frame(values: dict[str, np.ndarray]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    prefix: tuple[int, ...] | None = None
    for term in TORQUE_TRACTION_FRAME_SCHEMA.terms:
        if term.name not in values:
            raise KeyError(term.name)
        value = np.asarray(values[term.name], dtype=np.float32)
        if value.shape[-1] != term.dimension:
            raise ValueError(
                f"{term.name} has dimension {value.shape[-1]}, expected {term.dimension}"
            )
        if prefix is None:
            prefix = value.shape[:-1]
        elif prefix != value.shape[:-1]:
            raise ValueError(f"{term.name} batch prefix changed")
        arrays.append(value)
    result = np.concatenate(arrays, axis=-1)
    if result.shape[-1] != TORQUE_TRACTION_FRAME_SCHEMA.frame_dimension:
        raise RuntimeError("concatenated frame dimension changed")
    return result
