"""Single source of truth for action, force, observation, and history order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ACTION_DIM = 29
PHYSICS_DT_S = 0.005
DECIMATION = 4
POLICY_DT_S = PHYSICS_DT_S * DECIMATION
FORCE_FRAME = "matching_ankle_roll_link_local"
FORCE_ORDER = ("L_Fx", "L_Fy", "L_Fz", "R_Fx", "R_Fy", "R_Fz")
FORCE_UNIT = "N"
NORMALIZED_FORCE_UNIT = "F_local_N / (robot_mass_kg * 9.81)"
HISTORY_TIME_ORDER = "oldest_to_newest"

G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class ObservationTermSpec:
    """A named vector term before history expansion."""

    name: str
    dimension: int
    unit: str = "dimensionless"
    frame: str = ""
    order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation term name cannot be empty")
        if self.dimension <= 0:
            raise ValueError(f"{self.name}: dimension must be positive")
        if self.order and len(self.order) != self.dimension:
            raise ValueError(
                f"{self.name}: order has {len(self.order)} entries, "
                f"expected {self.dimension}"
            )


@dataclass(frozen=True)
class FlatHistorySchema:
    """Isaac baseline-compatible, term-major history schema.

    The flat order is ``term0[oldest..newest], term1[oldest..newest], ...``.
    Every history sample stores all components of its term contiguously.
    """

    name: str
    terms: tuple[ObservationTermSpec, ...]
    history_frames: int = 5
    flatten_order: str = "term_major_history_oldest_to_newest"
    schema_version: str = "traction_obs_v1"

    def __post_init__(self) -> None:
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate observation terms: {names}")
        if self.flatten_order != "term_major_history_oldest_to_newest":
            raise ValueError(f"unsupported flatten order: {self.flatten_order}")

    @property
    def per_frame_dimension(self) -> int:
        return sum(term.dimension for term in self.terms)

    @property
    def flat_dimension(self) -> int:
        return self.per_frame_dimension * self.history_frames

    def term_slice(self, name: str, history_index: int | None = None) -> slice:
        if not 0 <= (history_index if history_index is not None else 0) < self.history_frames:
            raise IndexError(history_index)
        offset = 0
        for term in self.terms:
            width = term.dimension * self.history_frames
            if term.name == name:
                if history_index is None:
                    return slice(offset, offset + width)
                start = offset + history_index * term.dimension
                return slice(start, start + term.dimension)
            offset += width
        raise KeyError(name)


_PROPRIO_TERMS = (
    ObservationTermSpec("base_ang_vel", 3, "rad/s", "base"),
    ObservationTermSpec("projected_gravity", 3, "dimensionless", "base"),
    ObservationTermSpec(
        "velocity_commands",
        3,
        "m/s,m/s,rad/s",
        "base_yaw",
        ("vx", "vy", "yaw_rate"),
    ),
    ObservationTermSpec("joint_pos_rel", ACTION_DIM, "rad", order=G1_29DOF_JOINT_ORDER),
    ObservationTermSpec("joint_vel_rel", ACTION_DIM, "rad/s", order=G1_29DOF_JOINT_ORDER),
    ObservationTermSpec("last_action", ACTION_DIM, "policy_action", order=G1_29DOF_JOINT_ORDER),
)
_RAW_FORCE_TERM = ObservationTermSpec(
    "raw_foot_force",
    6,
    NORMALIZED_FORCE_UNIT,
    FORCE_FRAME,
    FORCE_ORDER,
)

PRIVILEGED_TRACTION_SCHEMA = FlatHistorySchema(
    name="g1_29dof_privileged_traction",
    history_frames=1,
    terms=(
        ObservationTermSpec("current_proprio", 96, "policy_normalized"),
        ObservationTermSpec(
            "ground_friction_mu",
            2,
            "dimensionless",
            "",
            ("left_mu", "right_mu"),
        ),
        ObservationTermSpec(
            "ideal_foot_force",
            6,
            "N",
            FORCE_FRAME,
            FORCE_ORDER,
        ),
        ObservationTermSpec("force_normal", 2, "N"),
        ObservationTermSpec("force_tangent", 2, "N"),
        ObservationTermSpec("friction_utilization", 2),
        ObservationTermSpec("contact", 2, "bool_as_float"),
        ObservationTermSpec("foot_tangent_velocity", 4, "m/s"),
        ObservationTermSpec("slip_speed", 2, "m/s"),
        ObservationTermSpec("slip_label", 2, "bool_as_float"),
        ObservationTermSpec("base_linear_velocity", 3, "m/s", "base"),
        ObservationTermSpec("terrain_contact_randomization", 4),
        ObservationTermSpec("dynamics_randomization", 8),
    ),
)


def legacy_actor_schema(*, include_force: bool) -> FlatHistorySchema:
    """Return the audited five-frame actor schema (480 or 510 elements)."""
    terms = _PROPRIO_TERMS + ((_RAW_FORCE_TERM,) if include_force else ())
    return FlatHistorySchema(
        name="g1_29dof_actor_raw_force" if include_force else "g1_29dof_actor_proprio",
        terms=terms,
    )


def legacy_critic_schema(*, include_force: bool) -> FlatHistorySchema:
    """Return the audited five-frame critic schema (495 or 525 elements)."""
    terms = (
        ObservationTermSpec("base_lin_vel", 3, "m/s", "base"),
        *_PROPRIO_TERMS,
    ) + ((_RAW_FORCE_TERM,) if include_force else ())
    return FlatHistorySchema(
        name="g1_29dof_critic_raw_force" if include_force else "g1_29dof_critic_proprio",
        terms=terms,
    )


def old_to_new_flat_index(
    old: FlatHistorySchema,
    new: FlatHistorySchema,
) -> np.ndarray:
    """Map every old flat column to its semantic location in ``new``.

    Mapping is term- and history-aware; it does not assume new observations
    were appended to the flattened vector.
    """
    if old.history_frames != new.history_frames:
        raise ValueError(
            "history frame counts differ; temporal interpolation is not a "
            "checkpoint column-migration operation"
        )
    new_terms = {term.name: term for term in new.terms}
    mapping = np.empty(old.flat_dimension, dtype=np.int64)
    old_cursor = 0
    for old_term in old.terms:
        if old_term.name not in new_terms:
            raise KeyError(f"new schema is missing old term {old_term.name!r}")
        new_term = new_terms[old_term.name]
        if new_term.dimension != old_term.dimension:
            raise ValueError(
                f"term {old_term.name!r} changed dimension "
                f"{old_term.dimension} -> {new_term.dimension}"
            )
        for history_index in range(old.history_frames):
            old_slice = slice(old_cursor, old_cursor + old_term.dimension)
            new_slice = new.term_slice(old_term.name, history_index)
            mapping[old_slice] = np.arange(new_slice.start, new_slice.stop)
            old_cursor += old_term.dimension
    if len(np.unique(mapping)) != old.flat_dimension:
        raise RuntimeError("old-to-new observation mapping is not one-to-one")
    return mapping


@dataclass(frozen=True)
class TemporalStudentFrameSchema:
    """Deployable Student input before time-major history flattening."""

    terms: tuple[ObservationTermSpec, ...]
    policy_dt_s: float = POLICY_DT_S
    history_seconds: float = 0.30
    flatten_order: str = "time_major_oldest_to_newest_then_term_order"
    schema_version: str = "traction_temporal_student_v1"

    @property
    def history_frames(self) -> int:
        return max(1, int(math.ceil(self.history_seconds / self.policy_dt_s)))

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
            force_order=FORCE_ORDER,
            force_frame=FORCE_FRAME,
            force_unit=FORCE_UNIT,
            normalized_force_unit=NORMALIZED_FORCE_UNIT,
            joint_order=G1_29DOF_JOINT_ORDER,
        )
        return result

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


TEMPORAL_STUDENT_FRAME_SCHEMA = TemporalStudentFrameSchema(
    terms=(
        ObservationTermSpec("base_ang_vel", 3, "rad/s", "base"),
        ObservationTermSpec("projected_gravity", 3, "dimensionless", "base"),
        ObservationTermSpec("joint_pos_rel", ACTION_DIM, "rad", order=G1_29DOF_JOINT_ORDER),
        ObservationTermSpec("joint_vel_rel", ACTION_DIM, "rad/s", order=G1_29DOF_JOINT_ORDER),
        ObservationTermSpec("previous_action", ACTION_DIM, "policy_action", order=G1_29DOF_JOINT_ORDER),
        ObservationTermSpec(
            "raw_command",
            3,
            "m/s,m/s,rad/s",
            "base_yaw",
            ("vx", "vy", "yaw_rate"),
        ),
        ObservationTermSpec(
            "observed_foot_force",
            6,
            NORMALIZED_FORCE_UNIT,
            FORCE_FRAME,
            FORCE_ORDER,
        ),
        ObservationTermSpec(
            "foot_force_valid",
            2,
            "bool_as_float",
            "",
            ("left_valid", "right_valid"),
        ),
        ObservationTermSpec(
            "foot_force_age",
            2,
            "s",
            "",
            ("left_age", "right_age"),
        ),
    )
)


def concatenate_terms(
    schema: TemporalStudentFrameSchema,
    values: dict[str, np.ndarray],
) -> np.ndarray:
    """Concatenate one or more batches according to the canonical term order."""
    arrays = []
    prefix: tuple[int, ...] | None = None
    for term in schema.terms:
        if term.name not in values:
            raise KeyError(f"missing observation term {term.name!r}")
        value = np.asarray(values[term.name])
        if value.shape[-1] != term.dimension:
            raise ValueError(
                f"{term.name}: last dimension {value.shape[-1]}, "
                f"expected {term.dimension}"
            )
        if prefix is None:
            prefix = value.shape[:-1]
        elif value.shape[:-1] != prefix:
            raise ValueError(
                f"{term.name}: prefix shape {value.shape[:-1]}, expected {prefix}"
            )
        arrays.append(value)
    return np.concatenate(arrays, axis=-1)


def validate_force_vector(force: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite canonical six-axis force vector."""
    value = np.asarray(force, dtype=np.float32)
    if value.shape != (6,):
        raise ValueError(f"force must have shape (6,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("force contains NaN or Inf")
    return value


def terms_by_name(terms: Iterable[ObservationTermSpec]) -> dict[str, ObservationTermSpec]:
    return {term.name: term for term in terms}
