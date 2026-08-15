"""Validated HighEnd recovery-state banks and actor-history restoration.

The recovery curriculum is deliberately isolated from ordinary locomotion
resets.  A mechanically restored walking pose is not a valid Markov state for
the 1864-D Hall actor unless its proprioceptive, action, command and magnetic
histories are restored as well.  This module owns the versioned, pickle-free
bank ABI used by the reset event and the specialised recovery environment.

Nothing in this file converts Hall measurements to forces.  The actor history
contains only the original Hall ``Bx/By/Bz`` response and packet metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


SCHEMA_VERSION = "high_end_recovery_state_bank.v2"
TRAINING_ROLE = "training_high_end_state_bank"
VALIDATION_ROLE = "validation_high_end_state_bank"
DEVELOPMENT_ROLE = "development_smoke_not_train"
LOCKED_ACCEPTANCE_SEEDS = (500,)

POLICY_DIM = 1864
ACTION_DIM = 29
NUM_HALL_SENSORS = 15
HALL_HISTORY_LENGTH = 15
PROPRIO_HISTORY_LENGTH = 5

# Audited term-major Motion policy layout.  These slices must remain aligned
# with FootTractionMagneticMotionObservationsCfg.PolicyCfg.
POLICY_HISTORY_LAYOUT: dict[str, tuple[slice, int, int]] = {
    "base_ang_vel": (slice(0, 15), PROPRIO_HISTORY_LENGTH, 3),
    "projected_gravity": (slice(15, 30), PROPRIO_HISTORY_LENGTH, 3),
    "velocity_commands": (slice(30, 45), PROPRIO_HISTORY_LENGTH, 3),
    "joint_pos_rel": (slice(45, 190), PROPRIO_HISTORY_LENGTH, ACTION_DIM),
    "joint_vel_rel": (slice(190, 335), PROPRIO_HISTORY_LENGTH, ACTION_DIM),
    "last_action": (slice(335, 480), PROPRIO_HISTORY_LENGTH, ACTION_DIM),
    "foot_magnetic_array": (slice(480, 1830), HALL_HISTORY_LENGTH, 90),
    "foot_sample_period_lr": (slice(1830, 1860), HALL_HISTORY_LENGTH, 2),
}
VALID_SLICE = slice(1860, 1862)
MOTION_SLICE = slice(1862, 1864)


@dataclass(frozen=True)
class HighEndStateBank:
    """Device tensors plus immutable provenance for one validated bank."""

    path: str
    metadata: dict[str, object]
    arrays: dict[str, torch.Tensor]

    @property
    def sample_count(self) -> int:
        return int(self.arrays["observation"].shape[0])


_REQUIRED_WIDTHS: dict[str, tuple[int, ...]] = {
    "root_pose_local": (7,),
    "root_velocity": (6,),
    "joint_pos": (ACTION_DIM,),
    "joint_vel": (ACTION_DIM,),
    "observation": (POLICY_DIM,),
    "motion_feedback_initial_yaw": (),
    "straight_heading_reference_xy": (2,),
    "straight_track_origin_local_xy": (2,),
    "straight_track_lateral_axis": (2,),
    "hall_local_deformation": (2, NUM_HALL_SENSORS, 6),
    "hall_signal_filtered_absolute": (2, NUM_HALL_SENSORS, 3),
    "hall_signal_processed": (2, NUM_HALL_SENSORS, 3),
    "hall_signal_baseline": (2, NUM_HALL_SENSORS, 3),
    "hall_signal_drift": (2, NUM_HALL_SENSORS, 3),
    "hall_policy_gain": (2, NUM_HALL_SENSORS, 3),
    "hall_policy_cross_axis": (2, NUM_HALL_SENSORS, 3, 3),
    "hall_policy_zero_residual": (2, NUM_HALL_SENSORS, 3),
    "hall_policy_channel_keep": (2, NUM_HALL_SENSORS, 1),
    "hall_policy_foot_keep": (2, 1, 1),
    "hall_policy_delay_steps": (2,),
    "hall_reported_sample_period": (2,),
    "source_seed": (),
    "source_env_id": (),
    "source_rollout_step": (),
    "time_to_fall_s": (),
    "state_kind": (),
}


def _metadata(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    if "metadata_json" not in payload.files:
        raise ValueError(f"{path}: missing metadata_json")
    raw = np.asarray(payload["metadata_json"])
    if raw.shape != ():
        raise ValueError(f"{path}: metadata_json must be a scalar string")
    try:
        value = json.loads(str(raw.item()))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid metadata_json") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return value


def _integer_seed_set(value: object, *, field: str, path: Path) -> set[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: metadata {field} must be a non-empty list")
    result: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{path}: metadata {field} must contain integers")
        result.add(int(item))
    return result


def load_high_end_state_bank(
    path: str | Path,
    *,
    device: str | torch.device,
    allowed_roles: Iterable[str] = (TRAINING_ROLE,),
    locked_acceptance_seeds: Iterable[int] = LOCKED_ACCEPTANCE_SEEDS,
) -> HighEndStateBank:
    """Load a V2 bank and fail closed on schema, provenance or ABI drift."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"HighEnd state bank not found: {resolved}")
    allowed = set(allowed_roles)
    locked = {int(seed) for seed in locked_acceptance_seeds}
    with np.load(resolved, allow_pickle=False) as payload:
        metadata = _metadata(payload, resolved)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{resolved}: expected schema_version={SCHEMA_VERSION!r}, "
                f"got {metadata.get('schema_version')!r}"
            )
        role = metadata.get("dataset_role")
        if role not in allowed:
            raise ValueError(
                f"{resolved}: dataset_role={role!r} is not allowed; "
                f"expected one of {sorted(allowed)!r}"
            )
        source_seeds = _integer_seed_set(
            metadata.get("source_seeds"), field="source_seeds", path=resolved
        )
        leaked = source_seeds & locked
        if leaked:
            raise ValueError(
                f"{resolved}: locked acceptance seed leakage: {sorted(leaked)}"
            )
        declared_locked = _integer_seed_set(
            metadata.get("excluded_locked_seeds"),
            field="excluded_locked_seeds",
            path=resolved,
        )
        if not locked.issubset(declared_locked):
            raise ValueError(
                f"{resolved}: metadata does not declare all locked seeds as excluded"
            )

        arrays: dict[str, np.ndarray] = {}
        count: int | None = None
        for name, trailing_shape in _REQUIRED_WIDTHS.items():
            if name not in payload.files:
                raise ValueError(f"{resolved}: missing required field {name!r}")
            value = np.asarray(payload[name])
            if value.ndim != len(trailing_shape) + 1 or value.shape[1:] != trailing_shape:
                raise ValueError(
                    f"{resolved}: {name} must have shape [N{''.join(',' + str(v) for v in trailing_shape)}], "
                    f"got {value.shape}"
                )
            count = value.shape[0] if count is None else count
            if value.shape[0] != count:
                raise ValueError(f"{resolved}: state-bank arrays have different lengths")
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"{resolved}: {name} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"{resolved}: {name} contains NaN/Inf")
            arrays[name] = value

        # Variable-width state arrays are validated against their paired
        # metadata/config shapes rather than silently resized.
        for name, prefix, suffix in (
            ("hall_loading_history", (2, NUM_HALL_SENSORS), (6,)),
            ("hall_policy_history", (2,), (NUM_HALL_SENSORS, 3)),
        ):
            if name not in payload.files:
                raise ValueError(f"{resolved}: missing required field {name!r}")
            value = np.asarray(payload[name])
            if (
                value.ndim != len(prefix) + len(suffix) + 2
                or value.shape[0] != count
                or value.shape[1 : 1 + len(prefix)] != prefix
                or value.shape[-len(suffix) :] != suffix
                or value.shape[1 + len(prefix)] < 1
                or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{resolved}: invalid {name} shape/dtype/data: {value.shape}")
            arrays[name] = value

    assert count is not None
    if count <= 0:
        raise ValueError(f"{resolved}: state bank is empty")

    observation = arrays["observation"].astype(np.float32, copy=False)
    valid = observation[:, VALID_SLICE]
    if not np.all(valid > 0.5):
        raise ValueError(
            f"{resolved}: recovery training requires two valid Hall feet in every state"
        )
    commands = observation[:, 30:45].reshape(count, PROPRIO_HISTORY_LENGTH, 3)
    if not np.allclose(commands, commands[:, -1:, :], atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{resolved}: five-frame command history is inconsistent")
    if np.any(np.abs(commands[:, :, 1:]) > 1.0e-6):
        raise ValueError(f"{resolved}: recovery bank must contain straight commands")
    source_rows = arrays["source_seed"].astype(np.int64, copy=False)
    if set(np.unique(source_rows).tolist()) != source_seeds:
        raise ValueError(
            f"{resolved}: row source_seed values do not match metadata source_seeds"
        )
    if set(np.unique(source_rows).tolist()) & locked:
        raise ValueError(f"{resolved}: locked source seed appears in data rows")

    quat = arrays["root_pose_local"][:, 3:7]
    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=2.0e-3, rtol=0.0):
        raise ValueError(f"{resolved}: root quaternions are not normalized")
    lateral_axis = arrays["straight_track_lateral_axis"]
    if not np.allclose(np.linalg.norm(lateral_axis, axis=1), 1.0, atol=2.0e-3, rtol=0.0):
        raise ValueError(f"{resolved}: straight-track lateral axes are not normalized")
    heading_ref = arrays["straight_heading_reference_xy"]
    if not np.allclose(np.linalg.norm(heading_ref, axis=1), 1.0, atol=2.0e-3, rtol=0.0):
        raise ValueError(f"{resolved}: straight-heading references are not normalized")

    tensor_arrays: dict[str, torch.Tensor] = {}
    target_device = torch.device(device)
    integer_fields = {
        "hall_policy_delay_steps",
        "source_seed",
        "source_env_id",
        "source_rollout_step",
        "state_kind",
    }
    for name, value in arrays.items():
        dtype = torch.long if name in integer_fields else torch.float32
        tensor_arrays[name] = torch.as_tensor(value, device=target_device, dtype=dtype)
    return HighEndStateBank(str(resolved), metadata, tensor_arrays)


def policy_history_terms(observation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split audited term-major policy rows into unflattened histories."""

    if observation.ndim != 2 or observation.shape[1] != POLICY_DIM:
        raise ValueError(
            f"observation must have shape [N,{POLICY_DIM}], got {tuple(observation.shape)}"
        )
    if not torch.isfinite(observation).all():
        raise ValueError("observation contains NaN/Inf")
    result: dict[str, torch.Tensor] = {}
    for name, (span, history, width) in POLICY_HISTORY_LAYOUT.items():
        result[name] = observation[:, span].reshape(-1, history, width)
    return result


def seed_circular_buffer_logical(
    circular_buffer,
    env_ids: torch.Tensor,
    logical_oldest_to_newest: torch.Tensor,
) -> None:
    """Write selected logical histories without disturbing other envs.

    Isaac Lab stores ring-buffer time first and exposes a rolled/transposed
    oldest-to-newest view.  The pointer is shared across environments, so an
    env-subset restore must use the current pointer rather than resetting it.
    """

    if logical_oldest_to_newest.ndim < 3:
        raise ValueError("logical history must have shape [N,L,...]")
    length = int(logical_oldest_to_newest.shape[1])
    if length != int(circular_buffer.max_length):
        raise ValueError(
            f"history length {length} != buffer max_length {circular_buffer.max_length}"
        )
    ids = env_ids.to(device=circular_buffer._device, dtype=torch.long)
    values = logical_oldest_to_newest.to(device=circular_buffer._device)
    if values.shape[0] != ids.numel():
        raise ValueError("history batch does not match env_ids")
    if circular_buffer._buffer is None:
        # Allocate without mutating the shared pointer through append().
        circular_buffer._pointer = 0
        circular_buffer._buffer = torch.zeros(
            (length, circular_buffer.batch_size, *values.shape[2:]),
            device=circular_buffer._device,
            dtype=values.dtype,
        )
    expected = (length, circular_buffer.batch_size, *values.shape[2:])
    if tuple(circular_buffer._buffer.shape) != expected:
        raise ValueError(
            f"ring storage shape {tuple(circular_buffer._buffer.shape)} != {expected}"
        )
    # buffer property computes roll(raw, L-pointer-1).  Apply its inverse.
    logical_time_first = values.transpose(0, 1)
    raw = torch.roll(
        logical_time_first,
        shifts=int(circular_buffer._pointer) + 1 - length,
        dims=0,
    )
    circular_buffer._buffer[:, ids] = raw
    circular_buffer._num_pushes[ids] = length


__all__ = [
    "ACTION_DIM",
    "DEVELOPMENT_ROLE",
    "HighEndStateBank",
    "LOCKED_ACCEPTANCE_SEEDS",
    "MOTION_SLICE",
    "POLICY_DIM",
    "POLICY_HISTORY_LAYOUT",
    "SCHEMA_VERSION",
    "TRAINING_ROLE",
    "VALIDATION_ROLE",
    "VALID_SLICE",
    "load_high_end_state_bank",
    "policy_history_terms",
    "seed_circular_buffer_logical",
]
