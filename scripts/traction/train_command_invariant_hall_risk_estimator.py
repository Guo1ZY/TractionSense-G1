#!/usr/bin/env python3
"""Train a command-invariant Hall/proprio low-traction risk head.

The runtime model consumes exactly the canonical 1864-D Hall-motion actor
observation.  Its five-frame command history is masked *inside the model*.
Ground friction and spatial course stage are used only to construct offline
labels in this tool; contact, force and slip are never runtime inputs.

Two rollout formats are supported:

* spatial: ``observation``, ``fastbase_course_stage``, env id and step, plus
  an explicit H-L-H or L-H-L CLI selector (stage alone is not a material);
* switch: ``obs``, ``mu``, env id, step and ``time_since_switch_s``.

Every CLI input file remains an indivisible source/seed.  Training and
held-out seeds must be disjoint, preventing row-level leakage across a
trajectory.  This tool deliberately emits a research smoke artifact, never a
release/deployment claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    COMMAND_HISTORY_SLICE,
    COMMAND_MASKED_MODEL_VARIANT,
    COMMAND_MASKED_TRAILING_FEATURE_MODE,
    CommandMaskedHallRiskEstimator,
    build_hall_risk_estimator,
    command_masked_risk_schema,
    command_masked_risk_schema_sha256,
)
from unitree_rl_lab.traction.contact_slip import (  # noqa: E402
    CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
    CONTACT_POINT_TANGENTIAL_SLIP_KEY,
    CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
    CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
    LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    BASE_DIM,
    INPUT_DIM,
    MAGNETIC_SLICE,
    PERIOD_SLICE,
    VALID_SLICE,
)


MU_CLASS_RISK_TARGET = "offline low-friction class (not governor compatible)"
PROSPECTIVE_RISK_TARGET = "prospective contact-point slip/fall"
RUNTIME_MEASUREMENT_BOUNDARY = (
    "runtime input is Hall Bx/By/Bz history + proprioception only; command "
    "history [30:45) is internally masked; contact slip/falls, mu and stage "
    "are offline simulator labels, not inputs; no Hall-to-force and no "
    "Hall-to-friction inverse"
)


@dataclass(frozen=True)
class OfflinePart:
    observation: np.ndarray
    target: np.ndarray
    env_id: np.ndarray
    step: np.ndarray
    command_vx: np.ndarray
    phase: np.ndarray
    source_kind: str
    source_id: str
    seed: int
    path: Path
    audit: dict[str, int | float | str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_from_path(path: Path) -> int:
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:\D|$)", path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"{path}: no unique seed array and filename has no '_seedNNN' token"
        )
    return int(match.group(1))


def _single_seed(path: Path, data: np.lib.npyio.NpzFile) -> int:
    if "seed" not in data.files:
        return _seed_from_path(path)
    values = np.unique(np.asarray(data["seed"], dtype=np.int64).reshape(-1))
    if len(values) != 1:
        raise ValueError(
            f"{path}: each source must contain one complete seed, found {values.tolist()}"
        )
    return int(values[0])


def _array_key(data: np.lib.npyio.NpzFile, path: Path, *keys: str) -> str:
    for key in keys:
        if key in data.files:
            return key
    raise ValueError(f"{path}: missing one of required keys {keys}")


def _validate_observation(path: Path, observation: np.ndarray) -> None:
    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{path}: expected canonical observation [N,{INPUT_DIM}], "
            f"got {observation.shape}"
        )


def _both_hall_feet_valid(observation: np.ndarray) -> np.ndarray:
    valid = observation[:, VALID_SLICE]
    return np.isfinite(valid).all(axis=1) & (valid >= 0.5).all(axis=1)


def _spatial_transition_keep(
    stage: np.ndarray,
    env_id: np.ndarray,
    step: np.ndarray,
    washout_steps: int,
) -> np.ndarray:
    """Exclude the first N causal samples after each in-trajectory stage change."""

    keep = np.ones(len(stage), dtype=bool)
    if washout_steps <= 0:
        return keep
    for identifier in np.unique(env_id):
        indices = np.flatnonzero(env_id == identifier)
        # Rows are emitted in causal evaluator order.  A non-increasing step
        # marks a managed reset and starts a new physical trajectory.
        previous_step: int | None = None
        previous_stage: int | None = None
        transition_step: int | None = None
        for index in indices:
            current_step = int(step[index])
            current_stage = int(stage[index])
            if previous_step is not None and current_step <= previous_step:
                previous_stage = None
                transition_step = None
            if previous_stage is not None and current_stage != previous_stage:
                transition_step = current_step
            if transition_step is not None:
                elapsed = current_step - transition_step
                if 0 <= elapsed < washout_steps:
                    keep[index] = False
            previous_step = current_step
            previous_stage = current_stage
    return keep


def prospective_future_event_target(
    contact_slip: np.ndarray,
    fall: np.ndarray,
    rollout_id: np.ndarray,
    env_id: np.ndarray,
    phase: np.ndarray,
    step: np.ndarray,
    horizon_steps: int,
    contact_slip_threshold: float,
    future_slip_quantile: float = 0.75,
    contact_slip_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Label the next 0.24 s of physics without crossing physical segments.

    The switch collector stores ``obs`` before ``env.step`` and stores the
    corresponding ``contact_slip``/``fall`` after that same step.  Therefore
    the outcome aligned with observation row ``t`` is the *first* 20 ms of
    its prospective window.  For a 12-step horizon the exact window is
    ``t <= step < t + 12`` within the same ``(rollout_id, env_id, phase)``
    group.  Excluding the current row would silently shift a nominal 0.24 s
    target to 0.02--0.26 s.

    The requested quantile is evaluated with the conservative discrete
    ``lower`` rank.  At the default q=0.75 and 12 samples this is equivalent
    to requiring at least four samples (80 ms) at or above the slip threshold,
    which rejects isolated contact-solver spikes without interpolation
    ambiguity.  A target is also positive when any aligned outcome is a fall.
    Rows without the complete window are right-censored, so the final
    ``horizon_steps - 1`` rows are removed instead of being mislabeled safe.
    """

    arrays = tuple(
        np.asarray(value).reshape(-1)
        for value in (contact_slip, fall, rollout_id, env_id, phase, step)
    )
    count = len(arrays[0])
    if any(len(value) != count for value in arrays):
        raise ValueError("prospective label arrays must have identical lengths")
    if horizon_steps <= 0:
        raise ValueError("prospective horizon_steps must be positive")
    if contact_slip_threshold <= 0.0:
        raise ValueError("contact_slip_threshold must be positive")
    if not 0.0 <= future_slip_quantile <= 1.0:
        raise ValueError("future_slip_quantile must be in [0,1]")
    slip = np.asarray(arrays[0], dtype=np.float32)
    falls = np.asarray(arrays[1], dtype=bool)
    rollout = np.asarray(arrays[2], dtype=np.int64)
    environment = np.asarray(arrays[3], dtype=np.int64)
    phases = np.asarray(arrays[4], dtype=np.int64)
    steps = np.asarray(arrays[5], dtype=np.int64)
    if not np.isfinite(slip).all():
        raise ValueError("contact_slip must be finite for prospective labels")
    if contact_slip_valid is None:
        slip_valid = np.ones(count, dtype=bool)
    else:
        slip_valid = np.asarray(contact_slip_valid, dtype=bool).reshape(-1)
        if len(slip_valid) != count:
            raise ValueError(
                "contact_slip_valid must match prospective label arrays"
            )

    target = np.zeros(count, dtype=np.float32)
    label_valid = np.zeros(count, dtype=bool)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(zip(rollout, environment, phases, strict=True)):
        groups.setdefault(tuple(map(int, key)), []).append(index)
    for key, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        order = np.argsort(steps[indices], kind="stable")
        indices = indices[order]
        group_steps = steps[indices]
        if len(group_steps) > 1 and not np.all(np.diff(group_steps) == 1):
            raise ValueError(
                f"prospective group {key} must have unique contiguous policy steps"
            )
        for local_index, original_index in enumerate(indices):
            future_end = local_index + horizon_steps
            if future_end > len(indices):
                continue
            expected_end_step = int(group_steps[local_index]) + horizon_steps - 1
            if int(group_steps[future_end - 1]) != expected_end_step:
                raise ValueError(
                    f"prospective group {key} has incomplete future horizon"
                )
            future = indices[local_index:future_end]
            future_fall = bool(np.any(falls[future]))
            # A fall is a valid positive even if managed reset leaves no
            # meaningful contact point for that aligned outcome.  A negative
            # label, however, requires a complete horizon of valid contact
            # slip measurements; treating no-contact/NaN as zero would be an
            # unsafe false-negative shortcut.
            if not future_fall and not bool(slip_valid[future].all()):
                continue
            label_valid[original_index] = True
            sustained_slip = float(
                np.quantile(
                    slip[future][slip_valid[future]],
                    future_slip_quantile,
                    method="lower",
                )
            ) if bool(slip_valid[future].any()) else 0.0
            target[original_index] = float(
                sustained_slip >= contact_slip_threshold
                or future_fall
            )
    return target, label_valid


def load_spatial_part(
    path: Path,
    transition_washout_steps: int,
    course_pattern: str,
) -> OfflinePart:
    """Load one spatial source with an explicit material ordering.

    Stage ids encode longitudinal regions, not friction by themselves.  A
    H-L-H course has risk stage 1, whereas an L-H-L course has risk stages 0
    and 2.  Requiring this semantic at the call site prevents an inverted
    label when evaluating counterbalanced courses.
    """

    normalized_pattern = course_pattern.upper().replace("-", "")
    if normalized_pattern == "HLH":
        risk_stages = (1,)
    elif normalized_pattern == "LHL":
        risk_stages = (0, 2)
    else:
        raise ValueError("spatial course_pattern must be explicitly 'HLH' or 'LHL'")
    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        observation_key = _array_key(data, path, "observation")
        stage_key = _array_key(data, path, "fastbase_course_stage")
        env_key = _array_key(data, path, "fastbase_env_id", "env_id")
        step_key = _array_key(data, path, "fastbase_rollout_step", "step")
        observation = np.asarray(data[observation_key], dtype=np.float32)
        stage = np.asarray(data[stage_key], dtype=np.int16).reshape(-1)
        env_id = np.asarray(data[env_key], dtype=np.int64).reshape(-1)
        step = np.asarray(data[step_key], dtype=np.int64).reshape(-1)
        seed = _single_seed(path, data)
    _validate_observation(path, observation)
    count = len(observation)
    if not all(len(value) == count for value in (stage, env_id, step)):
        raise ValueError(f"{path}: spatial arrays have inconsistent lengths")
    recognized = np.isin(stage, (0, 1, 2))
    finite = np.isfinite(observation).all(axis=1)
    both_valid = _both_hall_feet_valid(observation)
    washout_keep = _spatial_transition_keep(
        stage, env_id, step, transition_washout_steps
    )
    keep = recognized & finite & both_valid & washout_keep
    target = np.isin(stage, risk_stages).astype(np.float32)
    source_id = f"spatial-{normalized_pattern}:{path.name}:seed{seed}"
    return OfflinePart(
        observation=observation[keep],
        target=target[keep],
        env_id=env_id[keep],
        step=step[keep],
        command_vx=observation[keep, COMMAND_HISTORY_SLICE.stop - 3],
        phase=stage[keep].astype(np.int64),
        source_kind="spatial",
        source_id=source_id,
        seed=seed,
        path=path,
        audit={
            "input_rows": count,
            "kept_rows": int(keep.sum()),
            "nonfinite_removed": int((~finite).sum()),
            "not_both_valid_removed": int((finite & ~both_valid).sum()),
            "transition_washout_removed": int(
                (recognized & finite & both_valid & ~washout_keep).sum()
            ),
            "positive_rows": int((target[keep] >= 0.5).sum()),
            "negative_rows": int((target[keep] < 0.5).sum()),
            "course_pattern": normalized_pattern,
            "risk_stages": ",".join(map(str, risk_stages)),
            "label": f"fastbase_course_stage in {risk_stages}",
        },
    )


def load_switch_part(
    path: Path,
    transition_washout_s: float,
    low_mu_max: float,
    high_mu_min: float,
    *,
    label_mode: str = "mu",
    future_horizon_s: float = 0.24,
    policy_dt_s: float = 0.02,
    contact_slip_threshold: float = 0.045,
    future_slip_quantile: float = 0.75,
    prospective_transition_washout_s: float = 0.0,
    allow_research_legacy_link_origin_slip: bool = False,
) -> OfflinePart:
    if label_mode not in ("mu", "prospective_slip_fall"):
        raise ValueError("switch label_mode must be 'mu' or 'prospective_slip_fall'")
    if future_horizon_s <= 0.0 or policy_dt_s <= 0.0:
        raise ValueError("future horizon and policy dt must be positive")
    horizon_steps_float = future_horizon_s / policy_dt_s
    horizon_steps = int(round(horizon_steps_float))
    if not np.isclose(horizon_steps_float, horizon_steps, atol=1.0e-8):
        raise ValueError("future_horizon_s must be an integer multiple of policy_dt_s")
    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        observation_key = _array_key(data, path, "obs")
        env_key = _array_key(data, path, "env_id")
        step_key = _array_key(data, path, "step")
        time_key = _array_key(data, path, "time_since_switch_s")
        mu_key = _array_key(data, path, "mu")
        observation = np.asarray(data[observation_key], dtype=np.float32)
        mu = np.asarray(data[mu_key], dtype=np.float32).reshape(-1)
        env_id = np.asarray(data[env_key], dtype=np.int64).reshape(-1)
        step = np.asarray(data[step_key], dtype=np.int64).reshape(-1)
        time_since_switch_s = np.asarray(data[time_key], dtype=np.float32).reshape(-1)
        row_valid = np.asarray(
            data["valid"] if "valid" in data.files else np.ones(len(observation)),
            dtype=bool,
        ).reshape(-1)
        phase_present = "phase" in data.files
        phase = np.asarray(
            data["phase"] if "phase" in data.files else np.zeros(len(observation)),
            dtype=np.int64,
        ).reshape(-1)
        actor_command = (
            np.asarray(data["actor_command"], dtype=np.float32)
            if "actor_command" in data.files
            else None
        )
        applied_command = (
            np.asarray(data["applied_command"], dtype=np.float32)
            if "applied_command" in data.files
            else None
        )
        if actor_command is not None:
            command_vx = actor_command[:, 0].copy()
        elif "cmd_vx" in data.files:
            command_vx = np.asarray(data["cmd_vx"], dtype=np.float32).reshape(-1)
        elif applied_command is not None:
            command_vx = applied_command[:, 0].copy()
        else:
            command_vx = observation[:, COMMAND_HISTORY_SLICE.stop - 3].copy()
        rollout_id = (
            np.asarray(data["rollout_id"], dtype=np.int64).reshape(-1)
            if "rollout_id" in data.files
            else None
        )
        contact_slip_alias = (
            np.asarray(data["contact_slip"], dtype=np.float32).reshape(-1)
            if "contact_slip" in data.files
            else None
        )
        contact_point_slip = (
            np.asarray(
                data[CONTACT_POINT_TANGENTIAL_SLIP_KEY], dtype=np.float32
            ).reshape(-1)
            if CONTACT_POINT_TANGENTIAL_SLIP_KEY in data.files
            else None
        )
        contact_point_slip_valid = (
            np.asarray(
                data[CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY], dtype=bool
            ).reshape(-1)
            if CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY in data.files
            else None
        )
        legacy_link_origin_slip = (
            np.asarray(
                data[LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY], dtype=np.float32
            ).reshape(-1)
            if LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY in data.files
            else None
        )
        fall = (
            np.asarray(data["fall"], dtype=bool).reshape(-1)
            if "fall" in data.files
            else None
        )
        done = (
            np.asarray(data["done"], dtype=bool).reshape(-1)
            if "done" in data.files
            else None
        )
        time_out = (
            np.asarray(data["time_out"], dtype=bool).reshape(-1)
            if "time_out" in data.files
            else None
        )
        hall_valid_lr = (
            np.asarray(data["hall_valid_lr"], dtype=np.float32)
            if "hall_valid_lr" in data.files
            else None
        )
        metadata_policy_dt = (
            float(np.asarray(data["policy_dt"]).reshape(()).item())
            if "policy_dt" in data.files
            else None
        )
        metadata_collect_stride = (
            int(np.asarray(data["collect_stride"]).reshape(()).item())
            if "collect_stride" in data.files
            else None
        )
        metadata_dataset_kind = (
            str(np.asarray(data["dataset_kind"]).reshape(()).item())
            if "dataset_kind" in data.files
            else None
        )
        metadata_contact_slip_schema = (
            str(np.asarray(data["contact_slip_schema"]).reshape(()).item())
            if "contact_slip_schema" in data.files
            else None
        )
        metadata_contact_slip_metric_key = (
            str(
                np.asarray(data["contact_slip_metric_key"])
                .reshape(())
                .item()
            )
            if "contact_slip_metric_key" in data.files
            else None
        )
        metadata_contact_slip_valid_key = (
            str(
                np.asarray(data["contact_slip_valid_key"])
                .reshape(())
                .item()
            )
            if "contact_slip_valid_key" in data.files
            else None
        )
        metadata_contact_slip_formula = (
            str(np.asarray(data["contact_slip_formula"]).reshape(()).item())
            if "contact_slip_formula" in data.files
            else None
        )
        metadata_legacy_contact_slip_metric_key = (
            str(
                np.asarray(data["legacy_contact_slip_metric_key"])
                .reshape(())
                .item()
            )
            if "legacy_contact_slip_metric_key" in data.files
            else None
        )
        metadata_manifest = None
        if "metadata_json" in data.files:
            try:
                metadata_manifest = json.loads(
                    str(np.asarray(data["metadata_json"]).reshape(()).item())
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}: invalid metadata_json") from exc
        seed = _single_seed(path, data)
    _validate_observation(path, observation)
    count = len(observation)
    if not all(
        len(value) == count
        for value in (mu, env_id, step, time_since_switch_s, row_valid, phase, command_vx)
    ):
        raise ValueError(f"{path}: switch arrays have inconsistent lengths")
    for name, value in (("done", done), ("time_out", time_out)):
        if value is not None and len(value) != count:
            raise ValueError(f"{path}: {name} length does not match observations")
    for name, value in (
        ("actor_command", actor_command),
        ("applied_command", applied_command),
    ):
        if value is not None:
            if value.shape != (count, 3):
                raise ValueError(
                    f"{path}: {name} must have shape {(count, 3)}, "
                    f"got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {name} must be finite")
    if actor_command is not None and not np.array_equal(
        actor_command, observation[:, COMMAND_HISTORY_SLICE.stop - 3 : COMMAND_HISTORY_SLICE.stop]
    ):
        raise ValueError(
            f"{path}: actor_command does not match the exact actor observation "
            "newest command frame [:,42:45]"
        )
    if hall_valid_lr is not None:
        if hall_valid_lr.shape != (count, 2):
            raise ValueError(
                f"{path}: hall_valid_lr must have shape {(count, 2)}, "
                f"got {hall_valid_lr.shape}"
            )
        if not np.array_equal(
            hall_valid_lr, observation[:, VALID_SLICE]
        ):
            raise ValueError(
                f"{path}: hall_valid_lr does not match exact actor obs[:,1860:1862]"
            )
    class_known = (mu <= low_mu_max) | (mu >= high_mu_min)
    finite = (
        np.isfinite(observation).all(axis=1)
        & np.isfinite(mu)
        & np.isfinite(time_since_switch_s)
        & np.isfinite(command_vx)
    )
    both_valid = _both_hall_feet_valid(observation)
    if label_mode == "mu":
        label_valid = class_known
        target = (mu <= low_mu_max).astype(np.float32)
        selected_washout_s = transition_washout_s
        washout_keep = time_since_switch_s + 1.0e-6 >= selected_washout_s
        current_state_valid = np.ones(count, dtype=bool)
        label_description = f"mu <= {low_mu_max:g}; safe iff mu >= {high_mu_min:g}"
        right_censored = np.zeros(count, dtype=bool)
        positive_washout_rescued = np.zeros(count, dtype=bool)
        contact_slip_source = "not_used_for_mu_label"
        contact_slip_valid = np.ones(count, dtype=bool)
    else:
        if not phase_present or rollout_id is None or fall is None:
            raise ValueError(
                f"{path}: prospective mode requires rollout_id, phase, "
                "the versioned contact-point slip metric and fall"
            )
        if contact_point_slip is not None:
            expected_metadata = {
                "contact_slip_schema": CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
                "contact_slip_metric_key": CONTACT_POINT_TANGENTIAL_SLIP_KEY,
                "contact_slip_valid_key": CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
                "contact_slip_formula": CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
                "legacy_contact_slip_metric_key": LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
                "actor_command_key": "actor_command",
                "actor_command_source": (
                    "exact pre-step actor observation newest command frame [:,42:45]"
                ),
                "applied_command_key": "applied_command",
                "applied_command_source": (
                    "base_velocity.vel_command_b[:,0:3] snapshot immediately before env.step"
                ),
            }
            observed_metadata = {
                "contact_slip_schema": metadata_contact_slip_schema,
                "contact_slip_metric_key": metadata_contact_slip_metric_key,
                "contact_slip_valid_key": metadata_contact_slip_valid_key,
                "contact_slip_formula": metadata_contact_slip_formula,
                "legacy_contact_slip_metric_key": (
                    metadata_legacy_contact_slip_metric_key
                ),
            }
            for key, expected in expected_metadata.items():
                if key in observed_metadata and observed_metadata[key] != expected:
                    raise ValueError(
                        f"{path}: {key} must be {expected!r}, got "
                        f"{observed_metadata[key]!r}"
                    )
                if not isinstance(metadata_manifest, dict) or (
                    metadata_manifest.get(key) != expected
                ):
                    raise ValueError(
                        f"{path}: metadata_json must record exact {key}={expected!r}"
                    )
            if contact_point_slip_valid is None:
                raise ValueError(
                    f"{path}: versioned contact-point slip requires "
                    f"{CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY}"
                )
            if legacy_link_origin_slip is None:
                raise ValueError(
                    f"{path}: new collector must retain diagnostic "
                    f"{LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY}"
                )
            if contact_slip_alias is None or not np.array_equal(
                contact_slip_alias, contact_point_slip
            ):
                raise ValueError(
                    f"{path}: contact_slip compatibility alias must exactly "
                    f"match {CONTACT_POINT_TANGENTIAL_SLIP_KEY}"
                )
            if actor_command is None or applied_command is None:
                raise ValueError(
                    f"{path}: versioned collector requires actor_command and "
                    "applied_command provenance"
                )
            contact_slip = contact_point_slip
            contact_slip_valid = contact_point_slip_valid
            contact_slip_source = CONTACT_POINT_TANGENTIAL_SLIP_KEY
        elif allow_research_legacy_link_origin_slip:
            # Explicit research-only compatibility for already collected
            # files.  Never infer legacy semantics from a version string: use
            # the clearly named field when available, otherwise the old
            # unversioned alias.  This path is surfaced in every audit and
            # checkpoint metadata and is not a deployment-quality label.
            if legacy_link_origin_slip is not None:
                contact_slip = legacy_link_origin_slip
                contact_slip_source = LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY
            elif contact_slip_alias is not None:
                contact_slip = contact_slip_alias
                contact_slip_source = "unversioned_contact_slip_legacy_assumed"
            else:
                raise ValueError(
                    f"{path}: no legacy link-origin slip field is available"
                )
            contact_slip_valid = np.ones(len(contact_slip), dtype=bool)
        else:
            raise ValueError(
                f"{path}: prospective labels require versioned "
                f"{CONTACT_POINT_TANGENTIAL_SLIP_KEY} schema "
                f"{CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA!r}; old link-origin "
                "datasets are rejected by default. For research-only "
                "comparison, explicitly enable "
                "allow_research_legacy_link_origin_slip."
            )
        if not all(len(value) == count for value in (rollout_id, contact_slip, fall)):
            raise ValueError(f"{path}: prospective arrays have inconsistent lengths")
        if len(contact_slip_valid) != count:
            raise ValueError(
                f"{path}: contact-point slip validity length mismatch"
            )
        if contact_slip_source == CONTACT_POINT_TANGENTIAL_SLIP_KEY and any(
            value is None
            for value in (
                metadata_collect_stride,
                metadata_policy_dt,
                metadata_dataset_kind,
            )
        ):
            raise ValueError(
                f"{path}: versioned prospective data requires policy_dt, "
                "collect_stride and dataset_kind metadata"
            )
        if metadata_collect_stride is not None and metadata_collect_stride != 1:
            raise ValueError(
                f"{path}: prospective labels require collect_stride=1, "
                f"metadata reports {metadata_collect_stride}"
            )
        if metadata_policy_dt is not None and not np.isclose(
            metadata_policy_dt, policy_dt_s, atol=1.0e-9, rtol=0.0
        ):
            raise ValueError(
                f"{path}: policy_dt metadata {metadata_policy_dt:g} does not "
                f"match requested {policy_dt_s:g}"
            )
        if metadata_dataset_kind is not None and metadata_dataset_kind != "switch":
            raise ValueError(
                f"{path}: prospective switch loader rejects "
                f"dataset_kind={metadata_dataset_kind!r}"
            )
        if done is not None or time_out is not None:
            if done is None or time_out is None:
                raise ValueError(
                    f"{path}: done and time_out provenance must be present together"
                )
            expected_fall = done & ~time_out
            if not np.array_equal(fall, expected_fall):
                raise ValueError(
                    f"{path}: fall must equal done & ~time_out for every row"
                )
            # Every managed reset must terminate the old rollout id before the
            # next same-env/same-phase observation.  Conversely, ordinary
            # consecutive samples must remain in one rollout.  This catches a
            # timeout-only reset that a fall-only generation counter misses.
            for phase_value in np.unique(phase):
                for env_value in np.unique(env_id[phase == phase_value]):
                    rows = np.flatnonzero(
                        (phase == phase_value) & (env_id == env_value)
                    )
                    rows = rows[np.argsort(step[rows], kind="stable")]
                    if len(rows) < 2:
                        continue
                    changed = rollout_id[rows[1:]] != rollout_id[rows[:-1]]
                    if not np.array_equal(changed, done[rows[:-1]]):
                        raise ValueError(
                            f"{path}: rollout_id reset segmentation mismatch "
                            f"for phase={int(phase_value)}, env={int(env_value)}"
                        )
        target, label_valid = prospective_future_event_target(
            contact_slip,
            fall,
            rollout_id,
            env_id,
            phase,
            step,
            horizon_steps,
            contact_slip_threshold,
            future_slip_quantile,
            contact_slip_valid,
        )
        selected_washout_s = prospective_transition_washout_s
        raw_washout_keep = (
            time_since_switch_s + 1.0e-6 >= selected_washout_s
        )
        # An optional washout may remove history-contaminated negatives, but
        # never a genuine future-event precursor at the start of H->L.
        positive_washout_rescued = (~raw_washout_keep) & (target >= 0.5)
        washout_keep = raw_washout_keep | (target >= 0.5)
        # ``fall`` belongs to the post-step outcome of the pre-step observation
        # stored on this row.  It is a valid causal positive, not an already
        # reset observation, and must not be discarded.
        current_state_valid = np.ones(count, dtype=bool)
        right_censored = ~label_valid
        label_description = (
            f"future contact_slip q={future_slip_quantile:g} >= "
            f"{contact_slip_threshold:g} or any fall at "
            f"t <= outcome < t+{future_horizon_s:g}s; current aligned "
            "post-step outcome included; discrete lower-rank quantile; "
            "complete within-phase/reset-safe horizon required"
        )
    keep = (
        label_valid
        & finite
        & both_valid
        & row_valid
        & washout_keep
        & current_state_valid
    )
    source_id = f"switch-{label_mode}:{path.name}:seed{seed}"
    return OfflinePart(
        observation=observation[keep],
        target=target[keep],
        env_id=env_id[keep],
        step=step[keep],
        command_vx=command_vx[keep],
        phase=phase[keep],
        source_kind="switch",
        source_id=source_id,
        seed=seed,
        path=path,
        audit={
            "input_rows": count,
            "kept_rows": int(keep.sum()),
            "nonfinite_removed": int((~finite).sum()),
            "not_both_valid_removed": int((finite & ~both_valid).sum()),
            "external_invalid_removed": int((finite & both_valid & ~row_valid).sum()),
            "transition_washout_s": float(selected_washout_s),
            "transition_washout_removed": int(
                (label_valid & finite & both_valid & row_valid & ~washout_keep).sum()
            ),
            "positive_transition_rows_rescued": int(positive_washout_rescued.sum()),
            "ambiguous_mu_removed": int((~class_known).sum()) if label_mode == "mu" else 0,
            "right_censored_horizon_removed": int(right_censored.sum()),
            "current_fall_rows_removed": int((label_valid & ~current_state_valid).sum()),
            "positive_rows": int((target[keep] >= 0.5).sum()),
            "negative_rows": int((target[keep] < 0.5).sum()),
            "label_mode": label_mode,
            "future_horizon_s": float(future_horizon_s),
            "future_horizon_steps": horizon_steps,
            "contact_slip_threshold": float(contact_slip_threshold),
            "future_slip_quantile": float(future_slip_quantile),
            "contact_slip_source": contact_slip_source,
            "contact_slip_schema": (
                CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
                if contact_slip_source == CONTACT_POINT_TANGENTIAL_SLIP_KEY
                else "research_legacy_link_origin"
            ),
            "contact_slip_valid_rows": int(contact_slip_valid.sum()),
            "research_legacy_link_origin_slip": bool(
                contact_slip_source != CONTACT_POINT_TANGENTIAL_SLIP_KEY
                and label_mode == "prospective_slip_fall"
            ),
            "metadata_policy_dt": metadata_policy_dt,
            "metadata_collect_stride": metadata_collect_stride,
            "metadata_dataset_kind": metadata_dataset_kind,
            "hall_valid_lr_cross_checked": hall_valid_lr is not None,
            "managed_reset_segmentation_checked": done is not None,
            "label": label_description,
        },
    )


def validate_strict_heldout(
    train_parts: list[OfflinePart], heldout_parts: list[OfflinePart]
) -> None:
    if not train_parts or not heldout_parts:
        raise ValueError("both train and heldout must contain complete source files")
    train_paths = {part.path for part in train_parts}
    heldout_paths = {part.path for part in heldout_parts}
    overlap_paths = train_paths & heldout_paths
    if overlap_paths:
        raise ValueError(f"train/heldout source overlap: {sorted(map(str, overlap_paths))}")
    train_seeds = {part.seed for part in train_parts}
    heldout_seeds = {part.seed for part in heldout_parts}
    overlap_seeds = train_seeds & heldout_seeds
    if overlap_seeds:
        raise ValueError(
            f"train/heldout seed overlap is forbidden: {sorted(overlap_seeds)}"
        )
    train_kinds = {part.source_kind for part in train_parts}
    heldout_kinds = {part.source_kind for part in heldout_parts}
    missing = train_kinds - heldout_kinds
    if missing:
        raise ValueError(f"heldout split lacks source kinds: {sorted(missing)}")
    for split_name, parts in (("train", train_parts), ("heldout", heldout_parts)):
        for part in parts:
            if len(part.observation) == 0:
                raise ValueError(f"{split_name} source has no rows after masks: {part.path}")
            classes = np.unique(part.target >= 0.5)
            if len(classes) != 2:
                raise ValueError(
                    f"{split_name} source must retain both classes: {part.source_id}"
                )


def concatenate_parts(parts: list[OfflinePart]) -> dict[str, np.ndarray]:
    return {
        "observation": np.concatenate([part.observation for part in parts]),
        "target": np.concatenate([part.target for part in parts]),
        "source_kind": np.concatenate(
            [np.full(len(part.target), part.source_kind, dtype="U16") for part in parts]
        ),
        "source_id": np.concatenate(
            [np.full(len(part.target), part.source_id, dtype="U256") for part in parts]
        ),
        "seed": np.concatenate(
            [np.full(len(part.target), part.seed, dtype=np.int64) for part in parts]
        ),
        "command_vx": np.concatenate([part.command_vx for part in parts]),
        "phase": np.concatenate([part.phase for part in parts]),
    }


def source_class_balanced_weights(
    source_kind: np.ndarray,
    source_id: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Balance kind, class, then complete source within each kind/class."""

    binary = target >= 0.5
    weight = np.zeros(len(target), dtype=np.float64)
    kinds = np.unique(source_kind)
    for kind in kinds:
        kind_rows = source_kind == kind
        present_classes = [value for value in (False, True) if np.any(kind_rows & (binary == value))]
        for class_value in present_classes:
            class_rows = kind_rows & (binary == class_value)
            files = np.unique(source_id[class_rows])
            group_mass = 1.0 / (len(kinds) * len(present_classes) * len(files))
            for file_id in files:
                rows = class_rows & (source_id == file_id)
                weight[rows] = group_mass / int(rows.sum())
    if np.any(weight <= 0.0) or not np.isfinite(weight).all():
        raise ValueError("failed to assign finite positive source/class weights")
    return (weight / weight.mean()).astype(np.float32)


def weight_mass_report(
    source_kind: np.ndarray, source_id: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    binary = target >= 0.5
    total = float(weight.sum())
    for kind in np.unique(source_kind):
        for class_value, class_name in ((False, "safe"), (True, "risk")):
            rows = (source_kind == kind) & (binary == class_value)
            if np.any(rows):
                result[f"{kind}/{class_name}"] = float(weight[rows].sum() / total)
    for file_id in np.unique(source_id):
        rows = source_id == file_id
        result[f"file/{file_id}"] = float(weight[rows].sum() / total)
    return result


def raw_feature_statistics(
    observation: np.ndarray, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            raw, _ = CommandMaskedHallRiskEstimator.raw_features(
                torch.from_numpy(observation[start : start + batch_size])
            )
            batches.append(raw)
    features = torch.cat(batches, dim=0)
    return features.mean(dim=0), features.std(dim=0).clamp_min(0.05)


def infer(
    model: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(observation), batch_size):
            batch = torch.from_numpy(observation[start : start + batch_size]).to(device)
            output.append(model(batch).cpu().numpy().reshape(-1))
    return np.concatenate(output)


def roc_auc(target: np.ndarray, prediction: np.ndarray) -> float | None:
    positive = target >= 0.5
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        return None
    order = np.argsort(prediction, kind="stable")
    ranks = np.empty(len(prediction), dtype=np.float64)
    ranks[order] = np.arange(1, len(prediction) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(prediction, return_inverse=True, return_counts=True)
    for group, count in enumerate(counts):
        if count > 1:
            selected = inverse == group
            ranks[selected] = ranks[selected].mean()
    n_positive = float(positive.sum())
    n_negative = float(negative.sum())
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1.0) / 2.0)
        / (n_positive * n_negative)
    )


def binary_metrics(
    target: np.ndarray, prediction: np.ndarray, threshold: float
) -> dict[str, int | float | bool | None]:
    positive = target >= 0.5
    negative = ~positive
    decision = prediction >= threshold
    return {
        "samples": int(len(target)),
        "positive_samples": int(positive.sum()),
        "negative_samples": int(negative.sum()),
        "auc": roc_auc(target, prediction),
        "recall": float(decision[positive].mean()) if np.any(positive) else None,
        "false_alarm_rate": (
            float(decision[negative].mean()) if np.any(negative) else None
        ),
        "mean_risk_positive": (
            float(prediction[positive].mean()) if np.any(positive) else None
        ),
        "mean_risk_safe": (
            float(prediction[negative].mean()) if np.any(negative) else None
        ),
        "finite": bool(np.isfinite(prediction).all()),
    }


def metrics_by_source(
    data: dict[str, np.ndarray], prediction: np.ndarray, threshold: float
) -> dict[str, object]:
    by_kind: dict[str, object] = {}
    for kind in np.unique(data["source_kind"]):
        rows = data["source_kind"] == kind
        by_kind[str(kind)] = binary_metrics(
            data["target"][rows], prediction[rows], threshold
        )
    by_file: dict[str, object] = {}
    for source_id in np.unique(data["source_id"]):
        rows = data["source_id"] == source_id
        by_file[str(source_id)] = binary_metrics(
            data["target"][rows], prediction[rows], threshold
        )
    by_command_phase: dict[str, object] = {}
    rounded_command = np.round(data["command_vx"].astype(np.float64), 4)
    for source_id in np.unique(data["source_id"]):
        source_rows = data["source_id"] == source_id
        for command in np.unique(rounded_command[source_rows]):
            command_rows = source_rows & (rounded_command == command)
            for phase in np.unique(data["phase"][command_rows]):
                rows = command_rows & (data["phase"] == phase)
                key = f"{source_id}/command_vx={command:.4f}/phase={int(phase)}"
                by_command_phase[key] = binary_metrics(
                    data["target"][rows], prediction[rows], threshold
                )
    return {
        "aggregate": binary_metrics(data["target"], prediction, threshold),
        "per_source_kind": by_kind,
        "per_source_file": by_file,
        "per_source_command_phase": by_command_phase,
    }


def grouped_evidence_permutation(
    observation: np.ndarray,
    source_id: np.ndarray,
    phase: np.ndarray,
    *,
    evidence: str,
    seed: int,
) -> np.ndarray:
    """Break sample-level evidence while preserving each source/phase marginal.

    A global permutation can turn an episode-level Hall calibration draw into
    an accidental source classifier.  We therefore permute only inside a
    complete source and material-phase group.  Hall permutation moves the
    magnetic history and its sample-period history together, while leaving
    packet validity and all proprioception untouched.  Proprio permutation
    moves the command-masked 480-D base history plus the two motion-feedback
    channels; Hall, period and packet validity remain untouched.

    This is an audit transform only.  It is never used to train the model.
    """

    value = np.asarray(observation, dtype=np.float32)
    sources = np.asarray(source_id).reshape(-1)
    phases = np.asarray(phase, dtype=np.int64).reshape(-1)
    if value.ndim != 2 or value.shape[1] != INPUT_DIM:
        raise ValueError(f"expected observation [N,{INPUT_DIM}], got {value.shape}")
    if len(sources) != len(value) or len(phases) != len(value):
        raise ValueError("permutation group arrays must match observation rows")
    if evidence not in ("hall", "proprio"):
        raise ValueError("evidence must be 'hall' or 'proprio'")

    result = value.copy()
    rng = np.random.default_rng(seed)
    for source in np.unique(sources):
        source_rows = sources == source
        for phase_value in np.unique(phases[source_rows]):
            rows = np.flatnonzero(source_rows & (phases == phase_value))
            if len(rows) < 2:
                continue
            shuffled = rng.permutation(rows)
            if evidence == "hall":
                # The Hall block and sample-period history are one coherent
                # measurement packet.  Validity is intentionally not moved:
                # prospective training already requires both feet valid, and
                # invalid-link safety belongs to the outer Health envelope.
                result[rows, MAGNETIC_SLICE.start : PERIOD_SLICE.stop] = value[
                    shuffled, MAGNETIC_SLICE.start : PERIOD_SLICE.stop
                ]
            else:
                result[rows, :BASE_DIM] = value[shuffled, :BASE_DIM]
                result[rows, VALID_SLICE.stop : INPUT_DIM] = value[
                    shuffled, VALID_SLICE.stop : INPUT_DIM
                ]
    return result


def evidence_reliance_diagnostics(
    model: nn.Module,
    data: dict[str, np.ndarray],
    baseline_prediction: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Measure whether held-out discrimination materially uses Hall evidence."""

    baseline_auc = roc_auc(data["target"], baseline_prediction)
    result: dict[str, object] = {"baseline_auc": baseline_auc}
    for offset, evidence in enumerate(("hall", "proprio"), start=1):
        permuted_observation = grouped_evidence_permutation(
            data["observation"],
            data["source_id"],
            data["phase"],
            evidence=evidence,
            seed=seed + offset,
        )
        permuted_prediction = infer(
            model, permuted_observation, device, batch_size
        )
        permuted_auc = roc_auc(data["target"], permuted_prediction)
        result[evidence] = {
            "permuted_auc": permuted_auc,
            "auc_drop": (
                None
                if baseline_auc is None or permuted_auc is None
                else float(baseline_auc - permuted_auc)
            ),
            "grouping": "within complete source_id and phase",
        }
    return result


def offline_research_gate(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    reliance: dict[str, object],
    *,
    threshold: float,
    primary_command_min: float,
    min_primary_auc: float,
    min_primary_recall: float,
    max_primary_false_alarm_rate: float,
    min_hall_auc_drop: float,
    command_invariance_exact: bool,
    strict_restore_delta: float,
) -> dict[str, object]:
    """Return an offline-only gate; passing never authorizes deployment."""

    failures: list[str] = []
    per_source: dict[str, object] = {}
    rounded_command = np.round(data["command_vx"].astype(np.float64), 4)
    for source in np.unique(data["source_id"]):
        rows = data["source_id"] == source
        command = float(np.median(np.abs(rounded_command[rows])))
        metrics = binary_metrics(data["target"][rows], prediction[rows], threshold)
        primary = command + 1.0e-9 >= primary_command_min
        per_source[str(source)] = {
            "command_vx_abs_median": command,
            "primary_operating_source": primary,
            **metrics,
        }
        if not primary:
            continue
        if metrics["auc"] is None or metrics["auc"] < min_primary_auc:
            failures.append(f"{source}: primary AUC below {min_primary_auc:g}")
        if metrics["recall"] is None or metrics["recall"] < min_primary_recall:
            failures.append(f"{source}: primary recall below {min_primary_recall:g}")
        if (
            metrics["false_alarm_rate"] is None
            or metrics["false_alarm_rate"] > max_primary_false_alarm_rate
        ):
            failures.append(
                f"{source}: primary FAR above {max_primary_false_alarm_rate:g}"
            )

    hall = reliance.get("hall", {})
    hall_auc_drop = hall.get("auc_drop") if isinstance(hall, dict) else None
    if hall_auc_drop is None or float(hall_auc_drop) < min_hall_auc_drop:
        failures.append(
            f"heldout Hall permutation AUC drop below {min_hall_auc_drop:g}"
        )
    if not command_invariance_exact:
        failures.append("command counterfactual invariance is not exact")
    # Identical weights evaluated with different BLAS batch partitioning may
    # differ by a few float32 ulps.  The restore probe above deliberately uses
    # the same batch shape, while this 1e-6 guard still fails closed on any
    # material checkpoint/schema drift.
    if not np.isfinite(strict_restore_delta) or strict_restore_delta > 1.0e-6:
        failures.append("strict checkpoint restore changed model output beyond 1e-6")
    if not np.isfinite(prediction).all():
        failures.append("heldout prediction contains non-finite values")
    return {
        "passed": not failures,
        "scope": "offline research gate only; never a deployment authorization",
        "thresholds": {
            "primary_command_min": primary_command_min,
            "min_primary_auc": min_primary_auc,
            "min_primary_recall": min_primary_recall,
            "max_primary_false_alarm_rate": max_primary_false_alarm_rate,
            "min_hall_permutation_auc_drop": min_hall_auc_drop,
        },
        "failures": failures,
        "per_source": per_source,
    }


def counterfactual_command_invariance(
    model: nn.Module,
    observation: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, int | float | bool]:
    sample = observation[: min(2048, len(observation))].copy()
    rng = np.random.default_rng(seed)
    counterfactual = sample.copy()
    counterfactual[:, COMMAND_HISTORY_SLICE] = rng.uniform(
        -20.0,
        20.0,
        size=(len(counterfactual), COMMAND_HISTORY_SLICE.stop - COMMAND_HISTORY_SLICE.start),
    ).astype(np.float32)
    baseline = infer(model, sample, device, batch_size)
    changed = infer(model, counterfactual, device, batch_size)
    delta = np.abs(baseline - changed)
    return {
        "samples": int(len(sample)),
        "command_slice_start": int(COMMAND_HISTORY_SLICE.start),
        "command_slice_stop": int(COMMAND_HISTORY_SLICE.stop),
        "max_abs_probability_delta": float(delta.max(initial=0.0)),
        "mean_abs_probability_delta": float(delta.mean()) if len(delta) else 0.0,
        "exact": bool(np.array_equal(baseline, changed)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-spatial-hlh", type=Path, action="append", default=[])
    parser.add_argument("--train-spatial-lhl", type=Path, action="append", default=[])
    parser.add_argument("--train-switch", type=Path, action="append", default=[])
    parser.add_argument("--heldout-spatial-hlh", type=Path, action="append", default=[])
    parser.add_argument("--heldout-spatial-lhl", type=Path, action="append", default=[])
    parser.add_argument("--heldout-switch", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-washout-steps", type=int, default=15)
    # The switch collector's first saved policy frame is t=0.04 s.  At
    # t=0.30 s the 15-frame Hall window still contains one pre-switch frame;
    # t=0.32 s is the first completely post-switch observation.
    parser.add_argument("--transition-washout-s", type=float, default=0.32)
    parser.add_argument(
        "--switch-label-mode",
        choices=("mu", "prospective_slip_fall"),
        default="mu",
        help=(
            "mu preserves the offline friction classifier; prospective_slip_fall "
            "labels future contact-slip/fall and is the governor-compatible mode."
        ),
    )
    parser.add_argument("--future-horizon-s", type=float, default=0.24)
    parser.add_argument("--policy-dt-s", type=float, default=0.02)
    parser.add_argument("--contact-slip-threshold", type=float, default=0.045)
    parser.add_argument("--future-slip-quantile", type=float, default=0.75)
    parser.add_argument(
        "--allow-research-legacy-link-origin-slip",
        action="store_true",
        help=(
            "Explicitly allow old unversioned ankle/link-origin planar-speed "
            "labels for research comparison only. Default prospective training "
            "requires the corrected contact-point tangential-speed schema."
        ),
    )
    parser.add_argument(
        "--prospective-transition-washout-s",
        type=float,
        default=0.0,
        help=(
            "Optional prospective-mode negative-row washout. Positive future-event "
            "precursors are always retained so H->L onset is not hidden."
        ),
    )
    parser.add_argument("--low-mu-max", type=float, default=0.25)
    parser.add_argument("--high-mu-min", type=float, default=0.75)
    parser.add_argument("--operating-threshold", type=float, default=0.50)
    parser.add_argument("--offline-primary-command-min", type=float, default=0.50)
    parser.add_argument("--offline-min-primary-auc", type=float, default=0.85)
    parser.add_argument("--offline-min-primary-recall", type=float, default=0.80)
    parser.add_argument("--offline-max-primary-far", type=float, default=0.10)
    parser.add_argument("--offline-min-hall-auc-drop", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.transition_washout_steps < 0
        or args.transition_washout_s < 0.0
        or args.prospective_transition_washout_s < 0.0
    ):
        raise ValueError("transition washout values must be non-negative")
    if not args.low_mu_max < args.high_mu_min:
        raise ValueError("--low-mu-max must be less than --high-mu-min")
    if not 0.0 < args.operating_threshold < 1.0:
        raise ValueError("--operating-threshold must be in (0,1)")
    if args.offline_primary_command_min < 0.0:
        raise ValueError("--offline-primary-command-min must be non-negative")
    for name in (
        "offline_min_primary_auc",
        "offline_min_primary_recall",
        "offline_max_primary_far",
        "offline_min_hall_auc_drop",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= args.future_slip_quantile <= 1.0:
        raise ValueError("--future-slip-quantile must be in [0,1]")
    if args.switch_label_mode == "prospective_slip_fall" and (
        args.train_spatial_hlh
        or args.train_spatial_lhl
        or args.heldout_spatial_hlh
        or args.heldout_spatial_lhl
    ):
        raise ValueError(
            "prospective_slip_fall mode cannot mix stage/friction spatial labels; "
            "use reset-safe switch NPZs only"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"requested unavailable device: {device}")

    train_parts = [
        load_spatial_part(path, args.transition_washout_steps, "HLH")
        for path in args.train_spatial_hlh
    ] + [
        load_spatial_part(path, args.transition_washout_steps, "LHL")
        for path in args.train_spatial_lhl
    ] + [
        load_switch_part(
            path,
            args.transition_washout_s,
            args.low_mu_max,
            args.high_mu_min,
            label_mode=args.switch_label_mode,
            future_horizon_s=args.future_horizon_s,
            policy_dt_s=args.policy_dt_s,
            contact_slip_threshold=args.contact_slip_threshold,
            future_slip_quantile=args.future_slip_quantile,
            prospective_transition_washout_s=(
                args.prospective_transition_washout_s
            ),
            allow_research_legacy_link_origin_slip=(
                args.allow_research_legacy_link_origin_slip
            ),
        )
        for path in args.train_switch
    ]
    heldout_parts = [
        load_spatial_part(path, args.transition_washout_steps, "HLH")
        for path in args.heldout_spatial_hlh
    ] + [
        load_spatial_part(path, args.transition_washout_steps, "LHL")
        for path in args.heldout_spatial_lhl
    ] + [
        load_switch_part(
            path,
            args.transition_washout_s,
            args.low_mu_max,
            args.high_mu_min,
            label_mode=args.switch_label_mode,
            future_horizon_s=args.future_horizon_s,
            policy_dt_s=args.policy_dt_s,
            contact_slip_threshold=args.contact_slip_threshold,
            future_slip_quantile=args.future_slip_quantile,
            prospective_transition_washout_s=(
                args.prospective_transition_washout_s
            ),
            allow_research_legacy_link_origin_slip=(
                args.allow_research_legacy_link_origin_slip
            ),
        )
        for path in args.heldout_switch
    ]
    validate_strict_heldout(train_parts, heldout_parts)
    train = concatenate_parts(train_parts)
    heldout = concatenate_parts(heldout_parts)

    feature_mean, feature_scale = raw_feature_statistics(
        train["observation"], args.batch_size
    )
    model = CommandMaskedHallRiskEstimator(
        feature_mean,
        feature_scale,
        trailing_feature_mode=COMMAND_MASKED_TRAILING_FEATURE_MODE,
    ).to(device)
    sample_weight = source_class_balanced_weights(
        train["source_kind"], train["source_id"], train["target"]
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["observation"]),
            torch.from_numpy(train["target"]),
            torch.from_numpy(sample_weight),
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
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        rows = 0
        for observation, target, weight in loader:
            observation = observation.to(device)
            target = target.to(device)
            weight = weight.to(device)
            logit, _ = model.learned_logit(observation)
            per_sample = nn.functional.binary_cross_entropy_with_logits(
                logit.reshape(-1), target, reduction="none"
            )
            loss = (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(observation)
            rows += len(observation)
        scheduler.step()
        epoch_loss = loss_sum / max(rows, 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(
                f"epoch={epoch + 1:03d}/{args.epochs} loss={epoch_loss:.6f}",
                flush=True,
            )

    train_prediction = infer(model, train["observation"], device, args.batch_size)
    heldout_prediction = infer(
        model, heldout["observation"], device, args.batch_size
    )
    invariance = counterfactual_command_invariance(
        model,
        heldout["observation"],
        device,
        args.batch_size,
        args.seed + 1,
    )
    reliance = evidence_reliance_diagnostics(
        model,
        heldout,
        heldout_prediction,
        device,
        args.batch_size,
        args.seed + 10_000,
    )

    schema = command_masked_risk_schema()
    schema_sha = command_masked_risk_schema_sha256()
    risk_target = (
        PROSPECTIVE_RISK_TARGET
        if args.switch_label_mode == "prospective_slip_fall"
        else MU_CLASS_RISK_TARGET
    )
    legacy_slip_used = any(
        bool(part.audit.get("research_legacy_link_origin_slip", False))
        for part in train_parts + heldout_parts
    )
    risk_target_metadata = {
        "mode": args.switch_label_mode,
        "future_horizon_s": args.future_horizon_s,
        "future_horizon_steps": int(round(args.future_horizon_s / args.policy_dt_s)),
        "policy_dt_s": args.policy_dt_s,
        "window": (
            "collector pre-step observation aligned to current post-step outcome; "
            "t <= outcome < t+horizon"
        ),
        "complete_future_horizon_required": True,
        "grouping": "exact (rollout_id, env_id, phase); never cross phase/reset",
        "contact_slip_threshold": args.contact_slip_threshold,
        "future_slip_quantile": args.future_slip_quantile,
        "quantile_method": "lower",
        "fall_rule": "any future fall in the same complete horizon",
        "contact_slip_schema": (
            "research_legacy_link_origin"
            if legacy_slip_used
            else CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
        ),
        "contact_slip_metric_key": (
            "legacy/unversioned research source"
            if legacy_slip_used
            else CONTACT_POINT_TANGENTIAL_SLIP_KEY
        ),
        "contact_slip_valid_key": (
            "assumed true (legacy research only)"
            if legacy_slip_used
            else CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY
        ),
        "contact_slip_formula": (
            "legacy ankle/link-origin planar-speed proxy"
            if legacy_slip_used
            else CONTACT_POINT_TANGENTIAL_SLIP_FORMULA
        ),
        "legacy_link_origin_research_override": bool(
            args.allow_research_legacy_link_origin_slip
        ),
    }
    checkpoint_payload: dict[str, object] = {
        "model": model.state_dict(),
        "model_variant": COMMAND_MASKED_MODEL_VARIANT,
        "input_dim": INPUT_DIM,
        "trailing_feature_mode": COMMAND_MASKED_TRAILING_FEATURE_MODE,
        "masked_input_slices": {
            "command_history": [COMMAND_HISTORY_SLICE.start, COMMAND_HISTORY_SLICE.stop]
        },
        "observation_schema": schema,
        "observation_schema_sha256": schema_sha,
        "schema_sha256": schema_sha,
        "measurement_boundary": RUNTIME_MEASUREMENT_BOUNDARY,
        "risk_target": risk_target,
        "risk_target_metadata": risk_target_metadata,
        "training_split": {
            "train_seeds": sorted({part.seed for part in train_parts}),
            "heldout_seeds": sorted({part.seed for part in heldout_parts}),
        },
    }
    # Exercise the same strict factory used by downstream runtime code before
    # writing the artifact.
    restored = build_hall_risk_estimator(checkpoint_payload).eval().to(device)
    restore_probe_observation = heldout["observation"][
        : min(1024, len(heldout["observation"]))
    ]
    restore_reference_prediction = infer(
        model, restore_probe_observation, device, args.batch_size
    )
    restored_prediction = infer(
        restored, restore_probe_observation, device, args.batch_size
    )
    strict_restore_delta = float(
        np.max(
            np.abs(
                restored_prediction - restore_reference_prediction
            ),
            initial=0.0,
        )
    )
    offline_gate = offline_research_gate(
        heldout,
        heldout_prediction,
        reliance,
        threshold=args.operating_threshold,
        primary_command_min=args.offline_primary_command_min,
        min_primary_auc=args.offline_min_primary_auc,
        min_primary_recall=args.offline_min_primary_recall,
        max_primary_false_alarm_rate=args.offline_max_primary_far,
        min_hall_auc_drop=args.offline_min_hall_auc_drop,
        command_invariance_exact=bool(invariance["exact"]),
        strict_restore_delta=strict_restore_delta,
    )
    if (
        args.switch_label_mode == "prospective_slip_fall"
        and legacy_slip_used
    ):
        offline_gate["passed"] = False
        offline_gate["failures"].append(
            "research-only legacy link-origin slip override is enabled"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "hall_command_masked_risk_estimator.pt"
    torch.save(checkpoint_payload, checkpoint)
    np.savez_compressed(
        args.output_dir / "heldout_predictions.npz",
        prediction=heldout_prediction.astype(np.float32),
        target=heldout["target"].astype(np.float32),
        source_kind=heldout["source_kind"],
        source_id=heldout["source_id"],
        seed=heldout["seed"],
        command_vx=heldout["command_vx"].astype(np.float32),
        phase=heldout["phase"].astype(np.int64),
    )
    summary = {
        "status": "RESEARCH_SMOKE_NOT_RELEASE",
        "release_candidate": False,
        "reason": (
            "offline heldout classification is necessary but not sufficient; "
            "closed-loop Isaac, fault, MuJoCo and hardware gates remain"
        ),
        "model_variant": COMMAND_MASKED_MODEL_VARIANT,
        "input_dim": INPUT_DIM,
        "trailing_feature_mode": COMMAND_MASKED_TRAILING_FEATURE_MODE,
        "masked_input_slices": checkpoint_payload["masked_input_slices"],
        "observation_schema_sha256": schema_sha,
        "measurement_boundary": checkpoint_payload["measurement_boundary"],
        "risk_target": risk_target,
        "risk_target_metadata": risk_target_metadata,
        "offline_label_policy": {
            "spatial_hlh": "risk iff fastbase_course_stage == 1",
            "spatial_lhl": "risk iff fastbase_course_stage in {0,2}",
            "switch_label_mode": args.switch_label_mode,
            "switch_mu": (
                f"risk iff mu <= {args.low_mu_max:g}; safe iff mu >= "
                f"{args.high_mu_min:g}; intermediate mu excluded"
            ),
            "switch_prospective": risk_target_metadata,
            "transition_washout_steps": args.transition_washout_steps,
            "transition_washout_s": args.transition_washout_s,
            "prospective_transition_washout_s": (
                args.prospective_transition_washout_s
            ),
            "both_valid_training_mask": True,
            "allow_research_legacy_link_origin_slip": bool(
                args.allow_research_legacy_link_origin_slip
            ),
        },
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": args.seed,
            "device": str(device),
            "heldout_never_used_for_statistics_or_epoch_selection": True,
        },
        "code_provenance": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "model_source": str(
                (SOURCE / "unitree_rl_lab" / "traction" / "hall_risk_estimator.py").resolve()
            ),
            "model_source_sha256": sha256(
                SOURCE / "unitree_rl_lab" / "traction" / "hall_risk_estimator.py"
            ),
        },
        "strict_split": {
            "unit": "whole source file and whole seed; no row-level split",
            "train_seeds": sorted({part.seed for part in train_parts}),
            "heldout_seeds": sorted({part.seed for part in heldout_parts}),
            "seed_overlap": [],
        },
        "source_class_weight_mass": weight_mass_report(
            train["source_kind"],
            train["source_id"],
            train["target"],
            sample_weight,
        ),
        "train_samples": int(len(train["target"])),
        "heldout_samples": int(len(heldout["target"])),
        "train": metrics_by_source(
            train, train_prediction, args.operating_threshold
        ),
        "heldout": metrics_by_source(
            heldout, heldout_prediction, args.operating_threshold
        ),
        "counterfactual_command_invariance": invariance,
        "heldout_evidence_reliance": reliance,
        "offline_research_gate": offline_gate,
        "strict_factory_restore_max_abs_delta": strict_restore_delta,
        "operating_threshold": args.operating_threshold,
        "train_source_audit": {
            part.source_id: part.audit for part in train_parts
        },
        "heldout_source_audit": {
            part.source_id: part.audit for part in heldout_parts
        },
        "sources": {
            "train": [
                {
                    "path": str(part.path),
                    "sha256": sha256(part.path),
                    "kind": part.source_kind,
                    "seed": part.seed,
                }
                for part in train_parts
            ],
            "heldout": [
                {
                    "path": str(part.path),
                    "sha256": sha256(part.path),
                    "kind": part.source_kind,
                    "seed": part.seed,
                }
                for part in heldout_parts
            ],
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_s": time.time() - started,
        "history": history,
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
