#!/usr/bin/env python3
"""Build a leak-free V2 HighEnd recovery bank from aligned Isaac state dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from unitree_rl_lab.traction.high_end_state_bank import (
    LOCKED_ACCEPTANCE_SEEDS,
    SCHEMA_VERSION,
    TRAINING_ROLE,
    VALIDATION_ROLE,
    load_high_end_state_bank,
)


DUMP_SCHEMA = "high_end_recovery_state_dump.v2"
ROLE_MAP = {
    TRAINING_ROLE: "training_high_end_state_dump",
    VALIDATION_ROLE: "validation_high_end_state_dump",
}

PASS_THROUGH_FIELDS = (
    "root_pose_local",
    "root_velocity",
    "joint_pos",
    "joint_vel",
    "observation",
    "motion_feedback_initial_yaw",
    "straight_heading_reference_xy",
    "straight_track_origin_local_xy",
    "straight_track_lateral_axis",
    "hall_local_deformation",
    "hall_loading_history",
    "hall_signal_filtered_absolute",
    "hall_signal_processed",
    "hall_signal_baseline",
    "hall_signal_drift",
    "hall_policy_history",
    "hall_policy_gain",
    "hall_policy_cross_axis",
    "hall_policy_zero_residual",
    "hall_policy_channel_keep",
    "hall_policy_foot_keep",
    "hall_policy_delay_steps",
    "hall_reported_sample_period",
    "source_seed",
    "source_env_id",
    "source_rollout_step",
    "time_to_fall_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    if "metadata_json" not in payload.files:
        raise ValueError(f"{path}: missing metadata_json")
    raw = np.asarray(payload["metadata_json"])
    if raw.shape != ():
        raise ValueError(f"{path}: metadata_json must be scalar")
    try:
        metadata = json.loads(str(raw.item()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid metadata_json") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must decode to an object")
    return metadata


def _load_dump(
    path: Path,
    *,
    expected_role: str,
    locked_seeds: set[int],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        metadata = _read_metadata(payload, resolved)
        if metadata.get("schema_version") != DUMP_SCHEMA:
            raise ValueError(
                f"{resolved}: expected {DUMP_SCHEMA}, got "
                f"{metadata.get('schema_version')!r}"
            )
        if metadata.get("dataset_role") != expected_role:
            raise ValueError(
                f"{resolved}: expected dataset_role={expected_role!r}, got "
                f"{metadata.get('dataset_role')!r}"
            )
        arrays: dict[str, np.ndarray] = {}
        count: int | None = None
        for name in (*PASS_THROUGH_FIELDS, "source_episode_fall"):
            if name not in payload.files:
                raise ValueError(f"{resolved}: missing {name}")
            value = np.asarray(payload[name])
            if value.ndim < 1:
                raise ValueError(f"{resolved}: {name} must have a row dimension")
            count = value.shape[0] if count is None else count
            if value.shape[0] != count:
                raise ValueError(f"{resolved}: row count mismatch at {name}")
            if name != "source_episode_fall" and (
                not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{resolved}: {name} has invalid dtype/data")
            arrays[name] = value
    assert count is not None
    if count <= 0:
        raise ValueError(f"{resolved}: empty state dump")
    source_seeds = set(np.unique(arrays["source_seed"].astype(np.int64)).tolist())
    declared = metadata.get("source_seeds")
    if not isinstance(declared, list) or source_seeds != {int(v) for v in declared}:
        raise ValueError(f"{resolved}: source seed rows/metadata disagree")
    leaked = source_seeds & locked_seeds
    if leaked:
        raise ValueError(f"{resolved}: locked seed leakage {sorted(leaked)}")
    metadata = dict(metadata)
    metadata["source_file"] = str(resolved)
    metadata["source_sha256"] = _sha256(resolved)
    return arrays, metadata


def _temporal_thin(
    indices: np.ndarray,
    *,
    source_seed: np.ndarray,
    source_env: np.ndarray,
    source_step: np.ndarray,
    minimum_gap_steps: int,
) -> np.ndarray:
    order = np.lexsort((source_step[indices], source_env[indices], source_seed[indices]))
    selected: list[int] = []
    last: dict[tuple[int, int], int] = {}
    for index in indices[order].tolist():
        key = (int(source_seed[index]), int(source_env[index]))
        step = int(source_step[index])
        if key not in last or step - last[key] >= minimum_gap_steps:
            selected.append(index)
            last[key] = step
    return np.asarray(selected, dtype=np.int64)


def _choose(
    arrays: dict[str, np.ndarray],
    *,
    near_fraction: float,
    nominal_min_future_fall_s: float,
    max_states: int,
    minimum_gap_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    obs = arrays["observation"].astype(np.float32, copy=False)
    pose = arrays["root_pose_local"].astype(np.float32, copy=False)
    velocity = arrays["root_velocity"].astype(np.float32, copy=False)
    ttf = arrays["time_to_fall_s"].astype(np.float32, copy=False)
    healthy = (obs[:, 1860:1862] > 0.5).all(axis=1)
    safe_height = (pose[:, 2] >= 0.58) & (pose[:, 2] <= 0.90)
    finite_speed = np.linalg.norm(velocity, axis=1) < 8.0
    common = healthy & safe_height & finite_speed
    near = common & (ttf >= 0.50) & (ttf <= 3.00)
    nominal = (
        common
        # A backbone that eventually late-falls may have no episode labelled
        # globally successful, even though its early HighEnd states are clean
        # and remain stable for several seconds.  Treat those far-pre-failure
        # rows as retention states instead of forcing the bank to learn only
        # from the terminal basin.  This is a training label, never an
        # acceptance claim: locked long-horizon rollouts still require zero
        # falls over the full course.
        & ((ttf < 0.0) | (ttf >= nominal_min_future_fall_s))
        & (np.abs(obs[:, 1862]) <= 0.40)
        & (np.abs(obs[:, 1863]) <= 0.30)
    )
    seed = arrays["source_seed"].astype(np.int64, copy=False)
    env = arrays["source_env_id"].astype(np.int64, copy=False)
    step = arrays["source_rollout_step"].astype(np.int64, copy=False)
    near_ids = _temporal_thin(
        np.flatnonzero(near),
        source_seed=seed,
        source_env=env,
        source_step=step,
        minimum_gap_steps=minimum_gap_steps,
    )
    nominal_ids = _temporal_thin(
        np.flatnonzero(nominal),
        source_seed=seed,
        source_env=env,
        source_step=step,
        minimum_gap_steps=minimum_gap_steps,
    )
    if not len(near_ids):
        raise ValueError("no 0.5--3.0 s pre-fall HighEnd states survived filtering")
    if not len(nominal_ids):
        raise ValueError("no nominal HighEnd states survived filtering")
    # Preserve the requested class mixture even when max_states exceeds the
    # available thinned rows.  The previous independent min() calls silently
    # changed a requested 70/30 bank into 40/60 whenever near-failure rows
    # were the limiting class.  Uniform reset sampling then trained mostly on
    # easy retention states, contradicting the recovery curriculum.
    maximum_total_from_near = int(np.floor(len(near_ids) / near_fraction))
    maximum_total_from_nominal = int(
        np.floor(len(nominal_ids) / (1.0 - near_fraction))
    )
    feasible_total = min(
        int(max_states), maximum_total_from_near, maximum_total_from_nominal
    )
    if feasible_total < 2:
        raise ValueError("not enough states to preserve the requested near/nominal ratio")
    near_take = int(round(feasible_total * near_fraction))
    nominal_take = feasible_total - near_take
    if near_take < 1 or nominal_take < 1:
        raise RuntimeError("requested class mixture produced an empty state class")
    if near_take > len(near_ids) or nominal_take > len(nominal_ids):
        # Round-to-nearest can exceed the limiting class by one.  Reduce the
        # total until both class counts fit; never silently distort the ratio.
        while feasible_total >= 2:
            feasible_total -= 1
            near_take = int(round(feasible_total * near_fraction))
            nominal_take = feasible_total - near_take
            if 0 < near_take <= len(near_ids) and 0 < nominal_take <= len(nominal_ids):
                break
        else:
            raise RuntimeError("failed to construct a feasible mixed state bank")
    near_selected = rng.choice(near_ids, size=near_take, replace=False)
    nominal_selected = rng.choice(nominal_ids, size=nominal_take, replace=False)
    selected = np.concatenate((near_selected, nominal_selected))
    kinds = np.concatenate(
        (
            np.ones(near_take, dtype=np.int64),
            np.zeros(nominal_take, dtype=np.int64),
        )
    )
    permutation = rng.permutation(len(selected))
    return selected[permutation], kinds[permutation], {
        "near_available": int(len(near_ids)),
        "nominal_available": int(len(nominal_ids)),
        "near_selected": int(near_take),
        "nominal_selected": int(nominal_take),
        "selected_near_fraction": float(near_take / (near_take + nominal_take)),
        "requested_near_fraction": float(near_fraction),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-role",
        choices=(TRAINING_ROLE, VALIDATION_ROLE),
        default=TRAINING_ROLE,
    )
    parser.add_argument("--locked-seed", type=int, action="append", default=None)
    parser.add_argument("--near-fraction", type=float, default=0.70)
    parser.add_argument(
        "--nominal-min-future-fall-s",
        type=float,
        default=4.0,
        help=(
            "A HighEnd row is eligible as a retention/nominal state when its "
            "episode never falls or the observed fall is at least this many "
            "seconds in the future. This does not change evaluation labels."
        ),
    )
    parser.add_argument("--max-states", type=int, default=4096)
    parser.add_argument("--minimum-gap-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.near_fraction < 1.0:
        raise ValueError("--near-fraction must be in (0,1)")
    if not np.isfinite(args.nominal_min_future_fall_s) or args.nominal_min_future_fall_s <= 3.0:
        raise ValueError("--nominal-min-future-fall-s must be finite and >3.0")
    if args.max_states < 2 or args.minimum_gap_steps < 1:
        raise ValueError("--max-states>=2 and --minimum-gap-steps>=1 are required")
    locked = set(args.locked_seed or LOCKED_ACCEPTANCE_SEEDS)
    source_arrays: list[dict[str, np.ndarray]] = []
    source_metadata: list[dict[str, object]] = []
    for path in args.input:
        arrays, metadata = _load_dump(
            path,
            expected_role=ROLE_MAP[args.dataset_role],
            locked_seeds=locked,
        )
        source_arrays.append(arrays)
        source_metadata.append(metadata)
    merged = {
        name: np.concatenate([arrays[name] for arrays in source_arrays], axis=0)
        for name in (*PASS_THROUGH_FIELDS, "source_episode_fall")
    }
    selected, state_kind, selection = _choose(
        merged,
        near_fraction=float(args.near_fraction),
        nominal_min_future_fall_s=float(args.nominal_min_future_fall_s),
        max_states=int(args.max_states),
        minimum_gap_steps=int(args.minimum_gap_steps),
        rng=np.random.default_rng(args.seed),
    )
    source_seeds = sorted(
        set(merged["source_seed"][selected].astype(np.int64).tolist())
    )
    if set(source_seeds) & locked:
        raise RuntimeError("locked seed reached final selected rows")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_role": args.dataset_role,
        "source_seeds": source_seeds,
        "excluded_locked_seeds": sorted(locked),
        "builder_seed": int(args.seed),
        "selection": {
            **selection,
            "near_definition": "0.50 <= future physical fall time <= 3.00 s",
            "nominal_definition": (
                "non-fall episode OR future fall time >= "
                f"{float(args.nominal_min_future_fall_s):.3f} s, "
                "with |body_vy|<=0.40 and |relative_heading|<=0.30; "
                "training retention label only, not an acceptance success"
            ),
            "minimum_gap_steps_per_source_env": int(args.minimum_gap_steps),
        },
        "sources": source_metadata,
        "actor_observation_dim": 1864,
        "actor_uses_force_contact_mu_slip_or_stage": False,
        "measurement_boundary": (
            "Hall histories are Bx/By/Bz plus packet period/validity; no force conversion"
        ),
    }
    output = {
        name: merged[name][selected]
        for name in PASS_THROUGH_FIELDS
    }
    output["state_kind"] = state_kind
    output["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(args.output)
    # The same strict loader used by Isaac reset is the final builder gate.
    validated = load_high_end_state_bank(
        args.output,
        device="cpu",
        allowed_roles=(args.dataset_role,),
        locked_acceptance_seeds=locked,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
                "samples": validated.sample_count,
                "source_seeds": source_seeds,
                "selection": selection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
