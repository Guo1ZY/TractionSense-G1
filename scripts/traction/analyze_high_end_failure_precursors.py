#!/usr/bin/env python3
"""Align locked HighEnd rollouts and quantify late-fall precursors.

This tool is diagnostic-only.  It refuses inputs that are not explicitly
marked ``locked_evaluation_only_do_not_train`` and never writes a state bank or
checkpoint.  Rows are expected to be pre-action state/observation at t with
fall/done labels from the transition caused by action[t].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICY_DIM = 1864
ACTION_DIM = 29
MOTION_VY_INDEX = 1862
MOTION_HEADING_INDEX = 1863
LEFT_LEG_ACTIONS = np.asarray((0, 3, 6, 9, 13, 17), dtype=np.int64)
RIGHT_LEG_ACTIONS = np.asarray((1, 4, 7, 10, 14, 18), dtype=np.int64)
REQUIRED_KEYS = (
    "metadata_json",
    "failure_observation",
    "failure_action",
    "failure_env_id",
    "failure_rollout_step",
    "failure_time_s",
    "failure_root_pose_local",
    "failure_root_lin_vel_b",
    "failure_root_ang_vel_b",
    "failure_foot_contact_lr",
    "failure_course_stage",
    "failure_effective_hall_gate",
    "failure_capture_delta_l2",
    "failure_stability_delta_l2",
    "failure_fall",
    "failure_done",
    "failure_time_out",
)


def _rolling_rms(value: np.ndarray, window: int) -> np.ndarray:
    if value.ndim != 1 or window <= 0:
        raise ValueError("rolling RMS requires a 1-D signal and positive window")
    squared = np.square(value.astype(np.float64))
    prefix = np.concatenate((np.zeros(1), np.cumsum(squared)))
    result = np.empty_like(squared)
    for index in range(value.size):
        start = max(0, index - window + 1)
        result[index] = math.sqrt(
            max((prefix[index + 1] - prefix[start]) / (index + 1 - start), 0.0)
        )
    return result.astype(np.float32)


def _quat_wxyz_to_roll_pitch(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape [N,4] in wxyz order")
    w, x, y, z = quaternion.T.astype(np.float64)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return roll.astype(np.float32), pitch.astype(np.float32)


def _persistent_first(mask: np.ndarray, count: int) -> int | None:
    if mask.ndim != 1 or count <= 0:
        raise ValueError("persistent crossing requires 1-D mask and positive count")
    if mask.size < count:
        return None
    run = np.convolve(mask.astype(np.int16), np.ones(count, dtype=np.int16), mode="valid")
    indices = np.flatnonzero(run == count)
    return int(indices[0]) if indices.size else None


def _robust_success_threshold(value: np.ndarray, floor: float) -> float:
    finite = value[np.isfinite(value)]
    if finite.size < 10:
        raise ValueError("not enough successful HighEnd rows to set a threshold")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_upper = median + 6.0 * 1.4826 * mad
    return max(float(floor), float(np.quantile(finite, 0.99)), robust_upper)


def load_trace(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_KEYS) - set(payload.files))
        if missing:
            raise KeyError(f"{path}: missing required arrays {missing}")
        arrays = {name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"}
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("schema_version") != "high_end_failure_precursor_trace.v1":
        raise ValueError("unsupported failure trace schema")
    if metadata.get("dataset_role") != "locked_evaluation_only_do_not_train":
        raise ValueError("precursor analysis requires a locked evaluation trace")
    observation = arrays["failure_observation"]
    action = arrays["failure_action"]
    if observation.ndim != 2 or observation.shape[1] != POLICY_DIM:
        raise ValueError(f"observation must be [N,{POLICY_DIM}]")
    if action.shape != (observation.shape[0], ACTION_DIM):
        raise ValueError(f"action must be [N,{ACTION_DIM}]")
    row_count = observation.shape[0]
    for name in REQUIRED_KEYS[3:]:
        if arrays[name].shape[0] != row_count:
            raise ValueError(f"{name}: row count is not aligned")
    for name, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise FloatingPointError(f"{name}: NaN/Inf in locked trace")
    if not np.array_equal(
        observation[:, MOTION_VY_INDEX],
        np.clip(arrays["failure_root_lin_vel_b"][:, 1], -1.5, 1.5),
    ):
        raise RuntimeError("obs[1862] no longer matches clipped body_vy")
    pairs = np.stack(
        (arrays["failure_env_id"], arrays["failure_rollout_step"]), axis=1
    )
    if np.unique(pairs, axis=0).shape[0] != row_count:
        raise RuntimeError("duplicate (env_id, rollout_step) rows")
    for env_id in np.unique(arrays["failure_env_id"]):
        steps = arrays["failure_rollout_step"][arrays["failure_env_id"] == env_id]
        if steps.size > 1 and not np.all(np.diff(steps) == 1):
            raise RuntimeError(f"env {int(env_id)} contains a step gap")
    return arrays, metadata


def derive_signals(arrays: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    observation = arrays["failure_observation"]
    action = arrays["failure_action"]
    env_ids = arrays["failure_env_id"]
    roll, pitch = _quat_wxyz_to_roll_pitch(arrays["failure_root_pose_local"][:, 3:7])
    result: dict[str, np.ndarray] = {
        "heading": observation[:, MOTION_HEADING_INDEX].astype(np.float32),
        "body_vx": arrays["failure_root_lin_vel_b"][:, 0].astype(np.float32),
        "body_vy": arrays["failure_root_lin_vel_b"][:, 1].astype(np.float32),
        "omega_x": arrays["failure_root_ang_vel_b"][:, 0].astype(np.float32),
        "omega_y": arrays["failure_root_ang_vel_b"][:, 1].astype(np.float32),
        "omega_z": arrays["failure_root_ang_vel_b"][:, 2].astype(np.float32),
        "omega_norm": np.linalg.norm(arrays["failure_root_ang_vel_b"], axis=1).astype(np.float32),
        "roll": roll,
        "pitch": pitch,
        "left_action_norm": np.linalg.norm(action[:, LEFT_LEG_ACTIONS], axis=1).astype(np.float32),
        "right_action_norm": np.linalg.norm(action[:, RIGHT_LEG_ACTIONS], axis=1).astype(np.float32),
        "action_saturation_ratio": np.mean(np.abs(action) >= 2.9, axis=1).astype(np.float32),
        "contact_left": arrays["failure_foot_contact_lr"][:, 0].astype(np.float32),
        "contact_right": arrays["failure_foot_contact_lr"][:, 1].astype(np.float32),
        "hall_gate": arrays["failure_effective_hall_gate"].astype(np.float32),
        "hall_residual_l2": arrays["failure_capture_delta_l2"].astype(np.float32),
        "stability_residual_l2": arrays["failure_stability_delta_l2"].astype(np.float32),
    }
    row_count = action.shape[0]
    for name in ("heading_dot", "body_vy_dot", "action_slew", "cadence_left", "cadence_right"):
        result[name] = np.zeros(row_count, dtype=np.float32)
    rms_window = max(1, int(round(0.50 / dt)))
    cadence_window = max(1, int(round(1.00 / dt)))
    result["heading_rms_0p5s"] = np.zeros(row_count, dtype=np.float32)
    result["body_vy_rms_0p5s"] = np.zeros(row_count, dtype=np.float32)
    for env_id in np.unique(env_ids):
        indices = np.flatnonzero(env_ids == env_id)
        result["heading_dot"][indices] = np.gradient(result["heading"][indices], dt).astype(np.float32)
        result["body_vy_dot"][indices] = np.gradient(result["body_vy"][indices], dt).astype(np.float32)
        delta = np.diff(action[indices], axis=0, prepend=action[indices[:1]])
        result["action_slew"][indices] = np.linalg.norm(delta, axis=1).astype(np.float32)
        result["heading_rms_0p5s"][indices] = _rolling_rms(result["heading"][indices], rms_window)
        result["body_vy_rms_0p5s"][indices] = _rolling_rms(result["body_vy"][indices], rms_window)
        for side, key in ((0, "cadence_left"), (1, "cadence_right")):
            contact = arrays["failure_foot_contact_lr"][indices, side].astype(np.int8)
            touchdown = np.maximum(contact - np.concatenate(([contact[0]], contact[:-1])), 0)
            rate = np.convolve(touchdown, np.ones(cadence_window), mode="full")[: contact.size]
            result[key][indices] = (rate / (cadence_window * dt)).astype(np.float32)
    return result


def analyze(
    arrays: dict[str, np.ndarray], metadata: dict[str, object], window_s: float, persistence: int
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    dt = float(metadata["policy_dt_s"])
    signals = derive_signals(arrays, dt)
    env_ids = arrays["failure_env_id"]
    failed_envs = sorted(int(v) for v in np.unique(env_ids[arrays["failure_fall"]]))
    all_envs = sorted(int(v) for v in np.unique(env_ids))
    success_envs = sorted(set(all_envs) - set(failed_envs))
    if not failed_envs or not success_envs:
        raise RuntimeError("analysis requires at least one failed and one successful rollout")
    success_high = np.isin(env_ids, success_envs) & (arrays["failure_course_stage"] == 2)
    threshold_specs = {
        "heading_abs": (np.abs(signals["heading"]), 0.25),
        "vy_abs": (np.abs(signals["body_vy"]), 0.30),
        "omega_norm": (signals["omega_norm"], 1.00),
        "action_slew": (signals["action_slew"], 1.00),
        "action_saturation_ratio": (signals["action_saturation_ratio"], 2.0 / ACTION_DIM),
    }
    thresholds = {
        name: _robust_success_threshold(value[success_high], floor)
        for name, (value, floor) in threshold_specs.items()
    }
    per_env: list[dict[str, object]] = []
    lead_times: dict[str, list[float]] = {name: [] for name in thresholds}
    for env_id in all_envs:
        indices = np.flatnonzero(env_ids == env_id)
        fall_local = np.flatnonzero(arrays["failure_fall"][indices])
        terminal_local = np.flatnonzero(arrays["failure_done"][indices])
        terminal_index = int(fall_local[0] if fall_local.size else terminal_local[-1])
        terminal_step = int(arrays["failure_rollout_step"][indices[terminal_index]])
        window_start_step = terminal_step - int(round(window_s / dt))
        eligible = (
            (arrays["failure_rollout_step"][indices] >= window_start_step)
            & (arrays["failure_course_stage"][indices] == 2)
            & (np.arange(indices.size) <= terminal_index)
        )
        row: dict[str, object] = {
            "env_id": env_id,
            "failed": bool(fall_local.size),
            "terminal_step": terminal_step,
            "terminal_time_s": float(arrays["failure_time_s"][indices[terminal_index]]),
        }
        for name, threshold in thresholds.items():
            value = threshold_specs[name][0][indices]
            local_candidates = np.flatnonzero(eligible)
            first = _persistent_first(value[local_candidates] > threshold, persistence)
            trigger_step = None
            lead_s = None
            if first is not None:
                trigger_local = int(local_candidates[first])
                trigger_step = int(arrays["failure_rollout_step"][indices[trigger_local]])
                lead_s = (terminal_step - trigger_step) * dt
                if fall_local.size:
                    lead_times[name].append(float(lead_s))
            row[f"{name}_threshold"] = float(threshold)
            row[f"{name}_trigger_step"] = trigger_step
            row[f"{name}_lead_s"] = lead_s
        high = eligible
        for name in ("heading", "body_vy", "omega_norm", "action_slew", "action_saturation_ratio"):
            values = signals[name][indices][high]
            row[f"{name}_rms"] = float(np.sqrt(np.mean(np.square(values)))) if values.size else None
        per_env.append(row)
    aggregate = {}
    for name, values in lead_times.items():
        aggregate[name] = {
            "detected_failures": len(values),
            "failure_count": len(failed_envs),
            "median_lead_s": float(np.median(values)) if values else None,
            "min_lead_s": float(np.min(values)) if values else None,
            "max_lead_s": float(np.max(values)) if values else None,
            "success_false_triggers": sum(
                row[f"{name}_trigger_step"] is not None for row in per_env if not row["failed"]
            ),
        }
    causal_order = sorted(
        (
            {"signal": name, **stats}
            for name, stats in aggregate.items()
            if stats["median_lead_s"] is not None
        ),
        key=lambda row: float(row["median_lead_s"]),
        reverse=True,
    )
    summary = {
        "definition": "locked-high-end-failure-precursor-analysis.v1",
        "source_metadata": metadata,
        "failed_envs": failed_envs,
        "successful_envs": success_envs,
        "analysis_window_s": window_s,
        "persistence_steps": persistence,
        "persistence_s": persistence * dt,
        "threshold_method": "max(success HighEnd q99, median+6*MAD, physical floor)",
        "thresholds": thresholds,
        "precursor_detection": aggregate,
        "causal_order_by_median_lead": causal_order,
        "warning": "exploratory 7-failure/1-success locked diagnostic; thresholds require held-out validation",
    }
    return summary, per_env, signals


def plot_aligned(
    arrays: dict[str, np.ndarray], signals: dict[str, np.ndarray], output: Path, window_s: float
) -> None:
    env_ids = arrays["failure_env_id"]
    failed = set(int(v) for v in np.unique(env_ids[arrays["failure_fall"]]))
    dt = float(np.median(np.diff(np.unique(arrays["failure_time_s"]))))
    grid = np.arange(-window_s, 0.5 * dt, dt)

    def aligned(name: str, failed_only: bool) -> list[np.ndarray]:
        rows = []
        for env_id in sorted(int(v) for v in np.unique(env_ids)):
            if (env_id in failed) != failed_only:
                continue
            idx = np.flatnonzero(env_ids == env_id)
            terminal = np.flatnonzero(arrays["failure_done"][idx])[-1]
            t = arrays["failure_time_s"][idx] - arrays["failure_time_s"][idx[terminal]]
            value = signals[name][idx]
            rows.append(np.interp(grid, t, value, left=np.nan, right=np.nan))
        return rows

    panels = (
        (("heading", "heading_rms_0p5s"), "heading [rad]"),
        (("body_vx", "body_vy"), "body velocity [m/s]"),
        (("omega_x", "omega_y", "omega_z"), "body angular velocity [rad/s]"),
        (("roll", "pitch"), "attitude [rad]"),
        (("left_action_norm", "right_action_norm"), "leg action L2"),
        (("action_slew", "action_saturation_ratio"), "action change / saturation"),
        (("contact_left", "contact_right", "cadence_left", "cadence_right"), "foot contact / cadence"),
        (("hall_gate", "hall_residual_l2", "stability_residual_l2"), "Hall/residual diagnostics"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    colors = ("#c43c39", "#2864b7", "#6b4ba1", "#d17a22")
    for axis, (names, ylabel) in zip(axes.ravel(), panels):
        for signal_index, name in enumerate(names):
            failed_rows = aligned(name, True)
            success_rows = aligned(name, False)
            for row in failed_rows:
                axis.plot(grid, row, color=colors[signal_index % len(colors)], alpha=0.12, lw=0.7)
            if failed_rows:
                axis.plot(
                    grid, np.nanmedian(np.stack(failed_rows), axis=0),
                    color=colors[signal_index % len(colors)], lw=2.0,
                    label=f"failed median: {name}",
                )
            for row in success_rows:
                axis.plot(grid, row, color="#248f55", ls="--", lw=1.5, label=f"success: {name}")
        axis.axvline(0.0, color="black", lw=1.0)
        axis.grid(alpha=0.25)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7, ncol=2, loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("time relative to fall/success terminal [s]")
    figure.suptitle(
        "Locked HighEnd precursor alignment: 7 failures vs 1 success\n"
        "thin=individual failures, solid=failed median, green dashed=success",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_s", type=float, default=4.0)
    parser.add_argument("--persistence_steps", type=int, default=5)
    args = parser.parse_args()
    if args.window_s <= 0.0 or args.persistence_steps <= 0:
        parser.error("window_s and persistence_steps must be positive")
    arrays, metadata = load_trace(args.input.expanduser().resolve())
    summary, per_env, signals = analyze(
        arrays, metadata, args.window_s, args.persistence_steps
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_env.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_env[0]))
        writer.writeheader()
        writer.writerows(per_env)
    plot_aligned(arrays, signals, output_dir / "aligned_precursors.png", args.window_s)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
