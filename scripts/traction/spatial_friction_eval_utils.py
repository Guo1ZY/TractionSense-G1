"""Pure helpers shared by the spatial-friction Isaac evaluator and tests.

This module deliberately has no Isaac Sim dependency.  Keeping the transition
tracker pure makes it possible to test the high--low--high accounting without
starting Kit, while the evaluator performs the actual USD/PhysX checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median
from typing import Sequence


WAIT_HIGH_START = 0
WAIT_LOW = 1
WAIT_HIGH_END = 2
COMPLETE = 3

# These labels describe the physical course region occupied by an observation
# row.  They are evaluator/dataset metadata, never deployable actor inputs.
CAPTURE_DIAGNOSTIC_STAGE_NAMES = ("HIGH_START", "LOW", "HIGH_END")


@dataclass(frozen=True)
class SpatialTransitionSample:
    """One privileged evaluation sample; never an actor observation."""

    local_x: float
    low_contact: bool
    done: bool = False
    high_start_contact: bool = True
    high_end_contact: bool = True


def advance_high_low_high_stage(
    stage: int,
    sample: SpatialTransitionSample,
    *,
    low_start_x: float = 0.0,
    high_end_x: float = 1.0,
) -> int:
    """Advance a causal high--low--high rollout state machine.

    A terminal/reset sample clears partial progress.  This prevents a fall on
    the orange patch followed by an episode reset on blue from being counted
    as a successful low-to-high recovery.
    """

    if stage not in (WAIT_HIGH_START, WAIT_LOW, WAIT_HIGH_END, COMPLETE):
        raise ValueError(f"invalid high--low--high stage: {stage}")
    if high_end_x <= low_start_x:
        raise ValueError("high_end_x must be greater than low_start_x")
    if sample.done:
        return WAIT_HIGH_START
    if stage == COMPLETE:
        return COMPLETE
    if stage == WAIT_HIGH_START:
        if (
            sample.local_x < low_start_x
            and not sample.low_contact
            and sample.high_start_contact
        ):
            return WAIT_LOW
        return stage
    if stage == WAIT_LOW:
        if sample.low_contact:
            return WAIT_HIGH_END
        return stage
    if (
        sample.local_x >= high_end_x
        and not sample.low_contact
        and sample.high_end_contact
    ):
        return COMPLETE
    return stage


def compress_contact_labels(labels: list[bool]) -> list[str]:
    """Compress boolean low-contact frames into human-readable regimes."""

    result: list[str] = []
    for low_contact in labels:
        name = "LOW" if low_contact else "HIGH"
        if not result or result[-1] != name:
            result.append(name)
    return result


def classify_hall_health(
    channel_keep: Sequence[Sequence[bool]],
    foot_keep: Sequence[bool],
) -> str:
    """Classify one episode's sampled Hall fault state.

    Delay is intentionally reported as a separate severity field: a delayed
    but otherwise complete array is not the same failure mode as missing
    spatial channels.  ``channel_keep`` is expected to contain two feet with
    one boolean per Hall package.
    """

    if len(channel_keep) != 2 or len(foot_keep) != 2:
        raise ValueError("Hall health inputs must contain exactly left/right feet")
    if not channel_keep[0] or len(channel_keep[0]) != len(channel_keep[1]):
        raise ValueError("left/right Hall channel arrays must be non-empty and equal length")
    online_feet = sum(bool(value) for value in foot_keep)
    if online_feet == 0:
        return "both_feet_offline"
    if online_feet == 1:
        return "single_foot_offline"
    live_channels = sum(bool(value) for foot in channel_keep for value in foot)
    total_channels = sum(len(foot) for foot in channel_keep)
    return "fully_healthy" if live_channels == total_channels else "channel_degraded"


def summarize_fastbase_capture_diagnostics(
    *,
    capture_probability: Sequence[float],
    effective_gate: Sequence[float],
    delta_l2: Sequence[float],
    course_stage: Sequence[int],
    rollout_step: Sequence[int],
    env_id: Sequence[int],
    step_dt_s: float,
    raw_capture_probability: Sequence[float] | None = None,
    activation_threshold: float = 0.5,
    release_threshold: float = 0.1,
    release_stable_steps: int = 3,
) -> dict[str, object]:
    """Summarize deployable FastBase capture signals by time and H-L-H region.

    All three signals must already have been computed from the 1864-D policy
    observation alone.  ``course_stage``, ``rollout_step`` and ``env_id`` are
    privileged *evaluation labels*: this helper only groups saved samples and
    cannot feed them back into an actor.

    ``delta_l2`` is the Euclidean norm of the 29-D residual action.  The LOW
    activation latency starts at an environment's first LOW sample.  The
    HIGH_END release latency requires ``release_stable_steps`` consecutive
    samples at or below ``release_threshold`` so a one-frame gate dip is not
    counted as recovery.
    """

    # Legacy datasets predate an explicit raw probability.  Treat their only
    # probability column as both raw and calibrated without changing results.
    raw_values = (
        list(capture_probability)
        if raw_capture_probability is None
        else list(raw_capture_probability)
    )
    columns = {
        "raw_capture_probability": raw_values,
        "capture_probability": list(capture_probability),
        "effective_gate": list(effective_gate),
        "delta_l2": list(delta_l2),
        "course_stage": list(course_stage),
        "rollout_step": list(rollout_step),
        "env_id": list(env_id),
    }
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"FastBase capture diagnostic columns are not aligned: {lengths}")
    if step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be positive")
    if not 0.0 <= activation_threshold <= 1.0:
        raise ValueError("activation_threshold must be in [0, 1]")
    if not 0.0 <= release_threshold <= 1.0:
        raise ValueError("release_threshold must be in [0, 1]")
    if release_stable_steps <= 0:
        raise ValueError("release_stable_steps must be positive")

    count = next(iter(lengths.values()), 0)
    stages = [int(value) for value in columns["course_stage"]]
    invalid_stages = sorted(
        {value for value in stages if value not in range(len(CAPTURE_DIAGNOSTIC_STAGE_NAMES))}
    )
    if invalid_stages:
        raise ValueError(f"invalid FastBase capture course stage(s): {invalid_stages}")
    steps = [int(value) for value in columns["rollout_step"]]
    envs = [int(value) for value in columns["env_id"]]
    if any(value < 0 for value in steps) or any(value < 0 for value in envs):
        raise ValueError("rollout_step and env_id must be non-negative")

    raw_probability = [
        float(value) for value in columns["raw_capture_probability"]
    ]
    probability = [float(value) for value in columns["capture_probability"]]
    gate = [float(value) for value in columns["effective_gate"]]
    delta = [float(value) for value in columns["delta_l2"]]
    if any(
        not math.isfinite(value)
        for value in raw_probability + probability + gate + delta
    ):
        raise ValueError("FastBase capture diagnostics must be finite")
    if any(
        value < 0.0 or value > 1.0
        for value in raw_probability + probability + gate
    ):
        raise ValueError(
            "raw_capture_probability/capture_probability/effective_gate must be in [0, 1]"
        )
    if any(value < 0.0 for value in delta):
        raise ValueError("delta_l2 must be non-negative")

    def capture_value_summary(values: Sequence[float]) -> dict[str, float | int | None]:
        result = _finite_summary(values)
        ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
        if not ordered:
            result["p95"] = None
            return result
        position = 0.95 * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        result["p95"] = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        return result

    def low_vs_high_auc(values: Sequence[float]) -> float | None:
        """Rank AUC with LOW=positive and both HIGH regions=negative."""

        labelled = sorted(
            (float(value), stage == 1) for value, stage in zip(values, stages)
        )
        positives = sum(label for _, label in labelled)
        negatives = len(labelled) - positives
        if positives == 0 or negatives == 0:
            return None
        positive_rank_sum = 0.0
        start = 0
        while start < len(labelled):
            end = start + 1
            while end < len(labelled) and labelled[end][0] == labelled[start][0]:
                end += 1
            # Ranks are one-based; ties receive their average rank.
            average_rank = 0.5 * ((start + 1) + end)
            positive_rank_sum += average_rank * sum(
                label for _, label in labelled[start:end]
            )
            start = end
        return (
            positive_rank_sum - positives * (positives + 1) / 2.0
        ) / (positives * negatives)

    def indexed_summary(indices: Sequence[int]) -> dict[str, object]:
        selected_steps = [steps[index] for index in indices]
        return {
            "samples": len(indices),
            "time_s": _finite_summary(
                [selected_step * step_dt_s for selected_step in selected_steps]
            ),
            "raw_capture_probability": capture_value_summary(
                [raw_probability[index] for index in indices]
            ),
            "capture_probability": capture_value_summary(
                [probability[index] for index in indices]
            ),
            "effective_gate": capture_value_summary([gate[index] for index in indices]),
            "capture_delta_l2": capture_value_summary(
                [delta[index] for index in indices]
            ),
            "effective_gate_ge_activation_fraction": (
                sum(gate[index] >= activation_threshold for index in indices)
                / max(len(indices), 1)
            ),
        }

    all_indices = list(range(count))
    by_stage = {
        name: indexed_summary(
            [index for index, value in enumerate(stages) if value == stage_id]
        )
        for stage_id, name in enumerate(CAPTURE_DIAGNOSTIC_STAGE_NAMES)
    }

    rows_by_env: dict[int, list[int]] = {}
    for index, value in enumerate(envs):
        rows_by_env.setdefault(value, []).append(index)
    low_activation_latency: list[float] = []
    high_end_release_latency: list[float] = []
    per_env: list[dict[str, object]] = []
    low_entered_envs = 0
    high_end_entered_envs = 0
    for current_env, indices in sorted(rows_by_env.items()):
        ordered = sorted(indices, key=lambda index: steps[index])

        def first_stage_index(stage_id: int) -> int | None:
            return next((index for index in ordered if stages[index] == stage_id), None)

        low_entry = first_stage_index(1)
        low_activation_step: int | None = None
        if low_entry is not None:
            low_entered_envs += 1
            entry_step = steps[low_entry]
            low_activation_step = next(
                (
                    steps[index]
                    for index in ordered
                    if steps[index] >= entry_step
                    and stages[index] == 1
                    and gate[index] >= activation_threshold
                ),
                None,
            )
            if low_activation_step is not None:
                low_activation_latency.append(
                    (low_activation_step - entry_step) * step_dt_s
                )

        high_entry = first_stage_index(2)
        high_release_step: int | None = None
        if high_entry is not None:
            high_end_entered_envs += 1
            stable = 0
            for index in ordered:
                if steps[index] < steps[high_entry]:
                    continue
                if stages[index] != 2:
                    stable = 0
                    continue
                stable = stable + 1 if gate[index] <= release_threshold else 0
                if stable >= release_stable_steps:
                    high_release_step = steps[index] - release_stable_steps + 1
                    high_end_release_latency.append(
                        (high_release_step - steps[high_entry]) * step_dt_s
                    )
                    break

        per_env.append(
            {
                "env_id": current_env,
                "low_entry_step": None if low_entry is None else steps[low_entry],
                "low_activation_step": low_activation_step,
                "high_end_entry_step": None if high_entry is None else steps[high_entry],
                "high_end_release_step": high_release_step,
            }
        )

    return {
        "definition": "fastbase-capture-observation-only-diagnostics-v2",
        "available": True,
        "sample_count": count,
        "step_dt_s": float(step_dt_s),
        "stage_encoding": {
            name: index for index, name in enumerate(CAPTURE_DIAGNOSTIC_STAGE_NAMES)
        },
        "overall": indexed_summary(all_indices),
        "by_stage": by_stage,
        "low_vs_high_auc": {
            "positive_stage": "LOW",
            "negative_stages": ["HIGH_START", "HIGH_END"],
            "raw_capture_probability": low_vs_high_auc(raw_probability),
            "capture_probability": low_vs_high_auc(probability),
            "effective_gate": low_vs_high_auc(gate),
            "capture_delta_l2": low_vs_high_auc(delta),
        },
        "low_activation": {
            "effective_gate_threshold": float(activation_threshold),
            "entered_envs": low_entered_envs,
            "activated_envs": len(low_activation_latency),
            "activation_fraction": len(low_activation_latency) / max(low_entered_envs, 1),
            "latency_s": _finite_summary(low_activation_latency),
        },
        "high_end_release": {
            "effective_gate_threshold": float(release_threshold),
            "stable_steps": int(release_stable_steps),
            "entered_envs": high_end_entered_envs,
            "released_envs": len(high_end_release_latency),
            "release_fraction": len(high_end_release_latency)
            / max(high_end_entered_envs, 1),
            "latency_s": _finite_summary(high_end_release_latency),
        },
        "per_env_transition_steps": per_env,
    }


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def analyze_transition_response(
    *,
    body_vx: Sequence[Sequence[float]],
    low_contact: Sequence[Sequence[bool]],
    high_end_contact: Sequence[Sequence[bool]],
    falls: Sequence[Sequence[bool]],
    dones: Sequence[Sequence[bool]],
    step_dt_s: float,
    low_speed_target_m_s: float,
    high_recovery_speed_m_s: float,
    horizons_s: Sequence[float] = (0.5, 1.0),
    recovery_stable_steps: int = 3,
) -> dict[str, object]:
    """Measure causal speed response for the first episode of every env.

    A low-entry reference is the first physically labelled low-contact frame.
    Deceleration at a horizon is ``entry_vx - horizon_vx``.  Missing samples
    caused by a termination remain missing and reduce ``survival_fraction``;
    they are never silently replaced by post-reset motion.

    High-patch recovery starts on the first physical high-end contact after a
    low entry.  Two thresholds are reported: an absolute deployability target
    and 90% of that rollout's pre-low speed.  The threshold must be held for
    ``recovery_stable_steps`` consecutive policy frames.
    """

    series = (body_vx, low_contact, high_end_contact, falls, dones)
    if not body_vx:
        raise ValueError("transition response needs at least one step")
    num_steps = len(body_vx)
    if any(len(item) != num_steps for item in series):
        raise ValueError("all transition time series must have equal step counts")
    num_envs = len(body_vx[0])
    if num_envs <= 0 or any(len(row) != num_envs for item in series for row in item):
        raise ValueError("transition time series must be rectangular [step, env]")
    if step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be positive")
    if low_speed_target_m_s < 0.0 or high_recovery_speed_m_s <= 0.0:
        raise ValueError("speed targets must be non-negative and recovery positive")
    if recovery_stable_steps <= 0:
        raise ValueError("recovery_stable_steps must be positive")
    clean_horizons = tuple(float(value) for value in horizons_s)
    if not clean_horizons or any(value <= 0.0 for value in clean_horizons):
        raise ValueError("deceleration horizons must be positive")
    horizon_steps = {
        horizon: max(1, int(round(horizon / step_dt_s))) for horizon in clean_horizons
    }

    per_env: list[dict[str, object]] = []
    first_fall_steps: list[int] = []
    for env_id in range(num_envs):
        low_entry_step: int | None = None
        low_entry_vx: float | None = None
        high_entry_step: int | None = None
        horizon_values: dict[float, float | None] = {
            horizon: None for horizon in clean_horizons
        }
        absolute_stable = 0
        relative_stable = 0
        absolute_recovery_step: int | None = None
        relative_recovery_step: int | None = None
        first_fall_step: int | None = None
        terminal_step: int | None = None

        for step in range(num_steps):
            fell = bool(falls[step][env_id])
            done = bool(dones[step][env_id])
            if fell and first_fall_step is None:
                first_fall_step = step
                first_fall_steps.append(step)

            # A done frame already contains post-reset state in Isaac Lab.
            # Use it only as a terminal marker, never as a response sample.
            if not done:
                velocity = float(body_vx[step][env_id])
                if low_entry_step is None and bool(low_contact[step][env_id]):
                    low_entry_step = step
                    low_entry_vx = velocity
                if low_entry_step is not None:
                    for horizon, offset in horizon_steps.items():
                        if horizon_values[horizon] is None and step >= low_entry_step + offset:
                            horizon_values[horizon] = velocity

                    if high_entry_step is None and bool(high_end_contact[step][env_id]):
                        high_entry_step = step
                    if high_entry_step is not None:
                        absolute_stable = (
                            absolute_stable + 1
                            if velocity >= high_recovery_speed_m_s
                            else 0
                        )
                        relative_target = 0.90 * float(low_entry_vx)
                        relative_stable = relative_stable + 1 if velocity >= relative_target else 0
                        if (
                            absolute_recovery_step is None
                            and absolute_stable >= recovery_stable_steps
                        ):
                            absolute_recovery_step = step
                        if (
                            relative_recovery_step is None
                            and relative_stable >= recovery_stable_steps
                        ):
                            relative_recovery_step = step

            if done:
                terminal_step = step
                break

        horizon_report: dict[str, object] = {}
        for horizon in clean_horizons:
            value = horizon_values[horizon]
            horizon_report[f"{horizon:g}s"] = {
                "vx_m_s": value,
                "deceleration_m_s": (
                    None
                    if value is None or low_entry_vx is None
                    else float(low_entry_vx) - float(value)
                ),
                "at_or_below_target": (
                    None if value is None else bool(value <= low_speed_target_m_s)
                ),
            }

        def recovery_time(recovery_step: int | None) -> float | None:
            if recovery_step is None or high_entry_step is None:
                return None
            # Attribute the first held sample to its first stable frame.
            first_stable = recovery_step - recovery_stable_steps + 1
            return max(0.0, (first_stable - high_entry_step) * step_dt_s)

        per_env.append(
            {
                "env_id": env_id,
                "first_fall_step": first_fall_step,
                "first_fall_time_s": (
                    None if first_fall_step is None else (first_fall_step + 1) * step_dt_s
                ),
                "terminal_step": terminal_step,
                "low_entry_step": low_entry_step,
                "low_entry_time_s": (
                    None if low_entry_step is None else (low_entry_step + 1) * step_dt_s
                ),
                "low_entry_vx_m_s": low_entry_vx,
                "deceleration": horizon_report,
                "high_end_entry_step": high_entry_step,
                "absolute_recovery_time_s": recovery_time(absolute_recovery_step),
                "relative_90pct_recovery_time_s": recovery_time(relative_recovery_step),
            }
        )

    entries = [row for row in per_env if row["low_entry_step"] is not None]
    deceleration: dict[str, object] = {}
    for horizon in clean_horizons:
        key = f"{horizon:g}s"
        samples = [
            row["deceleration"][key]
            for row in entries
            if row["deceleration"][key]["vx_m_s"] is not None
        ]
        deceleration[key] = {
            "sampled_envs": len(samples),
            "survival_fraction": len(samples) / max(len(entries), 1),
            "vx_m_s": _finite_summary([sample["vx_m_s"] for sample in samples]),
            "deceleration_m_s": _finite_summary(
                [sample["deceleration_m_s"] for sample in samples]
            ),
            "at_or_below_target_fraction": (
                sum(bool(sample["at_or_below_target"]) for sample in samples)
                / max(len(samples), 1)
            ),
        }

    high_entries = [row for row in per_env if row["high_end_entry_step"] is not None]

    def recovery_summary(field: str) -> dict[str, object]:
        values = [row[field] for row in high_entries if row[field] is not None]
        return {
            "recovered_envs": len(values),
            "recovery_fraction": len(values) / max(len(high_entries), 1),
            "time_s": _finite_summary(values),
        }

    first_fall_step = min(first_fall_steps) if first_fall_steps else None
    return {
        "definition": "first-episode-causal-response-v1",
        "step_dt_s": float(step_dt_s),
        "first_fall_step": first_fall_step,
        "first_fall_time_s": (
            None if first_fall_step is None else (first_fall_step + 1) * step_dt_s
        ),
        "low_speed_target_m_s": float(low_speed_target_m_s),
        "low_entry_envs": len(entries),
        "low_entry_speed_m_s": _finite_summary(
            [row["low_entry_vx_m_s"] for row in entries]
        ),
        "deceleration_after_low_contact": deceleration,
        "high_end_entry_envs": len(high_entries),
        "high_recovery_speed_m_s": float(high_recovery_speed_m_s),
        "recovery_stable_steps": int(recovery_stable_steps),
        "absolute_high_recovery": recovery_summary("absolute_recovery_time_s"),
        "relative_90pct_pre_low_recovery": recovery_summary(
            "relative_90pct_recovery_time_s"
        ),
        "per_env": per_env,
    }


def validate_motion_hall_risk_metadata(payload: object) -> dict[str, object]:
    """Fail closed on the deployable Motion Hall-risk checkpoint contract.

    The Motion actor's last two columns are ``[body_vy, relative_heading]``.
    Older risk artifacts called those columns sensor age, which is a silent
    semantic error even though their tensor width is still 1864.  Keep this
    validator free of Torch/Isaac imports so the evaluator contract can be
    exercised in a small CPU test.
    """

    if not isinstance(payload, dict):
        raise ValueError("Hall risk checkpoint payload must be a dictionary")
    if payload.get("input_dim") != 1864:
        raise ValueError(
            "Hall risk checkpoint must consume exactly 1864 actor columns"
        )
    if payload.get("trailing_feature_mode") != "motion_feedback":
        raise ValueError(
            "Hall risk checkpoint trailing_feature_mode must be "
            "'motion_feedback'; sensor_age artifacts are incompatible with "
            "Motion columns 1862:1864=[body_vy,relative_heading]"
        )
    boundary = payload.get("measurement_boundary")
    if not isinstance(boundary, str) or "Hall Bx/By/Bz" not in boundary:
        raise ValueError(
            "Hall risk checkpoint must explicitly declare its Hall-only "
            "measurement boundary"
        )
    boundary_lower = boundary.lower()
    explicit_inverse_exclusion = (
        "no hall-to-force" in boundary_lower
        and "no hall-to-friction" in boundary_lower
    )
    explicit_runtime_only = (
        "proprioception only" in boundary_lower
        and "offline simulator labels, not inputs" in boundary_lower
    )
    if not (explicit_inverse_exclusion or explicit_runtime_only):
        raise ValueError(
            "Hall risk checkpoint boundary must explicitly exclude Hall-to-force "
            "and Hall-to-friction inversion"
        )
    if payload.get("risk_target") != "prospective contact-point slip/fall":
        raise ValueError(
            "Hall command governor requires the prospective slip/fall risk target"
        )
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("Hall risk checkpoint must contain a non-empty model state")
    return {
        "input_dim": 1864,
        "trailing_feature_mode": "motion_feedback",
        "model_variant": str(payload.get("model_variant", "")),
        "risk_target": str(payload["risk_target"]),
        "measurement_boundary": boundary,
    }


def summarize_hall_command_governor_trace(
    *,
    risk_probability: Sequence[float],
    filtered_probability: Sequence[float],
    state: Sequence[int],
    requested_vx: Sequence[float],
    upstream_vx: Sequence[float],
    effective_vx: Sequence[float],
    valid: Sequence[bool],
    probing: Sequence[bool],
    prebrake: Sequence[bool],
    rollout_step: Sequence[int],
    env_id: Sequence[int],
    step_dt_s: float,
    low_speed_limit_m_s: float,
    high_speed_limit_m_s: float,
    recovery_fraction: float = 0.90,
) -> dict[str, object]:
    """Summarize a Hall-only governor trace without course/material labels.

    Latencies are referenced only to the governor's own HIGH->LOW->HIGH state
    transitions.  Physical patch/contact response remains a separate evaluator
    diagnostic and cannot influence this state machine or command output.
    """

    columns = (
        risk_probability,
        filtered_probability,
        state,
        requested_vx,
        upstream_vx,
        effective_vx,
        valid,
        probing,
        prebrake,
        rollout_step,
        env_id,
    )
    count = len(risk_probability)
    if any(len(column) != count for column in columns):
        raise ValueError("Hall governor diagnostic columns are not aligned")
    if step_dt_s <= 0.0 or not math.isfinite(float(step_dt_s)):
        raise ValueError("step_dt_s must be finite and positive")
    if low_speed_limit_m_s < 0.0 or high_speed_limit_m_s < low_speed_limit_m_s:
        raise ValueError("Hall governor speed limits are invalid")
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must be in (0,1]")

    raw_risk = [float(value) for value in risk_probability]
    ema_risk = [float(value) for value in filtered_probability]
    requested = [float(value) for value in requested_vx]
    upstream = [float(value) for value in upstream_vx]
    effective = [float(value) for value in effective_vx]
    states = [int(value) for value in state]
    steps = [int(value) for value in rollout_step]
    envs = [int(value) for value in env_id]
    if any(
        not math.isfinite(value)
        for values in (raw_risk, ema_risk, requested, upstream, effective)
        for value in values
    ):
        raise ValueError("Hall governor diagnostics must be finite")
    if any(value < 0.0 or value > 1.0 for value in raw_risk + ema_risk):
        raise ValueError("Hall risk probabilities must be in [0,1]")
    if any(value not in (0, 1, 2) for value in states):
        raise ValueError("Hall governor state must be UNKNOWN=0, LOW=1 or HIGH=2")
    if any(value < 0 for value in steps + envs):
        raise ValueError("rollout_step and env_id must be non-negative")

    rows_by_env: dict[int, list[int]] = {}
    for index, current_env in enumerate(envs):
        rows_by_env.setdefault(current_env, []).append(index)

    low_command_latencies: list[float] = []
    state_recovery_latencies: list[float] = []
    command_recovery_latencies: list[float] = []
    per_env: list[dict[str, object]] = []
    completed_hlh = 0
    for current_env, indices in sorted(rows_by_env.items()):
        ordered = sorted(indices, key=lambda index: steps[index])
        ordered_states = [states[index] for index in ordered]
        compressed: list[int] = []
        for value in ordered_states:
            if not compressed or value != compressed[-1]:
                compressed.append(value)

        first_high_position = next(
            (position for position, value in enumerate(ordered_states) if value == 2),
            None,
        )
        low_position = None
        if first_high_position is not None:
            low_position = next(
                (
                    position
                    for position in range(first_high_position + 1, len(ordered))
                    if ordered_states[position] == 1
                ),
                None,
            )
        recovered_high_position = None
        if low_position is not None:
            recovered_high_position = next(
                (
                    position
                    for position in range(low_position + 1, len(ordered))
                    if ordered_states[position] == 2
                ),
                None,
            )
        if recovered_high_position is not None:
            completed_hlh += 1

        low_command_position = None
        if low_position is not None:
            low_command_position = next(
                (
                    position
                    for position in range(low_position, len(ordered))
                    if abs(effective[ordered[position]])
                    <= low_speed_limit_m_s + 1.0e-6
                ),
                None,
            )
            if low_command_position is not None:
                low_command_latencies.append(
                    (steps[ordered[low_command_position]] - steps[ordered[low_position]])
                    * step_dt_s
                )

        high_command_position = None
        recovery_target = None
        if recovered_high_position is not None:
            recovery_row = ordered[recovered_high_position]
            recovery_target = recovery_fraction * min(
                abs(requested[recovery_row]), high_speed_limit_m_s
            )
            high_command_position = next(
                (
                    position
                    for position in range(recovered_high_position, len(ordered))
                    if abs(effective[ordered[position]]) + 1.0e-6 >= recovery_target
                ),
                None,
            )
            state_recovery_latencies.append(
                (steps[recovery_row] - steps[ordered[low_position]]) * step_dt_s
            )
            if high_command_position is not None:
                command_recovery_latencies.append(
                    (
                        steps[ordered[high_command_position]]
                        - steps[recovery_row]
                    )
                    * step_dt_s
                )

        def step_at(position: int | None) -> int | None:
            return None if position is None else steps[ordered[position]]

        per_env.append(
            {
                "env_id": current_env,
                "compressed_state_sequence": compressed,
                "first_high_step": step_at(first_high_position),
                "low_entry_step": step_at(low_position),
                "low_command_limit_step": step_at(low_command_position),
                "recovered_high_step": step_at(recovered_high_position),
                "high_command_recovery_step": step_at(high_command_position),
                "high_command_recovery_target_m_s": recovery_target,
            }
        )

    return {
        "definition": "hall-only-command-governor-response-v1",
        "input_contract": (
            "raw deployable 1864-D Hall/proprioception observation and Hall "
            "packet health only; no friction/contact/force/course-stage truth"
        ),
        "sample_count": count,
        "step_dt_s": float(step_dt_s),
        "state_encoding": {"UNKNOWN": 0, "LOW": 1, "HIGH": 2},
        "risk_probability": _finite_summary(raw_risk),
        "filtered_probability": _finite_summary(ema_risk),
        "valid_fraction": sum(bool(value) for value in valid) / max(count, 1),
        "probing_fraction": sum(bool(value) for value in probing) / max(count, 1),
        "prebrake_fraction": sum(bool(value) for value in prebrake) / max(count, 1),
        "state_fraction": {
            "UNKNOWN": sum(value == 0 for value in states) / max(count, 1),
            "LOW": sum(value == 1 for value in states) / max(count, 1),
            "HIGH": sum(value == 2 for value in states) / max(count, 1),
        },
        "completed_internal_hlh_envs": completed_hlh,
        "completed_internal_hlh_fraction": completed_hlh / max(len(rows_by_env), 1),
        "low_state_to_command_limit_s": _finite_summary(low_command_latencies),
        "low_state_to_recovered_high_state_s": _finite_summary(
            state_recovery_latencies
        ),
        "recovered_high_state_to_command_s": _finite_summary(
            command_recovery_latencies
        ),
        "per_env": per_env,
    }
