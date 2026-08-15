"""Metric computation for canonical traction NPZ trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_TRAJECTORY_FIELDS = (
    "timestamp_s",
    "environment_id",
    "command",
    "base_velocity",
    "rollout_action",
    "episode_done",
    "slip_speed_proxy",
    "slip_label",
)


def _mean(value: np.ndarray) -> float:
    return float(np.mean(value)) if value.size else float("nan")


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    label = label.astype(bool).reshape(-1)
    score = score.reshape(-1)
    positive = int(label.sum())
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_rank = np.arange(1, len(score) + 1, dtype=np.float64)
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_score)), len(score)]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        sorted_rank[start:stop] = sorted_rank[start:stop].mean()
    ranks = np.empty_like(sorted_rank)
    ranks[order] = sorted_rank
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _binary_metrics(label: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    label_bool = label.reshape(-1) > 0.5
    predicted = probability.reshape(-1) >= 0.5
    tp = int(np.count_nonzero(label_bool & predicted))
    fp = int(np.count_nonzero(~label_bool & predicted))
    fn = int(np.count_nonzero(label_bool & ~predicted))
    precision = (
        tp / (tp + fp)
        if tp + fp
        else (0.0 if tp + fn else float("nan"))
    )
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0.0
        else (0.0 if np.isfinite(precision + recall) else float("nan"))
    )
    return {
        "slip_precision": precision,
        "slip_recall": recall,
        "slip_f1": f1,
        "slip_auc": _auc(label_bool, probability),
    }


def _per_environment_sequences(data: dict[str, np.ndarray]):
    environment_ids = data["environment_id"].reshape(-1).astype(np.int64)
    timestamps = data["timestamp_s"].reshape(-1)
    for environment_id in np.unique(environment_ids):
        indices = np.flatnonzero(environment_ids == environment_id)
        yield environment_id, indices[np.argsort(timestamps[indices])]


def _slip_event_count(data: dict[str, np.ndarray]) -> int:
    count = 0
    labels = data["slip_label"] > 0.5
    for _, indices in _per_environment_sequences(data):
        sequence = labels[indices]
        count += int(sequence[0].sum())
        count += int(np.count_nonzero(sequence[1:] & ~sequence[:-1]))
    return count


def _action_smoothness(data: dict[str, np.ndarray]) -> float:
    differences = []
    actions = data["rollout_action"]
    done = data["episode_done"].reshape(-1).astype(bool)
    for _, indices in _per_environment_sequences(data):
        if len(indices) < 2:
            continue
        valid = ~done[indices[:-1]]
        differences.append(
            np.linalg.norm(actions[indices[1:]] - actions[indices[:-1]], axis=1)[valid]
        )
    return _mean(np.concatenate(differences)) if differences else float("nan")


def _detection_delay(data: dict[str, np.ndarray], threshold: float = 0.5) -> float:
    if "predicted_slip_probability" not in data:
        return float("nan")
    label = data["slip_label"] > 0.5
    predicted = data["predicted_slip_probability"] >= threshold
    timestamps = data["timestamp_s"].reshape(-1)
    delays = []
    for _, indices in _per_environment_sequences(data):
        for foot in range(2):
            sequence = label[indices, foot]
            starts = np.flatnonzero(sequence & np.r_[True, ~sequence[:-1]])
            for start in starts:
                false_after_start = np.flatnonzero(~sequence[start:])
                stop = (
                    start + int(false_after_start[0])
                    if false_after_start.size
                    else len(sequence)
                )
                detections = np.flatnonzero(
                    predicted[indices[start:stop], foot]
                )
                if detections.size:
                    delays.append(
                        timestamps[indices[start + detections[0]]]
                        - timestamps[indices[start]]
                    )
    return _mean(np.asarray(delays))


def bootstrap_mean_ci(
    value: np.ndarray,
    *,
    seed: int = 20260731,
    samples: int = 2000,
) -> tuple[float, float]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    resampled = generator.choice(value, size=(samples, value.size), replace=True)
    means = resampled.mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _episode_statistics(
    data: dict[str, np.ndarray],
    dt: float,
) -> tuple[float, float, int]:
    terminated = data.get("terminated", np.zeros_like(data["episode_done"]))
    truncated = data.get("truncated", np.zeros_like(data["episode_done"]))
    completed_lengths = []
    completed = 0
    falls = 0
    for _, indices in _per_environment_sequences(data):
        segment_length = 0
        for index in indices:
            segment_length += 1
            if bool(terminated[index].item() or truncated[index].item()):
                completed += 1
                falls += int(bool(terminated[index].item()))
                completed_lengths.append(segment_length * dt)
                segment_length = 0
    fall_rate = falls / completed if completed else float("nan")
    return _mean(np.asarray(completed_lengths)), fall_rate, completed


def evaluate_npz(path: str | Path) -> tuple[dict[str, float | str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata"
        }
    missing = sorted(set(REQUIRED_TRAJECTORY_FIELDS) - set(data))
    if missing:
        raise ValueError(
            f"{Path(path)} is not a canonical trajectory; missing fields: {missing}"
        )
    command = data["command"]
    base_velocity = data["base_velocity"]
    velocity_error = np.linalg.norm(command[:, :2] - base_velocity[:, :2], axis=1)
    command_speed = np.linalg.norm(command[:, :2], axis=1)
    actual_speed = np.linalg.norm(base_velocity[:, :2], axis=1)
    moving = command_speed > 0.05
    speed_ratio = actual_speed[moving] / command_speed[moving]
    dt = 0.02
    if len(data["timestamp_s"]) > len(np.unique(data["environment_id"])):
        first_environment = next(_per_environment_sequences(data))[1]
        if len(first_environment) > 1:
            dt = float(
                np.median(
                    np.diff(data["timestamp_s"][first_environment].reshape(-1))
                )
            )
    environment_count = len(np.unique(data["environment_id"]))
    episode_length, fall_rate, completed_episodes = _episode_statistics(data, dt)
    slip_speed = data["slip_speed_proxy"]
    slip_label = data["slip_label"] > 0.5
    orientation_error = (
        np.linalg.norm(data["projected_gravity"][:, :2], axis=1)
        if "projected_gravity" in data
        else np.asarray([], dtype=np.float32)
    )
    velocity_ci = bootstrap_mean_ci(velocity_error)
    metrics: dict[str, float | str] = {
        "dataset": str(Path(path).resolve()),
        "samples": float(len(command)),
        "environments": float(environment_count),
        "velocity_tracking_error_mean_m_s": _mean(velocity_error),
        "velocity_tracking_error_ci95_low_m_s": velocity_ci[0],
        "velocity_tracking_error_ci95_high_m_s": velocity_ci[1],
        "yaw_tracking_error_mean_rad_s": _mean(
            np.abs(command[:, 2:3] - data["base_yaw_rate"])
        )
        if "base_yaw_rate" in data
        else float("nan"),
        "actual_command_speed_ratio": _mean(speed_ratio),
        "completed_episodes": float(completed_episodes),
        "fall_rate": fall_rate,
        "episode_length_mean_s": episode_length,
        "rollout_duration_mean_per_environment_s": float(
            len(command) * dt / max(environment_count, 1)
        ),
        "orientation_error_mean": _mean(orientation_error),
        "action_smoothness": _action_smoothness(data),
        "slip_event_count": float(_slip_event_count(data)),
        "slip_duration_s": float(slip_label.sum() * dt),
        "slip_distance_proxy_m": float((slip_speed * slip_label).sum() * dt),
        "maximum_slip_speed_proxy_m_s": float(slip_speed.max(initial=0.0)),
        "left_slip_rate": _mean(slip_label[:, 0]),
        "right_slip_rate": _mean(slip_label[:, 1]),
        "slip_detection_delay_s": _detection_delay(data),
    }
    if "joint_torque" in data and "joint_velocity" in data:
        absolute_power = np.abs(data["joint_torque"] * data["joint_velocity"]).sum(
            axis=1
        )
        metrics["mean_absolute_mechanical_power_w"] = _mean(absolute_power)
        metrics["mechanical_energy_per_environment_j"] = float(
            absolute_power.sum() * dt / max(environment_count, 1)
        )
    else:
        metrics["mean_absolute_mechanical_power_w"] = float("nan")
        metrics["mechanical_energy_per_environment_j"] = float("nan")
    if "predicted_slip_probability" in data:
        metrics.update(
            _binary_metrics(
                data["slip_label"],
                data["predicted_slip_probability"],
            )
        )
    else:
        metrics.update(
            {
                "slip_precision": float("nan"),
                "slip_recall": float("nan"),
                "slip_f1": float("nan"),
                "slip_auc": float("nan"),
            }
        )
    if "predicted_traction_score" in data:
        target = data["traction_target"].reshape(-1)
        predicted = data["predicted_traction_score"].reshape(-1)
        metrics["traction_score_correlation"] = (
            float(np.corrcoef(target, predicted)[0, 1])
            if np.std(target) > 0.0 and np.std(predicted) > 0.0
            else float("nan")
        )
    else:
        metrics["traction_score_correlation"] = float("nan")
    metrics["latent_mse"] = (
        float(np.mean((data["student_latent"] - data["teacher_latent"]) ** 2))
        if "student_latent" in data
        else float("nan")
    )
    metrics["sensor_confidence_brier"] = (
        float(
            np.mean(
                (
                    data["predicted_sensor_confidence"].reshape(-1)
                    - data["sensor_valid"].min(axis=1)
                )
                ** 2
            )
        )
        if "predicted_sensor_confidence" in data
        else float("nan")
    )
    return metrics, data
