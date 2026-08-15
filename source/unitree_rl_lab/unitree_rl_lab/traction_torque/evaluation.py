"""Metrics and registered experiments for torque-based traction policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ISAAC_MUJOCO_SCENARIOS = (
    "high_friction", "low_friction", "very_low_friction", "abrupt_friction_drop",
    "friction_recovery", "left_right_asymmetric_friction", "forward_acceleration",
    "sudden_stop", "turning", "lateral_walking", "single_support", "double_support",
    "motor_torque_noise", "motor_torque_delay", "motor_state_dropout", "imu_bias",
    "mass_inertia_mismatch", "combined_randomization",
)

TORQUE_TRACTION_ABLATIONS = {
    "A0": "original proprio baseline",
    "A1": "proprio + raw leg tau_est",
    "A2": "proprio + analytical estimated force",
    "A3": "proprio + analytical force + 15-frame history",
    "A4": "analytical + temporal correction",
    "A5": "full Student without governor",
    "A6": "full Student with governor",
    "A7": "privileged Teacher upper bound",
    "A8": "no IMU linear acceleration",
    "A9": "no tau_est history",
    "A10": "no estimated force, only raw torque history",
    "A11": "1/5/10/15/25-frame history",
    "A12": "GRU vs TCN",
    "A13": "ideal dynamics vs randomized dynamics",
    "A14": "no contact classifier",
    "A15": "no estimator-confidence fallback",
}


def _binary_metrics(score: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    prediction = score >= threshold
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    positive, negative = int(target.sum()), int((~target).sum())
    auc = float("nan")
    if positive and negative:
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        ranks = np.empty(score.size, dtype=np.float64)
        start = 0
        while start < score.size:
            end = start + 1
            while end < score.size and sorted_score[end] == sorted_score[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        auc = float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc, "tp": tp, "fp": fp, "fn": fn}


def _force_direction_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate, truth = estimate.reshape(-1, 3), truth.reshape(-1, 3)
    valid = (np.linalg.norm(estimate, axis=1) > 1.0) & (np.linalg.norm(truth, axis=1) > 1.0)
    if not valid.any():
        return float("nan")
    cosine = np.sum(estimate[valid] * truth[valid], axis=1) / (
        np.linalg.norm(estimate[valid], axis=1) * np.linalg.norm(truth[valid], axis=1)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())


def _detection_delay_s(score: np.ndarray, target: np.ndarray, dt_s: float) -> float:
    score, target = np.asarray(score), np.asarray(target, dtype=bool)
    delays: list[float] = []
    for foot in range(target.shape[1]):
        onset = np.flatnonzero(target[:, foot] & ~np.r_[False, target[:-1, foot]])
        for index in onset:
            detected = np.flatnonzero(score[index : index + max(1, round(1.0 / dt_s)), foot] >= 0.5)
            if detected.size:
                delays.append(float(detected[0] * dt_s))
    return float(np.mean(delays)) if delays else float("nan")


def evaluate_rollout_npz(path: str | Path, *, policy_dt_s: float = 0.02) -> dict[str, object]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    estimate = np.asarray(data["estimated_force_local_n"], dtype=np.float64)
    truth = np.asarray(data["true_force_local_n"], dtype=np.float64)
    error = estimate - truth
    true_contact = np.asarray(data["true_contact"], dtype=bool)
    contact_probability = np.asarray(data["contact_probability"], dtype=np.float64)
    slip_speed = np.asarray(data["contact_point_slip_speed_m_s"], dtype=np.float64)
    true_slip = true_contact & (slip_speed > 0.12)
    slip_probability = np.asarray(data["slip_probability"], dtype=np.float64)
    command = np.asarray(data["raw_command"], dtype=np.float64)
    velocity = np.asarray(data["base_velocity"], dtype=np.float64)
    component_mae = np.abs(error).mean(axis=0)
    component_rmse = np.sqrt(np.square(error).mean(axis=0))
    nonfinite = sum(int((~np.isfinite(np.asarray(data[name]))).sum()) for name in data.files if np.asarray(data[name]).dtype.kind in "fci")
    steps = int(estimate.shape[0])
    metadata = data["metadata"].item() if "metadata" in data.files else {}
    fell = bool(metadata.get("fell", np.min(data["base_height_m"]) <= 0.31))
    return {
        "source": str(path.resolve()),
        "samples": steps,
        "survival_time_s": steps * policy_dt_s,
        "fell_by_height_threshold": fell,
        "minimum_base_height_m": float(np.min(data["base_height_m"])),
        "velocity_tracking_mae_m_s": float(np.abs(velocity[:, :2] - command[:, :2]).mean()),
        "yaw_tracking_mae_rad_s": float(np.abs(velocity[:, 2] - command[:, 2]).mean()),
        "force_component_mae_n": component_mae.tolist(),
        "force_component_rmse_n": component_rmse.tolist(),
        "force_mae_n": float(np.abs(error).mean()),
        "force_rmse_n": float(np.sqrt(np.square(error).mean())),
        "force_direction_error_deg": _force_direction_error_deg(estimate, truth),
        "contact": _binary_metrics(contact_probability, true_contact),
        "slip": _binary_metrics(slip_probability, true_slip),
        "slip_detection_delay_s": _detection_delay_s(slip_probability, true_slip, policy_dt_s),
        "ground_truth_slip_rate": float(true_slip.mean()),
        "maximum_contact_point_slip_speed_m_s": float(slip_speed.max()),
        "mean_traction_utilization": float(np.mean(data["traction_utilization"])),
        "governor_activation_ratio": float(np.mean(np.asarray(data["governor_state"]) != 0)),
        "mean_speed_scale": float(np.mean(data["speed_scale"])),
        "mean_acceleration_limit": float(np.mean(data["acceleration_limit"])),
        "mean_push_off_scale": float(np.mean(data["push_off_scale"])),
        "nonfinite_count": nonfinite,
        "slip_metric_definition": "ground-truth contact AND contact-point relative tangential speed > 0.12 m/s",
    }
