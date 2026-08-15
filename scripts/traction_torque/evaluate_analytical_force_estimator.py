#!/usr/bin/env python3
"""Evaluate analytical force/contact estimates from a collected NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {"precision": precision, "recall": recall, "f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


def evaluate(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    estimate = data["estimated_force_local_n"].reshape(-1, 2, 3)
    truth = data["true_force_local_n"].reshape(-1, 2, 3)
    error = estimate - truth
    axis = ("Fx", "Fy", "Fz")
    result: dict[str, object] = {
        "dataset": str(path.resolve()),
        "sample_feet": int(estimate.shape[0] * 2),
        "mae_n": {name: float(np.mean(np.abs(error[..., index]))) for index, name in enumerate(axis)},
        "rmse_n": {name: float(np.sqrt(np.mean(np.square(error[..., index])))) for index, name in enumerate(axis)},
        "nonfinite_count": int((~np.isfinite(estimate)).sum()),
    }
    true_norm = np.linalg.norm(truth, axis=-1)
    estimate_norm = np.linalg.norm(estimate, axis=-1)
    valid_direction = (true_norm > 20.0) & (estimate_norm > 1.0)
    cosine = np.sum(estimate * truth, axis=-1) / np.maximum(estimate_norm * true_norm, 1e-8)
    result["force_direction_error_deg"] = float(np.degrees(np.arccos(np.clip(cosine[valid_direction], -1, 1))).mean()) if valid_direction.any() else float("nan")
    contact_truth = data["true_contact"].reshape(-1, 2)
    contact_pred = data["contact_probability"].reshape(-1, 2) >= 0.5
    result["contact"] = _binary_metrics(contact_pred, contact_truth)
    swing = ~contact_truth
    result["swing_false_force_mean_n"] = float(estimate_norm[swing].mean()) if swing.any() else 0.0
    result["stance_force_correlation"] = {}
    for foot, name in enumerate(("left", "right")):
        mask = contact_truth[:, foot]
        if mask.sum() > 2:
            result["stance_force_correlation"][name] = [float(np.corrcoef(estimate[mask, foot, i], truth[mask, foot, i])[0, 1]) for i in range(3)]
    latency = data["estimator_latency_ms"].reshape(-1)
    result["estimator_latency_ms"] = {"mean": float(np.nanmean(latency)), "p95": float(np.nanpercentile(latency, 95))}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
