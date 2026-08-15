#!/usr/bin/env python3
"""Evaluate whether labelled high/low surfaces are separable from Hall time series.

The analysis uses trial-wise cross-validation. It reports statistical
separability only and never renames Hall features as force or friction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def _features(segment: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    db = segment.astype(np.float64) - baseline[None]
    centered = db - np.median(db, axis=0, keepdims=True)
    diff = np.diff(db, axis=0)
    abs_axis = np.mean(np.abs(db), axis=(0, 1))
    std_axis = np.std(db, axis=(0, 1))
    ptp_axis = np.ptp(db, axis=(0, 1))
    derivative_axis = np.sqrt(np.mean(diff * diff, axis=(0, 1)))
    site_rms = np.sqrt(np.mean(centered * centered, axis=(0, 2)))
    spatial = np.asarray([np.mean(site_rms), np.std(site_rms), np.max(site_rms)])
    derivative_norm = np.linalg.norm(diff, axis=2)
    temporal = np.asarray([
        np.mean(derivative_norm), np.std(derivative_norm), np.max(derivative_norm)
    ])
    db_norm = np.linalg.norm(db, axis=2)
    regions = np.asarray([np.mean(db_norm[:, start : start + 5]) for start in (0, 5, 10)])
    return np.concatenate((abs_axis, std_axis, ptp_axis, derivative_axis, spatial, temporal, regions))


def _build_windows(data: np.lib.npyio.NpzFile, window_s: float, step_s: float):
    hall = np.asarray(data["hall_xyz"])
    t = np.asarray(data["monotonic_ns"], dtype=np.float64) * 1.0e-9
    trials = np.asarray(data["trial_id"], dtype=np.int64)
    surface = np.asarray(data["surface"]).astype(str)
    phase = np.asarray(data["phase"]).astype(str)
    saturated = np.asarray(data["saturated"], dtype=bool)
    xs, ys, groups = [], [], []
    trial_rows = []
    for trial in sorted(set(trials[trials >= 0].tolist())):
        base_mask = (trials == trial) & (phase == "baseline_unloaded") & ~saturated
        shear_mask = (trials == trial) & (phase == "shear_probe") & ~saturated
        if np.count_nonzero(base_mask) < 50 or np.count_nonzero(shear_mask) < 100:
            raise ValueError(f"trial {trial} has insufficient baseline/shear data")
        labels = np.unique(surface[trials == trial])
        labels = labels[np.isin(labels, ("high", "low"))]
        if len(labels) != 1:
            raise ValueError(f"trial {trial} has invalid surface labels: {labels}")
        y = int(labels[0] == "low")
        baseline = np.median(hall[base_mask].astype(np.float64), axis=0)
        idx = np.flatnonzero(shear_mask)
        start_t, end_t = t[idx[0]], t[idx[-1]]
        trial_features = []
        cursor = start_t
        while cursor + window_s <= end_t + 1.0e-9:
            window_idx = idx[(t[idx] >= cursor) & (t[idx] < cursor + window_s)]
            if len(window_idx) >= 20:
                feature = _features(hall[window_idx], baseline)
                xs.append(feature)
                ys.append(y)
                groups.append(trial)
                trial_features.append(feature)
            cursor += step_s
        if not trial_features:
            raise ValueError(f"trial {trial} produced no valid windows")
        trial_rows.append({
            "trial": int(trial),
            "surface": labels[0],
            "frames_baseline": int(np.count_nonzero(base_mask)),
            "frames_shear": int(np.count_nonzero(shear_mask)),
            "windows": len(trial_features),
        })
    return np.asarray(xs), np.asarray(ys), np.asarray(groups), trial_rows


def _fit_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    z = (x_train - mean) / scale
    zt = (x_test - mean) / scale
    m0, m1 = np.mean(z[y_train == 0], axis=0), np.mean(z[y_train == 1], axis=0)
    centered = z - np.where(y_train[:, None] == 1, m1, m0)
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.75 * covariance + 0.25 * diagonal + np.eye(z.shape[1]) * 1.0e-4
    weight = np.linalg.solve(covariance, m1 - m0)
    threshold = 0.5 * float((m0 + m1) @ weight)
    return zt @ weight, threshold


def _cross_validate(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    scores = np.full(len(y), np.nan)
    predictions = np.zeros(len(y), dtype=np.int64)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if len(np.unique(y[train])) != 2:
            raise ValueError("each fold must contain both surfaces")
        fold_score, threshold = _fit_scores(x[train], y[train], x[test])
        scores[test] = fold_score
        predictions[test] = fold_score >= threshold
    balanced = 0.5 * (
        np.mean(predictions[y == 0] == 0) + np.mean(predictions[y == 1] == 1)
    )
    trial_y, trial_score, trial_pred = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        trial_y.append(int(y[mask][0]))
        trial_score.append(float(np.mean(scores[mask])))
        trial_pred.append(int(np.mean(predictions[mask]) >= 0.5))
    trial_y_a = np.asarray(trial_y)
    trial_score_a = np.asarray(trial_score)
    trial_pred_a = np.asarray(trial_pred)
    trial_balanced = 0.5 * (
        np.mean(trial_pred_a[trial_y_a == 0] == 0)
        + np.mean(trial_pred_a[trial_y_a == 1] == 1)
    )
    return {
        "window_auc": _auc(y, scores),
        "window_balanced_accuracy": float(balanced),
        "trial_auc": _auc(trial_y_a, trial_score_a),
        "trial_balanced_accuracy": float(trial_balanced),
        "trial_scores_low_positive": trial_score,
        "trial_labels_low_is_one": trial_y,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-s", type=float, default=0.50)
    parser.add_argument("--step-s", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with np.load(args.dataset, allow_pickle=False) as data:
            required = {"monotonic_ns", "trial_id", "surface", "phase", "hall_xyz", "saturated"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            if np.asarray(data["hall_xyz"]).shape[1:] != (15, 3):
                raise ValueError("hall_xyz must be [T,15,3]")
            x, y, groups, trials = _build_windows(data, args.window_s, args.step_s)
        result = _cross_validate(x, y, groups)
        result.update({
            "format": "g1-friction-hall-separability-report-v1",
            "dataset": str(args.dataset.resolve()),
            "measurement_boundary": "Hall temporal separability only; not force or friction coefficient regression",
            "window_s": args.window_s,
            "step_s": args.step_s,
            "feature_dim": int(x.shape[1]),
            "windows": int(len(x)),
            "trials": trials,
        })
        pass_gate = bool(result["trial_auc"] >= 0.80 and result["trial_balanced_accuracy"] >= 0.75)
        result["preliminary_gate"] = {
            "pass": pass_gate,
            "requirements": "trial-wise AUC>=0.80 and balanced accuracy>=0.75",
            "meaning": (
                "Under this controlled protocol, high/low labelled surfaces produced separable multi-frame Hall patterns."
                if pass_gate
                else "This dataset does not yet prove reliable high/low surface separability."
            ),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        print(text, end="")
        output = args.output or args.dataset.with_name("analysis.json")
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output)
        return 0 if pass_gate else 1
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
