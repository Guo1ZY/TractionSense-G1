#!/usr/bin/env python3
"""Train a trial-held-out Hall-only high/low friction classifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from friction_runtime import FEATURE_NAMES, MODEL_FORMAT, SENSOR_ORDER, extract_window_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=0.60)
    parser.add_argument("--stride-s", type=float, default=0.20)
    parser.add_argument("--nominal-rate", type=float, default=50.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[y == 1]
    negative = scores[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        0.5 * np.mean(prediction[y == 0] == 0)
        + 0.5 * np.mean(prediction[y == 1] == 1)
    )


def _fit_lda(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    z = (x - mean) / scale
    class_high = np.mean(z[y == 0], axis=0)
    class_low = np.mean(z[y == 1], axis=0)
    centered = z - np.where(y[:, None] == 1, class_low, class_high)
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    diagonal = np.diag(np.diag(covariance))
    regularized = 0.60 * covariance + 0.40 * diagonal + np.eye(x.shape[1]) * 1.0e-3
    weight = np.linalg.solve(regularized, class_low - class_high)
    bias = -0.5 * float((class_low + class_high) @ weight)
    return mean, scale, weight, bias


def _scores(x: np.ndarray, model: Tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> np.ndarray:
    mean, scale, weight, bias = model
    return ((x - mean) / scale) @ weight + bias


def _temperature(scores: np.ndarray, y: np.ndarray) -> float:
    labels = y.astype(np.float64)
    candidates = np.logspace(-2.0, 2.0, 401)
    losses = []
    for value in candidates:
        logits = np.clip(scores / value, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        losses.append(
            -float(np.mean(labels * np.log(probability + 1.0e-9) + (1.0 - labels) * np.log(1.0 - probability + 1.0e-9)))
        )
    return float(candidates[int(np.argmin(losses))])


def _load_windows(
    paths: Sequence[Path], window_frames: int, stride_frames: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    groups: List[int] = []
    manifests: List[Dict[str, Any]] = []
    contracts = set()
    for group, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            required = {"sequence", "valid", "age_s", "hall_xyz", "metadata_json"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            if metadata.get("format") != "g1-dual-foot-labelled-friction-trial-v1":
                raise ValueError(f"{path}: unsupported dataset format")
            if metadata.get("measurement") != "raw dual-foot Hall Bx/By/Bz and temperature only":
                raise ValueError(f"{path}: wrong measurement boundary")
            surface = metadata.get("surface_label")
            if surface not in ("high", "low"):
                raise ValueError(f"{path}: invalid surface label")
            contracts.add(
                (metadata.get("controller_mode"), round(float(metadata.get("requested_vx_mps")), 3))
            )
            sequence = np.asarray(data["sequence"], dtype=np.int64)
            valid = np.asarray(data["valid"], dtype=bool)
            age = np.asarray(data["age_s"], dtype=np.float64)
            hall = np.asarray(data["hall_xyz"])
            if hall.ndim != 4 or hall.shape[1:] != (2, 15, 3):
                raise ValueError(f"{path}: hall_xyz must be [T,2,15,3]")
            if not (len(sequence) == len(valid) == len(age) == len(hall)):
                raise ValueError(f"{path}: row count mismatch")
            if np.any(np.diff(sequence) != 1):
                raise ValueError(f"{path}: non-contiguous F0R1 sequence")
            healthy = np.all(valid, axis=1) & (np.max(age, axis=1) <= 0.20)
            count = 0
            for start in range(0, len(hall) - window_frames + 1, stride_frames):
                stop = start + window_frames
                if not np.all(healthy[start:stop]):
                    continue
                xs.append(extract_window_features(hall[start:stop]))
                ys.append(int(surface == "low"))
                groups.append(group)
                count += 1
            if count < 10:
                raise ValueError(f"{path}: fewer than ten healthy windows")
            manifests.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "surface": surface,
                    "trial_id": metadata.get("trial_id"),
                    "windows": count,
                    "controller_mode": metadata.get("controller_mode"),
                    "requested_vx_mps": metadata.get("requested_vx_mps"),
                }
            )
    if len(contracts) != 1:
        raise ValueError(
            "anti-confound gate failed: all high/low trials must use exactly one "
            "controller_mode and requested_vx_mps"
        )
    return np.asarray(xs), np.asarray(ys), np.asarray(groups), manifests


def main() -> int:
    args = parse_args()
    if args.window_s <= 0.0 or args.stride_s <= 0.0 or not 20.0 <= args.nominal_rate <= 200.0:
        raise ValueError("invalid window, stride, or rate")
    window_frames = int(round(args.window_s * args.nominal_rate))
    stride_frames = max(1, int(round(args.stride_s * args.nominal_rate)))
    x, y, groups, manifests = _load_windows(args.datasets, window_frames, stride_frames)
    group_labels = np.asarray([int(item["surface"] == "low") for item in manifests])
    if np.count_nonzero(group_labels == 0) < 3 or np.count_nonzero(group_labels == 1) < 3:
        raise ValueError("at least three independent trials per surface are required")

    cv_scores = np.full(len(y), np.nan, dtype=np.float64)
    cv_predictions = np.zeros(len(y), dtype=np.int64)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if len(np.unique(y[train])) != 2:
            raise ValueError("each held-out fold must retain both classes")
        model = _fit_lda(x[train], y[train])
        score = _scores(x[test], model)
        cv_scores[test] = score
        cv_predictions[test] = score >= 0.0

    trial_scores = np.asarray([np.mean(cv_scores[groups == group]) for group in np.unique(groups)])
    trial_predictions = trial_scores >= 0.0
    window_auc = _auc(y, cv_scores)
    trial_auc = _auc(group_labels, trial_scores)
    window_balanced = _balanced_accuracy(y, cv_predictions)
    trial_balanced = _balanced_accuracy(group_labels, trial_predictions)
    passed = bool(
        np.isfinite(trial_auc)
        and trial_auc >= 0.85
        and trial_balanced >= 0.80
        and window_auc >= 0.75
    )

    final_model = _fit_lda(x, y)
    temperature = _temperature(cv_scores, y)
    mean, scale, weight, bias = final_model
    document = {
        "format": MODEL_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "Hall Bx/By/Bz temporal features only",
        "force_conversion": False,
        "sensor_order": SENSOR_ORDER,
        "feature_names": FEATURE_NAMES,
        "linear_model": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weight": weight.tolist(),
            "bias": bias,
            "probability_temperature": temperature,
        },
        "runtime": {
            "window_frames": window_frames,
            "nominal_rate_hz": args.nominal_rate,
            "enter_low_probability": 0.80,
            "clear_low_probability": 0.20,
            "fallback_on_uncertain_or_fault": "waist_walk_speed_cap_0.25_mps",
        },
        "validation": {
            "method": "leave-one-entire-trial-out; no frame-random split",
            "passed": passed,
            "requirements": "trial AUC>=0.85, trial balanced accuracy>=0.80, window AUC>=0.75",
            "trial_auc": trial_auc,
            "trial_balanced_accuracy": trial_balanced,
            "window_auc": window_auc,
            "window_balanced_accuracy": window_balanced,
            "trial_scores_low_positive": trial_scores.tolist(),
            "trial_labels_low_is_one": group_labels.tolist(),
        },
        "training_data": manifests,
        "deployment_status": "OBSERVE_ONLY" if passed else "REJECTED_VALIDATION_FAILED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(document["validation"], ensure_ascii=False, indent=2))
    print(f"model={args.output} status={document['deployment_status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
