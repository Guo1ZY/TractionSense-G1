from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


DEPLOY = Path(__file__).resolve().parents[1] / "robot_deploy"
sys.path.insert(0, str(DEPLOY))

from friction_runtime import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_FORMAT,
    SENSOR_ORDER,
    FrictionDecisionStateMachine,
    LinearFrictionModel,
    extract_window_features,
)


def _window(frames: int = 30) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64)[:, None, None, None]
    foot = np.arange(2, dtype=np.float64)[None, :, None, None]
    site = np.arange(15, dtype=np.float64)[None, None, :, None]
    axis = np.arange(3, dtype=np.float64)[None, None, None, :]
    return 1000.0 + 4.0 * np.sin(0.31 * t + 0.2 * site + foot) + axis


def _model_document(passed: bool = True) -> dict:
    dim = len(FEATURE_NAMES)
    return {
        "format": MODEL_FORMAT,
        "measurement": "Hall Bx/By/Bz temporal features only",
        "sensor_order": SENSOR_ORDER,
        "feature_names": FEATURE_NAMES,
        "linear_model": {
            "mean": [0.0] * dim,
            "scale": [1.0] * dim,
            "weight": [0.0] * dim,
            "bias": 0.0,
            "probability_temperature": 1.0,
        },
        "runtime": {
            "window_frames": 30,
            "nominal_rate_hz": 50.0,
            "enter_low_probability": 0.8,
            "clear_low_probability": 0.2,
        },
        "validation": {"passed": passed},
    }


def test_feature_shape_finite_and_constant_baseline_invariant() -> None:
    values = _window()
    first = extract_window_features(values)
    second = extract_window_features(values + np.arange(90).reshape(1, 2, 15, 3) * 1234.0)
    assert first.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, atol=1.0e-10, rtol=1.0e-10)


def test_feature_rejects_wrong_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        extract_window_features(np.zeros((30, 15, 3)))
    values = _window()
    values[2, 0, 3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        extract_window_features(values)


def test_decision_defaults_conservative_and_hysteretic() -> None:
    state = FrictionDecisionStateMachine(
        enter_low_hold_s=0.30, clear_low_hold_s=0.80
    )
    for _ in range(14):
        output = state.update(0.95, 0.02, True)
        assert output.requested_mode == "waist_walk"
    output = state.update(0.95, 0.02, True)
    assert output.state == "LOW"
    assert output.speed_cap_mps == pytest.approx(0.25)

    for _ in range(39):
        output = state.update(0.05, 0.02, True)
        assert output.requested_mode == "waist_walk"
    output = state.update(0.05, 0.02, True)
    assert output.state == "HIGH"
    assert output.requested_mode == "walkrun"


def test_sensor_fault_immediately_falls_back_to_waist_walk() -> None:
    state = FrictionDecisionStateMachine(clear_low_hold_s=0.02)
    assert state.update(0.01, 0.02, True).state == "HIGH"
    output = state.update(0.01, 0.02, False)
    assert output.state == "DEGRADED"
    assert output.requested_mode == "waist_walk"
    assert output.speed_cap_mps == pytest.approx(0.25)


def test_model_loader_refuses_unvalidated_model_and_wrong_schema() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(_model_document(False)), encoding="utf-8")
        with pytest.raises(ValueError, match="not passed"):
            LinearFrictionModel.load(path)
        document = _model_document(True)
        document["feature_names"] = document["feature_names"][:-1]
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            LinearFrictionModel.load(path)


def test_loaded_zero_model_returns_half_probability() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(_model_document(True)), encoding="utf-8")
        model = LinearFrictionModel.load(path)
        assert model.probability_low(extract_window_features(_window())) == pytest.approx(0.5)
