from __future__ import annotations

import torch

from unitree_rl_lab.traction.layout_magnetic_student import (
    AGE_SLICE,
    INPUT_DIM,
    MAGNETIC_SLICE,
    SCHEMA,
    VALID_SLICE,
    LayoutMagneticStudent,
    schema_for_trailing_feature_mode,
)


def test_schema_has_exact_layout_and_no_force_input() -> None:
    schema = SCHEMA.to_dict()
    assert schema["input_dimension"] == 1864
    assert schema["sensor_order"] == [f"P{i:02d}" for i in range(15)] or tuple(
        schema["sensor_order"]
    ) == tuple(f"P{i:02d}" for i in range(15))
    assert len(schema["sensor_positions_xy"]) == 15
    forbidden = " ".join(schema["forbidden_student_inputs"])
    assert "force" in forbidden
    assert "ground_friction_mu" in forbidden


def test_invalid_hall_is_exact_baseline_fallback() -> None:
    model = LayoutMagneticStudent().eval()
    observation = torch.randn(3, INPUT_DIM)
    observation[:, MAGNETIC_SLICE] = 0.0
    observation[:, VALID_SLICE] = 0.0
    observation[:, AGE_SLICE] = 1.0
    with torch.inference_mode():
        action, _, _, confidence, residual = model.all_outputs(observation)
        baseline = model.baseline_actor(observation[:, :480])
    assert torch.equal(action, baseline)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(confidence, torch.zeros_like(confidence))


def test_residual_is_bounded() -> None:
    model = LayoutMagneticStudent(residual_limit=1.0).eval()
    observation = torch.randn(2, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = 0.0
    with torch.inference_mode():
        residual = model.all_outputs(observation)[-1]
    assert residual.abs().max() <= 1.0


def test_motion_feedback_trailing_channels_do_not_decay_hall_confidence() -> None:
    """body-vy/heading are proprioception, not BLE packet age."""

    model = LayoutMagneticStudent(trailing_feature_mode="motion_feedback").eval()
    observation = torch.zeros(2, INPUT_DIM)
    observation[:, VALID_SLICE] = 1.0
    # These values are legal lateral velocity / heading observations but would
    # look like fully stale packets if their semantics were mixed up.
    observation[:, AGE_SLICE] = torch.tensor([[1.2, -0.7], [0.4, 1.0]])
    with torch.inference_mode():
        confidence = model.physical_confidence(observation[:, VALID_SLICE.start : AGE_SLICE.stop])
    assert torch.allclose(confidence, torch.ones_like(confidence))
    schema = schema_for_trailing_feature_mode("motion_feedback").to_dict()
    assert schema["trailing_feature_names"] == [
        "body_lateral_velocity",
        "relative_heading_error",
    ]
    assert schema["slices"]["trailing_features"] == [1862, 1864]
    assert schema["slices"]["motion_feedback"] == [1862, 1864]
    assert "age_lr" not in schema["slices"]
    assert "motion_feedback_[body_vy,relative_heading]" in schema["flatten_order"]
    assert schema["hall_frame"].startswith("per_site_hall_ic_local_xyz")
    assert len(schema["hall_axis_yaw_deg"]) == 15
    assert schema["right_foot_axis_sign"] == [1.0, -1.0, 1.0]


def test_sensor_age_schema_keeps_distinct_trailing_semantics() -> None:
    schema = schema_for_trailing_feature_mode("sensor_age").to_dict()
    assert schema["trailing_feature_names"] == [
        "sensor_age_left",
        "sensor_age_right",
    ]
    assert schema["slices"]["trailing_features"] == [1862, 1864]
    assert schema["slices"]["age_lr"] == [1862, 1864]
    assert "motion_feedback" not in schema["slices"]
    assert schema["flatten_order"].endswith("age_left_right")
