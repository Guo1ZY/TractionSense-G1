from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRACTION_SCRIPTS = ROOT / "scripts" / "traction"
if str(TRACTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACTION_SCRIPTS))

from train_layout_magnetic_student import (
    AGE_SLICE,
    VALID_SLICE,
    apply_foot_dropout_metadata,
    filter_stable_action_samples,
)


def test_all_1864_offline_trainers_require_explicit_trailing_semantics() -> None:
    for relative in (
        "train_layout_magnetic_student.py",
        "train_hall_risk_estimator.py",
        "train_hall_slip_risk_estimator.py",
    ):
        source = (TRACTION_SCRIPTS / relative).read_text(encoding="utf-8")
        argument = source.split('"--trailing-feature-mode"', 1)[1].split(
            ")", 1
        )[0]
        assert "required=True" in argument
        assert 'default="sensor_age"' not in argument


def test_stable_action_filter_removes_only_post_reset_like_outliers() -> None:
    data = {
        "obs": np.zeros((3, 1864), dtype=np.float32),
        "mu": np.asarray([0.8, 0.08, 0.8], dtype=np.float32),
    }
    teacher = np.zeros((3, 29), dtype=np.float32)
    baseline = np.zeros((3, 29), dtype=np.float32)
    teacher[1, 0] = 12.0

    filtered, filtered_teacher, filtered_baseline, report = (
        filter_stable_action_samples(data, teacher, baseline, 3.0)
    )

    assert filtered["obs"].shape == (2, 1864)
    assert filtered_teacher.shape == filtered_baseline.shape == (2, 29)
    np.testing.assert_allclose(filtered["mu"], [0.8, 0.8])
    assert report == {
        "input_samples": 3,
        "kept_samples": 2,
        "dropped_post_reset_or_outlier_samples": 1,
    }


def test_motion_feedback_is_never_rewritten_as_packet_age_on_foot_dropout() -> None:
    observation = torch.zeros((2, 1864), dtype=torch.float32)
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = torch.tensor(((0.35, -0.20), (-0.70, 0.45)))
    original_motion = observation[:, AGE_SLICE].clone()
    dropout = torch.tensor(((True, False), (False, True)))

    apply_foot_dropout_metadata(observation, dropout, "motion_feedback")

    torch.testing.assert_close(
        observation[:, VALID_SLICE],
        torch.tensor(((0.0, 1.0), (1.0, 0.0))),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(observation[:, AGE_SLICE], original_motion)


def test_sensor_age_mode_marks_only_the_dropped_foot_stale() -> None:
    observation = torch.zeros((1, 1864), dtype=torch.float32)
    observation[:, VALID_SLICE] = 1.0
    observation[:, AGE_SLICE] = torch.tensor(((0.1, 0.2),))

    apply_foot_dropout_metadata(
        observation, torch.tensor(((False, True),)), "sensor_age"
    )

    torch.testing.assert_close(
        observation[:, VALID_SLICE], torch.tensor(((1.0, 0.0),))
    )
    torch.testing.assert_close(
        observation[:, AGE_SLICE], torch.tensor(((0.1, 1.0),))
    )
