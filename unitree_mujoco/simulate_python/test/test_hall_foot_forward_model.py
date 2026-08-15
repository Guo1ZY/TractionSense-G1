from __future__ import annotations

import numpy as np
import pytest

from hall_foot_forward_model import (
    HallFootForwardConfig,
    HallFootForwardModel,
    dipole_field,
)


def test_dipole_axis_direction_and_finite_magnitude() -> None:
    field = dipole_field(
        np.asarray((0.0, 0.0, 0.01)),
        np.asarray((0.0, 0.0, 0.01)),
        1.0e-4,
    )
    assert field[2] > 0.0
    assert abs(field[0]) < 1.0e-15
    assert abs(field[1]) < 1.0e-15
    assert np.isfinite(field).all()


def test_magnets_are_embedded_and_invalid_geometry_is_rejected() -> None:
    model = HallFootForwardModel()
    assert model.hall_to_tpu_top_distance_m == pytest.approx(0.002)
    with pytest.raises(ValueError):
        HallFootForwardModel(
            HallFootForwardConfig(
                magnet_embedding_depth_m=0.0001,
                magnet_thickness_m=0.001,
            )
        )


def test_local_load_response_and_unload_to_baseline() -> None:
    model = HallFootForwardModel()
    none = ([], [])
    np.testing.assert_array_equal(model.update(0.02, none), 0.0)
    contacts = (
        [(np.asarray((0.035, 0.0)), np.asarray((20.0, -10.0, 300.0)))],
        [],
    )
    loaded = model.update(0.02, contacts)
    assert np.max(np.abs(loaded[0])) > 1.0e-5
    assert np.max(np.abs(loaded[1])) == 0.0
    for _ in range(200):
        unloaded = model.update(0.02, none)
    assert np.max(np.abs(unloaded)) < 1.0e-5


def test_left_right_mirror_is_finite_and_not_naively_copied() -> None:
    model = HallFootForwardModel()
    left_xy = np.asarray((0.035, 0.012))
    right_xy = np.asarray((0.035, -0.012))
    output = model.update(
        0.02,
        (
            [(left_xy, np.asarray((15.0, 8.0, 250.0)))],
            [(right_xy, np.asarray((15.0, -8.0, 250.0)))],
        ),
    )
    assert np.isfinite(output).all()
    assert np.max(np.abs(output[0])) > 0.0
    assert np.max(np.abs(output[1])) > 0.0
    # Mirroring includes the chip/foot y-axis convention; direct tensor copy
    # would fail this sign-sensitive check.
    assert not np.array_equal(output[0], output[1])
