from __future__ import annotations

import math

import pytest
import torch

from unitree_rl_lab.sensors import HallFootSensor, HallFootSensorCfg
from unitree_rl_lab.sensors.hall_contact_distribution import (
    distribute_point_forces_to_hall_sites,
    indexed_buffer_indices,
)


def _rotation_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def test_indexed_buffer_indices_unpacks_filters_and_empty_groups() -> None:
    counts = torch.tensor(((2, 0, 1), (0, 2, 0)), dtype=torch.int32)
    starts = torch.tensor(((0, 2, 2), (3, 3, 5)), dtype=torch.int32)
    rows, indices = indexed_buffer_indices(counts, starts, buffer_length=7)
    assert torch.equal(rows, torch.tensor((0, 0, 0, 1, 1)))
    assert torch.equal(indices, torch.tensor((0, 1, 2, 3, 4)))

    empty_rows, empty_indices = indexed_buffer_indices(
        torch.zeros((2, 3), dtype=torch.int64),
        torch.zeros((2, 3), dtype=torch.int64),
        buffer_length=0,
    )
    assert empty_rows.shape == empty_indices.shape == (0,)


def test_indexed_buffer_indices_rejects_corrupt_metadata() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        indexed_buffer_indices(
            torch.tensor(((2,),), dtype=torch.int64),
            torch.tensor(((3,),), dtype=torch.int64),
            buffer_length=4,
        )
    with pytest.raises(TypeError, match="integer"):
        indexed_buffer_indices(
            torch.tensor(((1.0,),)),
            torch.tensor(((0.0,),)),
            buffer_length=1,
        )


def test_detailed_distribution_preserves_force_in_each_foot_frame() -> None:
    num_envs = 2
    hall_positions_f = torch.tensor(
        (
            ((0.08, 0.02, 0.0), (0.00, 0.00, 0.0), (-0.08, -0.02, 0.0)),
            ((0.08, -0.02, 0.0), (0.00, 0.00, 0.0), (-0.08, 0.02, 0.0)),
        )
    )
    foot_positions_w = torch.tensor(
        (
            ((1.0, 2.0, 0.2), (1.0, -2.0, 0.2)),
            ((3.0, 2.0, 0.2), (3.0, -2.0, 0.2)),
        )
    )
    rotations = torch.eye(3).repeat(num_envs, 2, 1, 1)
    rotations[0, 1] = _rotation_z(0.5 * math.pi)
    rotations[1, 0] = _rotation_z(-0.5 * math.pi)

    env_indices = torch.tensor((0, 0, 1, 1), dtype=torch.long)
    foot_indices = torch.tensor((0, 1, 0, 1), dtype=torch.long)
    local_points = torch.tensor(
        ((0.08, 0.02, 0.0), (0.08, -0.02, 0.0), (0.00, 0.00, 0.0), (-0.08, 0.02, 0.0))
    )
    local_forces = torch.tensor(
        ((1.0, 2.0, 30.0), (4.0, 5.0, 60.0), (-3.0, 7.0, 40.0), (8.0, -2.0, 50.0))
    )
    sample_rotations = rotations[env_indices, foot_indices]
    points_w = foot_positions_w[env_indices, foot_indices] + torch.einsum(
        "kij,kj->ki", sample_rotations, local_points
    )
    forces_w = torch.einsum("kij,kj->ki", sample_rotations, local_forces)

    output = distribute_point_forces_to_hall_sites(
        num_envs=num_envs,
        hall_positions_f=hall_positions_f,
        foot_positions_w=foot_positions_w,
        foot_rotations_w=rotations,
        point_forces_w=forces_w,
        contact_points_w=points_w,
        contact_env_indices=env_indices,
        contact_foot_indices=foot_indices,
        spread_sigma_f=torch.full((num_envs, 2, 1), 0.015),
    )
    assert output.shape == (2, 2, 3, 3)
    torch.testing.assert_close(
        output[env_indices, foot_indices].sum(dim=1),
        local_forces,
        atol=1.0e-5,
        rtol=1.0e-6,
    )
    # Samples were placed exactly at a Hall centre with a narrow kernel.
    assert output[0, 0, 0, 2] > 100.0 * output[0, 0, 1, 2]
    assert output[0, 1, 0, 2] > 100.0 * output[0, 1, 1, 2]


def test_distribution_is_stable_for_far_points_and_zero_contacts() -> None:
    hall_positions = torch.tensor(
        (((-0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),) * 2,
        dtype=torch.float32,
    )
    positions = torch.zeros((3, 2, 3))
    rotations = torch.eye(3).repeat(3, 2, 1, 1)
    empty = distribute_point_forces_to_hall_sites(
        num_envs=3,
        hall_positions_f=hall_positions,
        foot_positions_w=positions,
        foot_rotations_w=rotations,
        point_forces_w=torch.empty((0, 3)),
        contact_points_w=torch.empty((0, 3)),
        contact_env_indices=torch.empty(0, dtype=torch.long),
        contact_foot_indices=torch.empty(0, dtype=torch.long),
        spread_sigma_f=0.01,
    )
    assert empty.shape == (3, 2, 2, 3)
    assert torch.count_nonzero(empty) == 0

    force = torch.tensor(((3.0, -2.0, 100.0),))
    far = distribute_point_forces_to_hall_sites(
        num_envs=3,
        hall_positions_f=hall_positions,
        foot_positions_w=positions,
        foot_rotations_w=rotations,
        point_forces_w=force,
        contact_points_w=torch.tensor(((1.0e6, -1.0e6, 0.0),)),
        contact_env_indices=torch.tensor((2,)),
        contact_foot_indices=torch.tensor((1,)),
        spread_sigma_f=0.001,
    )
    assert torch.isfinite(far).all()
    torch.testing.assert_close(far[2, 1].sum(dim=0), force[0])


def test_two_contact_patches_remain_spatially_bimodal() -> None:
    hall_positions = torch.tensor(
        (
            ((0.10, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.10, 0.0, 0.0)),
            ((0.10, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.10, 0.0, 0.0)),
        )
    )
    output = distribute_point_forces_to_hall_sites(
        num_envs=1,
        hall_positions_f=hall_positions,
        foot_positions_w=torch.zeros((1, 2, 3)),
        foot_rotations_w=torch.eye(3).repeat(1, 2, 1, 1),
        point_forces_w=torch.tensor(((0.0, 0.0, 100.0), (0.0, 0.0, 100.0))),
        contact_points_w=torch.tensor(((0.10, 0.0, 0.0), (-0.10, 0.0, 0.0))),
        contact_env_indices=torch.tensor((0, 0)),
        contact_foot_indices=torch.tensor((0, 0)),
        spread_sigma_f=0.015,
    )
    normal = output[0, 0, :, 2]
    assert normal[0] > 1000.0 * normal[1]
    assert normal[2] > 1000.0 * normal[1]
    torch.testing.assert_close(normal.sum(), torch.tensor(200.0))


def test_distribution_fails_closed_on_nan_or_improper_rotation_shape() -> None:
    common = {
        "num_envs": 1,
        "hall_positions_f": torch.zeros((2, 1, 3)),
        "foot_positions_w": torch.zeros((1, 2, 3)),
        "foot_rotations_w": torch.eye(3).repeat(1, 2, 1, 1),
        "point_forces_w": torch.tensor(((0.0, 0.0, 1.0),)),
        "contact_points_w": torch.tensor(((float("nan"), 0.0, 0.0),)),
        "contact_env_indices": torch.tensor((0,)),
        "contact_foot_indices": torch.tensor((0,)),
        "spread_sigma_f": 0.01,
    }
    with pytest.raises(ValueError, match="finite"):
        distribute_point_forces_to_hall_sites(**common)
    common["contact_points_w"] = torch.zeros((1, 3))
    common["foot_rotations_w"] = torch.eye(3).repeat(2, 1, 1)
    with pytest.raises(ValueError, match="foot_rotations_w"):
        distribute_point_forces_to_hall_sites(**common)


def test_hall_sensor_accepts_local_detailed_force_without_schema_change() -> None:
    cfg = HallFootSensorCfg(
        hall_positions_normalized=((0.0, 0.0), (0.2, 0.0)),
        hall_axis_yaw_deg=(0.0,),
        contact_distribution_mode="detailed",
        auto_zero=False,
        noise_std=(0.0, 0.0, 0.0),
        resolution=0.0,
        low_pass_cutoff=0.0,
        drift_std_per_sqrt_s=0.0,
    )
    sensor = HallFootSensor(cfg)
    sensor.initialize(2, "cpu")
    foot_positions = torch.zeros((2, 2, 3))
    foot_quaternions = torch.zeros((2, 2, 4))
    foot_quaternions[..., 0] = 1.0
    local_force = torch.zeros((2, 2, 2, 3))
    local_force[0, 0, 0] = torch.tensor((10.0, -5.0, 100.0))
    # Detailed raw samples remain conservative even below the legacy 1 N
    # aggregate contact-state threshold.
    local_force[0, 1, 0, 2] = 0.1
    local_force[1, 1, 1] = torch.tensor((-4.0, 3.0, 80.0))
    output = sensor.update(
        0.02,
        foot_positions_w=foot_positions,
        foot_quaternions_w=foot_quaternions,
        local_contact_force_f=local_force,
    )
    debug = sensor.get_debug_data()
    assert output.shape == (2, 2, 2, 3)
    torch.testing.assert_close(debug["mechanical_driver_force_privileged"], local_force)
    assert debug["local_deformation"][0, 0, 0, 2] > 0.0
    assert debug["local_deformation"][1, 1, 1, 0] < 0.0

    with pytest.raises(ValueError, match="mutually exclusive"):
        sensor.update(
            0.02,
            foot_positions_w=foot_positions,
            foot_quaternions_w=foot_quaternions,
            contact_force_w=torch.zeros((2, 2, 3)),
            local_contact_force_f=local_force,
        )


def test_contact_distribution_config_validation_and_default_compatibility() -> None:
    assert HallFootSensorCfg().contact_distribution_mode == "aggregate"
    with pytest.raises(ValueError, match="contact_distribution_mode"):
        HallFootSensorCfg(contact_distribution_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contact_spread_sigma"):
        HallFootSensorCfg(contact_spread_sigma=0.0)
