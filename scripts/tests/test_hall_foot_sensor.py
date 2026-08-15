from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from unitree_rl_lab.sensors import (
    DEFAULT_HALL_POSITIONS_IMAGE_PX,
    DEFAULT_HALL_POSITIONS_NORMALIZED,
    HALL_LAYOUT_IMAGE_SIZE_PX,
    HALL_LAYOUT_SOLE_BOUNDS_PX,
    HALL_LAYOUT_SOURCE_IMAGE,
    DipoleMagneticFieldModel,
    HallFootSensor,
    HallFootSensorCfg,
)
from unitree_rl_lab.sensors.hall_sensor_noise import HallSensorSignalProcessor


def _quiet_cfg(**updates) -> HallFootSensorCfg:
    values = {
        "auto_zero": False,
        "noise_std": (0.0, 0.0, 0.0),
        "bias": (0.0, 0.0, 0.0),
        "resolution": 0.0,
        "low_pass_cutoff": 0.0,
        "drift_std_per_sqrt_s": 0.0,
    }
    values.update(updates)
    return HallFootSensorCfg(**values)


def _identity_foot_pose(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.zeros((batch, 2, 3))
    quaternion = torch.zeros((batch, 2, 4))
    quaternion[..., 0] = 1.0
    return position, quaternion


def test_a4_layout_is_three_ordered_five_point_crosses() -> None:
    assert len(DEFAULT_HALL_POSITIONS_IMAGE_PX) == 15
    normalized = torch.tensor(DEFAULT_HALL_POSITIONS_NORMALIZED)
    for start in (0, 5, 10):
        top, left, centre, right, bottom = normalized[start : start + 5]
        assert top[0] > torch.max(torch.stack((left[0], centre[0], right[0])))
        assert bottom[0] < torch.min(torch.stack((left[0], centre[0], right[0])))
        assert left[1] > centre[1] > right[1]
    cfg = HallFootSensorCfg()
    assert cfg.hall_package_size == (0.004, 0.004, 0.001)


def test_real_dashboard_layout_uses_exact_same_a4_sensor_centres() -> None:
    path = Path(HALL_LAYOUT_SOURCE_IMAGE).parent / "config" / "sensor_layout_a4_15.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(document["source_image_size_px"]) == HALL_LAYOUT_IMAGE_SIZE_PX
    assert tuple(document["sole_ink_bounds_px"]) == HALL_LAYOUT_SOLE_BOUNDS_PX
    assert document["sole_length_m"] == HallFootSensorCfg().sole_length
    assert document["sole_width_m"] == HallFootSensorCfg().sole_width
    sensors = document["sensors"]
    assert [item["output_id"] for item in sensors] == [f"P{i:02d}" for i in range(15)]
    torch.testing.assert_close(
        torch.tensor([item["source_pixel_uv"] for item in sensors]),
        torch.tensor(DEFAULT_HALL_POSITIONS_IMAGE_PX),
        atol=0.0,
        rtol=0.0,
    )
    dashboard_uv = torch.tensor([item["normalized_uv"] for item in sensors])
    dashboard_in_foot_frame = torch.stack(
        (0.5 - dashboard_uv[:, 1], 0.5 - dashboard_uv[:, 0]), dim=-1
    )
    torch.testing.assert_close(
        dashboard_in_foot_frame,
        torch.tensor(DEFAULT_HALL_POSITIONS_NORMALIZED),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_magnets_are_fully_embedded_inside_tpu_geometry() -> None:
    cfg = HallFootSensorCfg()
    tpu_top = -cfg.hall_to_tpu_top_distance
    tpu_bottom = tpu_top - cfg.tpu_thickness
    magnet_center = -cfg.initial_hall_magnet_distance
    assert math.isclose(
        tpu_top - magnet_center, cfg.magnet_embedding_depth, abs_tol=1.0e-12
    )
    assert magnet_center + 0.5 * cfg.magnet_thickness <= tpu_top
    assert magnet_center - 0.5 * cfg.magnet_thickness >= tpu_bottom


def test_dipole_axis_direction_and_order_of_magnitude() -> None:
    model = DipoleMagneticFieldModel(min_distance=1.0e-6)
    sensor = torch.tensor([[[[0.0, 0.0, 0.01]]]])
    magnet = torch.tensor([[[[[0.0, 0.0, 0.0]]]]])
    moment = torch.tensor([[[[[0.0, 0.0, 0.01]]]]])
    field = model.compute(sensor, magnet, moment)
    expected_bz = 1.0e-7 * 2.0 * 0.01 / 0.01**3
    assert torch.allclose(field[..., :2], torch.zeros_like(field[..., :2]), atol=1.0e-10)
    assert torch.allclose(field[..., 2], torch.tensor([[[expected_bz]]]), rtol=1.0e-5)

    equatorial_sensor = torch.tensor([[[[0.01, 0.0, 0.0]]]])
    equatorial = model.compute(equatorial_sensor, magnet, moment)
    assert equatorial[..., 2].item() < 0.0
    assert math.isclose(abs(equatorial[..., 2].item()), 0.5 * expected_bz, rel_tol=1.0e-5)


def test_four_magnet_superposition() -> None:
    model = DipoleMagneticFieldModel(min_distance=1.0e-6)
    sensor = torch.tensor([[[[0.0, 0.0, 0.006]]]])
    xy = ((-0.003, -0.003), (-0.003, 0.003), (0.003, -0.003), (0.003, 0.003))
    magnet = torch.tensor([[[[[x, y, 0.0] for x, y in xy]]]])
    moment = torch.zeros_like(magnet)
    moment[..., 2] = 0.01
    total = model.compute(sensor, magnet, moment)
    individual_sum = sum(
        model.compute(sensor, magnet[..., index : index + 1, :], moment[..., index : index + 1, :])
        for index in range(4)
    )
    assert torch.allclose(total, individual_sum, rtol=1.0e-6, atol=1.0e-10)
    assert abs(total[..., 0].item()) < 1.0e-9
    assert abs(total[..., 1].item()) < 1.0e-9
    assert total[..., 2].item() > 0.0


def test_world_to_hall_local_rotation_is_invariant() -> None:
    cfg = _quiet_cfg(
        hall_positions_normalized=((0.0, 0.0),),
        hall_axis_yaw_deg=(0.0,),
        mirror_right_y=False,
        right_hall_axis_sign=(1.0, 1.0, 1.0),
    )
    sensor = HallFootSensor(cfg)
    sensor.initialize(1, "cpu")
    position, quaternion = _identity_foot_pose(1)
    deformation = torch.zeros((1, 2, 1, 6))
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=deformation,
    )
    reference = sensor.get_debug_data()["ideal_magnetic_field"].clone()

    angle = 0.5 * math.pi
    quaternion[..., 0] = math.cos(0.5 * angle)
    quaternion[..., 3] = math.sin(0.5 * angle)
    position[..., 0] = 0.37
    position[..., 1] = -0.21
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=deformation,
    )
    rotated = sensor.get_debug_data()["ideal_magnetic_field"]
    assert torch.allclose(rotated, reference, rtol=2.0e-5, atol=2.0e-8)


def test_right_foot_position_and_axis_mirror() -> None:
    cfg = _quiet_cfg(hall_axis_yaw_deg=(0.0,) * 15)
    sensor = HallFootSensor(cfg)
    sensor.initialize(1, "cpu")
    assert torch.allclose(sensor.hall_positions_f[0, :, 0], sensor.hall_positions_f[1, :, 0])
    assert torch.allclose(sensor.hall_positions_f[0, :, 1], -sensor.hall_positions_f[1, :, 1])

    position, quaternion = _identity_foot_pose(1)
    deformation = torch.zeros((1, 2, 15, 6))
    deformation[..., 1] = 0.0008
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=deformation,
    )
    field = sensor.get_debug_data()["ideal_magnetic_field"]
    assert torch.allclose(field[:, 0, :, 0], field[:, 1, :, 0], rtol=1.0e-5, atol=1.0e-8)
    assert torch.allclose(field[:, 0, :, 1], -field[:, 1, :, 1], rtol=1.0e-5, atol=1.0e-8)
    assert torch.allclose(field[:, 0, :, 2], field[:, 1, :, 2], rtol=1.0e-5, atol=1.0e-8)


def test_reset_clears_and_reacquires_zero_load_baseline() -> None:
    cfg = _quiet_cfg(auto_zero=True, auto_zero_samples=2)
    sensor = HallFootSensor(cfg)
    sensor.initialize(2, "cpu")
    position, quaternion = _identity_foot_pose(2)
    unloaded = torch.zeros((2, 2, 15, 6))
    for _ in range(2):
        sensor.update(
            0.02,
            foot_positions_w=position,
            foot_quaternions_w=quaternion,
            local_deformation=unloaded,
        )
    assert sensor.get_debug_data()["valid_mask"].all()
    assert sensor.get_filtered_data().abs().max() == 0.0

    loaded = unloaded.clone()
    loaded[..., 2] = 0.001
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=loaded,
    )
    assert sensor.get_filtered_data().abs().max() > 0.0
    sensor.reset(torch.tensor([1]))
    assert sensor.get_filtered_data()[1].abs().max() == 0.0
    assert not sensor.get_debug_data()["valid_mask"][1].any()
    for _ in range(2):
        sensor.update(
            0.02,
            foot_positions_w=position,
            foot_quaternions_w=quaternion,
            local_deformation=loaded,
        )
    assert sensor.get_filtered_data()[1].abs().max() == 0.0


def test_noise_quantization_saturation_and_lowpass() -> None:
    cfg = _quiet_cfg(
        bias=(0.01, 0.0, 0.0),
        resolution=0.01,
        saturation_min=(-0.05, -0.05, -0.05),
        saturation_max=(0.05, 0.05, 0.05),
        low_pass_cutoff=1.0,
    )
    processor = HallSensorSignalProcessor(cfg, 1, 1, device="cpu", seed=7)
    high = torch.full((1, 2, 1, 3), 0.20)
    first = processor.update(high, 0.02)
    assert torch.allclose(first.raw, torch.full_like(first.raw, 0.05))
    zero = torch.zeros_like(high)
    second = processor.update(zero, 0.02)
    assert torch.allclose(second.raw[..., 0], torch.full_like(second.raw[..., 0], 0.01))
    assert torch.all((second.filtered_absolute[..., 0] > 0.01) & (second.filtered_absolute[..., 0] < 0.05))

    noisy_cfg = _quiet_cfg(noise_std=(0.001, 0.001, 0.001))
    a = HallSensorSignalProcessor(noisy_cfg, 1, 1, device="cpu", seed=11)
    b = HallSensorSignalProcessor(noisy_cfg, 1, 1, device="cpu", seed=11)
    noisy_a = a.update(zero, 0.02).raw
    noisy_b = b.update(zero, 0.02).raw
    assert torch.equal(noisy_a, noisy_b)
    assert noisy_a.abs().max() > 0.0


def test_nominal_policy_observation_is_dimensionless_hall_only() -> None:
    cfg = _quiet_cfg(observation_scale_t=0.02)
    sensor = HallFootSensor(cfg)
    sensor.initialize(2, "cpu")
    position, quaternion = _identity_foot_pose(2)
    deformation = torch.zeros((2, 2, 15, 6))
    deformation[..., 2] = 0.0007
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=deformation,
    )
    expected = torch.clamp(
        sensor.get_filtered_data() / cfg.observation_scale_t,
        cfg.observation_clip[0],
        cfg.observation_clip[1],
    )
    assert torch.allclose(sensor.get_policy_observation(), expected)
    assert sensor.get_reported_sample_period().shape == (2, 2)
    assert sensor.get_policy_valid_mask().all()


def test_structured_domain_randomization_and_whole_foot_fallback() -> None:
    cfg = _quiet_cfg(
        enable_domain_randomization=True,
        dead_channel_probability=0.0,
        foot_dropout_probability=1.0,
        maximum_packet_delay_steps=0,
        normal_stiffness_scale_range=(0.5, 0.7),
        shear_stiffness_scale_range=(1.3, 1.5),
        reported_sample_period_range=(0.03, 0.04),
    )
    sensor = HallFootSensor(cfg)
    sensor.initialize(4, "cpu", seed=19)
    debug = sensor.get_debug_data()
    assert torch.all((debug["mechanical_normal_scale"] >= 0.5) & (debug["mechanical_normal_scale"] <= 0.7))
    assert torch.all((debug["mechanical_shear_scale"] >= 1.3) & (debug["mechanical_shear_scale"] <= 1.5))
    assert torch.all((debug["reported_sample_period"] >= 0.03) & (debug["reported_sample_period"] <= 0.04))

    position, quaternion = _identity_foot_pose(4)
    deformation = torch.zeros((4, 2, 15, 6))
    deformation[..., 2] = 0.001
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        local_deformation=deformation,
    )
    assert torch.count_nonzero(sensor.get_policy_observation()) == 0
    assert not sensor.get_policy_valid_mask().any()


def test_partial_reset_resamples_only_selected_installation() -> None:
    cfg = _quiet_cfg(
        enable_domain_randomization=True,
        foot_dropout_probability=0.0,
        dead_channel_probability=0.0,
    )
    sensor = HallFootSensor(cfg)
    sensor.initialize(3, "cpu", seed=23)
    before = sensor.get_debug_data()["mechanical_normal_scale"].clone()
    sensor.reset(torch.tensor([1]))
    after = sensor.get_debug_data()["mechanical_normal_scale"]
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[2], after[2])
    assert not torch.equal(before[1], after[1])


def test_batched_shapes_local_response_and_numerical_stability() -> None:
    cfg = _quiet_cfg()
    sensor = HallFootSensor(cfg)
    sensor.initialize(8, "cpu")
    position, quaternion = _identity_foot_pose(8)
    force = torch.zeros((8, 2, 3))
    force[..., 2] = 240.0
    # Put the contact point directly below P00 in each foot.
    foot_point = sensor.hall_positions_f[:, 0].unsqueeze(0).expand(8, -1, -1)
    sensor.update(
        0.02,
        foot_positions_w=position,
        foot_quaternions_w=quaternion,
        contact_force_w=force,
        contact_point_w=foot_point,
    )
    debug = sensor.get_debug_data()
    assert sensor.get_raw_data().shape == (8, 2, 15, 3)
    assert sensor.get_filtered_data().shape == (8, 2, 15, 3)
    assert debug["magnetic_norm"].shape == (8, 2, 15)
    assert debug["local_deformation"].shape == (8, 2, 15, 6)
    assert debug["mechanical_driver_force_privileged"].shape == (8, 2, 15, 3)
    assert "contact_force" not in debug
    assert debug["valid_mask"].shape == (8, 2, 15)
    assert debug["local_deformation"][:, :, 0, 2].mean() > debug["local_deformation"][:, :, 14, 2].mean()
    assert all(torch.isfinite(value).all() for value in debug.values())
