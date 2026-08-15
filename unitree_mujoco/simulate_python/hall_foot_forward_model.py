"""Independent MuJoCo TPU-deformation -> embedded-magnet -> Hall forward model.

Contact force is used only as a simulator-internal mechanical load.  The public
output is the normalized dual-foot Bx/By/Bz residual; no magnetic-to-force or
magnetic-to-friction inverse exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np


FEET = 2
SENSORS = 15
AXES = 3
MU0_OVER_4PI = 1.0e-7

NORMALIZED_POSITIONS = np.asarray(
    (
        (0.353772, -0.021447),
        (0.283190, 0.162250),
        (0.282175, -0.015712),
        (0.277814, -0.210734),
        (0.185509, -0.019449),
        (0.064257, 0.007302),
        (-0.009736, 0.177225),
        (-0.012185, 0.001879),
        (-0.016400, -0.203782),
        (-0.088951, -0.007490),
        (-0.236309, -0.006487),
        (-0.291583, 0.146060),
        (-0.296870, -0.007403),
        (-0.295985, -0.176816),
        (-0.362058, -0.004481),
    ),
    dtype=np.float64,
)
CHIP_YAW_RAD = np.deg2rad(
    np.asarray((-90, -90, 0, 90, 0, -90, -90, 180, 90, -90, 0, -90, 180, 90, 0))
)


@dataclass(frozen=True)
class HallFootForwardConfig:
    sole_length_m: float = 0.21502
    sole_width_m: float = 0.08004
    sole_origin_x_m: float = 0.035
    magnet_spacing_x_m: float = 0.006
    magnet_spacing_y_m: float = 0.006
    magnet_embedding_depth_m: float = 0.004
    magnet_thickness_m: float = 0.001
    hall_magnet_distance_m: float = 0.006
    magnetic_moment_a_m2: float = 0.010
    normal_stiffness_n_m: float = 1.5e5
    shear_stiffness_n_m: float = 5.0e4
    local_damping_n_s_m: float = 3.0e3
    tpu_thickness_m: float = 0.010
    contact_spread_sigma_m: float = 0.035
    maximum_normal_compression_m: float = 0.004
    maximum_shear_displacement_m: float = 0.004
    maximum_local_rotation_rad: float = 0.35
    # A TPU layer is sheared not only by the bounded contact traction but also
    # by relative tangential motion while a foot is slipping.  This term is a
    # mechanical forward-model parameter (not a friction or force observation
    # exposed to the policy) and keeps low-grip slip visible in dB histories.
    slip_velocity_gain_s: float = 0.030
    observation_scale_t: float = 0.010
    observation_clip: float = 6.0
    low_pass_cutoff_hz: float = 20.0
    dipole_min_distance_m: float = 5.0e-4
    noise_std_normalized: float = 0.0

    def validate(self) -> None:
        positive = (
            self.sole_length_m,
            self.sole_width_m,
            self.magnet_thickness_m,
            self.hall_magnet_distance_m,
            self.magnetic_moment_a_m2,
            self.normal_stiffness_n_m,
            self.shear_stiffness_n_m,
            self.tpu_thickness_m,
            self.contact_spread_sigma_m,
            self.observation_scale_t,
            self.dipole_min_distance_m,
        )
        if min(positive) <= 0.0:
            raise ValueError("Hall/TPU dimensions and material parameters must be positive")
        magnet_top = self.magnet_embedding_depth_m - 0.5 * self.magnet_thickness_m
        magnet_bottom = self.magnet_embedding_depth_m + 0.5 * self.magnet_thickness_m
        if magnet_top < 0.0 or magnet_bottom > self.tpu_thickness_m:
            raise ValueError("the complete magnet thickness must be embedded inside TPU")
        if self.hall_magnet_distance_m < self.magnet_embedding_depth_m:
            raise ValueError("Hall sample must remain above the TPU top surface")


def randomized_config(
    base: HallFootForwardConfig, rng: np.random.Generator
) -> HallFootForwardConfig:
    """Episode-level sim-to-real randomization without changing interfaces."""
    return replace(
        base,
        magnetic_moment_a_m2=base.magnetic_moment_a_m2 * rng.uniform(0.72, 1.28),
        magnet_spacing_x_m=base.magnet_spacing_x_m * rng.uniform(0.85, 1.15),
        magnet_spacing_y_m=base.magnet_spacing_y_m * rng.uniform(0.85, 1.15),
        hall_magnet_distance_m=base.hall_magnet_distance_m * rng.uniform(0.90, 1.10),
        normal_stiffness_n_m=base.normal_stiffness_n_m * rng.uniform(0.65, 1.45),
        shear_stiffness_n_m=base.shear_stiffness_n_m * rng.uniform(0.65, 1.45),
        local_damping_n_s_m=base.local_damping_n_s_m * rng.uniform(0.60, 1.50),
        contact_spread_sigma_m=base.contact_spread_sigma_m * rng.uniform(0.80, 1.25),
        slip_velocity_gain_s=base.slip_velocity_gain_s * rng.uniform(0.70, 1.35),
        noise_std_normalized=rng.uniform(0.0, 0.025),
    )


def _euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
            (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
            (-sy, cy * sx, cx * cy),
        )
    )


def dipole_field(
    sensor_from_magnet_m: np.ndarray,
    moment_a_m2: np.ndarray,
    minimum_distance_m: float,
) -> np.ndarray:
    raw_distance = float(np.linalg.norm(sensor_from_magnet_m))
    distance = max(raw_distance, minimum_distance_m)
    direction = sensor_from_magnet_m / distance
    projection = float(moment_a_m2 @ direction)
    return MU0_OVER_4PI / distance**3 * (
        3.0 * projection * direction - moment_a_m2
    )


class HallFootForwardModel:
    """Scheme-A local compliance with four TPU-embedded magnets per Hall site."""

    def __init__(
        self,
        config: HallFootForwardConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config or HallFootForwardConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)
        left = np.empty((SENSORS, 2), dtype=np.float64)
        left[:, 0] = (
            self.config.sole_origin_x_m
            + NORMALIZED_POSITIONS[:, 0] * self.config.sole_length_m
        )
        left[:, 1] = NORMALIZED_POSITIONS[:, 1] * self.config.sole_width_m
        self.positions_xy_m = np.stack((left, left * np.asarray((1.0, -1.0))))
        self.deformation = np.zeros((FEET, SENSORS, 6), dtype=np.float64)
        self.filtered = np.zeros((FEET, SENSORS, AXES), dtype=np.float64)
        self.baseline = self._all_fields(np.zeros_like(self.deformation))

    @property
    def hall_to_tpu_top_distance_m(self) -> float:
        return (
            self.config.hall_magnet_distance_m
            - self.config.magnet_embedding_depth_m
        )

    def reset(self) -> None:
        self.deformation.fill(0.0)
        self.filtered.fill(0.0)

    def _field_for_site(self, foot: int, sensor: int, state: np.ndarray) -> np.ndarray:
        dx, dy, dz, roll, pitch, yaw = state
        rotation = _euler_xyz(roll, pitch, yaw)
        moment = rotation @ np.asarray((0.0, 0.0, self.config.magnetic_moment_a_m2))
        sx = 0.5 * self.config.magnet_spacing_x_m
        sy = 0.5 * self.config.magnet_spacing_y_m
        offsets = np.asarray(
            (
                (-sx, -sy, -self.config.hall_magnet_distance_m),
                (-sx, sy, -self.config.hall_magnet_distance_m),
                (sx, -sy, -self.config.hall_magnet_distance_m),
                (sx, sy, -self.config.hall_magnet_distance_m),
            )
        )
        field = np.zeros(AXES, dtype=np.float64)
        # One deformation belongs to one local TPU material element.  The four
        # inclusions are not simulated as separate rigid bodies.
        for offset in offsets:
            rotated = rotation @ offset
            sensor_from_magnet = -np.asarray((dx, dy, dz)) - rotated
            field += dipole_field(
                sensor_from_magnet,
                moment,
                self.config.dipole_min_distance_m,
            )
        c, s = np.cos(CHIP_YAW_RAD[sensor]), np.sin(CHIP_YAW_RAD[sensor])
        chip = np.asarray((c * field[0] + s * field[1], -s * field[0] + c * field[1], field[2]))
        if foot == 1:
            chip[1] *= -1.0
        return chip

    def _all_fields(self, deformation: np.ndarray) -> np.ndarray:
        result = np.empty((FEET, SENSORS, AXES), dtype=np.float64)
        for foot in range(FEET):
            for sensor in range(SENSORS):
                result[foot, sensor] = self._field_for_site(
                    foot, sensor, deformation[foot, sensor]
                )
        return result

    def update(
        self,
        dt_s: float,
        contacts: tuple[list[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ) -> np.ndarray:
        """Advance the hidden mechanics and return normalized local dB."""
        dt_s = float(np.clip(dt_s, 1.0e-4, 0.25))
        force = np.zeros((FEET, SENSORS, AXES), dtype=np.float64)
        variance = self.config.contact_spread_sigma_m**2
        for foot in range(FEET):
            for contact in contacts[foot]:
                position_xy, force_local = contact[:2]
                tangential_velocity = (
                    np.asarray(contact[2], dtype=np.float64)[:2]
                    if len(contact) >= 3
                    else np.zeros(2, dtype=np.float64)
                )
                delta = self.positions_xy_m[foot] - np.asarray(position_xy)[None]
                weight = np.exp(-0.5 * np.sum(delta**2, axis=1) / variance)
                weight /= max(float(weight.sum()), np.finfo(float).eps)
                force[foot, :, :2] += weight[:, None] * np.asarray(force_local)[:2]
                # Slip-induced shear is represented as an equivalent local
                # displacement.  It is filtered by the same TPU relaxation
                # below and remains internal to the magnetic forward model.
                force[foot, :, :2] += weight[:, None] * (
                    tangential_velocity[None] * self.config.shear_stiffness_n_m
                    * self.config.slip_velocity_gain_s
                )
                force[foot, :, 2] += weight * abs(float(np.asarray(force_local)[2]))
        target = np.zeros_like(self.deformation)
        target[..., 0] = np.clip(
            force[..., 0] / self.config.shear_stiffness_n_m,
            -self.config.maximum_shear_displacement_m,
            self.config.maximum_shear_displacement_m,
        )
        target[..., 1] = np.clip(
            force[..., 1] / self.config.shear_stiffness_n_m,
            -self.config.maximum_shear_displacement_m,
            self.config.maximum_shear_displacement_m,
        )
        target[..., 2] = np.clip(
            force[..., 2] / self.config.normal_stiffness_n_m,
            0.0,
            self.config.maximum_normal_compression_m,
        )
        target[..., 3] = np.clip(
            -target[..., 1] / self.config.tpu_thickness_m,
            -self.config.maximum_local_rotation_rad,
            self.config.maximum_local_rotation_rad,
        )
        target[..., 4] = np.clip(
            target[..., 0] / self.config.tpu_thickness_m,
            -self.config.maximum_local_rotation_rad,
            self.config.maximum_local_rotation_rad,
        )
        normal_tau = self.config.local_damping_n_s_m / self.config.normal_stiffness_n_m
        shear_tau = self.config.local_damping_n_s_m / self.config.shear_stiffness_n_m
        normal_alpha = 1.0 - np.exp(-dt_s / max(normal_tau, 1.0e-6))
        shear_alpha = 1.0 - np.exp(-dt_s / max(shear_tau, 1.0e-6))
        self.deformation[..., 2] += normal_alpha * (
            target[..., 2] - self.deformation[..., 2]
        )
        for index in (0, 1, 3, 4, 5):
            self.deformation[..., index] += shear_alpha * (
                target[..., index] - self.deformation[..., index]
            )
        absolute = self._all_fields(self.deformation)
        normalized = np.clip(
            (absolute - self.baseline) / self.config.observation_scale_t,
            -self.config.observation_clip,
            self.config.observation_clip,
        )
        if self.config.noise_std_normalized > 0.0:
            normalized += self.rng.normal(
                0.0, self.config.noise_std_normalized, normalized.shape
            )
        alpha = (
            1.0
            if self.config.low_pass_cutoff_hz <= 0.0
            else 1.0
            - np.exp(-2.0 * np.pi * self.config.low_pass_cutoff_hz * dt_s)
        )
        self.filtered += alpha * (normalized - self.filtered)
        return np.clip(
            self.filtered, -self.config.observation_clip, self.config.observation_clip
        ).astype(np.float32)


class MujocoHallContactReader:
    """Read per-contact local loads for the forward mechanics only."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.foot_body_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        if min(self.foot_body_ids) < 0:
            raise ValueError(f"missing G1 foot body: {self.foot_body_ids}")

    def _is_descendant(self, body_id: int, root_id: int) -> bool:
        while body_id >= 0:
            if body_id == root_id:
                return True
            parent = int(self.model.body_parentid[body_id])
            if parent == body_id:
                break
            body_id = parent
        return False

    def read(
        self, data: mujoco.MjData
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
        output: tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]] = ([], [])
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body_pair = (
                int(self.model.geom_bodyid[contact.geom[0]]),
                int(self.model.geom_bodyid[contact.geom[1]]),
            )
            foot = -1
            foot_side = -1
            for candidate, root in enumerate(self.foot_body_ids):
                if self._is_descendant(body_pair[0], root):
                    foot, foot_side = candidate, 0
                    break
                if self._is_descendant(body_pair[1], root):
                    foot, foot_side = candidate, 1
                    break
            if foot < 0:
                continue
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model, data, contact_index, contact_force
            )
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            world_force = frame.T @ contact_force[:3]
            if foot_side == 1:
                world_force *= -1.0
            root = self.foot_body_ids[foot]
            rotation = np.asarray(data.xmat[root]).reshape(3, 3)
            relative_world = np.asarray(contact.pos) - np.asarray(data.xpos[root])
            relative_local = rotation.T @ relative_world
            body_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, root, body_velocity, 1
            )
            tangential_velocity = body_velocity[:3] + np.cross(
                body_velocity[3:], relative_local
            )
            output[foot].append(
                (relative_local[:2], rotation.T @ world_force, tangential_velocity)
            )
        return output
