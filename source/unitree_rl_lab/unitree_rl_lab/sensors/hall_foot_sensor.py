"""Vectorized dual-foot flexible-magnet/Hall sensor for Isaac Lab.

The mechanics and magnetic field are deliberately separated.  Scheme A uses
one Kelvin--Voigt state per Hall site and never creates dynamic magnet bodies.
Scheme B consumes magnet poses derived from current deformable mesh nodes.
Both paths feed the same replaceable :class:`MagneticFieldModel` and Hall
electronics model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

import torch

from .hall_contact_distribution import distribute_point_forces_to_hall_sites
from .hall_sensor_config import HallFootSensorCfg
from .hall_sensor_noise import HallSensorSignalProcessor
from .magnetic_field_model import DipoleMagneticFieldModel, MagneticFieldModel


def _normalize(vector: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternions to local-to-parent matrices."""
    q = _normalize(quaternion_wxyz)
    w, x, y, z = q.unbind(dim=-1)
    result = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _rotation_z(angle_rad: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    result = torch.zeros((*angle_rad.shape, 3, 3), device=angle_rad.device, dtype=angle_rad.dtype)
    result[..., 0, 0] = c
    result[..., 0, 1] = -s
    result[..., 1, 0] = s
    result[..., 1, 1] = c
    result[..., 2, 2] = 1.0
    return result


def _euler_xyz_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Return intrinsic XYZ rotation matrices for roll, pitch, yaw."""
    roll, pitch, yaw = angles.unbind(dim=-1)
    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)
    result = torch.empty((*angles.shape[:-1], 3, 3), device=angles.device, dtype=angles.dtype)
    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = cz * sx * sy - cx * sz
    result[..., 0, 2] = sx * sz + cx * cz * sy
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = cx * sy * sz - cz * sx
    result[..., 2, 0] = -sy
    result[..., 2, 1] = cy * sx
    result[..., 2, 2] = cx * cy
    return result


@dataclass(frozen=True)
class DeformableMagnetPoseSample:
    """Scheme-B magnet state sampled from current deformable mesh nodes."""

    positions_w: torch.Tensor
    rotations_w: torch.Tensor
    local_deformation: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None


class MagnetPoseProvider(Protocol):
    """Protocol implemented by PhysX/Newton deformable mesh adapters."""

    def sample(
        self,
        foot_positions_w: torch.Tensor,
        foot_quaternions_w: torch.Tensor,
        dt: float,
    ) -> DeformableMagnetPoseSample: ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...


class DeformableNodeMagnetPoseProvider:
    """Bind every magnet to three current deformable-mesh nodes.

    ``nodal_positions_getter`` must return ``[N,2,V,3]`` world positions from
    the current Isaac Lab ``DeformableObject.data.nodal_pos_w`` tensors.  The
    center, +X, and +Y node-index tables have shape ``[2,S,4]``.  The three
    nodes define each magnet's position and orientation without using removed
    legacy deformable APIs.  Mesh generation and attachment constraints remain
    scene responsibilities because they depend on the final TPU tetrahedral
    mesh and measured mounting geometry.
    """

    def __init__(
        self,
        nodal_positions_getter: Callable[[], torch.Tensor],
        center_node_indices: torch.Tensor,
        x_node_indices: torch.Tensor,
        y_node_indices: torch.Tensor,
    ) -> None:
        self.get_nodal_positions = nodal_positions_getter
        self.center_node_indices = center_node_indices.to(dtype=torch.long)
        self.x_node_indices = x_node_indices.to(dtype=torch.long)
        self.y_node_indices = y_node_indices.to(dtype=torch.long)
        if not (
            self.center_node_indices.shape
            == self.x_node_indices.shape
            == self.y_node_indices.shape
        ):
            raise ValueError("deformable magnet node-index tables must have identical [2,S,4] shapes")
        if self.center_node_indices.ndim != 3 or self.center_node_indices.shape[0] != 2:
            raise ValueError("deformable magnet node indices must have shape [2,S,4]")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def sample(
        self,
        foot_positions_w: torch.Tensor,
        foot_quaternions_w: torch.Tensor,
        dt: float,
    ) -> DeformableMagnetPoseSample:
        del foot_positions_w, foot_quaternions_w, dt
        nodes = self.get_nodal_positions()
        if nodes.ndim != 4 or nodes.shape[1] != 2 or nodes.shape[-1] != 3:
            raise ValueError(f"nodal position getter must return [N,2,V,3], got {tuple(nodes.shape)}")
        indices = self.center_node_indices.to(nodes.device)
        foot_index = torch.arange(2, device=nodes.device).view(1, 2, 1, 1)
        env_index = torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1, 1, 1)

        def gather(table: torch.Tensor) -> torch.Tensor:
            table = table.to(nodes.device).unsqueeze(0).expand(nodes.shape[0], -1, -1, -1)
            return nodes[env_index, foot_index, table]

        center = gather(indices)
        x_point = gather(self.x_node_indices)
        y_point = gather(self.y_node_indices)
        x_axis = _normalize(x_point - center)
        y_seed = y_point - center
        y_axis = _normalize(y_seed - torch.sum(y_seed * x_axis, dim=-1, keepdim=True) * x_axis)
        z_axis = _normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))
        y_axis = _normalize(torch.linalg.cross(z_axis, x_axis, dim=-1))
        rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
        valid = torch.isfinite(center).all(dim=-1).all(dim=-1)
        return DeformableMagnetPoseSample(
            positions_w=torch.nan_to_num(center),
            rotations_w=torch.nan_to_num(rotation),
            valid_mask=valid,
        )


class HallFootSensor:
    """Dual-foot Hall array with batched ``[env,left/right,site,XYZ]`` output."""

    def __init__(
        self,
        cfg: HallFootSensorCfg,
        field_model: MagneticFieldModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.field_model = field_model or DipoleMagneticFieldModel(cfg.dipole_min_distance)
        self.initialized = False
        self.magnet_pose_provider: MagnetPoseProvider | None = None
        self._debug_visualizer = None

    def initialize(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        magnet_pose_provider: MagnetPoseProvider | None = None,
        seed: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.cfg.implementation_mode == "deformable" and magnet_pose_provider is None:
            raise RuntimeError(
                "implementation_mode='deformable' requires a MagnetPoseProvider bound to current deformable mesh nodes"
            )
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.num_sensors = self.cfg.num_hall_sensors
        self.magnet_pose_provider = magnet_pose_provider
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 104729)
        self._build_static_geometry()

        field_shape = (self.num_envs, 2, self.num_sensors, 3)
        self.local_deformation = torch.zeros(field_shape[:-1] + (6,), device=self.device)
        # Simulation-only mechanical driver for Scheme A.  It is privileged
        # ground truth, not a quantity measured or reconstructed by Hall ICs.
        self.mechanical_driver_force_privileged = torch.zeros(field_shape, device=self.device)
        self.loading_history = torch.zeros(
            (*field_shape[:-1], self.cfg.loading_history_length, 6),
            device=self.device,
        )
        self.hall_positions_w = torch.zeros(field_shape, device=self.device)
        self.hall_rotations_w = torch.zeros(
            (*field_shape[:-1], 3, 3), device=self.device
        )
        self.magnet_positions_w = torch.zeros(
            (*field_shape[:-1], self.cfg.magnets_per_hall, 3), device=self.device
        )
        self.magnetic_moments_w = torch.zeros_like(self.magnet_positions_w)
        self.ideal_field = torch.zeros(field_shape, device=self.device)
        self.valid_mask = torch.zeros(field_shape[:-1], dtype=torch.bool, device=self.device)
        self.sample_age = torch.zeros(field_shape[:-2], device=self.device)
        self._sample_accumulator = 0.0
        self.signal = HallSensorSignalProcessor(
            self.cfg,
            self.num_envs,
            self.num_sensors,
            device=self.device,
            seed=seed,
        )
        self._initialize_domain_randomization_buffers()
        self.initialized = True
        self.reset()
        if self.cfg.enable_debug_vis:
            self._initialize_debug_visualizer()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._require_initialized()
        ids = self._env_ids(env_ids)
        self.local_deformation[ids] = 0.0
        self.mechanical_driver_force_privileged[ids] = 0.0
        self.loading_history[ids] = 0.0
        self.hall_positions_w[ids] = 0.0
        self.hall_rotations_w[ids] = 0.0
        self.magnet_positions_w[ids] = 0.0
        self.magnetic_moments_w[ids] = 0.0
        self.ideal_field[ids] = 0.0
        self.valid_mask[ids] = False
        self.sample_age[ids] = 0.0
        self.signal.reset(ids)
        self._policy_history[ids] = 0.0
        self._policy_observation[ids] = 0.0
        self._sample_domain_randomization(ids)
        if env_ids is None:
            self._sample_accumulator = 0.0
        if self.magnet_pose_provider is not None:
            self.magnet_pose_provider.reset(ids)

    def update(
        self,
        dt: float,
        *,
        foot_positions_w: torch.Tensor,
        foot_quaternions_w: torch.Tensor,
        contact_force_w: torch.Tensor | None = None,
        contact_point_w: torch.Tensor | None = None,
        local_contact_force_f: torch.Tensor | None = None,
        local_deformation: torch.Tensor | None = None,
        temperature_c: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Advance mechanics and, when due, sample the Hall electronics.

        ``local_deformation`` is an optional ``[...,6]`` override ordered as
        ``dx,dy,dz,roll,pitch,yaw``; ``dz>0`` moves magnets toward the Hall IC.
        It is used by the platen validation scene and calibrated surrogates.

        ``contact_force_w`` is simulator-only input to the Scheme-A mechanical
        surrogate.  It is never inferred from magnetic data and is never part
        of the Hall sensor output.

        ``local_contact_force_f`` is the detailed-contact alternative, in
        newtons with shape ``[N,2,S,3]`` and canonical left/right foot-local
        axes.  Passing it bypasses the legacy total-force/mean-point Gaussian;
        it remains privileged mechanics and never enters the actor directly.
        """
        self._require_initialized()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_foot_pose(foot_positions_w, foot_quaternions_w)
        if not self.cfg.enabled:
            self.signal.raw.zero_()
            self.signal.processed.zero_()
            self.valid_mask.zero_()
            return self.get_filtered_data()
        foot_positions_w = foot_positions_w.to(device=self.device, dtype=self.dtype)
        foot_quaternions_w = foot_quaternions_w.to(device=self.device, dtype=self.dtype)
        foot_rotation_w = quaternion_to_matrix(foot_quaternions_w)

        if local_deformation is not None:
            if local_deformation.shape != self.local_deformation.shape:
                raise ValueError(
                    f"local_deformation must be {tuple(self.local_deformation.shape)}, got {tuple(local_deformation.shape)}"
                )
            self.local_deformation.copy_(local_deformation.to(self.device, self.dtype))
            self.mechanical_driver_force_privileged.zero_()
        elif self.cfg.implementation_mode == "approximate":
            self._update_approximate_deformation(
                dt,
                foot_positions_w,
                foot_rotation_w,
                contact_force_w,
                contact_point_w,
                local_contact_force_f,
            )

        self.loading_history.copy_(torch.roll(self.loading_history, shifts=1, dims=-2))
        self.loading_history[..., 0, :].copy_(self.local_deformation)
        self._sample_accumulator += dt
        self.sample_age.add_(dt)
        if self._sample_accumulator + 1.0e-12 < self.cfg.sensor_period:
            return self.get_filtered_data()
        sample_dt = self._sample_accumulator
        self._sample_accumulator = math.fmod(self._sample_accumulator, self.cfg.sensor_period)

        hall_rotation_w = self._hall_world_geometry(foot_positions_w, foot_rotation_w)
        provider_valid: torch.Tensor | None = None
        if self.cfg.implementation_mode == "deformable":
            assert self.magnet_pose_provider is not None
            sample = self.magnet_pose_provider.sample(foot_positions_w, foot_quaternions_w, sample_dt)
            self._validate_deformable_sample(sample)
            self.magnet_positions_w.copy_(sample.positions_w.to(self.device, self.dtype))
            rotations_w = sample.rotations_w.to(self.device, self.dtype)
            self._set_moments_from_rotations(rotations_w)
            if sample.local_deformation is not None:
                self.local_deformation.copy_(sample.local_deformation.to(self.device, self.dtype))
            provider_valid = sample.valid_mask
        else:
            self._approximate_magnet_world_geometry(foot_positions_w, foot_rotation_w)

        self.ideal_field.copy_(
            self.field_model.compute(
                self.hall_positions_w,
                self.magnet_positions_w,
                self.magnetic_moments_w,
                sensor_rotation_w=hall_rotation_w,
                local_deformation=self.local_deformation,
                loading_history=self.loading_history,
                dt=sample_dt,
            )
        )
        # A right PCB is mirrored in hardware.  This is an explicit channel-axis
        # remap after the proper 3-D rotation, not a left-handed pose matrix.
        right_sign = torch.as_tensor(self.cfg.right_hall_axis_sign, device=self.device)
        self.ideal_field[:, 1].mul_(right_sign)
        finite = torch.isfinite(self.ideal_field).all(dim=-1)
        if provider_valid is not None:
            provider_valid = provider_valid.to(device=self.device, dtype=torch.bool)
            if provider_valid.ndim == 4:
                provider_valid = provider_valid.all(dim=-1)
            finite.logical_and_(provider_valid)
        self.ideal_field.copy_(torch.nan_to_num(self.ideal_field))
        output = self.signal.update(self.ideal_field, sample_dt, temperature_c=temperature_c)
        self.valid_mask.copy_(finite & output.baseline_ready.squeeze(-1))
        self._update_policy_observation()
        self.sample_age.zero_()
        if self._debug_visualizer is not None:
            self._debug_visualizer.update(self.get_debug_data())
        return self.get_filtered_data()

    def distribute_detailed_contact_forces(
        self,
        *,
        foot_positions_w: torch.Tensor,
        foot_quaternions_w: torch.Tensor,
        point_forces_w: torch.Tensor,
        contact_points_w: torch.Tensor,
        contact_env_indices: torch.Tensor,
        contact_foot_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Map detailed PhysX point forces to ``[N,2,S,3]`` local forces.

        This method only performs the Scheme-A mechanical spatial assignment.
        The returned tensor is intended for :meth:`update` via
        ``local_contact_force_f``.  Inputs and outputs remain SI.
        """

        self._require_initialized()
        foot_positions_w = foot_positions_w.to(device=self.device, dtype=self.dtype)
        foot_quaternions_w = foot_quaternions_w.to(device=self.device, dtype=self.dtype)
        self._validate_foot_pose(foot_positions_w, foot_quaternions_w)
        return distribute_point_forces_to_hall_sites(
            num_envs=self.num_envs,
            hall_positions_f=self.hall_positions_f,
            foot_positions_w=foot_positions_w,
            foot_rotations_w=quaternion_to_matrix(foot_quaternions_w),
            point_forces_w=point_forces_w.to(device=self.device, dtype=self.dtype),
            contact_points_w=contact_points_w.to(device=self.device, dtype=self.dtype),
            contact_env_indices=contact_env_indices.to(device=self.device, dtype=torch.long),
            contact_foot_indices=contact_foot_indices.to(device=self.device, dtype=torch.long),
            spread_sigma_f=self.cfg.contact_spread_sigma * self._contact_spread_scale,
        )

    def get_raw_data(self) -> torch.Tensor:
        """Digitized absolute Hall values in tesla, shape ``[N,2,S,3]``."""
        self._require_initialized()
        return self.signal.raw

    def get_filtered_data(self) -> torch.Tensor:
        """Filtered, auto-zeroed Hall values in tesla, shape ``[N,2,S,3]``."""
        self._require_initialized()
        return self.signal.processed

    def get_policy_observation(self) -> torch.Tensor:
        """Dimensionless randomized Hall response used by the deployable actor.

        The transform represents installation/calibration uncertainty in the
        real count domain after per-channel normalization.  It does not expose
        or estimate contact force.
        """
        self._require_initialized()
        return self._policy_observation

    def get_policy_valid_mask(self) -> torch.Tensor:
        """Per-site validity after whole-foot packet dropout injection."""
        self._require_initialized()
        foot_valid = self._policy_foot_keep.squeeze(-1).squeeze(-1).to(torch.bool)
        return self.valid_mask & foot_valid.unsqueeze(-1)

    def get_reported_sample_period(self) -> torch.Tensor:
        """Per-environment/per-foot source period metadata in seconds."""
        self._require_initialized()
        return self._reported_sample_period

    def get_debug_data(self) -> dict[str, torch.Tensor]:
        """Return views of the current physics/electronics state.

        Only the magnetic/baseline/validity entries are Hall observables.
        ``mechanical_driver_force_privileged`` is simulator ground truth used
        to drive Scheme A and must not be treated as a Hall-derived force.
        """
        self._require_initialized()
        processed = self.get_filtered_data()
        return {
            "ideal_magnetic_field": self.ideal_field,
            "raw_magnetic_field": self.get_raw_data(),
            "filtered_magnetic_field": processed,
            "zero_load_baseline": self.signal.baseline,
            "magnetic_delta": processed,
            "magnetic_norm": torch.linalg.vector_norm(processed, dim=-1),
            "local_deformation": self.local_deformation,
            "mechanical_driver_force_privileged": self.mechanical_driver_force_privileged,
            "valid_mask": self.valid_mask,
            "sample_age": self.sample_age,
            "policy_observation": self._policy_observation,
            "policy_valid_mask": self.get_policy_valid_mask(),
            "reported_sample_period": self._reported_sample_period,
            "mechanical_normal_scale": self._normal_stiffness_scale,
            "mechanical_shear_scale": self._shear_stiffness_scale,
            "mechanical_damping_scale": self._damping_scale,
            "policy_channel_keep": self._policy_channel_keep,
            "policy_foot_keep": self._policy_foot_keep,
            "policy_delay_steps": self._policy_delay_steps,
            "hall_positions_w": self.hall_positions_w,
            "hall_rotations_w": self.hall_rotations_w,
            "magnet_positions_w": self.magnet_positions_w,
            "magnetic_moments_w": self.magnetic_moments_w,
        }

    def _initialize_domain_randomization_buffers(self) -> None:
        scalar_shape = (self.num_envs, 2, self.num_sensors)
        self._normal_stiffness_scale = torch.ones(scalar_shape, device=self.device)
        self._shear_stiffness_scale = torch.ones(scalar_shape, device=self.device)
        self._damping_scale = torch.ones(scalar_shape, device=self.device)
        self._contact_spread_scale = torch.ones((self.num_envs, 2, 1), device=self.device)
        self._magnetic_moment_scale = torch.ones(
            (self.num_envs, 2, self.num_sensors, self.cfg.magnets_per_hall, 1),
            device=self.device,
        )
        self._magnet_position_jitter = torch.zeros(
            (self.num_envs, 2, self.num_sensors, self.cfg.magnets_per_hall, 3),
            device=self.device,
        )
        self._policy_gain = torch.ones((*scalar_shape, 3), device=self.device)
        identity = torch.eye(3, device=self.device, dtype=self.dtype)
        self._policy_cross_axis = identity.view(1, 1, 1, 3, 3).expand(
            self.num_envs, 2, self.num_sensors, -1, -1
        ).clone()
        self._policy_zero_residual = torch.zeros((*scalar_shape, 3), device=self.device)
        self._policy_channel_keep = torch.ones((*scalar_shape, 1), device=self.device)
        self._policy_foot_keep = torch.ones((self.num_envs, 2, 1, 1), device=self.device)
        self._policy_delay_steps = torch.zeros(
            (self.num_envs, 2), dtype=torch.long, device=self.device
        )
        self._reported_sample_period = torch.full(
            (self.num_envs, 2), self.cfg.sensor_period, device=self.device
        )
        delay_depth = self.cfg.maximum_packet_delay_steps + 1
        self._policy_history = torch.zeros(
            (self.num_envs, 2, delay_depth, self.num_sensors, 3), device=self.device
        )
        self._policy_observation = torch.zeros(
            (self.num_envs, 2, self.num_sensors, 3), device=self.device
        )

    def _uniform(
        self,
        shape: tuple[int, ...],
        value_range: tuple[float, float],
    ) -> torch.Tensor:
        lower, upper = value_range
        return lower + (upper - lower) * torch.rand(
            shape, device=self.device, generator=self.generator
        )

    def _sample_domain_randomization(self, ids: torch.Tensor) -> None:
        count = int(ids.numel())
        if count == 0:
            return
        if not self.cfg.enable_domain_randomization:
            self._normal_stiffness_scale[ids] = 1.0
            self._shear_stiffness_scale[ids] = 1.0
            self._damping_scale[ids] = 1.0
            self._contact_spread_scale[ids] = 1.0
            self._magnetic_moment_scale[ids] = 1.0
            self._magnet_position_jitter[ids] = 0.0
            self._policy_gain[ids] = 1.0
            identity = torch.eye(3, device=self.device, dtype=self.dtype)
            self._policy_cross_axis[ids] = identity
            self._policy_zero_residual[ids] = 0.0
            self._policy_channel_keep[ids] = 1.0
            self._policy_foot_keep[ids] = 1.0
            self._policy_delay_steps[ids] = 0
            self._reported_sample_period[ids] = self.cfg.sensor_period
            return

        scalar_shape = (count, 2, self.num_sensors)
        self._normal_stiffness_scale[ids] = self._uniform(
            scalar_shape, self.cfg.normal_stiffness_scale_range
        )
        self._shear_stiffness_scale[ids] = self._uniform(
            scalar_shape, self.cfg.shear_stiffness_scale_range
        )
        self._damping_scale[ids] = self._uniform(
            scalar_shape, self.cfg.damping_scale_range
        )
        self._contact_spread_scale[ids] = self._uniform(
            (count, 2, 1), self.cfg.contact_spread_scale_range
        )
        self._magnetic_moment_scale[ids] = self._uniform(
            (count, 2, self.num_sensors, self.cfg.magnets_per_hall, 1),
            self.cfg.magnetic_moment_scale_range,
        )
        if self.cfg.magnet_position_jitter_std > 0.0:
            jitter = torch.randn(
                (count, 2, self.num_sensors, self.cfg.magnets_per_hall, 3),
                device=self.device,
                generator=self.generator,
            ) * self.cfg.magnet_position_jitter_std
            self._magnet_position_jitter[ids] = jitter.clamp(
                -3.0 * self.cfg.magnet_position_jitter_std,
                3.0 * self.cfg.magnet_position_jitter_std,
            )
        else:
            self._magnet_position_jitter[ids] = 0.0

        sensor_gain = self._uniform(
            (count, 2, self.num_sensors, 1), self.cfg.observation_sensor_gain_range
        )
        axis_gain = self._uniform(
            (count, 2, 1, 3), self.cfg.observation_axis_gain_range
        )
        self._policy_gain[ids] = sensor_gain * axis_gain
        identity = torch.eye(3, device=self.device, dtype=self.dtype).view(1, 1, 1, 3, 3)
        coupling = torch.randn(
            (count, 2, self.num_sensors, 3, 3),
            device=self.device,
            generator=self.generator,
        ) * self.cfg.observation_cross_axis_std
        # Keep diagonal gain in the explicit gain tensor; cross-axis terms are
        # off-diagonal only so the uncertainty remains interpretable.
        coupling = coupling * (1.0 - identity)
        self._policy_cross_axis[ids] = identity + coupling
        self._policy_zero_residual[ids] = torch.randn(
            (count, 2, self.num_sensors, 3),
            device=self.device,
            generator=self.generator,
        ) * self.cfg.observation_zero_residual_std
        self._policy_channel_keep[ids] = (
            torch.rand(
                (count, 2, self.num_sensors, 1),
                device=self.device,
                generator=self.generator,
            )
            >= self.cfg.dead_channel_probability
        ).to(self.dtype)
        self._policy_foot_keep[ids] = (
            torch.rand((count, 2, 1, 1), device=self.device, generator=self.generator)
            >= self.cfg.foot_dropout_probability
        ).to(self.dtype)
        self._reported_sample_period[ids] = self._uniform(
            (count, 2), self.cfg.reported_sample_period_range
        )
        self._policy_delay_steps[ids] = torch.randint(
            0,
            self.cfg.maximum_packet_delay_steps + 1,
            (count, 2),
            device=self.device,
            generator=self.generator,
        )

    def _update_policy_observation(self) -> None:
        value = self.signal.processed / self.cfg.observation_scale_t
        value = torch.einsum("...ij,...j->...i", self._policy_cross_axis, value)
        value = value * self._policy_gain + self._policy_zero_residual
        value = torch.clamp(
            value,
            self.cfg.observation_clip[0],
            self.cfg.observation_clip[1],
        )
        value = value * self._policy_channel_keep
        if self._policy_history.shape[2] > 1:
            self._policy_history[:, :, 1:].copy_(self._policy_history[:, :, :-1].clone())
        self._policy_history[:, :, 0].copy_(value)
        env_index = torch.arange(self.num_envs, device=self.device).view(-1, 1).expand(-1, 2)
        foot_index = torch.arange(2, device=self.device).view(1, -1).expand(self.num_envs, -1)
        delayed = self._policy_history[env_index, foot_index, self._policy_delay_steps]
        self._policy_observation.copy_(delayed * self._policy_foot_keep)

    def _build_static_geometry(self) -> None:
        normalized = torch.tensor(
            self.cfg.hall_positions_normalized,
            device=self.device,
            dtype=self.dtype,
        )
        xy = normalized * torch.tensor(
            (self.cfg.sole_length, self.cfg.sole_width),
            device=self.device,
            dtype=self.dtype,
        )
        xy = xy.unsqueeze(0).expand(2, -1, -1).clone()
        if self.cfg.mirror_right_y:
            xy[1, :, 1].neg_()
        yaw = torch.deg2rad(
            torch.tensor(
                (self.cfg.sole_yaw_deg, self.cfg.right_sole_yaw_deg),
                device=self.device,
                dtype=self.dtype,
            )
        )
        sole_rotation = _rotation_z(yaw)
        xy3 = torch.zeros((2, self.num_sensors, 3), device=self.device)
        xy3[..., :2] = xy
        xy3 = torch.einsum("fij,fsj->fsi", sole_rotation, xy3)
        origin = torch.tensor(self.cfg.sole_origin, device=self.device, dtype=self.dtype)
        xy3.add_(origin)
        xy3[..., 2] = origin[2] + self.cfg.hall_height
        self.hall_positions_f = xy3

        chip_yaw = self.cfg.hall_axis_yaw_deg
        if len(chip_yaw) == 1:
            chip_yaw = chip_yaw * self.num_sensors
        chip_rotation = _rotation_z(
            torch.deg2rad(torch.tensor(chip_yaw, device=self.device, dtype=self.dtype))
        )
        self.hall_rotation_f = torch.einsum(
            "fij,sjk->fsik", sole_rotation, chip_rotation
        )

        sx, sy = 0.5 * self.cfg.magnet_spacing_x, 0.5 * self.cfg.magnet_spacing_y
        self.magnet_offsets = torch.tensor(
            (
                (-sx, -sy, -self.cfg.initial_hall_magnet_distance),
                (-sx, sy, -self.cfg.initial_hall_magnet_distance),
                (sx, -sy, -self.cfg.initial_hall_magnet_distance),
                (sx, sy, -self.cfg.initial_hall_magnet_distance),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        direction = torch.tensor(self.cfg.magnetization_direction, device=self.device, dtype=self.dtype)
        self.base_magnetic_moment = _normalize(direction) * self.cfg.magnetic_moment

    def _update_approximate_deformation(
        self,
        dt: float,
        foot_positions_w: torch.Tensor,
        foot_rotation_w: torch.Tensor,
        contact_force_w: torch.Tensor | None,
        contact_point_w: torch.Tensor | None,
        local_contact_force_f: torch.Tensor | None,
    ) -> None:
        if local_contact_force_f is not None:
            if contact_force_w is not None or contact_point_w is not None:
                raise ValueError(
                    "local_contact_force_f is mutually exclusive with aggregate contact force/point inputs"
                )
            expected = (self.num_envs, 2, self.num_sensors, 3)
            if local_contact_force_f.shape != expected:
                raise ValueError(
                    f"local_contact_force_f must have shape {expected}, got "
                    f"{tuple(local_contact_force_f.shape)}"
                )
            local_force = local_contact_force_f.to(self.device, self.dtype)
            if not torch.isfinite(local_force).all():
                raise ValueError("local_contact_force_f must be finite")
            # A raw detailed stream contains entries only for actual contacts;
            # preserve every point contribution.  Re-applying the aggregate
            # contact threshold here would violate the detailed path's total
            # force conservation and suppress weak but spatially informative
            # onset/exit patches.
        else:
            if contact_force_w is None:
                force_w = torch.zeros((self.num_envs, 2, 3), device=self.device)
            else:
                if contact_force_w.shape != (self.num_envs, 2, 3):
                    raise ValueError("contact_force_w must have shape [num_envs,2,3]")
                force_w = torch.nan_to_num(contact_force_w.to(self.device, self.dtype))
            force_f = torch.einsum("...ji,...j->...i", foot_rotation_w, force_w)

            default_point_f = torch.tensor(
                self.cfg.sole_origin,
                device=self.device,
                dtype=self.dtype,
            ).view(1, 1, 3).expand(self.num_envs, 2, 3)
            if contact_point_w is None:
                point_f = default_point_f
                has_point = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
            else:
                if contact_point_w.shape != (self.num_envs, 2, 3):
                    raise ValueError("contact_point_w must have shape [num_envs,2,3]")
                point_w = contact_point_w.to(self.device, self.dtype)
                has_point = torch.isfinite(point_w).all(dim=-1)
                relative = torch.nan_to_num(point_w - foot_positions_w)
                measured_point_f = torch.einsum("...ji,...j->...i", foot_rotation_w, relative)
                point_f = torch.where(has_point.unsqueeze(-1), measured_point_f, default_point_f)

            delta_xy = self.hall_positions_f.unsqueeze(0)[..., :2] - point_f.unsqueeze(-2)[..., :2]
            sigma = self.cfg.contact_spread_sigma * self._contact_spread_scale
            weights = torch.exp(
                -0.5 * torch.sum(delta_xy * delta_xy, dim=-1) / torch.square(sigma)
            )
            # Without tracked contact points, retain a smooth whole-sole load instead
            # of creating a fictitious center spike.
            uniform = torch.ones_like(weights)
            weights = torch.where(has_point.unsqueeze(-1), weights, uniform)
            active = torch.linalg.vector_norm(force_f, dim=-1) > self.cfg.contact_force_threshold
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            weights = weights * active.unsqueeze(-1)
            local_force = weights.unsqueeze(-1) * force_f.unsqueeze(-2)
        self.mechanical_driver_force_privileged.copy_(local_force)

        target = torch.zeros_like(self.local_deformation)
        shear_stiffness = self.cfg.local_shear_stiffness * self._shear_stiffness_scale
        normal_stiffness = self.cfg.local_normal_stiffness * self._normal_stiffness_scale
        target[..., :2] = local_force[..., :2] / shear_stiffness.unsqueeze(-1)
        target[..., 2] = torch.clamp_min(local_force[..., 2], 0.0) / normal_stiffness
        target[..., :2].clamp_(-self.cfg.max_shear_displacement, self.cfg.max_shear_displacement)
        target[..., 2].clamp_(0.0, self.cfg.max_normal_compression)
        target[..., 3] = torch.clamp(
            -target[..., 1] / self.cfg.tpu_thickness,
            -self.cfg.max_local_rotation,
            self.cfg.max_local_rotation,
        )
        target[..., 4] = torch.clamp(
            target[..., 0] / self.cfg.tpu_thickness,
            -self.cfg.max_local_rotation,
            self.cfg.max_local_rotation,
        )

        damping = self.cfg.local_damping * self._damping_scale
        normal_tau = damping / normal_stiffness
        shear_tau = damping / shear_stiffness
        alpha_normal = 1.0 - torch.exp(-dt / normal_tau)
        alpha_shear = 1.0 - torch.exp(-dt / shear_tau)
        self.local_deformation[..., :2].lerp_(target[..., :2], alpha_shear.unsqueeze(-1))
        self.local_deformation[..., 2:3].lerp_(target[..., 2:3], alpha_normal.unsqueeze(-1))
        self.local_deformation[..., 3:6].lerp_(target[..., 3:6], alpha_shear.unsqueeze(-1))

    def _hall_world_geometry(
        self,
        foot_positions_w: torch.Tensor,
        foot_rotation_w: torch.Tensor,
    ) -> torch.Tensor:
        hall_f = self.hall_positions_f.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        self.hall_positions_w.copy_(
            foot_positions_w.unsqueeze(-2) + torch.einsum("...ij,...sj->...si", foot_rotation_w, hall_f)
        )
        self.hall_rotations_w.copy_(
            torch.einsum(
                "...ij,...sjk->...sik",
                foot_rotation_w,
                self.hall_rotation_f.unsqueeze(0).expand(self.num_envs, -1, -1, -1, -1),
            )
        )
        return self.hall_rotations_w

    def _approximate_magnet_world_geometry(
        self,
        foot_positions_w: torch.Tensor,
        foot_rotation_w: torch.Tensor,
    ) -> None:
        deformation_rotation = _euler_xyz_matrix(self.local_deformation[..., 3:6])
        randomized_offset = self.magnet_offsets.view(1, 1, 1, -1, 3) + self._magnet_position_jitter
        offset = torch.einsum("...ij,...mj->...mi", deformation_rotation, randomized_offset)
        translation = self.local_deformation[..., :3].unsqueeze(-2)
        magnet_f = self.hall_positions_f.unsqueeze(0).unsqueeze(-2) + translation + offset
        self.magnet_positions_w.copy_(
            foot_positions_w.unsqueeze(-2).unsqueeze(-2)
            + torch.einsum("...ij,...smj->...smi", foot_rotation_w, magnet_f)
        )
        moment_f = torch.einsum("...ij,j->...i", deformation_rotation, self.base_magnetic_moment)
        moment_f = moment_f.unsqueeze(-2).expand(-1, -1, -1, self.cfg.magnets_per_hall, -1)
        moment_f = moment_f * self._magnetic_moment_scale
        self.magnetic_moments_w.copy_(
            torch.einsum("...ij,...smj->...smi", foot_rotation_w, moment_f)
        )

    def _set_moments_from_rotations(self, magnet_rotations_w: torch.Tensor) -> None:
        if magnet_rotations_w.shape != (*self.magnet_positions_w.shape[:-1], 3, 3):
            raise ValueError("deformable magnet rotations must be [N,2,S,4,3,3]")
        self.magnetic_moments_w.copy_(
            torch.einsum("...ij,j->...i", magnet_rotations_w, self.base_magnetic_moment)
            * self._magnetic_moment_scale
        )

    def _validate_deformable_sample(self, sample: DeformableMagnetPoseSample) -> None:
        if sample.positions_w.shape != self.magnet_positions_w.shape:
            raise ValueError(
                f"deformable magnet positions {tuple(sample.positions_w.shape)} != {tuple(self.magnet_positions_w.shape)}"
            )
        expected_rotation = (*self.magnet_positions_w.shape[:-1], 3, 3)
        if sample.rotations_w.shape != expected_rotation:
            raise ValueError(
                f"deformable magnet rotations {tuple(sample.rotations_w.shape)} != {expected_rotation}"
            )
        if sample.local_deformation is not None and sample.local_deformation.shape != self.local_deformation.shape:
            raise ValueError("deformable local_deformation must be [N,2,S,6]")

    def _validate_foot_pose(self, position: torch.Tensor, quaternion: torch.Tensor) -> None:
        if position.shape != (self.num_envs, 2, 3):
            raise ValueError(f"foot_positions_w must be [{self.num_envs},2,3]")
        if quaternion.shape != (self.num_envs, 2, 4):
            raise ValueError(f"foot_quaternions_w must be [{self.num_envs},2,4] scalar-first")

    def _initialize_debug_visualizer(self) -> None:
        try:
            from .hall_sensor_visualizer import HallSensorVisualizer

            self._debug_visualizer = HallSensorVisualizer(self.cfg)
        except (ImportError, RuntimeError) as error:
            import warnings

            warnings.warn(f"Hall debug visualization disabled: {error}", stacklevel=2)
            self._debug_visualizer = None

    def _env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("HallFootSensor.initialize(...) must be called first")
