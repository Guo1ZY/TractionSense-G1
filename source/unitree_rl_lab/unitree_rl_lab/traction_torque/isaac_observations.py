"""Isaac Lab adapter using native state for analytical force estimation.

ContactSensor values are intentionally absent from this module. Ground-truth
contact is supplied by separate Teacher/evaluation code only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import NamedTuple

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

from .analytical_force_estimator import AnalyticalDualFootForceEstimator, AnalyticalForceEstimatorInput
from .contact_estimator import HybridContactEstimator, HybridContactInput
from .dynamics import inverse_dynamics_torque_residual
from .randomization import TorqueDynamicsObservationModel, TorqueDynamicsRandomizationCfg
from .schema import LEFT_LEG_ACTION_INDICES, RIGHT_LEG_ACTION_INDICES, TORQUE_TRACTION_FRAME_SCHEMA, TORQUE_TRACTION_JOINT_INDICES
from .torque_filter import JointStateFilter
from .traction_estimator import TractionEstimatorInput, TractionStateEstimator


class IsaacTorqueTractionPacket(NamedTuple):
    frame: torch.Tensor
    analytical_force_local_n: torch.Tensor
    contact_probability: torch.Tensor
    contact_state: torch.Tensor
    force_confidence: torch.Tensor
    residual_norm_nm: torch.Tensor
    condition_score: torch.Tensor
    traction_utilization: torch.Tensor
    slip_probability: torch.Tensor
    traction_margin: torch.Tensor
    friction_lower_bound: torch.Tensor
    slip_event_mu_estimate: torch.Tensor
    slip_state: torch.Tensor
    slip_duration_s: torch.Tensor
    foot_planar_velocity_m_s: torch.Tensor
    imu_linear_acceleration_m_s2: torch.Tensor
    tau_est_nm: torch.Tensor
    tau_est_minus_model_nm: torch.Tensor


@dataclass(frozen=True)
class IsaacTorqueEstimatorCfg:
    asset_name: str = "robot"
    left_foot_body_name: str = "left_ankle_roll_link"
    right_foot_body_name: str = "right_ankle_roll_link"
    gravity_m_s2: float = 9.81
    force_clip_normalized: float = 3.0
    randomization_stage: int = 0
    seed: int = 20260803


class _IsaacTorqueEstimatorState:
    def __init__(self, env, cfg: IsaacTorqueEstimatorCfg) -> None:
        self.env, self.cfg = env, cfg
        self.robot: Articulation = env.scene[cfg.asset_name]
        if self.robot.joint_names != list(__import__("unitree_rl_lab.traction.schema", fromlist=["G1_29DOF_JOINT_ORDER"]).G1_29DOF_JOINT_ORDER):
            raise RuntimeError("Isaac joint/action order changed from canonical G1-29DoF schema")
        self.foot_body_ids = tuple(self.robot.body_names.index(name) for name in (cfg.left_foot_body_name, cfg.right_foot_body_name))
        self.leg_indices = (LEFT_LEG_ACTION_INDICES, RIGHT_LEG_ACTION_INDICES)
        n, device = env.num_envs, env.device
        self.filter = JointStateFilter(n, 29, device=device)
        self.force_estimator = AnalyticalDualFootForceEstimator(n, device=device)
        self.contact_estimator = HybridContactEstimator(n, device=device)
        self.traction_estimator = TractionStateEstimator(n, device=device)
        self.randomization = TorqueDynamicsObservationModel(
            n, 29, 35,
            cfg=TorqueDynamicsRandomizationCfg(curriculum_stage=cfg.randomization_stage),
            device=device, seed=cfg.seed, leg_joint_indices=self.leg_indices,
        )
        self.previous_foot_velocity = torch.zeros(n, 2, 2, device=device)
        self.previous_force = torch.zeros(n, 2, 3, device=device)
        self.last_step = -1
        self.packet: IsaacTorqueTractionPacket | None = None
        self.last_compute_latency_ms = float("nan")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = None if env_ids is None else torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        self.filter.reset(ids)
        self.force_estimator.reset(ids)
        self.contact_estimator.reset(ids)
        self.traction_estimator.reset(ids)
        self.randomization.reset(ids)
        if ids is None:
            self.previous_foot_velocity.zero_(); self.previous_force.zero_()
        else:
            self.previous_foot_velocity[ids] = 0.0; self.previous_force[ids] = 0.0
        self.last_step = -1

    def _step_index(self) -> int:
        value = getattr(self.env, "common_step_counter", 0)
        return int(value.item() if isinstance(value, torch.Tensor) else value)

    def compute(self) -> IsaacTorqueTractionPacket:
        step = self._step_index()
        if self.packet is not None and step == self.last_step:
            return self.packet
        benchmark = bool(getattr(self.env, "torque_estimator_benchmark_synchronize", False))
        if benchmark and torch.cuda.is_available() and str(self.env.device).startswith("cuda"):
            torch.cuda.synchronize(self.env.device)
        start_time = time.perf_counter()
        robot, n = self.robot, self.env.num_envs
        q, dq, tau_raw = robot.data.joint_pos, robot.data.joint_vel, robot.data.applied_torque
        qdd_filtered, tau_filtered = self.filter.update(dq, tau_raw)
        root_linear_acc_w = robot.data.body_com_lin_acc_w[:, 0]
        root_angular_acc_w = robot.data.body_com_ang_acc_w[:, 0]
        generalized_acceleration = torch.cat((root_linear_acc_w, root_angular_acc_w, qdd_filtered), dim=-1)
        view = robot.root_physx_view
        mass_matrix = view.get_generalized_mass_matrices()
        bias_force = view.get_coriolis_and_centrifugal_compensation_forces() + view.get_gravity_compensation_forces()
        jacobian_w = view.get_jacobians()[:, list(self.foot_body_ids), :3, :]
        foot_rotation_w = matrix_from_quat(robot.data.body_quat_w[:, list(self.foot_body_ids)])
        jacobian_local_full = torch.matmul(foot_rotation_w.transpose(-1, -2), jacobian_w)
        leg_jacobian = torch.stack((
            jacobian_local_full[:, 0, :, [6 + index for index in self.leg_indices[0]]],
            jacobian_local_full[:, 1, :, [6 + index for index in self.leg_indices[1]]],
        ), dim=1)
        gravity_w = torch.tensor((0.0, 0.0, -self.cfg.gravity_m_s2), device=self.env.device).expand(n, 3)
        imu_specific_force_b = quat_apply_inverse(robot.data.root_quat_w, root_linear_acc_w - gravity_w)
        randomized = self.randomization.update(
            joint_position=q, joint_velocity=dq, joint_acceleration=qdd_filtered,
            tau_est=tau_filtered, imu_linear_acceleration=imu_specific_force_b,
            mass_matrix=mass_matrix, bias_force=bias_force, leg_jacobian=leg_jacobian,
        )
        randomized_qdd = torch.cat((root_linear_acc_w, root_angular_acc_w, randomized.joint_acceleration), dim=-1)
        dynamics = inverse_dynamics_torque_residual(randomized.mass_matrix, randomized.bias_force, randomized_qdd, randomized.tau_est)
        contact_leg_force = torch.stack((
            dynamics.contact_generalized_joint_force[:, list(self.leg_indices[0])],
            dynamics.contact_generalized_joint_force[:, list(self.leg_indices[1])],
        ), dim=1)
        foot_pos = robot.data.body_pos_w[:, list(self.foot_body_ids)] - self.env.scene.env_origins[:, None, :]
        foot_vel = robot.data.body_lin_vel_w[:, list(self.foot_body_ids)]
        prior_contact = self.contact_estimator.probability
        kinematic_contact = ((foot_pos[..., 2] < 0.10) & (foot_vel[..., 2].abs() < 0.5)).float()
        # Kinematics bootstraps the force solve; force then reinforces the
        # probabilistic classifier.  This avoids a zero-force/zero-contact
        # deadlock without injecting simulator contact truth.
        force_contact = torch.maximum(prior_contact, kinematic_contact)
        force_result = self.force_estimator.update(AnalyticalForceEstimatorInput(randomized.leg_jacobian, contact_leg_force, force_contact))
        force_n = force_result.force_local_n.reshape(n, 2, 3)
        leg_tau = torch.stack((randomized.tau_est[:, list(self.leg_indices[0])], randomized.tau_est[:, list(self.leg_indices[1])]), dim=1)
        contact = self.contact_estimator.update(HybridContactInput(
            foot_pos[..., 2], foot_vel[..., 2], foot_vel[..., :2], leg_tau,
            force_n[..., 2], torch.stack((randomized.joint_position[:, list(self.leg_indices[0])], randomized.joint_position[:, list(self.leg_indices[1])]), dim=1),
            randomized.imu_linear_acceleration,
        ))
        foot_acceleration = (foot_vel[..., :2] - self.previous_foot_velocity) / self.env.step_dt
        force_growth = torch.linalg.vector_norm((force_n - self.previous_force) / self.env.step_dt, dim=-1)
        traction = self.traction_estimator.update(TractionEstimatorInput(
            force_result.force_local_n, contact.probability, foot_vel[..., :2], foot_acceleration,
            force_growth, force_result.residual_norm_nm, randomized.imu_linear_acceleration, force_result.confidence,
        ))
        self.previous_foot_velocity.copy_(foot_vel[..., :2]); self.previous_force.copy_(force_n)
        mass = view.get_masses().sum(dim=1, keepdim=True).to(
            device=force_result.force_local_n.device,
            dtype=force_result.force_local_n.dtype,
        )
        normalized_force = (force_result.force_local_n / (mass * self.cfg.gravity_m_s2)).clamp(-self.cfg.force_clip_normalized, self.cfg.force_clip_normalized)
        effort = robot.data.joint_effort_limits[:, list(TORQUE_TRACTION_JOINT_INDICES)].clamp_min(1.0)
        frame = torch.cat((
            robot.data.root_ang_vel_b * 0.2,
            robot.data.projected_gravity_b,
            self.env.command_manager.get_command("base_velocity"),
            randomized.joint_position - robot.data.default_joint_pos,
            randomized.joint_velocity * 0.05,
            self.env.action_manager.action,
            randomized.tau_est[:, list(TORQUE_TRACTION_JOINT_INDICES)] / effort,
            normalized_force,
            contact.probability,
            force_result.confidence,
            foot_vel[..., :2].reshape(n, 4),
            randomized.imu_linear_acceleration / self.cfg.gravity_m_s2,
        ), dim=-1)
        if frame.shape != (n, TORQUE_TRACTION_FRAME_SCHEMA.frame_dimension):
            raise RuntimeError(f"torque Student frame changed shape: {tuple(frame.shape)}")
        self.packet = IsaacTorqueTractionPacket(
            frame, force_result.force_local_n, contact.probability, contact.state,
            force_result.confidence, force_result.residual_norm_nm, force_result.condition_score,
            traction.traction_utilization, traction.slip_probability, traction.traction_margin,
            traction.friction_lower_bound, traction.slip_event_mu_estimate, traction.state,
            traction.slip_duration_s, foot_vel[..., :2], randomized.imu_linear_acceleration,
            randomized.tau_est, dynamics.tau_est_minus_model,
        )
        self.last_step, self.packet = step, self.packet
        if benchmark and torch.cuda.is_available() and str(self.env.device).startswith("cuda"):
            torch.cuda.synchronize(self.env.device)
        self.last_compute_latency_ms = (time.perf_counter() - start_time) * 1000.0
        return self.packet


class IsaacTorqueTractionFrame(ManagerTermBase):
    """Observation term yielding exactly one deployment-native 125-D frame."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        estimator_cfg = IsaacTorqueEstimatorCfg(
            asset_name=params.get("asset_cfg", SceneEntityCfg("robot")).name,
            randomization_stage=int(params.get("randomization_stage", 0)),
            seed=int(params.get("seed", 20260803)),
        )
        if not hasattr(env, "_isaac_torque_traction_state"):
            env._isaac_torque_traction_state = _IsaacTorqueEstimatorState(env, estimator_cfg)
        self.state: _IsaacTorqueEstimatorState = env._isaac_torque_traction_state

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.state.reset(env_ids)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), randomization_stage: int = 0, seed: int = 20260803) -> torch.Tensor:
        del env, asset_cfg, randomization_stage, seed
        return self.state.compute().frame


def torque_traction_packet(env) -> IsaacTorqueTractionPacket:
    """Read the cached native-signal packet for logging/rewards/evaluation."""
    if not hasattr(env, "_isaac_torque_traction_state"):
        raise RuntimeError("torque-traction observation term has not been initialized")
    return env._isaac_torque_traction_state.compute()
