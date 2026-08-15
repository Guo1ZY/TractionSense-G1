"""MuJoCo native-state analytical foot-force estimator.

This module never calls ``mj_contactForce``. Contact forces from MuJoCo are
reserved for the separate truth bridge and evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


TRACTION_SOURCE = Path("/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab")
if str(TRACTION_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRACTION_SOURCE))

from unitree_rl_lab.traction.schema import G1_29DOF_JOINT_ORDER  # noqa: E402
from unitree_rl_lab.traction_torque.analytical_force_estimator import AnalyticalDualFootForceEstimator, AnalyticalForceEstimatorInput  # noqa: E402
from unitree_rl_lab.traction_torque.contact_estimator import HybridContactEstimator, HybridContactInput  # noqa: E402
from unitree_rl_lab.traction_torque.dynamics import inverse_dynamics_torque_residual  # noqa: E402
from unitree_rl_lab.traction_torque.schema import EstimatedDualFootForce, LEFT_LEG_ACTION_INDICES, RIGHT_LEG_ACTION_INDICES  # noqa: E402
from unitree_rl_lab.traction_torque.torque_filter import JointStateFilter  # noqa: E402
from unitree_rl_lab.traction_torque.traction_estimator import TractionEstimatorInput, TractionStateEstimator  # noqa: E402
from unitree_rl_lab.traction_torque.randomization import TorqueDynamicsObservationModel, TorqueDynamicsRandomizationCfg  # noqa: E402


@dataclass(frozen=True)
class MujocoTorqueEstimatorOutput:
    estimated: EstimatedDualFootForce
    force_local_n: np.ndarray
    contact_probability: np.ndarray
    contact_state: np.ndarray
    confidence: np.ndarray
    traction_utilization: np.ndarray
    slip_probability: np.ndarray
    traction_margin: np.ndarray
    friction_lower_bound: np.ndarray
    slip_state: np.ndarray
    slip_duration_s: np.ndarray
    tau_est_nm: np.ndarray
    foot_planar_velocity_m_s: np.ndarray
    imu_linear_acceleration_m_s2: np.ndarray


class MujocoTorqueForceEstimator:
    """Full 35-DoF MuJoCo inverse dynamics plus per-leg force solve."""

    def __init__(self, model: mujoco.MjModel, *, dt: float = 0.02, randomization_stage: int = 0, seed: int = 20260803) -> None:
        if model.nv != 35 or model.nq != 36:
            raise ValueError(f"expected free-base G1 nq=36,nv=35; got {model.nq},{model.nv}")
        self.model, self.dt = model, dt
        self.joint_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_29DOF_JOINT_ORDER])
        if np.any(self.joint_ids < 0):
            raise ValueError("MuJoCo model is missing a canonical G1 joint")
        self.qpos_address = model.jnt_qposadr[self.joint_ids]
        self.dof_address = model.jnt_dofadr[self.joint_ids]
        self.model_joint_index = self.dof_address - 6
        self.leg_indices = (LEFT_LEG_ACTION_INDICES, RIGHT_LEG_ACTION_INDICES)
        self.foot_body_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("left_ankle_roll_link", "right_ankle_roll_link"))
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.imu_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        self.filter = JointStateFilter(1, 29, cfg=__import__("unitree_rl_lab.traction_torque.torque_filter", fromlist=["JointStateFilterCfg"]).JointStateFilterCfg(dt=dt))
        self.force_estimator = AnalyticalDualFootForceEstimator(1, cfg=__import__("unitree_rl_lab.traction_torque.analytical_force_estimator", fromlist=["AnalyticalForceEstimatorCfg"]).AnalyticalForceEstimatorCfg(dt=dt))
        self.contact_estimator = HybridContactEstimator(1, cfg=__import__("unitree_rl_lab.traction_torque.contact_estimator", fromlist=["HybridContactEstimatorCfg"]).HybridContactEstimatorCfg(dt=dt))
        self.traction_estimator = TractionStateEstimator(1, cfg=__import__("unitree_rl_lab.traction_torque.traction_estimator", fromlist=["TractionStateEstimatorCfg"]).TractionStateEstimatorCfg(dt=dt))
        self.randomization = TorqueDynamicsObservationModel(
            1, 29, 35,
            cfg=TorqueDynamicsRandomizationCfg(curriculum_stage=randomization_stage, dt=dt),
            seed=seed,
            leg_joint_indices=(LEFT_LEG_ACTION_INDICES, RIGHT_LEG_ACTION_INDICES),
        )
        self.randomization_stage = randomization_stage
        self.previous_velocity = torch.zeros(1, 2, 2)
        self.previous_force = torch.zeros(1, 2, 3)

    def reset(self) -> None:
        self.filter.reset(); self.force_estimator.reset(); self.contact_estimator.reset(); self.traction_estimator.reset(); self.randomization.reset()
        self.previous_velocity.zero_(); self.previous_force.zero_()

    def _body_velocity_world(self, data: mujoco.MjData, body_id: int) -> np.ndarray:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        return velocity

    def update(self, data: mujoco.MjData) -> MujocoTorqueEstimatorOutput:
        model = self.model
        q = torch.from_numpy(np.asarray(data.qpos[self.qpos_address], dtype=np.float32))[None]
        dq = torch.from_numpy(np.asarray(data.qvel[self.dof_address], dtype=np.float32))[None]
        tau_canonical = torch.from_numpy(np.asarray(data.qfrc_actuator[self.dof_address], dtype=np.float32))[None]
        qdd_joint, tau_filtered = self.filter.update(dq, tau_canonical)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        jacobians = []
        positions, velocities = [], []
        for leg, body_id in enumerate(self.foot_body_ids):
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
            indices = [int(self.dof_address[index]) for index in self.leg_indices[leg]]
            jacobians.append(rotation.T @ jacp[:, indices])
            positions.append(np.asarray(data.xpos[body_id]).copy())
            velocities.append(self._body_velocity_world(data, body_id)[3:])
        jacobian = torch.from_numpy(np.asarray(jacobians, dtype=np.float32))[None]
        if self.imu_sensor_id >= 0:
            address = int(model.sensor_adr[self.imu_sensor_id]); dimension = int(model.sensor_dim[self.imu_sensor_id])
            imu_np = np.asarray(data.sensordata[address : address + dimension], dtype=np.float32)
        else:
            imu_np = np.zeros(3, dtype=np.float32)
        randomized = self.randomization.update(
            joint_position=q,
            joint_velocity=dq,
            joint_acceleration=qdd_joint,
            tau_est=tau_filtered,
            imu_linear_acceleration=torch.from_numpy(imu_np)[None],
            mass_matrix=torch.from_numpy(mass.astype(np.float32))[None],
            bias_force=torch.from_numpy(np.asarray(data.qfrc_bias, dtype=np.float32))[None],
            leg_jacobian=jacobian,
        )
        q, dq, qdd_joint, tau_filtered = randomized.joint_position, randomized.joint_velocity, randomized.joint_acceleration, randomized.tau_est
        jacobian = randomized.leg_jacobian
        qdd_model = np.zeros(29, dtype=np.float32)
        qdd_model[self.model_joint_index] = qdd_joint[0].numpy()
        tau_model = np.zeros(29, dtype=np.float32)
        tau_model[self.model_joint_index] = tau_filtered[0].numpy()
        generalized_acceleration = torch.from_numpy(np.concatenate((np.asarray(data.qacc[:6], dtype=np.float32), qdd_model)))[None]
        dynamics = inverse_dynamics_torque_residual(
            randomized.mass_matrix,
            randomized.bias_force,
            generalized_acceleration,
            torch.from_numpy(tau_model)[None],
        )
        leg_model_indices = tuple(
            tuple(int(self.model_joint_index[index]) for index in indices)
            for indices in self.leg_indices
        )
        leg_force = torch.stack((
            dynamics.contact_generalized_joint_force[:, list(leg_model_indices[0])],
            dynamics.contact_generalized_joint_force[:, list(leg_model_indices[1])],
        ), dim=1)
        position = torch.from_numpy(np.asarray(positions, dtype=np.float32))[None]
        velocity = torch.from_numpy(np.asarray(velocities, dtype=np.float32))[None]
        prior = self.contact_estimator.probability
        kinematic = ((position[..., 2] < 0.10) & (velocity[..., 2].abs() < 0.5)).float()
        force = self.force_estimator.update(AnalyticalForceEstimatorInput(jacobian, leg_force, torch.maximum(prior, kinematic)))
        force_foot = force.force_local_n.reshape(1, 2, 3)
        leg_tau = torch.stack((tau_filtered[:, list(self.leg_indices[0])], tau_filtered[:, list(self.leg_indices[1])]), dim=1)
        imu = randomized.imu_linear_acceleration
        imu_np = imu.numpy()[0]
        contact = self.contact_estimator.update(HybridContactInput(
            position[..., 2], velocity[..., 2], velocity[..., :2], leg_tau, force_foot[..., 2],
            torch.stack((q[:, list(self.leg_indices[0])], q[:, list(self.leg_indices[1])]), dim=1), imu,
        ))
        acceleration = (velocity[..., :2] - self.previous_velocity) / self.dt
        growth = torch.linalg.vector_norm((force_foot - self.previous_force) / self.dt, dim=-1)
        traction = self.traction_estimator.update(TractionEstimatorInput(
            force.force_local_n, contact.probability, velocity[..., :2], acceleration, growth,
            force.residual_norm_nm, imu, force.confidence,
        ))
        self.previous_velocity.copy_(velocity[..., :2]); self.previous_force.copy_(force_foot)
        f, p, c, residual, condition = (value.detach().numpy()[0] for value in (force.force_local_n, contact.probability, force.confidence, force.residual_norm_nm, force.condition_score))
        common = EstimatedDualFootForce(float(data.time), f[:3], f[3:], float(p[0]), float(p[1]), float(c[0]), float(c[1]), float(residual[0]), float(residual[1]), float(condition[0]), float(condition[1]))
        return MujocoTorqueEstimatorOutput(
            common, f, p, contact.state.numpy()[0], c,
            traction.traction_utilization.numpy()[0], traction.slip_probability.numpy()[0],
            traction.traction_margin.numpy()[0], traction.friction_lower_bound.numpy()[0],
            traction.state.numpy()[0], traction.slip_duration_s.numpy()[0],
            tau_filtered.numpy()[0], velocity.numpy()[0, :, :2], imu_np,
        )
