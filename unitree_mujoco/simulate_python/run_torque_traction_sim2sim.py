#!/usr/bin/env python3
"""Fixed-policy native-torque MuJoCo Sim2Sim; no training or contact input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRACTION_SOURCE = Path("/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab")
for path in (ROOT / "simulate_python", TRACTION_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torque_contact_truth import MujocoTorqueTruthBridge  # noqa: E402
from torque_force_estimator import MujocoTorqueForceEstimator  # noqa: E402
from unitree_rl_lab.traction.deployment import DEFAULT_JOINT_POSITION, G1_29DOF_JOINT_ORDER, JOINT_DAMPING, JOINT_EFFORT_LIMIT, JOINT_STIFFNESS, POLICY_DT_S  # noqa: E402
from unitree_rl_lab.traction_torque.governor import TorqueTractionCommandGovernor  # noqa: E402
from unitree_rl_lab.traction_torque.schema import TORQUE_TRACTION_FRAME_SCHEMA, TORQUE_TRACTION_JOINT_INDICES  # noqa: E402


def set_friction(model: mujoco.MjModel, estimator: MujocoTorqueForceEstimator, left: float, right: float) -> None:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_friction[floor, 0] = min(left, right)
    bridge = MujocoTorqueTruthBridge(model).force_bridge
    for geom, body in enumerate(model.geom_bodyid):
        if bridge._is_descendant(int(body), estimator.foot_body_ids[0]):
            model.geom_friction[geom, 0] = left
        elif bridge._is_descendant(int(body), estimator.foot_body_ids[1]):
            model.geom_friction[geom, 0] = right


class Controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, policy_path: Path, *, governor_enabled: bool, randomization_stage: int = 0, seed: int = 20260803) -> None:
        self.model, self.data = model, data
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.estimator = MujocoTorqueForceEstimator(model, dt=POLICY_DT_S, randomization_stage=randomization_stage, seed=seed)
        self.truth = MujocoTorqueTruthBridge(model)
        self.governor = TorqueTractionCommandGovernor(1, enabled=governor_enabled)
        self.governor_enabled = governor_enabled
        self.history = np.zeros((15, 125), dtype=np.float32)
        self.history_initialized = False
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.position_target = np.asarray(DEFAULT_JOINT_POSITION, dtype=np.float64).copy()
        self.joint_ids = self.estimator.joint_ids
        self.qpos_address, self.dof_address = self.estimator.qpos_address, self.estimator.dof_address
        self.actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0] for joint in self.joint_ids])
        self.pelvis_id = self.estimator.pelvis_id
        self.mass = float(model.body_mass.sum())
        self.slip_duration = torch.zeros(1, 2)

    def initialize(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_address] = np.asarray(DEFAULT_JOINT_POSITION)
        self.data.qvel[:] = 0.0
        self.previous_action.fill(0); self.position_target[:] = DEFAULT_JOINT_POSITION
        self.estimator.reset(); self.governor.reset(); self.history.fill(0); self.history_initialized = False
        mujoco.mj_forward(self.model, self.data)

    def _base_velocity_local(self) -> np.ndarray:
        velocity = np.zeros(6); mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_id, velocity, 1)
        return velocity

    def policy_step(self, command: np.ndarray) -> dict[str, np.ndarray | float | int]:
        estimate = self.estimator.update(self.data)
        base_velocity = self._base_velocity_local()
        rotation = np.asarray(self.data.xmat[self.pelvis_id]).reshape(3, 3)
        projected_gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        effort = np.asarray(JOINT_EFFORT_LIMIT)[list(TORQUE_TRACTION_JOINT_INDICES)]
        frame = np.concatenate((
            base_velocity[:3] * 0.2, projected_gravity, command,
            self.data.qpos[self.qpos_address] - np.asarray(DEFAULT_JOINT_POSITION),
            self.data.qvel[self.dof_address] * 0.05, self.previous_action,
            estimate.tau_est_nm[list(TORQUE_TRACTION_JOINT_INDICES)] / effort,
            np.clip(estimate.force_local_n / (self.mass * 9.81), -3.0, 3.0),
            estimate.contact_probability, estimate.confidence,
            estimate.foot_planar_velocity_m_s.reshape(4), estimate.imu_linear_acceleration_m_s2 / 9.81,
        )).astype(np.float32)
        if frame.shape != (125,):
            raise RuntimeError(f"MuJoCo frame changed shape to {frame.shape}")
        if not self.history_initialized:
            self.history[:] = frame; self.history_initialized = True
        else:
            self.history[:-1] = self.history[1:]; self.history[-1] = frame
        history = torch.from_numpy(self.history[None])
        with torch.inference_mode():
            first = self.policy(history)
        action, learned_force, learned_contact, slip, utilization, margin, confidence = first
        slipping = (slip > 0.6) & (learned_contact > 0.5)
        self.slip_duration = torch.where(slipping, self.slip_duration + POLICY_DT_S, torch.zeros_like(self.slip_duration))
        governor = self.governor.update(
            raw_command=torch.from_numpy(command[None]), slip_probability=slip,
            traction_utilization=utilization, traction_margin=margin,
            contact_probability=learned_contact, estimator_confidence=confidence,
            foot_relative_velocity=torch.from_numpy(estimate.foot_planar_velocity_m_s[None]),
            slip_duration=self.slip_duration, current_velocity=torch.from_numpy(base_velocity[3:][None].astype(np.float32)),
        )
        governed_history = history.clone(); governed_history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")] = governor.adjusted_command
        with torch.inference_mode():
            final = self.policy(governed_history)
        action = torch.nan_to_num(final[0]).clamp(-4.0, 4.0)[0].numpy()
        self.previous_action = action.copy(); self.position_target = np.asarray(DEFAULT_JOINT_POSITION) + 0.25 * action
        truth = self.truth.read(self.data)  # metrics only, after policy inputs are complete
        return {
            "estimated_force_local_n": estimate.force_local_n, "true_force_local_n": truth.force_local_n,
            "contact_probability": estimate.contact_probability, "true_contact": truth.contact_count > 0,
            "contact_point_slip_speed_m_s": truth.contact_point_slip_speed_m_s,
            "analytical_confidence": estimate.confidence, "learned_confidence": confidence[0].numpy(),
            "slip_probability": slip[0].numpy(), "traction_utilization": utilization[0].numpy(),
            "traction_margin": margin[0].numpy(), "raw_command": command.copy(),
            "adjusted_command": governor.adjusted_command[0].numpy(), "speed_scale": governor.speed_scale.item(),
            "acceleration_limit": governor.acceleration_limit.item(), "yaw_limit": governor.yaw_limit.item(),
            "push_off_scale": governor.push_off_scale.item(), "governor_state": governor.state.item(),
            "base_velocity": base_velocity[3:].copy(), "action": action.copy(),
        }

    def physics_step(self) -> None:
        torque = np.asarray(JOINT_STIFFNESS) * (self.position_target - self.data.qpos[self.qpos_address]) - np.asarray(JOINT_DAMPING) * self.data.qvel[self.dof_address]
        self.data.ctrl[self.actuator_ids] = np.clip(torque, -np.asarray(JOINT_EFFORT_LIMIT), np.asarray(JOINT_EFFORT_LIMIT))
        mujoco.mj_step(self.model, self.data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--duration_s", type=float, default=4.0)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--left_friction", type=float); parser.add_argument("--right_friction", type=float)
    parser.add_argument("--transition_friction", type=float)
    parser.add_argument("--command", type=float, nargs=3, default=(0.6, 0.0, 0.0))
    parser.add_argument("--disable_governor", action="store_true")
    parser.add_argument("--randomization_stage", type=int, choices=range(6), default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model)); model.opt.timestep = 0.005; data = mujoco.MjData(model)
    controller = Controller(model, data, args.policy, governor_enabled=not args.disable_governor, randomization_stage=args.randomization_stage, seed=args.seed)
    left = args.friction if args.left_friction is None else args.left_friction; right = args.friction if args.right_friction is None else args.right_friction
    set_friction(model, controller.estimator, left, right); controller.initialize()
    decimation = int(round(POLICY_DT_S / model.opt.timestep)); command = np.asarray(args.command, dtype=np.float32)
    records: dict[str, list] = {}; transitioned = False; fell = False; nonfinite = 0
    for _ in range(int(np.ceil(args.duration_s / POLICY_DT_S))):
        if args.transition_friction is not None and not transitioned and data.time >= args.duration_s / 2:
            set_friction(model, controller.estimator, args.transition_friction, args.transition_friction); transitioned = True
        sample = controller.policy_step(command)
        records.setdefault("timestamp_s", []).append(float(data.time)); records.setdefault("base_height_m", []).append(float(data.qpos[2]))
        records.setdefault("ground_friction_mu", []).append(np.asarray((args.transition_friction, args.transition_friction)) if transitioned else np.asarray((left, right)))
        for key, value in sample.items():
            records.setdefault(key, []).append(value); nonfinite += int((~np.isfinite(np.asarray(value))).sum())
        for _ in range(decimation): controller.physics_step()
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.30:
            fell = data.qpos[2] < 0.30; break
    arrays = {key: np.asarray(value) for key, value in records.items()}; args.output.parent.mkdir(parents=True, exist_ok=True)
    rollout_metadata = {
        "mode": "fixed_policy_no_mujoco_training",
        "contact_truth_policy_input": False,
        "foot_slip_metric": "contact-point relative tangential velocity",
        "requested_duration_s": args.duration_s,
        "fell": bool(fell),
        "randomization_stage": args.randomization_stage,
        "seed": args.seed,
    }
    np.savez_compressed(args.output, **arrays, metadata=np.asarray(rollout_metadata, dtype=object))
    error = arrays["estimated_force_local_n"] - arrays["true_force_local_n"]
    summary = {
        "mode": "mujoco_fixed_policy_sim2sim", "policy_steps": len(arrays["timestamp_s"]),
        "physics_dt_s": model.opt.timestep, "policy_dt_s": POLICY_DT_S, "action_dimension": 29,
        "governor_enabled": not args.disable_governor, "randomization_stage": args.randomization_stage,
        "seed": args.seed, "fell": bool(fell), "nonfinite_count": nonfinite,
        "minimum_base_height_m": float(arrays["base_height_m"].min()),
        "velocity_tracking_mae_m_s": float(np.abs(arrays["base_velocity"][:, 0] - arrays["raw_command"][:, 0]).mean()),
        "force_mae_n": np.abs(error).mean(axis=0).tolist(), "force_rmse_n": np.sqrt(np.square(error).mean(axis=0)).tolist(),
        "ground_truth_slip_rate": float((arrays["contact_point_slip_speed_m_s"] > 0.10).mean()),
        "governor_activation_ratio": float((arrays["governor_state"] != 0).mean()),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
