#!/usr/bin/env python3
"""Run a fixed exported traction Student in MuJoCo; never trains or updates it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRACTION_SOURCE = Path(
    "/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab"
)
if str(ROOT / "simulate_python") not in sys.path:
    sys.path.insert(0, str(ROOT / "simulate_python"))
if str(TRACTION_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRACTION_SOURCE))

from traction_force_bridge import MujocoFootForceBridge  # noqa: E402
from unitree_rl_lab.traction import (  # noqa: E402
    DualFootForceInput,
    ProprioceptiveState,
    TactileDomainRandomizationCfg,
    TactileObservationModel,
    TractionPolicyRuntime,
)
from unitree_rl_lab.traction.deployment import (  # noqa: E402
    DEFAULT_JOINT_POSITION,
    G1_29DOF_JOINT_ORDER,
    JOINT_DAMPING,
    JOINT_EFFORT_LIMIT,
    JOINT_STIFFNESS,
    POLICY_DT_S,
)


class FixedMujocoController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        *,
        seed: int,
        tactile_stage: int,
        sensor_invalid: bool,
        governor_enabled: bool,
    ) -> None:
        self.model = model
        self.data = data
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.runtime = TractionPolicyRuntime(
            self.policy,
            governor_enabled=governor_enabled,
        )
        self.bridge = MujocoFootForceBridge(model)
        self.tactile = TactileObservationModel(
            1,
            cfg=TactileDomainRandomizationCfg(dt=POLICY_DT_S),
            seed=seed,
            curriculum_stage=tactile_stage,
        )
        self.sensor_invalid = sensor_invalid
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.position_target = np.asarray(
            DEFAULT_JOINT_POSITION,
            dtype=np.float64,
        )
        self.pelvis_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                for name in G1_29DOF_JOINT_ORDER
            ]
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("one or more canonical 29-DOF joints are missing")
        self.qpos_address = model.jnt_qposadr[self.joint_ids]
        self.dof_address = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.asarray(
            [
                np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)[0]
                for joint_id in self.joint_ids
            ]
        )
        self.stiffness = np.asarray(JOINT_STIFFNESS)
        self.damping = np.asarray(JOINT_DAMPING)
        self.effort_limit = np.asarray(JOINT_EFFORT_LIMIT)

    def initialize(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_address] = np.asarray(DEFAULT_JOINT_POSITION)
        self.data.qvel[:] = 0.0
        self.previous_action.fill(0.0)
        self.position_target[:] = np.asarray(DEFAULT_JOINT_POSITION)
        self.runtime.reset()
        self.tactile.reset()
        mujoco.mj_forward(self.model, self.data)

    def _body_velocity_local(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_id,
            velocity,
            1,
        )
        return velocity

    def _foot_tangent_velocity_local(self) -> np.ndarray:
        result = np.zeros((2, 2), dtype=np.float64)
        velocity = np.zeros(6, dtype=np.float64)
        for foot, body_id in enumerate(self.bridge.foot_body_ids):
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                1,
            )
            result[foot] = velocity[3:5]
        return result

    def policy_step(
        self,
        command: np.ndarray,
    ) -> dict[str, np.ndarray | float | int | tuple[str, ...]]:
        ideal = self.bridge.read(self.data)
        tactile = self.tactile(torch.from_numpy(ideal.force_local_n)[None])
        observed = tactile.force_xyz_n[0].numpy()
        valid = tactile.valid[0].numpy()
        age = tactile.sample_age_s[0].numpy()
        if self.sensor_invalid:
            valid[:] = False
            age[:] = 1.0
        force_input = DualFootForceInput(
            timestamp=float(self.data.time),
            left_force_xyz=observed[:3],
            right_force_xyz=observed[3:],
            left_valid=bool(valid[0]),
            right_valid=bool(valid[1]),
            left_age=float(age[0]),
            right_age=float(age[1]),
            left_source="mujoco_contact_bridge",
            right_source="mujoco_contact_bridge",
        )
        local_velocity = self._body_velocity_local()
        foot_tangent_velocity = self._foot_tangent_velocity_local()
        local_to_world = self.data.xmat[self.pelvis_id].reshape(3, 3)
        projected_gravity = local_to_world.T @ np.asarray([0.0, 0.0, -1.0])
        state = ProprioceptiveState(
            timestamp=float(self.data.time),
            base_angular_velocity=local_velocity[:3],
            projected_gravity=projected_gravity,
            joint_position=self.data.qpos[self.qpos_address],
            joint_velocity=self.data.qvel[self.dof_address],
            previous_action=self.previous_action,
            base_linear_velocity=local_velocity[3:],
        )
        with torch.inference_mode():
            output = self.runtime.step(state, force_input, command)
        self.previous_action = output.action[0].numpy().copy()
        self.position_target = output.joint_position_target[0].numpy().copy()
        return {
            "ideal_force": ideal.force_local_n,
            "observed_force": observed,
            "contact_count": np.asarray(ideal.contact_count),
            "effective_contact_friction": self._effective_contact_friction(),
            "valid": valid,
            "age": age,
            "base_velocity": local_velocity[3:].copy(),
            "base_yaw_rate": float(local_velocity[2]),
            "projected_gravity": projected_gravity.copy(),
            "foot_tangent_velocity_proxy": foot_tangent_velocity.reshape(4),
            "slip_speed_proxy": np.linalg.norm(
                foot_tangent_velocity,
                axis=1,
            ),
            "action": self.previous_action.copy(),
            "adjusted_command": output.governor.adjusted_command[0].numpy(),
            "acceleration_limit": output.governor.acceleration_limit.item(),
            "deceleration_limit": output.governor.deceleration_limit.item(),
            "speed_scale": output.governor.speed_scale.item(),
            "yaw_limit": output.governor.yaw_limit.item(),
            "push_off_scale": output.governor.push_off_scale.item(),
            "slip_probability": output.slip_probability[0].numpy(),
            "traction_score": output.traction_score.item(),
            "sensor_confidence": output.sensor_confidence.item(),
            "governor_state": output.governor.state.item(),
            "safety_flags": output.safety_flags,
        }

    def _effective_contact_friction(self) -> np.ndarray:
        # Zero denotes no current contact; contact_count disambiguates it from μ.
        result = np.zeros(2, dtype=np.float64)
        for contact in self.data.contact:
            body = (
                int(self.model.geom_bodyid[contact.geom[0]]),
                int(self.model.geom_bodyid[contact.geom[1]]),
            )
            for foot, foot_root in enumerate(self.bridge.foot_body_ids):
                if (
                    self.bridge._is_descendant(body[0], foot_root)
                    or self.bridge._is_descendant(body[1], foot_root)
                ):
                    result[foot] = max(result[foot], float(contact.friction[0]))
        return result

    def physics_step(self) -> None:
        position = self.data.qpos[self.qpos_address]
        velocity = self.data.qvel[self.dof_address]
        torque = (
            self.stiffness * (self.position_target - position)
            - self.damping * velocity
        )
        torque = np.clip(torque, -self.effort_limit, self.effort_limit)
        self.data.ctrl[self.actuator_ids] = torque
        mujoco.mj_step(self.model, self.data)


def _set_contact_friction(
    model: mujoco.MjModel,
    bridge: MujocoFootForceBridge,
    left_coefficient: float,
    right_coefficient: float,
) -> None:
    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_id < 0:
        raise ValueError("floor geom not found")
    model.geom_friction[floor_id, 0] = min(left_coefficient, right_coefficient)
    for geom_id, body_id in enumerate(model.geom_bodyid):
        if bridge._is_descendant(int(body_id), bridge.foot_body_ids[0]):
            model.geom_friction[geom_id, 0] = left_coefficient
        elif bridge._is_descendant(int(body_id), bridge.foot_body_ids[1]):
            model.geom_friction[geom_id, 0] = right_coefficient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "unitree_robots/g1/scene_29dof.xml",
    )
    parser.add_argument("--duration_s", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--left_friction", type=float)
    parser.add_argument("--right_friction", type=float)
    parser.add_argument("--transition_friction", type=float)
    parser.add_argument("--command", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--tactile_stage", type=int, choices=range(6), default=0)
    parser.add_argument("--sensor_invalid", action="store_true")
    parser.add_argument(
        "--disable_governor",
        action="store_true",
        help="Ablation/reference mode: pass the raw command through unchanged.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.friction <= 0.0:
        raise ValueError("duration and friction must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    controller = FixedMujocoController(
        model,
        data,
        args.policy,
        seed=args.seed,
        tactile_stage=args.tactile_stage,
        sensor_invalid=args.sensor_invalid,
        governor_enabled=not args.disable_governor,
    )
    left_friction = (
        args.friction if args.left_friction is None else args.left_friction
    )
    right_friction = (
        args.friction if args.right_friction is None else args.right_friction
    )
    _set_contact_friction(
        model,
        controller.bridge,
        left_friction,
        right_friction,
    )
    controller.initialize()
    decimation = int(round(POLICY_DT_S / model.opt.timestep))
    if decimation != 4:
        raise RuntimeError(f"MuJoCo decimation is {decimation}, expected 4")
    command = np.asarray(args.command, dtype=np.float32)
    records: dict[str, list[np.ndarray | float | int]] = {}
    flags: set[str] = set()
    transition_done = False
    nonfinite = 0
    fell = False
    policy_steps = int(np.ceil(args.duration_s / POLICY_DT_S))
    for policy_index in range(policy_steps):
        if (
            args.transition_friction is not None
            and not transition_done
            and data.time >= args.duration_s / 2.0
        ):
            _set_contact_friction(
                model,
                controller.bridge,
                args.transition_friction,
                args.transition_friction,
            )
            transition_done = True
        sample = controller.policy_step(command)
        flags.update(sample.pop("safety_flags"))  # type: ignore[arg-type]
        records.setdefault("timestamp_s", []).append(float(data.time))
        records.setdefault("base_height_m", []).append(float(data.qpos[2]))
        records.setdefault("ground_friction_mu", []).append(
            np.asarray(
                (
                    args.transition_friction
                    if transition_done
                    else left_friction,
                    args.transition_friction
                    if transition_done
                    else right_friction,
                )
            )
        )
        records.setdefault("raw_command", []).append(command.copy())
        for key, value in sample.items():
            records.setdefault(key, []).append(value)  # type: ignore[arg-type]
            nonfinite += int(np.count_nonzero(~np.isfinite(np.asarray(value))))
        for _ in range(decimation):
            controller.physics_step()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("MuJoCo state contains NaN or Inf")
        if data.qpos[2] < 0.30:
            fell = True
            break
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["terminated_fall"] = np.asarray([fell], dtype=bool)
    arrays["final_base_height_m"] = np.asarray([data.qpos[2]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        metadata=np.asarray(
            [
                "fixed_policy_no_training_or_finetuning",
                "force_order=L_Fx,L_Fy,L_Fz,R_Fx,R_Fy,R_Fz",
                "force_frame=matching_ankle_roll_link_local",
                "foot_tangent_velocity_not_recorded",
                f"seed={args.seed}",
                f"governor_enabled={not args.disable_governor}",
            ]
        ),
    )
    summary = {
        "mode": "mujoco_fixed_policy_sim2sim",
        "policy_steps": len(arrays["timestamp_s"]),
        "physics_dt_s": model.opt.timestep,
        "policy_dt_s": POLICY_DT_S,
        "decimation": decimation,
        "action_dimension": 29,
        "governor_enabled": not args.disable_governor,
        "nonfinite": nonfinite,
        "fell": fell,
        "minimum_base_height_m": float(arrays["base_height_m"].min()),
        "maximum_action_abs": float(np.abs(arrays["action"]).max()),
        "force_nonzero_samples": int(
            np.count_nonzero(np.linalg.norm(arrays["ideal_force"], axis=1) > 1.0e-6)
        ),
        "safety_flags": sorted(flags),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
