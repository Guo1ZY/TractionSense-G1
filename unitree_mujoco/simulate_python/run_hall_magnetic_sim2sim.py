#!/usr/bin/env python3
"""Run the frozen Hall-only G1 policy in an independent MuJoCo forward model.

The causal path is MuJoCo contacts -> local TPU compliance -> four embedded
magnet poses -> dipole Bx/By/Bz -> electronics/history -> policy.  Contact
force and ground friction are recorded only as evaluator truth and never enter
the 1864-D policy/risk observation.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
LAB_SOURCE = Path("/home/mosense/guo/unitree_rl_lab/source/unitree_rl_lab")
for path in (ROOT / "simulate_python", LAB_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hall_foot_forward_model import (  # noqa: E402
    AXES,
    FEET,
    SENSORS,
    HallFootForwardConfig,
    HallFootForwardModel,
    MujocoHallContactReader,
    randomized_config,
)
from unitree_rl_lab.traction.deployment import (  # noqa: E402
    DEFAULT_JOINT_POSITION,
    JOINT_DAMPING,
    JOINT_EFFORT_LIMIT,
    JOINT_STIFFNESS,
    POLICY_DT_S,
)
from unitree_rl_lab.traction.hall_governor import (  # noqa: E402
    HallTractionGovernor,
    HallTractionGovernorCfg,
)
from unitree_rl_lab.traction.layout_magnetic_student import (  # noqa: E402
    INPUT_DIM,
)
from unitree_rl_lab.traction.schema import G1_29DOF_JOINT_ORDER  # noqa: E402


BASE_HISTORY = 5
HALL_HISTORY = 15
ACTION_DIM = 29
PHYSICS_DT = 0.005


class History:
    def __init__(self, length: int) -> None:
        self.values: deque[np.ndarray] = deque(maxlen=length)
        self.length = length

    def append(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).copy()
        if not self.values:
            for _ in range(self.length):
                self.values.append(value.copy())
        else:
            self.values.append(value)

    def flat(self) -> np.ndarray:
        if len(self.values) != self.length:
            raise RuntimeError("history is not initialized")
        return np.concatenate(tuple(self.values))


class HallElectronics:
    """Episode-fixed Hall uncertainty; no force calibration is performed."""

    def __init__(self, rng: np.random.Generator, randomized: bool) -> None:
        self.rng = rng
        if randomized:
            self.gain = rng.uniform(0.78, 1.22, (FEET, SENSORS, AXES))
            self.bias = rng.normal(0.0, 0.08, (FEET, SENSORS, AXES))
            self.cross_axis = np.eye(AXES)[None, None] + rng.normal(
                0.0, 0.04, (FEET, SENSORS, AXES, AXES)
            )
            self.bad = rng.random((FEET, SENSORS, 1)) < 0.02
            self.delay = int(rng.integers(0, 4))
            self.noise_std = 0.025
        else:
            self.gain = np.ones((FEET, SENSORS, AXES))
            self.bias = np.zeros((FEET, SENSORS, AXES))
            self.cross_axis = np.broadcast_to(
                np.eye(AXES), (FEET, SENSORS, AXES, AXES)
            ).copy()
            self.bad = np.zeros((FEET, SENSORS, 1), dtype=bool)
            self.delay = 0
            self.noise_std = 0.0
        self.queue: deque[np.ndarray] = deque(maxlen=self.delay + 1)

    def apply(self, value: np.ndarray) -> np.ndarray:
        output = np.einsum("fsab,fsb->fsa", self.cross_axis, value)
        output = output * self.gain + self.bias
        if self.noise_std:
            output += self.rng.normal(0.0, self.noise_std, output.shape)
        output = np.where(self.bad, 0.0, output)
        output = np.clip(output, -6.0, 6.0).astype(np.float32)
        if not self.queue:
            for _ in range(self.delay + 1):
                self.queue.append(output.copy())
        else:
            self.queue.append(output)
        return self.queue[0].copy()


class HallOnlyMujocoController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_onnx: Path,
        risk_onnx: Path,
        *,
        seed: int,
        randomized_hall: bool,
        fault_mode: str,
        governor_enabled: bool,
        low_speed_limit: float,
        high_speed_limit: float,
        low_reprobe_s: float,
        low_probability: float,
        high_probability: float,
        critical_probability: float,
        critical_hold_s: float,
        reference_alpha: float,
        relative_low_rise: float,
        relative_high_drop: float,
        low_hold_s: float,
        high_hold_s: float,
        probe_speed_limit: float,
        probe_duration_s: float,
        probe_relative_clear_drop: float,
        crawl_pulse_s: float,
        accel_rate: float,
        decel_rate: float,
    ) -> None:
        self.model = model
        self.data = data
        self.rng = np.random.default_rng(seed)
        base_cfg = HallFootForwardConfig()
        hall_cfg = randomized_config(base_cfg, self.rng) if randomized_hall else base_cfg
        self.hall_model = HallFootForwardModel(hall_cfg, seed=seed)
        self.electronics = HallElectronics(self.rng, randomized_hall)
        self.contact_reader = MujocoHallContactReader(model)
        self.fault_mode = fault_mode
        self.governor_enabled = bool(governor_enabled)
        self.policy = ort.InferenceSession(
            str(policy_onnx), providers=["CPUExecutionProvider"]
        )
        self.risk = ort.InferenceSession(
            str(risk_onnx), providers=["CPUExecutionProvider"]
        )
        self.policy_input_name = self.policy.get_inputs()[0].name
        self.risk_input_name = self.risk.get_inputs()[0].name
        policy_shape = self.policy.get_inputs()[0].shape
        self.policy_input_dim = int(policy_shape[-1]) if isinstance(policy_shape[-1], int) else INPUT_DIM
        if self.policy_input_dim not in (480, INPUT_DIM):
            raise ValueError(
                f"Hall sim2sim supports the 480-D proprio baseline or 1864-D Hall actor, "
                f"got policy input {policy_shape}"
            )
        self.governor = HallTractionGovernor(
            1,
            POLICY_DT_S,
            "cpu",
            HallTractionGovernorCfg(
                low_speed_limit=low_speed_limit,
                high_speed_limit=high_speed_limit,
                low_lateral_limit=0.05,
                high_lateral_limit=0.35,
                low_yaw_limit=0.15,
                high_yaw_limit=0.80,
                probability_low_enter=low_probability,
                probability_high_enter=high_probability,
                probability_critical_enter=critical_probability,
                critical_hold_s=critical_hold_s,
                state_reference_ema_alpha=reference_alpha,
                relative_low_rise=relative_low_rise,
                relative_high_drop=relative_high_drop,
                low_hold_s=low_hold_s,
                high_hold_s=high_hold_s,
                probe_speed_limit=probe_speed_limit,
                probe_duration_s=probe_duration_s,
                probe_relative_clear_drop=probe_relative_clear_drop,
                crawl_pulse_s=crawl_pulse_s,
                low_reprobe_s=low_reprobe_s,
                linear_accel_rate=accel_rate,
                linear_decel_rate=decel_rate,
            ),
        )
        self.default_joint = np.asarray(DEFAULT_JOINT_POSITION, dtype=np.float64)
        self.stiffness = np.asarray(JOINT_STIFFNESS, dtype=np.float64)
        self.damping = np.asarray(JOINT_DAMPING, dtype=np.float64)
        self.effort = np.asarray(JOINT_EFFORT_LIMIT, dtype=np.float64)
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in G1_29DOF_JOINT_ORDER
            ]
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("canonical G1 29-DOF joints are missing")
        self.qpos_address = model.jnt_qposadr[self.joint_ids]
        self.dof_address = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.asarray(
            [
                int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0])
                for joint in self.joint_ids
            ]
        )
        self.pelvis_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self.pelvis_id < 0:
            raise ValueError("pelvis body is missing")
        self.target = self.default_joint.copy()
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.applied_command = np.zeros(3, dtype=np.float32)
        self.base_histories = {
            key: History(BASE_HISTORY)
            for key in ("ang", "gravity", "command", "q", "qd", "action")
        }
        self.hall_history = History(HALL_HISTORY)
        self.period_history = History(HALL_HISTORY)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_address] = self.default_joint
        self.data.qvel[:] = 0.0
        self.target[:] = self.default_joint
        self.previous_action.fill(0.0)
        self.applied_command.fill(0.0)
        self.governor.reset()
        self.hall_model.reset()
        mujoco.mj_forward(self.model, self.data)

    def _body_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_id,
            velocity,
            1,
        )
        rotation = np.asarray(self.data.xmat[self.pelvis_id]).reshape(3, 3)
        gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        return velocity[:3], velocity[3:], gravity

    def _health(self, hall: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = np.ones(FEET, dtype=np.float32)
        age = np.zeros(FEET, dtype=np.float32)
        if self.fault_mode == "full":
            hall[:] = 0.0
            valid[:] = 0.0
            age[:] = 1.0
        elif self.fault_mode == "left":
            hall[0] = 0.0
            valid[0] = 0.0
            age[0] = 1.0
        elif self.fault_mode == "right":
            hall[1] = 0.0
            valid[1] = 0.0
            age[1] = 1.0
        elif self.fault_mode == "bad_channels":
            hall[:, ::4] = 0.0
        return hall, valid, age

    def policy_step(self, requested_command: np.ndarray) -> dict[str, np.ndarray | float | int]:
        contacts = self.contact_reader.read(self.data)
        magnetic = self.hall_model.update(POLICY_DT_S, contacts)
        magnetic = self.electronics.apply(magnetic)
        magnetic, valid, age = self._health(magnetic)
        angular, linear, gravity = self._body_state()
        q = self.data.qpos[self.qpos_address]
        qd = self.data.qvel[self.dof_address]

        self.base_histories["ang"].append(0.2 * angular)
        self.base_histories["gravity"].append(gravity)
        self.base_histories["command"].append(self.applied_command)
        self.base_histories["q"].append(q - self.default_joint)
        self.base_histories["qd"].append(0.05 * qd)
        self.base_histories["action"].append(self.previous_action)
        self.hall_history.append(magnetic.reshape(-1))
        self.period_history.append(
            np.full(FEET, POLICY_DT_S, dtype=np.float32)
        )
        observation = np.concatenate(
            (
                self.base_histories["ang"].flat(),
                self.base_histories["gravity"].flat(),
                self.base_histories["command"].flat(),
                self.base_histories["q"].flat(),
                self.base_histories["qd"].flat(),
                self.base_histories["action"].flat(),
                self.hall_history.flat(),
                self.period_history.flat(),
                valid,
                np.clip(age / 0.25, 0.0, 1.0),
            )
        ).astype(np.float32)[None]
        if observation.shape != (1, INPUT_DIM) or not np.isfinite(observation).all():
            raise RuntimeError(f"invalid Hall policy observation {observation.shape}")
        policy_observation = observation[:, : self.policy_input_dim]
        action = self.policy.run(None, {self.policy_input_name: policy_observation})[0][0]
        risk = float(
            self.risk.run(None, {self.risk_input_name: observation})[0].reshape(-1)[0]
        )
        if self.governor_enabled:
            governed, state = self.governor.update(
                torch.from_numpy(np.asarray(requested_command, dtype=np.float32)[None]),
                torch.tensor((risk,), dtype=torch.float32),
                torch.tensor((bool(valid.all()),)),
            )
            self.applied_command = governed[0].numpy().copy()
            governor_state = int(state[0])
        else:
            self.applied_command = np.asarray(
                requested_command, dtype=np.float32
            ).copy()
            governor_state = -1
        self.previous_action = np.clip(action, -3.0, 3.0).astype(np.float32)
        self.target = self.default_joint + 0.25 * self.previous_action
        truth_force = np.zeros((FEET, AXES), dtype=np.float64)
        for foot in range(FEET):
            for contact in contacts[foot]:
                force = contact[1]
                truth_force[foot] += force
        return {
            "observation": observation[0],
            "hall": magnetic,
            "valid": valid,
            "age": age,
            "risk": risk,
            "governor_state": governor_state,
            "applied_command": self.applied_command.copy(),
            "action": self.previous_action.copy(),
            "base_velocity": linear,
            "projected_gravity": gravity,
            "truth_contact_force": truth_force,
            "maximum_compression_m": float(np.max(self.hall_model.deformation[..., 2])),
        }

    def physics_step(self) -> None:
        position = self.data.qpos[self.qpos_address]
        velocity = self.data.qvel[self.dof_address]
        torque = self.stiffness * (self.target - position) - self.damping * velocity
        self.data.ctrl[self.actuator_ids] = np.clip(torque, -self.effort, self.effort)
        mujoco.mj_step(self.model, self.data)


def set_friction(model: mujoco.MjModel, coefficient: float) -> None:
    updated = 0
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name and name.startswith("floor"):
            model.geom_friction[geom] = (coefficient, 0.005, 0.0001)
            model.geom_priority[geom] = 1
            updated += 1
    if not updated:
        raise ValueError("no floor geom found")


def adaptive_recovery_gate(
    requested_vx: float,
    sequence: tuple[float, ...],
    mean_vx: list[float],
    mean_applied: list[float],
    mean_risk: list[float],
) -> bool:
    """Require meaningful recovery, not just a tiny relative phase change."""
    if len(sequence) < 3 or sequence[-1] < sequence[1] + 0.30:
        return True
    requested_abs = abs(float(requested_vx))
    minimum_recovered_command = min(0.20, 0.35 * requested_abs)
    minimum_recovered_velocity = min(0.08, 0.20 * requested_abs)
    return bool(
        mean_applied[-1] >= mean_applied[1] + 0.05
        and mean_applied[-1] >= minimum_recovered_command
        and mean_risk[-1] <= mean_risk[1] - 0.03
        and mean_vx[-1] >= mean_vx[1] + 0.03
        and mean_vx[-1] >= minimum_recovered_velocity
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-onnx", type=Path, required=True)
    parser.add_argument("--risk-onnx", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "unitree_robots/g1/scene_29dof.xml",
    )
    parser.add_argument("--duration-s", type=float, default=9.0)
    parser.add_argument("--command-ramp-s", type=float, default=1.0)
    parser.add_argument("--command", type=float, nargs=3, default=(0.6, 0.0, 0.0))
    parser.add_argument(
        "--friction-sequence", type=float, nargs="+", default=(0.8, 0.2, 0.8)
    )
    parser.add_argument("--randomized-hall", action="store_true")
    parser.add_argument(
        "--disable-governor",
        action="store_true",
        help="Offline active-probe collection only; never use for deployment safety tests.",
    )
    parser.add_argument("--governor-low-speed", type=float, default=0.22)
    parser.add_argument("--governor-high-speed", type=float, default=0.60)
    parser.add_argument(
        "--governor-low-reprobe-s",
        type=float,
        default=10.0,
        help="Bounded LOW-state Hall re-probe interval.",
    )
    parser.add_argument("--governor-low-probability", type=float, default=0.65)
    parser.add_argument("--governor-high-probability", type=float, default=0.55)
    parser.add_argument("--governor-critical-probability", type=float, default=0.95)
    parser.add_argument("--governor-critical-hold-s", type=float, default=0.04)
    parser.add_argument("--governor-reference-alpha", type=float, default=0.002)
    parser.add_argument("--governor-relative-low-rise", type=float, default=0.20)
    parser.add_argument("--governor-relative-high-drop", type=float, default=0.20)
    parser.add_argument(
        "--governor-low-hold-s",
        type=float,
        default=0.10,
        help="Required duration of non-critical low-traction evidence.",
    )
    parser.add_argument("--governor-high-hold-s", type=float, default=0.10)
    parser.add_argument("--governor-probe-speed", type=float, default=0.50)
    parser.add_argument("--governor-probe-duration-s", type=float, default=1.60)
    parser.add_argument(
        "--governor-probe-relative-clear-drop",
        type=float,
        default=0.20,
        help=(
            "Minimum causal risk decrease from probe start to its latter-half "
            "mean that permits a tentative HIGH state."
        ),
    )
    parser.add_argument("--governor-crawl-pulse-s", type=float, default=0.45)
    parser.add_argument("--governor-accel-rate", type=float, default=1.50)
    parser.add_argument("--governor-decel-rate", type=float, default=1.00)
    parser.add_argument(
        "--fault-mode",
        choices=("none", "full", "left", "right", "bad_channels"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_s <= 0.0 or min(args.friction_sequence) <= 0.0:
        raise ValueError("duration and friction must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    controller = HallOnlyMujocoController(
        model,
        data,
        args.policy_onnx,
        args.risk_onnx,
        seed=args.seed,
        randomized_hall=args.randomized_hall,
        fault_mode=args.fault_mode,
        governor_enabled=not args.disable_governor,
        low_speed_limit=args.governor_low_speed,
        high_speed_limit=args.governor_high_speed,
        low_reprobe_s=args.governor_low_reprobe_s,
        low_probability=args.governor_low_probability,
        high_probability=args.governor_high_probability,
        critical_probability=args.governor_critical_probability,
        critical_hold_s=args.governor_critical_hold_s,
        reference_alpha=args.governor_reference_alpha,
        relative_low_rise=args.governor_relative_low_rise,
        relative_high_drop=args.governor_relative_high_drop,
        low_hold_s=args.governor_low_hold_s,
        high_hold_s=args.governor_high_hold_s,
        probe_speed_limit=args.governor_probe_speed,
        probe_duration_s=args.governor_probe_duration_s,
        probe_relative_clear_drop=args.governor_probe_relative_clear_drop,
        crawl_pulse_s=args.governor_crawl_pulse_s,
        accel_rate=args.governor_accel_rate,
        decel_rate=args.governor_decel_rate,
    )
    controller.reset()
    requested = np.asarray(args.command, dtype=np.float32)
    sequence = tuple(float(value) for value in args.friction_sequence)
    phase_duration = args.duration_s / len(sequence)
    set_friction(model, sequence[0])
    records: dict[str, list[np.ndarray | float | int]] = {}
    decimation = int(round(POLICY_DT_S / PHYSICS_DT))
    fell = False
    nonfinite = 0
    for step in range(int(np.ceil(args.duration_s / POLICY_DT_S))):
        phase = min(int(data.time / phase_duration), len(sequence) - 1)
        if step == 0 or int((data.time - POLICY_DT_S) / phase_duration) != phase:
            set_friction(model, sequence[phase])
        ramp = min(data.time / max(args.command_ramp_s, 1.0e-6), 1.0)
        sample = controller.policy_step(requested * ramp)
        records.setdefault("time_s", []).append(float(data.time))
        records.setdefault("phase", []).append(phase)
        records.setdefault("ground_friction_mu", []).append(sequence[phase])
        records.setdefault("base_height_m", []).append(float(data.qpos[2]))
        for key, value in sample.items():
            records.setdefault("obs" if key == "observation" else key, []).append(value)
            nonfinite += int(np.count_nonzero(~np.isfinite(value)))
        tilt = float(np.linalg.norm(np.asarray(sample["projected_gravity"])[:2]))
        fell |= bool(data.qpos[2] < 0.45 or tilt > 0.80)
        for _ in range(decimation):
            controller.physics_step()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["requested_command"] = np.broadcast_to(
        requested, (len(arrays["time_s"]), 3)
    ).copy()
    arrays["mu"] = arrays["ground_friction_mu"].astype(np.float32)
    arrays["cmd_vx"] = np.full(len(arrays["time_s"]), requested[0], dtype=np.float32)
    arrays["sample_weight"] = np.ones(len(arrays["time_s"]), dtype=np.float32)
    arrays["terminated_fall"] = np.asarray((fell,), dtype=np.bool_)
    arrays["nonfinite_count"] = np.asarray((nonfinite,), dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    phase_masks = [arrays["phase"] == index for index in range(len(sequence))]
    steady_masks = []
    for mask in phase_masks:
        indices = np.flatnonzero(mask)
        steady = np.zeros_like(mask)
        steady[indices[len(indices) * 2 // 3 :]] = True
        steady_masks.append(steady)
    mean_vx = [
        float(arrays["base_velocity"][mask, 0].mean()) for mask in steady_masks
    ]
    mean_applied = [
        float(arrays["applied_command"][mask, 0].mean()) for mask in steady_masks
    ]
    mean_risk = [float(arrays["risk"][mask].mean()) for mask in steady_masks]
    safety_gate = not fell and nonfinite == 0
    adaptive_gate = args.fault_mode != "none" or adaptive_recovery_gate(
        requested[0], sequence, mean_vx, mean_applied, mean_risk
    )
    fault_gate = True
    if args.fault_mode != "none":
        fault_gate = (
            float(np.max(np.abs(arrays["applied_command"]))) <= 1.0e-6
            and float(np.min(arrays["risk"]))
            >= (0.999999 if args.fault_mode == "full" else 0.85)
        )
    overall = safety_gate and adaptive_gate and fault_gate
    summary = {
        "status": "PASS" if overall else "NEEDS_CALIBRATION",
        "measurement_boundary": (
            "policy/risk input contains Hall Bx/By/Bz + proprioception only; "
            "contact force and friction are evaluator truth"
        ),
        "mechanical_model": "local TPU compliance with four embedded magnets per Hall site",
        "seed": args.seed,
        "friction_sequence": sequence,
        "command": requested.tolist(),
        "randomized_hall": args.randomized_hall,
        "governor_enabled": not args.disable_governor,
        "fault_mode": args.fault_mode,
        "fell": fell,
        "nonfinite": nonfinite,
        "minimum_base_height_m": float(arrays["base_height_m"].min()),
        "mean_vx_by_phase": mean_vx,
        "mean_applied_vx_by_phase": mean_applied,
        "mean_risk_by_phase": mean_risk,
        "maximum_abs_hall": float(np.abs(arrays["hall"]).max()),
        "maximum_compression_mm": float(arrays["maximum_compression_m"].max() * 1000.0),
        "gates": {
            "finite_and_no_fall": safety_gate,
            "high_friction_recovery": adaptive_gate,
            "full_fault_fail_safe": fault_gate,
        },
        "trajectory": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
