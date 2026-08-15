#!/usr/bin/env python3
"""Collect native G1 state, analytical estimates, and Isaac truth to NPZ."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=20260803)
parser.add_argument("--checkpoint", type=Path, default=Path("model/rl/model_49999.pt"))
parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/dataset_stage0.npz"))
parser.add_argument("--randomization_stage", type=int, choices=range(6), default=0)
parser.add_argument("--benchmark_latency", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion.agents.torque_traction_rsl_cfg import TorqueTractionStudentRunnerCfg  # noqa: E402
from unitree_rl_lab.traction.isaac_observations import _true_force_local_n  # noqa: E402
from unitree_rl_lab.traction_torque.isaac_teacher import torque_teacher_observation  # noqa: E402
from unitree_rl_lab.traction_torque.teacher_schema import (  # noqa: E402
    TORQUE_TEACHER_FRAME_DIM,
    TORQUE_TEACHER_HISTORY_FRAMES,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.utils.partial_checkpoint import load_partial_into_runner  # noqa: E402


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task = "Unitree-G1-29dof-Velocity-TorqueTractionStudent"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs, entry_point_key="play_env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.observations.policy.torque_traction_frame.params["randomization_stage"] = args.randomization_stage
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    base.torque_estimator_benchmark_synchronize = args.benchmark_latency
    torque_cfg = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_torque_traction_student_env_cfg"
    )
    left, right, feet = (
        torque_cfg.TORQUE_LEFT_FOOT_SENSOR_CFG.copy(),
        torque_cfg.TORQUE_RIGHT_FOOT_SENSOR_CFG.copy(),
        torque_cfg.TORQUE_FOOT_FORCE_ASSET_CFG.copy(),
    )
    for entity in (left, right, feet):
        entity.resolve(base.scene)
    wrapped = RslRlVecEnvWrapper(env)
    runner_cfg = handle_deprecated_rsl_rl_cfg(TorqueTractionStudentRunnerCfg(), version("rsl-rl-lib"))
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=base.device)
    load_stats = load_partial_into_runner(runner, str(args.checkpoint.resolve()), device=base.device, verbose=False)
    policy = runner.get_inference_policy(device=base.device)
    observation = wrapped.get_observations()
    teacher_history = torch.zeros(
        args.num_envs,
        TORQUE_TEACHER_HISTORY_FRAMES,
        TORQUE_TEACHER_FRAME_DIM,
        device=base.device,
    )
    records: dict[str, list[np.ndarray]] = {name: [] for name in (
        "timestamp_s", "environment_id", "q", "dq", "qdd_filtered", "tau_est_nm",
        "imu_linear_acceleration_m_s2", "foot_position_w_m", "foot_velocity_w_m_s",
        "estimated_force_local_n", "true_force_local_n", "contact_probability",
        "true_contact", "force_confidence", "residual_norm_nm", "condition_score",
        "traction_utilization", "slip_probability", "traction_margin", "friction_lower_bound",
        "slip_event_mu_estimate", "slip_state", "ground_friction_mu", "base_velocity_b",
        "command", "action", "student_history", "teacher_history", "terminated", "truncated", "estimator_latency_ms",
    )}
    for step in range(args.warmup_steps + args.steps):
        with torch.inference_mode():
            action = policy(observation)
        observation, reward, terminated, info = wrapped.step(action)
        truncated = info.get("time_outs", torch.zeros_like(terminated)) if isinstance(info, dict) else torch.zeros_like(terminated)
        teacher_history[terminated] = 0.0
        teacher_history = torch.roll(teacher_history, shifts=-1, dims=1)
        teacher_history[:, -1] = torque_teacher_observation(base, left, right, feet)
        if step < args.warmup_steps:
            continue
        packet = base._isaac_torque_traction_state.packet
        truth = _true_force_local_n(base, left, right, feet)
        truth_contact = truth.reshape(args.num_envs, 2, 3)[..., 2].abs() > 10.0
        robot = base.scene["robot"]
        mu = getattr(base, "ground_friction_mu_buf", torch.full((args.num_envs, 2), float("nan"), device=base.device))
        if mu.ndim == 1:
            mu = mu[:, None].expand(-1, 2)
        values = {
            "timestamp_s": torch.full((args.num_envs, 1), (step - args.warmup_steps) * base.step_dt, device=base.device),
            "environment_id": torch.arange(args.num_envs, device=base.device)[:, None],
            "q": robot.data.joint_pos, "dq": robot.data.joint_vel,
            "qdd_filtered": base._isaac_torque_traction_state.filter.acceleration,
            "tau_est_nm": packet.tau_est_nm,
            "imu_linear_acceleration_m_s2": packet.imu_linear_acceleration_m_s2,
            "foot_position_w_m": robot.data.body_pos_w[:, list(base._isaac_torque_traction_state.foot_body_ids)],
            "foot_velocity_w_m_s": robot.data.body_lin_vel_w[:, list(base._isaac_torque_traction_state.foot_body_ids)],
            "estimated_force_local_n": packet.analytical_force_local_n,
            "true_force_local_n": truth,
            "contact_probability": packet.contact_probability,
            "true_contact": truth_contact,
            "force_confidence": packet.force_confidence,
            "residual_norm_nm": packet.residual_norm_nm,
            "condition_score": packet.condition_score,
            "traction_utilization": packet.traction_utilization,
            "slip_probability": packet.slip_probability,
            "traction_margin": packet.traction_margin,
            "friction_lower_bound": packet.friction_lower_bound,
            "slip_event_mu_estimate": packet.slip_event_mu_estimate,
            "slip_state": packet.slip_state,
            "ground_friction_mu": mu,
            "base_velocity_b": robot.data.root_lin_vel_b,
            "command": base.command_manager.get_command("base_velocity"),
            "action": action,
            "student_history": observation["policy"].reshape(args.num_envs, 15, 125),
            "teacher_history": teacher_history,
            "terminated": terminated[:, None], "truncated": truncated[:, None],
            "estimator_latency_ms": torch.full((args.num_envs, 1), base._isaac_torque_traction_state.last_compute_latency_ms, device=base.device),
        }
        for name, value in values.items():
            records[name].append(value.detach().cpu().numpy())
    output = {name: np.stack(chunks, axis=0) for name, chunks in records.items()}
    output["metadata"] = np.asarray({
        "task": task, "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps,
        "policy_dt_s": base.step_dt, "randomization_stage": args.randomization_stage,
        "checkpoint": str(args.checkpoint.resolve()), "load_stats": load_stats,
        "force_order": ["L_Fx", "L_Fy", "L_Fz", "R_Fx", "R_Fy", "R_Fz"],
        "force_frame": "+x toe, +y robot-left, +z up; matching ankle_roll_link local",
        "foot_velocity_is_contact_point_truth": False,
        "teacher_history_is_policy_input": False,
        "teacher_history_shape": [TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_FRAME_DIM],
    }, dtype=object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print({"output": str(args.output.resolve()), "samples": args.steps * args.num_envs, "nonfinite_estimated_force": int((~np.isfinite(output["estimated_force_local_n"])).sum())})
    wrapped.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
