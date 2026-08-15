#!/usr/bin/env python3
"""Reproducible, finite Isaac Lab rollout for the unchanged G1 proprio baseline.

This entry point is intentionally diagnostic-only.  It never exports or
overwrites a policy and writes all generated records below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from importlib.metadata import version

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Unitree-G1-29dof-Velocity",
    choices=("Unitree-G1-29dof-Velocity",),
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=20260731)
parser.add_argument("--command-vx", type=float, default=0.8)
parser.add_argument("--command-vy", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--disable-fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    BasePPORunnerCfg,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _json_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_stats(value: torch.Tensor) -> dict[str, float | int]:
    finite = torch.isfinite(value)
    clean = torch.nan_to_num(value.detach())
    return {
        "shape": list(value.shape),
        "nonfinite": int((~finite).sum().item()),
        "min": _json_float(clean.min()),
        "max": _json_float(clean.max()),
        "mean": _json_float(clean.mean()),
    }


def main() -> int:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    action_dim = int(base_env.action_manager.total_action_dim)
    if action_dim != 29:
        raise RuntimeError(f"Expected the preserved 29-D action, got {action_dim}")

    wrapped_env = RslRlVecEnvWrapper(env)
    runner_cfg = handle_deprecated_rsl_rl_cfg(
        BasePPORunnerCfg(), version("rsl-rl-lib")
    )
    runner_cfg.seed = args_cli.seed
    runner = OnPolicyRunner(
        wrapped_env, runner_cfg.to_dict(), log_dir=None, device=base_env.device
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=base_env.device)

    cmd_term = base_env.command_manager.get_term("base_velocity")
    fixed_command = torch.tensor(
        [args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw],
        device=base_env.device,
        dtype=torch.float32,
    )
    cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_term.cfg.rel_standing_envs = 0.0
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False

    robot = base_env.scene["robot"]
    policy_dim = int(base_env.observation_manager.group_obs_dim["policy"][-1])
    critic_dim = int(base_env.observation_manager.group_obs_dim["critic"][-1])
    policy_terms = list(base_env.observation_manager.active_terms["policy"])
    critic_terms = list(base_env.observation_manager.active_terms["critic"])
    policy_term_dims = [
        [int(value) for value in dim]
        for dim in base_env.observation_manager.group_obs_term_dim["policy"]
    ]
    critic_term_dims = [
        [int(value) for value in dim]
        for dim in base_env.observation_manager.group_obs_term_dim["critic"]
    ]

    obs = wrapped_env.get_observations()
    trajectories: dict[str, list[np.ndarray]] = {
        "command": [],
        "base_velocity_body": [],
        "base_angular_velocity_body": [],
        "projected_gravity_body": [],
        "root_height": [],
        "joint_position": [],
        "joint_velocity": [],
        "action": [],
        "reward": [],
        "done": [],
    }
    obs_nonfinite = 0
    critic_nonfinite = 0
    action_nonfinite = 0
    action_saturated = 0
    total_actions = 0
    previous_action: torch.Tensor | None = None
    action_delta_l2: list[torch.Tensor] = []

    for _ in range(args_cli.steps):
        cmd_term.vel_command_b[:] = fixed_command
        with torch.inference_mode():
            actor_obs = obs["policy"]
            critic_obs = obs["critic"]
            actions = policy(obs)
            obs_nonfinite += int((~torch.isfinite(actor_obs)).sum().item())
            critic_nonfinite += int((~torch.isfinite(critic_obs)).sum().item())
            action_nonfinite += int((~torch.isfinite(actions)).sum().item())
            action_saturated += int((actions.abs() >= 0.999).sum().item())
            total_actions += int(actions.numel())
            if previous_action is not None:
                action_delta_l2.append(
                    torch.linalg.vector_norm(actions - previous_action, dim=-1)
                )
            previous_action = actions.clone()
            obs, reward, done, _ = wrapped_env.step(actions)

        trajectories["command"].append(
            cmd_term.vel_command_b.detach().cpu().numpy().copy()
        )
        trajectories["base_velocity_body"].append(
            robot.data.root_lin_vel_b.detach().cpu().numpy().copy()
        )
        trajectories["base_angular_velocity_body"].append(
            robot.data.root_ang_vel_b.detach().cpu().numpy().copy()
        )
        trajectories["projected_gravity_body"].append(
            robot.data.projected_gravity_b.detach().cpu().numpy().copy()
        )
        trajectories["root_height"].append(
            robot.data.root_pos_w[:, 2].detach().cpu().numpy().copy()
        )
        trajectories["joint_position"].append(
            robot.data.joint_pos.detach().cpu().numpy().copy()
        )
        trajectories["joint_velocity"].append(
            robot.data.joint_vel.detach().cpu().numpy().copy()
        )
        trajectories["action"].append(actions.detach().cpu().numpy().copy())
        trajectories["reward"].append(reward.detach().cpu().numpy().copy())
        trajectories["done"].append(done.detach().cpu().numpy().copy())

    arrays = {key: np.stack(value, axis=0) for key, value in trajectories.items()}
    np.savez_compressed(output_dir / "baseline_trajectory.npz", **arrays)

    velocity = torch.from_numpy(arrays["base_velocity_body"])
    command = torch.from_numpy(arrays["command"])
    gravity = torch.from_numpy(arrays["projected_gravity_body"])
    height = torch.from_numpy(arrays["root_height"])
    done = torch.from_numpy(arrays["done"]).bool()
    reward = torch.from_numpy(arrays["reward"])
    actions = torch.from_numpy(arrays["action"])
    warmup = min(50, max(0, args_cli.steps // 5))
    eval_slice = slice(warmup, None)
    xy_error = torch.linalg.vector_norm(
        velocity[eval_slice, :, :2] - command[eval_slice, :, :2], dim=-1
    )
    yaw_error = (
        torch.from_numpy(arrays["base_angular_velocity_body"])[
            eval_slice, :, 2
        ]
        - command[eval_slice, :, 2]
    ).abs()
    orientation_error = torch.linalg.vector_norm(
        gravity[eval_slice, :, :2], dim=-1
    )
    commanded_vx = command[eval_slice, :, 0]
    vx_ratio_mask = commanded_vx.abs() > 1.0e-4
    vx_ratio = (
        velocity[eval_slice, :, 0][vx_ratio_mask]
        / commanded_vx[vx_ratio_mask]
    )
    action_delta = (
        torch.cat(action_delta_l2)
        if action_delta_l2
        else torch.zeros(1, dtype=torch.float32)
    )
    body_masses = robot.root_physx_view.get_masses().detach().cpu()
    default_joint_position = robot.data.default_joint_pos[0].detach().cpu()
    default_joint_stiffness = robot.data.default_joint_stiffness[0].detach().cpu()
    default_joint_damping = robot.data.default_joint_damping[0].detach().cpu()

    results = {
        "status": "passed" if obs_nonfinite + critic_nonfinite + action_nonfinite == 0 else "failed",
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "physics_dt_s": float(base_env.physics_dt),
        "policy_dt_s": float(base_env.step_dt),
        "decimation": int(env_cfg.decimation),
        "action_dimension": action_dim,
        "actor_observation_dimension": policy_dim,
        "critic_observation_dimension": critic_dim,
        "actor_terms": [
            {"name": name, "flattened_shape": dim}
            for name, dim in zip(policy_terms, policy_term_dims)
        ],
        "critic_terms": [
            {"name": name, "flattened_shape": dim}
            for name, dim in zip(critic_terms, critic_term_dims)
        ],
        "joint_order": list(robot.joint_names),
        "body_order": list(robot.body_names),
        "body_mass_kg_env0": {
            name: float(value)
            for name, value in zip(robot.body_names, body_masses[0].tolist())
        },
        "total_robot_mass_kg": {
            "min": float(body_masses.sum(dim=1).min()),
            "max": float(body_masses.sum(dim=1).max()),
            "mean": float(body_masses.sum(dim=1).mean()),
        },
        "default_joint_position_rad": {
            name: float(value)
            for name, value in zip(robot.joint_names, default_joint_position.tolist())
        },
        "pd_stiffness": {
            name: float(value)
            for name, value in zip(robot.joint_names, default_joint_stiffness.tolist())
        },
        "pd_damping": {
            name: float(value)
            for name, value in zip(robot.joint_names, default_joint_damping.tolist())
        },
        "command": fixed_command.detach().cpu().tolist(),
        "nonfinite": {
            "actor_observations": obs_nonfinite,
            "critic_observations": critic_nonfinite,
            "actions": action_nonfinite,
        },
        "termination_count": int(done.sum().item()),
        "environments_with_termination": int(done.any(dim=0).sum().item()),
        "mean_reward": _json_float(reward.mean()),
        "velocity_tracking_error_xy_mean_m_s": _json_float(xy_error.mean()),
        "velocity_tracking_error_xy_max_m_s": _json_float(xy_error.max()),
        "yaw_tracking_error_mean_rad_s": _json_float(yaw_error.mean()),
        "actual_vx_mean_m_s": _json_float(velocity[eval_slice, :, 0].mean()),
        "actual_over_command_vx_mean": _json_float(vx_ratio.mean()),
        "orientation_error_xy_mean": _json_float(orientation_error.mean()),
        "root_height_mean_m": _json_float(height[eval_slice].mean()),
        "root_height_min_m": _json_float(height.min()),
        "action_abs_max": _json_float(actions.abs().max()),
        "action_saturation_fraction_at_abs_ge_0_999": (
            float(action_saturated) / float(max(total_actions, 1))
        ),
        "action_delta_l2_mean": _json_float(action_delta.mean()),
        "initial_actor_observation": _tensor_stats(obs["policy"]),
        "initial_critic_observation": _tensor_stats(obs["critic"]),
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_snapshot = {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "physics_dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "policy_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "episode_length_s": float(env_cfg.episode_length_s),
        "robot_asset": str(env_cfg.scene.robot.spawn.usd_path),
        "action": {
            "type": "joint_position",
            "dimension": action_dim,
            "scale": float(env_cfg.actions.JointPositionAction.scale),
            "use_default_offset": bool(
                env_cfg.actions.JointPositionAction.use_default_offset
            ),
            "joint_order": list(robot.joint_names),
        },
        "observations": {
            "actor_dimension": policy_dim,
            "critic_dimension": critic_dim,
            "history_order": "term-major; oldest-to-newest within each term",
            "actor_terms": results["actor_terms"],
            "critic_terms": results["critic_terms"],
        },
        "command": {
            "fixed_vx_m_s": args_cli.command_vx,
            "fixed_vy_m_s": args_cli.command_vy,
            "fixed_yaw_rad_s": args_cli.command_yaw,
        },
        "checkpoint": str(checkpoint),
    }
    (output_dir / "env_cfg.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)

    wrapped_env.close()
    simulation_app.close()
    return 0 if results["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
