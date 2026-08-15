#!/usr/bin/env python3
"""Action-level H->L comparison on the 0.8 m/s mu=0.28 course.

For one policy, record the first-episode per-joint actions and per-foot stance
fraction in HIGH_START vs LOW.  This answers whether a "mild" friction change
produces hidden joint-level adaptation even when coarse gait parameters (step
length/cadence/speed) barely move.  Supports both the trained plain 480-D actor
and the R5 1864-D FastBase actor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

import cli_args


TASK = (
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideLongDemo"
)
ACTION_DIM = 29
REGION_NAMES = ("high_start", "low", "high_end")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=450)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--low_patch_mu", type=float, default=0.28)
    parser.add_argument("--command_vx", type=float, default=0.80)
    parser.add_argument("--floor_width_m", type=float, default=30.0)
    parser.add_argument("--metric_warmup_steps", type=int, default=100)
    parser.add_argument("--proprio_checkpoint", type=Path, default=None)
    parser.add_argument("--rsl_rl_cfg_entry_point", type=str, default=None)
    parser.add_argument("--summary_json", type=Path, required=True)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
if not ((args_cli.proprio_checkpoint is None) ^ (args_cli.checkpoint is None)):
    raise SystemExit("provide exactly one of --proprio_checkpoint / --checkpoint")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.traction.proprio_baseline import load_proprio_baseline  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

EVAL_ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}


def _build_policy(base_env, wrapped_env):
    if args_cli.proprio_checkpoint is not None:
        path = Path(args_cli.proprio_checkpoint).expanduser().resolve()
        policy = load_proprio_baseline(path, device=base_env.device)

        def call(observation):
            return policy(observation)

        return call, {
            "kind": "plain_480d_proprio_actor",
            "path": str(path),
        }
    path = Path(args_cli.checkpoint).expanduser().resolve()
    if not isinstance(args_cli.rsl_rl_cfg_entry_point, str) or ":" not in (
        args_cli.rsl_rl_cfg_entry_point
    ):
        raise ValueError("--rsl_rl_cfg_entry_point must be MODULE:ClassName")
    agent_cfg = resolve_callable(args_cli.rsl_rl_cfg_entry_point)()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if algorithm_cfg is not None and hasattr(
        algorithm_cfg, "capture_gate_warmup_updates"
    ):
        algorithm_cfg.capture_gate_warmup_updates = 0
    runner_class = resolve_callable(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
    runner = runner_class(
        wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(path), load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG), strict=True)
    policy = runner.get_inference_policy(device=base_env.device)

    def call(observation):
        output = policy(observation)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    return call, {"kind": "rsl_rl_fastbase_1864", "path": str(path)}


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    width = float(args_cli.floor_width_m)
    for attr in ("friction_high_start", "friction_low", "friction_high_end"):
        patch = getattr(env_cfg.scene, attr)
        size = tuple(float(item) for item in patch.spawn.size)
        patch.spawn.size = (size[0], width, size[2])
    env_cfg.scene.friction_low.spawn.physics_material.static_friction = float(
        args_cli.low_patch_mu
    )
    env_cfg.scene.friction_low.spawn.physics_material.dynamic_friction = float(
        args_cli.low_patch_mu
    )
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.stop_fraction = 0.0
    env_cfg.commands.base_velocity.crawl_fraction = 0.0
    env_cfg.commands.base_velocity.cruise_speed_range = (
        float(args_cli.command_vx),
        float(args_cli.command_vx),
    )
    env_cfg.commands.base_velocity.high_speed_range = (
        float(args_cli.command_vx),
        float(args_cli.command_vx),
    )
    if hasattr(env_cfg, "hall_sensor_cfg"):
        env_cfg.hall_sensor_cfg.enable_domain_randomization = False
        env_cfg.hall_sensor_cfg.enable_debug_vis = False

    raw_env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=100.0)
    base_env = env.unwrapped
    policy, policy_meta = _build_policy(base_env, env)

    robot = base_env.scene["robot"]
    n = int(base_env.num_envs)
    left_contact = base_env.scene["left_hall_contact"]
    right_contact = base_env.scene["right_hall_contact"]
    observation = env.get_observations()

    first_episode_active = torch.ones(n, dtype=torch.bool, device=base_env.device)
    env_origin_x = base_env.scene.env_origins[:, 0].detach()
    region = {
        name: {
            "frames": 0,
            "vx_sum": 0.0,
            "action_sum": torch.zeros(ACTION_DIM, device=base_env.device),
            "stance_sum_left": 0.0,
            "stance_sum_right": 0.0,
        }
        for name in REGION_NAMES
    }
    fall_events = 0

    for step in range(int(args_cli.steps)):
        policy_obs = observation["policy"]
        with torch.inference_mode():
            action = policy(observation)
        stage = base_env.spatial_course_stage_buf.long()
        force_l = torch.linalg.vector_norm(
            left_contact.data.net_forces_w[:, 0, :], dim=-1
        )
        force_r = torch.linalg.vector_norm(
            right_contact.data.net_forces_w[:, 0, :], dim=-1
        )
        in_contact_l = force_l > 5.0
        in_contact_r = force_r > 5.0
        local_x = robot.data.root_pos_w[:, 0] - env_origin_x
        mask = first_episode_active & (step >= int(args_cli.metric_warmup_steps))
        for idx, name in enumerate(REGION_NAMES):
            region_mask = mask & (stage == idx)
            count = int(region_mask.sum().item())
            if count == 0:
                continue
            region[name]["frames"] += count
            region[name]["vx_sum"] += float(
                robot.data.root_lin_vel_b[:, 0][region_mask].sum().item()
            )
            region[name]["action_sum"] += action.detach()[region_mask].sum(dim=0)
            region[name]["stance_sum_left"] += float(
                in_contact_l[region_mask].sum().item()
            )
            region[name]["stance_sum_right"] += float(
                in_contact_r[region_mask].sum().item()
            )

        completed_pre = local_x >= 17.5
        observation, _, done, extras = env.step(action)
        done = done.bool()
        timeout = extras.get("time_outs")
        if timeout is None:
            timeout = torch.zeros_like(done)
        else:
            timeout = timeout.to(device=done.device, dtype=torch.bool)
        fall = done & ~timeout & ~completed_pre
        fall_events += int((first_episode_active & fall).sum().item())
        first_episode_active &= ~done
        if not bool(first_episode_active.any().item()):
            break

    output = {}
    for name in REGION_NAMES:
        frames = region[name]["frames"]
        output[name] = {
            "frames": frames,
            "mean_vx_m_s": (
                region[name]["vx_sum"] / frames if frames else None
            ),
            "joint_action_mean": (
                (region[name]["action_sum"] / frames).detach().cpu().tolist()
                if frames
                else None
            ),
            "stance_fraction_left": (
                region[name]["stance_sum_left"] / frames if frames else None
            ),
            "stance_fraction_right": (
                region[name]["stance_sum_right"] / frames if frames else None
            ),
        }
    high = torch.as_tensor(
        output["high_start"]["joint_action_mean"], device=base_env.device
    )
    low = torch.as_tensor(
        output["low"]["joint_action_mean"], device=base_env.device
    )
    delta = (low - high).detach().cpu()
    joint_names = list(robot.data.joint_names)
    ranked = sorted(
        zip(joint_names, delta.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    low_vs_high = {
        "joint_action_delta": delta.tolist(),
        "joint_action_delta_abs_mean": float(delta.abs().mean().item()),
        "joint_action_delta_rms": float(delta.square().mean().sqrt().item()),
        "top_joints_by_abs_delta": ranked[:10],
        "vx_ratio": (
            output["low"]["mean_vx_m_s"] / output["high_start"]["mean_vx_m_s"]
            if output["low"]["mean_vx_m_s"]
            else None
        ),
        "stance_fraction_delta_left": (
            output["low"]["stance_fraction_left"]
            - output["high_start"]["stance_fraction_left"]
        ),
        "stance_fraction_delta_right": (
            output["low"]["stance_fraction_right"]
            - output["high_start"]["stance_fraction_right"]
        ),
    }
    payload = {
        "policy": policy_meta,
        "course": {
            "task": TASK,
            "low_patch_mu": float(args_cli.low_patch_mu),
            "high_patch_mu": 0.90,
            "command_vx_m_s": float(args_cli.command_vx),
            "floor_width_m": width,
        },
        "run": {
            "seed": int(args_cli.seed),
            "num_envs": n,
            "steps": int(args_cli.steps),
            "metric_warmup_steps": int(args_cli.metric_warmup_steps),
        },
        "regions": output,
        "low_vs_high_start": low_vs_high,
        "fall_events": fall_events,
        "joint_names": joint_names,
    }
    summary_path = args_cli.summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(
        f"[action-level] {policy_meta['kind']} frames H={output['high_start']['frames']} "
        f"L={output['low']['frames']} vx_ratio={low_vs_high['vx_ratio']:.3f} "
        f"abs_delta={low_vs_high['joint_action_delta_abs_mean']:.4f} "
        f"falls={fall_events}",
        flush=True,
    )
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
