#!/usr/bin/env python3
"""Record a headless video of the 1864-D R5 policy on the slope/stairs task.

Reads the R5 checkpoint (read-only) and writes per-environment videos into a
fresh directory.  Nothing is exported into the R5 log directory and no file
under the original model is modified: this is a view/recording utility for the
new SlopeStairsV1 task.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=(
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideTransitionRetentionSlopeStairsV1"
))
parser.add_argument(
    "--video_dir",
    type=Path,
    default=Path("/home/mosense/guo/video/isaac_slope_stairs_v1"),
)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--command_vx", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--view",
    action="store_true",
    default=False,
    help="Open the interactive viewer instead of recording a video.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if args_cli.view and getattr(args_cli, "headless", False):
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _force_forward_command(base_env, vx: float) -> None:
    term = base_env.command_manager.get_term("base_velocity")
    is_standing = getattr(term, "is_standing_env", None)
    if is_standing is not None:
        is_standing[:] = False
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[:, 0] = float(vx)
        term.vel_command_b[:, 1:] = 0.0


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=5,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = int(args_cli.seed)
    # The Hall debug point instancers dominate per-frame render cost; the
    # video only needs the robot + terrain, so disable them for speed.
    env_cfg.hall_sensor_cfg.enable_debug_vis = False
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    checkpoint = Path(
        args_cli.checkpoint
        or "/home/mosense/guo/unitree_rl_lab/logs/rsl_rl/"
        "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
        "cadence_stride_transition_retention_r5/"
        "2026-08-13_20-43-49_transition_retention_r5_rebalanced/model_399.pt"
    ).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None if args_cli.view else "rgb_array",
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_dir = args_cli.video_dir.expanduser().resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    if not args_cli.view:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class = resolve_callable(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
    if not isinstance(runner_class, type) or not issubclass(runner_class, OnPolicyRunner):
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
    )
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    _force_forward_command(env.unwrapped, args_cli.command_vx)
    observation = env.get_observations()
    max_steps = int(1 << 30) if args_cli.view else args_cli.video_length
    n_envs = int(env.unwrapped.num_envs)

    def policy_input(obs):
        # Feed only the policy group to the actor.
        return TensorDict({"policy": obs["policy"]}, batch_size=[n_envs])

    for _step in range(max_steps):
        with torch.inference_mode():
            action = policy(policy_input(observation))
        observation, _, done, _extras = env.step(action)
        if not args_cli.view and bool(torch.as_tensor(done).all()):
            break
        if args_cli.view and bool(torch.as_tensor(done).any()):
            observation, _extras = env.reset()

    env.close()
    if not args_cli.view:
        print(f"[record] videos written under {video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
