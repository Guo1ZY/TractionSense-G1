# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--viewer_eye",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Override the Isaac viewer camera position.",
)
parser.add_argument(
    "--viewer_lookat",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Override the Isaac viewer camera target.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Load the checkpoint, export ONNX/JIT, and exit before the play loop.",
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Teleop base velocity with keyboard (W/S/A/D or arrows). Overrides random velocity commands.",
)
parser.add_argument(
    "--vx_max",
    type=float,
    default=1.0,
    help="Max forward speed (m/s) when using --keyboard (default 1.0, match training limit).",
)
parser.add_argument(
    "--vy_max",
    type=float,
    default=0.3,
    help="Max lateral speed (m/s) when using --keyboard.",
)
parser.add_argument(
    "--wz_max",
    type=float,
    default=0.2,
    help="Max yaw rate (rad/s) when using --keyboard.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# keyboard teleop needs a visible window (not headless)
if args_cli.keyboard and getattr(args_cli, "headless", False):
    print("[WARN] --keyboard requires a GUI window; disabling headless.")
    args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

# Isaac Lab 2.3.x: pretrained helper path differs across minor versions
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    try:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ImportError:
        try:
            from isaaclab.utils.assets import get_published_pretrained_checkpoint
        except ImportError:

            def get_published_pretrained_checkpoint(*args, **kwargs):
                return None


from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
    if args_cli.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Compat with rsl-rl-lib 2.3.x config fields
    installed_version = version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit (API differs across rsl-rl versions)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    installed_version = version("rsl-rl-lib")
    from packaging import version as pkg_version

    if pkg_version.parse(installed_version) >= pkg_version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.export_only:
        print(f"[INFO] Export-only complete: {export_model_dir}")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # optional keyboard teleop → override base_velocity command each step
    keyboard = None
    if args_cli.keyboard:
        import numpy as np
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args_cli.vx_max,
                v_y_sensitivity=args_cli.vy_max,
                omega_z_sensitivity=args_cli.wz_max,
                sim_device=env.unwrapped.device,
            )
        )
        # Map WASD as well as default arrows/numpad
        keyboard._INPUT_KEY_MAPPING.update(
            {
                "W": np.asarray([1.0, 0.0, 0.0]) * args_cli.vx_max,
                "S": np.asarray([-1.0, 0.0, 0.0]) * args_cli.vx_max,
                "A": np.asarray([0.0, 1.0, 0.0]) * args_cli.vy_max,
                "D": np.asarray([0.0, -1.0, 0.0]) * args_cli.vy_max,
                "Q": np.asarray([0.0, 0.0, 1.0]) * args_cli.wz_max,
                "E": np.asarray([0.0, 0.0, -1.0]) * args_cli.wz_max,
            }
        )
        # stop auto random resampling so keyboard fully owns the command
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
        cmd_term.cfg.rel_standing_envs = 0.0
        cmd_term.is_standing_env[:] = False
        print("[INFO] Keyboard teleop enabled.")
        print(keyboard)
        print("\tAlso: W/S forward-back, A/D strafe, Q/E yaw, L reset cmd")
        print(f"\tSpeed scale: vx_max={args_cli.vx_max}, vy_max={args_cli.vy_max}, wz_max={args_cli.wz_max}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if keyboard is not None:
                # (vx, vy, wz) from keyboard → all envs
                cmd = keyboard.advance()  # shape (3,)
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                cmd_term.vel_command_b[:] = cmd.unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                # clamp to training-friendly limits
                cmd_term.vel_command_b[:, 0].clamp_(-args_cli.vx_max, args_cli.vx_max)
                cmd_term.vel_command_b[:, 1].clamp_(-args_cli.vy_max, args_cli.vy_max)
                cmd_term.vel_command_b[:, 2].clamp_(-args_cli.wz_max, args_cli.wz_max)

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if pkg_version.parse(installed_version) >= pkg_version.parse("4.0.0"):
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            elif policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time or args_cli.keyboard:
            if sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
