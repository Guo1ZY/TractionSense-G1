#!/usr/bin/env python3
"""Per-terrain acceptance eval + focused headless videos for SlopeStairsV1.

Runs the final SlopeStairsV1 model in the five-terrain Play task (flat,
slope_up, slope_down, stairs_up, stairs_down).  First it reports, for each
terrain, how many policy steps it survived and why it terminated.  Then it
records one focused video per terrain, with the camera following that
environment, so the failure mode is directly inspectable.
"""

from __future__ import annotations

import argparse
import math
from importlib.metadata import version
from pathlib import Path

import imageio
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

TASK = (
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideTransitionRetentionSlopeStairsV1"
)
TERRAINS = ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down")
VIEWS = {
    "flat": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "slope_up": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "slope_down": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "stairs_up": ((6.0, -4.0, 2.0), (3.5, 0.0, 0.5)),
    "stairs_down": ((6.0, -4.0, 2.0), (3.5, 0.0, 0.5)),
}
BAD_ORIENTATION_LIMIT = 0.8  # matches TerminationsCfg.bad_orientation
MIN_BASE_HEIGHT = 0.2  # matches TerminationsCfg.base_height

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=TASK)
parser.add_argument(
    "--video_dir",
    type=Path,
    default=Path("/home/mosense/guo/video/isaac_slope_stairs_v1_per_env"),
)
parser.add_argument("--video_length", type=int, default=800)
parser.add_argument("--command_vx", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip_videos", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

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


def _roll_pitch_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _parse_env(render_mode: str):
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=5,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.hall_sensor_cfg.enable_debug_vis = False
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (1280, 720)
    return env_cfg, gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)


def _load_policy(env):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    checkpoint = Path(
        args_cli.checkpoint
        or (
            "/home/mosense/guo/unitree_rl_lab/scripts/rsl_rl/logs/rsl_rl/"
            "unitree_g1_29dof_velocity_foot_traction_hall_spatial_"
            "cadence_stride_transition_retention_slope_stairs_v1/"
        )
    ).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = sorted(checkpoint.glob("*/model_*.pt"))[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_class = resolve_callable(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
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
    return env, policy, getattr(agent_cfg, "clip_actions", None)


def _policy_input(obs, n_envs):
    return TensorDict({"policy": obs["policy"]}, batch_size=[n_envs])


def _fix_spawn(env, yaw: float = 0.0) -> None:
    """Align every robot to a fixed heading before a rollout.

    The training reset samples yaw uniformly, which makes world-frame
    displacement meaningless for a per-terrain traversal test.  Keep the
    sampled center spawn position, only zero the heading/velocities.
    """
    asset = env.unwrapped.scene["robot"]
    root = asset.data.root_state_w.clone()
    root[:, 3:7] = torch.tensor(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        device=root.device,
        dtype=root.dtype,
    )
    root[:, 7:13] = 0.0
    asset.write_root_state_to_sim(root)


def run_stats(env, policy) -> None:
    _fix_spawn(env)
    _force_forward_command(env.unwrapped, args_cli.command_vx)
    observation = env.get_observations()
    initial_pos = env.unwrapped.scene["robot"].data.root_state_w[:, :3].clone()
    max_steps = int(args_cli.video_length)
    n_envs = int(env.unwrapped.num_envs)
    done_step = torch.full((n_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
    cause = [""] * n_envs
    term_names = list(env.unwrapped.termination_manager._term_names)
    for step in range(max_steps):
        root_prev = env.unwrapped.scene["robot"].data.root_state_w.clone()
        with torch.inference_mode():
            action = policy(_policy_input(observation, n_envs))
        observation, _, done, _extras = env.step(action)
        d = torch.as_tensor(done, device=env.unwrapped.device)
        last_dones = env.unwrapped.termination_manager._last_episode_dones
        for i in range(n_envs):
            if done_step[i] >= 0:
                continue
            if not bool(d[i]):
                continue
            done_step[i] = step
            fired = [
                name
                for j, name in enumerate(term_names)
                if bool(last_dones[i, j])
            ]
            cause[i] = ",".join(fired) or "unknown"
        if bool(d.all()):
            break
    print(
        "\n[stats] per-terrain rollout, cmd=%.2f m/s, seed=%d"
        % (args_cli.command_vx, args_cli.seed)
    )
    mu = env.unwrapped.ground_friction_mu_buf
    print(f"[stats] {'terrain':<12}{'steps':>8}{'cause':<22}{'dx(m)':>8}{'mu':>7}")
    for i, name in enumerate(TERRAINS):
        steps = int(done_step[i]) if done_step[i] >= 0 else max_steps
        root = env.unwrapped.scene["robot"].data.root_state_w[i]
        dx = float(root[0] - initial_pos[i, 0])
        reason = cause[i] if done_step[i] >= 0 else "time_out/survived"
        print(f"[stats] {name:<12}{steps:>8}{reason:<22}{dx:>8.2f}{float(mu[i]):>7.2f}")


def run_video(env, policy, terrain: str) -> None:
    """Record one focused video for ``terrain`` on the existing env instance."""
    focus = TERRAINS.index(terrain)
    controller = env.unwrapped.viewport_camera_controller
    controller.set_view_env_index(focus)
    eye, lookat = VIEWS[terrain]
    controller.update_view_location(eye=eye, lookat=lookat)
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    _fix_spawn(env)
    _force_forward_command(env.unwrapped, args_cli.command_vx)
    observation = env.get_observations()
    n_envs = int(env.unwrapped.num_envs)
    video_dir = args_cli.video_dir.expanduser().resolve() / terrain
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{terrain}_seed{args_cli.seed}_vx{args_cli.command_vx}.mp4"
    writer = imageio.get_writer(
        video_path, fps=50, macro_block_size=None, codec="libx264", quality=7
    )
    try:
        for _step in range(args_cli.video_length):
            with torch.inference_mode():
                action = policy(_policy_input(observation, n_envs))
            observation, _, done, _extras = env.step(action)
            frame = env.unwrapped.render()
            if isinstance(frame, list):
                frame = frame[focus]
            writer.append_data(frame)
            if bool(torch.as_tensor(done)[focus]):
                break
    finally:
        writer.close()
    print(f"[record] {terrain} video written to {video_path}")


def main() -> int:
    env_cfg, raw_env = _parse_env("rgb_array")
    env, policy, clip_actions = _load_policy(raw_env)
    run_stats(env, policy)
    if not args_cli.skip_videos:
        for terrain in TERRAINS:
            run_video(env, policy, terrain)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
