#!/usr/bin/env python3
"""Capture one clean PNG of a slope/stairs terrain column for paper figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

TASK = (
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideTransitionRetentionSlopeStairsV1"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--shot",
    required=True,
    choices=["flat", "slope_up", "slope_down", "stairs_up", "stairs_down"],
)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from PIL import Image  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

COLUMN = {
    "flat": 0,
    "slope_up": 1,
    "slope_down": 2,
    "stairs_up": 3,
    "stairs_down": 4,
}
VIEWS = {
    "flat": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "slope_up": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "slope_down": ((6.5, -5.0, 2.6), (3.0, 0.0, 0.4)),
    "stairs_up": ((6.0, -4.0, 2.0), (3.5, 0.0, 0.5)),
    "stairs_down": ((6.0, -4.0, 2.0), (3.5, 0.0, 0.5)),
}


def main() -> int:
    env_cfg = parse_env_cfg(
        TASK,
        device="cuda:0",
        num_envs=5,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = 42
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = COLUMN[args.shot]
    eye, lookat = VIEWS[args.shot]
    env_cfg.viewer.eye = eye
    env_cfg.viewer.lookat = lookat
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.hall_sensor_cfg.enable_debug_vis = False

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    for _ in range(args.steps):
        env.step(env.action_space.sample())
    # warm up the renderer, then keep the final image
    rgb = None
    for _ in range(5):
        rgb = env.render()
    if rgb is None or rgb.size == 0:
        raise RuntimeError("renderer returned an empty image")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.out)
    print(f"[shot] saved {args.out} ({rgb.shape})")
    # Kit shutdown after saving is flaky in this build; the PNG is already
    # flushed, so exit directly instead of hanging in env.close().
    os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
