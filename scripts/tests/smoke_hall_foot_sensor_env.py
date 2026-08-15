#!/usr/bin/env python3
"""Small current-stack smoke test for the Hall-enabled G1 manager environment."""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Unitree-G1-29dof-Velocity-Foot-TractionMagneticStudent",
)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--enable_debug_vis", action="store_true")
parser.add_argument(
    "--detailed_contact",
    action="store_true",
    help="exercise raw PhysX contact-patch/friction-anchor Hall distribution",
)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument(
    "--full_task",
    action="store_true",
    help="also construct the large training reward/teacher/event stack",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
faulthandler.dump_traceback_later(45.0, repeat=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.sensors import sync_hall_sensor_cfg_to_policy_terms  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _clear_manager_terms(manager_cfg) -> None:
    """Keep the manager's valid config object but disable each term."""
    for name in tuple(manager_cfg.__dict__):
        if not name.startswith("_"):
            setattr(manager_cfg, name, None)


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("num_envs and steps must be positive")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.hall_sensor_cfg.enable_debug_vis = args_cli.enable_debug_vis
    if args_cli.detailed_contact:
        env_cfg.hall_sensor_cfg.contact_distribution_mode = "detailed"
    sync_hall_sensor_cfg_to_policy_terms(env_cfg.observations, env_cfg.hall_sensor_cfg)
    if not args_cli.full_task:
        # This is an integration test for scene/contact/Hall/policy wiring.
        # The large teacher, reward, curriculum and DR stack is orthogonal and
        # substantially slows a cold Isaac launch on CPU.
        env_cfg.observations.critic = None
        env_cfg.observations.teacher = None
        for manager_cfg in (
            env_cfg.events,
            env_cfg.rewards,
            env_cfg.terminations,
            env_cfg.curriculum,
        ):
            _clear_manager_terms(manager_cfg)
    print("[hall-smoke] constructing environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    print("[hall-smoke] environment ready", flush=True)
    actions = torch.zeros(
        (base_env.num_envs, base_env.action_manager.total_action_dim),
        device=base_env.device,
    )
    observation, _ = env.reset()
    print("[hall-smoke] reset complete", flush=True)
    if torch.cuda.is_available() and str(base_env.device).startswith("cuda"):
        torch.cuda.synchronize(base_env.device)
    rollout_start = time.perf_counter()
    for step in range(args_cli.steps):
        observation, _, _, _, _ = env.step(actions)
        if step == 0:
            print("[hall-smoke] first step complete", flush=True)
    if torch.cuda.is_available() and str(base_env.device).startswith("cuda"):
        torch.cuda.synchronize(base_env.device)
    rollout_elapsed_s = time.perf_counter() - rollout_start

    assert observation is not None
    sensor = getattr(base_env, "_hall_foot_sensor", None)
    if sensor is None:
        raise RuntimeError("Hall observation did not create the HallFootSensor")
    raw = sensor.get_raw_data()
    filtered = sensor.get_filtered_data()
    debug = sensor.get_debug_data()
    expected = (base_env.num_envs, 2, env_cfg.hall_sensor_cfg.num_hall_sensors, 3)
    assert raw.shape == expected, (raw.shape, expected)
    assert filtered.shape == expected, (filtered.shape, expected)
    assert torch.isfinite(raw).all() and torch.isfinite(filtered).all()
    assert debug["local_deformation"].shape == (*expected[:-1], 6)
    assert debug["mechanical_driver_force_privileged"].shape == expected
    policy_dim = base_env.observation_manager.group_obs_dim["policy"][-1]
    assert policy_dim == 1864, policy_dim
    print(
        {
            "task": args_cli.task,
            "num_envs": base_env.num_envs,
            "policy_dim": policy_dim,
            "hall_shape": tuple(raw.shape),
            "contact_distribution_mode": env_cfg.hall_sensor_cfg.contact_distribution_mode,
            "rollout_elapsed_s": rollout_elapsed_s,
            "policy_steps_per_s": args_cli.steps / max(rollout_elapsed_s, 1.0e-9),
            "env_steps_per_s": (
                args_cli.steps * base_env.num_envs / max(rollout_elapsed_s, 1.0e-9)
            ),
            "foot_order": ["left_foot", "right_foot"],
            "raw_range_T": [float(raw.min()), float(raw.max())],
            "filtered_range_T": [float(filtered.min()), float(filtered.max())],
            "max_compression_m": float(debug["local_deformation"][..., 2].max()),
            "valid_fraction": float(debug["valid_mask"].float().mean()),
            "deformable_embedding_rest_error_m": (
                float(base_env._hall_deformable_sole_adapter.embedding_rest_error_m)
                if hasattr(base_env, "_hall_deformable_sole_adapter")
                else None
            ),
        },
        flush=True,
    )
    env.close()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Immediate Kit teardown can terminate the interpreter before Python's
        # default exception hook runs, so emit the diagnostic first.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Sim 5.1 can deadlock during graceful CUDA teardown; this standalone
    # test owns the process, so immediate cleanup is safe.
    simulation_app.close(skip_cleanup=True)
