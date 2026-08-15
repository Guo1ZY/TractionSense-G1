#!/usr/bin/env python3
"""Evaluate the 480-D proprio-only bare G1 policy (model_49999) on the
physical High--Low--High friction course.

The environment remains the standard 1864-D Hall course so the protocol,
floor materials, seeds and episode horizon exactly match the R3/R5 acceptance
suite.  The bare actor consumes only observation columns ``0:480`` (the legacy
proprioceptive block): no Hall, no friction, no contact or privileged course
state ever reaches the policy.  Privileged root state is used exclusively for
evaluation metrics.

Examples
--------
    python scripts/rsl_rl/eval_bare480_spatial_friction_course.py \\
        --checkpoint model/rl/model_49999.pt --seed 450 --num_envs 16 \\
        --steps 3500 --low_patch_mu 0.28 --summary_json out.json --headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

import cli_args


TASK = (
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideLongDemo"
)
LEGACY_DIM = 480
ACTION_DIM = 29
COURSE_SUCCESS_MIN_X = 17.5  # LongDemo authored success line
DEFAULT_CHECKPOINT = (
    "/home/mosense/guo/unitree_rl_lab/model/rl/model_49999.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=450)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument(
        "--steps",
        type=int,
        default=3500,
        help="Policy steps; 3500 is 70 s at the audited 50 Hz policy rate.",
    )
    parser.add_argument("--floor_width_m", type=float, default=30.0)
    parser.add_argument(
        "--low_patch_mu",
        type=float,
        default=0.28,
        help="Static/dynamic friction of the middle LOW patch (High patches stay 0.90).",
    )
    parser.add_argument("--metric_warmup_steps", type=int, default=100)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--trace_npz", type=Path, default=None)
    parser.add_argument("--print_progress", action="store_true")
    parser.add_argument("--disable_fabric", action="store_true")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
checkpoint_path = Path(args_cli.checkpoint or DEFAULT_CHECKPOINT).expanduser().resolve()
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    raise SystemExit("--num_envs and --steps must be positive")
if not 0 <= args_cli.metric_warmup_steps < args_cli.steps:
    raise SystemExit("--metric_warmup_steps must be in [0, steps)")
if not math.isfinite(args_cli.low_patch_mu) or args_cli.low_patch_mu <= 0:
    raise SystemExit("--low_patch_mu must be a positive finite number")
if not math.isfinite(args_cli.floor_width_m) or args_cli.floor_width_m < 3.2:
    raise SystemExit("--floor_width_m must be finite and at least 3.2 m")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion.mdp.spatial_friction_state import (  # noqa: E402
    SPATIAL_HIGH_END,
    SPATIAL_LOW,
    spatial_course_success_mask,
)
from unitree_rl_lab.traction.proprio_baseline import load_proprio_baseline  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _rms(sum_square: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(sum_square / count.clamp_min(1))


def _quantile(value: np.ndarray, q: float) -> float | None:
    return float(np.quantile(value, q)) if value.size else None


def main() -> int:
    wall_start = time.perf_counter()

    def progress(label: str) -> None:
        if args_cli.print_progress:
            print(
                f"[bare480-progress] {label}: "
                f"{time.perf_counter() - wall_start:.3f}s",
                flush=True,
            )

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.scene.num_envs = int(args_cli.num_envs)

    width = float(args_cli.floor_width_m)
    for attr in ("friction_high_start", "friction_low", "friction_high_end"):
        patch = getattr(env_cfg.scene, attr)
        size = tuple(float(item) for item in patch.spawn.size)
        patch.spawn.size = (size[0], width, size[2])

    low_patch = env_cfg.scene.friction_low
    material = low_patch.spawn.physics_material
    material.static_friction = float(args_cli.low_patch_mu)
    material.dynamic_friction = float(args_cli.low_patch_mu)
    progress("configuration ready")

    raw_env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=100.0)
    base = env.unwrapped
    progress("environment ready")

    policy = load_proprio_baseline(checkpoint_path, device=base.device)
    progress("480-D bare policy ready")

    observation = env.get_observations()
    policy_obs = observation["policy"]
    if tuple(policy_obs.shape) != (int(base.num_envs), 1864):
        raise RuntimeError(
            f"expected [N,1864] policy observation, got {tuple(policy_obs.shape)}"
        )
    robot = base.scene["robot"]
    n = int(base.num_envs)
    dt = float(base.step_dt)
    if not math.isclose(dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"expected 50 Hz policy dt=0.02 s, got {dt}")

    active = torch.ones(n, dtype=torch.bool, device=base.device)
    fallen = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    completed = torch.zeros_like(active)
    terminal_step = torch.full((n,), -1, dtype=torch.long, device=base.device)
    first_fall_step = torch.full_like(terminal_step, -1)
    env_origin_x = base.scene.env_origins[:, 0].detach()
    env_origin_y = base.scene.env_origins[:, 1].detach()

    sums = {
        key: torch.zeros(n, device=base.device)
        for key in ("vx", "vy", "vy2", "heading2", "omega2", "tilt2", "slew2")
    }
    low_sums = {
        key: torch.zeros(n, device=base.device)
        for key in ("vx", "vy2", "heading2", "omega2", "tilt2")
    }
    he_sums = {
        key: torch.zeros(n, device=base.device)
        for key in ("vx", "heading2", "post_low_residual2")
    }
    metric_count = torch.zeros(n, dtype=torch.long, device=base.device)
    low_count = torch.zeros_like(metric_count)
    he_count = torch.zeros_like(metric_count)
    sat_elements = torch.zeros(n, device=base.device)
    action_elements = torch.zeros(n, device=base.device)

    episode_heading = torch.zeros(n, device=base.device)
    low_entry_heading = torch.full((n,), float("nan"), device=base.device)
    entered_low = torch.zeros(n, dtype=torch.bool, device=base.device)

    start_x = robot.data.root_pos_w[:, 0].detach().clone()
    last_active_x = start_x.clone()
    last_active_cross = (robot.data.root_pos_w[:, 1] - env_origin_y).detach().clone()
    max_abs_cross = last_active_cross.abs().clone()
    previous_action = policy_obs[:, 335:480].reshape(n, 5, ACTION_DIM)[:, -1]

    nan_detected = False
    nan_component: str | None = None
    total_fall_events = 0
    executed_steps = 0

    for step in range(int(args_cli.steps)):
        policy_obs = observation["policy"]
        with torch.inference_mode():
            action = policy(observation)
        if tuple(action.shape) != (n, ACTION_DIM):
            raise RuntimeError(
                f"action must be [N,{ACTION_DIM}], got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            nan_detected = True
            nan_component = "action"
            break

        pre_active = active.clone()
        local_x = robot.data.root_pos_w[:, 0] - env_origin_x
        cross = robot.data.root_pos_w[:, 1] - env_origin_y
        last_active_x = torch.where(pre_active, local_x, last_active_x)
        last_active_cross = torch.where(pre_active, cross, last_active_cross)
        max_abs_cross = torch.maximum(
            max_abs_cross,
            torch.where(
                pre_active, cross.abs(), torch.zeros_like(cross)
            ),
        )
        stage = base.spatial_course_stage_buf.long()
        high_end_contact = base.spatial_high_end_contact_buf.bool()
        completed_pre = spatial_course_success_mask(
            stage, high_end_contact, local_x, COURSE_SUCCESS_MIN_X
        )
        in_low = stage == SPATIAL_LOW
        in_high_end = stage == SPATIAL_HIGH_END

        vx = robot.data.root_lin_vel_b[:, 0]
        vy = robot.data.root_lin_vel_b[:, 1]
        omega = robot.data.root_ang_vel_b
        omega_norm = torch.linalg.vector_norm(omega, dim=1)
        gravity = policy_obs[:, 27:30]
        tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        slew = torch.linalg.vector_norm(action - previous_action, dim=1)
        sat = (action.abs() >= 2.9).sum(dim=1).float()
        state_values = torch.stack((vx, vy, omega_norm, tilt, slew), dim=1)
        if not torch.isfinite(state_values).all():
            nan_detected = True
            nan_component = "state_metrics"
            break

        heading_now = episode_heading.clone()
        newly_low = in_low & ~entered_low
        low_entry_heading[newly_low] = heading_now[newly_low]
        entered_low |= in_low
        post_low_residual = torch.where(
            entered_low & in_high_end,
            heading_now - low_entry_heading,
            torch.zeros_like(heading_now),
        )

        metric_mask = pre_active & (step >= int(args_cli.metric_warmup_steps))
        metric_count += metric_mask.long()
        low_mask = metric_mask & in_low
        he_mask = metric_mask & in_high_end & entered_low
        low_count += low_mask.long()
        he_count += he_mask.long()
        for name, value in (
            ("vx", vx),
            ("vy", vy),
            ("vy2", vy.square()),
            ("heading2", heading_now.square()),
            ("omega2", omega_norm.square()),
            ("tilt2", tilt.square()),
            ("slew2", slew.square()),
        ):
            sums[name] += torch.where(metric_mask, value, torch.zeros_like(value))
        for name, value in (
            ("vx", vx),
            ("vy2", vy.square()),
            ("heading2", heading_now.square()),
            ("omega2", omega_norm.square()),
            ("tilt2", tilt.square()),
        ):
            low_sums[name] += torch.where(low_mask, value, torch.zeros_like(value))
        for name, value in (
            ("vx", vx),
            ("heading2", heading_now.square()),
            ("post_low_residual2", post_low_residual.square()),
        ):
            he_sums[name] += torch.where(he_mask, value, torch.zeros_like(value))
        sat_elements += torch.where(metric_mask, sat, torch.zeros_like(sat))
        action_elements += metric_mask.float() * ACTION_DIM

        previous_action = action.detach().clone()
        observation, _, done, extras = env.step(action)
        done = done.bool()
        timeout = extras.get("time_outs")
        if timeout is None:
            timeout = torch.zeros_like(done)
        else:
            timeout = timeout.to(device=done.device, dtype=torch.bool)

        fall = done & ~timeout & ~completed_pre
        new_fall = pre_active & fall
        new_timeout = pre_active & timeout
        new_complete = pre_active & completed_pre
        first_fall_step[new_fall] = step + 1
        terminal_step[new_fall | new_timeout | new_complete] = step + 1
        fallen |= new_fall
        timed_out |= new_timeout
        completed |= new_complete
        total_fall_events += int(fall.sum().item())
        active &= ~done

        episode_heading = episode_heading + omega[:, 2].detach() * dt
        episode_heading[done] = 0.0
        low_entry_heading[done] = float("nan")
        entered_low[done] = False
        executed_steps = step + 1
        if args_cli.print_progress and step % 250 == 0:
            progress(f"rollout step {step}/{args_cli.steps}")
        if not bool(active.any().item()):
            break

    terminal_step = torch.where(
        terminal_step >= 0,
        terminal_step,
        torch.full_like(terminal_step, executed_steps),
    )
    mean_vx = sums["vx"] / metric_count.clamp_min(1)
    per_env = {
        "mean_vx": mean_vx,
        "mean_vy": sums["vy"] / metric_count.clamp_min(1),
        "vy_rms": _rms(sums["vy2"], metric_count),
        "heading_rms": _rms(sums["heading2"], metric_count),
        "omega_rms": _rms(sums["omega2"], metric_count),
        "tilt_rms": _rms(sums["tilt2"], metric_count),
        "action_slew_rms": _rms(sums["slew2"], metric_count),
        "action_saturation_fraction": sat_elements
        / action_elements.clamp_min(1),
        "low_mean_vx": low_sums["vx"] / low_count.clamp_min(1),
        "low_vy_rms": _rms(low_sums["vy2"], low_count),
        "low_heading_rms": _rms(low_sums["heading2"], low_count),
        "low_omega_rms": _rms(low_sums["omega2"], low_count),
        "low_tilt_rms": _rms(low_sums["tilt2"], low_count),
        "high_end_mean_vx": he_sums["vx"] / he_count.clamp_min(1),
        "high_end_heading_rms": _rms(he_sums["heading2"], he_count),
        "post_low_residual_rms": _rms(he_sums["post_low_residual2"], he_count),
        "progress_m": last_active_x - start_x,
        "final_cross_track_m": last_active_cross,
        "survival_s": terminal_step.float() * dt,
        "max_abs_cross_track_m": max_abs_cross,
    }
    arrays = {key: value.detach().cpu().numpy() for key, value in per_env.items()}
    valid_env = metric_count.detach().cpu().numpy() > 0
    if not bool(valid_env.any()):
        raise RuntimeError("no environment produced steady-state metrics")

    mean_vx_np = arrays["mean_vx"][valid_env]
    low_valid = low_count.detach().cpu().numpy() > 0
    he_valid = he_count.detach().cpu().numpy() > 0

    def _agg_rmse(name: str, valid: np.ndarray) -> float | None:
        if not bool(valid.any()):
            return None
        value = arrays[name][valid]
        if not np.isfinite(value).all():
            return None
        return float(np.sqrt(np.mean(np.square(value))))

    aggregate = {
        "fall_event_count": total_fall_events,
        "unique_env_first_fall_count": int(fallen.sum().item()),
        "completion_count": int(completed.sum().item()),
        "timeout_count": int(timed_out.sum().item()),
        "survival_fraction": float((~fallen).float().mean().item()),
        "earliest_first_fall_s": (
            float(first_fall_step[first_fall_step >= 0].min().item() * dt)
            if bool((first_fall_step >= 0).any().item())
            else None
        ),
        "mean_survival_s": float(arrays["survival_s"].mean()),
        "metric_valid_envs": int(valid_env.sum()),
        "mean_body_vx_m_s": float(mean_vx_np.mean()),
        "minimum_per_env_mean_vx_m_s": float(mean_vx_np.min()),
        "low_mean_body_vx_m_s": (
            float(arrays["low_mean_vx"][low_valid].mean()) if low_valid.any() else None
        ),
        "high_end_mean_body_vx_m_s": (
            float(arrays["high_end_mean_vx"][he_valid].mean())
            if he_valid.any()
            else None
        ),
        "mean_body_vy_m_s": float(arrays["mean_vy"][valid_env].mean()),
        "body_vy_rms_m_s": _agg_rmse("vy_rms", valid_env),
        "heading_rms_rad": _agg_rmse("heading_rms", valid_env),
        "low_heading_rms_rad": _agg_rmse("low_heading_rms", low_valid),
        "high_end_heading_rms_rad": _agg_rmse("high_end_heading_rms", he_valid),
        "post_low_heading_residual_rms_rad": _agg_rmse(
            "post_low_residual_rms", he_valid
        ),
        "angular_velocity_rms_rad_s": _agg_rmse("omega_rms", valid_env),
        "low_angular_velocity_rms_rad_s": _agg_rmse("low_omega_rms", low_valid),
        "tilt_rms": _agg_rmse("tilt_rms", valid_env),
        "action_slew_rms": _agg_rmse("action_slew_rms", valid_env),
        "action_saturation_fraction": float(
            sat_elements.sum().item() / action_elements.sum().clamp_min(1).item()
        ),
        "mean_progress_m": float(arrays["progress_m"].mean()),
        "maximum_progress_m": float(arrays["progress_m"].max()),
        "minimum_progress_m": float(arrays["progress_m"].min()),
        "mean_final_cross_track_m": float(arrays["final_cross_track_m"].mean()),
        "maximum_abs_final_cross_track_m": float(
            np.abs(arrays["final_cross_track_m"]).max()
        ),
        "p95_abs_max_cross_track_m": _quantile(
            arrays["max_abs_cross_track_m"], 0.95
        ),
        "maximum_abs_cross_track_m": float(arrays["max_abs_cross_track_m"].max()),
        "nan_detected": bool(nan_detected),
        "nan_component": nan_component,
        "executed_steps": executed_steps,
    }

    payload = {
        "policy": {
            "kind": "original_model_49999_proprio_baseline",
            "path": str(checkpoint_path),
            "consumed_observation_dimension": LEGACY_DIM,
            "environment_observation_dimension": 1864,
        },
        "course": {
            "task": TASK,
            "low_patch_mu": float(args_cli.low_patch_mu),
            "high_patch_mu": 0.90,
            "floor_width_m": width,
            "success_min_local_x_m": COURSE_SUCCESS_MIN_X,
            "episode_length_s": float(env_cfg.episode_length_s),
        },
        "run": {
            "seed": int(args_cli.seed),
            "num_envs": n,
            "requested_steps": int(args_cli.steps),
            "executed_steps": executed_steps,
            "policy_dt_s": dt,
            "metric_warmup_steps": int(args_cli.metric_warmup_steps),
        },
        "aggregate": aggregate,
    }
    summary_path = args_cli.summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n"
    )
    progress(f"summary written to {summary_path}")
    print(
        f"[bare480] seed={args_cli.seed} mu_low={args_cli.low_patch_mu} "
        f"falls={aggregate['fall_event_count']} "
        f"completed={aggregate['completion_count']}/{n} "
        f"mean_vx={aggregate['mean_body_vx_m_s']:.3f} "
        f"low_vx={aggregate['low_mean_body_vx_m_s']} "
        f"heading_rms={aggregate['heading_rms_rad']}",
        flush=True,
    )
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
