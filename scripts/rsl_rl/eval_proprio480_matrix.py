#!/usr/bin/env python3
"""Constant-mu and time-based High->Low->High matrix for 480-D/R5 policies.

The environment is the audited 1864-D Hall LongDemo course so the physics,
floor patches, contact sensors and episode mechanics exactly match the R5
acceptance suite.  Two actor backends are supported:

* ``--proprio_checkpoint PATH`` loads a plain 512-256-128 ELU 480-D actor mean
  and feeds it only the first 480 columns of ``policy`` (legacy proprioceptive
  history).  No Hall/contact/force/mu term ever reaches the actor.
* ``--checkpoint PATH --rsl_rl_cfg_entry_point MODULE:CLASS`` reconstructs the
  R5 1864-D FastBase actor from its training runner (actor-only load).

Privileged state (root velocity, contact force, contact-point slip) is used
exclusively for evaluation metrics and is never passed to the policy.

Examples
--------
    python scripts/rsl_rl/eval_proprio480_matrix.py --mode constant --mu 0.10 \\
        --proprio_checkpoint logs/rsl_rl/.../model_400.pt --seed 450 \\
        --summary_json out.json --headless

    python scripts/rsl_rl/eval_proprio480_matrix.py --mode time_hlh \\
        --checkpoint logs/rsl_rl/.../model_399.pt \\
        --rsl_rl_cfg_entry_point unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FootTractionHallSpatialCadenceStrideTransitionRetentionR5PPORunnerCfg \\
        --seed 450 --summary_json out.json --headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

import cli_args


TASK = (
    "Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-"
    "SpatialFrictionCadenceStrideLongDemo"
)
ACTION_DIM = 29
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
TIME_HLH_SEGMENTS = ((0, 300), (300, 800), (800, 1200))  # 0.8/0.10/0.8 in steps
TIME_HLH_MU = (0.80, 0.10, 0.80)
COURSE_SUCCESS_MIN_X = 17.5

EVAL_ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("constant", "time_hlh"), required=True)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--command_vx", type=float, default=0.30)
    parser.add_argument(
        "--seeds",
        type=str,
        default="450",
        help="Comma-separated seeds; every seed gets a fresh environment.",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--metric_warmup_steps", type=int, default=100)
    parser.add_argument("--floor_width_m", type=float, default=30.0)
    parser.add_argument(
        "--proprio_checkpoint",
        type=Path,
        default=None,
        help="Plain 480-D RSL actor checkpoint (legacy MLP layout).",
    )
    parser.add_argument(
        "--rsl_rl_cfg_entry_point",
        type=str,
        default=None,
        help="Runner config 'MODULE:ClassName' used to reconstruct an RSL actor.",
    )
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--print_progress", action="store_true")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
try:
    SEEDS = tuple(int(item) for item in args_cli.seeds.split(",") if item.strip())
except ValueError:
    raise SystemExit("--seeds must be a comma-separated list of integers")
if not SEEDS or any(item < 0 for item in SEEDS):
    raise SystemExit("--seeds must contain non-negative integers")

if args_cli.mode == "constant":
    if args_cli.mu is None or not math.isfinite(args_cli.mu) or args_cli.mu <= 0.0:
        raise SystemExit("--mode constant requires a positive finite --mu")
    segment_bounds = [(0, int(args_cli.steps))]
    segment_mu = [float(args_cli.mu)]
else:
    segment_bounds = list(TIME_HLH_SEGMENTS)
    segment_mu = list(TIME_HLH_MU)
    args_cli.steps = int(segment_bounds[-1][1])

if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    raise SystemExit("--num_envs and --steps must be positive")
if not 0 <= args_cli.metric_warmup_steps < args_cli.steps:
    raise SystemExit("--metric_warmup_steps must be in [0, steps)")
if not (
    (args_cli.proprio_checkpoint is None)
    ^ (args_cli.checkpoint is None)
):
    raise SystemExit("provide exactly one of --proprio_checkpoint / --checkpoint")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.traction.contact_slip import (  # noqa: E402
    static_ground_contact_point_tangential_speed,
)
from unitree_rl_lab.traction.proprio_baseline import load_proprio_baseline  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _rms(sum_square: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(sum_square / count.clamp_min(1))


def _force_robot_mu(uenv, mu: float) -> None:
    """Carry the exact effective floor coefficient on the robot material.

    The three floor patches are static colliders (no runtime physx view).
    This evaluator instead fixes every patch at friction=1.0 (their ``multiply``
    combine mode is preserved) and writes the target mu into the robot's rigid
    material properties, exactly like the audited friction-matrix ``_force_mu``.
    Effective contact friction is therefore ``robot_mu * patch(1.0)``.
    """

    n = uenv.num_envs
    env_ids_cpu = torch.arange(n, device="cpu")
    robot = uenv.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = mu
    materials[env_ids_cpu, :, 1] = mu
    materials[env_ids_cpu, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
    if hasattr(uenv, "ground_friction_mu_buf"):
        uenv.ground_friction_mu_buf[:] = mu
    if hasattr(uenv, "effective_friction_mu_buf"):
        uenv.effective_friction_mu_buf[:] = mu


def _build_policy(base_env, wrapped_env):
    if args_cli.proprio_checkpoint is not None:
        path = Path(args_cli.proprio_checkpoint).expanduser().resolve()
        policy = load_proprio_baseline(path, device=base_env.device)

        def call(observation) -> torch.Tensor:
            return policy(observation)

        return call, {
            "kind": "plain_480d_proprio_actor",
            "path": str(path),
            "consumed_observation_dimension": 480,
            "environment_observation_dimension": 1864,
        }

    path = Path(args_cli.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
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
    if not isinstance(runner_class, type) or not issubclass(
        runner_class, OnPolicyRunner
    ):
        raise RuntimeError(f"unsupported checkpoint runner {agent_cfg.class_name!r}")
    runner = runner_class(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(str(path), load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG), strict=True)
    policy = runner.get_inference_policy(device=base_env.device)

    def call(observation) -> torch.Tensor:
        output = policy(observation)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    return call, {
        "kind": "rsl_rl_fastbase_1864",
        "path": str(path),
        "runner_cfg": args_cli.rsl_rl_cfg_entry_point,
        "consumed_observation_dimension": 1864,
        "environment_observation_dimension": 1864,
    }


def run_seed(seed: int) -> dict:
    """Build one seeded environment, roll out the policy and return metrics."""

    wall_start = time.perf_counter()

    def progress(label: str) -> None:
        if args_cli.print_progress:
            print(
                f"[matrix-progress] {label}: "
                f"{time.perf_counter() - wall_start:.1f}s",
                flush=True,
            )

    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = seed
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    width = float(args_cli.floor_width_m)
    for attr in ("friction_high_start", "friction_low", "friction_high_end"):
        patch = getattr(env_cfg.scene, attr)
        size = tuple(float(item) for item in patch.spawn.size)
        patch.spawn.size = (size[0], width, size[2])
        patch.spawn.physics_material.static_friction = 1.0
        patch.spawn.physics_material.dynamic_friction = 1.0
    # The robot material now carries the evaluator's exact mu.  Neutralize the
    # inherited startup/reset material-scale DR so it cannot overwrite it.
    for name in ("physics_material", "physics_material_reset"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
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
    if env_cfg.episode_length_s < float(args_cli.steps) * 0.02:
        env_cfg.episode_length_s = float(args_cli.steps) * 0.02 + 5.0
    progress("configuration ready")

    raw_env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=100.0)
    base_env = env.unwrapped
    progress("environment ready")

    policy, policy_meta = _build_policy(base_env, env)
    progress("policy ready")

    observation = env.get_observations()
    policy_obs = observation["policy"]
    if tuple(policy_obs.shape) != (int(base_env.num_envs), 1864):
        raise RuntimeError(
            f"expected [N,1864] policy observation, got {tuple(policy_obs.shape)}"
        )
    robot = base_env.scene["robot"]
    n = int(base_env.num_envs)
    dt = float(base_env.step_dt)
    if not math.isclose(dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"expected 50 Hz policy dt=0.02 s, got {dt}")
    foot_body_ids = tuple(
        int(robot.data.body_names.index(name)) for name in FOOT_BODY_NAMES
    )
    left_contact = base_env.scene["left_hall_contact"]
    right_contact = base_env.scene["right_hall_contact"]

    active = torch.ones(n, dtype=torch.bool, device=base_env.device)
    fallen = torch.zeros_like(active)
    completed = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    fall_event_count = 0
    first_fall_step = torch.full((n,), -1, dtype=torch.long, device=base_env.device)
    terminal_step = torch.full_like(first_fall_step, -1)
    episode_heading = torch.zeros(n, device=base_env.device)

    num_segments = len(segment_bounds)
    sums = {
        key: torch.zeros(n, device=base_env.device)
        for key in ("vx", "vy2", "heading2", "omega2", "tilt2")
    }
    counts = torch.zeros(n, dtype=torch.long, device=base_env.device)
    seg_sums = [
        {key: torch.zeros(n, device=base_env.device) for key in sums}
        for _ in range(num_segments)
    ]
    seg_counts = [
        torch.zeros(n, dtype=torch.long, device=base_env.device)
        for _ in range(num_segments)
    ]
    seg_slip_sum = [
        torch.zeros(n, device=base_env.device) for _ in range(num_segments)
    ]
    seg_slip_valid = [
        torch.zeros(n, dtype=torch.long, device=base_env.device)
        for _ in range(num_segments)
    ]
    gait_contact_prev = torch.zeros(n, 2, dtype=torch.bool, device=base_env.device)
    gait_air_steps = torch.zeros(n, 2, dtype=torch.long, device=base_env.device)
    gait_steps_since_td = torch.full(
        (n, 2), 10_000, dtype=torch.long, device=base_env.device
    )
    gait_last_x = torch.full(
        (n, 2), float("nan"), device=base_env.device
    )
    gait_region = torch.full((n,), -1, dtype=torch.long, device=base_env.device)
    gait_stats = [
        {"exposure": 0.0, "touchdowns": 0, "stride_sum": 0.0, "stride_count": 0}
        for _ in range(num_segments)
    ]

    start_x = robot.data.root_pos_w[:, 0].detach().clone()
    env_origin_x = base_env.scene.env_origins[:, 0].detach()
    previous_action = policy_obs[:, 335:480].reshape(n, 5, ACTION_DIM)[:, -1]
    _force_robot_mu(base_env, segment_mu[0])

    for step in range(int(args_cli.steps)):
        if args_cli.mode == "time_hlh":
            for seg_idx in range(num_segments):
                if step == segment_bounds[seg_idx][0]:
                    _force_robot_mu(base_env, segment_mu[seg_idx])

        policy_obs = observation["policy"]
        with torch.inference_mode():
            action = policy(observation)
        if tuple(action.shape) != (n, ACTION_DIM):
            raise RuntimeError(
                f"action must be [N,{ACTION_DIM}], got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise RuntimeError("non-finite policy action")

        pre_active = active.clone()
        local_x = robot.data.root_pos_w[:, 0] - env_origin_x
        vx = robot.data.root_lin_vel_b[:, 0]
        vy = robot.data.root_lin_vel_b[:, 1]
        omega = robot.data.root_ang_vel_b
        omega_norm = torch.linalg.vector_norm(omega, dim=1)
        gravity = policy_obs[:, 27:30]
        tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        completed_pre = local_x >= COURSE_SUCCESS_MIN_X

        seg_id = torch.full((n,), -1, dtype=torch.long, device=base_env.device)
        for seg_idx, (lo, hi) in enumerate(segment_bounds):
            seg_id[(step >= lo) & (step < hi)] = seg_idx

        metric_mask = pre_active & (step >= int(args_cli.metric_warmup_steps))
        counts += metric_mask.long()
        for name, value in (
            ("vx", vx),
            ("vy2", vy.square()),
            ("heading2", episode_heading.square()),
            ("omega2", omega_norm.square()),
            ("tilt2", tilt.square()),
        ):
            sums[name] += torch.where(metric_mask, value, torch.zeros_like(value))
        for seg_idx in range(num_segments):
            mask = metric_mask & (seg_id == seg_idx)
            seg_counts[seg_idx] += mask.long()
            for name, value in (
                ("vx", vx),
                ("vy2", vy.square()),
                ("heading2", episode_heading.square()),
                ("omega2", omega_norm.square()),
                ("tilt2", tilt.square()),
            ):
                seg_sums[seg_idx][name] += torch.where(
                    mask, value, torch.zeros_like(value)
                )

        # Privileged contact-point slip (evaluation metric only).
        try:
            slip_result = static_ground_contact_point_tangential_speed(
                robot.data.body_com_pos_w[:, foot_body_ids, :],
                robot.data.body_com_lin_vel_w[:, foot_body_ids, :],
                robot.data.body_com_ang_vel_w[:, foot_body_ids, :],
                (left_contact.data.contact_pos_w, right_contact.data.contact_pos_w),
                (left_contact.data.force_matrix_w, right_contact.data.force_matrix_w),
                min_normal_force_n=5.0,
            )
            slip_speed = slip_result.speed_per_foot.mean(dim=1)
            slip_valid = slip_result.valid_per_env & metric_mask
            for seg_idx in range(num_segments):
                mask = slip_valid & (seg_id == seg_idx)
                seg_slip_sum[seg_idx] += torch.where(
                    mask, slip_speed, torch.zeros_like(slip_speed)
                )
                seg_slip_valid[seg_idx] += mask.long()
        except Exception as exc:  # pragma: no cover - fail soft for non-Hall scenes
            print(f"[WARN] slip metric unavailable: {exc}", flush=True)

        # Touchdown-based cadence/stride (same definition as the course suite).
        force_l = torch.linalg.vector_norm(left_contact.data.net_forces_w[:, 0, :], dim=-1)
        force_r = torch.linalg.vector_norm(right_contact.data.net_forces_w[:, 0, :], dim=-1)
        gait_contact = torch.stack((force_l > 5.0, force_r > 5.0), dim=1)
        region_changed = seg_id != gait_region
        gait_last_x[region_changed] = float("nan")
        gait_steps_since_td[region_changed] = 10_000
        gait_air_steps[region_changed] = 0
        gait_region = seg_id.clone()
        gait_touchdown = (
            gait_contact
            & ~gait_contact_prev
            & (gait_air_steps >= 3)
            & (gait_steps_since_td >= 8)
            & metric_mask[:, None]
        )
        forward_at_td = local_x[:, None].expand(-1, 2)
        stride = torch.abs(forward_at_td - gait_last_x)
        for seg_idx in range(num_segments):
            region_mask = metric_mask & (seg_id == seg_idx)
            region_td = gait_touchdown & region_mask[:, None]
            valid_stride = region_td & torch.isfinite(gait_last_x)
            gait_stats[seg_idx]["exposure"] += float(
                region_mask.sum().item()
            ) * dt
            gait_stats[seg_idx]["touchdowns"] += int(region_td.sum().item())
            gait_stats[seg_idx]["stride_sum"] += float(stride[valid_stride].sum().item())
            gait_stats[seg_idx]["stride_count"] += int(valid_stride.sum().item())
        gait_last_x[gait_touchdown] = forward_at_td[gait_touchdown]
        gait_steps_since_td[gait_touchdown] = 0
        gait_steps_since_td += 1
        gait_air_steps = torch.where(
            gait_contact,
            torch.zeros_like(gait_air_steps),
            gait_air_steps + 1,
        )
        gait_contact_prev = gait_contact

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
        fall_event_count += int(fall.sum().item())
        active &= ~done

        episode_heading = episode_heading + omega[:, 2].detach() * dt
        episode_heading[done] = 0.0

        if args_cli.print_progress and step % 250 == 0:
            progress(f"rollout step {step}/{args_cli.steps}")
        if not bool(active.any().item()):
            break

    mean_vx = sums["vx"] / counts.clamp_min(1)
    per_env = {
        "mean_vx": mean_vx.detach().cpu().numpy(),
        "vy_rms": _rms(sums["vy2"], counts).detach().cpu().numpy(),
        "heading_rms": _rms(sums["heading2"], counts).detach().cpu().numpy(),
        "omega_rms": _rms(sums["omega2"], counts).detach().cpu().numpy(),
        "tilt_rms": _rms(sums["tilt2"], counts).detach().cpu().numpy(),
        "first_fall_s": first_fall_step.detach().cpu().numpy().astype(float) * dt,
    }
    segments = []
    for seg_idx in range(num_segments):
        count = seg_counts[seg_idx]
        valid = count > 0
        slip = torch.where(
            valid & (seg_slip_valid[seg_idx] > 0),
            seg_slip_sum[seg_idx] / seg_slip_valid[seg_idx].clamp_min(1),
            torch.full_like(seg_slip_sum[seg_idx], float("nan")),
        )
        cadence = (
            gait_stats[seg_idx]["touchdowns"] / gait_stats[seg_idx]["exposure"]
            if gait_stats[seg_idx]["exposure"] > 0
            else float("nan")
        )
        stride = (
            gait_stats[seg_idx]["stride_sum"]
            / gait_stats[seg_idx]["stride_count"]
            if gait_stats[seg_idx]["stride_count"] > 0
            else float("nan")
        )
        segments.append(
            {
                "index": seg_idx,
                "step_bounds": segment_bounds[seg_idx],
                "mu": segment_mu[seg_idx],
                "exposure_s": gait_stats[seg_idx]["exposure"],
                "mean_vx_m_s": float(
                    (seg_sums[seg_idx]["vx"] / count.clamp_min(1))
                    .mean()
                    .item()
                ),
                "vy_rms_m_s": float(
                    _rms(seg_sums[seg_idx]["vy2"], count).mean().item()
                ),
                "heading_rms_rad": float(
                    _rms(seg_sums[seg_idx]["heading2"], count).mean().item()
                ),
                "omega_rms_rad_s": float(
                    _rms(seg_sums[seg_idx]["omega2"], count).mean().item()
                ),
                "tilt_rms": float(
                    _rms(seg_sums[seg_idx]["tilt2"], count).mean().item()
                ),
                "mean_slip_m_s": (
                    float(torch.nanmean(slip).item())
                    if bool(valid.any() & (seg_slip_valid[seg_idx] > 0).any())
                    else None
                ),
                "step_frequency_hz": cadence,
                "mean_stride_length_m": stride,
                "mean_step_length_m": stride / 2.0 if math.isfinite(stride) else None,
            }
        )

    aggregate = {
        "fall_event_count": fall_event_count,
        "unique_env_first_fall_count": int(fallen.sum().item()),
        "survival_completion_count": int((~fallen).sum().item()),
        "course_success_count": int(completed.sum().item()),
        "timeout_count": int(timed_out.sum().item()),
        "mean_body_vx_m_s": float(mean_vx.mean().item()),
        "minimum_per_env_mean_vx_m_s": float(mean_vx.min().item()),
        "body_vy_rms_m_s": float(per_env["vy_rms"].mean()),
        "heading_rms_rad": float(per_env["heading_rms"].mean()),
        "angular_velocity_rms_rad_s": float(per_env["omega_rms"].mean()),
        "tilt_rms": float(per_env["tilt_rms"].mean()),
        "earliest_first_fall_s": (
            float(
                np.nanmin(
                    np.where(
                        per_env["first_fall_s"] >= 0,
                        per_env["first_fall_s"],
                        np.nan,
                    )
                )
            )
            if (per_env["first_fall_s"] >= 0).any()
            else None
        ),
        "mean_first_fall_s": (
            float(
                np.nanmean(
                    np.where(
                        per_env["first_fall_s"] >= 0,
                        per_env["first_fall_s"],
                        np.nan,
                    )
                )
            )
            if (per_env["first_fall_s"] >= 0).any()
            else None
        ),
    }

    payload = {
        "policy": policy_meta,
        "mode": args_cli.mode,
        "segments": segments,
        "course": {
            "task": TASK,
            "command_vx_m_s": float(args_cli.command_vx),
            "floor_width_m": float(args_cli.floor_width_m),
            "success_min_local_x_m": COURSE_SUCCESS_MIN_X,
            "policy_dt_s": dt,
            "episode_length_s": float(env_cfg.episode_length_s),
            "mu_application": (
                "robot_rigid_material_mu * static_patch_1.0 (multiply combine)"
            ),
        },
        "run": {
            "seed": seed,
            "num_envs": n,
            "steps": int(args_cli.steps),
            "metric_warmup_steps": int(args_cli.metric_warmup_steps),
        },
        "aggregate": aggregate,
    }
    print(
        f"[matrix] mode={args_cli.mode} seed={seed} "
        f"falls={aggregate['fall_event_count']} "
        f"survived={aggregate['survival_completion_count']}/{n} "
        f"mean_vx={aggregate['mean_body_vx_m_s']:.3f}",
        flush=True,
    )
    env.close()
    return payload


def main() -> int:
    summary_path = args_cli.summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if len(SEEDS) == 1:
        payload = run_seed(SEEDS[0])
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n"
        )
        print(f"[matrix] summary written to {summary_path}", flush=True)
        simulation_app.close()
        return 0
    results = {}
    for seed in SEEDS:
        payload = run_seed(seed)
        results[str(seed)] = payload
        per_seed = summary_path.with_name(
            f"{summary_path.stem}_seed{seed}{summary_path.suffix}"
        )
        per_seed.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n"
        )
    summary_path.write_text(
        json.dumps({"seeds": SEEDS, "results": results}, indent=2, sort_keys=False)
        + "\n"
    )
    print(f"[matrix] summary written to {summary_path}", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
