#!/usr/bin/env python3
"""Fixed-seed multi-μ evaluation matrix for Foot policies.

Runs short rollouts at fixed velocity commands across friction levels and
prints a CSV-like summary (forward speed, lateral drift, yaw, slip, resets).

Usage (after conda activate isaaclab-v2 + Isaac Lab env):

  cd TractionSense-G1
  python scripts/rsl_rl/eval_friction_matrix.py \\
    --task Unitree-G1-29dof-Velocity-Foot-Adaptive-V2 \\
    --checkpoint logs/rsl_rl/.../model_XXXX.pt \\
    --num_envs 64 --headless

Smoke (no long train): keep --max_steps 200.

Does NOT send commands to a real robot.
"""

from __future__ import annotations

import argparse
import csv
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

# App launch pattern matches train.py / play.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Friction evaluation matrix")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=250)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument(
    "--command_ramp_steps",
    type=int,
    default=-1,
    help="0 for a step command; negative uses the full warmup interval",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument("--collect_npz", type=Path, default=None)
parser.add_argument(
    "--collect_dagger_npz",
    type=Path,
    default=None,
    help="Collect exact magnetic policy + 641-D Teacher observation pairs.",
)
parser.add_argument(
    "--dagger_execution_teacher_onnx",
    type=Path,
    default=None,
    help=(
        "In switch DAgger mode, execute this 641-D Oracle Teacher while still "
        "recording the exact deployable Student input."
    ),
)
parser.add_argument(
    "--shared_policy",
    type=Path,
    default=None,
    help="SharedMagneticPolicy .pt; bypasses the RSL-RL checkpoint loader.",
)
parser.add_argument(
    "--lateral_estimator",
    type=Path,
    default=None,
    help="1862-D body-vy estimator .pt; overwrites policy channel 1862 online.",
)
parser.add_argument(
    "--nominal_magnetic_sensor",
    action="store_true",
    help="Keep both feet valid and disable synthetic packet-drop faults.",
)
parser.add_argument("--collect_stride", type=int, default=5)
parser.add_argument(
    "--recovery_steps",
    type=int,
    default=25,
    help="Mark this many post-fall steps as recovery samples in DAgger output.",
)
parser.add_argument(
    "--failure_weight",
    type=float,
    default=8.0,
    help="Training priority written for the exact pre-fall DAgger state.",
)
parser.add_argument(
    "--recovery_weight",
    type=float,
    default=4.0,
    help="Training priority written for post-reset recovery DAgger states.",
)
parser.add_argument("--vx", type=float, nargs="+", default=[0.5, 1.0, 1.5])
parser.add_argument(
    "--mu_bins",
    type=float,
    nargs="+",
    default=[0.08, 0.20, 0.40, 0.80, 1.20],
    help="Exact static/dynamic robot material friction used for each rollout",
)
parser.add_argument(
    "--switch_sequence",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Enable one continuous rollout and apply this friction sequence while "
        "keeping vx fixed, e.g. --switch_sequence 1.2 0.08 1.2"
    ),
)
parser.add_argument(
    "--switch_phase_steps",
    type=int,
    default=150,
    help="Policy steps per friction phase in switch mode (50 Hz by default).",
)
parser.add_argument(
    "--switch_settle_steps",
    type=int,
    default=25,
    help="Initial steps excluded from each phase's steady-state metrics.",
)
parser.add_argument(
    "--switch_response_window_steps",
    type=int,
    default=10,
    help="Moving-average window used to estimate transition response time.",
)
parser.add_argument(
    "--switch_max_response_s",
    type=float,
    default=1.0,
    help="Acceptance gate for both high->low and low->high response.",
)
parser.add_argument(
    "--ablate_foot_sensor",
    action="store_true",
    help="Zero deployable foot/Hall channels for a causal sensor ablation.",
)
parser.add_argument(
    "--output_summary",
    type=Path,
    default=None,
    help="Switch-mode Markdown summary path (defaults beside --output_csv).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402
import unitree_rl_lab.tasks  # noqa: F401, E402


def _force_command(env, vx: float):
    """Set constant base velocity command if command term supports ranges."""
    try:
        term = env.unwrapped.command_manager.get_term("base_velocity")
        term.cfg.ranges.lin_vel_x = (vx, vx)
        term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        term.cfg.ranges.ang_vel_z = (0.0, 0.0)
        term.cfg.rel_standing_envs = 0.0
        if hasattr(term.cfg, "rel_spin_envs"):
            term.cfg.rel_spin_envs = 0.0
        if hasattr(term.cfg, "high_speed_fraction"):
            term.cfg.high_speed_fraction = 0.0
        # resample all
        env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        term._resample_command(env_ids)
        term.is_standing_env[:] = False
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1:] = 0.0
    except Exception as e:
        print(f"[warn] could not force command: {e}")


def _set_command_value(env, vx: float):
    """Update the active command without resampling/resetting its history."""
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1:] = 0.0


def _force_mu(env, mu: float):
    """Assign one exact material to every robot shape and synchronize teacher μ."""
    uenv = env.unwrapped
    n = uenv.num_envs
    robot = uenv.scene["robot"]
    env_ids_cpu = torch.arange(n, device="cpu")
    materials = robot.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = mu
    materials[env_ids_cpu, :, 1] = mu
    materials[env_ids_cpu, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
    if not hasattr(uenv, "ground_friction_mu_buf"):
        uenv.ground_friction_mu_buf = torch.full((n,), mu, device=uenv.device)
    uenv.ground_friction_mu_buf[:] = mu
    if hasattr(uenv, "effective_friction_mu_buf"):
        uenv.effective_friction_mu_buf[:] = mu
    if hasattr(uenv, "ground_friction_regime_buf"):
        regime = 0 if mu <= 0.25 else 2 if mu >= 0.75 else 1
        uenv.ground_friction_regime_buf[:] = regime
    if hasattr(uenv, "friction_switch_is_high_buf"):
        uenv.friction_switch_is_high_buf[:] = mu >= 0.75
    if hasattr(uenv, "friction_switch_target_mu_buf"):
        uenv.friction_switch_target_mu_buf[:] = mu


def _yaw_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return wrapped yaw for Isaac Lab's scalar-first world quaternion."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _ablate_foot_observation(observation):
    """Return an observation copy with deployable foot evidence removed."""

    # Isaac Lab may return either a regular dict or a TensorDict-like mapping,
    # depending on whether the value came from reset/get_observations or step.
    # Test the mapping contract instead of the concrete container type.
    try:
        policy_obs = observation["policy"].clone()
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(
            "foot ablation requires an observation with a policy group"
        ) from exc
    ablated = observation.clone() if hasattr(observation, "clone") else dict(observation)
    dim = int(policy_obs.shape[1])
    if dim == 1864:
        # Hall history and per-foot validity.  Preserve proprioception, sample
        # timing, and the independent lateral/heading feedback channels.
        policy_obs[:, 480:1830] = 0.0
        policy_obs[:, 1860:1862] = 0.0
    elif dim == 640:
        policy_obs[:, 480:640] = 0.0
    elif dim == 510:
        policy_obs[:, 480:510] = 0.0
    else:
        raise ValueError(
            f"unsupported policy observation dimension for foot ablation: {dim}"
        )
    ablated["policy"] = policy_obs
    return ablated


def _response_time(
    values: list[float],
    previous_steady: float,
    current_steady: float,
    dt: float,
    window_steps: int,
) -> float:
    """Time to complete 80% of a speed transition with a short dwell."""

    if not values:
        return float("nan")
    delta = current_steady - previous_steady
    if abs(delta) < 0.05:
        return 0.0
    window = max(int(window_steps), 1)
    array = np.asarray(values, dtype=np.float64)
    if len(array) < window:
        smoothed = array
    else:
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(array, kernel, mode="valid")
    boundary = previous_steady + 0.80 * delta
    dwell = max(int(round(0.20 / max(dt, 1.0e-6))), 1)
    if delta > 0.0:
        reached = smoothed >= boundary
    else:
        reached = smoothed <= boundary
    for index in range(len(reached)):
        if bool(np.all(reached[index : index + dwell])) and index + dwell <= len(
            reached
        ):
            # A valid moving average ending at index+window-1 has already
            # consumed that much physical response time.
            return float((index + window) * dt)
    return float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _run_switch_evaluation(
    env,
    policy,
    robot,
    contact_sensor,
    foot_body_ids,
    foot_sensor_ids,
    execution_teacher=None,
) -> None:
    """Run one high/low/high trajectory and report causal gait adaptation."""

    sequence = [float(value) for value in args_cli.switch_sequence]
    if len(sequence) < 2:
        raise ValueError("--switch_sequence requires at least two friction phases")
    if args_cli.switch_phase_steps <= 0:
        raise ValueError("--switch_phase_steps must be positive")
    if not 0 <= args_cli.switch_settle_steps < args_cli.switch_phase_steps:
        raise ValueError(
            "--switch_settle_steps must be in [0, switch_phase_steps)"
        )
    if len(args_cli.vx) != 1:
        raise ValueError("switch mode requires exactly one --vx value")
    command_vx = float(args_cli.vx[0])
    uenv = env.unwrapped
    n = int(uenv.num_envs)
    dt = float(uenv.step_dt)

    env.reset()
    if args_cli.nominal_magnetic_sensor:
        for name in ("magnetic_episode_valid_lr_buf", "magnetic_valid_lr_buf"):
            if hasattr(uenv, name):
                getattr(uenv, name).fill_(1.0)
        for name in (
            "magnetic_episode_age_lr_buf",
            "magnetic_age_lr_buf",
            "structured_foot_current_age_buf",
            "structured_foot_sample_dropout_prob_buf",
            "structured_foot_burst_dropout_prob_buf",
            "structured_foot_burst_remaining_buf",
        ):
            if hasattr(uenv, name):
                getattr(uenv, name).zero_()
        if hasattr(uenv, "structured_foot_current_valid_buf"):
            uenv.structured_foot_current_valid_buf.fill_(1.0)
        if hasattr(uenv, "magnetic_last_step_buf"):
            uenv.magnetic_last_step_buf.fill_(-1)
            uenv.magnetic_packet_cache = {}

    _force_mu(env, sequence[0])
    _force_command(env, command_vx)
    ramp_steps = (
        args_cli.warmup_steps
        if args_cli.command_ramp_steps < 0
        else min(args_cli.command_ramp_steps, args_cli.warmup_steps)
    )
    if ramp_steps > 0:
        _set_command_value(env, 0.0)
    obs = env.get_observations()

    phase_data = []
    for index, mu in enumerate(sequence):
        phase_data.append(
            {
                "phase": index,
                "mu": mu,
                "vx": [],
                "abs_vy": [],
                "slip": [],
                "fn": [],
                "ft": [],
                "early_slip": [],
                "steady_slip": [],
                "touchdowns": 0.0,
                "stride_sum": 0.0,
                "stride_count": 0,
                "falls": 0,
            }
        )
    time_rows = []
    dagger_policy_obs = []
    dagger_teacher_obs = []
    dagger_mu = []
    dagger_cmd = []
    dagger_weight = []
    dagger_phase = []
    dagger_time_since_switch = []

    prev_contact = (
        torch.abs(
            contact_sensor.data.net_forces_w[:, foot_sensor_ids, 2]
        )
        > 5.0
    )
    air_steps = torch.zeros((n, 2), dtype=torch.int64, device=uenv.device)
    steps_since_touchdown = torch.full(
        (n, 2), 10_000, dtype=torch.int64, device=uenv.device
    )
    minimum_air_steps = max(int(round(0.08 / dt)), 1)
    minimum_touchdown_gap_steps = max(int(round(0.20 / dt)), 1)
    last_touchdown_forward = torch.full(
        (n, 2), float("nan"), device=uenv.device
    )
    phase_origin = robot.data.root_pos_w[:, :2].clone()
    phase_yaw = _yaw_from_wxyz(robot.data.root_quat_w).clone()
    current_phase = -1
    total_steps = args_cli.warmup_steps + len(sequence) * args_cli.switch_phase_steps

    for step in range(total_steps):
        if step < ramp_steps:
            _set_command_value(
                env, command_vx * float(step + 1) / float(ramp_steps)
            )
        if step >= args_cli.warmup_steps:
            phase_index = (
                step - args_cli.warmup_steps
            ) // args_cli.switch_phase_steps
            local_step = (
                step - args_cli.warmup_steps
            ) % args_cli.switch_phase_steps
            if phase_index != current_phase:
                current_phase = phase_index
                _force_mu(env, sequence[phase_index])
                phase_origin = robot.data.root_pos_w[:, :2].clone()
                phase_yaw = _yaw_from_wxyz(robot.data.root_quat_w).clone()
                prev_contact = (
                    torch.abs(
                        contact_sensor.data.net_forces_w[
                            :, foot_sensor_ids, 2
                        ]
                    )
                    > 5.0
                )
                air_steps.zero_()
                steps_since_touchdown.fill_(10_000)
                last_touchdown_forward.fill_(float("nan"))
        else:
            phase_index = -1
            local_step = -1

        policy_obs = (
            _ablate_foot_observation(obs)
            if args_cli.ablate_foot_sensor
            else obs
        )
        with torch.inference_mode():
            actions = policy(policy_obs)
            if execution_teacher is not None:
                if "teacher" not in obs:
                    raise ValueError(
                        "Teacher execution requires a teacher observation group"
                    )
                actions = execution_teacher(obs["teacher"])
        if (
            args_cli.collect_dagger_npz is not None
            and phase_index >= 0
            # The first observation after forcing a new material still
            # describes the previous simulator step.  Skip it so the Oracle
            # μ and deployable sensor history are time-aligned.
            and local_step > 0
            and local_step % max(args_cli.collect_stride, 1) == 0
        ):
            if "teacher" not in obs:
                raise ValueError(
                    "switch DAgger collection requires a teacher observation group"
                )
            actor_obs = getattr(policy, "last_policy_observation", None)
            if actor_obs is None:
                actor_obs = obs["policy"]
            teacher_obs = obs["teacher"]
            if actor_obs.shape[1] != 1864 or teacher_obs.shape[1] != 641:
                raise ValueError(
                    "switch DAgger observation mismatch: "
                    f"policy={actor_obs.shape}, teacher={teacher_obs.shape}"
                )
            dagger_policy_obs.append(
                actor_obs.detach().cpu().numpy().astype(np.float32)
            )
            dagger_teacher_obs.append(
                teacher_obs.detach().cpu().numpy().astype(np.float32)
            )
            dagger_mu.append(
                np.full(n, sequence[phase_index], dtype=np.float32)
            )
            dagger_cmd.append(np.full(n, command_vx, dtype=np.float32))
            transition_weight = (
                4.0
                if phase_index > 0 and (local_step + 1) * dt <= 1.0
                else 1.0
            )
            dagger_weight.append(
                np.full(n, transition_weight, dtype=np.float32)
            )
            dagger_phase.append(
                np.full(n, phase_index, dtype=np.int16)
            )
            dagger_time_since_switch.append(
                np.full(n, (local_step + 1) * dt, dtype=np.float32)
            )
        obs, _, dones, extras = env.step(actions)

        if phase_index < 0:
            continue
        data = phase_data[phase_index]
        timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
        if timeouts is None:
            falls = dones.bool()
        else:
            falls = dones.bool() & ~timeouts.to(device=dones.device).bool()
        data["falls"] += int(falls.sum().item())

        vel = robot.data.root_lin_vel_b
        foot_vel = robot.data.body_lin_vel_w[:, foot_body_ids, :2]
        foot_speed = torch.linalg.norm(foot_vel, dim=-1)
        fn = torch.abs(
            contact_sensor.data.net_forces_w[:, foot_sensor_ids, 2]
        )
        ft = torch.linalg.norm(
            contact_sensor.data.net_forces_w[:, foot_sensor_ids, :2], dim=-1
        )
        contact = fn > 5.0
        contact_weight = contact.float()
        contact_slip = (
            (foot_speed * contact_weight).sum()
            / contact_weight.sum().clamp(min=1.0)
        )
        mean_vx = float(vel[:, 0].mean().item())
        mean_abs_vy = float(torch.abs(vel[:, 1]).mean().item())
        mean_slip = float(contact_slip.item())
        mean_fn = float(fn.sum(dim=1).mean().item())
        mean_ft = float(ft.sum(dim=1).mean().item())
        data["vx"].append(mean_vx)
        data["abs_vy"].append(mean_abs_vy)
        data["slip"].append(mean_slip)
        data["fn"].append(mean_fn)
        data["ft"].append(mean_ft)
        if local_step < max(int(round(0.50 / dt)), 1):
            data["early_slip"].append(mean_slip)
        if local_step >= args_cli.switch_settle_steps:
            data["steady_slip"].append(mean_slip)

            displacement = robot.data.root_pos_w[:, :2] - phase_origin
            forward = (
                torch.cos(phase_yaw) * displacement[:, 0]
                + torch.sin(phase_yaw) * displacement[:, 1]
            )
            # Reject one-frame force chatter and repeated impacts from a
            # sliding foot.  Cadence should describe deliberate swing cycles,
            # not noisy contact threshold crossings on the low-friction phase.
            touchdown = (
                contact
                & ~prev_contact
                & (air_steps >= minimum_air_steps)
                & (steps_since_touchdown >= minimum_touchdown_gap_steps)
            )
            valid_stride = touchdown & torch.isfinite(
                last_touchdown_forward
            )
            stride = torch.abs(
                forward[:, None] - last_touchdown_forward
            )
            data["stride_sum"] += float(stride[valid_stride].sum().item())
            data["stride_count"] += int(valid_stride.sum().item())
            data["touchdowns"] += float(touchdown.sum().item())
            last_touchdown_forward[touchdown] = forward[:, None].expand(
                -1, 2
            )[touchdown]
            steps_since_touchdown[touchdown] = 0
        steps_since_touchdown += 1
        air_steps = torch.where(
            contact, torch.zeros_like(air_steps), air_steps + 1
        )
        prev_contact = contact

        time_rows.append(
            {
                "time_s": (
                    step - args_cli.warmup_steps + 1
                )
                * dt,
                "phase": phase_index,
                "time_since_switch_s": (local_step + 1) * dt,
                "mu": sequence[phase_index],
                "cmd_vx": command_vx,
                "mean_vx": mean_vx,
                "mean_abs_vy": mean_abs_vy,
                "mean_contact_slip": mean_slip,
                "mean_foot_fn": mean_fn,
                "mean_foot_ft": mean_ft,
                "falls_cumulative": data["falls"],
            }
        )

    phase_rows = []
    previous_steady = None
    steady_start = args_cli.switch_settle_steps
    for data in phase_data:
        phase_duration = (
            args_cli.switch_phase_steps - steady_start
        ) * dt
        steady_vx_values = data["vx"][steady_start:]
        steady_vx = _mean(steady_vx_values)
        response = (
            float("nan")
            if previous_steady is None
            else _response_time(
                data["vx"],
                previous_steady,
                steady_vx,
                dt,
                args_cli.switch_response_window_steps,
            )
        )
        stride = (
            data["stride_sum"] / data["stride_count"]
            if data["stride_count"] > 0
            else float("nan")
        )
        step_frequency = data["touchdowns"] / max(
            n * phase_duration, 1.0e-6
        )
        early_slip = _mean(data["early_slip"])
        steady_slip = _mean(data["steady_slip"])
        slip_reduction = (
            (early_slip - steady_slip) / max(early_slip, 1.0e-6)
            if np.isfinite(early_slip) and np.isfinite(steady_slip)
            else float("nan")
        )
        phase_rows.append(
            {
                "phase": data["phase"],
                "mu": data["mu"],
                "cmd_vx": command_vx,
                "steady_vx": steady_vx,
                "steady_abs_vy": _mean(data["abs_vy"][steady_start:]),
                "steady_contact_slip": steady_slip,
                "early_contact_slip": early_slip,
                "slip_reduction_fraction": slip_reduction,
                "step_frequency_hz": step_frequency,
                "mean_stride_length_m": stride,
                "mean_step_length_m": 0.5 * stride,
                "response_time_s": response,
                "falls": data["falls"],
            }
        )
        previous_steady = steady_vx

    low_rows = [row for row in phase_rows if row["mu"] <= 0.25]
    high_rows = [row for row in phase_rows if row["mu"] >= 0.75]
    low_vx = _mean([row["steady_vx"] for row in low_rows])
    high_vx = _mean([row["steady_vx"] for row in high_rows])
    low_cadence = _mean([row["step_frequency_hz"] for row in low_rows])
    high_cadence = _mean([row["step_frequency_hz"] for row in high_rows])
    low_step = _mean([row["mean_step_length_m"] for row in low_rows])
    high_step = _mean([row["mean_step_length_m"] for row in high_rows])
    responses = [
        row["response_time_s"]
        for row in phase_rows[1:]
        if np.isfinite(row["response_time_s"])
    ]
    all_responses_observed = len(responses) == len(phase_rows) - 1
    total_falls = sum(int(row["falls"]) for row in phase_rows)
    gates = [
        (
            "全程无摔倒",
            float(total_falls),
            total_falls == 0,
            "= 0",
        ),
        (
            "高低摩擦稳态速度差",
            high_vx - low_vx,
            bool(high_vx - low_vx >= 0.35),
            ">= 0.35 m/s",
        ),
        (
            "低摩擦稳态限速",
            low_vx,
            bool(low_vx <= 0.45),
            "<= 0.45 m/s",
        ),
        (
            "高摩擦速度恢复",
            high_vx,
            bool(high_vx >= 0.75 * command_vx),
            ">= 75% cmd",
        ),
        (
            "切换响应时间",
            max(responses, default=float("inf")),
            bool(
                all_responses_observed
                and max(responses) <= args_cli.switch_max_response_s
            ),
            f"<= {args_cli.switch_max_response_s:.2f} s",
        ),
        (
            "高摩擦步频恢复",
            high_cadence - low_cadence,
            bool(high_cadence - low_cadence >= 0.10),
            ">= 0.10 Hz",
        ),
        (
            "高摩擦步长恢复",
            high_step - low_step,
            bool(high_step - low_step >= 0.03),
            ">= 0.03 m",
        ),
    ]

    output_csv = args_cli.output_csv
    if output_csv is None:
        output_csv = Path("friction_switch_phases.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    phase_fields = list(phase_rows[0])
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=phase_fields)
        writer.writeheader()
        writer.writerows(phase_rows)
    time_csv = output_csv.with_name(
        f"{output_csv.stem}.timeseries{output_csv.suffix}"
    )
    with time_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(time_rows[0]))
        writer.writeheader()
        writer.writerows(time_rows)

    summary = args_cli.output_summary
    if summary is None:
        summary = output_csv.with_name(f"{output_csv.stem}.summary.md")
    overall = "PASS" if all(item[2] for item in gates) else "NEEDS_TRAINING"
    lines = [
        "# Friction-switch adaptation",
        "",
        f"- Overall: **{overall}**",
        f"- Command: `{command_vx:.3f} m/s` (unchanged across phases)",
        f"- Sequence: `{sequence}`",
        f"- Foot-sensor ablation: `{args_cli.ablate_foot_sensor}`",
        f"- Environments / seed: `{n}` / `{args_cli.seed}`",
        "",
        "## Per-phase behavior",
        "",
        "| phase | μ | vx | |vy| | slip | cadence | step length | response | falls |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        response_text = (
            f"{row['response_time_s']:.3f}"
            if np.isfinite(row["response_time_s"])
            else "n/a"
        )
        lines.append(
            f"| {row['phase']} | {row['mu']:.3f} | {row['steady_vx']:.3f} | "
            f"{row['steady_abs_vy']:.3f} | {row['steady_contact_slip']:.3f} | "
            f"{row['step_frequency_hz']:.3f} | {row['mean_step_length_m']:.3f} | "
            f"{response_text} | {row['falls']} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "| gate | value | result | target |",
        "|---|---:|:---:|---:|",
    ]
    for name, value, passed, target in gates:
        lines.append(
            f"| {name} | {value:.3f} | {'PASS' if passed else 'WARN'} | {target} |"
        )
    lines.append("")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines), encoding="utf-8")

    if args_cli.collect_dagger_npz is not None:
        if not dagger_policy_obs:
            raise RuntimeError("no switch DAgger observations were collected")
        args_cli.collect_dagger_npz.parent.mkdir(
            parents=True, exist_ok=True
        )
        np.savez_compressed(
            args_cli.collect_dagger_npz,
            obs=np.concatenate(dagger_policy_obs, axis=0),
            teacher_obs=np.concatenate(dagger_teacher_obs, axis=0),
            mu=np.concatenate(dagger_mu, axis=0),
            cmd_vx=np.concatenate(dagger_cmd, axis=0),
            sample_weight=np.concatenate(dagger_weight, axis=0),
            phase=np.concatenate(dagger_phase, axis=0),
            time_since_switch_s=np.concatenate(
                dagger_time_since_switch, axis=0
            ),
        )
        print(
            f"[info] switch DAgger dataset: {args_cli.collect_dagger_npz} "
            f"shape={sum(len(item) for item in dagger_policy_obs)}x1864"
        )

    print("\n".join(lines))
    print(f"[info] phase CSV: {output_csv}")
    print(f"[info] time-series CSV: {time_csv}")
    print(f"[info] summary: {summary}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    env_cfg.seed = args_cli.seed

    # A matrix must isolate friction: use a flat plane and prevent episode
    # resets from silently drawing a new random material.
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if hasattr(env_cfg.events, "physics_material_reset"):
        env_cfg.events.physics_material_reset = None
    if args_cli.switch_sequence is not None and hasattr(
        env_cfg.events, "friction_switch"
    ):
        # Switch evaluation applies exact, synchronized phase values itself.
        # Training retains the event-driven low/high alternation.
        env_cfg.events.friction_switch = None
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.shared_policy is None:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if args_cli.checkpoint:
            print(f"[info] load {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        shared_script_dir = Path(__file__).resolve().parents[3] / "scripts"
        sys.path.insert(0, str(shared_script_dir))
        from train_shared_magnetic_policy import INPUT_DIM, SharedMagneticPolicy

        payload = torch.load(args_cli.shared_policy, map_location="cpu", weights_only=False)
        if "model" in payload:
            shared_model = SharedMagneticPolicy().to(env.unwrapped.device).eval()
            shared_model.load_state_dict(payload["model"], strict=True)
            shared_description = "single shared magnetic actor"
        elif payload.get("policy_type") == "estimator_guided_magnetic_teacher":
            from export_estimator_guided_magnetic_teacher import build_runtime

            shared_model = build_runtime(payload).to(env.unwrapped.device).eval()
            shared_description = (
                "calibration-gated estimator-guided magnetic Teacher"
            )
        elif (
            "safe_checkpoint" in payload
            and "fast_checkpoint" in payload
            and "stable_checkpoint" in payload
        ):
            from export_jointwise_magnetic_ensemble import load_policy

            metrics = payload.get("metrics", {})
            if "boost_factor" in metrics:
                from export_high_friction_speed_boost_policy import (
                    HighFrictionSpeedBoostPolicy,
                )

                shared_model = HighFrictionSpeedBoostPolicy(
                    load_policy(Path(payload["safe_checkpoint"])),
                    load_policy(Path(payload["fast_checkpoint"])),
                    load_policy(Path(payload["stable_checkpoint"])),
                    float(metrics["stable_lateral_weight"]),
                    float(metrics.get("stable_arm_weight", 0.0)),
                    float(metrics["residual_center"]),
                    float(metrics["residual_sharpness"]),
                    float(metrics["evidence_center"]),
                    float(metrics["evidence_sharpness"]),
                    float(metrics["boost_factor"]),
                    float(metrics["traction_center"]),
                    float(metrics["traction_sharpness"]),
                    float(metrics["command_center"]),
                    float(metrics["command_sharpness"]),
                    metrics.get(
                        "stable_branch_command",
                        "boosted internal command",
                    )
                    == "boosted internal command",
                ).to(env.unwrapped.device).eval()
                shared_description = (
                    "calibration-gated high-friction speed compensation"
                )
            else:
                from export_confidence_gated_magnetic_policy import (
                    CalibrationGatedPolicy,
                )
                from export_jointwise_magnetic_ensemble import (
                    JointwiseEnsemble,
                )

                fast_model = JointwiseEnsemble(
                    load_policy(Path(payload["fast_checkpoint"])),
                    load_policy(Path(payload["stable_checkpoint"])),
                    float(metrics["stable_lateral_weight"]),
                    float(metrics.get("stable_arm_weight", 0.0)),
                )
                shared_model = CalibrationGatedPolicy(
                    load_policy(Path(payload["safe_checkpoint"])),
                    fast_model,
                    float(metrics["residual_center"]),
                    float(metrics["residual_sharpness"]),
                    float(metrics["evidence_center"]),
                    float(metrics["evidence_sharpness"]),
                ).to(env.unwrapped.device).eval()
                shared_description = (
                    "calibration-confidence gated magnetic ensemble"
                )
        elif "fast_checkpoint" in payload and "stable_checkpoint" in payload:
            from export_jointwise_magnetic_ensemble import (
                JointwiseEnsemble,
                load_policy,
            )

            metrics = payload.get("metrics", {})
            shared_model = JointwiseEnsemble(
                load_policy(Path(payload["fast_checkpoint"])),
                load_policy(Path(payload["stable_checkpoint"])),
                float(metrics["stable_lateral_weight"]),
                float(metrics.get("stable_arm_weight", 0.0)),
            ).to(env.unwrapped.device).eval()
            shared_description = "jointwise fast/stable magnetic ensemble"
        else:
            raise ValueError(
                f"{args_cli.shared_policy}: no supported runtime policy state"
            )
        if int(payload.get("input_dim", INPUT_DIM)) != INPUT_DIM:
            raise ValueError(
                f"shared policy input mismatch: checkpoint={payload.get('input_dim')} "
                f"code={INPUT_DIM}"
            )

        lateral_estimator = None
        lateral_estimate = None
        last_policy_observation = None
        if args_cli.lateral_estimator is not None:
            from train_lateral_velocity_estimator import (
                LateralVelocityEstimator,
                NormalizedLateralVelocityEstimator,
            )

            estimator_payload = torch.load(
                args_cli.lateral_estimator,
                map_location="cpu",
                weights_only=False,
            )
            estimator_core = LateralVelocityEstimator(
                int(estimator_payload["input_dim"])
            )
            estimator_core.load_state_dict(estimator_payload["model"], strict=True)
            lateral_estimator = NormalizedLateralVelocityEstimator(
                estimator_core,
                np.asarray(estimator_payload["mean"], dtype=np.float32),
                np.asarray(estimator_payload["scale"], dtype=np.float32),
            ).to(env.unwrapped.device).eval()
            print(
                f"[info] body-vy estimator {args_cli.lateral_estimator} "
                f"(input={estimator_payload['input_dim']}, alpha=0.35)"
            )

        def policy(observation):
            nonlocal lateral_estimate, last_policy_observation
            actor_observation = observation["policy"]
            if actor_observation.shape[1] != INPUT_DIM:
                raise ValueError(
                    f"environment policy observation is {actor_observation.shape[1]}, "
                    f"expected {INPUT_DIM}"
                )
            if lateral_estimator is not None:
                actor_observation = actor_observation.clone()
                estimate = lateral_estimator(actor_observation[:, :1862])
                if lateral_estimate is None:
                    lateral_estimate = estimate
                else:
                    lateral_estimate = 0.65 * lateral_estimate + 0.35 * estimate
                actor_observation[:, 1862] = lateral_estimate
            # Preserve the exact tensor consumed by the Student.  In
            # particular, channel 1862 must contain the deployable body-vy
            # estimate instead of the simulator's privileged ground truth.
            last_policy_observation = actor_observation.detach().clone()
            policy.last_policy_observation = last_policy_observation
            return shared_model(actor_observation)

        print(
            f"[info] load shared magnetic policy {args_cli.shared_policy} "
            f"({shared_description}, input={INPUT_DIM}, output=29)"
        )
    robot = env.unwrapped.scene["robot"]
    foot_body_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    foot_sensor_ids = contact_sensor.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]

    if args_cli.switch_sequence is not None:
        execution_teacher = None
        if args_cli.dagger_execution_teacher_onnx is not None:
            if args_cli.collect_dagger_npz is None:
                raise ValueError(
                    "--dagger_execution_teacher_onnx requires --collect_dagger_npz"
                )
            shared_script_dir = Path(__file__).resolve().parents[3] / "scripts"
            sys.path.insert(0, str(shared_script_dir))
            from distill_traction_student import load_actor

            execution_teacher = load_actor(
                args_cli.dagger_execution_teacher_onnx, 641
            ).to(env.unwrapped.device).eval()
            print(
                "[info] switch trajectories execute Oracle Teacher "
                f"{args_cli.dagger_execution_teacher_onnx}"
            )
        _run_switch_evaluation(
            env,
            policy,
            robot,
            contact_sensor,
            foot_body_ids,
            foot_sensor_ids,
            execution_teacher,
        )
        env.close()
        simulation_app.close()
        return

    fields = [
        "mu",
        "cmd_vx",
        "mean_vx",
        "mean_vy",
        "mean_abs_vy",
        "mean_abs_wz",
        "mean_contact_slip",
        "mean_foot_fn",
        "mean_foot_ft",
        "mean_force_ratio",
        "mean_abs_lateral_pos",
        "mean_lateral_pos",
        "max_mean_abs_lateral_pos",
        "final_mean_abs_lateral_pos",
        "mean_abs_action",
        "done_per_env",
        "fall_per_env",
        "steps",
        "seed",
    ]
    fall_fields = [
        "mu",
        "cmd_vx",
        "seed",
        "phase",
        "step",
        "env_id",
        "termination",
        "pre_root_z",
        "pre_tilt",
        "pre_abs_wxy",
        "pre_abs_vxy",
        "pre_action_mean",
        "pre_action_max",
        "action_delay",
        "motor_strength",
        "stiffness_scale",
        "damping_scale",
        "mass_scale",
        "torso_com_x",
        "torso_com_y",
        "torso_com_z",
        "foot_sensor_delay",
        "foot_sensor_alpha",
        "foot_sensor_gain",
    ]
    rows = []
    fall_rows = []
    collected_obs = []
    collected_mu = []
    collected_cmd = []
    collected_seed = []
    dagger_policy_obs = []
    dagger_teacher_obs = []
    dagger_mu = []
    dagger_cmd = []
    dagger_seed = []
    dagger_step = []
    dagger_fall = []
    dagger_recovery = []
    dagger_sample_weight = []

    # Cache the episode-invariant dynamics draw for per-fall diagnostics.  This
    # makes rare 1/64 failures actionable instead of reducing them to one
    # aggregate scalar.  Missing fields degrade to NaN for non-robust tasks.
    uenv = env.unwrapped
    nan_vec = torch.full((uenv.num_envs,), float("nan"), device=uenv.device)

    def _env_mean_ratio(value, reference):
        try:
            value_t = torch.as_tensor(value, device=uenv.device, dtype=torch.float32)
            ref_t = torch.as_tensor(reference, device=uenv.device, dtype=torch.float32)
            valid = torch.abs(ref_t) > 1.0e-6
            ratio = torch.where(valid, value_t / ref_t, torch.nan)
            return torch.nanmean(ratio, dim=1)
        except Exception:
            return nan_vec

    motor_strength = getattr(uenv, "motor_strength_scale_buf", nan_vec)
    stiffness_scale = _env_mean_ratio(robot.data.joint_stiffness, robot.data.default_joint_stiffness)
    damping_scale = _env_mean_ratio(robot.data.joint_damping, robot.data.default_joint_damping)
    try:
        mass_scale = _env_mean_ratio(robot.root_physx_view.get_masses(), robot.data.default_mass)
    except Exception:
        mass_scale = nan_vec
    try:
        torso_id = int(robot.find_bodies(["torso_link"], preserve_order=True)[0][0])
        coms = torch.as_tensor(robot.root_physx_view.get_coms(), device=uenv.device)
        torso_com = coms[:, torso_id, :3]
    except Exception:
        torso_com = torch.full((uenv.num_envs, 3), float("nan"), device=uenv.device)
    try:
        action_term = uenv.action_manager.get_term("JointPositionAction")
    except Exception:
        action_term = None

    print(",".join(fields))
    for mu in args_cli.mu_bins:
        for vx in args_cli.vx:
            env.reset()
            if args_cli.shared_policy is not None:
                lateral_estimate = None
                last_policy_observation = None
            if args_cli.nominal_magnetic_sensor:
                uenv = env.unwrapped
                for name in (
                    "magnetic_episode_valid_lr_buf",
                    "magnetic_valid_lr_buf",
                ):
                    if hasattr(uenv, name):
                        getattr(uenv, name).fill_(1.0)
                for name in (
                    "magnetic_episode_age_lr_buf",
                    "magnetic_age_lr_buf",
                    "structured_foot_current_age_buf",
                    "structured_foot_sample_dropout_prob_buf",
                    "structured_foot_burst_dropout_prob_buf",
                    "structured_foot_burst_remaining_buf",
                ):
                    if hasattr(uenv, name):
                        getattr(uenv, name).zero_()
                if hasattr(uenv, "structured_foot_current_valid_buf"):
                    uenv.structured_foot_current_valid_buf.fill_(1.0)
                if hasattr(uenv, "magnetic_last_step_buf"):
                    uenv.magnetic_last_step_buf.fill_(-1)
                    uenv.magnetic_packet_cache = {}
            _force_mu(env, mu)
            _force_command(env, vx)
            ramp_steps = (
                args_cli.warmup_steps
                if args_cli.command_ramp_steps < 0
                else min(args_cli.command_ramp_steps, args_cli.warmup_steps)
            )
            if ramp_steps > 0:
                _set_command_value(env, 0.0)
            obs = env.get_observations()
            vx_acc = []
            vy_signed_acc = []
            vy_acc = []
            wz_acc = []
            slip_acc = []
            fn_acc = []
            ft_acc = []
            force_ratio_acc = []
            lateral_pos_acc = []
            lateral_pos_signed_acc = []
            action_acc = []
            dones_total = 0
            falls_total = 0
            recovery_remaining = torch.zeros(
                args_cli.num_envs,
                device=env.unwrapped.device,
                dtype=torch.int64,
            )
            steps = 0
            reference_xy = None
            reference_yaw = None
            total_steps = args_cli.warmup_steps + args_cli.max_steps
            for step in range(total_steps):
                if step < ramp_steps:
                    _set_command_value(env, vx * float(step + 1) / float(ramp_steps))
                with torch.inference_mode():
                    actions = policy(obs)
                if args_cli.collect_dagger_npz is not None:
                    if last_policy_observation is None:
                        raise RuntimeError("exact Student policy input was not captured")
                    dagger_policy_pre = last_policy_observation
                    dagger_teacher_pre = obs["teacher"].detach().clone()
                    dagger_recovery_pre = recovery_remaining > 0
                # Snapshot immediately before stepping because managed envs
                # reset a terminated robot inside env.step().
                pre_root_z = robot.data.root_pos_w[:, 2].clone()
                pre_tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
                pre_abs_wxy = torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)
                pre_abs_vxy = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
                pre_action_mean = torch.abs(actions).mean(dim=1)
                pre_action_max = torch.abs(actions).amax(dim=1)
                if action_term is not None and hasattr(action_term, "_delay_buffer"):
                    pre_action_delay = action_term._delay_buffer.time_lags.clone()
                else:
                    pre_action_delay = nan_vec
                pre_foot_delay = getattr(uenv, "structured_foot_delay_steps_buf", nan_vec).clone()
                pre_foot_alpha = getattr(uenv, "structured_foot_lowpass_alpha_buf", None)
                if pre_foot_alpha is None:
                    pre_foot_alpha = nan_vec
                else:
                    pre_foot_alpha = pre_foot_alpha.reshape(uenv.num_envs, -1).mean(dim=1).clone()
                pre_foot_gain = getattr(uenv, "structured_foot_gain_buf", None)
                if pre_foot_gain is None:
                    pre_foot_gain = nan_vec
                else:
                    pre_foot_gain = pre_foot_gain.reshape(uenv.num_envs, -1).mean(dim=1).clone()
                obs, rew, dones, extras = env.step(actions)

                # Count safety events over the complete commanded rollout,
                # including the command-ramp/warm-up interval.  Previously the
                # early ``continue`` below silently discarded warm-up falls,
                # which made abrupt-command acceptance results optimistic.
                dones_total += int(dones.float().sum().item())
                timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
                if timeouts is None:
                    falls = dones.bool()
                else:
                    falls = dones.bool() & ~timeouts.to(device=dones.device).bool()
                falls_total += int(falls.float().sum().item())
                if falls.any():
                    try:
                        bad_orientation = uenv.termination_manager.get_term("bad_orientation").bool()
                        low_height = uenv.termination_manager.get_term("base_height").bool()
                    except Exception:
                        bad_orientation = torch.zeros_like(falls)
                        low_height = torch.zeros_like(falls)
                    phase = "warmup" if step < args_cli.warmup_steps else "measure"
                    phase_step = step if phase == "warmup" else step - args_cli.warmup_steps
                    for env_id in torch.nonzero(falls, as_tuple=False).flatten().tolist():
                        cause = "bad_orientation" if bool(bad_orientation[env_id]) else "base_height" if bool(low_height[env_id]) else "unknown"
                        fall_rows.append(
                            {
                                "mu": f"{mu:.3f}",
                                "cmd_vx": f"{vx:.3f}",
                                "seed": args_cli.seed,
                                "phase": phase,
                                "step": phase_step,
                                "env_id": env_id,
                                "termination": cause,
                                "pre_root_z": f"{pre_root_z[env_id].item():.5f}",
                                "pre_tilt": f"{pre_tilt[env_id].item():.5f}",
                                "pre_abs_wxy": f"{pre_abs_wxy[env_id].item():.5f}",
                                "pre_abs_vxy": f"{pre_abs_vxy[env_id].item():.5f}",
                                "pre_action_mean": f"{pre_action_mean[env_id].item():.5f}",
                                "pre_action_max": f"{pre_action_max[env_id].item():.5f}",
                                "action_delay": f"{pre_action_delay[env_id].item():.0f}",
                                "motor_strength": f"{motor_strength[env_id].item():.5f}",
                                "stiffness_scale": f"{stiffness_scale[env_id].item():.5f}",
                                "damping_scale": f"{damping_scale[env_id].item():.5f}",
                                "mass_scale": f"{mass_scale[env_id].item():.5f}",
                                "torso_com_x": f"{torso_com[env_id, 0].item():.5f}",
                                "torso_com_y": f"{torso_com[env_id, 1].item():.5f}",
                                "torso_com_z": f"{torso_com[env_id, 2].item():.5f}",
                                "foot_sensor_delay": f"{pre_foot_delay[env_id].item():.0f}",
                                "foot_sensor_alpha": f"{pre_foot_alpha[env_id].item():.5f}",
                                "foot_sensor_gain": f"{pre_foot_gain[env_id].item():.5f}",
                            }
                        )

                if args_cli.collect_dagger_npz is not None:
                    if args_cli.shared_policy is None:
                        raise ValueError("--collect_dagger_npz requires --shared_policy")
                    if (
                        dagger_policy_pre.shape[1] != 1864
                        or dagger_teacher_pre.shape[1] != 641
                    ):
                        raise ValueError(
                            "DAgger observation mismatch: "
                            f"policy={dagger_policy_pre.shape}, "
                            f"teacher={dagger_teacher_pre.shape}"
                        )

                    scheduled = step % max(args_cli.collect_stride, 1) == 0
                    # Scheduled samples cover the complete state distribution.
                    # Off-stride pre-fall states are added explicitly so rare
                    # failures can never disappear between collection ticks.
                    if scheduled:
                        selected = torch.arange(
                            args_cli.num_envs, device=dones.device
                        )
                    else:
                        selected = torch.nonzero(
                            falls, as_tuple=False
                        ).flatten()
                    if selected.numel() > 0:
                        selected_fall = falls.index_select(0, selected)
                        selected_recovery = dagger_recovery_pre.index_select(
                            0, selected
                        )
                        priority = torch.ones(
                            selected.numel(),
                            device=dones.device,
                            dtype=torch.float32,
                        )
                        priority = torch.where(
                            selected_recovery,
                            torch.full_like(
                                priority, args_cli.recovery_weight
                            ),
                            priority,
                        )
                        priority = torch.where(
                            selected_fall,
                            torch.full_like(
                                priority, args_cli.failure_weight
                            ),
                            priority,
                        )
                        dagger_policy_obs.append(
                            dagger_policy_pre.index_select(0, selected)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        dagger_teacher_obs.append(
                            dagger_teacher_pre.index_select(0, selected)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        count = selected.numel()
                        dagger_mu.append(
                            np.full(count, mu, dtype=np.float32)
                        )
                        dagger_cmd.append(
                            np.full(count, vx, dtype=np.float32)
                        )
                        dagger_seed.append(
                            np.full(count, args_cli.seed, dtype=np.int32)
                        )
                        dagger_step.append(
                            np.full(count, step, dtype=np.int32)
                        )
                        dagger_fall.append(
                            selected_fall.cpu().numpy().astype(np.bool_)
                        )
                        dagger_recovery.append(
                            selected_recovery.cpu().numpy().astype(np.bool_)
                        )
                        dagger_sample_weight.append(
                            priority.cpu().numpy().astype(np.float32)
                        )

                recovery_remaining = torch.clamp(
                    recovery_remaining - 1, min=0
                )
                if falls.any():
                    recovery_remaining[falls] = max(
                        args_cli.recovery_steps, 0
                    )
                if step < args_cli.warmup_steps:
                    if step == args_cli.warmup_steps - 1:
                        reference_xy = robot.data.root_pos_w[:, :2].clone()
                        reference_yaw = _yaw_from_wxyz(robot.data.root_quat_w).clone()
                    continue
                if reference_xy is None:
                    reference_xy = robot.data.root_pos_w[:, :2].clone()
                    reference_yaw = _yaw_from_wxyz(robot.data.root_quat_w).clone()
                # Managed environments reset terminated robots inside env.step().
                # Start a new local path for them instead of counting the reset
                # position jump as lateral drift.
                done_mask = dones.bool()
                if done_mask.any():
                    reference_xy[done_mask] = robot.data.root_pos_w[done_mask, :2]
                    reference_yaw[done_mask] = _yaw_from_wxyz(
                        robot.data.root_quat_w[done_mask]
                    )
                steps += 1
                vel = robot.data.root_lin_vel_b
                vx_acc.append(vel[:, 0].mean().item())
                vy_signed_acc.append(vel[:, 1].mean().item())
                vy_acc.append(torch.abs(vel[:, 1]).mean().item())
                wz_acc.append(torch.abs(robot.data.root_ang_vel_b[:, 2]).mean().item())
                foot_vel = robot.data.body_lin_vel_w[:, foot_body_ids, :2]
                foot_speed = torch.linalg.norm(foot_vel, dim=-1)
                fn = torch.abs(contact_sensor.data.net_forces_w[:, foot_sensor_ids, 2])
                ft = torch.linalg.norm(
                    contact_sensor.data.net_forces_w[:, foot_sensor_ids, :2], dim=-1
                )
                contact = (fn > 5.0).float()
                contact_slip = (foot_speed * contact).sum() / contact.sum().clamp(min=1.0)
                slip_acc.append(contact_slip.item())
                fn_acc.append(fn.sum(dim=1).mean().item())
                ft_acc.append(ft.sum(dim=1).mean().item())
                force_ratio_acc.append((ft / (fn + 5.0)).mean().item())
                displacement = robot.data.root_pos_w[:, :2] - reference_xy
                # Project world displacement onto the lateral axis at the start
                # of this measured path. This remains valid with randomized yaw.
                local_y = (
                    -torch.sin(reference_yaw) * displacement[:, 0]
                    + torch.cos(reference_yaw) * displacement[:, 1]
                )
                lateral_pos_acc.append(torch.abs(local_y).mean().item())
                lateral_pos_signed_acc.append(local_y.mean().item())
                action_acc.append(torch.abs(actions).mean().item())
                if args_cli.collect_npz is not None and step % max(args_cli.collect_stride, 1) == 0:
                    actor_obs = obs["policy"]
                    collected_obs.append(actor_obs.detach().cpu().numpy().astype(np.float32))
                    collected_mu.append(np.full(args_cli.num_envs, mu, dtype=np.float32))
                    collected_cmd.append(np.full(args_cli.num_envs, vx, dtype=np.float32))
                    collected_seed.append(np.full(args_cli.num_envs, args_cli.seed, dtype=np.int32))
            mean = lambda xs: sum(xs) / max(len(xs), 1)
            row = {
                "mu": f"{mu:.3f}",
                "cmd_vx": f"{vx:.3f}",
                "mean_vx": f"{mean(vx_acc):.4f}",
                "mean_vy": f"{mean(vy_signed_acc):.4f}",
                "mean_abs_vy": f"{mean(vy_acc):.4f}",
                "mean_abs_wz": f"{mean(wz_acc):.4f}",
                "mean_contact_slip": f"{mean(slip_acc):.4f}",
                "mean_foot_fn": f"{mean(fn_acc):.4f}",
                "mean_foot_ft": f"{mean(ft_acc):.4f}",
                "mean_force_ratio": f"{mean(force_ratio_acc):.4f}",
                "mean_abs_lateral_pos": f"{mean(lateral_pos_acc):.4f}",
                "mean_lateral_pos": f"{mean(lateral_pos_signed_acc):.4f}",
                "max_mean_abs_lateral_pos": f"{max(lateral_pos_acc, default=0.0):.4f}",
                "final_mean_abs_lateral_pos": (
                    f"{lateral_pos_acc[-1] if lateral_pos_acc else 0.0:.4f}"
                ),
                "mean_abs_action": f"{mean(action_acc):.4f}",
                "done_per_env": f"{dones_total / args_cli.num_envs:.4f}",
                "fall_per_env": f"{falls_total / args_cli.num_envs:.4f}",
                "steps": steps,
                "seed": args_cli.seed,
            }
            rows.append(row)
            print(",".join(str(row[name]) for name in fields), flush=True)

    if args_cli.output_csv is not None:
        args_cli.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args_cli.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[info] evaluation CSV: {args_cli.output_csv}")
        fall_csv = args_cli.output_csv.with_suffix(".falls.csv")
        with fall_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fall_fields)
            writer.writeheader()
            writer.writerows(fall_rows)
        print(f"[info] fall diagnostics: {fall_csv} ({len(fall_rows)} events)")

    if args_cli.collect_npz is not None:
        args_cli.collect_npz.parent.mkdir(parents=True, exist_ok=True)
        if not collected_obs:
            raise RuntimeError("no policy observations were collected")
        np.savez_compressed(
            args_cli.collect_npz,
            obs=np.concatenate(collected_obs, axis=0),
            mu=np.concatenate(collected_mu, axis=0),
            cmd_vx=np.concatenate(collected_cmd, axis=0),
            seed=np.concatenate(collected_seed, axis=0),
        )
        print(
            f"[info] observation dataset: {args_cli.collect_npz} "
            f"shape={np.concatenate(collected_mu).shape[0]}x{collected_obs[0].shape[1]}"
        )
    if args_cli.collect_dagger_npz is not None:
        args_cli.collect_dagger_npz.parent.mkdir(parents=True, exist_ok=True)
        if not dagger_policy_obs:
            raise RuntimeError("no DAgger observations were collected")
        np.savez_compressed(
            args_cli.collect_dagger_npz,
            obs=np.concatenate(dagger_policy_obs, axis=0),
            teacher_obs=np.concatenate(dagger_teacher_obs, axis=0),
            mu=np.concatenate(dagger_mu, axis=0),
            cmd_vx=np.concatenate(dagger_cmd, axis=0),
            seed=np.concatenate(dagger_seed, axis=0),
            step=np.concatenate(dagger_step, axis=0),
            fall=np.concatenate(dagger_fall, axis=0),
            recovery=np.concatenate(dagger_recovery, axis=0),
            sample_weight=np.concatenate(
                dagger_sample_weight, axis=0
            ),
        )
        fall_count = int(np.concatenate(dagger_fall).sum())
        recovery_count = int(np.concatenate(dagger_recovery).sum())
        print(
            f"[info] DAgger dataset: {args_cli.collect_dagger_npz} "
            f"shape={sum(len(item) for item in dagger_policy_obs)}x1864 "
            f"pre_fall={fall_count} recovery={recovery_count}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
