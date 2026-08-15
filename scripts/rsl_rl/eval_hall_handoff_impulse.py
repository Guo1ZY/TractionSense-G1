#!/usr/bin/env python3
"""Evaluate a Hall-only G1 policy after a high-momentum low-grip handoff.

The evaluator first lets the selected policy settle at an exact low-friction
coefficient and a fixed command.  It then adds a body-forward velocity impulse
(``delta-v`` in m/s) without resetting observation histories.  This reproduces
the state-distribution hole exposed by the high->low hybrid controller: the
recovery actor receives real Hall/proprioceptive history while inheriting
substantial forward momentum.

Only the simulator/evaluator uses exact friction and termination state.  The
actor is runtime-audited to consume the exact 1864-D Hall/proprioception policy
group; contact force, true friction, and slip are never actor inputs.

Example:

  python scripts/rsl_rl/eval_hall_handoff_impulse.py \
    --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-LowGripRecovery \
    --checkpoint logs/rsl_rl/.../model_5850.pt \
    --vx_impulses 0.35 0.55 0.75 --mu_bins 0.14 0.17 0.20 \
    --command 0.16 --num_envs 64 --seed 390 --headless

For the paired original-policy pass, keep every environment argument identical
and replace ``--checkpoint ...`` with
``--proprio_baseline_checkpoint model/rl/model_49999.pt``.  For the strongest
case-level pairing, invoke one ``mu``/``delta-v`` cell per fresh process so a
policy-dependent managed reset in one cell cannot advance the random stream of
the next cell.

This script never sends commands to a physical robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

# Match the established train/play import path without importing Isaac before
# AppLauncher has created the SimulationApp.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(
    description="Hall-only Stage7 high-momentum handoff evaluation"
)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=390)
parser.add_argument(
    "--vx_impulses",
    type=float,
    nargs="+",
    default=[0.35, 0.55, 0.75],
    help="Body-forward velocity increments in m/s (these are delta-v bins).",
)
parser.add_argument(
    "--mu_bins",
    type=float,
    nargs="+",
    default=[0.14, 0.17, 0.20],
    help="Exact static/dynamic friction coefficients used by physics only.",
)
parser.add_argument("--command", type=float, default=0.16)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=100,
    help="Policy steps used to establish the low-grip gait before delta-v.",
)
parser.add_argument(
    "--measure_steps",
    type=int,
    default=150,
    help="Policy steps recorded after delta-v (150 is about 3 s at 50 Hz).",
)
parser.add_argument(
    "--one_second_speed_limit",
    type=float,
    default=0.25,
    help="Absolute body-forward speed required one second after delta-v.",
)
parser.add_argument("--max_abs_roll", type=float, default=0.45)
parser.add_argument("--max_abs_pitch", type=float, default=0.45)
parser.add_argument(
    "--nominal_hall",
    action="store_true",
    help="Disable Hall domain randomization for a nominal-sensor pass.",
)
parser.add_argument(
    "--output_csv",
    type=Path,
    default=None,
    help="Per-case CSV output path; a JSON manifest is written beside it.",
)
parser.add_argument(
    "--fail_on_gate",
    action="store_true",
    help="Exit with status 2 when any matrix case fails a safety gate.",
)
parser.add_argument(
    "--proprio_baseline_checkpoint",
    type=Path,
    default=None,
    help=(
        "Evaluate the original 480-D proprioceptive actor (for example "
        "model/rl/model_49999.pt) through the same 1864-D Hall environment. "
        "This is mutually exclusive with --checkpoint."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if bool(args_cli.checkpoint) == bool(args_cli.proprio_baseline_checkpoint):
    parser.error(
        "select exactly one policy: --checkpoint for the Hall actor or "
        "--proprio_baseline_checkpoint for the original 480-D actor"
    )
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.warmup_steps < 0 or args_cli.measure_steps <= 0:
    parser.error("--warmup_steps must be >= 0 and --measure_steps must be positive")
if any(value < 0.0 for value in args_cli.vx_impulses):
    parser.error("--vx_impulses must contain non-negative delta-v values")
if any(value <= 0.0 for value in args_cli.mu_bins):
    parser.error("--mu_bins must contain positive friction coefficients")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.sensors import (  # noqa: E402
    audit_hall_sensor_cfg_policy_terms,
    sync_hall_sensor_cfg_to_policy_terms,
)
from unitree_rl_lab.traction.handoff_metrics import (  # noqa: E402
    body_forward_axis_world,
    one_second_deceleration,
    roll_pitch_from_wxyz,
)
from unitree_rl_lab.traction.proprio_baseline import (  # noqa: E402
    HALL_POLICY_DIM,
    LEGACY_PROPRIO_DIM,
    load_proprio_baseline,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


EXPECTED_POLICY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "foot_magnetic_array",
    "foot_sample_period_lr",
    "foot_sensor_valid_lr",
    "foot_sensor_age_lr",
)
FORBIDDEN_POLICY_TOKENS = (
    "contact",
    "contact_force",
    "foot_force",
    "ground_friction",
    "friction_mu",
    "slip",
)


def _policy_tensor(
    observation, consumed_observation_dimension: int = HALL_POLICY_DIM
) -> torch.Tensor:
    try:
        tensor = observation["policy"]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("environment did not return a policy observation group") from exc
    # Keep the Hall environment boundary literal and explicit.  Baseline mode
    # adapts *inside* this interface; it never weakens the 1864-D contract.
    if tensor.ndim != 2 or tensor.shape[1] != 1864:
        raise RuntimeError(
            "Stage7 handoff evaluation requires the exact deployable 1864-D "
            f"Hall actor observation, got {tuple(tensor.shape)}"
        )
    if consumed_observation_dimension not in (
        LEGACY_PROPRIO_DIM,
        HALL_POLICY_DIM,
    ):
        raise ValueError(
            "consumed observation dimension must be 480 or 1864, got "
            f"{consumed_observation_dimension}"
        )
    if not torch.isfinite(tensor[:, :consumed_observation_dimension]).all():
        raise FloatingPointError(
            "non-finite value in the consumed policy observation prefix "
            f"[0:{consumed_observation_dimension}]"
        )
    return tensor


def _audit_policy_terms(env) -> tuple[str, ...]:
    manager = env.unwrapped.observation_manager
    active_terms = getattr(manager, "active_terms", {})
    terms = tuple(active_terms.get("policy", ()))
    if terms != EXPECTED_POLICY_TERMS:
        raise RuntimeError(
            "Hall-only policy term mismatch: "
            f"expected={EXPECTED_POLICY_TERMS}, active={terms}"
        )
    lowered = tuple(term.lower() for term in terms)
    forbidden = [
        term
        for term in lowered
        if any(token in term for token in FORBIDDEN_POLICY_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"privileged actor observation terms are forbidden: {forbidden}")
    return terms


def _force_command(env, command: float) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = float(command)
    term.vel_command_b[:, 1:] = 0.0


def _force_mu(env, mu: float) -> None:
    """Set exact physics friction and synchronize privileged reward buffers."""

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    env_ids_cpu = torch.arange(uenv.num_envs, device="cpu")
    materials = robot.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = float(mu)
    materials[env_ids_cpu, :, 1] = float(mu)
    materials[env_ids_cpu, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
    if not hasattr(uenv, "ground_friction_mu_buf"):
        uenv.ground_friction_mu_buf = torch.full(
            (uenv.num_envs,), float(mu), device=uenv.device
        )
    uenv.ground_friction_mu_buf.fill_(float(mu))
    if hasattr(uenv, "effective_friction_mu_buf"):
        uenv.effective_friction_mu_buf.fill_(float(mu))
    if hasattr(uenv, "ground_friction_regime_buf"):
        regime = 0 if mu <= 0.25 else 2 if mu >= 0.75 else 1
        uenv.ground_friction_regime_buf.fill_(regime)


def _hall_cfg_audit_payload(env_cfg) -> dict[str, object]:
    """Return the effective per-term Hall DR settings recorded in the manifest."""

    term_names = audit_hall_sensor_cfg_policy_terms(
        env_cfg.observations, env_cfg.hall_sensor_cfg
    )
    cfg = env_cfg.hall_sensor_cfg
    return {
        "synced_policy_terms": list(term_names),
        "enable_domain_randomization": bool(cfg.enable_domain_randomization),
        "foot_dropout_probability": float(cfg.foot_dropout_probability),
        "dead_channel_probability": float(cfg.dead_channel_probability),
        "maximum_packet_delay_steps": int(cfg.maximum_packet_delay_steps),
        "observation_cross_axis_std": float(cfg.observation_cross_axis_std),
        "observation_zero_residual_std": float(cfg.observation_zero_residual_std),
    }


def _apply_body_forward_delta_v(robot, delta_v: float) -> torch.Tensor:
    root_velocity_w = robot.data.root_vel_w.clone()
    forward_w = body_forward_axis_world(robot.data.root_quat_w)
    root_velocity_w[:, :3] += float(delta_v) * forward_w
    robot.write_root_velocity_to_sim(root_velocity_w)
    return robot.data.root_lin_vel_b[:, 0].detach().clone()


def _fall_mask(dones: torch.Tensor, extras) -> torch.Tensor:
    timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
    if timeouts is None:
        return dones.bool()
    return dones.bool() & ~timeouts.to(device=dones.device).bool()


def _finite_runtime_state(
    observation,
    actions: torch.Tensor,
    robot,
    consumed_observation_dimension: int = HALL_POLICY_DIM,
) -> tuple[bool, str]:
    checks = (
        (
            "policy_observation",
            observation["policy"][:, :consumed_observation_dimension],
        ),
        ("action", actions),
        ("root_pose", robot.data.root_state_w[:, :7]),
        ("root_velocity", robot.data.root_state_w[:, 7:]),
        ("joint_position", robot.data.joint_pos),
        ("joint_velocity", robot.data.joint_vel),
    )
    for name, tensor in checks:
        if not torch.isfinite(tensor).all():
            return False, name
    return True, ""


def _float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.8g}"


def _strict_json(value):
    """Replace non-finite floats with JSON ``null`` recursively."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return value


def _run_case(
    env,
    policy,
    mu: float,
    delta_v: float,
    *,
    consumed_observation_dimension: int = HALL_POLICY_DIM,
) -> dict[str, object]:
    env.reset()
    _force_mu(env, mu)
    _force_command(env, args_cli.command)
    observation = env.get_observations()
    _policy_tensor(observation, consumed_observation_dimension)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = float(uenv.step_dt)

    warmup_falls = 0
    warmup_failed = torch.zeros(
        uenv.num_envs, dtype=torch.bool, device=uenv.device
    )
    post_reset_fall_events = 0
    nan_detected = False
    first_nan_s = float("nan")
    first_nan_component = ""
    for step in range(args_cli.warmup_steps):
        _force_command(env, args_cli.command)
        _policy_tensor(observation, consumed_observation_dimension)
        with torch.inference_mode():
            actions = policy(observation)
        finite, component = _finite_runtime_state(
            observation,
            actions,
            robot,
            consumed_observation_dimension,
        )
        if not finite:
            nan_detected = True
            first_nan_s = step * dt
            first_nan_component = f"warmup:{component}"
            break
        observation, _, dones, extras = env.step(actions)
        warmup_step_falls = _fall_mask(dones, extras)
        post_reset_fall_events += int(
            (warmup_step_falls & warmup_failed).sum().item()
        )
        warmup_falls += int(warmup_step_falls.sum().item())
        warmup_failed |= warmup_step_falls

    if nan_detected:
        return {
            "mu": mu,
            "delta_v_m_s": delta_v,
            "command_m_s": args_cli.command,
            "warmup_falls": warmup_falls,
            "warmup_fall_envs": int(warmup_failed.sum().item()),
            "eligible_envs": int((~warmup_failed).sum().item()),
            "falls": 0,
            "fall_envs": 0,
            "post_reset_fall_events": post_reset_fall_events,
            "first_fall_s": float("nan"),
            "first_fall_env": -1,
            "initial_vx_mean_m_s": float("nan"),
            "vx_1s_mean_m_s": float("nan"),
            "decel_1s_mean_m_s": float("nan"),
            "speed_1s_pass_fraction": 0.0,
            "max_abs_roll_rad": float("nan"),
            "max_abs_pitch_rad": float("nan"),
            "nan_detected": True,
            "first_nan_s": first_nan_s,
            "first_nan_component": first_nan_component,
            "gate_pass": False,
        }

    _force_command(env, args_cli.command)
    initial_forward_speed = _apply_body_forward_delta_v(robot, delta_v)
    eligible = ~warmup_failed
    one_second_step = max(int(math.ceil(1.0 / dt)), 1)
    if args_cli.measure_steps < one_second_step:
        raise ValueError(
            f"--measure_steps={args_cli.measure_steps} is shorter than one second "
            f"({one_second_step} policy steps at dt={dt:.6f})"
        )

    fallen = torch.zeros(uenv.num_envs, dtype=torch.bool, device=uenv.device)
    one_second_survivor = torch.zeros_like(fallen)
    falls_total = 0
    first_fall_s = float("nan")
    first_fall_env = -1
    vx_after_one_second = torch.full_like(initial_forward_speed, float("nan"))
    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    for step in range(args_cli.measure_steps):
        _force_command(env, args_cli.command)
        try:
            _policy_tensor(observation, consumed_observation_dimension)
        except FloatingPointError:
            nan_detected = True
            first_nan_s = step * dt
            first_nan_component = "measure:policy_observation"
            break
        with torch.inference_mode():
            actions = policy(observation)
        finite, component = _finite_runtime_state(
            observation,
            actions,
            robot,
            consumed_observation_dimension,
        )
        if not finite:
            nan_detected = True
            first_nan_s = step * dt
            first_nan_component = f"measure:{component}"
            break

        # Capture diagnostics before ManagerBasedRLEnv resets terminated rows.
        roll, pitch = roll_pitch_from_wxyz(robot.data.root_quat_w)
        alive = eligible & ~fallen
        if torch.any(alive):
            max_abs_roll = max(max_abs_roll, float(torch.abs(roll[alive]).max().item()))
            max_abs_pitch = max(max_abs_pitch, float(torch.abs(pitch[alive]).max().item()))

        observation, _, dones, extras = env.step(actions)
        falls = _fall_mask(dones, extras)
        eligible_falls = falls & eligible
        post_reset_fall_events += int(
            (falls & (warmup_failed | fallen)).sum().item()
        )
        falls_total += int(eligible_falls.sum().item())
        new_falls = eligible_falls & ~fallen
        if torch.any(new_falls) and not math.isfinite(first_fall_s):
            first_fall_s = (step + 1) * dt
            first_fall_env = int(torch.nonzero(new_falls, as_tuple=False)[0].item())
        fallen |= eligible_falls

        if step + 1 == one_second_step:
            current_vx = robot.data.root_lin_vel_b[:, 0].detach().clone()
            one_second_survivor = eligible & ~fallen
            vx_after_one_second[one_second_survivor] = current_vx[
                one_second_survivor
            ]

    deceleration = one_second_deceleration(
        initial_forward_speed,
        vx_after_one_second,
        # Later failures still fail the rollout gate, but must not erase a
        # valid one-second measurement that was observed before that failure.
        valid_mask=one_second_survivor,
        speed_limit=args_cli.one_second_speed_limit,
    )
    gate_pass = (
        warmup_falls == 0
        and not torch.any(fallen).item()
        and not nan_detected
        and deceleration["pass_fraction"] >= 1.0 - 1.0e-8
        and max_abs_roll <= args_cli.max_abs_roll
        and max_abs_pitch <= args_cli.max_abs_pitch
    )
    return {
        "mu": mu,
        "delta_v_m_s": delta_v,
        "command_m_s": args_cli.command,
        "warmup_falls": warmup_falls,
        "warmup_fall_envs": int(warmup_failed.sum().item()),
        "eligible_envs": int(eligible.sum().item()),
        "falls": falls_total,
        "fall_envs": int(fallen.sum().item()),
        "post_reset_fall_events": post_reset_fall_events,
        "first_fall_s": first_fall_s,
        "first_fall_env": first_fall_env,
        "initial_vx_mean_m_s": deceleration["initial_mean_m_s"],
        "vx_1s_mean_m_s": deceleration["after_mean_m_s"],
        "decel_1s_mean_m_s": deceleration["reduction_mean_m_s"],
        "speed_1s_pass_fraction": deceleration["pass_fraction"],
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
        "nan_detected": nan_detected,
        "first_nan_s": first_nan_s,
        "first_nan_component": first_nan_component,
        "gate_pass": bool(gate_pass),
    }


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="env_cfg_entry_point",
    )
    if getattr(env_cfg, "sensor_output_mode", "hall") != "hall":
        raise RuntimeError("Stage7 evaluator only accepts sensor_output_mode='hall'")
    env_cfg.seed = args_cli.seed
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None
    for event_name in ("physics_material_reset", "friction_switch", "push_robot"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    if args_cli.nominal_hall and hasattr(env_cfg, "hall_sensor_cfg"):
        env_cfg.hall_sensor_cfg.enable_domain_randomization = False
    if not hasattr(env_cfg, "hall_sensor_cfg"):
        raise RuntimeError("Stage7 Hall evaluator requires env_cfg.hall_sensor_cfg")
    sync_hall_sensor_cfg_to_policy_terms(
        env_cfg.observations, env_cfg.hall_sensor_cfg
    )
    hall_cfg_audit = _hall_cfg_audit_payload(env_cfg)
    if args_cli.nominal_hall and hall_cfg_audit["enable_domain_randomization"]:
        raise RuntimeError("--nominal_hall failed to disable policy-term Hall DR")

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy_terms = _audit_policy_terms(env)
    action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
    try:
        action_scale = float(action_term.cfg.scale)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "handoff comparison requires a scalar JointPositionAction scale"
        ) from exc

    baseline_mode = args_cli.proprio_baseline_checkpoint is not None
    empirical_normalization = bool(
        getattr(agent_cfg, "empirical_normalization", False)
    )
    if baseline_mode and empirical_normalization:
        raise RuntimeError(
            "the original model_49999 actor has no empirical-normalizer state; "
            "baseline comparison requires empirical_normalization=False"
        )
    if baseline_mode and not math.isclose(
        action_scale, 0.25, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(
            "model_49999 expects JointPositionAction scale=0.25 rad, got "
            f"{action_scale}"
        )

    # Construct the identical 1864-D runner in both modes.  Besides preserving
    # the established Hall path, this keeps policy-construction RNG consumption
    # identical before the first environment reset.  The 480-D checkpoint is
    # intentionally never loaded into this runner, which avoids input-layer
    # expansion or partial-load ambiguity.
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    if baseline_mode:
        checkpoint = (
            Path(args_cli.proprio_baseline_checkpoint).expanduser().resolve()
        )
        # LegacyLocomotionActor construction initializes parameters before the
        # strict state load.  Restore CPU RNG afterward so policy choice cannot
        # alter the environment randomization sequence.
        with torch.random.fork_rng(devices=[]):
            policy = load_proprio_baseline(
                checkpoint, device=env.unwrapped.device
            )
        policy_kind = "original_proprioceptive_baseline"
        consumed_observation_dimension = LEGACY_PROPRIO_DIM
        print(f"[info] load 480-D proprio baseline checkpoint: {checkpoint}")
    else:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[info] load Hall-only checkpoint: {checkpoint}")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_kind = "hall_actor"
        consumed_observation_dimension = HALL_POLICY_DIM

    if args_cli.output_csv is None:
        output_stem = (
            "stage7_handoff_impulse_proprio_baseline"
            if baseline_mode
            else "stage7_handoff_impulse"
        )
        output_csv = Path(
            f"artifacts/hall_speed_demo/{output_stem}_seed{args_cli.seed}.csv"
        )
    else:
        output_csv = args_cli.output_csv
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    print(
        "mu,delta_v_m_s,warmup_falls,falls,fall_envs,first_fall_s,"
        "initial_vx,vx_1s,decel_1s,speed_1s_pass,max_roll,max_pitch,nan,gate"
    )
    for mu in args_cli.mu_bins:
        for delta_v in args_cli.vx_impulses:
            row = _run_case(
                env,
                policy,
                float(mu),
                float(delta_v),
                consumed_observation_dimension=(
                    consumed_observation_dimension
                ),
            )
            rows.append(row)
            print(
                f"{mu:.3f},{delta_v:.3f},{row['warmup_falls']},{row['falls']},"
                f"{row['fall_envs']},{_float(float(row['first_fall_s']))},"
                f"{_float(float(row['initial_vx_mean_m_s']))},"
                f"{_float(float(row['vx_1s_mean_m_s']))},"
                f"{_float(float(row['decel_1s_mean_m_s']))},"
                f"{_float(float(row['speed_1s_pass_fraction']))},"
                f"{_float(float(row['max_abs_roll_rad']))},"
                f"{_float(float(row['max_abs_pitch_rad']))},"
                f"{int(bool(row['nan_detected']))},{'PASS' if row['gate_pass'] else 'FAIL'}"
            )

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = output_csv.with_suffix(".json")
    payload = {
        "status": "PASS" if all(bool(row["gate_pass"]) for row in rows) else "FAIL",
        "measurement_boundary": (
            "environment input is exactly 1864-D Hall Bx/By/Bz history plus "
            "proprioception; the Hall actor consumes all 1864 dimensions while "
            "the original baseline consumes only the audited first 480; friction "
            "and termination are evaluator-only labels"
        ),
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "policy_kind": policy_kind,
        "environment_observation_dimension": HALL_POLICY_DIM,
        "consumed_observation_dimension": consumed_observation_dimension,
        "empirical_normalization": empirical_normalization,
        "clip_actions": agent_cfg.clip_actions,
        "joint_position_action_scale_rad": action_scale,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "policy_terms": policy_terms,
        "consumed_policy_terms": (
            policy_terms[:6] if baseline_mode else policy_terms
        ),
        "reset_censoring": (
            "warm-up first-fall environments and every post-first-fall managed "
            "reset state are excluded from primary impulse metrics"
        ),
        "paired_case_protocol": (
            "use identical task/seed/mu/delta-v/command/Hall settings; for exact "
            "case pairing run one mu/delta-v cell per fresh process"
        ),
        "config": {
            "vx_impulses_m_s": args_cli.vx_impulses,
            "mu_bins": args_cli.mu_bins,
            "command_m_s": args_cli.command,
            "warmup_steps": args_cli.warmup_steps,
            "measure_steps": args_cli.measure_steps,
            "one_second_speed_limit_m_s": args_cli.one_second_speed_limit,
            "max_abs_roll_rad": args_cli.max_abs_roll,
            "max_abs_pitch_rad": args_cli.max_abs_pitch,
            "nominal_hall": args_cli.nominal_hall,
            "effective_hall_cfg": hall_cfg_audit,
        },
        "rows": rows,
    }
    manifest.write_text(
        json.dumps(_strict_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[info] CSV: {output_csv}")
    print(f"[info] JSON: {manifest}")
    env.close()
    failed = payload["status"] != "PASS"
    return 2 if failed and args_cli.fail_on_gate else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
