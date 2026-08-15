#!/usr/bin/env python3
"""Fair long-horizon high-friction evaluation for the G1 Hall backbone.

The physical course is forty metres of identical ``mu=0.90`` static ground.
There is no LOW patch, friction switch, course-success truncation, command
governor, or stability envelope.  This isolates the late heading/lateral
instability of the locomotion backbone from Hall traction adaptation.

All policies run behind the exact same 1864-D Hall/proprio environment:

* ``--proprio_baseline_checkpoint`` strictly consumes only columns ``0:480``
  from the original Unitree ``model_49999.pt`` actor.
* ``--checkpoint`` strictly loads a native 1864-D Hall actor.
* ``--high_speed_backbone_checkpoint`` strictly loads the isolated 482-D
  high-speed actor: the same 480-D prefix plus current body-y velocity and
  relative heading.  The complete Hall group remains available in parallel.
* ``--holosoma_onnx`` reconstructs HoloSoma FastSAC's audited 100-D
  proprioceptive interface from the live articulation.  It consumes no Hall or
  privileged ground/contact truth and maps its 29 joint targets through the
  same Isaac action term used by every other policy.

The evaluator permanently censors an environment at its first terminal event.
Managed-reset motion is never allowed to improve speed or stability metrics.
This script never sends commands to a physical robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from importlib.metadata import version
from pathlib import Path
import sys
import time

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
    "UniformHighFrictionLongBackbone482"
)
POLICY_DIM = 1864
LEGACY_DIM = 480
HIGH_SPEED_DIM = 482
ACTION_DIM = 29
COMMAND_SLICE = slice(30, 45)
MOTION_FEEDBACK_SLICE = slice(1862, 1864)
EVAL_ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}
TRACE_COLUMNS = (
    "step",
    "time_s",
    "env_id",
    "vx",
    "vy",
    "heading",
    "omega_x",
    "omega_y",
    "omega_z",
    "tilt",
    "action_norm",
    "action_slew",
    "action_saturation_fraction",
    "hall_valid_left",
    "hall_valid_right",
)
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
    # Historical field name retained by the trained 1864-D ABI.  The exact
    # callable and runtime value are audited below and must be
    # lateral_motion_feedback=[body_vy, relative_heading], never packet age.
    "foot_sensor_age_lr",
)
EXPECTED_POLICY_FUNCTIONS = (
    "base_ang_vel",
    "projected_gravity",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "hall_magnetic_array",
    "hall_sample_period_lr",
    "hall_sensor_valid_lr",
    "lateral_motion_feedback",
)
EXPECTED_POLICY_TERM_DIMS = (
    (15,),
    (15,),
    (15,),
    (145,),
    (145,),
    (145,),
    (1350,),
    (30,),
    (2,),
    (2,),
)
EXPECTED_POLICY_HISTORY_LENGTHS = (5, 5, 5, 5, 5, 5, 15, 15, 0, 0)
EXPECTED_POLICY_SLICES = (
    (0, 15),
    (15, 30),
    (30, 45),
    (45, 190),
    (190, 335),
    (335, 480),
    (480, 1830),
    (1830, 1860),
    (1860, 1862),
    (1862, 1864),
)
EXPECTED_HIGH_SPEED_TERMS = EXPECTED_POLICY_TERMS[:6] + ("foot_sensor_age_lr",)
EXPECTED_HIGH_SPEED_FUNCTIONS = EXPECTED_POLICY_FUNCTIONS[:6] + (
    "lateral_motion_feedback",
)
EXPECTED_HIGH_SPEED_TERM_DIMS = EXPECTED_POLICY_TERM_DIMS[:6] + ((2,),)
EXPECTED_HIGH_SPEED_HISTORY_LENGTHS = EXPECTED_POLICY_HISTORY_LENGTHS[:6] + (0,)
FORBIDDEN_POLICY_TOKENS = (
    "contact",
    "force",
    "friction",
    "ground_mu",
    "slip",
    "stage",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--steps",
    type=int,
    default=1500,
    help="Policy steps; 1500 is 30 seconds at the audited 50 Hz policy rate.",
)
parser.add_argument("--seed", type=int, default=550)
parser.add_argument("--command", type=float, default=0.80)
parser.add_argument(
    "--floor_width_m",
    type=float,
    default=None,
    help=(
        "Evaluation-only width override applied equally to all three coplanar "
        "floor patches.  Default None preserves the task's 3.2 m width."
    ),
)
parser.add_argument(
    "--floor_length_m",
    type=float,
    default=None,
    help=(
        "Evaluation-only runout override for the trailing high-friction patch "
        "so 70 s rollouts cannot walk off the forward edge.  Default None "
        "preserves the task's authored 40 m layout."
    ),
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help=(
        "Evaluation-only episode length override so 70 s rollouts are not "
        "truncated by the task's authored 30 s horizon.  Default None keeps "
        "the task value."
    ),
)
parser.add_argument(
    "--rsl_rl_cfg_entry_point",
    default=None,
    help=(
        "Agent config entry point used to load --checkpoint.  Required for "
        "FastBase gate/residual checkpoints evaluated on a task whose default "
        "agent config uses a plain MLP actor."
    ),
)
parser.add_argument(
    "--metric_warmup_steps",
    type=int,
    default=100,
    help="Exclude startup from steady-state metrics, never from fall counting.",
)
parser.add_argument(
    "--proprio_baseline_checkpoint",
    type=Path,
    default=None,
    help="Original 480-D model_49999 checkpoint; exclusive with --checkpoint.",
)
parser.add_argument(
    "--high_speed_backbone_checkpoint",
    type=Path,
    default=None,
    help=(
        "Native 482-D high-speed checkpoint; exclusive with --checkpoint and "
        "--proprio_baseline_checkpoint."
    ),
)
parser.add_argument(
    "--holosoma_onnx",
    type=Path,
    default=None,
    help=(
        "Bundled HoloSoma G1 FastSAC ONNX; exclusive with all RSL checkpoints. "
        "This is an Isaac sim-to-sim baseline, not the MuJoCo result."
    ),
)
parser.add_argument(
    "--training_profile",
    action="store_true",
    help="Use the training env (targeted pushes/material DR) instead of play cfg.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
    "--hardened_hall",
    action="store_true",
    help="Enable strong Hall dropout/dead-channel/delay randomization.",
)
parser.add_argument(
    "--hall_contact_distribution",
    choices=("aggregate", "detailed"),
    default=None,
    help="Override only the Hall mechanical contact distribution for an A/B run.",
)
parser.add_argument("--summary_json", type=Path, required=True)
parser.add_argument("--trace_npz", type=Path, default=None)
parser.add_argument("--video", action="store_true", help="Record the natural rollout.")
parser.add_argument("--video_length", type=int, default=1500)
parser.add_argument(
    "--video_dir",
    type=Path,
    default=Path("artifacts/uniform_high_friction_video"),
)
parser.add_argument(
    "--video_eye",
    type=str,
    default=None,
    help="Comma-separated x,y,z viewer eye override.",
)
parser.add_argument(
    "--video_lookat",
    type=str,
    default=None,
    help="Comma-separated x,y,z viewer look-at override.",
)
parser.add_argument("--minimum_mean_vx", type=float, default=0.69)
parser.add_argument("--minimum_per_env_mean_vx", type=float, default=0.65)
parser.add_argument("--maximum_heading_rms", type=float, default=0.25)
parser.add_argument("--maximum_body_vy_rms", type=float, default=0.25)
parser.add_argument("--maximum_angular_velocity_rms", type=float, default=1.50)
parser.add_argument("--maximum_action_saturation_fraction", type=float, default=0.05)
parser.add_argument("--fail_on_gate", action="store_true")
parser.add_argument(
    "--print_progress",
    action="store_true",
    help="Print construction and rollout timing for throughput diagnosis.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if sum(
    path is not None
    for path in (
        args_cli.checkpoint,
        args_cli.proprio_baseline_checkpoint,
        args_cli.high_speed_backbone_checkpoint,
        args_cli.holosoma_onnx,
    )
) != 1:
    parser.error(
        "select exactly one policy: --checkpoint for a Hall actor, "
        "--proprio_baseline_checkpoint for model_49999, or "
        "--high_speed_backbone_checkpoint for the 482-D actor, or "
        "--holosoma_onnx for HoloSoma FastSAC"
    )
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num_envs and --steps must be positive")
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if not 0 <= args_cli.metric_warmup_steps < args_cli.steps:
    parser.error("--metric_warmup_steps must be in [0, steps)")
for name in (
    "command",
    "minimum_mean_vx",
    "minimum_per_env_mean_vx",
    "maximum_heading_rms",
    "maximum_body_vy_rms",
    "maximum_angular_velocity_rms",
    "maximum_action_saturation_fraction",
):
    if not math.isfinite(float(getattr(args_cli, name))):
        parser.error(f"--{name} must be finite")

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402
import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion import mdp  # noqa: E402
from unitree_rl_lab.sensors import (  # noqa: E402
    audit_hall_sensor_cfg_policy_terms,
    sync_hall_sensor_cfg_to_policy_terms,
)
from unitree_rl_lab.traction.proprio_baseline import (  # noqa: E402
    load_high_speed_backbone,
    load_hall_backbone,
    load_proprio_baseline,
)
from unitree_rl_lab.traction.holosoma_baseline import (  # noqa: E402
    HOLOSOMA_OBSERVATION_DIM,
    HoloSomaFastSacIsaacPolicy,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _policy_tensor(observation) -> torch.Tensor:
    try:
        tensor = observation["policy"]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("environment did not return the policy group") from exc
    if tensor.ndim != 2 or tensor.shape[1] != POLICY_DIM:
        raise RuntimeError(
            f"long high-friction evaluation requires [N,{POLICY_DIM}], "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("policy observation contains NaN/Inf")
    return tensor


def _high_speed_tensor(observation) -> torch.Tensor:
    try:
        tensor = observation["high_speed_policy"]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError(
            "environment did not return the high_speed_policy group"
        ) from exc
    if tensor.ndim != 2 or tensor.shape[1] != HIGH_SPEED_DIM:
        raise RuntimeError(
            f"high-speed evaluation requires [N,{HIGH_SPEED_DIM}], got "
            f"{tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("high-speed observation contains NaN/Inf")
    return tensor


def _callable_name(function) -> str:
    return str(getattr(function, "__name__", type(function).__name__))


def _require_same_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        max_abs = float((actual - expected).abs().max().item())
        raise RuntimeError(f"{name} value/order mismatch (max_abs={max_abs:.9g})")


def _audit_policy_schema(base_env) -> dict[str, object]:
    """Fail closed on the complete 1864-D actor observation ABI.

    The final config field deliberately has a legacy name.  Checking only
    active-term names would therefore either reject the correct Motion task or
    silently accept packet age.  This audit checks its exact callable plus all
    term dimensions/history/slices before any actor inference.
    """

    manager = base_env.observation_manager
    active_terms = getattr(manager, "active_terms", {})
    terms = tuple(active_terms.get("policy", ()))
    if terms != EXPECTED_POLICY_TERMS:
        raise RuntimeError(
            "actor observation terms/order changed: "
            f"expected={EXPECTED_POLICY_TERMS}, actual={terms}"
        )
    forbidden = [
        term
        for term in terms
        if any(token in term.lower() for token in FORBIDDEN_POLICY_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"privileged actor terms are forbidden: {forbidden}")
    policy_dim = int(manager.group_obs_dim["policy"][-1])
    if policy_dim != POLICY_DIM:
        raise RuntimeError(f"expected actor dimension {POLICY_DIM}, got {policy_dim}")
    term_dims = tuple(
        tuple(int(value) for value in shape)
        for shape in manager.group_obs_term_dim["policy"]
    )
    if term_dims != EXPECTED_POLICY_TERM_DIMS:
        raise RuntimeError(
            "actor term dimensions changed: "
            f"expected={EXPECTED_POLICY_TERM_DIMS}, actual={term_dims}"
        )
    term_cfgs = tuple(manager._group_obs_term_cfgs["policy"])
    functions = tuple(_callable_name(cfg.func) for cfg in term_cfgs)
    if functions != EXPECTED_POLICY_FUNCTIONS:
        raise RuntimeError(
            "actor term callables changed: "
            f"expected={EXPECTED_POLICY_FUNCTIONS}, actual={functions}"
        )
    if term_cfgs[-1].func is not mdp.lateral_motion_feedback:
        raise RuntimeError(
            "legacy field foot_sensor_age_lr is not bound to the authoritative "
            "mdp.lateral_motion_feedback callable"
        )
    history_lengths = tuple(int(cfg.history_length) for cfg in term_cfgs)
    if history_lengths != EXPECTED_POLICY_HISTORY_LENGTHS:
        raise RuntimeError(
            "actor observation history lengths changed: "
            f"expected={EXPECTED_POLICY_HISTORY_LENGTHS}, actual={history_lengths}"
        )
    for term, cfg, history_length in zip(terms, term_cfgs, history_lengths):
        if history_length > 0 and not bool(cfg.flatten_history_dim):
            raise RuntimeError(f"policy history term {term!r} is not flattened")
    motion_params = term_cfgs[-1].params
    if (
        motion_params.get("asset_name") != "robot"
        or float(motion_params.get("lateral_velocity_clip", math.nan)) != 1.5
        or float(motion_params.get("heading_error_clip", math.nan)) != 1.0
        or tuple(term_cfgs[-1].clip) != (-1.5, 1.5)
    ):
        raise RuntimeError(
            "final actor columns must be robot [body_vy,relative_heading] "
            "with configured clips [1.5 m/s,1.0 rad]"
        )
    return {
        "terms": terms,
        "functions": functions,
        "term_dims": term_dims,
        "history_lengths": history_lengths,
        "term_cfgs": term_cfgs,
    }


def _audit_high_speed_schema(base_env) -> dict[str, object]:
    """Fail closed on the independent 482-D actor group."""

    manager = base_env.observation_manager
    terms = tuple(manager.active_terms.get("high_speed_policy", ()))
    if terms != EXPECTED_HIGH_SPEED_TERMS:
        raise RuntimeError(
            "high-speed actor terms/order changed: "
            f"expected={EXPECTED_HIGH_SPEED_TERMS}, actual={terms}"
        )
    forbidden = [
        term
        for term in terms
        if any(token in term.lower() for token in FORBIDDEN_POLICY_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"privileged high-speed actor terms: {forbidden}")
    dimension = int(manager.group_obs_dim["high_speed_policy"][-1])
    if dimension != HIGH_SPEED_DIM:
        raise RuntimeError(
            f"expected high-speed dimension {HIGH_SPEED_DIM}, got {dimension}"
        )
    dims = tuple(
        tuple(int(value) for value in shape)
        for shape in manager.group_obs_term_dim["high_speed_policy"]
    )
    if dims != EXPECTED_HIGH_SPEED_TERM_DIMS:
        raise RuntimeError(
            "high-speed term dimensions changed: "
            f"expected={EXPECTED_HIGH_SPEED_TERM_DIMS}, actual={dims}"
        )
    cfgs = tuple(manager._group_obs_term_cfgs["high_speed_policy"])
    functions = tuple(_callable_name(cfg.func) for cfg in cfgs)
    if functions != EXPECTED_HIGH_SPEED_FUNCTIONS:
        raise RuntimeError(
            "high-speed callables changed: "
            f"expected={EXPECTED_HIGH_SPEED_FUNCTIONS}, actual={functions}"
        )
    if cfgs[-1].func is not mdp.lateral_motion_feedback:
        raise RuntimeError("482-D tail is not mdp.lateral_motion_feedback")
    histories = tuple(int(cfg.history_length) for cfg in cfgs)
    if histories != EXPECTED_HIGH_SPEED_HISTORY_LENGTHS:
        raise RuntimeError(
            "high-speed histories changed: "
            f"expected={EXPECTED_HIGH_SPEED_HISTORY_LENGTHS}, actual={histories}"
        )
    return {
        "terms": terms,
        "functions": functions,
        "term_dims": dims,
        "history_lengths": histories,
    }


def _audit_high_speed_runtime_values(
    observation, progress=lambda _label: None
) -> dict[str, object]:
    """Prove 482-D = policy[0:480] + policy[1862:1864] bit-for-bit."""

    progress("482-D audit: reading 1864-D source")
    policy = _policy_tensor(observation)
    progress("482-D audit: reading independent 482-D group")
    high_speed = _high_speed_tensor(observation)
    # Compare the two source slices independently.  Besides making the ABI
    # proof more explicit, this avoids a CUDA stream stall observed in Isaac
    # Sim 5.1 when a freshly concatenated tensor was passed to torch.equal
    # before the first physics step.
    progress("482-D audit: comparing proprio prefix")
    _require_same_tensor(
        "high-speed 482-D proprio prefix",
        high_speed[:, :LEGACY_DIM],
        policy[:, :LEGACY_DIM],
    )
    progress("482-D audit: comparing motion tail")
    _require_same_tensor(
        "high-speed 482-D motion tail",
        high_speed[:, LEGACY_DIM:HIGH_SPEED_DIM],
        policy[:, MOTION_FEEDBACK_SLICE],
    )
    progress("482-D audit: mapping verified")
    return {
        "mapping": ["policy[0:480]", "policy[1862:1864]"],
        "tail_values": ["body_vy_m_s", "relative_heading_rad"],
        "runtime_mapping_max_abs_error": 0.0,
    }


def _audit_policy_runtime_values(
    base_env, observation, schema: dict[str, object]
) -> dict[str, object]:
    """Verify flattened slices and the physical meaning of the final columns."""

    manager = base_env.observation_manager
    policy = _policy_tensor(observation)
    terms = schema["terms"]
    term_cfgs = schema["term_cfgs"]
    history_buffers = manager._group_obs_term_history_buffer["policy"]
    slice_report: dict[str, list[int]] = {}
    for term, cfg, expected_slice, expected_dim in zip(
        terms, term_cfgs, EXPECTED_POLICY_SLICES, EXPECTED_POLICY_TERM_DIMS
    ):
        start, stop = expected_slice
        if stop - start != math.prod(expected_dim):
            raise RuntimeError(f"internal actor slice definition is invalid for {term!r}")
        actual = policy[:, start:stop]
        if int(cfg.history_length) > 0:
            buffered = history_buffers[term].buffer.reshape(base_env.num_envs, -1)
            _require_same_tensor(f"policy slice {term}", actual, buffered)
        slice_report[str(term)] = [start, stop]

    motion_cfg = term_cfgs[-1]
    direct_motion = motion_cfg.func(base_env, **motion_cfg.params).clone()
    direct_motion.clip_(min=motion_cfg.clip[0], max=motion_cfg.clip[1])
    motion_actual = policy[:, MOTION_FEEDBACK_SLICE]
    _require_same_tensor(
        "motion feedback [body_vy,relative_heading]",
        motion_actual,
        direct_motion,
    )
    robot = base_env.scene["robot"]
    expected_body_vy = robot.data.root_lin_vel_b[:, 1].clamp(-1.5, 1.5)
    _require_same_tensor("motion feedback body_vy", motion_actual[:, 0], expected_body_vy)
    return {
        "policy_slices": slice_report,
        "trailing_config_field": "foot_sensor_age_lr",
        "trailing_feature_mode": "motion_feedback",
        "trailing_values": ["body_vy_m_s", "relative_heading_rad"],
        "runtime_value_max_abs_error": 0.0,
    }


def _force_command(base_env, command: float) -> None:
    term = base_env.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = float(command)
    term.vel_command_b[:, 1:] = 0.0


def _audit_command_history(policy_obs: torch.Tensor, command: float) -> None:
    actual = policy_obs[:, COMMAND_SLICE].reshape(-1, 5, 3)
    expected = torch.zeros_like(actual)
    expected[:, :, 0] = float(command)
    if not torch.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
        error = float((actual - expected).abs().max().item())
        raise RuntimeError(
            "five-frame actor command history is not the requested constant "
            f"command; max_abs_error={error:.9g}"
        )


def _fall_timeout_masks(done: torch.Tensor, extras) -> tuple[torch.Tensor, torch.Tensor]:
    timeout = extras.get("time_outs") if isinstance(extras, dict) else None
    if timeout is None:
        timeout = torch.zeros_like(done, dtype=torch.bool)
    else:
        timeout = timeout.to(device=done.device, dtype=torch.bool)
    return done.bool() & ~timeout, done.bool() & timeout


def _rms(sum_square: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(sum_square / count.clamp_min(1))


def _quantile(value: np.ndarray, q: float) -> float | None:
    return float(np.quantile(value, q)) if value.size else None


def _floor_cfg_audit(env_cfg) -> dict[str, object]:
    result: dict[str, object] = {}
    for attr in ("friction_high_start", "friction_low", "friction_high_end"):
        cfg = getattr(env_cfg.scene, attr)
        material = cfg.spawn.physics_material
        size = tuple(float(item) for item in cfg.spawn.size)
        entry = {
            "prim_path": str(cfg.prim_path),
            "size_m": list(size),
            "center_m": [float(item) for item in cfg.init_state.pos],
            "static_friction": float(material.static_friction),
            "dynamic_friction": float(material.dynamic_friction),
            "friction_combine_mode": str(material.friction_combine_mode),
            "opacity": float(cfg.spawn.visual_material.opacity),
        }
        if (
            not math.isclose(entry["static_friction"], 0.90, abs_tol=1.0e-12)
            or not math.isclose(entry["dynamic_friction"], 0.90, abs_tol=1.0e-12)
            or entry["friction_combine_mode"] != "multiply"
            or not math.isclose(entry["opacity"], 1.0, abs_tol=1.0e-12)
        ):
            raise RuntimeError(f"uniform high-friction floor audit failed: {entry}")
        result[attr] = entry
    return result


def _hall_cfg_payload(env_cfg) -> dict[str, object]:
    cfg = env_cfg.hall_sensor_cfg
    audit_hall_sensor_cfg_policy_terms(env_cfg.observations, cfg)
    return {
        "enable_domain_randomization": bool(cfg.enable_domain_randomization),
        "foot_dropout_probability": float(cfg.foot_dropout_probability),
        "dead_channel_probability": float(cfg.dead_channel_probability),
        "maximum_packet_delay_steps": int(cfg.maximum_packet_delay_steps),
        "contact_distribution_mode": str(cfg.contact_distribution_mode),
    }


def _load_rsl_runner_policy(
    env, base_env, cfg_entry_point: str
):
    """Load a FastBase gate/residual checkpoint through its RSL runner config.

    The uniform high-friction task's default agent config is a plain MLP, so a
    native FastBase checkpoint (gate/residual/teacher keys) cannot be loaded
    by :func:`load_hall_backbone`.  This path builds the runner from the
    supplied FastBase-compatible config entry point and loads the actor only,
    never the critic/optimizer, before returning the deterministic inference
    policy.
    """

    if not isinstance(cfg_entry_point, str) or ":" not in cfg_entry_point:
        raise ValueError(
            "rsl_rl_cfg_entry_point must be 'module.qualname:ClassName'"
        )
    agent_cfg = resolve_callable(cfg_entry_point)()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, version("rsl-rl-lib")
    )
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if isinstance(algorithm_cfg, dict):
        if "capture_gate_warmup_updates" in algorithm_cfg:
            algorithm_cfg["capture_gate_warmup_updates"] = 0
    elif algorithm_cfg is not None and hasattr(
        algorithm_cfg, "capture_gate_warmup_updates"
    ):
        algorithm_cfg.capture_gate_warmup_updates = 0
    runner_class = resolve_callable(
        getattr(agent_cfg, "class_name", "OnPolicyRunner")
    )
    if not isinstance(runner_class, type) or not issubclass(
        runner_class, OnPolicyRunner
    ):
        raise RuntimeError(
            f"unsupported checkpoint runner {agent_cfg.class_name!r}"
        )
    runner = runner_class(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(
        str(Path(args_cli.checkpoint).expanduser().resolve()),
        load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG),
        strict=True,
    )
    return runner.get_inference_policy(device=base_env.device)


def main() -> int:
    wall_start = time.perf_counter()

    def progress(label: str) -> None:
        if args_cli.print_progress:
            print(
                f"[uniform-high-progress] {label}: "
                f"{time.perf_counter() - wall_start:.3f}s",
                flush=True,
            )

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    entry_key = (
        "env_cfg_entry_point" if args_cli.training_profile else "play_env_cfg_entry_point"
    )
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key=entry_key,
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.hall_contact_distribution is not None:
        env_cfg.hall_sensor_cfg.contact_distribution_mode = (
            args_cli.hall_contact_distribution
        )
        sync_hall_sensor_cfg_to_policy_terms(
            env_cfg.observations, env_cfg.hall_sensor_cfg
        )
    if args_cli.hardened_hall:
        env_cfg.hall_sensor_cfg.enable_domain_randomization = True
        env_cfg.hall_sensor_cfg.foot_dropout_probability = 0.10
        env_cfg.hall_sensor_cfg.dead_channel_probability = 0.08
        env_cfg.hall_sensor_cfg.maximum_packet_delay_steps = 5
        sync_hall_sensor_cfg_to_policy_terms(
            env_cfg.observations, env_cfg.hall_sensor_cfg
        )
    if args_cli.floor_width_m is not None:
        width = float(args_cli.floor_width_m)
        if not math.isfinite(width) or width < 3.2:
            raise ValueError("floor_width_m must be finite and at least 3.2 m")
        for attr in ("friction_high_start", "friction_low", "friction_high_end"):
            patch = getattr(env_cfg.scene, attr)
            size = tuple(float(item) for item in patch.spawn.size)
            patch.spawn.size = (size[0], width, size[2])
    if args_cli.floor_length_m is not None:
        length = float(args_cli.floor_length_m)
        low = getattr(env_cfg.scene, "friction_low")
        high_end = getattr(env_cfg.scene, "friction_high_end")
        low_size = tuple(float(item) for item in low.spawn.size)
        low_center = tuple(float(item) for item in low.init_state.pos)
        low_end_x = low_center[0] + low_size[0] / 2.0
        new_high_end_length = length - low_end_x
        if not math.isfinite(length) or new_high_end_length <= 0:
            raise ValueError(
                f"floor_length_m={length!r} leaves no trailing patch length"
            )
        high_end_size = tuple(float(item) for item in high_end.spawn.size)
        high_end.spawn.size = (
            new_high_end_length,
            high_end_size[1],
            high_end_size[2],
        )
        high_end.init_state.pos = (
            low_end_x + new_high_end_length / 2.0,
            low_center[1],
            low_center[2],
        )
    if args_cli.episode_length_s is not None:
        length = float(args_cli.episode_length_s)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("episode_length_s must be finite and positive")
        env_cfg.episode_length_s = length
    if args_cli.video_eye is not None or args_cli.video_lookat is not None:
        viewer = getattr(env_cfg, "viewer", None)
        if viewer is None:
            raise RuntimeError("viewer override requires env_cfg.viewer")
        def _viewer_xyz(value, name):
            parts = [float(item) for item in value.split(",")]
            if len(parts) != 3 or not all(math.isfinite(item) for item in parts):
                raise ValueError(f"{name} must be three finite comma-separated numbers")
            return (parts[0], parts[1], parts[2])
        if args_cli.video_eye is not None:
            viewer.eye = _viewer_xyz(args_cli.video_eye, "video_eye")
        if args_cli.video_lookat is not None:
            viewer.lookat = _viewer_xyz(args_cli.video_lookat, "video_lookat")
    floor_cfg = _floor_cfg_audit(env_cfg)
    hall_cfg = _hall_cfg_payload(env_cfg)
    progress("configuration ready")
    if env_cfg.events.spatial_friction_update is not None:
        raise RuntimeError("uniform high-friction task must not update spatial stage")
    if env_cfg.terminations.course_success is not None:
        raise RuntimeError("uniform high-friction task must not truncate at a course boundary")
    if not args_cli.training_profile and env_cfg.events.push_robot is not None:
        raise RuntimeError("nominal long-horizon play profile must not inject pushes")

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    raw_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        args_cli.video_dir.mkdir(parents=True, exist_ok=True)
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(args_cli.video_dir.resolve()),
            step_trigger=lambda step: step == 0,
            video_length=min(args_cli.video_length, max(args_cli.steps, 1)),
            disable_logger=True,
        )
    progress("gym environment ready")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    progress("RSL wrapper ready")
    base = env.unwrapped
    actor_schema = _audit_policy_schema(base)
    policy_terms = tuple(actor_schema["terms"])
    high_speed_mode = args_cli.high_speed_backbone_checkpoint is not None
    high_speed_schema = None
    if "high_speed_policy" in base.observation_manager.active_terms:
        high_speed_schema = _audit_high_speed_schema(base)
    if high_speed_mode and high_speed_schema is None:
        raise RuntimeError(
            "482-D checkpoint selected but task has no high_speed_policy group"
        )
    progress("actor term audit ready")
    action_term = base.action_manager.get_term("JointPositionAction")
    action_scale = float(action_term.cfg.scale)
    if not math.isclose(action_scale, 0.25, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"expected JointPositionAction scale 0.25, got {action_scale}")

    baseline_mode = args_cli.proprio_baseline_checkpoint is not None
    holosoma_mode = args_cli.holosoma_onnx is not None
    progress("action ABI ready")
    # Network constructors initialize temporary weights.  Forking RNG in both
    # modes guarantees that policy choice cannot alter environment random draws.
    with torch.random.fork_rng(devices=[]):
        progress("policy RNG sandbox entered")
        if holosoma_mode:
            checkpoint = args_cli.holosoma_onnx.expanduser().resolve()
            policy = HoloSomaFastSacIsaacPolicy(
                checkpoint,
                robot=base.scene["robot"],
                action_term=action_term,
                command_x_m_s=float(args_cli.command),
                policy_dt_s=float(base.step_dt),
            )
            progress("HoloSoma FastSAC adapter ready")
            policy_kind = "holosoma_fastsac_g1_29dof_100d_baseline"
            consumed_dimension = HOLOSOMA_OBSERVATION_DIM
        elif baseline_mode:
            checkpoint = args_cli.proprio_baseline_checkpoint.expanduser().resolve()
            policy = load_proprio_baseline(checkpoint, device=base.device)
            progress("baseline mean loaded")
            policy_kind = "original_model_49999_proprio_baseline"
            consumed_dimension = LEGACY_DIM
        elif high_speed_mode:
            checkpoint = (
                args_cli.high_speed_backbone_checkpoint.expanduser().resolve()
            )
            policy = load_high_speed_backbone(checkpoint, device=base.device)
            progress("high-speed 482-D mean loaded")
            policy_kind = "native_high_speed_backbone_482_candidate"
            consumed_dimension = HIGH_SPEED_DIM
        else:
            checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
            if args_cli.rsl_rl_cfg_entry_point is not None:
                policy = _load_rsl_runner_policy(
                    env, base, args_cli.rsl_rl_cfg_entry_point
                )
                policy_kind = "native_fastbase_rsl_runner_candidate"
            else:
                policy = load_hall_backbone(checkpoint, device=base.device)
                policy_kind = "native_hall_backbone_candidate"
            consumed_dimension = POLICY_DIM
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    progress("policy ready")

    _force_command(base, args_cli.command)
    observation = env.get_observations()
    progress("initial observations ready")
    policy_obs = _policy_tensor(observation)
    _audit_command_history(policy_obs, args_cli.command)
    actor_runtime_audit = _audit_policy_runtime_values(base, observation, actor_schema)
    progress("1864-D actor runtime value audit ready")
    high_speed_runtime_audit = (
        _audit_high_speed_runtime_values(observation, progress=progress)
        if high_speed_schema is not None
        else None
    )
    progress("actor runtime value audit ready")
    robot = base.scene["robot"]
    n = int(base.num_envs)
    dt = float(base.step_dt)
    if not math.isclose(dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"expected 50 Hz policy dt=0.02 s, got {dt}")
    effective_mu = getattr(base, "effective_friction_mu_buf", None)
    if effective_mu is None or effective_mu.shape != (n,):
        raise RuntimeError("uniform high-friction runtime buffer is missing")
    if not torch.isfinite(effective_mu).all() or bool((effective_mu <= 0).any().item()):
        raise RuntimeError("effective friction buffer is invalid")

    active = torch.ones(n, dtype=torch.bool, device=base.device)
    fallen = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    terminal_step = torch.full((n,), -1, dtype=torch.long, device=base.device)
    first_fall_step = torch.full_like(terminal_step, -1)
    first_fall_cross_track = torch.full(
        (n,), float("nan"), dtype=robot.data.root_pos_w.dtype, device=base.device
    )
    env_origin_y = base.scene.env_origins[:, 1].detach()
    initial_cross_track = robot.data.root_pos_w[:, 1] - env_origin_y
    max_abs_cross_track = initial_cross_track.abs().detach().clone()
    metric_count = torch.zeros(n, dtype=torch.long, device=base.device)
    sums = {
        name: torch.zeros(n, device=base.device)
        for name in (
            "vx",
            "vy",
            "vy2",
            "heading2",
            "omega2",
            "tilt2",
            "action_slew2",
            "hall_both_valid",
        )
    }
    saturation_elements = torch.zeros(n, device=base.device)
    action_elements = torch.zeros(n, device=base.device)
    start_x = robot.data.root_pos_w[:, 0].detach().clone()
    last_active_x = start_x.clone()
    last_active_cross_track = initial_cross_track.clone()
    previous_action = policy_obs[:, 335:480].reshape(n, 5, ACTION_DIM)[:, -1]
    trace_rows: list[np.ndarray] = []
    nan_detected = False
    nan_component: str | None = None
    executed_steps = 0

    for step in range(int(args_cli.steps)):
        _force_command(base, args_cli.command)
        policy_obs = _policy_tensor(observation)
        # Config and initial observations are audited exactly above.  Keeping
        # the history check on device here avoids a CUDA synchronization every
        # step; the maximum is consumed and fail-closed after the rollout.
        command_actual = policy_obs[:, COMMAND_SLICE].reshape(-1, 5, 3)
        command_expected = torch.zeros_like(command_actual)
        command_expected[:, :, 0] = float(args_cli.command)
        if step == 0:
            command_history_max_error = torch.zeros((), device=base.device)
        command_history_max_error = torch.maximum(
            command_history_max_error,
            (command_actual - command_expected).abs().amax(),
        )
        with torch.inference_mode():
            action = policy(observation)
        if action.shape != (n, ACTION_DIM):
            raise RuntimeError(f"action must be [N,{ACTION_DIM}], got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            nan_detected = True
            nan_component = "action"
            break

        pre_active = active.clone()
        pre_step_x = robot.data.root_pos_w[:, 0].detach().clone()
        pre_step_cross_track = (
            robot.data.root_pos_w[:, 1] - env_origin_y
        ).detach().clone()
        last_active_x = torch.where(pre_active, pre_step_x, last_active_x)
        last_active_cross_track = torch.where(
            pre_active, pre_step_cross_track, last_active_cross_track
        )
        max_abs_cross_track = torch.maximum(
            max_abs_cross_track,
            torch.where(
                pre_active,
                pre_step_cross_track.abs(),
                torch.zeros_like(pre_step_cross_track),
            ),
        )
        vx = robot.data.root_lin_vel_b[:, 0]
        vy = robot.data.root_lin_vel_b[:, 1]
        omega_xyz = robot.data.root_ang_vel_b
        omega = torch.linalg.vector_norm(omega_xyz, dim=1)
        heading = policy_obs[:, MOTION_FEEDBACK_SLICE][:, 1]
        gravity = policy_obs[:, 27:30]
        tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        action_slew = torch.linalg.vector_norm(action - previous_action, dim=1)
        action_norm = torch.linalg.vector_norm(action, dim=1)
        action_sat = (action.abs() >= 2.9).float().mean(dim=1)
        hall_valid = policy_obs[:, 1860:1862] >= 0.5
        state_values = torch.stack(
            (
                vx,
                vy,
                omega_xyz[:, 0],
                omega_xyz[:, 1],
                omega_xyz[:, 2],
                heading,
                tilt,
                action_slew,
                action_norm,
                action_sat,
            ),
            dim=1,
        )
        if not torch.isfinite(state_values).all():
            nan_detected = True
            nan_component = "state_metrics"
            break

        metric_mask = pre_active & (step >= int(args_cli.metric_warmup_steps))
        metric_count += metric_mask.long()
        for name, value in (
            ("vx", vx),
            ("vy", vy),
            ("vy2", vy.square()),
            ("heading2", heading.square()),
            ("omega2", omega.square()),
            ("tilt2", tilt.square()),
            ("action_slew2", action_slew.square()),
            ("hall_both_valid", hall_valid.all(dim=1).float()),
        ):
            sums[name] += torch.where(metric_mask, value, torch.zeros_like(value))
        saturation_elements += torch.where(
            metric_mask,
            (action.abs() >= 2.9).sum(dim=1).float(),
            torch.zeros(n, device=base.device),
        )
        action_elements += metric_mask.float() * ACTION_DIM

        if args_cli.trace_npz is not None:
            ids = torch.nonzero(pre_active, as_tuple=False).flatten()
            if ids.numel():
                # Exactly one device-to-host transfer per step.  The previous
                # per-field transfers dominated a 2-env diagnostic rollout.
                packed = torch.stack(
                    (
                        torch.full_like(vx, float(step)),
                        torch.full_like(vx, float(step * dt)),
                        torch.arange(n, device=base.device, dtype=vx.dtype),
                        vx,
                        vy,
                        heading,
                        omega_xyz[:, 0],
                        omega_xyz[:, 1],
                        omega_xyz[:, 2],
                        tilt,
                        action_norm,
                        action_slew,
                        action_sat,
                        hall_valid[:, 0].float(),
                        hall_valid[:, 1].float(),
                    ),
                    dim=1,
                )
                trace_rows.append(
                    packed.index_select(0, ids).detach().cpu().numpy()
                )

        previous_action = action.detach().clone()
        observation, _, done, extras = env.step(action)
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy(done.bool())
        fall, timeout = _fall_timeout_masks(done, extras)
        new_fall = pre_active & fall
        new_timeout = pre_active & timeout
        first_fall_step[new_fall] = step + 1
        first_fall_cross_track[new_fall] = pre_step_cross_track[new_fall]
        terminal_step[new_fall | new_timeout] = step + 1
        fallen |= new_fall
        timed_out |= new_timeout
        active &= ~done.bool()
        executed_steps = step + 1
        if args_cli.print_progress:
            progress(f"rollout step {executed_steps}/{args_cli.steps}")
        if not bool(active.any().item()):
            break

    terminal_step = torch.where(
        terminal_step >= 0,
        terminal_step,
        torch.full_like(terminal_step, executed_steps),
    )
    command_history_error = float(command_history_max_error.item())
    if command_history_error > 1.0e-6:
        raise RuntimeError(
            "five-frame command history changed during rollout; "
            f"max_abs_error={command_history_error:.9g}"
        )
    mean_vx = sums["vx"] / metric_count.clamp_min(1)
    per_env = {
        "mean_vx": mean_vx,
        "mean_vy": sums["vy"] / metric_count.clamp_min(1),
        "vy_rms": _rms(sums["vy2"], metric_count),
        "heading_rms": _rms(sums["heading2"], metric_count),
        "omega_rms": _rms(sums["omega2"], metric_count),
        "tilt_rms": _rms(sums["tilt2"], metric_count),
        "action_slew_rms": _rms(sums["action_slew2"], metric_count),
        "action_saturation_fraction": saturation_elements / action_elements.clamp_min(1),
        "hall_both_valid_fraction": sums["hall_both_valid"] / metric_count.clamp_min(1),
        # The managed environment resets terminal rows inside env.step().
        # Retain the final causal pre-step root position instead of measuring
        # reset-to-start displacement for a fall or horizon timeout.
        "progress_m": last_active_x - start_x,
        "final_cross_track_m": last_active_cross_track,
        "survival_s": terminal_step.float() * dt,
        "max_abs_cross_track_m": max_abs_cross_track,
    }
    arrays = {key: value.detach().cpu().numpy() for key, value in per_env.items()}
    valid_metric_env = metric_count.detach().cpu().numpy() > 0
    if not bool(valid_metric_env.all()):
        raise RuntimeError("one or more first episodes ended before steady metrics began")
    mean_vx_np = arrays["mean_vx"]
    aggregate = {
        "fall_event_count": int(fallen.sum().item()),
        "unique_env_first_fall_count": int(fallen.sum().item()),
        "timeout_count": int(timed_out.sum().item()),
        "survival_fraction": float((~fallen).float().mean().item()),
        "earliest_first_fall_s": (
            float(first_fall_step[first_fall_step >= 0].min().item() * dt)
            if bool((first_fall_step >= 0).any().item())
            else None
        ),
        "mean_survival_s": float(arrays["survival_s"].mean()),
        "mean_body_vx_m_s": float(mean_vx_np.mean()),
        "minimum_per_env_mean_vx_m_s": float(mean_vx_np.min()),
        "p05_per_env_mean_vx_m_s": _quantile(mean_vx_np, 0.05),
        "mean_body_vy_m_s": float(arrays["mean_vy"].mean()),
        "mean_final_cross_track_m": float(arrays["final_cross_track_m"].mean()),
        "maximum_abs_final_cross_track_m": float(
            np.abs(arrays["final_cross_track_m"]).max()
        ),
        "heading_rms_rad": float(np.sqrt(np.mean(np.square(arrays["heading_rms"])))),
        "body_vy_rms_m_s": float(np.sqrt(np.mean(np.square(arrays["vy_rms"])))),
        "angular_velocity_rms_rad_s": float(np.sqrt(np.mean(np.square(arrays["omega_rms"])))),
        "tilt_rms": float(np.sqrt(np.mean(np.square(arrays["tilt_rms"])))),
        "action_slew_rms": float(np.sqrt(np.mean(np.square(arrays["action_slew_rms"])))),
        "action_saturation_fraction": float(
            saturation_elements.sum().item() / action_elements.sum().clamp_min(1).item()
        ),
        "hall_both_valid_fraction": float(
            sums["hall_both_valid"].sum().item() / metric_count.sum().clamp_min(1).item()
        ),
        "mean_progress_m": float(arrays["progress_m"].mean()),
        "minimum_progress_m": float(arrays["progress_m"].min()),
        "maximum_abs_cross_track_m": float(arrays["max_abs_cross_track_m"].max()),
        "p95_abs_cross_track_m": _quantile(arrays["max_abs_cross_track_m"], 0.95),
        "nan_detected": bool(nan_detected),
        "nan_component": nan_component,
    }
    gates = {
        "zero_falls": aggregate["fall_event_count"] == 0,
        "finite_rollout": not nan_detected,
        "mean_vx": aggregate["mean_body_vx_m_s"] >= args_cli.minimum_mean_vx,
        "every_env_vx": (
            aggregate["minimum_per_env_mean_vx_m_s"]
            >= args_cli.minimum_per_env_mean_vx
        ),
        "heading_rms": aggregate["heading_rms_rad"] <= args_cli.maximum_heading_rms,
        "body_vy_rms": aggregate["body_vy_rms_m_s"] <= args_cli.maximum_body_vy_rms,
        "angular_velocity_rms": (
            aggregate["angular_velocity_rms_rad_s"]
            <= args_cli.maximum_angular_velocity_rms
        ),
        "action_saturation": (
            aggregate["action_saturation_fraction"]
            <= args_cli.maximum_action_saturation_fraction
        ),
    }
    gate_pass = all(bool(value) for value in gates.values())

    report = {
        "format": "uniform-high-friction-long-eval-v1",
        "status": "PASS" if gate_pass else "FAIL",
        "task": args_cli.task,
        "seed": int(args_cli.seed),
        "num_envs": n,
        "requested_steps": int(args_cli.steps),
        "executed_steps": executed_steps,
        "step_dt_s": dt,
        "requested_duration_s": float(args_cli.steps * dt),
        "metric_warmup_steps": int(args_cli.metric_warmup_steps),
        "policy": {
            "kind": policy_kind,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "rsl_rl_cfg_entry_point_override": (
                args_cli.rsl_rl_cfg_entry_point
                if args_cli.rsl_rl_cfg_entry_point is not None
                else None
            ),
            "environment_observation_dimension": POLICY_DIM,
            "consumed_observation_dimension": consumed_dimension,
            "policy_terms": list(policy_terms),
            "policy_functions": list(actor_schema["functions"]),
            "policy_term_dims": [list(shape) for shape in actor_schema["term_dims"]],
            "policy_history_lengths": list(actor_schema["history_lengths"]),
            **actor_runtime_audit,
            "uses_force_contact_mu_slip_or_stage": False,
            "clip_actions": float(agent_cfg.clip_actions),
            "joint_position_action_scale_rad": action_scale,
            "external_policy_adapter": (
                policy.manifest() if holosoma_mode else None
            ),
            "high_speed_policy_group": (
                {
                    "terms": list(high_speed_schema["terms"]),
                    "functions": list(high_speed_schema["functions"]),
                    "term_dims": [
                        list(shape) for shape in high_speed_schema["term_dims"]
                    ],
                    "history_lengths": list(
                        high_speed_schema["history_lengths"]
                    ),
                    **high_speed_runtime_audit,
                }
                if high_speed_schema is not None
                else None
            ),
        },
        "environment": {
            "profile": "training" if args_cli.training_profile else "nominal_play",
            "requested_command_m_s": float(args_cli.command),
            "floor_width_override_m": (
                float(args_cli.floor_width_m)
                if args_cli.floor_width_m is not None
                else None
            ),
            "floor_length_override_m": (
                float(args_cli.floor_length_m)
                if args_cli.floor_length_m is not None
                else None
            ),
            "episode_length_override_s": (
                float(args_cli.episode_length_s)
                if args_cli.episode_length_s is not None
                else None
            ),
            "floor": floor_cfg,
            "effective_mu_min_at_reset": float(effective_mu.min().item()),
            "effective_mu_max_at_reset": float(effective_mu.max().item()),
            "has_friction_transition": False,
            "has_course_success_truncation": False,
            "has_targeted_pushes": bool(args_cli.training_profile),
            "hall": hall_cfg,
            "command_history_max_abs_error": command_history_error,
        },
        "censoring": (
            "first episode only; first terminal sample and every managed-reset "
            "sample are excluded from subsequent motion metrics"
        ),
        "aggregate": aggregate,
        "gate_thresholds": {
            "minimum_mean_vx_m_s": float(args_cli.minimum_mean_vx),
            "minimum_per_env_mean_vx_m_s": float(args_cli.minimum_per_env_mean_vx),
            "maximum_heading_rms_rad": float(args_cli.maximum_heading_rms),
            "maximum_body_vy_rms_m_s": float(args_cli.maximum_body_vy_rms),
            "maximum_angular_velocity_rms_rad_s": float(
                args_cli.maximum_angular_velocity_rms
            ),
            "maximum_action_saturation_fraction": float(
                args_cli.maximum_action_saturation_fraction
            ),
        },
        "gates": {**gates, "pass": gate_pass},
        "per_env": [
            {
                "env_id": index,
                "fall": bool(fallen[index].item()),
                "timeout": bool(timed_out[index].item()),
                "first_fall_s": (
                    float(first_fall_step[index].item() * dt)
                    if first_fall_step[index].item() >= 0
                    else None
                ),
                "survival_s": float(arrays["survival_s"][index]),
                "mean_vx_m_s": float(arrays["mean_vx"][index]),
                "body_vy_rms_m_s": float(arrays["vy_rms"][index]),
                "heading_rms_rad": float(arrays["heading_rms"][index]),
                "angular_velocity_rms_rad_s": float(arrays["omega_rms"][index]),
                "tilt_rms": float(arrays["tilt_rms"][index]),
                "action_slew_rms": float(arrays["action_slew_rms"][index]),
                "action_saturation_fraction": float(
                    arrays["action_saturation_fraction"][index]
                ),
                "hall_both_valid_fraction": float(
                    arrays["hall_both_valid_fraction"][index]
                ),
                "progress_m": float(arrays["progress_m"][index]),
                "max_abs_cross_track_m": float(
                    arrays["max_abs_cross_track_m"][index]
                ),
                "first_fall_cross_track_m": (
                    float(first_fall_cross_track[index].item())
                    if bool(fallen[index].item())
                    else None
                ),
            }
            for index in range(n)
        ],
    }
    output = args_cli.summary_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict_json(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if args_cli.trace_npz is not None:
        trace_path = args_cli.trace_npz.expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = (
            np.concatenate(trace_rows, axis=0)
            if trace_rows
            else np.empty((0, len(TRACE_COLUMNS)), dtype=np.float32)
        )
        payload = {
            name: matrix[:, index]
            for index, name in enumerate(TRACE_COLUMNS)
        }
        payload.update(
            {
                "format": np.asarray("uniform-high-friction-long-trace-v1"),
                "seed": np.asarray(args_cli.seed, dtype=np.int64),
                "policy_dt_s": np.asarray(dt, dtype=np.float64),
                "requested_command_m_s": np.asarray(args_cli.command, dtype=np.float32),
                "checkpoint_sha256": np.asarray(_sha256(checkpoint)),
            }
        )
        np.savez_compressed(trace_path, **payload)
    print(json.dumps({"output": str(output), **aggregate, "pass": gate_pass}, indent=2))
    env.close()
    return 2 if args_cli.fail_on_gate and not gate_pass else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit can terminate the interpreter with status zero from close().
        # Preserve a failed smoke's shell-visible status instead.
        import os

        os._exit(1)
    if exit_code != 0:
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(exit_code)
    # Isaac Sim 5.1 can block during full CUDA teardown in standalone smoke
    # processes; this process owns Kit, so fast cleanup is safe.
    simulation_app.close(skip_cleanup=True)
