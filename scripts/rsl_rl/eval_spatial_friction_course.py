#!/usr/bin/env python3
"""Smoke-test and evaluate the physical Hall high--low--high course.

The evaluator has three deliberately separate responsibilities:

1. inspect the live USD stage and require three *static* CollisionAPI cuboids
   with the authored PhysX friction materials;
2. teleport a standing robot to each patch and verify the privileged contact
   label/contact-filter order is HIGH--LOW--HIGH;
3. optionally run a PT, TorchScript or ONNX actor and report whether one
   uninterrupted episode naturally traversed HIGH--LOW--HIGH.

Exact material identity and ``spatial_low_contact_buf`` are used only by this
evaluator.  A runtime audit requires the deployable actor group to remain the
1864-D Hall/proprioception schema; no friction, terrain or contact-force truth
is passed to the actor.

Examples
--------

Fast integration smoke test (no policy, no recording)::

  python scripts/rsl_rl/eval_spatial_friction_course.py \
    --headless --num_envs 2 --steps 8 --label_probe_steps 8

Evaluate a native RSL-RL checkpoint and record the first rollout::

  python scripts/rsl_rl/eval_spatial_friction_course.py \
    --checkpoint logs/rsl_rl/.../model_250.pt --num_envs 4 --steps 500 \
    --video --video_dir artifacts/spatial_friction_video

ONNX and TorchScript actors use ``--onnx`` and ``--torchscript`` respectively.
This script only drives Isaac Sim; it never sends commands to a real robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args


DEFAULT_TASK = (
    "Unitree-G1-29dof-Velocity-Foot-"
    "TractionMagneticMotionStudent-SpatialFriction"
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
    "foot_sensor_age_lr",
)
# Exact actor ABI.  The final *term name* is retained for checkpoint/config
# compatibility, but its callable and values in every Motion task must be
# ``lateral_motion_feedback = [body_vy, relative_heading]``.  It is not Hall
# packet age.  Keeping all term dimensions and slices here turns silent column
# insertions/reordering into a hard evaluator failure before policy inference.
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
EXPECTED_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
LEG_ACTION_EQUAL_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
)
LEG_ACTION_OPPOSITE_PAIRS = (
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
)
EXPECTED_JOINT_IDS_MAP = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)
FORBIDDEN_POLICY_TOKENS = (
    "contact",
    "force",
    "friction",
    "material",
    "terrain",
    "slip",
    "spatial",
    "ground_mu",
)
PATCHES = (
    ("FrictionHighStart", 0.90, -0.50, 0),
    ("FrictionLow", 0.16, 0.50, 1),
    ("FrictionHighEnd", 0.90, 1.50, 2),
)
EVAL_ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}

# The first 480 policy columns are concatenated term-by-term, not as five
# interleaved 96-D frames.  ``velocity_commands`` therefore occupies 30:45 as
# five consecutive [vx, vy, yaw] samples.  Keep these indices explicit: the
# former frame-major rewrite (6, 102, 198, 294, 390) silently overwrote body
# angular velocity and joint-history columns before calling the recovery
# expert.
RECOVERY_COMMAND_VX_INDICES = (30, 33, 36, 39, 42)
RECOVERY_COMMAND_VY_INDICES = (31, 34, 37, 40, 43)
RECOVERY_COMMAND_YAW_INDICES = (32, 35, 38, 41, 44)
RECOVERY_POLICY_OBSERVATION_DIM = 1864


def _with_recovery_command_history(observation, forward_command: float):
    """Return a copy with only the five-frame command term rewritten.

    This counterfactual command is an input to the frozen recovery policy
    only.  It never changes the environment command or the baseline policy
    observation.
    """

    if observation.ndim != 2 or observation.shape[1] != RECOVERY_POLICY_OBSERVATION_DIM:
        raise ValueError(
            "recovery command rewrite requires observation [N,1864], got "
            f"{tuple(observation.shape)}"
        )
    command = float(forward_command)
    if not math.isfinite(command):
        raise ValueError("recovery command must be finite")
    rewritten = observation.clone()
    rewritten[:, list(RECOVERY_COMMAND_VX_INDICES)] = command
    rewritten[:, list(RECOVERY_COMMAND_VY_INDICES)] = 0.0
    rewritten[:, list(RECOVERY_COMMAND_YAW_INDICES)] = 0.0
    return rewritten


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help=(
        "Natural rollout policy steps. Defaults to 3500 for the 24 m "
        "CadenceStrideLongDemo task and 400 for short courses."
    ),
)
parser.add_argument("--seed", type=int, default=390)
parser.add_argument("--command", type=float, default=0.80, help="Independent forward command in m/s.")
parser.add_argument(
    "--hall_contact_distribution",
    choices=("aggregate", "detailed"),
    default=None,
    help=(
        "Override only the Scheme-A mechanical contact-to-Hall driver for a "
        "paired A/B. The actor schema and floor physics remain unchanged."
    ),
)
parser.add_argument(
    "--hall_contact_force_atol",
    type=float,
    default=None,
    help=(
        "Evaluation-only override for the detailed Hall contact force-sum "
        "audit absolute tolerance (N).  Default None keeps the task value."
    ),
)
parser.add_argument(
    "--hall_contact_force_rtol",
    type=float,
    default=None,
    help=(
        "Evaluation-only override for the detailed Hall contact force-sum "
        "audit relative tolerance.  Default None keeps the task value."
    ),
)
parser.add_argument(
    "--hall_contact_audit_warn_only",
    action="store_true",
    help=(
        "Evaluation-only downgrade of the detailed Hall contact force-sum "
        "audit from fail-closed to warning.  Preserves the detailed Hall "
        "signal while tolerating single-step contact reconstruction "
        "transients; mismatches are still reported to stderr."
    ),
)
parser.add_argument(
    "--floor_width_m",
    type=float,
    default=None,
    help=(
        "Evaluation-only width override applied equally to all three coplanar "
        "floor patches.  Default None preserves the task's authored width."
    ),
)
parser.add_argument(
    "--low_patch_mu",
    type=float,
    default=None,
    help=(
        "Evaluation-only override for the LOW patch friction coefficient, "
        "applied to both the collider material and the privileged stage "
        "buffer.  Default None preserves the task's authored mu."
    ),
)
parser.add_argument(
    "--drift_metric_warmup_steps",
    type=int,
    default=100,
    help=(
        "Exclude startup steps from drift-gate RMS statistics, mirroring the "
        "uniform high-friction evaluator's 100-step warmup.  Fall counting is "
        "never warmed up."
    ),
)
parser.add_argument(
    "--hall_health_envelope",
    action="store_true",
    help=(
        "Enable the health-only Hall command fallback. Disabled by default so "
        "legacy evaluation behavior and datasets remain unchanged."
    ),
)
parser.add_argument("--health_single_foot_speed", type=float, default=0.25)
parser.add_argument(
    "--health_max_packet_age",
    type=float,
    default=0.25,
    help=(
        "Maximum delivered Hall packet age in seconds (default 0.25, aligned "
        "with the robot foot-bridge stale watchdog)."
    ),
)
parser.add_argument("--health_accel_rate", type=float, default=0.30)
parser.add_argument("--health_decel_rate", type=float, default=2.00)
parser.add_argument("--health_recovery_hold", type=float, default=0.50)
parser.add_argument(
    "--high_speed_stability_envelope",
    action="store_true",
    help=(
        "Enable the deployable proprio/IMU high-speed command envelope. "
        "Disabled by default. Reset-relative heading checks are disabled for "
        "non-zero yaw commands; tilt, roll/pitch-rate, and action emergency "
        "checks remain active."
    ),
)
parser.add_argument(
    "--stability_heading_correction",
    action="store_true",
    help=(
        "Opt in to bounded straight-heading yaw correction inside the stability "
        "envelope. Requires --high_speed_stability_envelope and is disabled by "
        "default so v1 command behavior is unchanged."
    ),
)
parser.add_argument("--stability_heading_gain", type=float, default=0.80)
parser.add_argument("--stability_heading_yaw_cap", type=float, default=0.40)
parser.add_argument(
    "--stability_heading_integral_gain",
    type=float,
    default=0.0,
    help="Leaky-integral gain on signed heading error for PI auto-straightening.",
)
parser.add_argument(
    "--stability_heading_integral_abs_cap",
    type=float,
    default=0.20,
    help="Maximum integral contribution to the yaw correction in radians.",
)
parser.add_argument(
    "--stability_heading_integral_decay",
    type=float,
    default=0.995,
    help="Per-step leak of the heading integral in [0, 1].",
)
parser.add_argument(
    "--stability_heading_correction_always",
    action="store_true",
    help="Apply heading correction continuously instead of only in WARN+ states.",
)
parser.add_argument(
    "--stability_conservative_preset",
    action="store_true",
    help=(
        "Diagnostic earlier intervention preset for sustained high-speed "
        "heading/tilt divergence. Requires --high_speed_stability_envelope; "
        "normal low-risk 0.8 m/s commands remain unchanged."
    ),
)
parser.add_argument(
    "--stability_early_heading_preset",
    action="store_true",
    help=(
        "Diagnostic heading-only early intervention. It leaves tilt, angular-"
        "rate and action emergency thresholds at their nominal values. Requires "
        "--high_speed_stability_envelope."
    ),
)
parser.add_argument(
    "--stability_recovery_checkpoint",
    type=Path,
    default=None,
    help=(
        "Opt in to a smooth EMERGENCY-state action handoff to the frozen "
        "Stage7 low-speed RSL actor checkpoint. Requires "
        "--high_speed_stability_envelope and is disabled by default."
    ),
)
parser.add_argument(
    "--stability_recovery_command",
    type=float,
    default=0.16,
    help="Private counterfactual forward command for the Stage7 recovery actor.",
)
parser.add_argument(
    "--stability_recovery_blend_in_s",
    type=float,
    default=0.20,
    help="EMERGENCY handoff ramp time in seconds (reviewed range 0.15--0.30).",
)
parser.add_argument(
    "--stability_recovery_blend_out_s",
    type=float,
    default=0.30,
    help="Safe-state release ramp time in seconds (reviewed range 0.15--0.30).",
)
parser.add_argument("--onnx", type=Path, default=None, help="ONNX actor path (alternative to --checkpoint).")
parser.add_argument(
    "--torchscript",
    type=Path,
    default=None,
    help="TorchScript/JIT actor path (alternative to --checkpoint).",
)
parser.add_argument(
    "--rsl_rl_cfg_entry_point",
    default=None,
    help=(
        "Override the agent config entry point used to build the RSL runner "
        "for --checkpoint.  Required for stability-branch checkpoints that the "
        "course's default FastBase config cannot construct."
    ),
)
parser.add_argument(
    "--label_probe_steps",
    type=int,
    default=12,
    help="Settling steps at each teleported patch used only for privileged smoke checks.",
)
parser.add_argument(
    "--skip_label_probe",
    action="store_true",
    help="Skip physical H-L-H teleport/contact label verification.",
)
parser.add_argument(
    "--probe_min_fraction",
    type=float,
    default=0.50,
    help="Minimum expected label and dominant-filter fraction in each probe.",
)
parser.add_argument(
    "--require_rollout_hlh",
    action="store_true",
    help="Exit non-zero unless a policy completes H-L-H without an intervening termination.",
)
parser.add_argument("--video", action="store_true", help="Record the natural rollout, not teleport probes.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_dir", type=Path, default=Path("artifacts/spatial_friction_video"))
parser.add_argument(
    "--video_eye",
    type=str,
    default=None,
    help="Comma-separated x,y,z viewer eye override for multi-env recordings.",
)
parser.add_argument(
    "--video_lookat",
    type=str,
    default=None,
    help="Comma-separated x,y,z viewer look-at override for multi-env recordings.",
)
parser.add_argument("--summary_json", type=Path, default=None)
parser.add_argument("--trace_npz", type=Path, default=None)
parser.add_argument("--state_dump_npz", type=Path, default=None)
parser.add_argument(
    "--state_dump_role",
    choices=(
        "locked_evaluation_only_do_not_train",
        "training_high_end_state_dump",
        "validation_high_end_state_dump",
        "development_smoke_not_train",
    ),
    default="locked_evaluation_only_do_not_train",
    help=(
        "Explicit provenance role for --state_dump_npz. Training/validation "
        "dumps are intermediate V2 sources and still require the offline "
        "bank builder before reset use."
    ),
)
parser.add_argument(
    "--state_dump_locked_seed",
    action="append",
    type=int,
    default=None,
    help="Acceptance seed that a training state dump must exclude (default: 500).",
)
parser.add_argument(
    "--failure_analysis_npz",
    type=Path,
    default=None,
    help=(
        "Save a row-aligned, first-episode diagnostic trace for HighEnd "
        "failure-precursor analysis. This is read-only evaluation data and "
        "must never be used to train on a locked acceptance seed."
    ),
)
parser.add_argument(
    "--dataset_npz",
    type=Path,
    default=None,
    help="Save alive all-env observation/action samples for teacher-mixture distillation.",
)
parser.add_argument("--trace_env_id", type=int, default=0)
parser.add_argument(
    "--hybrid_baseline_onnx",
    type=Path,
    default=None,
    help="High-traction 1864-D actor for the causal Hall hybrid evaluator.",
)
parser.add_argument(
    "--hybrid_recovery_onnx",
    type=Path,
    default=None,
    help="Low-grip recovery 1864-D actor selected only by the Hall risk gate.",
)
parser.add_argument(
    "--hall_risk_checkpoint",
    type=Path,
    default=None,
    help="Slip-risk estimator checkpoint used by the hybrid evaluator.",
)
parser.add_argument(
    "--hall_command_governor",
    action="store_true",
    help=(
        "Opt in to the strict single-RSL-checkpoint Hall-only command governor. "
        "This path is independent of the three-actor hybrid and requires "
        "--checkpoint plus --hall_command_risk_checkpoint."
    ),
)
parser.add_argument(
    "--hall_command_risk_checkpoint",
    type=Path,
    default=None,
    help=(
        "Independent 1864-D motion_feedback prospective-risk .pt used only by "
        "--hall_command_governor. It never replaces the locomotion actor."
    ),
)
parser.add_argument("--hall_governor_low_speed", type=float, default=0.10)
parser.add_argument("--hall_governor_high_speed", type=float, default=0.90)
parser.add_argument("--hall_governor_critical_speed", type=float, default=0.0)
parser.add_argument("--hall_governor_low_probability", type=float, default=0.65)
parser.add_argument("--hall_governor_high_probability", type=float, default=0.45)
parser.add_argument("--hall_governor_critical_probability", type=float, default=0.85)
parser.add_argument("--hall_governor_critical_hold_s", type=float, default=0.04)
parser.add_argument("--hall_governor_probability_alpha", type=float, default=0.20)
parser.add_argument("--hall_governor_relative_low_rise", type=float, default=0.12)
parser.add_argument("--hall_governor_relative_high_drop", type=float, default=0.12)
parser.add_argument("--hall_governor_low_hold_s", type=float, default=0.10)
parser.add_argument("--hall_governor_high_hold_s", type=float, default=0.40)
parser.add_argument("--hall_governor_low_reprobe_s", type=float, default=2.50)
parser.add_argument("--hall_governor_probe_duration_s", type=float, default=0.45)
parser.add_argument("--hall_governor_probe_speed", type=float, default=0.25)
parser.add_argument("--hall_governor_accel_rate", type=float, default=0.50)
parser.add_argument("--hall_governor_decel_rate", type=float, default=2.00)
parser.add_argument(
    "--hall_governor_allow_absolute_high_clear",
    action="store_true",
    help=(
        "Allow a sustained calibrated low prospective-risk score to release "
        "LOW. This is opt-in because a risk head must be validated with the "
        "specific actor/checkpoint before it is an acceptance result."
    ),
)
parser.add_argument("--hybrid_risk_start", type=float, default=0.55)
parser.add_argument("--hybrid_risk_full", type=float, default=0.75)
parser.add_argument("--hybrid_on_steps", type=int, default=5)
parser.add_argument("--hybrid_off_steps", type=int, default=15)
parser.add_argument(
    "--hybrid_max_active_steps",
    type=int,
    default=60,
    help="Bounded low-grip recovery authority (0 disables the bound).",
)
parser.add_argument("--hybrid_recovery_command", type=float, default=0.16)
parser.add_argument(
    "--blend_onnx",
    type=Path,
    default=None,
    help="Optional second 1864-D actor for a fixed action blend with --onnx.",
)
parser.add_argument("--blend_alpha", type=float, default=0.5)
parser.add_argument(
    "--hardened_hall",
    action="store_true",
    help="Evaluate with the strengthened foot-drop/dead-channel/delay DR profile.",
)
parser.add_argument(
    "--expected_low_mu",
    type=float,
    default=None,
    help="Expected physical low-patch friction; inferred for staged tasks when omitted.",
)
parser.add_argument(
    "--low_speed_target",
    type=float,
    default=None,
    help="Low-patch response target; inferred as Mild=.45, Medium=.32, final=.24 m/s.",
)
parser.add_argument(
    "--high_recovery_speed",
    type=float,
    default=0.70,
    help="Absolute speed that must be held on the final high patch for recovery timing.",
)
parser.add_argument(
    "--recovery_stable_steps",
    type=int,
    default=3,
    help="Consecutive policy frames required to declare high-patch speed recovery.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O for dynamic state (USD authoring checks always run).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
state_dump_locked_seeds = set(args_cli.state_dump_locked_seed or [500])
if args_cli.state_dump_npz is None and (
    args_cli.state_dump_role != "locked_evaluation_only_do_not_train"
    or args_cli.state_dump_locked_seed is not None
):
    parser.error("--state_dump_role/--state_dump_locked_seed require --state_dump_npz")
if (
    args_cli.state_dump_npz is not None
    and args_cli.state_dump_role
    in {"training_high_end_state_dump", "validation_high_end_state_dump"}
    and int(args_cli.seed) in state_dump_locked_seeds
):
    parser.error(
        f"seed {args_cli.seed} is locked for acceptance and cannot be used as "
        f"{args_cli.state_dump_role}"
    )
if args_cli.steps is None:
    args_cli.steps = (
        3500
        if "LongDemo" in args_cli.task
        else 1200
        if "CadenceStrideRetention" in args_cli.task
        else 400
    )


def _runtime_course_geometry() -> dict[str, float | bool]:
    """Return the explicit local-X geometry selected by the task play cfg.

    CadenceStride deliberately trains on the short transition-dense course but
    uses a 24 m long play course.  Keeping this mapping explicit prevents the
    evaluator from silently applying the old 0/1 m diagnostic boundaries to
    that long visualization scene.
    """

    # The ordinary CadenceStride task still has a short play cfg.  Only the
    # explicit LongDemo registry ID selects the 24 m scene.
    retention_course = "CadenceStrideRetention" in args_cli.task
    long_course = "CadenceStrideLongDemo" in args_cli.task or (
        "LongDemo" in args_cli.task
    )
    if retention_course:
        return {
            "long_course": True,
            "retention_training_course": True,
            "low_start_x_m": 0.0,
            "low_end_x_m": 2.0,
            "success_x_m": 9.5,
            "high_start_probe_x_m": -1.5,
            "low_probe_x_m": 1.0,
            "high_end_probe_x_m": 6.0,
        }
    if long_course:
        return {
            "long_course": True,
            "low_start_x_m": 0.0,
            "low_end_x_m": 6.0,
            "success_x_m": 17.5,
            "high_start_probe_x_m": -3.0,
            "low_probe_x_m": 3.0,
            "high_end_probe_x_m": 12.0,
        }
    return {
        "long_course": False,
        "low_start_x_m": 0.0,
        "low_end_x_m": 1.0,
        "success_x_m": 2.60,
        "high_start_probe_x_m": -0.50,
        "low_probe_x_m": 0.50,
        "high_end_probe_x_m": 1.50,
    }


def _runtime_patches():
    geometry = _runtime_course_geometry()
    if getattr(args_cli, "low_patch_mu", None) is not None:
        low_mu = float(getattr(args_cli, "low_patch_mu"))
    elif args_cli.expected_low_mu is not None:
        low_mu = float(args_cli.expected_low_mu)
    elif (
        args_cli.task.endswith("SpatialFrictionMild")
        or "SpatialFrictionMild" in args_cli.task
    ):
        low_mu = 0.45
    elif (
        args_cli.task.endswith("SpatialFrictionMedium")
        or "SpatialFrictionMedium" in args_cli.task
        or "CadenceStride" in args_cli.task
    ):
        low_mu = 0.28
    else:
        low_mu = 0.16
    if not 0.0 < low_mu < 0.90:
        raise ValueError(f"expected_low_mu must be in (0, 0.90), got {low_mu}")
    return (
        (
            "FrictionHighStart",
            0.90,
            geometry["high_start_probe_x_m"],
            0,
        ),
        ("FrictionLow", low_mu, geometry["low_probe_x_m"], 1),
        (
            "FrictionHighEnd",
            0.90,
            geometry["high_end_probe_x_m"],
            2,
        ),
    )


def _runtime_low_speed_target() -> float:
    if args_cli.low_speed_target is not None:
        return float(args_cli.low_speed_target)
    if "CadenceStride" in args_cli.task:
        # This isolated curriculum intentionally has no privileged LOW speed
        # cap.  Treat the unchanged request as the retention reference.
        return float(args_cli.command)
    if (
        args_cli.task.endswith("SpatialFrictionMild")
        or "SpatialFrictionMild" in args_cli.task
    ):
        return 0.45
    if (
        args_cli.task.endswith("SpatialFrictionMedium")
        or "SpatialFrictionMedium" in args_cli.task
    ):
        return 0.32
    return 0.24

selected_policies = sum(
    path is not None
    for path in (
        args_cli.checkpoint,
        args_cli.onnx,
        args_cli.torchscript,
        args_cli.hybrid_baseline_onnx,
    )
)
hybrid_requested = any(
    path is not None
    for path in (
        args_cli.hybrid_baseline_onnx,
        args_cli.hybrid_recovery_onnx,
        args_cli.hall_risk_checkpoint,
    )
)
if selected_policies > 1 or (hybrid_requested and selected_policies > 1):
    parser.error("select one ordinary policy or the complete hybrid policy set")
if hybrid_requested and not all(
    path is not None
    for path in (
        args_cli.hybrid_baseline_onnx,
        args_cli.hybrid_recovery_onnx,
        args_cli.hall_risk_checkpoint,
    )
):
    parser.error("hybrid mode requires baseline, recovery and Hall-risk paths")
if args_cli.hall_command_governor:
    if args_cli.checkpoint is None or args_cli.hall_command_risk_checkpoint is None:
        parser.error(
            "--hall_command_governor requires --checkpoint and "
            "--hall_command_risk_checkpoint"
        )
    if hybrid_requested or args_cli.onnx is not None or args_cli.torchscript is not None:
        parser.error(
            "--hall_command_governor is a single-RSL-checkpoint path and cannot "
            "be combined with ONNX/TorchScript or the three-actor hybrid"
        )
    if args_cli.blend_onnx is not None:
        parser.error("--hall_command_governor cannot be combined with --blend_onnx")
    if args_cli.high_speed_stability_envelope or args_cli.stability_recovery_checkpoint:
        parser.error(
            "--hall_command_governor cannot be stacked with the stability "
            "envelope; its audited order ends at the rewritten single actor"
        )
elif args_cli.hall_command_risk_checkpoint is not None:
    parser.error(
        "--hall_command_risk_checkpoint requires --hall_command_governor"
    )
if args_cli.blend_onnx is not None and args_cli.onnx is None:
    parser.error("--blend_onnx requires --onnx as the first actor")
if args_cli.blend_onnx is not None and (
    args_cli.checkpoint is not None or args_cli.torchscript is not None
):
    parser.error("--blend_onnx is only supported with --onnx")
if not 0.0 <= args_cli.blend_alpha <= 1.0:
    parser.error("blend_alpha must be in [0, 1]")
if not 0.0 <= args_cli.hybrid_risk_start < args_cli.hybrid_risk_full <= 1.0:
    parser.error("hybrid risk thresholds must satisfy 0 <= start < full <= 1")
if args_cli.hybrid_on_steps <= 0 or args_cli.hybrid_off_steps <= 0:
    parser.error("hybrid hysteresis step counts must be positive")
if args_cli.hybrid_max_active_steps < 0:
    parser.error("hybrid_max_active_steps must be non-negative")
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num_envs and --steps must be positive")
if args_cli.drift_metric_warmup_steps < 0:
    parser.error("--drift_metric_warmup_steps must be non-negative")
if not math.isfinite(args_cli.command):
    parser.error("--command must be finite")
for name in ("hall_contact_force_atol", "hall_contact_force_rtol"):
    value = getattr(args_cli, name)
    if value is not None and (not math.isfinite(value) or value < 0.0):
        parser.error(f"--{name} must be finite and non-negative")
if not math.isfinite(args_cli.health_single_foot_speed) or args_cli.health_single_foot_speed < 0.0:
    parser.error("--health_single_foot_speed must be finite and non-negative")
if not math.isfinite(args_cli.health_max_packet_age) or args_cli.health_max_packet_age <= 0.0:
    parser.error("--health_max_packet_age must be finite and positive")
if not math.isfinite(args_cli.health_accel_rate) or args_cli.health_accel_rate <= 0.0:
    parser.error("--health_accel_rate must be finite and positive")
if not math.isfinite(args_cli.health_decel_rate) or args_cli.health_decel_rate <= 0.0:
    parser.error("--health_decel_rate must be finite and positive")
if not math.isfinite(args_cli.health_recovery_hold) or args_cli.health_recovery_hold < 0.0:
    parser.error("--health_recovery_hold must be finite and non-negative")
if args_cli.stability_heading_correction and not args_cli.high_speed_stability_envelope:
    parser.error(
        "--stability_heading_correction requires --high_speed_stability_envelope"
    )
if (
    args_cli.stability_recovery_checkpoint is not None
    and not args_cli.high_speed_stability_envelope
):
    parser.error(
        "--stability_recovery_checkpoint requires --high_speed_stability_envelope"
    )
if args_cli.stability_conservative_preset and not args_cli.high_speed_stability_envelope:
    parser.error(
        "--stability_conservative_preset requires --high_speed_stability_envelope"
    )
if args_cli.stability_early_heading_preset and not args_cli.high_speed_stability_envelope:
    parser.error(
        "--stability_early_heading_preset requires --high_speed_stability_envelope"
    )
if args_cli.stability_conservative_preset and args_cli.stability_early_heading_preset:
    parser.error("stability presets are mutually exclusive")
if args_cli.stability_recovery_checkpoint is not None and hybrid_requested:
    parser.error(
        "--stability_recovery_checkpoint cannot be stacked with the Hall-risk hybrid"
    )
if args_cli.stability_recovery_checkpoint is not None and selected_policies != 1:
    parser.error(
        "--stability_recovery_checkpoint requires exactly one baseline policy"
    )
if not math.isfinite(args_cli.stability_heading_gain) or args_cli.stability_heading_gain < 0.0:
    parser.error("--stability_heading_gain must be finite and non-negative")
if (
    not math.isfinite(args_cli.stability_heading_yaw_cap)
    or args_cli.stability_heading_yaw_cap <= 0.0
):
    parser.error("--stability_heading_yaw_cap must be finite and positive")
if (
    not math.isfinite(args_cli.stability_heading_integral_gain)
    or args_cli.stability_heading_integral_gain < 0.0
):
    parser.error("--stability_heading_integral_gain must be finite and non-negative")
if (
    not math.isfinite(args_cli.stability_heading_integral_abs_cap)
    or args_cli.stability_heading_integral_abs_cap <= 0.0
):
    parser.error("--stability_heading_integral_abs_cap must be finite and positive")
if (
    not math.isfinite(args_cli.stability_heading_integral_decay)
    or not 0.0 <= args_cli.stability_heading_integral_decay <= 1.0
):
    parser.error("--stability_heading_integral_decay must be finite and in [0, 1]")
if (
    not math.isfinite(args_cli.stability_recovery_command)
    or args_cli.stability_recovery_command < 0.0
):
    parser.error("--stability_recovery_command must be finite and non-negative")
for name, value in (
    ("--stability_recovery_blend_in_s", args_cli.stability_recovery_blend_in_s),
    ("--stability_recovery_blend_out_s", args_cli.stability_recovery_blend_out_s),
):
    if not math.isfinite(value) or not 0.15 <= value <= 0.30:
        parser.error(f"{name} must be finite and in [0.15, 0.30]")
if args_cli.label_probe_steps <= 0:
    parser.error("--label_probe_steps must be positive")
if not 0.0 < args_cli.probe_min_fraction <= 1.0:
    parser.error("--probe_min_fraction must be in (0, 1]")
if not 0 <= args_cli.trace_env_id < args_cli.num_envs:
    parser.error("--trace_env_id must be smaller than --num_envs")
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if _runtime_low_speed_target() < 0.0:
    parser.error("--low_speed_target must be non-negative")
if args_cli.high_recovery_speed <= 0.0:
    parser.error("--high_recovery_speed must be positive")
if args_cli.recovery_stable_steps <= 0:
    parser.error("--recovery_stable_steps must be positive")
for name, value in (
    ("--hall_governor_low_speed", args_cli.hall_governor_low_speed),
    ("--hall_governor_high_speed", args_cli.hall_governor_high_speed),
    ("--hall_governor_critical_speed", args_cli.hall_governor_critical_speed),
    ("--hall_governor_low_hold_s", args_cli.hall_governor_low_hold_s),
    ("--hall_governor_high_hold_s", args_cli.hall_governor_high_hold_s),
    ("--hall_governor_low_reprobe_s", args_cli.hall_governor_low_reprobe_s),
    ("--hall_governor_probe_duration_s", args_cli.hall_governor_probe_duration_s),
    ("--hall_governor_probe_speed", args_cli.hall_governor_probe_speed),
    ("--hall_governor_accel_rate", args_cli.hall_governor_accel_rate),
    ("--hall_governor_decel_rate", args_cli.hall_governor_decel_rate),
):
    if not math.isfinite(value) or value < 0.0:
        parser.error(f"{name} must be finite and non-negative")
if args_cli.hall_governor_high_speed < args_cli.hall_governor_low_speed:
    parser.error("Hall governor high speed must be >= low speed")
if not (
    0.0
    <= args_cli.hall_governor_high_probability
    <= args_cli.hall_governor_low_probability
    <= args_cli.hall_governor_critical_probability
    <= 1.0
):
    parser.error(
        "Hall governor probabilities must satisfy 0 <= high <= low <= critical <= 1"
    )
if not 0.0 < args_cli.hall_governor_probability_alpha <= 1.0:
    parser.error("--hall_governor_probability_alpha must be in (0,1]")
for name, value in (
    ("--hall_governor_relative_low_rise", args_cli.hall_governor_relative_low_rise),
    ("--hall_governor_relative_high_drop", args_cli.hall_governor_relative_high_drop),
):
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        parser.error(f"{name} must be in [0,1]")
if args_cli.hall_command_governor and (
    args_cli.hall_governor_probe_duration_s <= 0.0
    or args_cli.hall_governor_probe_speed <= 0.0
    or args_cli.hall_governor_accel_rate <= 0.0
    or args_cli.hall_governor_decel_rate <= 0.0
):
    parser.error("enabled Hall governor probe/rate values must be positive")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.traction.hall_risk_estimator import (  # noqa: E402
    build_hall_risk_estimator,
)
from unitree_rl_lab.traction.contact_slip import (  # noqa: E402
    static_ground_contact_point_tangential_speed,
)
from unitree_rl_lab.traction.hall_governor import (  # noqa: E402
    HallTractionGovernor,
    HallTractionGovernorCfg,
)
from unitree_rl_lab.traction.health_envelope import (  # noqa: E402
    HealthEnvelope,
    HealthEnvelopeCfg,
    rewrite_command_history,
    summarize_health_envelope_trace,
)
from unitree_rl_lab.traction.high_speed_stability_envelope import (  # noqa: E402
    HighSpeedStabilityEnvelope,
    HighSpeedStabilityEnvelopeCfg,
    summarize_high_speed_stability_trace,
)
from unitree_rl_lab.traction.stability_recovery_blend import (  # noqa: E402
    FrozenStage7RecoveryActor,
    StabilityRecoveryBlend,
    StabilityRecoveryBlendCfg,
)
from unitree_rl_lab.traction.layout_magnetic_student import VALID_SLICE  # noqa: E402
from unitree_rl_lab.sensors.hall_sensor_config import (  # noqa: E402
    sync_hall_sensor_cfg_to_policy_terms,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

# Pure helper lives outside the Isaac extension so it can be unit-tested
# without launching Kit.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "traction"))
from spatial_friction_eval_utils import (  # noqa: E402
    COMPLETE,
    SpatialTransitionSample,
    advance_high_low_high_stage,
    analyze_transition_response,
    classify_hall_health,
    compress_contact_labels,
    summarize_hall_command_governor_trace,
    summarize_fastbase_capture_diagnostics,
    validate_motion_hall_risk_metadata,
)


def _strict_json(value):
    """Convert NumPy values and replace non-finite floats for strict JSON."""

    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _force_command(base_env, command: float) -> None:
    """Apply a command that contains no patch/material side channel."""

    term = base_env.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = float(command)
    term.vel_command_b[:, 1:] = 0.0


def _apply_effective_command(base_env, command: torch.Tensor) -> None:
    """Apply a per-environment deployable command without privileged inputs."""

    term = base_env.command_manager.get_term("base_velocity")
    expected = (base_env.num_envs, 3)
    if command.shape != expected:
        raise ValueError(
            f"effective command must have shape {expected}, got {tuple(command.shape)}"
        )
    command = command.to(device=base_env.device, dtype=term.vel_command_b.dtype)
    if not torch.isfinite(command).all():
        raise FloatingPointError("health envelope produced a non-finite command")
    term.vel_command_b[:, :3].copy_(command)
    term.is_standing_env.copy_(command.abs().amax(dim=1) <= 1.0e-7)


def _policy_tensor(observation) -> torch.Tensor:
    if torch.is_tensor(observation):
        tensor = observation
    else:
        try:
            tensor = observation["policy"]
        except (KeyError, TypeError, IndexError) as exc:
            raise RuntimeError("environment did not return a policy observation group") from exc
    if tensor.ndim != 2 or tensor.shape[1] != 1864:
        raise RuntimeError(
            "spatial Hall evaluation requires [num_envs, 1864] actor input, "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("non-finite Hall/proprioception policy observation")
    return tensor


def _read_health_envelope_inputs(
    base_env, observation
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read only robot-available Hall packet health for the safety envelope."""

    policy_observation = _policy_tensor(observation)
    valid_observation = policy_observation[:, VALID_SLICE]
    valid = torch.isfinite(valid_observation) & (valid_observation > 0.5)
    debug = base_env._hall_foot_sensor.get_debug_data()
    age_s = debug["sample_age"]
    delay_steps = debug["policy_delay_steps"]
    reported_period_s = debug["reported_sample_period"]
    hall_observation = debug["policy_observation"]
    expected_age = (base_env.num_envs, 2)
    for name, value in (
        ("sample_age", age_s),
        ("policy_delay_steps", delay_steps),
        ("reported_sample_period", reported_period_s),
    ):
        if value.shape != expected_age:
            raise RuntimeError(
                f"Hall {name} must have shape {expected_age}, got {tuple(value.shape)}"
            )
    if hall_observation.ndim != 4 or hall_observation.shape[:2] != expected_age:
        raise RuntimeError(
            "Hall policy observation must have shape [num_envs,2,sensors,3], "
            f"got {tuple(hall_observation.shape)}"
        )
    # ``finite`` is computed from the same normalized Hall signal available to
    # deployment.  No mechanical/contact driver or material label is read.
    finite = torch.isfinite(hall_observation).all(dim=(2, 3))
    # The delivered sample can intentionally come from an older policy buffer.
    # Account for that causal delay so ``age_s`` has the same timestamp-age
    # meaning expected from a real BLE packet rather than merely reporting the
    # time since the simulator generated its newest, not-yet-delivered sample.
    delivered_age_s = age_s + delay_steps.to(age_s.dtype) * reported_period_s
    return valid, delivered_age_s, finite


def _rewrite_actor_command_history(observation, effective_command: torch.Tensor):
    """Return an actor input whose five command frames match the applied command."""

    rewritten_policy = rewrite_command_history(
        _policy_tensor(observation), effective_command
    )
    if torch.is_tensor(observation):
        return rewritten_policy
    try:
        rewritten = observation.clone()
        rewritten["policy"] = rewritten_policy
    except (AttributeError, KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("unable to clone and rewrite policy observation group") from exc
    return rewritten


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StrictHallCommandGovernor:
    """Independent Motion-safe risk head plus the causal command governor.

    Only :meth:`predict` receives the raw 1864-D Hall/proprio observation.
    Material identity, contact, force, course position/stage and simulator slip
    are structurally absent from this interface.
    """

    def __init__(
        self,
        risk_checkpoint: Path,
        *,
        actor_checkpoint: Path,
        num_envs: int,
        dt: float,
        device,
        cfg: HallTractionGovernorCfg,
    ) -> None:
        path = risk_checkpoint.expanduser().resolve()
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise FileNotFoundError(
                f"Hall command risk checkpoint must be an existing .pt file: {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.metadata = validate_motion_hall_risk_metadata(payload)
        state_dict = payload["model"]
        for name, value in state_dict.items():
            if not torch.is_tensor(value):
                raise RuntimeError(f"Hall risk model state {name!r} is not a tensor")
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Hall risk model state {name!r} contains NaN/Inf"
                )
        self.risk_model = build_hall_risk_estimator(payload).to(device).eval()
        self.governor = HallTractionGovernor(num_envs, dt, device, cfg)
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = _sha256_file(path)
        actor_path = actor_checkpoint.expanduser().resolve()
        if not actor_path.is_file():
            raise FileNotFoundError(f"RSL locomotion checkpoint does not exist: {actor_path}")
        self.actor_checkpoint_path = str(actor_path)
        self.actor_checkpoint_sha256 = _sha256_file(actor_path)

        # Construction must fail before rollout if even a finite, schema-valid
        # Motion observation produces an invalid score.
        probe = torch.zeros((1, 1864), device=device)
        probe[:, 1830:1860] = float(dt)
        probe[:, VALID_SLICE] = 1.0
        with torch.inference_mode():
            self._validate_probability(self.risk_model(probe), expected_batch=1)

    @staticmethod
    def _validate_probability(value: torch.Tensor, *, expected_batch: int) -> torch.Tensor:
        if not torch.is_tensor(value) or value.numel() != expected_batch:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise RuntimeError(
                "Hall risk head must return one scalar per environment, "
                f"got {shape}"
            )
        probability = value.reshape(expected_batch)
        if not torch.isfinite(probability).all():
            raise FloatingPointError("Hall risk head returned NaN/Inf")
        if torch.any((probability < 0.0) | (probability > 1.0)):
            raise RuntimeError("Hall risk head probability lies outside [0,1]")
        return probability

    def predict(self, raw_policy_observation: torch.Tensor) -> torch.Tensor:
        if raw_policy_observation.ndim != 2 or raw_policy_observation.shape[1] != 1864:
            raise RuntimeError(
                "Hall risk head requires raw Motion observation [num_envs,1864]"
            )
        with torch.inference_mode():
            value = self.risk_model(raw_policy_observation)
        return self._validate_probability(
            value, expected_batch=raw_policy_observation.shape[0]
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.governor.reset(env_ids)

    def update(
        self,
        requested_command: torch.Tensor,
        risk_probability: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.governor.update(
            requested_command,
            risk_probability,
            valid=valid,
        )

    def audit_report(self) -> dict[str, object]:
        return {
            "enabled": True,
            "status": "diagnostic_only_not_actor_specific_acceptance",
            "risk_checkpoint": self.checkpoint_path,
            "risk_checkpoint_sha256": self.checkpoint_sha256,
            "risk_model_role": "independent_observation_only_command_risk_head",
            "locomotion_actor_checkpoint": self.actor_checkpoint_path,
            "locomotion_actor_checkpoint_sha256": self.actor_checkpoint_sha256,
            "locomotion_actor_role": "single_rsl_joint_action_policy",
            "risk_model": dict(self.metadata),
            "governor_config": dict(vars(self.governor.cfg)),
            "causal_order": [
                "raw_1864_motion_observation_to_independent_risk",
                "optional_hall_packet_health_upper_bound",
                "hall_traction_governor",
                "apply_effective_command",
                "rewrite_all_five_command_frames_30_to_45",
                "single_rsl_actor",
            ],
            "decision_input_contract": (
                "Hall Bx/By/Bz history, proprioception, motion feedback and Hall "
                "packet health only; no true mu/contact/force/slip/course stage"
            ),
            "command_history_indices": {
                "vx": list(RECOVERY_COMMAND_VX_INDICES),
                "vy": list(RECOVERY_COMMAND_VY_INDICES),
                "yaw": list(RECOVERY_COMMAND_YAW_INDICES),
            },
        }


def _callable_name(value) -> str:
    return str(getattr(value, "__name__", type(value).__name__))


def _require_same_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        max_abs = float((actual - expected).abs().max().item())
        raise RuntimeError(f"{name} value/order mismatch (max_abs={max_abs:.9g})")


def _audit_actor_boundary(base_env, observation) -> dict[str, object]:
    """Fail closed on any runtime actor ABI or observation-order mismatch."""

    manager = base_env.observation_manager
    terms = tuple(manager.active_terms.get("policy", ()))
    policy_dim = int(manager.group_obs_dim["policy"][-1])
    if policy_dim != 1864:
        raise RuntimeError(f"expected deploy actor dimension 1864, got {policy_dim}")
    if terms != EXPECTED_POLICY_TERMS:
        raise RuntimeError(
            f"unexpected Hall actor terms; expected={EXPECTED_POLICY_TERMS}, active={terms}"
        )
    term_dims = tuple(
        tuple(int(value) for value in shape)
        for shape in manager.group_obs_term_dim["policy"]
    )
    if term_dims != EXPECTED_POLICY_TERM_DIMS:
        raise RuntimeError(
            "unexpected Hall actor term dimensions; "
            f"expected={EXPECTED_POLICY_TERM_DIMS}, active={term_dims}"
        )
    term_cfgs = tuple(manager._group_obs_term_cfgs["policy"])
    functions = tuple(_callable_name(cfg.func) for cfg in term_cfgs)
    if functions != EXPECTED_POLICY_FUNCTIONS:
        raise RuntimeError(
            "unexpected Hall actor term callables; "
            f"expected={EXPECTED_POLICY_FUNCTIONS}, active={functions}"
        )
    history_lengths = tuple(int(cfg.history_length) for cfg in term_cfgs)
    if history_lengths != EXPECTED_POLICY_HISTORY_LENGTHS:
        raise RuntimeError(
            "unexpected Hall actor history lengths; "
            f"expected={EXPECTED_POLICY_HISTORY_LENGTHS}, active={history_lengths}"
        )
    for term, cfg, history_length in zip(terms, term_cfgs, history_lengths):
        if history_length > 0 and not bool(cfg.flatten_history_dim):
            raise RuntimeError(f"policy history term {term!r} is not flattened")

    # Guard the less visible configuration details that can preserve the same
    # tensor dimension while changing its physical meaning.
    if term_cfgs[2].params.get("command_name") != "base_velocity":
        raise RuntimeError("velocity command observation is not base_velocity")
    if not math.isclose(
        float(term_cfgs[0].scale), 0.2, rel_tol=0.0, abs_tol=1.0e-7
    ) or not math.isclose(
        float(term_cfgs[4].scale), 0.05, rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise RuntimeError("base_ang_vel/joint_vel_rel observation scale mismatch")
    if tuple(term_cfgs[6].clip) != (-6.0, 6.0):
        raise RuntimeError("Hall magnetic observation clip must be [-6, 6]")
    if tuple(term_cfgs[7].clip) != (0.001, 0.25):
        raise RuntimeError("Hall sample-period clip must be [0.001, 0.25] seconds")
    if tuple(term_cfgs[8].clip) != (0.0, 1.0):
        raise RuntimeError("Hall valid-mask clip must be [0, 1]")
    motion_params = term_cfgs[9].params
    if (
        motion_params.get("asset_name") != "robot"
        or float(motion_params.get("lateral_velocity_clip", math.nan)) != 1.5
        or float(motion_params.get("heading_error_clip", math.nan)) != 1.0
    ):
        raise RuntimeError(
            "final two actor columns must be robot [body_vy, relative_heading] "
            "with clips [1.5 m/s, 1.0 rad]"
        )

    hall_asset_cfg = term_cfgs[6].params.get("asset_cfg")
    foot_body_names = tuple(getattr(hall_asset_cfg, "body_names", ()))
    if foot_body_names != EXPECTED_FOOT_BODY_NAMES or not bool(
        getattr(hall_asset_cfg, "preserve_order", False)
    ):
        raise RuntimeError(
            "Hall foot body order must be explicit left then right with preserve_order=True; "
            f"got names={foot_body_names}"
        )
    foot_body_ids = tuple(int(value) for value in hall_asset_cfg.body_ids)
    robot_body_names = tuple(base_env.scene["robot"].data.body_names[index] for index in foot_body_ids)
    if robot_body_names != EXPECTED_FOOT_BODY_NAMES:
        raise RuntimeError(
            "resolved Hall foot body IDs are not left then right: "
            f"ids={foot_body_ids}, names={robot_body_names}"
        )

    action_terms = tuple(base_env.action_manager.active_terms)
    if action_terms != ("JointPositionAction",):
        raise RuntimeError(f"unexpected policy action terms: {action_terms}")
    action_term = base_env.action_manager._terms["JointPositionAction"]
    if int(action_term.action_dim) != 29 or int(base_env.action_manager.total_action_dim) != 29:
        raise RuntimeError("Hall locomotion actor must produce exactly 29 joint actions")
    action_scale = action_term._scale
    if isinstance(action_scale, (float, int)):
        action_scale_tensor = torch.full(
            (29,), float(action_scale), device=base_env.device
        )
    else:
        action_scale_tensor = torch.as_tensor(action_scale, device=base_env.device).reshape(-1)
    if action_scale_tensor.numel() not in (29, 29 * base_env.num_envs):
        raise RuntimeError(f"unexpected JointPositionAction scale shape {tuple(action_scale_tensor.shape)}")
    if not torch.allclose(
        action_scale_tensor,
        torch.full_like(action_scale_tensor, 0.25),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("JointPositionAction scale must be 0.25 rad per actor unit")
    if not bool(action_term.cfg.use_default_offset):
        raise RuntimeError("JointPositionAction must use the trained default-position offset")
    actor_joint_names = tuple(action_term._joint_names)
    sdk_joint_names = tuple(base_env.cfg.scene.robot.joint_sdk_names)
    if len(actor_joint_names) != 29 or len(sdk_joint_names) != 29:
        raise RuntimeError(
            f"expected 29 actor/SDK joints, got {len(actor_joint_names)}/{len(sdk_joint_names)}"
        )
    try:
        joint_ids_map = tuple(sdk_joint_names.index(name) for name in actor_joint_names)
    except ValueError as exc:
        raise RuntimeError("actor joint name is absent from robot SDK order") from exc
    if joint_ids_map != EXPECTED_JOINT_IDS_MAP:
        raise RuntimeError(
            "actor-to-SDK joint map changed; this would permute the 29 policy actions: "
            f"expected={EXPECTED_JOINT_IDS_MAP}, active={joint_ids_map}"
        )

    policy = _policy_tensor(observation)
    history_buffers = manager._group_obs_term_history_buffer["policy"]
    slice_report: dict[str, list[int]] = {}
    for term, cfg, expected_slice, expected_dim in zip(
        terms, term_cfgs, EXPECTED_POLICY_SLICES, EXPECTED_POLICY_TERM_DIMS
    ):
        start, stop = expected_slice
        if stop - start != math.prod(expected_dim):
            raise RuntimeError(f"internal actor ABI slice error for {term!r}")
        actual = policy[:, start:stop]
        if cfg.history_length > 0:
            buffered = history_buffers[term].buffer.reshape(base_env.num_envs, -1)
            _require_same_tensor(f"policy slice {term}", actual, buffered)
        slice_report[term] = [start, stop]

    # Independent sensor/debug comparisons catch an accidentally reversed
    # history, foot, site or XYZ flatten order even when the total is 1864.
    hall_sensor = getattr(base_env, "_hall_foot_sensor", None)
    if hall_sensor is None:
        raise RuntimeError("actor boundary audit requires initialized HallFootSensor")
    hall_sensor_seed = getattr(base_env, "_hall_foot_sensor_seed", None)
    effective_env_seed = getattr(base_env.cfg, "seed", None)
    if hall_sensor_seed is None or int(hall_sensor_seed) != int(effective_env_seed):
        raise RuntimeError(
            "HallFootSensor RNG seed does not match the effective environment seed: "
            f"sensor={hall_sensor_seed!r}, env={effective_env_seed!r}"
        )
    debug = hall_sensor.get_debug_data()
    hall_latest = policy[:, 1830 - 90 : 1830]
    _require_same_tensor(
        "latest Hall frame [left,right,P00..P14,XYZ]",
        hall_latest,
        debug["policy_observation"].reshape(base_env.num_envs, 90),
    )
    _require_same_tensor(
        "latest Hall sample period [left,right]",
        policy[:, 1858:1860],
        debug["reported_sample_period"].reshape(base_env.num_envs, 2),
    )
    _require_same_tensor(
        "Hall valid [left,right]",
        policy[:, 1860:1862],
        debug["policy_valid_mask"].all(dim=-1).to(policy.dtype),
    )
    direct_motion = term_cfgs[9].func(base_env, **term_cfgs[9].params).clone()
    direct_motion.clip_(min=term_cfgs[9].clip[0], max=term_cfgs[9].clip[1])
    _require_same_tensor(
        "motion feedback [body_vy,relative_heading]",
        policy[:, 1862:1864],
        direct_motion,
    )

    leaked = [
        term
        for term in terms
        if any(token in term.lower() for token in FORBIDDEN_POLICY_TOKENS)
    ]
    if leaked:
        raise RuntimeError(f"privileged material/contact truth leaked into actor: {leaked}")
    return {
        "policy_dim": policy_dim,
        "policy_terms": list(terms),
        "policy_functions": list(functions),
        "policy_term_dims": [list(shape) for shape in term_dims],
        "policy_history_lengths": list(history_lengths),
        "policy_slices": slice_report,
        "trailing_feature_mode": "motion_feedback",
        "trailing_feature_names": ["body_lateral_velocity", "relative_heading_error"],
        "foot_order": ["left", "right"],
        "resolved_foot_body_ids": list(foot_body_ids),
        "resolved_foot_body_names": list(robot_body_names),
        "action_dim": int(action_term.action_dim),
        "action_scale_rad": 0.25,
        "action_uses_default_offset": True,
        "actor_joint_names": list(actor_joint_names),
        "sdk_joint_names": list(sdk_joint_names),
        "actor_to_sdk_joint_ids_map": list(joint_ids_map),
        "action_delay_steps": [
            int(getattr(action_term.cfg, "min_delay", 0)),
            int(getattr(action_term.cfg, "max_delay", 0)),
        ],
        "action_delay_probabilities": list(
            getattr(action_term.cfg, "delay_probabilities", ()) or ()
        ),
        "hall_flatten_order": "time_oldest_to_newest,left/right,P00..P14,XYZ",
        "hall_sensor_rng_seed": int(hall_sensor_seed),
        "runtime_value_order_checks": {
            "history_buffers_match_actor_slices": True,
            "latest_hall_frame_matches_sensor_packet": True,
            "latest_period_matches_sensor_packet": True,
            "valid_lr_matches_sensor_packet": True,
            "motion_feedback_matches_direct_function": True,
        },
        "truth_leaks": leaked,
    }


def _audit_usd_patches(base_env) -> list[dict[str, object]]:
    """Inspect actual authored USD schemas rather than trusting config text."""

    stage = base_env.scene.stage
    reports: list[dict[str, object]] = []
    for env_path in base_env.scene.env_prim_paths:
        for name, expected_mu, _, _ in _runtime_patches():
            root_path = f"{env_path}/{name}"
            mesh_path = f"{root_path}/geometry/mesh"
            material_path = f"{root_path}/geometry/material"
            root_prim = stage.GetPrimAtPath(root_path)
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            material_prim = stage.GetPrimAtPath(material_path)
            if not root_prim.IsValid() or not mesh_prim.IsValid() or not material_prim.IsValid():
                raise RuntimeError(
                    "missing spatial floor prim(s): "
                    f"root={root_path}, mesh={mesh_path}, material={material_path}"
                )
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"floor mesh lacks CollisionAPI: {mesh_path}")
            if root_prim.HasAPI(UsdPhysics.RigidBodyAPI) or mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"floor patch must be static (no RigidBodyAPI): {root_path}")

            material_api = UsdPhysics.MaterialAPI(material_prim)
            physx_api = PhysxSchema.PhysxMaterialAPI(material_prim)
            static_mu = float(material_api.GetStaticFrictionAttr().Get())
            dynamic_mu = float(material_api.GetDynamicFrictionAttr().Get())
            combine = str(physx_api.GetFrictionCombineModeAttr().Get())
            if not math.isclose(static_mu, expected_mu, rel_tol=0.0, abs_tol=1.0e-6):
                raise RuntimeError(f"wrong static friction at {material_path}: {static_mu}")
            if not math.isclose(dynamic_mu, expected_mu, rel_tol=0.0, abs_tol=1.0e-6):
                raise RuntimeError(f"wrong dynamic friction at {material_path}: {dynamic_mu}")
            if combine != "multiply":
                raise RuntimeError(f"expected multiply friction combine at {material_path}, got {combine!r}")
            reports.append(
                {
                    "env_path": env_path,
                    "collider_path": mesh_path,
                    "material_path": material_path,
                    "static": True,
                    "static_friction": static_mu,
                    "dynamic_friction": dynamic_mu,
                    "friction_combine_mode": combine,
                }
            )
    return reports


def _audit_contact_and_hall(base_env, observation) -> dict[str, object]:
    expected_filters = tuple(
        f"{base_env.scene.env_regex_ns}/{name}/geometry/mesh"
        for name, _, _, _ in _runtime_patches()
    )
    contact_report: dict[str, object] = {}
    for side in ("left", "right"):
        sensor = base_env.scene.sensors[f"{side}_hall_contact"]
        configured = tuple(sensor.cfg.filter_prim_paths_expr)
        force_matrix = sensor.data.force_matrix_w
        friction_forces = sensor.data.friction_forces_w
        if configured != expected_filters:
            raise RuntimeError(
                f"{side} Hall contact filter order mismatch: {configured} != {expected_filters}"
            )
        if sensor.contact_physx_view.filter_count != 3:
            raise RuntimeError(f"{side} Hall ContactSensor resolved != 3 filters")
        expected_shape = (base_env.num_envs, 1, 3, 3)
        if force_matrix is None or tuple(force_matrix.shape) != expected_shape:
            shape = None if force_matrix is None else tuple(force_matrix.shape)
            raise RuntimeError(f"{side} force_matrix_w shape {shape}, expected {expected_shape}")
        if not torch.isfinite(force_matrix).all():
            raise FloatingPointError(f"non-finite {side} Hall contact-force matrix")
        if friction_forces is None or tuple(friction_forces.shape) != expected_shape:
            shape = None if friction_forces is None else tuple(friction_forces.shape)
            raise RuntimeError(
                f"{side} friction_forces_w shape {shape}, expected {expected_shape}"
            )
        # Missing filtered pairs are NaN by ContactSensor contract; Inf is an
        # actual solver/data error and must fail the rollout.
        if torch.isinf(friction_forces).any():
            raise FloatingPointError(f"infinite {side} Hall friction-force matrix")
        contact_report[side] = {
            "filter_paths": list(configured),
            "filter_count": int(sensor.contact_physx_view.filter_count),
            "force_matrix_shape": list(force_matrix.shape),
            "friction_force_shape": list(friction_forces.shape),
            "friction_force_abs_max_N": float(
                torch.nan_to_num(friction_forces).abs().max().item()
            ),
        }

    hall_sensor = getattr(base_env, "_hall_foot_sensor", None)
    if hall_sensor is None:
        raise RuntimeError("Hall actor observation did not initialize HallFootSensor")
    raw = hall_sensor.get_raw_data()
    filtered = hall_sensor.get_filtered_data()
    expected_hall_shape = (base_env.num_envs, 2, 15, 3)
    if tuple(raw.shape) != expected_hall_shape or tuple(filtered.shape) != expected_hall_shape:
        raise RuntimeError(
            f"Hall tensor shapes raw={tuple(raw.shape)}, filtered={tuple(filtered.shape)}, "
            f"expected={expected_hall_shape}"
        )
    if not torch.isfinite(raw).all() or not torch.isfinite(filtered).all():
        raise FloatingPointError("non-finite raw/filtered Hall data")
    debug = hall_sensor.get_debug_data()
    normal_scale = debug["mechanical_normal_scale"]
    shear_scale = debug["mechanical_shear_scale"]
    delay_steps = debug["policy_delay_steps"]
    channel_keep = debug["policy_channel_keep"]
    foot_keep = debug["policy_foot_keep"]
    for name, value in (
        ("mechanical_normal_scale", normal_scale),
        ("mechanical_shear_scale", shear_scale),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite sampled Hall DR tensor {name}")
    _policy_tensor(observation)
    return {
        "hall_contact_distribution_mode": str(
            getattr(hall_sensor.cfg, "contact_distribution_mode", "unknown")
        ),
        "contact_sensors": contact_report,
        "hall_raw_shape": list(raw.shape),
        "hall_filtered_shape": list(filtered.shape),
        "hall_raw_abs_max_T": float(raw.abs().max().item()),
        "hall_filtered_abs_max_T": float(filtered.abs().max().item()),
        # A small robot-independent probe makes multi-seed Hall DR auditable
        # without exposing simulator force/contact truth to the actor.
        "hall_randomization_probe": {
            "normal_scale_first6": normal_scale.reshape(-1)[:6].detach().cpu().tolist(),
            "shear_scale_first6": shear_scale.reshape(-1)[:6].detach().cpu().tolist(),
            "delay_steps_first6": delay_steps.reshape(-1)[:6].detach().cpu().tolist(),
            "channel_keep_count": int(channel_keep.count_nonzero().item()),
            "foot_keep_count": int(foot_keep.count_nonzero().item()),
        },
    }


def _teleport_standing(base_env, local_x: float) -> None:
    """Place every robot at one course x while retaining its default stance."""

    robot = base_env.scene["robot"]
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += base_env.scene.env_origins[env_ids]
    root_state[:, 0] = base_env.scene.env_origins[env_ids, 0] + float(local_x)
    root_state[:, 1] = base_env.scene.env_origins[env_ids, 1]
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids],
        torch.zeros_like(robot.data.default_joint_vel[env_ids]),
        env_ids=env_ids,
    )
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)


def _contact_patch_state(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dominant patch index and >=5 N contact after combining feet."""

    force_by_filter = torch.zeros((base_env.num_envs, 3), device=base_env.device)
    for side in ("left", "right"):
        matrix = base_env.scene.sensors[f"{side}_hall_contact"].data.force_matrix_w
        if matrix is None:
            raise RuntimeError(f"{side} Hall ContactSensor has no force_matrix_w")
        # [N, one sensed foot, three filters, xyz] -> [N, three filters]
        force_by_filter += torch.linalg.vector_norm(torch.nan_to_num(matrix[:, 0]), dim=-1)
    return force_by_filter.argmax(dim=-1), force_by_filter.sum(dim=-1) >= 5.0


def _run_label_probe(env) -> dict[str, object]:
    """Physically verify contact-filter order and privileged H-L-H labels."""

    base_env = env.unwrapped
    action = torch.zeros(
        (base_env.num_envs, base_env.action_manager.total_action_dim),
        device=base_env.device,
    )
    reports: list[dict[str, object]] = []
    labels_for_compression: list[bool] = []
    for name, _, local_x, expected_filter in _runtime_patches():
        env.reset()
        _teleport_standing(base_env, local_x)
        low_samples: list[torch.Tensor] = []
        filter_samples: list[torch.Tensor] = []
        observation = None
        # Ignore the first half while contacts refresh after the teleport.
        keep_from = args_cli.label_probe_steps // 2
        for step in range(args_cli.label_probe_steps):
            _force_command(base_env, args_cli.command)
            observation, _, _, _, _ = env.step(action)
            if step >= keep_from:
                if not hasattr(base_env, "spatial_low_contact_buf"):
                    raise RuntimeError("privileged spatial_low_contact_buf was not initialized")
                low_samples.append(base_env.spatial_low_contact_buf.detach().clone())
                dominant_filter, _ = _contact_patch_state(base_env)
                filter_samples.append(dominant_filter.detach().clone())
        if observation is None:
            raise RuntimeError("label probe did not step the environment")
        low_stack = torch.stack(low_samples)
        filter_stack = torch.stack(filter_samples)
        low_fraction = float(low_stack.float().mean().item())
        filter_fraction = float((filter_stack == expected_filter).float().mean().item())
        expected_low = expected_filter == 1
        label_fraction = low_fraction if expected_low else 1.0 - low_fraction
        if label_fraction < args_cli.probe_min_fraction:
            raise RuntimeError(
                f"{name} privileged label mismatch: expected_low={expected_low}, "
                f"matching_fraction={label_fraction:.3f}"
            )
        if filter_fraction < args_cli.probe_min_fraction:
            raise RuntimeError(
                f"{name} contact filter mismatch: expected index {expected_filter}, "
                f"matching_fraction={filter_fraction:.3f}"
            )
        labels_for_compression.append(expected_low)
        reports.append(
            {
                "patch": name,
                "local_x_m": local_x,
                "expected_low": expected_low,
                "low_label_fraction": low_fraction,
                "expected_filter_index": expected_filter,
                "dominant_filter_fraction": filter_fraction,
            }
        )
    compressed = compress_contact_labels(labels_for_compression)
    if compressed != ["HIGH", "LOW", "HIGH"]:
        raise RuntimeError(f"expected privileged H-L-H probe, got {compressed}")
    return {"compressed_labels": compressed, "patches": reports}


class _ZeroPolicy:
    kind = "zero_action_smoke"

    def __init__(self, base_env):
        self._shape = (base_env.num_envs, base_env.action_manager.total_action_dim)
        self._device = base_env.device

    def __call__(self, observation) -> torch.Tensor:
        _policy_tensor(observation)
        return torch.zeros(self._shape, device=self._device)


class _OnnxPolicy:
    kind = "onnx"

    def __init__(self, path: Path, base_env):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("--onnx requires onnxruntime") from exc
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"ONNX actor must have one input, found {len(inputs)}")
        self._input_name = inputs[0].name
        self._input_shape = inputs[0].shape
        if len(self._input_shape) != 2 or self._input_shape[-1] != 1864:
            raise RuntimeError(f"ONNX input must be [batch, 1864], got {self._input_shape}")
        self._device = base_env.device

    def __call__(self, observation) -> torch.Tensor:
        array = _policy_tensor(observation).detach().cpu().numpy().astype(np.float32, copy=False)
        fixed_batch = self._input_shape[0] if isinstance(self._input_shape[0], int) else None
        if fixed_batch is None or fixed_batch == array.shape[0]:
            output = self._session.run(None, {self._input_name: array})[0]
        elif fixed_batch == 1:
            output = np.concatenate(
                [self._session.run(None, {self._input_name: row[None]})[0] for row in array],
                axis=0,
            )
        else:
            raise RuntimeError(
                f"ONNX fixed batch {fixed_batch} is incompatible with num_envs={array.shape[0]}"
            )
        actions = torch.as_tensor(output, device=self._device, dtype=torch.float32)
        if actions.ndim != 2:
            raise RuntimeError(f"ONNX actor output must be rank two, got {tuple(actions.shape)}")
        return actions


class _BlendPolicy:
    kind = "fixed_onnx_action_blend"

    def __init__(self, first: _OnnxPolicy, second: _OnnxPolicy, alpha: float):
        self._first = first
        self._second = second
        self._alpha = float(alpha)

    def __call__(self, observation) -> torch.Tensor:
        first = self._first(observation)
        second = self._second(observation)
        return torch.lerp(first, second, self._alpha)


class _CausalHallHybridPolicy:
    """Frozen fast actor plus a Hall-risk-gated low-grip recovery actor.

    The selector consumes only the deployable 1864-D observation.  Friction,
    contact filters and the spatial course label remain evaluator diagnostics;
    they never enter this action path.
    """

    kind = "causal_hall_hybrid_onnx"

    def __init__(
        self,
        baseline_path: Path,
        recovery_path: Path,
        risk_checkpoint: Path,
        base_env,
        risk_start: float,
        risk_full: float,
        on_steps: int,
        off_steps: int,
        max_active_steps: int,
        recovery_command: float,
    ):
        self._baseline = _OnnxPolicy(baseline_path, base_env)
        self._recovery = _OnnxPolicy(recovery_path, base_env)
        payload = torch.load(
            risk_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Hall risk checkpoint must contain a dictionary")
        self._risk = build_hall_risk_estimator(payload).to(base_env.device).eval()
        self._start = float(risk_start)
        self._full = float(risk_full)
        self._on_steps = int(on_steps)
        self._off_steps = int(off_steps)
        self._max_active_steps = int(max_active_steps)
        self._recovery_command = float(recovery_command)
        self._active = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.bool)
        self._high_count = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        self._low_count = torch.zeros_like(self._high_count)
        self._active_age = torch.zeros_like(self._high_count)
        self._last_gate = torch.zeros(base_env.num_envs, device=base_env.device)

    def __call__(self, observation) -> torch.Tensor:
        obs = _policy_tensor(observation)
        with torch.inference_mode():
            risk = self._risk(obs).reshape(-1).clamp(0.0, 1.0)
            healthy = torch.isfinite(obs[:, VALID_SLICE]).all(dim=1) & (
                obs[:, VALID_SLICE] > 0.5
            ).all(dim=1)
            high = (risk >= self._full) & healthy
            low = risk <= self._start
            self._high_count = torch.where(
                high, self._high_count + 1, torch.zeros_like(self._high_count)
            )
            self._low_count = torch.where(
                low, self._low_count + 1, torch.zeros_like(self._low_count)
            )
            self._active = torch.where(
                ~healthy,
                torch.zeros_like(self._active),
                torch.where(
                self._high_count >= self._on_steps,
                torch.ones_like(self._active),
                torch.where(
                    self._low_count >= self._off_steps,
                    torch.zeros_like(self._active),
                    self._active,
                ),
                ),
            )
            self._active_age = torch.where(
                self._active,
                self._active_age + 1,
                torch.zeros_like(self._active_age),
            )
            if self._max_active_steps:
                expired = self._active_age >= self._max_active_steps
                self._active = torch.where(expired, torch.zeros_like(self._active), self._active)
                self._active_age = torch.where(
                    expired, torch.zeros_like(self._active_age), self._active_age
                )
            # Once the causal state machine has latched, use a full recovery
            # action.  This avoids repeatedly mixing two policies with
            # incompatible command/history distributions.
            gate = self._active.to(dtype=torch.float32)
        baseline = self._baseline(obs)
        # The Stage7 recovery expert was trained with a 0.16 m/s crawl
        # command.  Align only its internal command history during the bounded
        # handoff; the environment/requested command remains unchanged.
        recovery_obs = _with_recovery_command_history(
            obs, self._recovery_command
        )
        recovery = self._recovery(recovery_obs)
        self._last_gate = gate.detach()
        return torch.lerp(baseline, recovery, gate[:, None])


class _TorchScriptPolicy:
    kind = "torchscript"

    def __init__(self, path: Path, base_env):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._model = torch.jit.load(str(path), map_location=base_env.device).eval()

    def __call__(self, observation) -> torch.Tensor:
        output = self._model(_policy_tensor(observation))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError("TorchScript actor did not return a rank-two action tensor")
        return output


def _disable_eval_capture_gate_warmup(agent_cfg) -> None:
    """Disable a training-only phase before constructing an evaluation runner.

    Loading an older, already-trained FastBase residual into a newly configured
    gate-only warm-up would correctly fail the *training resume* invariant that
    the residual head must still be exact zero.  Evaluation neither updates nor
    resumes that optimizer phase, so force its update count to zero before the
    runner is constructed.  Checkpoint actor tensors and inference outputs are
    untouched.
    """

    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if isinstance(algorithm_cfg, dict):
        if "capture_gate_warmup_updates" in algorithm_cfg:
            algorithm_cfg["capture_gate_warmup_updates"] = 0
    elif algorithm_cfg is not None and hasattr(
        algorithm_cfg, "capture_gate_warmup_updates"
    ):
        algorithm_cfg.capture_gate_warmup_updates = 0


class _RslPolicy:
    kind = "rsl_rl_pt"

    def __init__(self, wrapped_env, base_env):
        path = Path(args_cli.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if args_cli.rsl_rl_cfg_entry_point:
            if not isinstance(args_cli.rsl_rl_cfg_entry_point, str) or ":" not in (
                args_cli.rsl_rl_cfg_entry_point
            ):
                raise ValueError(
                    "rsl_rl_cfg_entry_point must be 'module.qualname:ClassName'"
                )
            agent_cfg = resolve_callable(args_cli.rsl_rl_cfg_entry_point)()
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        else:
            agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
        _disable_eval_capture_gate_warmup(agent_cfg)
        runner_class = resolve_callable(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
        if not isinstance(runner_class, type) or not issubclass(runner_class, OnPolicyRunner):
            raise RuntimeError(f"unsupported checkpoint runner {agent_cfg.class_name!r}")
        runner = runner_class(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        # Evaluation needs only the deterministic actor.  Loading optimizer or
        # iteration state can conflict with a newer training-only gate warm-up
        # (different optimizer groups/counters) and cannot affect inference.
        runner.load(
            str(path),
            load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG),
            strict=True,
        )
        self._policy = runner.get_inference_policy(device=base_env.device)
        self._capture_module = self._resolve_capture_module(runner)

    @staticmethod
    def _resolve_capture_module(runner):
        """Find the optional observation-only FastBase diagnostic interface.

        The current native RSL actor owns ``FastBaseHallCaptureResidual`` as
        ``runner.alg.actor.mlp``.  Looking up the small callable protocol keeps
        ordinary RSL actors fully compatible and avoids importing a concrete
        training class into this evaluator.
        """

        actor = getattr(runner.alg, "actor", None)
        candidate = getattr(actor, "mlp", None)
        required = (
            "capture_probability",
            "effective_capture_probability",
            "capture_delta",
        )
        if candidate is not None and all(
            callable(getattr(candidate, name, None)) for name in required
        ):
            return candidate
        return None

    def __call__(self, observation) -> torch.Tensor:
        _policy_tensor(observation)
        return self._policy(observation)

    def capture_diagnostics(self, observation) -> dict[str, torch.Tensor] | None:
        """Read FastBase internals using only the deployable observation.

        Course stage, contact filters and friction values are intentionally
        absent from this call.  They are attached later as evaluator labels.
        """

        if self._capture_module is None:
            return None
        policy_observation = _policy_tensor(observation)
        raw_reader = getattr(self._capture_module, "raw_capture_probability", None)
        probability = self._capture_module.capture_probability(policy_observation).reshape(-1)
        raw_probability = (
            raw_reader(policy_observation).reshape(-1)
            if callable(raw_reader)
            else probability
        )
        effective_gate = self._capture_module.effective_capture_probability(
            policy_observation
        ).reshape(-1)
        delta = self._capture_module.capture_delta(policy_observation)
        expected_batch = policy_observation.shape[0]
        if (
            raw_probability.shape != (expected_batch,)
            or probability.shape != (expected_batch,)
            or effective_gate.shape != (expected_batch,)
        ):
            raise RuntimeError(
                "FastBase capture gates must return one value per environment, "
                f"got raw={tuple(raw_probability.shape)}, "
                f"probability={tuple(probability.shape)}, "
                f"effective={tuple(effective_gate.shape)}"
            )
        if delta.ndim != 2 or delta.shape[0] != expected_batch:
            raise RuntimeError(
                "FastBase capture_delta must return [num_envs, action_dim], "
                f"got {tuple(delta.shape)}"
            )
        delta_l2 = torch.linalg.vector_norm(delta, ord=2, dim=1)
        stability_authority_reader = getattr(
            self._capture_module, "stability_authority", None
        )
        stability_delta_reader = getattr(
            self._capture_module, "stability_delta", None
        )
        stability_authority = None
        stability_delta_l2 = None
        stability_delta_abs_max = None
        if callable(stability_authority_reader) != callable(stability_delta_reader):
            raise RuntimeError(
                "FastBase stability diagnostic interface is only partially implemented"
            )
        if callable(stability_authority_reader):
            stability_authority = stability_authority_reader(
                policy_observation
            ).reshape(-1)
            stability_delta = stability_delta_reader(policy_observation)
            if stability_delta.shape != delta.shape:
                raise RuntimeError(
                    "FastBase stability_delta shape changed: "
                    f"{tuple(stability_delta.shape)} != {tuple(delta.shape)}"
                )
            stability_delta_l2 = torch.linalg.vector_norm(
                stability_delta, ord=2, dim=1
            )
            stability_delta_abs_max = stability_delta.abs().amax(dim=1)
            stability_limit = float(self._capture_module.stability_limit)
            if bool(
                (stability_delta_abs_max > stability_limit + 1.0e-6)
                .any()
                .item()
            ):
                raise RuntimeError("FastBase stability residual exceeded its hard limit")
        if not all(
            torch.isfinite(value).all()
            for value in (
                raw_probability,
                probability,
                effective_gate,
                delta_l2,
                *(() if stability_authority is None else (
                    stability_authority,
                    stability_delta_l2,
                    stability_delta_abs_max,
                )),
            )
        ):
            raise FloatingPointError("non-finite FastBase capture diagnostic")
        if bool(((raw_probability < 0.0) | (raw_probability > 1.0)).any().item()):
            raise RuntimeError("FastBase raw_capture_probability left [0, 1]")
        if bool(((probability < 0.0) | (probability > 1.0)).any().item()):
            raise RuntimeError("FastBase capture_probability left [0, 1]")
        if bool(((effective_gate < 0.0) | (effective_gate > 1.0)).any().item()):
            raise RuntimeError("FastBase effective capture gate left [0, 1]")
        result = {
            "raw_capture_probability": raw_probability,
            "capture_probability": probability,
            "effective_gate": effective_gate,
            "delta_l2": delta_l2,
        }
        if stability_authority is not None:
            if bool(
                ((stability_authority < 0.0) | (stability_authority > 1.0))
                .any()
                .item()
            ):
                raise RuntimeError("FastBase stability authority left [0, 1]")
            result.update(
                {
                    "stability_authority": stability_authority,
                    "stability_delta_l2": stability_delta_l2,
                    "stability_delta_abs_max": stability_delta_abs_max,
                }
            )
        return result


def _read_fastbase_capture_diagnostics(policy, observation):
    """Return optional actor diagnostics without changing the actor boundary."""

    read = getattr(policy, "capture_diagnostics", None)
    return read(observation) if callable(read) else None


def _course_stage_from_local_x(local_x: torch.Tensor) -> torch.Tensor:
    """Diagnostic-only H/L/H region labels aligned with pre-action samples."""

    geometry = _runtime_course_geometry()
    low_start_x = float(geometry["low_start_x_m"])
    low_end_x = float(geometry["low_end_x_m"])

    return torch.where(
        local_x < low_start_x,
        torch.zeros_like(local_x, dtype=torch.long),
        torch.where(
            local_x < low_end_x,
            torch.ones_like(local_x, dtype=torch.long),
            torch.full_like(local_x, 2, dtype=torch.long),
        ),
    )


def _fall_mask(dones: torch.Tensor, extras) -> torch.Tensor:
    timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
    if timeouts is None:
        return dones.bool()
    return dones.bool() & ~timeouts.to(device=dones.device).bool()


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(dtype=torch.float32).reshape(-1)
    return {
        "min": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "max": float(flat.max().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


def _capture_initial_hall_fault_state(base_env) -> dict[str, object]:
    """Capture the sampled episode-correlated Hall state before rollout.

    These tensors prove that an effective hardened run really sampled faults;
    reporting only non-zero configured probabilities is insufficient.
    """

    sensor = base_env._hall_foot_sensor
    debug = sensor.get_debug_data()
    channel_keep = debug["policy_channel_keep"].squeeze(-1).bool()
    foot_keep = debug["policy_foot_keep"].squeeze(-1).squeeze(-1).bool()
    delay = debug["policy_delay_steps"].to(dtype=torch.long)
    sample_period = debug["reported_sample_period"]
    env_rows: list[dict[str, object]] = []
    for env_id in range(base_env.num_envs):
        channels = channel_keep[env_id].detach().cpu().tolist()
        feet = foot_keep[env_id].detach().cpu().tolist()
        delays = delay[env_id].detach().cpu().tolist()
        env_rows.append(
            {
                "env_id": env_id,
                "group": classify_hall_health(channels, feet),
                "online_feet": int(sum(bool(value) for value in feet)),
                "live_channels_left": int(channel_keep[env_id, 0].sum().item()),
                "live_channels_right": int(channel_keep[env_id, 1].sum().item()),
                "delay_steps_left": int(delays[0]),
                "delay_steps_right": int(delays[1]),
            }
        )
    group_counts: dict[str, int] = {}
    for row in env_rows:
        name = str(row["group"])
        group_counts[name] = group_counts.get(name, 0) + 1

    cross_axis = sensor._policy_cross_axis
    identity = torch.eye(3, device=cross_axis.device).view(1, 1, 1, 3, 3)
    off_diagonal = (cross_axis - identity) * (1.0 - identity)
    return {
        "health_group_counts": group_counts,
        "envs": env_rows,
        "sampled_statistics": {
            "channel_keep_fraction": float(channel_keep.float().mean().item()),
            "foot_keep_fraction": float(foot_keep.float().mean().item()),
            "delay_steps": _tensor_stats(delay),
            "reported_sample_period_s": _tensor_stats(sample_period),
            "normal_stiffness_scale": _tensor_stats(debug["mechanical_normal_scale"]),
            "shear_stiffness_scale": _tensor_stats(debug["mechanical_shear_scale"]),
            "damping_scale": _tensor_stats(debug["mechanical_damping_scale"]),
            "magnetic_moment_scale": _tensor_stats(sensor._magnetic_moment_scale),
            "magnet_position_jitter_m": _tensor_stats(sensor._magnet_position_jitter),
            "observation_gain": _tensor_stats(sensor._policy_gain),
            "observation_cross_axis_off_diagonal": _tensor_stats(off_diagonal),
            "observation_zero_residual": _tensor_stats(sensor._policy_zero_residual),
        },
    }


def _effective_hall_cfg(hall_cfg) -> dict[str, object]:
    fields = (
        "implementation_mode",
        "sensor_sample_rate",
        "low_pass_cutoff",
        "auto_zero",
        "auto_zero_samples",
        "enable_domain_randomization",
        "normal_stiffness_scale_range",
        "shear_stiffness_scale_range",
        "damping_scale_range",
        "contact_spread_scale_range",
        "magnetic_moment_scale_range",
        "magnet_position_jitter_std",
        "observation_sensor_gain_range",
        "observation_axis_gain_range",
        "observation_cross_axis_std",
        "observation_zero_residual_std",
        "dead_channel_probability",
        "foot_dropout_probability",
        "reported_sample_period_range",
        "maximum_packet_delay_steps",
        "detailed_contact_force_atol",
        "detailed_contact_force_rtol",
        "detailed_contact_fail_on_audit_mismatch",
    )
    result: dict[str, object] = {}
    for name in fields:
        value = getattr(hall_cfg, name)
        result[name] = list(value) if isinstance(value, tuple) else value
    return result


def _mean_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _leg_action_pairs(base_env) -> list[tuple[int, int, bool]]:
    """Resolve mirrored leg joint names to JointPositionAction indices."""

    robot = base_env.scene["robot"]
    pairs: list[tuple[int, int, bool]] = []
    for source, opposite in (
        (LEG_ACTION_EQUAL_PAIRS, False),
        (LEG_ACTION_OPPOSITE_PAIRS, True),
    ):
        for left_name, right_name in source:
            left_index = int(robot.find_joints(left_name)[0][0])
            right_index = int(robot.find_joints(right_name)[0][0])
            pairs.append((left_index, right_index, opposite))
    return pairs


def _summarize_hall_health_performance(
    health: dict[str, object],
    *,
    completed: torch.Tensor,
    fallen: torch.Tensor,
    per_env_region_speed: dict[str, list[float]],
    response: dict[str, object],
) -> dict[str, object]:
    response_rows = response["per_env"]
    rows_by_group: dict[str, list[dict[str, object]]] = {}
    for row in health["envs"]:
        rows_by_group.setdefault(str(row["group"]), []).append(row)
    result: dict[str, object] = {}
    for group, rows in sorted(rows_by_group.items()):
        ids = [int(row["env_id"]) for row in rows]
        decel_05 = [
            response_rows[env_id]["deceleration"]["0.5s"]["deceleration_m_s"]
            for env_id in ids
        ]
        decel_10 = [
            response_rows[env_id]["deceleration"]["1s"]["deceleration_m_s"]
            for env_id in ids
        ]
        recovery = [
            response_rows[env_id]["absolute_recovery_time_s"] for env_id in ids
        ]
        result[group] = {
            "envs": len(ids),
            "fall_envs": int(fallen[ids].sum().item()),
            "completed_hlh_envs": int(completed[ids].sum().item()),
            "mean_body_vx_m_s": {
                name: _mean_or_none([per_env_region_speed[name][env_id] for env_id in ids])
                for name in ("high_start", "low", "high_end")
            },
            "mean_deceleration_0_5s_m_s": _mean_or_none(
                [float(value) for value in decel_05 if value is not None]
            ),
            "mean_deceleration_1_0s_m_s": _mean_or_none(
                [float(value) for value in decel_10 if value is not None]
            ),
            "absolute_high_recovery_fraction": (
                sum(value is not None for value in recovery) / max(len(ids), 1)
            ),
            "mean_absolute_high_recovery_time_s": _mean_or_none(
                [float(value) for value in recovery if value is not None]
            ),
        }
    return result


def _run_rollout(
    env,
    policy,
    health_envelope: HealthEnvelope | None = None,
    hall_command_governor: _StrictHallCommandGovernor | None = None,
    stability_envelope: HighSpeedStabilityEnvelope | None = None,
    stability_recovery_blend: StabilityRecoveryBlend | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base_env = env.unwrapped
    observation, _ = env.reset()
    _policy_tensor(observation)
    if health_envelope is not None:
        health_envelope.reset()
    if hall_command_governor is not None:
        if stability_envelope is not None or stability_recovery_blend is not None:
            raise RuntimeError(
                "strict Hall command governor cannot be stacked with stability layers"
            )
        hall_command_governor.reset()
    if stability_envelope is not None:
        stability_envelope.reset()
    if stability_recovery_blend is not None:
        if stability_envelope is None:
            raise RuntimeError(
                "stability recovery requires the high-speed stability envelope"
            )
        stability_recovery_blend.reset()
    initial_hall_fault_state = _capture_initial_hall_fault_state(base_env)
    stages = [0] * base_env.num_envs
    completed = torch.zeros(
        base_env.num_envs, dtype=torch.bool, device=base_env.device
    )
    # Reaching the first HIGH_END contact completes the causal H-L-H state
    # machine, but it does not end the managed episode.  Keep sampling that
    # first episode until its real done/course-success/fall transition so gate
    # release and recovered high-speed behavior are not truncated at entry.
    first_episode_active = torch.ones(
        base_env.num_envs, dtype=torch.bool, device=base_env.device
    )
    course_success_events = 0
    fallen = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    first_fall_step = torch.full(
        (base_env.num_envs,), -1, dtype=torch.long, device=base_env.device
    )
    first_fall_cross_track = torch.full(
        (base_env.num_envs,), float("nan"), device=base_env.device
    )
    max_abs_cross_track = torch.zeros(base_env.num_envs, device=base_env.device)
    drift_vy_sum = torch.zeros(base_env.num_envs, device=base_env.device)
    drift_vy_sq_sum = torch.zeros(base_env.num_envs, device=base_env.device)
    drift_heading_sq_sum = torch.zeros(base_env.num_envs, device=base_env.device)
    drift_sample_count = torch.zeros(
        base_env.num_envs, dtype=torch.long, device=base_env.device
    )
    drift_region_heading_sq = {
        key: torch.zeros(base_env.num_envs, device=base_env.device)
        for key in ("high_start", "low", "high_end")
    }
    drift_region_heading_sum = {
        key: torch.zeros(base_env.num_envs, device=base_env.device)
        for key in ("high_start", "low", "high_end")
    }
    drift_region_vy_sq = {
        key: torch.zeros(base_env.num_envs, device=base_env.device)
        for key in ("high_start", "low", "high_end")
    }
    drift_region_count = {
        key: torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
        for key in ("high_start", "low", "high_end")
    }
    drift_low_action_asym_sum = torch.zeros(
        base_env.num_envs, device=base_env.device
    )
    drift_low_action_count = torch.zeros(
        base_env.num_envs, dtype=torch.long, device=base_env.device
    )
    drift_low_slip_asym_sum = torch.zeros(
        base_env.num_envs, device=base_env.device
    )
    drift_low_slip_count = torch.zeros(
        base_env.num_envs, dtype=torch.long, device=base_env.device
    )
    leg_action_pairs = _leg_action_pairs(base_env)
    robot_body_names = base_env.scene["robot"].data.body_names
    foot_body_ids = tuple(
        int(robot_body_names.index(name)) for name in EXPECTED_FOOT_BODY_NAMES
    )
    drift_final_cross_track = torch.full(
        (base_env.num_envs,), float("nan"), device=base_env.device
    )
    falls_total = 0
    nan_detected = False
    region_speed_sum = {"high_start": 0.0, "low": 0.0, "high_end": 0.0}
    region_speed_count = {key: 0 for key in region_speed_sum}
    per_env_region_speed_sum = {
        key: torch.zeros(base_env.num_envs, device=base_env.device)
        for key in region_speed_sum
    }
    per_env_region_speed_count = {
        key: torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
        for key in region_speed_sum
    }
    # First-episode touchdown diagnostics expose the learned cadence/stride
    # mechanism without prescribing it.  Contact is read from the two
    # dedicated Hall contact sensors, while forward distance is the robot-root
    # course coordinate at each same-foot touchdown.  No value below is passed
    # to the actor or used to alter the command.
    left_gait_contact = base_env.scene["left_hall_contact"]
    right_gait_contact = base_env.scene["right_hall_contact"]

    def _gait_contact_state() -> torch.Tensor:
        rows = []
        expected = (base_env.num_envs, 1, 3)
        for side, sensor in (
            ("left", left_gait_contact),
            ("right", right_gait_contact),
        ):
            force = sensor.data.net_forces_w
            if not isinstance(force, torch.Tensor) or tuple(force.shape) != expected:
                raise RuntimeError(
                    f"{side} Hall contact force must have shape {expected}, "
                    f"got {getattr(force, 'shape', None)}"
                )
            if not torch.isfinite(force).all():
                raise RuntimeError(f"{side} Hall contact force contains NaN/Inf")
            rows.append(torch.linalg.vector_norm(force[:, 0, :], dim=-1) > 5.0)
        return torch.stack(rows, dim=1)

    gait_prev_contact = _gait_contact_state()
    gait_air_steps = torch.zeros(
        (base_env.num_envs, 2), dtype=torch.int64, device=base_env.device
    )
    gait_steps_since_touchdown = torch.full(
        (base_env.num_envs, 2), 10_000, dtype=torch.int64, device=base_env.device
    )
    gait_last_touchdown_forward = torch.full(
        (base_env.num_envs, 2), float("nan"), device=base_env.device
    )
    gait_region_id = torch.full(
        (base_env.num_envs,), -1, dtype=torch.int64, device=base_env.device
    )
    gait_minimum_air_steps = max(int(round(0.08 / float(base_env.step_dt))), 1)
    gait_minimum_touchdown_gap_steps = max(
        int(round(0.20 / float(base_env.step_dt))), 1
    )
    gait_region_data = {
        name: {
            "failure_free_exposure_s": 0.0,
            "touchdowns": 0,
            "stride_sum_m": 0.0,
            "stride_count": 0,
        }
        for name in region_speed_sum
    }
    label_trace: list[bool] = []
    trace_x: list[float] = []
    trace_vx: list[float] = []
    trace_low: list[bool] = []
    trace_contact_patch: list[int] = []
    trace_hall: list[np.ndarray] = []
    trace_valid: list[np.ndarray] = []
    recovery_state_pose: list[np.ndarray] = []
    recovery_state_velocity: list[np.ndarray] = []
    recovery_state_joint_pos: list[np.ndarray] = []
    recovery_state_joint_vel: list[np.ndarray] = []
    recovery_state_observation: list[np.ndarray] = []
    recovery_state_motion_initial_yaw: list[np.ndarray] = []
    recovery_state_heading_reference: list[np.ndarray] = []
    recovery_state_track_origin_local: list[np.ndarray] = []
    recovery_state_track_lateral_axis: list[np.ndarray] = []
    recovery_state_env_id: list[np.ndarray] = []
    recovery_state_rollout_step: list[np.ndarray] = []
    recovery_hall_local_deformation: list[np.ndarray] = []
    recovery_hall_loading_history: list[np.ndarray] = []
    recovery_hall_signal_filtered_absolute: list[np.ndarray] = []
    recovery_hall_signal_processed: list[np.ndarray] = []
    recovery_hall_signal_baseline: list[np.ndarray] = []
    recovery_hall_signal_drift: list[np.ndarray] = []
    recovery_hall_policy_history: list[np.ndarray] = []
    recovery_hall_policy_gain: list[np.ndarray] = []
    recovery_hall_policy_cross_axis: list[np.ndarray] = []
    recovery_hall_policy_zero_residual: list[np.ndarray] = []
    recovery_hall_policy_channel_keep: list[np.ndarray] = []
    recovery_hall_policy_foot_keep: list[np.ndarray] = []
    recovery_hall_policy_delay_steps: list[np.ndarray] = []
    recovery_hall_reported_sample_period: list[np.ndarray] = []
    dataset_obs: list[np.ndarray] = []
    dataset_actions: list[np.ndarray] = []
    dataset_low: list[np.ndarray] = []
    failure_obs_rows: list[np.ndarray] = []
    failure_action_rows: list[np.ndarray] = []
    failure_env_rows: list[np.ndarray] = []
    failure_step_rows: list[np.ndarray] = []
    failure_time_rows: list[np.ndarray] = []
    failure_local_x_rows: list[np.ndarray] = []
    failure_root_pose_rows: list[np.ndarray] = []
    failure_root_lin_vel_b_rows: list[np.ndarray] = []
    failure_root_ang_vel_b_rows: list[np.ndarray] = []
    failure_joint_pos_rows: list[np.ndarray] = []
    failure_joint_vel_rows: list[np.ndarray] = []
    failure_contact_rows: list[np.ndarray] = []
    failure_stage_rows: list[np.ndarray] = []
    failure_gate_rows: list[np.ndarray] = []
    failure_capture_delta_rows: list[np.ndarray] = []
    failure_stability_authority_rows: list[np.ndarray] = []
    failure_stability_delta_rows: list[np.ndarray] = []
    failure_fall_rows: list[np.ndarray] = []
    failure_done_rows: list[np.ndarray] = []
    failure_timeout_rows: list[np.ndarray] = []
    capture_raw_probability_rows: list[np.ndarray] = []
    capture_probability_rows: list[np.ndarray] = []
    capture_effective_gate_rows: list[np.ndarray] = []
    capture_delta_l2_rows: list[np.ndarray] = []
    stability_authority_rows: list[np.ndarray] = []
    stability_delta_l2_rows: list[np.ndarray] = []
    stability_delta_abs_max_rows: list[np.ndarray] = []
    capture_stage_rows: list[np.ndarray] = []
    capture_step_rows: list[np.ndarray] = []
    capture_time_rows: list[np.ndarray] = []
    capture_env_rows: list[np.ndarray] = []
    capture_interface_seen = False
    health_requested_rows: list[np.ndarray] = []
    health_target_rows: list[np.ndarray] = []
    health_effective_rows: list[np.ndarray] = []
    health_state_rows: list[np.ndarray] = []
    health_valid_rows: list[np.ndarray] = []
    health_age_rows: list[np.ndarray] = []
    health_finite_rows: list[np.ndarray] = []
    health_foot_healthy_rows: list[np.ndarray] = []
    health_recovery_timer_rows: list[np.ndarray] = []
    health_intervened_rows: list[np.ndarray] = []
    health_step_rows: list[np.ndarray] = []
    health_time_rows: list[np.ndarray] = []
    health_env_rows: list[np.ndarray] = []
    hall_governor_risk_rows: list[np.ndarray] = []
    hall_governor_filtered_risk_rows: list[np.ndarray] = []
    hall_governor_state_rows: list[np.ndarray] = []
    hall_governor_requested_rows: list[np.ndarray] = []
    hall_governor_upstream_rows: list[np.ndarray] = []
    hall_governor_effective_rows: list[np.ndarray] = []
    hall_governor_valid_rows: list[np.ndarray] = []
    hall_governor_probing_rows: list[np.ndarray] = []
    hall_governor_prebrake_rows: list[np.ndarray] = []
    hall_governor_step_rows: list[np.ndarray] = []
    hall_governor_time_rows: list[np.ndarray] = []
    hall_governor_env_rows: list[np.ndarray] = []
    stability_upstream_rows: list[np.ndarray] = []
    stability_effective_rows: list[np.ndarray] = []
    stability_state_rows: list[np.ndarray] = []
    stability_reason_rows: list[np.ndarray] = []
    stability_intervened_rows: list[np.ndarray] = []
    stability_command_mean_rows: list[np.ndarray] = []
    stability_heading_command_mean_rows: list[np.ndarray] = []
    stability_heading_enabled_rows: list[np.ndarray] = []
    stability_heading_signed_rows: list[np.ndarray] = []
    stability_heading_rows: list[np.ndarray] = []
    stability_heading_correction_active_rows: list[np.ndarray] = []
    stability_heading_correction_yaw_rows: list[np.ndarray] = []
    stability_omega_rows: list[np.ndarray] = []
    stability_tilt_rows: list[np.ndarray] = []
    stability_previous_action_norm_rows: list[np.ndarray] = []
    stability_current_action_norm_rows: list[np.ndarray] = []
    stability_action_slew_norm_rows: list[np.ndarray] = []
    stability_action_saturation_count_rows: list[np.ndarray] = []
    stability_warn_count_rows: list[np.ndarray] = []
    stability_limit_count_rows: list[np.ndarray] = []
    stability_hard_limit_count_rows: list[np.ndarray] = []
    stability_recovery_count_rows: list[np.ndarray] = []
    stability_step_rows: list[np.ndarray] = []
    stability_time_rows: list[np.ndarray] = []
    stability_env_rows: list[np.ndarray] = []
    stability_recovery_gate_rows: list[np.ndarray] = []
    stability_recovery_active_rows: list[np.ndarray] = []
    stability_recovery_baseline_action_rows: list[np.ndarray] = []
    stability_recovery_expert_action_rows: list[np.ndarray] = []
    stability_recovery_output_action_rows: list[np.ndarray] = []
    stability_recovery_step_rows: list[np.ndarray] = []
    stability_recovery_time_rows: list[np.ndarray] = []
    stability_recovery_env_rows: list[np.ndarray] = []
    all_vx: list[list[float]] = []
    all_low: list[list[bool]] = []
    all_high_end_contact: list[list[bool]] = []
    all_falls: list[list[bool]] = []
    all_dones: list[list[bool]] = []
    steps_run = 0

    for step_index in range(args_cli.steps):
        health_output = None
        stability_output = None
        failure_keep_for_step = None
        hall_governor_risk = None
        hall_governor_state = None
        hall_governor_effective = None
        hall_governor_valid = None
        raw_governor_observation = None
        # Strict causal order: infer risk from the untouched environment
        # observation before any health cap or counterfactual command rewrite.
        # This callable has no material/contact/force/course-stage argument.
        if hall_command_governor is not None:
            raw_governor_observation = _policy_tensor(observation)
            hall_governor_risk = hall_command_governor.predict(
                raw_governor_observation
            )
            requested_command = torch.zeros(
                (base_env.num_envs, 3), device=base_env.device
            )
            requested_command[:, 0] = float(args_cli.command)
        if health_envelope is None:
            if hall_command_governor is not None:
                upstream_command = requested_command
            elif stability_envelope is None:
                _force_command(base_env, args_cli.command)
            else:
                upstream_command = torch.zeros(
                    (base_env.num_envs, 3), device=base_env.device
                )
                upstream_command[:, 0] = float(args_cli.command)
        else:
            requested_command = torch.zeros(
                (base_env.num_envs, 3), device=base_env.device
            )
            requested_command[:, 0] = float(args_cli.command)
            health_valid, health_age_s, health_finite = _read_health_envelope_inputs(
                base_env, observation
            )
            health_output = health_envelope.update(
                requested_command=requested_command,
                valid=health_valid,
                age_s=health_age_s,
                finite=health_finite,
            )
            upstream_command = health_output.effective_command

        if hall_command_governor is not None:
            assert raw_governor_observation is not None
            assert hall_governor_risk is not None
            valid_lr = raw_governor_observation[:, VALID_SLICE]
            hall_governor_valid = (
                torch.isfinite(raw_governor_observation).all(dim=1)
                & torch.isfinite(valid_lr).all(dim=1)
                & (valid_lr > 0.5).all(dim=1)
            )
            if health_output is not None:
                # Packet health can only make the decision more conservative.
                # Its already-slewed command is the governor request, and a
                # stale/unhealthy foot invalidates traction release.
                hall_governor_valid &= health_output.foot_healthy.all(dim=1)
            hall_governor_effective, hall_governor_state = (
                hall_command_governor.update(
                    upstream_command,
                    hall_governor_risk,
                    valid=hall_governor_valid,
                )
            )
            _apply_effective_command(base_env, hall_governor_effective)
            observation = _rewrite_actor_command_history(
                observation, hall_governor_effective
            )
            # Fail before actor inference if any of the five term-major frames
            # differs from the command actually applied to the simulator.
            synchronized = _policy_tensor(observation)
            for command_index in RECOVERY_COMMAND_VX_INDICES:
                if not torch.equal(
                    synchronized[:, command_index : command_index + 3],
                    hall_governor_effective.to(
                        device=synchronized.device, dtype=synchronized.dtype
                    ),
                ):
                    raise RuntimeError(
                        "Hall governor failed to synchronize all five actor "
                        "command-history frames"
                    )
        elif health_envelope is not None or stability_envelope is not None:
            effective_command = upstream_command
            if stability_envelope is not None:
                stability_output = stability_envelope.update(
                    policy_observation=_policy_tensor(observation),
                    upstream_command=upstream_command,
                )
                # Stability is downstream of packet health and is mathematically
                # only-attenuating, so the composition always selects the more
                # conservative forward command without another policy input.
                effective_command = stability_output.effective_command
            _apply_effective_command(base_env, effective_command)
            # Rewrite all five command frames before either policy inference or
            # observation-only FastBase diagnostics.  The actor therefore sees
            # exactly the command applied to the environment.
            observation = _rewrite_actor_command_history(
                observation, effective_command
            )
        first_episode_active_before = first_episode_active.clone()
        robot = base_env.scene["robot"]
        local_x_before = robot.data.root_pos_w[:, 0] - base_env.scene.env_origins[:, 0]
        if args_cli.state_dump_npz is not None:
            # Select states only after the contact-driven H->L->H latch has
            # entered HIGH_END.  A fixed x=3 m threshold silently discarded
            # most of the 1--3 s precursor window on the 2 m retention course
            # and was wrong for the 6 m LongDemo course.  The stage is used
            # only to build/reset the privileged state bank; it is never part
            # of the 1864-D actor observation.
            course_stage = getattr(base_env, "spatial_course_stage_buf", None)
            if not isinstance(course_stage, torch.Tensor) or tuple(course_stage.shape) != (
                base_env.num_envs,
            ):
                raise RuntimeError(
                    "state dump requires spatial_course_stage_buf with shape "
                    f"({base_env.num_envs},), got {getattr(course_stage, 'shape', None)}"
                )
            high_end_start_x = float(_runtime_course_geometry()["low_end_x_m"])
            state_keep = (
                first_episode_active_before
                & (course_stage == 2)
                & (local_x_before >= high_end_start_x)
                & (robot.data.root_lin_vel_b[:, 0] >= 0.55)
            )
            if bool(state_keep.any().item()):
                kept_state_ids = torch.nonzero(
                    state_keep, as_tuple=False
                ).squeeze(1)

                def _required_state(name: str, trailing_shape: tuple[int, ...]):
                    value = getattr(base_env, name, None)
                    expected = (base_env.num_envs, *trailing_shape)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                        raise RuntimeError(
                            f"state-dump field {name} must have shape {expected}, "
                            f"got {getattr(value, 'shape', None)}"
                        )
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(
                            f"state-dump field {name} contains NaN/Inf"
                        )
                    return value

                root_state = robot.data.root_state_w
                local_position = root_state[:, :3] - base_env.scene.env_origins
                recovery_state_pose.append(
                    torch.cat((local_position, root_state[:, 3:7]), dim=-1)[state_keep]
                    .detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_velocity.append(
                    root_state[:, 7:13][state_keep].detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_joint_pos.append(
                    robot.data.joint_pos[state_keep].detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_joint_vel.append(
                    robot.data.joint_vel[state_keep].detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_observation.append(
                    _policy_tensor(observation)[state_keep].detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_motion_initial_yaw.append(
                    _required_state("motion_feedback_initial_yaw", ())[state_keep]
                    .detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_heading_reference.append(
                    _required_state("straight_heading_reference_xy", (2,))[state_keep]
                    .detach().cpu().numpy().astype(np.float32)
                )
                track_origin_local = (
                    _required_state("straight_track_origin_xy", (2,))
                    - base_env.scene.env_origins[:, :2]
                )
                recovery_state_track_origin_local.append(
                    track_origin_local[state_keep]
                    .detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_track_lateral_axis.append(
                    _required_state("straight_track_lateral_axis", (2,))[state_keep]
                    .detach().cpu().numpy().astype(np.float32)
                )
                recovery_state_env_id.append(
                    kept_state_ids.detach().cpu().numpy().astype(np.int32)
                )
                recovery_state_rollout_step.append(
                    np.full(
                        kept_state_ids.numel(), step_index, dtype=np.int32
                    )
                )

                sensor = getattr(base_env, "_hall_foot_sensor", None)
                if sensor is None:
                    raise RuntimeError(
                        "state dump requires an initialized HallFootSensor"
                    )

                def _append_sensor(rows: list[np.ndarray], value: torch.Tensor, name: str):
                    if value.shape[0] != base_env.num_envs or not torch.isfinite(value).all():
                        raise RuntimeError(
                            f"Hall state-dump field {name} has invalid shape/data"
                        )
                    rows.append(
                        value[state_keep].detach().cpu().numpy().astype(np.float32)
                    )

                _append_sensor(
                    recovery_hall_local_deformation,
                    sensor.local_deformation,
                    "local_deformation",
                )
                _append_sensor(
                    recovery_hall_loading_history,
                    sensor.loading_history,
                    "loading_history",
                )
                _append_sensor(
                    recovery_hall_signal_filtered_absolute,
                    sensor.signal.filtered_absolute,
                    "signal.filtered_absolute",
                )
                _append_sensor(
                    recovery_hall_signal_processed,
                    sensor.signal.processed,
                    "signal.processed",
                )
                _append_sensor(
                    recovery_hall_signal_baseline,
                    sensor.signal.baseline,
                    "signal.baseline",
                )
                _append_sensor(
                    recovery_hall_signal_drift,
                    sensor.signal.drift,
                    "signal.drift",
                )
                _append_sensor(
                    recovery_hall_policy_history,
                    sensor._policy_history,
                    "policy_history",
                )
                _append_sensor(
                    recovery_hall_policy_gain,
                    sensor._policy_gain,
                    "policy_gain",
                )
                _append_sensor(
                    recovery_hall_policy_cross_axis,
                    sensor._policy_cross_axis,
                    "policy_cross_axis",
                )
                _append_sensor(
                    recovery_hall_policy_zero_residual,
                    sensor._policy_zero_residual,
                    "policy_zero_residual",
                )
                _append_sensor(
                    recovery_hall_policy_channel_keep,
                    sensor._policy_channel_keep,
                    "policy_channel_keep",
                )
                _append_sensor(
                    recovery_hall_policy_foot_keep,
                    sensor._policy_foot_keep,
                    "policy_foot_keep",
                )
                _append_sensor(
                    recovery_hall_policy_delay_steps,
                    sensor._policy_delay_steps,
                    "policy_delay_steps",
                )
                _append_sensor(
                    recovery_hall_reported_sample_period,
                    sensor._reported_sample_period,
                    "reported_sample_period",
                )
        # The transition state is deliberately used only for labeling the
        # offline teacher mixture; it is never passed to the actor.
        low_before = torch.as_tensor(
            [stage == 2 for stage in stages], device=base_env.device, dtype=torch.bool
        )
        pre_step_root = base_env.scene["robot"].data.root_pos_w
        cross_track_pre_step = (
            pre_step_root[:, 1] - base_env.scene.env_origins[:, 1]
        )
        # First-episode-only drift gate accumulation.  ``relative_heading`` is
        # the same deployable reset-relative heading used by the uniform
        # high-friction evaluator (policy column 1863); body vy is read from
        # the live articulation.  Post-fall managed resets never re-enter these
        # sums, and the maximum cross-track is likewise first-episode-only.
        policy_observation_drift = _policy_tensor(observation)
        drift_keep = first_episode_active_before & (
            step_index >= args_cli.drift_metric_warmup_steps
        )
        drift_vy = robot.data.root_lin_vel_b[:, 1]
        drift_heading = policy_observation_drift[:, 1863]
        drift_vy_sum += torch.where(
            drift_keep, drift_vy.detach(), torch.zeros_like(drift_vy)
        )
        drift_vy_sq_sum += torch.where(
            drift_keep, drift_vy.square().detach(), torch.zeros_like(drift_vy)
        )
        drift_heading_sq_sum += torch.where(
            drift_keep,
            drift_heading.square().detach(),
            torch.zeros_like(drift_heading),
        )
        drift_sample_count += drift_keep.long()
        drift_final_cross_track = torch.where(
            first_episode_active_before,
            cross_track_pre_step.detach(),
            drift_final_cross_track,
        )
        drift_geometry = _runtime_course_geometry()
        drift_local_x = (
            pre_step_root[:, 0] - base_env.scene.env_origins[:, 0]
        )
        drift_low = base_env.spatial_low_contact_buf.bool()
        drift_region_masks = {
            "high_start": (
                (drift_local_x < float(drift_geometry["low_start_x_m"]))
                & ~drift_low
            ),
            "low": drift_low,
            "high_end": (
                (drift_local_x >= float(drift_geometry["low_end_x_m"]))
                & ~drift_low
            ),
        }
        for name, mask in drift_region_masks.items():
            keep = drift_keep & mask
            drift_region_heading_sq[name] += torch.where(
                keep,
                drift_heading.square().detach(),
                torch.zeros_like(drift_heading),
            )
            drift_region_heading_sum[name] += torch.where(
                keep,
                drift_heading.detach(),
                torch.zeros_like(drift_heading),
            )
            drift_region_vy_sq[name] += torch.where(
                keep,
                drift_vy.square().detach(),
                torch.zeros_like(drift_vy),
            )
            drift_region_count[name] += keep.long()
        slip_keep = drift_keep & drift_region_masks["low"]
        if bool(slip_keep.any().item()):
            left_sensor = base_env.scene["left_hall_contact"]
            right_sensor = base_env.scene["right_hall_contact"]
            slip_result = static_ground_contact_point_tangential_speed(
                robot.data.body_com_pos_w[:, foot_body_ids, :],
                robot.data.body_com_lin_vel_w[:, foot_body_ids, :],
                robot.data.body_com_ang_vel_w[:, foot_body_ids, :],
                (left_sensor.data.contact_pos_w, right_sensor.data.contact_pos_w),
                (left_sensor.data.force_matrix_w, right_sensor.data.force_matrix_w),
                min_normal_force_n=5.0,
            )
            slip_asymmetry = (
                slip_result.speed_per_foot[:, 0]
                - slip_result.speed_per_foot[:, 1]
            )
            drift_low_slip_asym_sum += torch.where(
                slip_keep,
                slip_asymmetry.detach(),
                torch.zeros_like(slip_asymmetry),
            )
            drift_low_slip_count += slip_keep.long()
        max_abs_cross_track = torch.maximum(
            max_abs_cross_track,
            torch.where(
                first_episode_active_before,
                cross_track_pre_step.abs().detach(),
                torch.zeros_like(cross_track_pre_step),
            ),
        )
        recovery_output = None
        with torch.inference_mode():
            baseline_actions = policy(observation)
            capture_diagnostics = _read_fastbase_capture_diagnostics(policy, observation)
            if stability_recovery_blend is None:
                actions = baseline_actions
            else:
                if stability_output is None:
                    raise RuntimeError(
                        "stability recovery did not receive a stability state"
                    )
                recovery_output = stability_recovery_blend.update(
                    policy_observation=_policy_tensor(observation),
                    baseline_action=baseline_actions,
                    stability_state=stability_output.state,
                )
                actions = recovery_output.action
        low_action_keep = drift_keep & drift_region_masks["low"]
        if bool(low_action_keep.any().item()):
            asymmetry = torch.zeros(
                base_env.num_envs, device=base_env.device
            )
            for left_index, right_index, opposite in leg_action_pairs:
                delta = (
                    actions[:, left_index] + actions[:, right_index]
                    if opposite
                    else actions[:, left_index] - actions[:, right_index]
                )
                asymmetry += torch.square(delta)
            drift_low_action_asym_sum += torch.where(
                low_action_keep, asymmetry, torch.zeros_like(asymmetry)
            )
            drift_low_action_count += low_action_keep.long()
        if actions.shape != (base_env.num_envs, base_env.action_manager.total_action_dim):
            raise RuntimeError(
                "policy action shape mismatch: "
                f"got {tuple(actions.shape)}, expected "
                f"{(base_env.num_envs, base_env.action_manager.total_action_dim)}"
            )
        if not torch.isfinite(actions).all():
            nan_detected = True
            break
        if hall_command_governor is not None and bool(
            first_episode_active_before.any().item()
        ):
            assert hall_governor_risk is not None
            assert hall_governor_state is not None
            assert hall_governor_effective is not None
            assert hall_governor_valid is not None
            keep = first_episode_active_before.to(device=base_env.device)

            def _hall_governor_numpy(value, dtype):
                return value[keep].detach().cpu().numpy().astype(dtype)

            hall_governor_risk_rows.append(
                _hall_governor_numpy(hall_governor_risk, np.float32)
            )
            hall_governor_filtered_risk_rows.append(
                _hall_governor_numpy(
                    hall_command_governor.governor.probability_ema,
                    np.float32,
                )
            )
            hall_governor_state_rows.append(
                _hall_governor_numpy(hall_governor_state, np.int8)
            )
            hall_governor_requested_rows.append(
                _hall_governor_numpy(requested_command, np.float32)
            )
            hall_governor_upstream_rows.append(
                _hall_governor_numpy(upstream_command, np.float32)
            )
            hall_governor_effective_rows.append(
                _hall_governor_numpy(hall_governor_effective, np.float32)
            )
            hall_governor_valid_rows.append(
                _hall_governor_numpy(hall_governor_valid, np.bool_)
            )
            hall_governor_probing_rows.append(
                _hall_governor_numpy(
                    hall_command_governor.governor.probing, np.bool_
                )
            )
            hall_governor_prebrake_rows.append(
                _hall_governor_numpy(
                    hall_command_governor.governor.prebrake_active, np.bool_
                )
            )
            kept_env_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            hall_governor_step_rows.append(
                np.full(kept_count, step_index, dtype=np.int32)
            )
            hall_governor_time_rows.append(
                np.full(
                    kept_count,
                    step_index * float(base_env.step_dt),
                    dtype=np.float32,
                )
            )
            hall_governor_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
        if recovery_output is not None and bool(first_episode_active_before.any().item()):
            keep = first_episode_active_before.to(device=base_env.device)

            def _recovery_numpy(value, dtype):
                return value[keep].detach().cpu().numpy().astype(dtype)

            stability_recovery_gate_rows.append(
                _recovery_numpy(recovery_output.gate, np.float32)
            )
            stability_recovery_active_rows.append(
                _recovery_numpy(recovery_output.active, np.bool_)
            )
            stability_recovery_baseline_action_rows.append(
                _recovery_numpy(recovery_output.baseline_action, np.float32)
            )
            stability_recovery_expert_action_rows.append(
                _recovery_numpy(recovery_output.recovery_action, np.float32)
            )
            stability_recovery_output_action_rows.append(
                _recovery_numpy(recovery_output.action, np.float32)
            )
            kept_env_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            stability_recovery_step_rows.append(
                np.full(kept_count, step_index, dtype=np.int32)
            )
            stability_recovery_time_rows.append(
                np.full(
                    kept_count,
                    step_index * float(base_env.step_dt),
                    dtype=np.float32,
                )
            )
            stability_recovery_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
        if health_output is not None and bool(first_episode_active_before.any().item()):
            keep = first_episode_active_before.to(device=base_env.device)
            health_requested_rows.append(
                health_output.requested_command[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            health_target_rows.append(
                health_output.target_command[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            health_effective_rows.append(
                health_output.effective_command[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            health_state_rows.append(
                health_output.state[keep].detach().cpu().numpy().astype(np.int8)
            )
            health_valid_rows.append(
                health_output.valid[keep].detach().cpu().numpy().astype(np.bool_)
            )
            health_age_rows.append(
                health_output.packet_age_s[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            health_finite_rows.append(
                health_output.finite[keep].detach().cpu().numpy().astype(np.bool_)
            )
            health_foot_healthy_rows.append(
                health_output.foot_healthy[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.bool_)
            )
            health_recovery_timer_rows.append(
                health_output.recovery_timer_s[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            health_intervened_rows.append(
                health_output.intervened[keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.bool_)
            )
            kept_env_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            health_step_rows.append(np.full(kept_count, step_index, dtype=np.int32))
            health_time_rows.append(
                np.full(
                    kept_count,
                    step_index * float(base_env.step_dt),
                    dtype=np.float32,
                )
            )
            health_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
        if stability_output is not None and bool(first_episode_active_before.any().item()):
            keep = first_episode_active_before.to(device=base_env.device)

            def _stability_numpy(value, dtype):
                return value[keep].detach().cpu().numpy().astype(dtype)

            stability_upstream_rows.append(
                _stability_numpy(stability_output.upstream_command, np.float32)
            )
            stability_effective_rows.append(
                _stability_numpy(stability_output.effective_command, np.float32)
            )
            stability_state_rows.append(
                _stability_numpy(stability_output.state, np.int8)
            )
            stability_reason_rows.append(
                _stability_numpy(stability_output.reason_mask, np.int16)
            )
            stability_intervened_rows.append(
                _stability_numpy(stability_output.intervened, np.bool_)
            )
            stability_command_mean_rows.append(
                _stability_numpy(stability_output.command_mean, np.float32)
            )
            stability_heading_command_mean_rows.append(
                _stability_numpy(
                    stability_output.heading_command_mean, np.float32
                )
            )
            stability_heading_enabled_rows.append(
                _stability_numpy(stability_output.heading_enabled, np.bool_)
            )
            stability_heading_signed_rows.append(
                _stability_numpy(stability_output.heading_error, np.float32)
            )
            stability_heading_rows.append(
                _stability_numpy(stability_output.heading_error_abs, np.float32)
            )
            stability_heading_correction_active_rows.append(
                _stability_numpy(
                    stability_output.heading_correction_active, np.bool_
                )
            )
            stability_heading_correction_yaw_rows.append(
                _stability_numpy(
                    stability_output.heading_correction_yaw, np.float32
                )
            )
            stability_omega_rows.append(
                _stability_numpy(stability_output.omega_xy, np.float32)
            )
            stability_tilt_rows.append(
                _stability_numpy(stability_output.tilt, np.float32)
            )
            stability_previous_action_norm_rows.append(
                _stability_numpy(stability_output.previous_action_norm, np.float32)
            )
            stability_current_action_norm_rows.append(
                _stability_numpy(stability_output.current_action_norm, np.float32)
            )
            stability_action_slew_norm_rows.append(
                _stability_numpy(stability_output.action_slew_norm, np.float32)
            )
            stability_action_saturation_count_rows.append(
                _stability_numpy(stability_output.action_saturation_count, np.int8)
            )
            stability_warn_count_rows.append(
                _stability_numpy(stability_output.warn_count, np.int16)
            )
            stability_limit_count_rows.append(
                _stability_numpy(stability_output.limit_count, np.int16)
            )
            stability_hard_limit_count_rows.append(
                _stability_numpy(stability_output.hard_limit_count, np.int16)
            )
            stability_recovery_count_rows.append(
                _stability_numpy(stability_output.recovery_count, np.int16)
            )
            kept_env_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            stability_step_rows.append(
                np.full(kept_count, step_index, dtype=np.int32)
            )
            stability_time_rows.append(
                np.full(
                    kept_count,
                    step_index * float(base_env.step_dt),
                    dtype=np.float32,
                )
            )
            stability_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
        if capture_diagnostics is not None:
            capture_interface_seen = True
            keep = first_episode_active_before.to(device=base_env.device)
            diagnostic_stage = _course_stage_from_local_x(local_x_before)
            capture_raw_probability_rows.append(
                capture_diagnostics["raw_capture_probability"][keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            capture_probability_rows.append(
                capture_diagnostics["capture_probability"][keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            capture_effective_gate_rows.append(
                capture_diagnostics["effective_gate"][keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            capture_delta_l2_rows.append(
                capture_diagnostics["delta_l2"][keep]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if "stability_authority" in capture_diagnostics:
                stability_authority_rows.append(
                    capture_diagnostics["stability_authority"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                stability_delta_l2_rows.append(
                    capture_diagnostics["stability_delta_l2"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                stability_delta_abs_max_rows.append(
                    capture_diagnostics["stability_delta_abs_max"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            capture_stage_rows.append(
                diagnostic_stage[keep].detach().cpu().numpy().astype(np.uint8)
            )
            kept_env_ids = torch.nonzero(keep, as_tuple=False).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            capture_step_rows.append(
                np.full(kept_count, step_index, dtype=np.int32)
            )
            capture_time_rows.append(
                np.full(kept_count, step_index * float(base_env.step_dt), dtype=np.float32)
            )
            capture_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
        if args_cli.failure_analysis_npz is not None and bool(
            first_episode_active_before.any().item()
        ):
            # All rows below describe the exact pre-action state consumed by
            # the actor.  Fall/done labels are appended after env.step using
            # this same frozen mask, so a row at t is causally aligned with
            # the transition caused by action[t].  Nothing here is fed back
            # into the policy or environment.
            failure_keep_for_step = first_episode_active_before.to(
                device=base_env.device
            )
            policy_observation = _policy_tensor(observation)
            kept_env_ids = torch.nonzero(
                failure_keep_for_step, as_tuple=False
            ).squeeze(1)
            kept_count = int(kept_env_ids.numel())
            root_state_pre = robot.data.root_state_w
            root_local_position = (
                root_state_pre[:, :3] - base_env.scene.env_origins
            )
            root_pose_local = torch.cat(
                (root_local_position, root_state_pre[:, 3:7]), dim=-1
            )
            geometric_stage = _course_stage_from_local_x(local_x_before)
            failure_obs_rows.append(
                policy_observation[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_action_rows.append(
                actions[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_env_rows.append(
                kept_env_ids.detach().cpu().numpy().astype(np.int32)
            )
            failure_step_rows.append(
                np.full(kept_count, step_index, dtype=np.int32)
            )
            failure_time_rows.append(
                np.full(
                    kept_count,
                    step_index * float(base_env.step_dt),
                    dtype=np.float32,
                )
            )
            failure_local_x_rows.append(
                local_x_before[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_root_pose_rows.append(
                root_pose_local[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_root_lin_vel_b_rows.append(
                robot.data.root_lin_vel_b[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_root_ang_vel_b_rows.append(
                robot.data.root_ang_vel_b[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_joint_pos_rows.append(
                robot.data.joint_pos[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_joint_vel_rows.append(
                robot.data.joint_vel[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.float32)
            )
            failure_contact_rows.append(
                gait_prev_contact[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.bool_)
            )
            failure_stage_rows.append(
                geometric_stage[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.uint8)
            )
            if capture_diagnostics is None:
                zero = np.zeros(kept_count, dtype=np.float32)
                failure_gate_rows.append(zero.copy())
                failure_capture_delta_rows.append(zero.copy())
                failure_stability_authority_rows.append(zero.copy())
                failure_stability_delta_rows.append(zero.copy())
            else:
                failure_gate_rows.append(
                    capture_diagnostics["effective_gate"][failure_keep_for_step]
                    .detach().cpu().numpy().astype(np.float32)
                )
                failure_capture_delta_rows.append(
                    capture_diagnostics["delta_l2"][failure_keep_for_step]
                    .detach().cpu().numpy().astype(np.float32)
                )
                if "stability_authority" in capture_diagnostics:
                    failure_stability_authority_rows.append(
                        capture_diagnostics["stability_authority"]
                        [failure_keep_for_step]
                        .detach().cpu().numpy().astype(np.float32)
                    )
                    failure_stability_delta_rows.append(
                        capture_diagnostics["stability_delta_l2"]
                        [failure_keep_for_step]
                        .detach().cpu().numpy().astype(np.float32)
                    )
                else:
                    zero = np.zeros(kept_count, dtype=np.float32)
                    failure_stability_authority_rows.append(zero.copy())
                    failure_stability_delta_rows.append(zero.copy())
        if args_cli.dataset_npz is not None and bool(
            first_episode_active_before.any().item()
        ):
            keep = first_episode_active_before.to(device=base_env.device)
            policy_observation = _policy_tensor(observation)
            dataset_obs.append(policy_observation[keep].detach().cpu().numpy().astype(np.float32))
            dataset_actions.append(actions[keep].detach().cpu().numpy().astype(np.float32))
            # stage==2 is the latched LOW segment; high-start and high-end use
            # the fast teacher in the subsequent distillation pass.
            dataset_low.append(low_before[keep].detach().cpu().numpy().astype(np.bool_))
        observation, _, dones, extras = env.step(actions)
        steps_run += 1
        _policy_tensor(observation)
        course_success = (
            base_env.termination_manager.get_term("course_success").bool()
            & first_episode_active_before
        )
        course_success_events += int(course_success.sum().item())
        completed |= course_success
        robot = base_env.scene["robot"]
        local_x = robot.data.root_pos_w[:, 0] - base_env.scene.env_origins[:, 0]
        cross_track = robot.data.root_pos_w[:, 1] - base_env.scene.env_origins[:, 1]
        falls = _fall_mask(dones, extras) & first_episode_active_before
        new_falls = falls & (first_fall_step < 0)
        first_fall_step[new_falls] = int(step_index)
        # Envs that terminated this step have already been reset, so the
        # current cross-track is the new episode's spawn position.  Record the
        # pre-step cross-track from the frame that actually terminated instead.
        pre_step_cross = cross_track_pre_step[new_falls]
        if new_falls.any().item():
            first_fall_cross_track[new_falls] = pre_step_cross.detach()
        vx = robot.data.root_lin_vel_b[:, 0]
        low = base_env.spatial_low_contact_buf.bool()
        contact_patch, in_contact = _contact_patch_state(base_env)
        falls = _fall_mask(dones, extras) & first_episode_active_before
        new_falls = falls & (first_fall_step < 0)
        first_fall_step[new_falls] = int(step_index)
        first_fall_cross_track[new_falls] = cross_track[new_falls].detach()
        falls_total += int(falls.sum().item())
        fallen |= falls
        first_episode_terminal = dones.bool() & first_episode_active_before
        if failure_keep_for_step is not None:
            failure_fall_rows.append(
                falls[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.bool_)
            )
            failure_done_rows.append(
                first_episode_terminal[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.bool_)
            )
            failure_timeout_rows.append(
                (first_episode_terminal & ~falls)[failure_keep_for_step]
                .detach().cpu().numpy().astype(np.bool_)
            )
        first_episode_active &= ~first_episode_terminal
        if health_envelope is not None and bool(dones.bool().any().item()):
            health_envelope.reset(
                torch.nonzero(dones.bool(), as_tuple=False).squeeze(1)
            )
        if hall_command_governor is not None and bool(dones.bool().any().item()):
            hall_command_governor.reset(
                torch.nonzero(dones.bool(), as_tuple=False).squeeze(1)
            )
        if stability_envelope is not None and bool(dones.bool().any().item()):
            stability_envelope.reset(
                torch.nonzero(dones.bool(), as_tuple=False).squeeze(1)
            )
        if stability_recovery_blend is not None and bool(dones.bool().any().item()):
            stability_recovery_blend.reset(
                torch.nonzero(dones.bool(), as_tuple=False).squeeze(1)
            )
        high_end_contact = in_contact & (contact_patch == 2)

        all_vx.append(vx.detach().cpu().tolist())
        all_low.append(low.detach().cpu().tolist())
        all_high_end_contact.append(high_end_contact.detach().cpu().tolist())
        all_falls.append(falls.detach().cpu().tolist())
        all_dones.append(first_episode_terminal.detach().cpu().tolist())

        for env_id in range(base_env.num_envs):
            if not bool(first_episode_active_before[env_id].item()):
                continue
            if bool(course_success[env_id].item()):
                # ManagerBasedRLEnv has already reset this environment before
                # returning from step(), so local_x below is the new episode's
                # spawn position.  The timeout term itself is the authoritative
                # proof that the preceding episode completed H--L--H.
                stages[env_id] = COMPLETE
                continue
            stages[env_id] = advance_high_low_high_stage(
                stages[env_id],
                SpatialTransitionSample(
                    local_x=float(local_x[env_id].item()),
                    low_contact=bool(low[env_id].item()),
                    done=bool(dones[env_id].item()),
                    high_start_contact=bool(
                        (in_contact[env_id] & (contact_patch[env_id] == 0)).item()
                    ),
                    high_end_contact=bool(
                        (in_contact[env_id] & (contact_patch[env_id] == 2)).item()
                    ),
                ),
            )
            if stages[env_id] == COMPLETE:
                completed[env_id] = True

        geometry = _runtime_course_geometry()
        low_start_x = float(geometry["low_start_x_m"])
        low_end_x = float(geometry["low_end_x_m"])
        masks = {
            # Returned state is post-reset for done environments; exclude that
            # frame so a successful truncation does not pollute high-start data.
            "high_start": (
                (local_x < low_start_x)
                & ~low
                & ~dones.bool()
                & first_episode_active_before
            ),
            "low": low & ~dones.bool() & first_episode_active_before,
            "high_end": (
                (local_x >= low_end_x)
                & ~low
                & ~dones.bool()
                & first_episode_active_before
            ),
        }
        for name, mask in masks.items():
            region_speed_sum[name] += float(vx[mask].sum().item())
            region_speed_count[name] += int(mask.sum().item())
            per_env_region_speed_sum[name] += torch.where(
                mask, vx, torch.zeros_like(vx)
            )
            per_env_region_speed_count[name] += mask.to(dtype=torch.long)

        gait_contact = _gait_contact_state()
        next_gait_region = torch.full_like(gait_region_id, -1)
        for region_index, name in enumerate(("high_start", "low", "high_end")):
            next_gait_region[masks[name]] = region_index
        region_changed = next_gait_region != gait_region_id
        gait_last_touchdown_forward[region_changed] = float("nan")
        gait_steps_since_touchdown[region_changed] = 10_000
        gait_air_steps[region_changed] = 0
        gait_region_id = next_gait_region

        gait_touchdown = (
            gait_contact
            & ~gait_prev_contact
            & (gait_air_steps >= gait_minimum_air_steps)
            & (gait_steps_since_touchdown >= gait_minimum_touchdown_gap_steps)
            & first_episode_active_before[:, None]
            & ~dones.bool()[:, None]
        )
        forward_at_touchdown = local_x[:, None].expand(-1, 2)
        gait_stride = torch.abs(
            forward_at_touchdown - gait_last_touchdown_forward
        )
        for name in ("high_start", "low", "high_end"):
            region_mask = masks[name]
            region_touchdown = gait_touchdown & region_mask[:, None]
            valid_stride = region_touchdown & torch.isfinite(
                gait_last_touchdown_forward
            )
            gait_region_data[name]["failure_free_exposure_s"] += (
                float(region_mask.sum().item()) * float(base_env.step_dt)
            )
            gait_region_data[name]["touchdowns"] += int(
                region_touchdown.sum().item()
            )
            gait_region_data[name]["stride_sum_m"] += float(
                gait_stride[valid_stride].sum().item()
            )
            gait_region_data[name]["stride_count"] += int(
                valid_stride.sum().item()
            )
        gait_last_touchdown_forward[gait_touchdown] = forward_at_touchdown[
            gait_touchdown
        ]
        gait_steps_since_touchdown[gait_touchdown] = 0
        gait_steps_since_touchdown += 1
        gait_air_steps = torch.where(
            gait_contact,
            torch.zeros_like(gait_air_steps),
            gait_air_steps + 1,
        )
        gait_prev_contact = gait_contact

        hall_sensor = base_env._hall_foot_sensor
        hall = hall_sensor.get_filtered_data()
        valid = hall_sensor.get_policy_valid_mask()
        if not torch.isfinite(hall).all() or not torch.isfinite(robot.data.root_state_w).all():
            nan_detected = True
            break
        trace_id = args_cli.trace_env_id
        # A done frame already contains the managed reset state.  Keep the
        # saved trace strictly within the selected environment's first episode.
        trace_frame_valid = bool(
            (first_episode_active_before[trace_id] & ~dones.bool()[trace_id]).item()
        )
        if trace_frame_valid:
            trace_x.append(float(local_x[trace_id].item()))
            trace_vx.append(float(vx[trace_id].item()))
            trace_low.append(bool(low[trace_id].item()))
            trace_contact_patch.append(
                int(contact_patch[trace_id].item())
                if bool(in_contact[trace_id].item())
                else -1
            )
            label_trace.append(bool(low[trace_id].item()))
            trace_hall.append(hall[trace_id].detach().cpu().numpy().copy())
            trace_valid.append(valid[trace_id].detach().cpu().numpy().copy())

    means = {
        name: (
            region_speed_sum[name] / region_speed_count[name]
            if region_speed_count[name]
            else float("nan")
        )
        for name in region_speed_sum
    }
    gait_adaptation: dict[str, object] = {
        "definition": "first-episode-touchdown-cadence-stride-v1",
        "contact_threshold_n": 5.0,
        "minimum_air_time_s": gait_minimum_air_steps * float(base_env.step_dt),
        "minimum_touchdown_gap_s": (
            gait_minimum_touchdown_gap_steps * float(base_env.step_dt)
        ),
        "command_m_s": float(args_cli.command),
        "command_is_identical_across_regions": True,
        "mechanism_is_diagnostic_not_prescribed": True,
        "regions": {},
    }
    high_start_cadence = float("nan")
    high_start_step = float("nan")
    high_start_stride = float("nan")
    high_start_speed = abs(float(means["high_start"]))
    for name in ("high_start", "low", "high_end"):
        data = gait_region_data[name]
        exposure = float(data["failure_free_exposure_s"])
        touchdowns = int(data["touchdowns"])
        stride_count = int(data["stride_count"])
        cadence = touchdowns / exposure if exposure > 0.0 else float("nan")
        stride = (
            float(data["stride_sum_m"]) / stride_count
            if stride_count > 0
            else float("nan")
        )
        step_length = 0.5 * stride
        speed = abs(float(means[name]))
        if name == "high_start":
            high_start_cadence = cadence
            high_start_step = step_length
            high_start_stride = stride

        def _finite_ratio(value: float, reference: float) -> float:
            if (
                not math.isfinite(value)
                or not math.isfinite(reference)
                or abs(reference) <= 1.0e-8
            ):
                return float("nan")
            return value / reference

        gait_adaptation["regions"][name] = {
            "failure_free_exposure_s": exposure,
            "touchdowns": touchdowns,
            "step_frequency_hz": cadence,
            "mean_stride_length_m": stride,
            "mean_step_length_m": step_length,
            "stride_samples": stride_count,
            "mean_body_vx_m_s": speed,
            "cadence_times_step_m_s": cadence * step_length,
            "cadence_vs_high_start_ratio": _finite_ratio(
                cadence, high_start_cadence
            ),
            "step_length_vs_high_start_ratio": _finite_ratio(
                step_length, high_start_step
            ),
            "stride_length_vs_high_start_ratio": _finite_ratio(
                stride, high_start_stride
            ),
            "vx_vs_high_start_ratio": _finite_ratio(speed, high_start_speed),
        }
    gait_adaptation["high_end_recovery"] = {
        "cadence_ratio": gait_adaptation["regions"]["high_end"][
            "cadence_vs_high_start_ratio"
        ],
        "step_length_ratio": gait_adaptation["regions"]["high_end"][
            "step_length_vs_high_start_ratio"
        ],
        "stride_length_ratio": gait_adaptation["regions"]["high_end"][
            "stride_length_vs_high_start_ratio"
        ],
        "vx_ratio": gait_adaptation["regions"]["high_end"][
            "vx_vs_high_start_ratio"
        ],
    }
    per_env_region_speed: dict[str, list[float]] = {}
    for name in region_speed_sum:
        sums = per_env_region_speed_sum[name]
        counts = per_env_region_speed_count[name]
        values = torch.where(
            counts > 0,
            sums / counts.clamp_min(1),
            torch.full_like(sums, float("nan")),
        )
        per_env_region_speed[name] = values.detach().cpu().tolist()
    response = analyze_transition_response(
        body_vx=all_vx,
        low_contact=all_low,
        high_end_contact=all_high_end_contact,
        falls=all_falls,
        dones=all_dones,
        step_dt_s=float(base_env.step_dt),
        low_speed_target_m_s=_runtime_low_speed_target(),
        high_recovery_speed_m_s=float(args_cli.high_recovery_speed),
        recovery_stable_steps=args_cli.recovery_stable_steps,
    )
    first_fall_cross = first_fall_cross_track.detach().cpu().tolist()
    max_cross = max_abs_cross_track.detach().cpu().tolist()
    response["first_fall_cross_track_m"] = [
        None if isinstance(value, float) and math.isnan(value) else float(value)
        for value in first_fall_cross
    ]
    response["max_abs_cross_track_m"] = [float(value) for value in max_cross]
    response["course_half_width_m"] = (
        None
        if args_cli.floor_width_m is None
        else float(args_cli.floor_width_m) / 2.0
    )
    if args_cli.floor_width_m is not None:
        half_width = float(args_cli.floor_width_m) / 2.0
        edge_exit = sum(
            1
            for value in first_fall_cross
            if value is not None and abs(value) >= 0.95 * half_width
        )
        response["edge_exit_fall_envs"] = edge_exit
        response["dynamic_fall_envs"] = int(fallen.sum().item()) - edge_exit
    else:
        response["edge_exit_fall_envs"] = None
        response["dynamic_fall_envs"] = None
    drift_counts = drift_sample_count.detach().cpu()
    drift_vy_mean = drift_vy_sum.detach().cpu() / drift_counts.clamp_min(1)
    drift_vy_rms = torch.sqrt(
        drift_vy_sq_sum.detach().cpu() / drift_counts.clamp_min(1)
    )
    drift_heading_rms = torch.sqrt(
        drift_heading_sq_sum.detach().cpu() / drift_counts.clamp_min(1)
    )
    drift_vy_list = [float(value) for value in drift_vy_rms]
    drift_vy_mean_list = [float(value) for value in drift_vy_mean]
    drift_heading_list = [float(value) for value in drift_heading_rms]
    drift_max_cross_list = [float(value) for value in max_cross]
    drift_final_list = [
        None if isinstance(value, float) and math.isnan(value) else float(value)
        for value in drift_final_cross_track.detach().cpu()
    ]
    drift_final_valid = [value for value in drift_final_list if value is not None]
    vy_rms_aggregate = float(np.sqrt(np.mean(np.square(drift_vy_list))))
    mean_vy_aggregate = float(np.mean(drift_vy_mean_list))
    heading_rms_aggregate = float(
        np.sqrt(np.mean(np.square(drift_heading_list)))
    )
    mean_final_cross = (
        float(np.mean(drift_final_valid)) if drift_final_valid else None
    )
    max_abs_final_cross = (
        float(np.max(np.abs(drift_final_valid))) if drift_final_valid else None
    )
    drift_region_rms: dict[str, dict[str, float]] = {}
    for name in ("high_start", "low", "high_end"):
        counts = drift_region_count[name].detach().cpu()
        heading_rms = torch.sqrt(
            drift_region_heading_sq[name].detach().cpu() / counts.clamp_min(1)
        )
        heading_mean = (
            drift_region_heading_sum[name].detach().cpu() / counts.clamp_min(1)
        )
        vy_rms = torch.sqrt(
            drift_region_vy_sq[name].detach().cpu() / counts.clamp_min(1)
        )
        drift_region_rms[name] = {
            "heading_rms_rad": float(
                np.sqrt(np.mean(np.square([float(value) for value in heading_rms])))
            ),
            "heading_mean_rad": float(
                np.mean([float(value) for value in heading_mean])
            ),
            "vy_rms_m_s": float(
                np.sqrt(np.mean(np.square([float(value) for value in vy_rms])))
            ),
            "sampled_envs": int((counts > 0).sum().item()),
        }
    if int(drift_low_action_count.sum().item()) > 0:
        drift_region_rms["low"]["action_asymmetry_mean_rad2"] = float(
            drift_low_action_asym_sum.sum().item()
            / drift_low_action_count.sum().item()
        )
    if int(drift_low_slip_count.sum().item()) > 0:
        drift_region_rms["low"]["foot_slip_asymmetry_mean_m_s"] = float(
            drift_low_slip_asym_sum.sum().item()
            / drift_low_slip_count.sum().item()
        )
    edge_exit_count = response.get("edge_exit_fall_envs")
    dynamic_fall_count = response.get("dynamic_fall_envs")
    drift_thresholds = {
        "maximum_body_vy_rms_m_s": 0.25,
        "maximum_heading_rms_rad": 0.25,
        "edge_margin_fraction": 0.95,
    }
    response["drift_gate"] = {
        "definition": "first-episode-deployable-drift-v1",
        "warmup_steps": int(args_cli.drift_metric_warmup_steps),
        "sampled_envs": int((drift_counts > 0).sum().item()),
        "per_env_body_vy_rms_m_s": drift_vy_list,
        "per_env_mean_body_vy_m_s": drift_vy_mean_list,
        "per_env_heading_rms_rad": drift_heading_list,
        "per_env_max_abs_cross_track_m": drift_max_cross_list,
        "per_env_final_cross_track_m": drift_final_list,
        "aggregate_body_vy_rms_m_s": vy_rms_aggregate,
        "aggregate_mean_body_vy_m_s": mean_vy_aggregate,
        "aggregate_heading_rms_rad": heading_rms_aggregate,
        "lateral_bias_fraction": (
            abs(mean_vy_aggregate) / vy_rms_aggregate
            if vy_rms_aggregate > 1.0e-9
            else None
        ),
        "max_abs_cross_track_m": float(np.max(drift_max_cross_list)),
        "p95_abs_cross_track_m": float(np.percentile(drift_max_cross_list, 95)),
        "mean_final_cross_track_m": mean_final_cross,
        "max_abs_final_cross_track_m": max_abs_final_cross,
        "per_region": drift_region_rms,
        "course_half_width_m": response["course_half_width_m"],
        "edge_exit_fall_envs": edge_exit_count,
        "dynamic_fall_envs": dynamic_fall_count,
        "thresholds": drift_thresholds,
        "gates": {
            "body_vy_rms": vy_rms_aggregate <= drift_thresholds[
                "maximum_body_vy_rms_m_s"
            ],
            "heading_rms": heading_rms_aggregate
            <= drift_thresholds["maximum_heading_rms_rad"],
            "zero_dynamic_falls": (
                True if dynamic_fall_count is None else dynamic_fall_count == 0
            ),
        },
        "pass": (
            vy_rms_aggregate <= drift_thresholds["maximum_body_vy_rms_m_s"]
            and heading_rms_aggregate <= drift_thresholds["maximum_heading_rms_rad"]
            and (dynamic_fall_count is None or dynamic_fall_count == 0)
        ),
    }
    hall_health_performance = _summarize_hall_health_performance(
        initial_hall_fault_state,
        completed=completed,
        fallen=fallen,
        per_env_region_speed=per_env_region_speed,
        response=response,
    )
    capture_diagnostic_arrays: dict[str, np.ndarray] = {}
    capture_diagnostic_summary = None
    stability_residual_diagnostic_summary = None
    if capture_interface_seen:
        capture_diagnostic_arrays = {
            "fastbase_raw_capture_probability": np.concatenate(
                capture_raw_probability_rows, axis=0
            ),
            "fastbase_capture_probability": np.concatenate(
                capture_probability_rows, axis=0
            ),
            "fastbase_effective_gate": np.concatenate(
                capture_effective_gate_rows, axis=0
            ),
            "fastbase_capture_delta_l2": np.concatenate(
                capture_delta_l2_rows, axis=0
            ),
            "fastbase_course_stage": np.concatenate(capture_stage_rows, axis=0),
            "fastbase_rollout_step": np.concatenate(capture_step_rows, axis=0),
            "fastbase_time_s": np.concatenate(capture_time_rows, axis=0),
            "fastbase_env_id": np.concatenate(capture_env_rows, axis=0),
        }
        capture_diagnostic_summary = summarize_fastbase_capture_diagnostics(
            raw_capture_probability=capture_diagnostic_arrays[
                "fastbase_raw_capture_probability"
            ],
            capture_probability=capture_diagnostic_arrays[
                "fastbase_capture_probability"
            ],
            effective_gate=capture_diagnostic_arrays["fastbase_effective_gate"],
            delta_l2=capture_diagnostic_arrays["fastbase_capture_delta_l2"],
            course_stage=capture_diagnostic_arrays["fastbase_course_stage"],
            rollout_step=capture_diagnostic_arrays["fastbase_rollout_step"],
            env_id=capture_diagnostic_arrays["fastbase_env_id"],
            step_dt_s=float(base_env.step_dt),
        )
        if stability_authority_rows:
            capture_diagnostic_arrays.update(
                {
                    "fastbase_stability_authority": np.concatenate(
                        stability_authority_rows, axis=0
                    ),
                    "fastbase_stability_delta_l2": np.concatenate(
                        stability_delta_l2_rows, axis=0
                    ),
                    "fastbase_stability_delta_abs_max": np.concatenate(
                        stability_delta_abs_max_rows, axis=0
                    ),
                }
            )

            def _finite_numpy_stats(values: np.ndarray) -> dict[str, object]:
                flat = np.asarray(values, dtype=np.float64).reshape(-1)
                if flat.size == 0:
                    return {
                        "count": 0,
                        "mean": None,
                        "median": None,
                        "p95": None,
                        "max": None,
                    }
                if not np.isfinite(flat).all():
                    raise FloatingPointError(
                        "stability residual diagnostics are non-finite"
                    )
                return {
                    "count": int(flat.size),
                    "mean": float(flat.mean()),
                    "median": float(np.median(flat)),
                    "p95": float(np.percentile(flat, 95.0)),
                    "max": float(flat.max()),
                }

            stability_residual_diagnostic_summary = {
                "definition": "deployable-proprio-stability-residual-v1",
                "actor_observation_dim": 1864,
                "inputs": "proprio_history_0_480_plus_motion_feedback_1862_1864",
                "uses_force_contact_mu_or_stage": False,
                "overall": {
                    "authority": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_authority"
                        ]
                    ),
                    "delta_l2": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_delta_l2"
                        ]
                    ),
                    "delta_abs_max": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_delta_abs_max"
                        ]
                    ),
                },
                "by_stage": {},
            }
            stages_array = capture_diagnostic_arrays["fastbase_course_stage"]
            for stage_id, stage_name in enumerate(
                ("HIGH_START", "LOW", "HIGH_END")
            ):
                stage_mask = stages_array == stage_id
                stability_residual_diagnostic_summary["by_stage"][stage_name] = {
                    "authority": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_authority"
                        ][stage_mask]
                    ),
                    "delta_l2": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_delta_l2"
                        ][stage_mask]
                    ),
                    "delta_abs_max": _finite_numpy_stats(
                        capture_diagnostic_arrays[
                            "fastbase_stability_delta_abs_max"
                        ][stage_mask]
                    ),
                }
    health_diagnostic_arrays: dict[str, np.ndarray] = {}
    health_diagnostic_summary = None
    if health_envelope is not None:
        health_diagnostic_arrays = {
            "health_requested_command": (
                np.concatenate(health_requested_rows, axis=0)
                if health_requested_rows
                else np.empty((0, 3), dtype=np.float32)
            ),
            "health_target_command": (
                np.concatenate(health_target_rows, axis=0)
                if health_target_rows
                else np.empty((0, 3), dtype=np.float32)
            ),
            "health_effective_command": (
                np.concatenate(health_effective_rows, axis=0)
                if health_effective_rows
                else np.empty((0, 3), dtype=np.float32)
            ),
            "health_state": (
                np.concatenate(health_state_rows, axis=0)
                if health_state_rows
                else np.empty((0,), dtype=np.int8)
            ),
            "health_valid": (
                np.concatenate(health_valid_rows, axis=0)
                if health_valid_rows
                else np.empty((0, 2), dtype=np.bool_)
            ),
            "health_age_s": (
                np.concatenate(health_age_rows, axis=0)
                if health_age_rows
                else np.empty((0, 2), dtype=np.float32)
            ),
            "health_finite": (
                np.concatenate(health_finite_rows, axis=0)
                if health_finite_rows
                else np.empty((0, 2), dtype=np.bool_)
            ),
            "health_foot_healthy": (
                np.concatenate(health_foot_healthy_rows, axis=0)
                if health_foot_healthy_rows
                else np.empty((0, 2), dtype=np.bool_)
            ),
            "health_recovery_timer_s": (
                np.concatenate(health_recovery_timer_rows, axis=0)
                if health_recovery_timer_rows
                else np.empty((0,), dtype=np.float32)
            ),
            "health_intervened": (
                np.concatenate(health_intervened_rows, axis=0)
                if health_intervened_rows
                else np.empty((0,), dtype=np.bool_)
            ),
            "health_rollout_step": (
                np.concatenate(health_step_rows, axis=0)
                if health_step_rows
                else np.empty((0,), dtype=np.int32)
            ),
            "health_time_s": (
                np.concatenate(health_time_rows, axis=0)
                if health_time_rows
                else np.empty((0,), dtype=np.float32)
            ),
            "health_env_id": (
                np.concatenate(health_env_rows, axis=0)
                if health_env_rows
                else np.empty((0,), dtype=np.int32)
            ),
        }
        health_diagnostic_summary = summarize_health_envelope_trace(
            requested_command=health_diagnostic_arrays["health_requested_command"],
            effective_command=health_diagnostic_arrays["health_effective_command"],
            state=health_diagnostic_arrays["health_state"],
            valid=health_diagnostic_arrays["health_valid"],
            age_s=health_diagnostic_arrays["health_age_s"],
            finite=health_diagnostic_arrays["health_finite"],
            foot_healthy=health_diagnostic_arrays["health_foot_healthy"],
            intervened=health_diagnostic_arrays["health_intervened"],
        )
        health_diagnostic_summary["enabled"] = True
        health_diagnostic_summary["config"] = {
            "single_foot_speed_cap_m_s": float(
                health_envelope.cfg.single_foot_speed_cap
            ),
            "max_packet_age_s": float(health_envelope.cfg.max_packet_age_s),
            "linear_accel_rate_m_s2": float(
                health_envelope.cfg.linear_accel_rate
            ),
            "linear_decel_rate_m_s2": float(
                health_envelope.cfg.linear_decel_rate
            ),
            "recovery_hold_s": float(health_envelope.cfg.recovery_hold_s),
        }
    hall_governor_diagnostic_arrays: dict[str, np.ndarray] = {}
    hall_governor_diagnostic_summary = None
    if hall_command_governor is not None:
        def _hall_governor_concat(rows, empty_shape, dtype):
            return (
                np.concatenate(rows, axis=0)
                if rows
                else np.empty(empty_shape, dtype=dtype)
            )

        hall_governor_diagnostic_arrays = {
            "hall_governor_risk_probability": _hall_governor_concat(
                hall_governor_risk_rows, (0,), np.float32
            ),
            "hall_governor_filtered_probability": _hall_governor_concat(
                hall_governor_filtered_risk_rows, (0,), np.float32
            ),
            "hall_governor_state": _hall_governor_concat(
                hall_governor_state_rows, (0,), np.int8
            ),
            "hall_governor_requested_command": _hall_governor_concat(
                hall_governor_requested_rows, (0, 3), np.float32
            ),
            "hall_governor_health_bounded_command": _hall_governor_concat(
                hall_governor_upstream_rows, (0, 3), np.float32
            ),
            "hall_governor_effective_command": _hall_governor_concat(
                hall_governor_effective_rows, (0, 3), np.float32
            ),
            "hall_governor_valid": _hall_governor_concat(
                hall_governor_valid_rows, (0,), np.bool_
            ),
            "hall_governor_probing": _hall_governor_concat(
                hall_governor_probing_rows, (0,), np.bool_
            ),
            "hall_governor_prebrake": _hall_governor_concat(
                hall_governor_prebrake_rows, (0,), np.bool_
            ),
            "hall_governor_rollout_step": _hall_governor_concat(
                hall_governor_step_rows, (0,), np.int32
            ),
            "hall_governor_time_s": _hall_governor_concat(
                hall_governor_time_rows, (0,), np.float32
            ),
            "hall_governor_env_id": _hall_governor_concat(
                hall_governor_env_rows, (0,), np.int32
            ),
        }
        hall_governor_diagnostic_summary = summarize_hall_command_governor_trace(
            risk_probability=hall_governor_diagnostic_arrays[
                "hall_governor_risk_probability"
            ],
            filtered_probability=hall_governor_diagnostic_arrays[
                "hall_governor_filtered_probability"
            ],
            state=hall_governor_diagnostic_arrays["hall_governor_state"],
            requested_vx=hall_governor_diagnostic_arrays[
                "hall_governor_requested_command"
            ][:, 0],
            upstream_vx=hall_governor_diagnostic_arrays[
                "hall_governor_health_bounded_command"
            ][:, 0],
            effective_vx=hall_governor_diagnostic_arrays[
                "hall_governor_effective_command"
            ][:, 0],
            valid=hall_governor_diagnostic_arrays["hall_governor_valid"],
            probing=hall_governor_diagnostic_arrays["hall_governor_probing"],
            prebrake=hall_governor_diagnostic_arrays["hall_governor_prebrake"],
            rollout_step=hall_governor_diagnostic_arrays[
                "hall_governor_rollout_step"
            ],
            env_id=hall_governor_diagnostic_arrays["hall_governor_env_id"],
            step_dt_s=float(base_env.step_dt),
            low_speed_limit_m_s=float(
                hall_command_governor.governor.cfg.low_speed_limit
            ),
            high_speed_limit_m_s=float(
                hall_command_governor.governor.cfg.high_speed_limit
            ),
        )
        hall_governor_diagnostic_summary["manifest"] = (
            hall_command_governor.audit_report()
        )
        hall_governor_diagnostic_summary["health_upper_bound_enabled"] = bool(
            health_envelope is not None
        )
        hall_governor_diagnostic_summary["actor_command_history"] = {
            "synchronized_frames": 5,
            "term_slice": [30, 45],
            "verified_before_every_actor_call": True,
        }
    stability_diagnostic_arrays: dict[str, np.ndarray] = {}
    stability_diagnostic_summary = None
    if stability_envelope is not None:
        def _stability_concat(rows, empty_shape, dtype):
            return (
                np.concatenate(rows, axis=0)
                if rows
                else np.empty(empty_shape, dtype=dtype)
            )

        stability_diagnostic_arrays = {
            "stability_upstream_command": _stability_concat(
                stability_upstream_rows, (0, 3), np.float32
            ),
            "stability_effective_command": _stability_concat(
                stability_effective_rows, (0, 3), np.float32
            ),
            "stability_state": _stability_concat(
                stability_state_rows, (0,), np.int8
            ),
            "stability_reason_mask": _stability_concat(
                stability_reason_rows, (0,), np.int16
            ),
            "stability_intervened": _stability_concat(
                stability_intervened_rows, (0,), np.bool_
            ),
            "stability_command_mean": _stability_concat(
                stability_command_mean_rows, (0, 3), np.float32
            ),
            "stability_heading_command_mean": _stability_concat(
                stability_heading_command_mean_rows, (0,), np.float32
            ),
            "stability_heading_enabled": _stability_concat(
                stability_heading_enabled_rows, (0,), np.bool_
            ),
            "stability_heading_error": _stability_concat(
                stability_heading_signed_rows, (0,), np.float32
            ),
            "stability_heading_error_abs": _stability_concat(
                stability_heading_rows, (0,), np.float32
            ),
            "stability_heading_correction_active": _stability_concat(
                stability_heading_correction_active_rows, (0,), np.bool_
            ),
            "stability_heading_correction_yaw": _stability_concat(
                stability_heading_correction_yaw_rows, (0,), np.float32
            ),
            "stability_omega_xy": _stability_concat(
                stability_omega_rows, (0,), np.float32
            ),
            "stability_tilt": _stability_concat(
                stability_tilt_rows, (0,), np.float32
            ),
            "stability_previous_action_norm": _stability_concat(
                stability_previous_action_norm_rows, (0,), np.float32
            ),
            "stability_current_action_norm": _stability_concat(
                stability_current_action_norm_rows, (0,), np.float32
            ),
            "stability_action_slew_norm": _stability_concat(
                stability_action_slew_norm_rows, (0,), np.float32
            ),
            "stability_action_saturation_count": _stability_concat(
                stability_action_saturation_count_rows, (0,), np.int8
            ),
            "stability_warn_count": _stability_concat(
                stability_warn_count_rows, (0,), np.int16
            ),
            "stability_limit_count": _stability_concat(
                stability_limit_count_rows, (0,), np.int16
            ),
            "stability_hard_limit_count": _stability_concat(
                stability_hard_limit_count_rows, (0,), np.int16
            ),
            "stability_recovery_count": _stability_concat(
                stability_recovery_count_rows, (0,), np.int16
            ),
            "stability_rollout_step": _stability_concat(
                stability_step_rows, (0,), np.int32
            ),
            "stability_time_s": _stability_concat(
                stability_time_rows, (0,), np.float32
            ),
            "stability_env_id": _stability_concat(
                stability_env_rows, (0,), np.int32
            ),
        }
        stability_diagnostic_summary = summarize_high_speed_stability_trace(
            upstream_command=stability_diagnostic_arrays[
                "stability_upstream_command"
            ],
            effective_command=stability_diagnostic_arrays[
                "stability_effective_command"
            ],
            state=stability_diagnostic_arrays["stability_state"],
            reason_mask=stability_diagnostic_arrays["stability_reason_mask"],
            intervened=stability_diagnostic_arrays["stability_intervened"],
            heading_enabled=stability_diagnostic_arrays[
                "stability_heading_enabled"
            ],
            heading_command_mean=stability_diagnostic_arrays[
                "stability_heading_command_mean"
            ],
            heading_error=stability_diagnostic_arrays[
                "stability_heading_error"
            ],
            heading_error_abs=stability_diagnostic_arrays[
                "stability_heading_error_abs"
            ],
            heading_correction_active=stability_diagnostic_arrays[
                "stability_heading_correction_active"
            ],
            heading_correction_yaw=stability_diagnostic_arrays[
                "stability_heading_correction_yaw"
            ],
            omega_xy=stability_diagnostic_arrays["stability_omega_xy"],
            tilt=stability_diagnostic_arrays["stability_tilt"],
        )
        stability_diagnostic_summary["enabled"] = True
        stability_diagnostic_summary["config"] = {
            name: value for name, value in vars(stability_envelope.cfg).items()
        }
        stability_diagnostic_summary["turning_limit"] = (
            "reset-relative heading thresholds are disabled when abs(mean yaw "
            "command) exceeds turning_yaw_command_threshold; the optional yaw "
            "correction is also transparent in that case. Integrate commanded "
            "heading before enabling either for general turning"
        )
    stability_recovery_diagnostic_arrays: dict[str, np.ndarray] = {}
    stability_recovery_diagnostic_summary = None
    if stability_recovery_blend is not None:
        def _recovery_concat(rows, empty_shape, dtype):
            return (
                np.concatenate(rows, axis=0)
                if rows
                else np.empty(empty_shape, dtype=dtype)
            )

        stability_recovery_diagnostic_arrays = {
            "stability_recovery_gate": _recovery_concat(
                stability_recovery_gate_rows, (0,), np.float32
            ),
            "stability_recovery_active": _recovery_concat(
                stability_recovery_active_rows, (0,), np.bool_
            ),
            "stability_recovery_baseline_action": _recovery_concat(
                stability_recovery_baseline_action_rows, (0, 29), np.float32
            ),
            "stability_recovery_expert_action": _recovery_concat(
                stability_recovery_expert_action_rows, (0, 29), np.float32
            ),
            "stability_recovery_output_action": _recovery_concat(
                stability_recovery_output_action_rows, (0, 29), np.float32
            ),
            "stability_recovery_rollout_step": _recovery_concat(
                stability_recovery_step_rows, (0,), np.int32
            ),
            "stability_recovery_time_s": _recovery_concat(
                stability_recovery_time_rows, (0,), np.float32
            ),
            "stability_recovery_env_id": _recovery_concat(
                stability_recovery_env_rows, (0,), np.int32
            ),
        }
        gate = stability_recovery_diagnostic_arrays["stability_recovery_gate"]
        active = stability_recovery_diagnostic_arrays["stability_recovery_active"]
        baseline_action = stability_recovery_diagnostic_arrays[
            "stability_recovery_baseline_action"
        ]
        output_action = stability_recovery_diagnostic_arrays[
            "stability_recovery_output_action"
        ]
        action_delta_l2 = np.linalg.norm(output_action - baseline_action, axis=1)
        nonzero = gate > 1.0e-7
        first_step = (
            int(
                stability_recovery_diagnostic_arrays[
                    "stability_recovery_rollout_step"
                ][nonzero].min()
            )
            if np.any(nonzero)
            else None
        )
        stability_recovery_diagnostic_summary = {
            "enabled": True,
            "checkpoint": getattr(
                stability_recovery_blend.recovery_actor,
                "checkpoint_path",
                None,
            ),
            "sample_count": int(gate.size),
            "emergency_target_fraction": (
                float(active.mean()) if active.size else 0.0
            ),
            "blend_nonzero_fraction": (
                float(nonzero.mean()) if nonzero.size else 0.0
            ),
            "blend_full_fraction": (
                float((gate >= 1.0 - 1.0e-7).mean()) if gate.size else 0.0
            ),
            "first_blend_rollout_step": first_step,
            "gate": {
                "min": float(gate.min()) if gate.size else 0.0,
                "mean": float(gate.mean()) if gate.size else 0.0,
                "max": float(gate.max()) if gate.size else 0.0,
            },
            "output_minus_baseline_action_l2": {
                "mean": float(action_delta_l2.mean()) if action_delta_l2.size else 0.0,
                "max": float(action_delta_l2.max()) if action_delta_l2.size else 0.0,
            },
            "config": {
                name: value
                for name, value in vars(stability_recovery_blend.cfg).items()
            },
            "input_contract": (
                "deployable 1864-D observation + baseline action + stability "
                "state only; no friction/contact/force/course-stage truth"
            ),
            "last_action_contract": (
                "recovery actor newest action-history sample is the actual "
                "previous blended output"
            ),
        }
    report = {
        "policy_kind": policy.kind,
        "steps_requested": args_cli.steps,
        "steps_run": steps_run,
        "command_m_s": args_cli.command,
        "course_geometry": _runtime_course_geometry(),
        "completed_hlh_envs": int(completed.sum().item()),
        "completed_hlh_fraction": float(completed.float().mean().item()),
        "course_success_events": course_success_events,
        "fall_events": falls_total,
        "fall_envs": int(fallen.sum().item()),
        "nan_detected": nan_detected,
        "trace_env_compressed_labels": compress_contact_labels(label_trace),
        "mean_body_vx_m_s": means,
        "region_frame_counts": region_speed_count,
        "gait_adaptation": gait_adaptation,
        "first_episode_only": True,
        "transition_response": response,
        "initial_hall_fault_state": initial_hall_fault_state,
        "hall_health_performance": hall_health_performance,
    }
    if capture_diagnostic_summary is not None:
        report["fastbase_capture_diagnostics"] = capture_diagnostic_summary
    if stability_residual_diagnostic_summary is not None:
        report["fastbase_stability_residual_diagnostics"] = (
            stability_residual_diagnostic_summary
        )
    if health_diagnostic_summary is not None:
        report["health_envelope"] = health_diagnostic_summary
    if hall_governor_diagnostic_summary is not None:
        report["hall_command_governor"] = hall_governor_diagnostic_summary
    if stability_diagnostic_summary is not None:
        report["high_speed_stability_envelope"] = stability_diagnostic_summary
    if stability_recovery_diagnostic_summary is not None:
        report["stability_recovery_blend"] = stability_recovery_diagnostic_summary
    trace = {
        "local_x_m": np.asarray(trace_x, dtype=np.float32),
        "body_vx_m_s": np.asarray(trace_vx, dtype=np.float32),
        "low_contact_privileged": np.asarray(trace_low, dtype=np.bool_),
        "contact_patch_privileged": np.asarray(trace_contact_patch, dtype=np.int8),
        "hall_filtered_T": np.asarray(trace_hall, dtype=np.float32),
        "hall_valid": np.asarray(trace_valid, dtype=np.float32),
    }
    trace.update(health_diagnostic_arrays)
    trace.update(hall_governor_diagnostic_arrays)
    trace.update(stability_diagnostic_arrays)
    trace.update(stability_recovery_diagnostic_arrays)
    trace.update(capture_diagnostic_arrays)
    if args_cli.state_dump_npz is not None:
        state_payload = {
            "recovery_root_pose_local": (recovery_state_pose, (0, 7), np.float32),
            "recovery_root_velocity": (recovery_state_velocity, (0, 6), np.float32),
            "recovery_joint_pos": (recovery_state_joint_pos, (0, 29), np.float32),
            "recovery_joint_vel": (recovery_state_joint_vel, (0, 29), np.float32),
            "recovery_observation": (recovery_state_observation, (0, 1864), np.float32),
            "recovery_motion_feedback_initial_yaw": (
                recovery_state_motion_initial_yaw, (0,), np.float32
            ),
            "recovery_straight_heading_reference_xy": (
                recovery_state_heading_reference, (0, 2), np.float32
            ),
            "recovery_straight_track_origin_local_xy": (
                recovery_state_track_origin_local, (0, 2), np.float32
            ),
            "recovery_straight_track_lateral_axis": (
                recovery_state_track_lateral_axis, (0, 2), np.float32
            ),
            "recovery_source_env_id": (recovery_state_env_id, (0,), np.int32),
            "recovery_source_rollout_step": (
                recovery_state_rollout_step, (0,), np.int32
            ),
            "recovery_hall_local_deformation": (
                recovery_hall_local_deformation, (0, 2, 15, 6), np.float32
            ),
            "recovery_hall_loading_history": (
                recovery_hall_loading_history, (0, 2, 15, 1, 6), np.float32
            ),
            "recovery_hall_signal_filtered_absolute": (
                recovery_hall_signal_filtered_absolute, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_signal_processed": (
                recovery_hall_signal_processed, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_signal_baseline": (
                recovery_hall_signal_baseline, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_signal_drift": (
                recovery_hall_signal_drift, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_policy_history": (
                recovery_hall_policy_history, (0, 2, 1, 15, 3), np.float32
            ),
            "recovery_hall_policy_gain": (
                recovery_hall_policy_gain, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_policy_cross_axis": (
                recovery_hall_policy_cross_axis, (0, 2, 15, 3, 3), np.float32
            ),
            "recovery_hall_policy_zero_residual": (
                recovery_hall_policy_zero_residual, (0, 2, 15, 3), np.float32
            ),
            "recovery_hall_policy_channel_keep": (
                recovery_hall_policy_channel_keep, (0, 2, 15, 1), np.float32
            ),
            "recovery_hall_policy_foot_keep": (
                recovery_hall_policy_foot_keep, (0, 2, 1, 1), np.float32
            ),
            "recovery_hall_policy_delay_steps": (
                recovery_hall_policy_delay_steps, (0, 2), np.int64
            ),
            "recovery_hall_reported_sample_period": (
                recovery_hall_reported_sample_period, (0, 2), np.float32
            ),
        }
        state_counts: dict[str, int] = {}
        for name, (rows, empty_shape, dtype) in state_payload.items():
            trace[name] = (
                np.concatenate(rows, axis=0).astype(dtype, copy=False)
                if rows else np.empty(empty_shape, dtype=dtype)
            )
            state_counts[name] = int(trace[name].shape[0])
        if len(set(state_counts.values())) != 1:
            raise RuntimeError(
                "HighEnd state-dump arrays lost row alignment: "
                f"{state_counts}"
            )
        source_env = trace["recovery_source_env_id"].astype(np.int64, copy=False)
        source_step = trace["recovery_source_rollout_step"].astype(np.int64, copy=False)
        fall_step_by_env = first_fall_step.detach().cpu().numpy().astype(np.int64)
        selected_fall_step = fall_step_by_env[source_env]
        trace["recovery_source_episode_fall"] = selected_fall_step >= 0
        trace["recovery_time_to_fall_s"] = np.where(
            selected_fall_step >= source_step,
            (selected_fall_step - source_step) * float(base_env.step_dt),
            -1.0,
        ).astype(np.float32)
    if args_cli.dataset_npz is not None:
        trace["dataset_observation"] = (
            np.concatenate(dataset_obs, axis=0) if dataset_obs else np.empty((0, 1864), dtype=np.float32)
        )
        trace["dataset_action"] = (
            np.concatenate(dataset_actions, axis=0) if dataset_actions else np.empty((0, 29), dtype=np.float32)
        )
        trace["dataset_low"] = (
            np.concatenate(dataset_low, axis=0) if dataset_low else np.empty((0,), dtype=np.bool_)
        )
    if args_cli.failure_analysis_npz is not None:
        failure_payload = {
            "failure_observation": (failure_obs_rows, (0, 1864), np.float32),
            "failure_action": (failure_action_rows, (0, 29), np.float32),
            "failure_env_id": (failure_env_rows, (0,), np.int32),
            "failure_rollout_step": (failure_step_rows, (0,), np.int32),
            "failure_time_s": (failure_time_rows, (0,), np.float32),
            "failure_local_x_m": (failure_local_x_rows, (0,), np.float32),
            "failure_root_pose_local": (failure_root_pose_rows, (0, 7), np.float32),
            "failure_root_lin_vel_b": (failure_root_lin_vel_b_rows, (0, 3), np.float32),
            "failure_root_ang_vel_b": (failure_root_ang_vel_b_rows, (0, 3), np.float32),
            "failure_joint_pos": (failure_joint_pos_rows, (0, 29), np.float32),
            "failure_joint_vel": (failure_joint_vel_rows, (0, 29), np.float32),
            "failure_foot_contact_lr": (failure_contact_rows, (0, 2), np.bool_),
            "failure_course_stage": (failure_stage_rows, (0,), np.uint8),
            "failure_effective_hall_gate": (failure_gate_rows, (0,), np.float32),
            "failure_capture_delta_l2": (failure_capture_delta_rows, (0,), np.float32),
            "failure_stability_authority": (failure_stability_authority_rows, (0,), np.float32),
            "failure_stability_delta_l2": (failure_stability_delta_rows, (0,), np.float32),
            "failure_fall": (failure_fall_rows, (0,), np.bool_),
            "failure_done": (failure_done_rows, (0,), np.bool_),
            "failure_time_out": (failure_timeout_rows, (0,), np.bool_),
        }
        row_counts: dict[str, int] = {}
        for name, (rows, empty_shape, dtype) in failure_payload.items():
            trace[name] = (
                np.concatenate(rows, axis=0)
                if rows else np.empty(empty_shape, dtype=dtype)
            )
            row_counts[name] = int(trace[name].shape[0])
        if len(set(row_counts.values())) != 1:
            raise RuntimeError(
                "failure-analysis arrays lost causal row alignment: "
                f"{row_counts}"
            )
    return report, trace


def main() -> int:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
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
    if args_cli.hall_contact_distribution is not None:
        hall_cfg = getattr(env_cfg, "hall_sensor_cfg", None)
        if hall_cfg is None:
            raise RuntimeError(
                "--hall_contact_distribution requires hall_sensor_cfg"
            )
        hall_cfg.contact_distribution_mode = args_cli.hall_contact_distribution
        sync_hall_sensor_cfg_to_policy_terms(env_cfg.observations, hall_cfg)
    if (
        args_cli.hall_contact_force_atol is not None
        or args_cli.hall_contact_force_rtol is not None
        or args_cli.hall_contact_audit_warn_only
    ):
        hall_cfg = getattr(env_cfg, "hall_sensor_cfg", None)
        if hall_cfg is None:
            raise RuntimeError(
                "Hall contact force tolerance override requires hall_sensor_cfg"
            )
        if args_cli.hall_contact_force_atol is not None:
            hall_cfg.detailed_contact_force_atol = float(
                args_cli.hall_contact_force_atol
            )
        if args_cli.hall_contact_force_rtol is not None:
            hall_cfg.detailed_contact_force_rtol = float(
                args_cli.hall_contact_force_rtol
            )
        if args_cli.hall_contact_audit_warn_only:
            hall_cfg.detailed_contact_fail_on_audit_mismatch = False
        sync_hall_sensor_cfg_to_policy_terms(env_cfg.observations, hall_cfg)
    if args_cli.hardened_hall:
        hall_cfg = getattr(env_cfg, "hall_sensor_cfg", None)
        if hall_cfg is None:
            raise RuntimeError("--hardened_hall requires hall_sensor_cfg")
        hall_cfg.enable_domain_randomization = True
        hall_cfg.foot_dropout_probability = 0.10
        hall_cfg.dead_channel_probability = 0.08
        hall_cfg.maximum_packet_delay_steps = 5
        sync_hall_sensor_cfg_to_policy_terms(env_cfg.observations, hall_cfg)
    if args_cli.floor_width_m is not None:
        width = float(args_cli.floor_width_m)
        if not math.isfinite(width) or width < 3.2:
            raise ValueError("floor_width_m must be finite and at least 3.2 m")
        for attr in ("friction_high_start", "friction_low", "friction_high_end"):
            patch = getattr(env_cfg.scene, attr)
            size = tuple(float(item) for item in patch.spawn.size)
            patch.spawn.size = (size[0], width, size[2])
    if args_cli.low_patch_mu is not None:
        low_mu = float(args_cli.low_patch_mu)
        if not math.isfinite(low_mu) or not 0.05 <= low_mu <= 0.85:
            raise ValueError("low_patch_mu must be finite and in [0.05, 0.85]")
        material = env_cfg.scene.friction_low.spawn.physics_material
        material.static_friction = low_mu
        material.dynamic_friction = low_mu
        for event_name in ("spatial_friction_reset", "spatial_friction_update"):
            event = getattr(env_cfg.events, event_name, None)
            if event is not None and isinstance(event.params, dict):
                event.params["low_patch_mu"] = low_mu
    raw_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    base_env = raw_env.unwrapped
    try:
        observation, _ = raw_env.reset(seed=args_cli.seed)
        zero_actions = torch.zeros(
            (base_env.num_envs, base_env.action_manager.total_action_dim),
            device=base_env.device,
        )
        _force_command(base_env, args_cli.command)
        observation, _, _, _, _ = raw_env.step(zero_actions)

        actor_audit = _audit_actor_boundary(base_env, observation)
        usd_patches = _audit_usd_patches(base_env)
        runtime_audit = _audit_contact_and_hall(base_env, observation)
        probe_report = None
        if not args_cli.skip_label_probe:
            probe_report = _run_label_probe(raw_env)

        if args_cli.video:
            args_cli.video_dir.mkdir(parents=True, exist_ok=True)
            raw_env = gym.wrappers.RecordVideo(
                raw_env,
                video_folder=str(args_cli.video_dir.resolve()),
                step_trigger=lambda step: step == 0,
                video_length=min(args_cli.video_length, max(args_cli.steps, 1)),
                disable_logger=True,
            )

        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        if hybrid_requested:
            policy = _CausalHallHybridPolicy(
                args_cli.hybrid_baseline_onnx,
                args_cli.hybrid_recovery_onnx,
                args_cli.hall_risk_checkpoint,
                base_env,
                args_cli.hybrid_risk_start,
                args_cli.hybrid_risk_full,
                args_cli.hybrid_on_steps,
                args_cli.hybrid_off_steps,
                args_cli.hybrid_max_active_steps,
                args_cli.hybrid_recovery_command,
            )
        elif args_cli.blend_onnx:
            policy = _BlendPolicy(
                _OnnxPolicy(args_cli.onnx, base_env),
                _OnnxPolicy(args_cli.blend_onnx, base_env),
                args_cli.blend_alpha,
            )
        elif args_cli.checkpoint:
            policy = _RslPolicy(env, base_env)
        elif args_cli.onnx:
            policy = _OnnxPolicy(args_cli.onnx, base_env)
        elif args_cli.torchscript:
            policy = _TorchScriptPolicy(args_cli.torchscript, base_env)
        else:
            policy = _ZeroPolicy(base_env)

        health_envelope = None
        if args_cli.hall_health_envelope:
            health_envelope = HealthEnvelope(
                num_envs=base_env.num_envs,
                dt=float(base_env.step_dt),
                device=base_env.device,
                cfg=HealthEnvelopeCfg(
                    single_foot_speed_cap=args_cli.health_single_foot_speed,
                    max_packet_age_s=args_cli.health_max_packet_age,
                    linear_accel_rate=args_cli.health_accel_rate,
                    linear_decel_rate=args_cli.health_decel_rate,
                    recovery_hold_s=args_cli.health_recovery_hold,
                ),
            )

        hall_command_governor = None
        if args_cli.hall_command_governor:
            hall_command_governor = _StrictHallCommandGovernor(
                args_cli.hall_command_risk_checkpoint,
                actor_checkpoint=Path(args_cli.checkpoint),
                num_envs=base_env.num_envs,
                dt=float(base_env.step_dt),
                device=base_env.device,
                cfg=HallTractionGovernorCfg(
                    low_speed_limit=float(args_cli.hall_governor_low_speed),
                    high_speed_limit=float(args_cli.hall_governor_high_speed),
                    critical_speed_limit=float(
                        args_cli.hall_governor_critical_speed
                    ),
                    probability_low_enter=float(
                        args_cli.hall_governor_low_probability
                    ),
                    probability_high_enter=float(
                        args_cli.hall_governor_high_probability
                    ),
                    probability_critical_enter=float(
                        args_cli.hall_governor_critical_probability
                    ),
                    critical_hold_s=float(
                        args_cli.hall_governor_critical_hold_s
                    ),
                    probability_ema_alpha=float(
                        args_cli.hall_governor_probability_alpha
                    ),
                    relative_low_rise=float(
                        args_cli.hall_governor_relative_low_rise
                    ),
                    relative_high_drop=float(
                        args_cli.hall_governor_relative_high_drop
                    ),
                    allow_absolute_high_clear=bool(
                        args_cli.hall_governor_allow_absolute_high_clear
                    ),
                    low_hold_s=float(args_cli.hall_governor_low_hold_s),
                    high_hold_s=float(args_cli.hall_governor_high_hold_s),
                    low_reprobe_s=float(args_cli.hall_governor_low_reprobe_s),
                    probe_duration_s=float(
                        args_cli.hall_governor_probe_duration_s
                    ),
                    probe_speed_limit=float(args_cli.hall_governor_probe_speed),
                    linear_accel_rate=float(args_cli.hall_governor_accel_rate),
                    linear_decel_rate=float(args_cli.hall_governor_decel_rate),
                ),
            )

        stability_envelope = None
        if args_cli.high_speed_stability_envelope:
            conservative = bool(args_cli.stability_conservative_preset)
            early_heading = bool(args_cli.stability_early_heading_preset)
            early_heading_or_conservative = conservative or early_heading
            stability_envelope = HighSpeedStabilityEnvelope(
                num_envs=base_env.num_envs,
                device=base_env.device,
                cfg=HighSpeedStabilityEnvelopeCfg(
                    warn_heading_threshold=(
                        0.25 if early_heading_or_conservative else 0.40
                    ),
                    warn_persistence_steps=(
                        3 if early_heading_or_conservative else 5
                    ),
                    warn_speed_cap=(
                        0.45 if early_heading_or_conservative else 0.55
                    ),
                    limit_heading_threshold=(
                        0.32 if early_heading_or_conservative else 0.45
                    ),
                    limit_persistence_steps=(
                        3 if early_heading_or_conservative else 5
                    ),
                    hard_limit_heading_threshold=(
                        0.38 if early_heading_or_conservative else 0.48
                    ),
                    hard_limit_persistence_steps=(
                        2 if early_heading_or_conservative else 3
                    ),
                    limit_speed_cap=(
                        0.25 if early_heading_or_conservative else 0.40
                    ),
                    emergency_omega_xy_threshold=0.80 if conservative else 1.20,
                    emergency_tilt_threshold=0.12 if conservative else 0.18,
                    emergency_action_norm_threshold=3.5 if conservative else 4.0,
                    emergency_action_component_threshold=(
                        2.2 if conservative else 2.5
                    ),
                    emergency_speed_cap=0.10 if conservative else 0.25,
                    recovery_heading_threshold=(
                        0.18 if early_heading_or_conservative else 0.30
                    ),
                    recovery_tilt_threshold=0.07 if conservative else 0.10,
                    recovery_omega_xy_threshold=0.50 if conservative else 0.80,
                    recovery_persistence_steps=15 if conservative else 10,
                    enable_heading_correction=bool(
                        args_cli.stability_heading_correction
                    ),
                    heading_correction_gain=float(
                        args_cli.stability_heading_gain
                    ),
                    heading_correction_abs_cap=float(
                        args_cli.stability_heading_yaw_cap
                    ),
                    heading_correction_integral_gain=float(
                        args_cli.stability_heading_integral_gain
                    ),
                    heading_correction_integral_abs_cap=float(
                        args_cli.stability_heading_integral_abs_cap
                    ),
                    heading_correction_integral_decay=float(
                        args_cli.stability_heading_integral_decay
                    ),
                    heading_correction_activate_always=bool(
                        args_cli.stability_heading_correction_always
                    ),
                ),
            )

        stability_recovery_blend = None
        if args_cli.stability_recovery_checkpoint is not None:
            recovery_actor = FrozenStage7RecoveryActor.from_checkpoint(
                args_cli.stability_recovery_checkpoint,
                device=base_env.device,
            )
            stability_recovery_blend = StabilityRecoveryBlend(
                recovery_actor,
                num_envs=base_env.num_envs,
                dt=float(base_env.step_dt),
                device=base_env.device,
                cfg=StabilityRecoveryBlendCfg(
                    recovery_forward_command=float(
                        args_cli.stability_recovery_command
                    ),
                    blend_in_time_s=float(
                        args_cli.stability_recovery_blend_in_s
                    ),
                    blend_out_time_s=float(
                        args_cli.stability_recovery_blend_out_s
                    ),
                ),
            )

        rollout, trace = _run_rollout(
            env,
            policy,
            health_envelope=health_envelope,
            hall_command_governor=hall_command_governor,
            stability_envelope=stability_envelope,
            stability_recovery_blend=stability_recovery_blend,
        )
        summary = {
            "format": "spatial-friction-course-eval-v2",
            "task": args_cli.task,
            "seed": args_cli.seed,
            "num_envs": base_env.num_envs,
            "step_dt_s": float(base_env.step_dt),
            "hall_fault_profile": {
                "requested_hardened": bool(args_cli.hardened_hall),
                "domain_randomization_enabled": bool(
                    getattr(env_cfg.hall_sensor_cfg, "enable_domain_randomization", False)
                ),
                "foot_dropout_probability": float(
                    env_cfg.hall_sensor_cfg.foot_dropout_probability
                ),
                "dead_channel_probability": float(
                    env_cfg.hall_sensor_cfg.dead_channel_probability
                ),
                "maximum_packet_delay_steps": int(
                    env_cfg.hall_sensor_cfg.maximum_packet_delay_steps
                ),
            },
            "effective_hall_config": _effective_hall_cfg(env_cfg.hall_sensor_cfg),
            "actor_boundary": actor_audit,
            "usd_physx": {
                "checked_colliders": len(usd_patches),
                "patches": usd_patches,
            },
            "runtime_shapes": runtime_audit,
            "privileged_label_probe": probe_report,
            "natural_rollout": rollout,
            "video_dir": str(args_cli.video_dir.resolve()) if args_cli.video else None,
        }
        strict_summary = _strict_json(summary)
        print("[spatial-friction] PASS runtime smoke", flush=True)
        if probe_report is not None:
            print(
                "[spatial-friction] privileged probe:",
                " -> ".join(probe_report["compressed_labels"]),
                flush=True,
            )
        print(json.dumps(strict_summary, ensure_ascii=False, indent=2), flush=True)

        if args_cli.summary_json is not None:
            args_cli.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args_cli.summary_json.write_text(
                json.dumps(strict_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args_cli.trace_npz is not None:
            args_cli.trace_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args_cli.trace_npz, **trace)
        if args_cli.state_dump_npz is not None:
            args_cli.state_dump_npz.parent.mkdir(parents=True, exist_ok=True)
            count = int(trace["recovery_observation"].shape[0])
            if count <= 0:
                raise RuntimeError(
                    "--state_dump_npz selected no valid HighEnd states"
                )
            checkpoint_path = (
                Path(args_cli.checkpoint).expanduser().resolve()
                if args_cli.checkpoint is not None else None
            )
            state_metadata = {
                "schema_version": "high_end_recovery_state_dump.v2",
                "dataset_role": str(args_cli.state_dump_role),
                "source_seeds": [int(args_cli.seed)],
                "excluded_locked_seeds": sorted(state_dump_locked_seeds),
                "task": str(args_cli.task),
                "policy_dt_s": float(env.unwrapped.step_dt),
                "command_m_s": float(args_cli.command),
                "actor_checkpoint": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "actor_checkpoint_sha256": (
                    _sha256_file(checkpoint_path)
                    if checkpoint_path is not None else None
                ),
                "actor_observation_dim": 1864,
                "measurement_boundary": (
                    "actor contains multi-frame Hall Bx/By/Bz and packet "
                    "metadata; no Hall-to-force conversion"
                ),
                "state_alignment": (
                    "pre-action articulation, actor observation, path "
                    "references and Hall electronics state at the same t"
                ),
                "requires_offline_bank_builder": True,
                "privileged_fields_are_reset_context_only_not_actor_input": True,
            }
            state_output = {
                "root_pose_local": trace["recovery_root_pose_local"],
                "root_velocity": trace["recovery_root_velocity"],
                "joint_pos": trace["recovery_joint_pos"],
                "joint_vel": trace["recovery_joint_vel"],
                "observation": trace["recovery_observation"],
                "motion_feedback_initial_yaw": trace[
                    "recovery_motion_feedback_initial_yaw"
                ],
                "straight_heading_reference_xy": trace[
                    "recovery_straight_heading_reference_xy"
                ],
                "straight_track_origin_local_xy": trace[
                    "recovery_straight_track_origin_local_xy"
                ],
                "straight_track_lateral_axis": trace[
                    "recovery_straight_track_lateral_axis"
                ],
                "hall_local_deformation": trace[
                    "recovery_hall_local_deformation"
                ],
                "hall_loading_history": trace[
                    "recovery_hall_loading_history"
                ],
                "hall_signal_filtered_absolute": trace[
                    "recovery_hall_signal_filtered_absolute"
                ],
                "hall_signal_processed": trace[
                    "recovery_hall_signal_processed"
                ],
                "hall_signal_baseline": trace[
                    "recovery_hall_signal_baseline"
                ],
                "hall_signal_drift": trace["recovery_hall_signal_drift"],
                "hall_policy_history": trace["recovery_hall_policy_history"],
                "hall_policy_gain": trace["recovery_hall_policy_gain"],
                "hall_policy_cross_axis": trace[
                    "recovery_hall_policy_cross_axis"
                ],
                "hall_policy_zero_residual": trace[
                    "recovery_hall_policy_zero_residual"
                ],
                "hall_policy_channel_keep": trace[
                    "recovery_hall_policy_channel_keep"
                ],
                "hall_policy_foot_keep": trace[
                    "recovery_hall_policy_foot_keep"
                ],
                "hall_policy_delay_steps": trace[
                    "recovery_hall_policy_delay_steps"
                ],
                "hall_reported_sample_period": trace[
                    "recovery_hall_reported_sample_period"
                ],
                "source_seed": np.full(count, int(args_cli.seed), dtype=np.int64),
                "source_env_id": trace["recovery_source_env_id"].astype(
                    np.int64, copy=False
                ),
                "source_rollout_step": trace[
                    "recovery_source_rollout_step"
                ].astype(np.int64, copy=False),
                "source_episode_fall": trace[
                    "recovery_source_episode_fall"
                ],
                "time_to_fall_s": trace["recovery_time_to_fall_s"],
                "metadata_json": np.asarray(
                    json.dumps(state_metadata, ensure_ascii=False, sort_keys=True)
                ),
            }
            lengths = {
                name: int(value.shape[0])
                for name, value in state_output.items()
                if name != "metadata_json"
            }
            if set(lengths.values()) != {count}:
                raise RuntimeError(
                    f"HighEnd V2 state dump lost row alignment: {lengths}"
                )
            np.savez_compressed(args_cli.state_dump_npz, **state_output)
        if args_cli.failure_analysis_npz is not None:
            args_cli.failure_analysis_npz.parent.mkdir(
                parents=True, exist_ok=True
            )
            failure_keys = sorted(
                name for name in trace if name.startswith("failure_")
            )
            checkpoint_path = (
                Path(args_cli.checkpoint).expanduser().resolve()
                if args_cli.checkpoint is not None else None
            )
            metadata = {
                "schema_version": "high_end_failure_precursor_trace.v1",
                "dataset_role": "locked_evaluation_only_do_not_train",
                "task": str(args_cli.task),
                "seed": int(args_cli.seed),
                "command_m_s": float(args_cli.command),
                "policy_dt_s": float(env.unwrapped.step_dt),
                "actor_checkpoint": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "actor_checkpoint_sha256": (
                    _sha256_file(checkpoint_path)
                    if checkpoint_path is not None else None
                ),
                "row_alignment": (
                    "pre_action_state_and_observation_at_t, action_t, "
                    "fall_done_timeout_from_env_step_t"
                ),
                "actor_uses_force_contact_mu_slip_or_stage": False,
                "privileged_fields_are_offline_diagnostics_only": True,
            }
            payload = {name: trace[name] for name in failure_keys}
            payload["metadata_json"] = np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            )
            np.savez_compressed(args_cli.failure_analysis_npz, **payload)
        if args_cli.dataset_npz is not None:
            args_cli.dataset_npz.parent.mkdir(parents=True, exist_ok=True)
            dataset_payload = {
                "observation": trace["dataset_observation"],
                "action": trace["dataset_action"],
                "low": trace["dataset_low"],
            }
            # Ordinary actors preserve the exact legacy three-array dataset.
            # A native FastBase actor adds observation-only capture signals
            # plus evaluator labels for offline temporal/stage diagnosis.
            for name in (
                "fastbase_raw_capture_probability",
                "fastbase_capture_probability",
                "fastbase_effective_gate",
                "fastbase_capture_delta_l2",
                "fastbase_stability_authority",
                "fastbase_stability_delta_l2",
                "fastbase_stability_delta_abs_max",
                "fastbase_course_stage",
                "fastbase_rollout_step",
                "fastbase_time_s",
                "fastbase_env_id",
            ):
                if name in trace:
                    dataset_payload[name] = trace[name]
            # Health-envelope fields are present only when explicitly enabled.
            # They are aligned with the saved first-episode actor samples and
            # contain robot-available packet health, never privileged friction
            # or contact truth.
            for name in (
                "health_requested_command",
                "health_target_command",
                "health_effective_command",
                "health_state",
                "health_valid",
                "health_age_s",
                "health_finite",
                "health_foot_healthy",
                "health_recovery_timer_s",
                "health_intervened",
                "health_rollout_step",
                "health_time_s",
                "health_env_id",
            ):
                if name in trace:
                    dataset_payload[name] = trace[name]
            # The strict single-PT Hall governor stays independently opt-in.
            # Its risk/state/command rows are deployable decision signals and
            # never contain the evaluator's material/contact/course labels.
            for name in (
                "hall_governor_risk_probability",
                "hall_governor_filtered_probability",
                "hall_governor_state",
                "hall_governor_requested_command",
                "hall_governor_health_bounded_command",
                "hall_governor_effective_command",
                "hall_governor_valid",
                "hall_governor_probing",
                "hall_governor_prebrake",
                "hall_governor_rollout_step",
                "hall_governor_time_s",
                "hall_governor_env_id",
            ):
                if name in trace:
                    dataset_payload[name] = trace[name]
            # The stability envelope is independently opt-in.  Its arrays are
            # actor/proprio-only and align with the same first-episode rows.
            for name in (
                "stability_upstream_command",
                "stability_effective_command",
                "stability_state",
                "stability_reason_mask",
                "stability_intervened",
                "stability_command_mean",
                "stability_heading_command_mean",
                "stability_heading_enabled",
                "stability_heading_error",
                "stability_heading_error_abs",
                "stability_heading_correction_active",
                "stability_heading_correction_yaw",
                "stability_omega_xy",
                "stability_tilt",
                "stability_previous_action_norm",
                "stability_current_action_norm",
                "stability_action_slew_norm",
                "stability_action_saturation_count",
                "stability_warn_count",
                "stability_limit_count",
                "stability_hard_limit_count",
                "stability_recovery_count",
                "stability_rollout_step",
                "stability_time_s",
                "stability_env_id",
            ):
                if name in trace:
                    dataset_payload[name] = trace[name]
            np.savez_compressed(args_cli.dataset_npz, **dataset_payload)
        if args_cli.require_rollout_hlh and rollout["completed_hlh_envs"] == 0:
            print(
                "[spatial-friction] FAIL: no uninterrupted natural H-L-H completion",
                file=sys.stderr,
                flush=True,
            )
            return 2
        if rollout["nan_detected"]:
            print("[spatial-friction] FAIL: non-finite rollout state", file=sys.stderr, flush=True)
            return 3
        return 0
    finally:
        raw_env.close()


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
