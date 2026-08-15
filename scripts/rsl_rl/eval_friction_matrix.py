#!/usr/bin/env python3
"""Fixed-seed multi-μ evaluation matrix for Foot policies.

Runs short rollouts at fixed velocity commands across friction levels and
prints a CSV-like summary (forward speed, lateral drift, yaw, slip, resets).

Usage (after conda activate isaaclab-v2 + Isaac Lab env):

  cd /home/mosense/guo/unitree_rl_lab
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
import hashlib
import json
import os
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
from unitree_rl_lab.traction.contact_slip import (  # noqa: E402
    CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
    CONTACT_POINT_TANGENTIAL_SLIP_KEY,
    CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
    CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
    LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
    legacy_link_origin_planar_speed,
    static_ground_contact_point_tangential_speed,
)


# Evaluation is an actor-only operation.  Loading a training optimizer or
# iteration counter is both unnecessary and unsafe when a checkpoint was
# produced by a stricter gradient-isolation variant of the same deployable
# actor (for example GateBceOnly).  Keep this identical to the spatial-course
# evaluator so every RSL checkpoint is reconstructed from actor tensors only.
EVAL_ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}


def _selected_actor_checkpoint(cli_namespace) -> tuple[str | None, str]:
    """Return the artifact that actually supplies rollout actions.

    An explicitly supplied DAgger execution teacher overrides the policy under
    evaluation.  Otherwise the shared ONNX/PT policy takes precedence over an
    RSL checkpoint, matching the evaluator's runtime dispatch below.
    """

    for attribute, source in (
        ("dagger_execution_teacher_onnx", "dagger_execution_teacher_onnx"),
        ("shared_onnx", "shared_onnx"),
        ("shared_policy", "shared_policy"),
        ("checkpoint", "rsl_checkpoint"),
    ):
        value = getattr(cli_namespace, attribute, None)
        if value:
            return os.fspath(value), source
    return None, "unresolved"


def _collection_metadata(
    *,
    dataset_kind: str,
    task: str,
    seed: int,
    policy_dt: float,
    collect_stride: int,
    actor_checkpoint: str | os.PathLike[str] | None,
    actor_source: str,
    hall_contact_distribution_mode: str = "unknown",
) -> dict[str, np.ndarray]:
    """Build pickle-free scalar provenance for a ``--collect_npz`` file.

    Unicode and numeric NumPy scalars remain readable with
    ``np.load(..., allow_pickle=False)``.  In particular, do not save a Python
    dict/object array as metadata: that silently forces downstream tools to
    enable pickle for a dataset that can contain externally supplied paths.
    """

    stride = max(int(collect_stride), 1)
    checkpoint_path = ""
    checkpoint_sha256 = ""
    if actor_checkpoint:
        candidate = Path(actor_checkpoint).expanduser().resolve(strict=False)
        checkpoint_path = os.fspath(candidate)
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()

    manifest = {
        "schema_version": 2,
        "dataset_kind": str(dataset_kind),
        "task": str(task),
        "seed": int(seed),
        "policy_dt": float(policy_dt),
        "collect_stride": stride,
        "requested_collect_stride": int(collect_stride),
        "prospective_steps_contiguous": stride == 1,
        "actor_source": str(actor_source),
        "actor_checkpoint": checkpoint_path,
        "actor_checkpoint_sha256": checkpoint_sha256,
        "hall_valid_lr_source": "exact_actor_obs[:,1860:1862]",
        "outcome_alignment": "pre-step actor obs -> same env.step outcome",
        "actor_command_key": "actor_command",
        "actor_command_source": (
            "exact pre-step actor observation newest command frame [:,42:45]"
        ),
        "applied_command_key": "applied_command",
        "applied_command_source": (
            "base_velocity.vel_command_b[:,0:3] snapshot immediately before env.step"
        ),
        "managed_reset_command_history_repair": (
            "newest frame repaired for all rows; all five [30:45) frames "
            "reinitialized to the evaluator request only for managed-reset rows"
        ),
        "contact_slip_schema": CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
        "contact_slip_metric_key": CONTACT_POINT_TANGENTIAL_SLIP_KEY,
        "contact_slip_valid_key": CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
        "contact_slip_formula": CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
        "contact_slip_reference_velocity": "static ground velocity = 0 m/s",
        "contact_slip_aggregation": (
            "normal-force-magnitude weighted over finite active filters and "
            "left/right feet"
        ),
        "legacy_contact_slip_metric_key": LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
        "legacy_contact_slip_formula": (
            "unweighted mean norm(body_com_lin_vel_w_xy) over feet with "
            "abs(net_normal_force_w_z)>5N"
        ),
        "hall_contact_distribution_mode": str(
            hall_contact_distribution_mode
        ),
    }
    return {
        "policy_dt": np.asarray(float(policy_dt), dtype=np.float64),
        "collect_stride": np.asarray(stride, dtype=np.int32),
        "collect_stride_requested": np.asarray(
            int(collect_stride), dtype=np.int32
        ),
        "task": np.asarray(str(task), dtype=np.str_),
        "metadata_seed": np.asarray(int(seed), dtype=np.int64),
        "dataset_kind": np.asarray(str(dataset_kind), dtype=np.str_),
        "actor_source": np.asarray(str(actor_source), dtype=np.str_),
        "actor_checkpoint": np.asarray(checkpoint_path, dtype=np.str_),
        "actor_checkpoint_sha256": np.asarray(
            checkpoint_sha256, dtype=np.str_
        ),
        "contact_slip_schema": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA, dtype=np.str_
        ),
        "contact_slip_metric_key": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_KEY, dtype=np.str_
        ),
        "contact_slip_valid_key": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY, dtype=np.str_
        ),
        "contact_slip_formula": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_FORMULA, dtype=np.str_
        ),
        "legacy_contact_slip_metric_key": np.asarray(
            LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY, dtype=np.str_
        ),
        "hall_contact_distribution_mode": np.asarray(
            str(hall_contact_distribution_mode), dtype=np.str_
        ),
        "metadata_json": np.asarray(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
    }


def _disable_eval_capture_gate_warmup(agent_cfg) -> None:
    """Disable a training-only gate warm-up before constructing an evaluator.

    This does not alter checkpoint tensors or the calibrated actor forward
    path.  It only prevents the optimizer curriculum from rejecting an
    already-released residual head during actor-only inference.
    """

    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if algorithm_cfg is not None and hasattr(
        algorithm_cfg, "capture_gate_warmup_updates"
    ):
        algorithm_cfg.capture_gate_warmup_updates = 0

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
parser.add_argument(
    "--video",
    action="store_true",
    help="Record an Isaac Sim RGB rollout with Hall debug markers enabled.",
)
parser.add_argument("--video_length", type=int, default=450)
parser.add_argument(
    "--video_dir", type=Path, default=Path("artifacts/hall_rough_terrain/videos")
)
parser.add_argument(
    "--camera_terrain",
    choices=("flat", "slope_up", "slope_down", "stairs_up", "stairs_down"),
    default="stairs_up",
    help="Terrain family followed by the fixed overview camera.",
)
parser.add_argument(
    "--camera_follow",
    action="store_true",
    help="Follow env-0's torso during video capture so long trajectories stay visible.",
)
parser.add_argument(
    "--dynamic_friction_visual",
    action="store_true",
    help=(
        "Update the spawned terrain display color whenever the exact PhysX "
        "friction coefficient changes; visualization only."
    ),
)
parser.add_argument(
    "--preserve_task_terrain",
    action="store_true",
    help="Evaluate on the task's generator terrain instead of forcing a plane.",
)
parser.add_argument(
    "--terrain_max_init_level",
    type=int,
    default=-1,
    help="Maximum initial generator row when --preserve_task_terrain is set.",
)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument(
    "--hall_trace_npz",
    type=Path,
    default=None,
    help=(
        "Switch mode only: save frame-synchronous Hall raw field, auto-zeroed "
        "dB, baseline, and validity for one environment.  This contains no "
        "force reconstructed from Hall data."
    ),
)
parser.add_argument(
    "--hall_trace_env_id",
    type=int,
    default=0,
    help="Environment row stored by --hall_trace_npz (default: camera env 0).",
)
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
        "During DAgger collection, execute this 641-D Oracle Teacher while still "
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
    "--shared_onnx",
    type=Path,
    default=None,
    help=(
        "Deployable 1864->29 magnetic policy ONNX. This is useful when the "
        "training-time .pt wrapper is unavailable; inference is performed "
        "with ONNX Runtime and the observation dimensions are checked."
    ),
)
parser.add_argument(
    "--hall_recovery_onnx",
    type=Path,
    default=None,
    help=(
        "Optional deployable 1864->29 Hall recovery action ONNX used by the "
        "anchored governor only while traction is confirmed LOW.  HIGH and "
        "the initial bounded probe keep the original --shared_onnx action; "
        "invalid Hall data always falls back to that original action."
    ),
)
parser.add_argument(
    "--hall_hybrid_blend_start",
    type=float,
    default=0.55,
    help="Risk probability at which the optional hybrid recovery action starts blending in.",
)
parser.add_argument(
    "--hall_hybrid_blend_full",
    type=float,
    default=0.75,
    help="Risk probability at which the optional hybrid recovery action reaches full weight.",
)
parser.add_argument(
    "--hall_recovery_low_blend_floor",
    type=float,
    default=0.70,
    help=(
        "Minimum recovery-action blend after the causal Hall governor has "
        "confirmed LOW. HIGH and UNKNOWN always retain the original actor "
        "unless --hall_recovery_on_probe is explicitly set."
    ),
)
parser.add_argument(
    "--hall_recovery_max_low_s",
    type=float,
    default=0.0,
    help=(
        "Maximum valid LOW-state time for the optional Hall recovery action. "
        "0 keeps the legacy unlimited behavior; a positive value makes the "
        "residual a bounded transient correction and then returns to the "
        "original actor while the command governor remains active."
    ),
)
parser.add_argument(
    "--hall_recovery_on_probe",
    action="store_true",
    help=(
        "Explicitly allow the optional Hall recovery action during a bounded "
        "governor probe.  Disabled by default so an initial high-traction "
        "probe cannot perturb the audited original gait."
    ),
)
parser.add_argument(
    "--hall_governed_command_reflex",
    action="store_true",
    help=(
        "In confirmed LOW, immediately re-evaluate the audited original "
        "actor with its five-frame velocity-command history replaced by the "
        "causal governor output. This removes the ordinary observation-history "
        "delay without using contact force, friction, or slip truth."
    ),
)
parser.add_argument(
    "--lateral_estimator",
    type=Path,
    default=None,
    help="1862-D body-vy estimator .pt; overwrites policy channel 1862 online.",
)
parser.add_argument(
    "--forward_velocity_estimator",
    type=Path,
    default=None,
    help=(
        "Normalized 1864-D Hall/proprio forward-speed estimator .pt.  It is "
        "diagnostic-only here and never feeds simulator velocity or friction "
        "back into the actor."
    ),
)
parser.add_argument(
    "--forward_velocity_filter_alpha",
    type=float,
    default=0.35,
    help="EMA alpha used for the diagnostic forward-speed estimate.",
)
parser.add_argument(
    "--hall_risk_checkpoint",
    type=Path,
    default=None,
    help=(
        "Independent HallTractionRiskEstimator .pt or deployable .onnx. "
        "This changes only the command governor and never the locomotion "
        "action policy."
    ),
)
parser.add_argument(
    "--anchored_hall_governor",
    action="store_true",
    help=(
        "Run the original proprioceptive actor unchanged and use the Hall "
        "risk model only as a causal command governor.  Requires both "
        "--shared_onnx and --hall_risk_checkpoint; this is the recommended "
        "non-regression validation/deployment path."
    ),
)
parser.add_argument(
    "--nominal_magnetic_sensor",
    action="store_true",
    help="Keep both feet valid and disable synthetic packet-drop faults.",
)
parser.add_argument(
    "--detailed_hall_contact",
    action="store_true",
    help=(
        "Use dedicated per-foot contact-filter positions/forces to distribute "
        "the mechanical Hall driver. Disabled by default, preserving the "
        "task's aggregate Hall-contact model."
    ),
)
parser.add_argument(
    "--hall_traction_governor",
    action="store_true",
    help=(
        "Apply the deployable Hall-risk speed/acceleration/turn governor. "
        "Requires a layout-aware --shared_policy, or "
        "--anchored_hall_governor with the original actor."
    ),
)
parser.add_argument("--governor_low_speed", type=float, default=0.10)
parser.add_argument("--governor_high_speed", type=float, default=0.90)
parser.add_argument(
    "--governor_critical_speed",
    type=float,
    default=0.0,
    help=(
        "Forward speed retained under confirmed critical Hall risk.  The "
        "default zero is fail-stop; a validated nonzero crawl can be used "
        "for simulation-only controlled probing."
    ),
)
parser.add_argument("--governor_low_probability", type=float, default=0.65)
parser.add_argument("--governor_high_probability", type=float, default=0.50)
parser.add_argument("--governor_critical_probability", type=float, default=0.85)
parser.add_argument("--governor_critical_hold", type=float, default=0.04)
parser.add_argument(
    "--governor_probability_alpha",
    type=float,
    default=0.20,
    help="EMA alpha for the causal Hall risk state machine.",
)
parser.add_argument("--governor_reference_alpha", type=float, default=0.01)
parser.add_argument(
    "--governor_reference_settle_s",
    type=float,
    default=0.60,
    help=(
        "HIGH-state causal Hall-reference settling window after a bounded "
        "probe or traction recovery. Relative-rise braking is disabled only "
        "during this short initialization window."
    ),
)
parser.add_argument(
    "--governor_reference_settle_alpha",
    type=float,
    default=0.25,
    help="Reference EMA alpha used only inside --governor_reference_settle_s.",
)
parser.add_argument(
    "--governor_prebrake_probability",
    type=float,
    default=None,
    help=(
        "Optional Hall-risk absolute floor for a causal pre-brake before the "
        "normal LOW confirmation. Must be provided with "
        "--governor_prebrake_relative_rise."
    ),
)
parser.add_argument(
    "--governor_prebrake_relative_rise",
    type=float,
    default=None,
    help="Required Hall-risk increase from the settled HIGH reference for pre-brake.",
)
parser.add_argument(
    "--governor_prebrake_speed",
    type=float,
    default=None,
    help="Optional forward speed cap while the Hall-only pre-brake is active.",
)
parser.add_argument(
    "--governor_prebrake_reflex_floor",
    type=float,
    default=0.80,
    help=(
        "Minimum blend of the same actor re-evaluated at the pre-brake "
        "command. Only applies with --hall_governed_command_reflex."
    ),
)
parser.add_argument("--governor_relative_low_rise", type=float, default=0.12)
parser.add_argument("--governor_relative_high_drop", type=float, default=0.12)
parser.add_argument(
    "--governor_relative_low_min_probability",
    type=float,
    default=None,
    help=(
        "Optional minimum Hall-risk probability for a relative-rise brake. "
        "Defaults to --governor_high_probability; lowering it is a validated "
        "early-transition guard, not a friction estimate."
    ),
)
parser.add_argument(
    "--governor_allow_absolute_high_clear",
    action="store_true",
    help=(
        "Allow a sustained calibrated low Hall-risk probability to release LOW. "
        "Use only with a validated prospective-slip risk head; the default "
        "keeps baseline-robust relative-only release."
    ),
)
parser.add_argument("--governor_probe_speed", type=float, default=0.25)
parser.add_argument("--governor_probe_duration", type=float, default=0.45)
parser.add_argument(
    "--governor_initial_probe_ignore_critical",
    action="store_true",
    help=(
        "For a valid Hall stream only, finish the bounded initial excitation "
        "probe before critical-risk braking.  Use only with a risk head "
        "trained on low-speed probe data; invalid/stale Hall remains fail-safe."
    ),
)
parser.add_argument(
    "--governor_allow_critical_reprobe",
    action="store_true",
    help=(
        "Permit a short, speed-bounded active Hall probe after a sustained "
        "critical state so a genuinely recovered high-traction surface can "
        "be identified.  Invalid/stale Hall data can never take this path."
    ),
)
parser.add_argument(
    "--governor_critical_reprobe",
    type=float,
    default=2.50,
    help="Seconds in a valid critical LOW state before one bounded re-probe.",
)
parser.add_argument(
    "--governor_probe_relative_clear_drop", type=float, default=0.08
)
parser.add_argument("--governor_crawl_pulse", type=float, default=0.45)
parser.add_argument("--governor_low_reprobe", type=float, default=2.50)
parser.add_argument("--governor_low_hold", type=float, default=0.10)
parser.add_argument("--governor_high_hold", type=float, default=0.40)
parser.add_argument("--governor_accel_rate", type=float, default=0.50)
parser.add_argument("--governor_decel_rate", type=float, default=2.00)
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
    "--vy",
    type=float,
    default=0.0,
    help="Fixed lateral velocity command used with every --vx case.",
)
parser.add_argument(
    "--wz",
    type=float,
    default=0.0,
    help="Fixed yaw-rate command used with every --vx case.",
)
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
    help=(
        "Maximum last-low -> final-HighEnd recovery response used by the "
        "adaptive-gait acceptance gate."
    ),
)
parser.add_argument(
    "--switch_min_high_end_vx_recovery_ratio",
    type=float,
    default=0.85,
    help=(
        "Minimum abs(vx) recovery of the final high-friction phase relative "
        "to the first high-friction phase."
    ),
)
parser.add_argument(
    "--switch_min_high_end_step_length_recovery_ratio",
    type=float,
    default=0.85,
    help=(
        "Minimum step-length recovery of the final high-friction phase "
        "relative to the first high-friction phase."
    ),
)
parser.add_argument(
    "--switch_max_tilt_deg",
    type=float,
    default=20.0,
    help="Maximum first-fall-censored/pre-fall base tilt safety gate.",
)
parser.add_argument(
    "--switch_max_steady_abs_vy_mps",
    type=float,
    default=0.25,
    help="Maximum per-phase steady mean absolute lateral velocity safety gate.",
)
parser.add_argument(
    "--switch_max_contact_point_slip_mps",
    type=float,
    default=None,
    help=(
        "Optional calibrated ceiling for corrected contact-point tangential "
        "slip.  When omitted, corrected slip is diagnostic-only and the "
        "summary cannot claim a fully calibrated slip gate."
    ),
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

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.sensors import (  # noqa: E402
    sync_hall_sensor_cfg_to_policy_terms,
)
from unitree_rl_lab.traction.hall_governor import (  # noqa: E402
    HIGH,
    LOW,
    UNKNOWN,
    HallTractionGovernor,
    HallTractionGovernorCfg,
)


def _configure_nominal_hall_sensor_cfg(
    env_cfg,
    *,
    enabled: bool,
    sync_fn,
) -> tuple[str, ...]:
    """Install a reset-persistent nominal Hall configuration before env creation.

    Hall sensors are constructed lazily from the independent ``hall_cfg``
    copies owned by observation terms.  Editing runtime buffers after one
    reset is therefore insufficient: a later managed reset legitimately
    samples from the original randomized term config again.  Updating the
    top-level config and synchronizing every Hall term makes the sensor's own
    reset path deterministically restore identity gain/cross-axis, no packet
    delay, and no channel/foot dropout for every episode.

    ``enabled=False`` is an exact no-op so hardened/default evaluation keeps
    the task's configured domain-randomization profile.
    """

    if not enabled:
        return ()
    hall_cfg = getattr(env_cfg, "hall_sensor_cfg", None)
    if hall_cfg is None:
        raise ValueError(
            "--nominal_magnetic_sensor requires a task with top-level "
            "hall_sensor_cfg"
        )
    hall_cfg.enable_domain_randomization = False
    synchronized_terms = tuple(sync_fn(env_cfg.observations, hall_cfg))
    if not synchronized_terms:
        raise RuntimeError(
            "--nominal_magnetic_sensor found no Hall observation terms to synchronize"
        )
    return synchronized_terms


def _configure_detailed_hall_contact_cfg(
    env_cfg,
    *,
    enabled: bool,
    sync_fn,
) -> tuple[str, tuple[str, ...]]:
    """Select detailed Hall contact before environment construction.

    The disabled path is an exact no-op and reports the task's effective
    top-level mode.  The enabled path fails closed unless both the modern
    ``contact_distribution_mode`` field and all Hall policy terms are present;
    otherwise an apparent aggregate/detailed A/B would silently compare
    mismatched sensor configs.
    """

    hall_cfg = getattr(env_cfg, "hall_sensor_cfg", None)
    if hall_cfg is None:
        if enabled:
            raise ValueError(
                "--detailed_hall_contact requires a task with top-level "
                "hall_sensor_cfg"
            )
        return "unavailable", ()
    if not hasattr(hall_cfg, "contact_distribution_mode"):
        if enabled:
            raise ValueError(
                "--detailed_hall_contact requires HallFootSensorCfg."
                "contact_distribution_mode; refusing a silent fallback"
            )
        return "unknown", ()

    current_mode = str(hall_cfg.contact_distribution_mode)
    if current_mode not in ("aggregate", "detailed"):
        raise ValueError(
            "Hall contact_distribution_mode must be 'aggregate' or "
            f"'detailed', got {current_mode!r}"
        )
    if not enabled:
        return current_mode, ()

    hall_cfg.contact_distribution_mode = "detailed"
    synchronized_terms = tuple(sync_fn(env_cfg.observations, hall_cfg))
    if not synchronized_terms:
        raise RuntimeError(
            "--detailed_hall_contact found no Hall observation terms to synchronize"
        )
    if str(hall_cfg.contact_distribution_mode) != "detailed":
        raise RuntimeError("detailed Hall contact configuration did not persist")
    return "detailed", synchronized_terms


def _force_command(env, vx: float, vy: float = 0.0, wz: float = 0.0):
    """Set constant base velocity command if command term supports ranges."""
    try:
        term = env.unwrapped.command_manager.get_term("base_velocity")
        term.cfg.ranges.lin_vel_x = (vx, vx)
        term.cfg.ranges.lin_vel_y = (vy, vy)
        term.cfg.ranges.ang_vel_z = (wz, wz)
        term.cfg.rel_standing_envs = 0.0
        if hasattr(term.cfg, "rel_spin_envs"):
            term.cfg.rel_spin_envs = 0.0
        if hasattr(term.cfg, "high_speed_fraction"):
            term.cfg.high_speed_fraction = 0.0
        # resample all
        env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        term._resample_command(env_ids)
        term.is_standing_env[:] = False
        for attribute in ("is_spin_env", "is_heading_env"):
            flag = getattr(term, attribute, None)
            if flag is not None:
                flag[:] = False
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1] = vy
        term.vel_command_b[:, 2] = wz
    except Exception as e:
        print(f"[warn] could not force command: {e}")


def _set_command_value(env, vx: float, vy: float = 0.0, wz: float = 0.0):
    """Update the active command without resampling/resetting its history."""
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.is_standing_env[:] = False
    # Fixed-command evaluation must also disable state left behind by custom
    # generators.  Otherwise _update_command() can zero/rotate the value again
    # inside env.step even after vel_command_b was assigned below.
    for attribute in ("is_spin_env", "is_heading_env"):
        flag = getattr(term, attribute, None)
        if flag is not None:
            flag[:] = False
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = vy
    term.vel_command_b[:, 2] = wz


def _synchronize_evaluator_command_observation(
    env,
    observation,
    expected_command: torch.Tensor,
    managed_resets: torch.Tensor,
) -> None:
    """Repair fixed-command state after generator resampling, fail closed.

    Isaac's managed reset resamples custom command terms and resets the
    observation history before returning from ``env.step``.  Some custom
    generators intentionally ignore ``cfg.ranges``, so changing those ranges
    is insufficient.  This helper writes the command term for every row,
    repairs only the newest policy-history frame during ordinary operation,
    and fills all five frames only for newly reset rows.  Thus a normal ramp's
    older causal history is left bit-identical.
    """

    uenv = env.unwrapped
    num_envs = int(uenv.num_envs)
    if (
        not isinstance(expected_command, torch.Tensor)
        or expected_command.shape != (num_envs, 3)
        or not expected_command.is_floating_point()
        or not torch.isfinite(expected_command).all()
    ):
        raise ValueError(
            "expected_command must be a finite floating tensor with shape "
            f"[{num_envs},3]"
        )
    if (
        not isinstance(managed_resets, torch.Tensor)
        or managed_resets.shape != (num_envs,)
    ):
        raise ValueError(
            f"managed_resets must have shape [{num_envs}]"
        )
    reset_mask = managed_resets.to(
        device=expected_command.device, dtype=torch.bool
    )

    term = uenv.command_manager.get_term("base_velocity")
    if term.vel_command_b.shape[0] != num_envs or term.vel_command_b.shape[1] < 3:
        raise RuntimeError("base_velocity command tensor has an invalid shape")
    term.is_standing_env[:] = False
    for attribute in ("is_spin_env", "is_heading_env"):
        flag = getattr(term, attribute, None)
        if flag is not None:
            flag[:] = False
    term.vel_command_b[:, :3] = expected_command

    try:
        history = uenv.observation_manager._group_obs_term_history_buffer[
            "policy"
        ]["velocity_commands"]
        policy_observation = observation["policy"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "fixed-command synchronization requires policy/velocity_commands "
            "history"
        ) from exc
    if (
        history.max_length != 5
        or history._buffer is None
        or history._buffer.ndim != 3
        or tuple(history._buffer.shape[1:]) != (num_envs, 3)
        or not 0 <= int(history._pointer) < history.max_length
    ):
        shape = None if history._buffer is None else tuple(history._buffer.shape)
        raise RuntimeError(
            "policy velocity_commands history must be initialized as "
            f"[5,{num_envs},3], got {shape}"
        )
    if (
        not isinstance(policy_observation, torch.Tensor)
        or policy_observation.ndim != 2
        or policy_observation.shape[0] != num_envs
        or policy_observation.shape[1] < 45
    ):
        raise RuntimeError(
            "policy observation must expose the canonical command slice [30:45)"
        )

    # CircularBuffer.buffer is oldest -> newest.  _pointer identifies the
    # newest storage slot before that chronological view is materialized.
    history._buffer[history._pointer, :, :3] = expected_command
    policy_observation[:, 42:45] = expected_command
    if bool(reset_mask.any().item()):
        history._buffer[:, reset_mask, :3] = expected_command[reset_mask].unsqueeze(0)
        policy_observation[reset_mask, 30:45] = expected_command[
            reset_mask
        ].repeat(1, 5)


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
    _update_friction_visual(env, mu)


def _friction_visual_color(mu: float) -> tuple[float, float, float]:
    """Map μ to a high-contrast opaque floor color for recordings."""
    # Blue = low traction, amber = intermediate, green = high traction.
    value = float(np.clip(mu, 0.08, 1.20))
    if value <= 0.25:
        t = (value - 0.08) / 0.17
        low = np.asarray((0.03, 0.16, 0.88), dtype=np.float32)
        mid = np.asarray((0.95, 0.45, 0.03), dtype=np.float32)
        color = (1.0 - t) * low + t * mid
    elif value <= 0.75:
        t = (value - 0.25) / 0.50
        mid = np.asarray((0.95, 0.45, 0.03), dtype=np.float32)
        high = np.asarray((0.05, 0.72, 0.16), dtype=np.float32)
        color = (1.0 - t) * mid + t * high
    else:
        t = (value - 0.75) / 0.45
        high = np.asarray((0.05, 0.72, 0.16), dtype=np.float32)
        very_high = np.asarray((0.02, 0.42, 0.08), dtype=np.float32)
        color = (1.0 - t) * high + t * very_high
    return tuple(float(component) for component in color)


def _update_friction_visual(env, mu: float) -> None:
    """Change terrain shader/display colors without touching PhysX materials.

    Isaac Lab's TerrainImporter binds a PreviewSurface under
    ``<terrain_prim>/visualMaterial/Shader``.  We update that shader and the
    mesh displayColor as a compatibility fallback for generator/plane/USD
    terrains.  All operations are guarded because visualization must never
    make headless training fail.
    """
    if not args_cli.dynamic_friction_visual:
        return
    try:
        from pxr import Gf, Usd, UsdGeom, UsdShade
        import omni.usd

        terrain = getattr(env.unwrapped.scene, "terrain", None)
        paths = list(getattr(terrain, "terrain_prim_paths", []))
        if not paths:
            paths = ["/World/terrain"]
        stage = omni.usd.get_context().get_stage()
        color = _friction_visual_color(mu)
        color_vec = Gf.Vec3f(*color)
        for terrain_path in paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                # Isaac Sim's stock grid plane is an imported USD and stores
                # its tint at Looks/theGrid/Shader.inputs:diffuse_tint.  Mesh
                # terrains use the PreviewSurface path handled below.
                for prim in Usd.PrimRange(terrain_prim):
                    if not prim.IsA(UsdShade.Shader):
                        continue
                    shader = UsdShade.Shader(prim)
                    for input_name in (
                        "diffuse_tint",
                        "diffuseColor",
                        "diffuse_color",
                        "base_color",
                    ):
                        material_input = shader.GetInput(input_name)
                        if material_input:
                            material_input.Set(color_vec)
            mesh = stage.GetPrimAtPath(f"{terrain_path}/mesh")
            if mesh.IsValid():
                display = UsdGeom.Primvar(
                    mesh.GetAttribute("primvars:displayColor")
                )
                if display:
                    display.SetInterpolation(UsdGeom.Tokens.constant)
                    display.Set([color_vec])
            shader_prim = stage.GetPrimAtPath(
                f"{terrain_path}/visualMaterial/Shader"
            )
            if shader_prim.IsValid():
                shader = UsdShade.Shader(shader_prim)
                diffuse = shader.GetInput("diffuseColor")
                if diffuse:
                    diffuse.Set(color_vec)
                roughness = shader.GetInput("roughness")
                if roughness:
                    roughness.Set(0.78 if mu <= 0.25 else 0.62)
    except Exception as exc:
        # This is an optional GUI/video aid.  Do not hide a physics result or
        # break a parallel rollout if a particular Isaac terrain backend uses
        # a material path that cannot be edited at runtime.
        if not getattr(_update_friction_visual, "_warned", False):
            print(f"[warn] dynamic friction visual unavailable: {exc}")
            _update_friction_visual._warned = True


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


def _exact_actor_policy_observation(policy, submitted_observation) -> torch.Tensor:
    """Return the exact 2-D policy tensor consumed by the current actor call.

    Wrapped policies expose their post-transform input through
    ``last_policy_observation``.  A native RSL actor consumes the submitted
    ``policy`` group directly, including any explicit Hall ablation applied by
    this evaluator.  Falling back to the *raw* environment observation here
    would silently save nominal Hall data for an ablated rollout.
    """

    exact = getattr(policy, "last_policy_observation", None)
    if exact is None:
        try:
            exact = submitted_observation["policy"]
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(
                "policy input capture requires a submitted policy observation group"
            ) from exc
    if not isinstance(exact, torch.Tensor) or exact.ndim != 2:
        raise ValueError(
            "captured actor policy input must be a rank-2 torch.Tensor, "
            f"got {type(exact).__name__} shape={getattr(exact, 'shape', None)}"
        )
    return exact


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


def _high_start_recovery_response_time(
    values: list[float],
    high_start_speed: float,
    recovery_ratio: float,
    dt: float,
    window_steps: int,
) -> float:
    """Time for HighEnd speed to regain a fraction of HighStart speed.

    This is intentionally distinct from ``_response_time``: the latter closes
    a fraction of the observed low-to-high delta and can look fast even when
    the final gait never recovers its original high-friction performance.
    """

    if (
        not np.isfinite(high_start_speed)
        or high_start_speed <= 0.0
        or not np.isfinite(recovery_ratio)
        or recovery_ratio <= 0.0
        or not np.isfinite(dt)
        or dt <= 0.0
        or not values
    ):
        return float("nan")
    window = max(int(window_steps), 1)
    speed = np.abs(np.asarray(values, dtype=np.float64))
    if len(speed) < window:
        smoothed = speed
        effective_window = 1
    else:
        smoothed = np.convolve(
            speed, np.ones(window, dtype=np.float64) / window, mode="valid"
        )
        effective_window = window
    reached = np.isfinite(smoothed) & (
        smoothed >= recovery_ratio * high_start_speed
    )
    dwell = max(int(round(0.20 / dt)), 1)
    for index in range(len(reached)):
        end = index + dwell
        if end <= len(reached) and bool(np.all(reached[index:end])):
            return float((index + effective_window) * dt)
    return float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _finite_mean(values: list[float]) -> float:
    """Return the mean of finite samples, or NaN when none remain.

    First-fall censoring can leave a late rollout step without any surviving
    environments.  Keeping that step as NaN is useful in the time-series CSV,
    but it must not erase valid pre-fall samples from the phase aggregate.
    """

    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if bool(finite.any()) else float("nan")


def _finite_max(values: list[float]) -> float:
    """Return a maximum only when every safety sample is finite."""

    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(array.max()) if bool(np.isfinite(array).all()) else float("nan")


def _projected_gravity_tilt_degrees(
    projected_gravity_b: torch.Tensor,
) -> torch.Tensor:
    """Return full 0--180 degree base tilt from projected gravity.

    Upright G1 has projected gravity close to ``[0, 0, -1]``.  Using
    ``atan2(norm(g_xy), -g_z)`` preserves an inverted pose as 180 degrees,
    unlike ``asin(norm(g_xy))`` which aliases it back to zero.  The input is
    read before ``env.step`` so a managed reset cannot erase the pre-fall
    attitude used by the safety diagnostics.
    """

    if (
        not isinstance(projected_gravity_b, torch.Tensor)
        or projected_gravity_b.ndim != 2
        or projected_gravity_b.shape[1] != 3
    ):
        raise ValueError(
            "projected gravity must be a torch.Tensor with shape [N,3], got "
            f"{type(projected_gravity_b).__name__} "
            f"{getattr(projected_gravity_b, 'shape', None)}"
        )
    horizontal = torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=1)
    return torch.rad2deg(torch.atan2(horizontal, -projected_gravity_b[:, 2]))


def _build_switch_gait_diagnostics(
    phase_rows: list[dict[str, object]],
    *,
    low_mu_max: float = 0.25,
    high_mu_min: float = 0.75,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Enrich switch phases without assuming that low friction must be slow.

    Cadence counts alternating foot touchdowns, while ``mean_step_length_m``
    is half the same-foot stride.  Their product should therefore close to
    forward speed.  A valid low-friction adaptation may increase cadence and
    shorten steps while preserving speed; those ratios are diagnostics, not
    fixed-speed gates.  Recovery is deliberately the *last* high-friction
    phase relative to the *first*, never an average of all high phases.
    """

    if not phase_rows:
        raise ValueError("switch gait diagnostics require at least one phase")
    if not low_mu_max < high_mu_min:
        raise ValueError("low_mu_max must be less than high_mu_min")

    rows = [dict(row) for row in phase_rows]
    rows.sort(key=lambda row: int(row["phase"]))
    phases = [int(row["phase"]) for row in rows]
    if len(set(phases)) != len(phases):
        raise ValueError("switch phase ids must be unique")

    high_indices = [
        index
        for index, row in enumerate(rows)
        if float(row["mu"]) >= high_mu_min
    ]
    if not high_indices:
        raise ValueError("switch diagnostics require a high-friction phase")
    high_start_index = high_indices[0]
    high_end_index = high_indices[-1]
    high_start = rows[high_start_index]

    def finite_ratio(numerator: float, denominator: float) -> float:
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or abs(denominator) <= 1.0e-8
        ):
            return float("nan")
        return float(numerator / denominator)

    high_start_vx = abs(float(high_start["steady_vx"]))
    high_start_cadence = float(high_start["step_frequency_hz"])
    high_start_step = float(high_start["mean_step_length_m"])
    high_start_stride = float(high_start["mean_stride_length_m"])

    for index, row in enumerate(rows):
        cadence = float(row["step_frequency_hz"])
        step_length = float(row["mean_step_length_m"])
        stride_length = float(row["mean_stride_length_m"])
        forward_speed = abs(float(row["steady_vx"]))
        estimate = cadence * step_length
        closure_error = estimate - forward_speed
        row["kinematic_speed_estimate_mps"] = estimate
        row["kinematic_closure_error_mps"] = closure_error
        row["kinematic_closure_relative_error"] = finite_ratio(
            abs(closure_error), forward_speed
        )
        row["cadence_vs_high_start_ratio"] = finite_ratio(
            cadence, high_start_cadence
        )
        row["step_length_vs_high_start_ratio"] = finite_ratio(
            step_length, high_start_step
        )
        row["stride_length_vs_high_start_ratio"] = finite_ratio(
            stride_length, high_start_stride
        )
        row["vx_vs_high_start_ratio"] = finite_ratio(
            forward_speed, high_start_vx
        )
        is_low = float(row["mu"]) <= low_mu_max
        row["low_mu_cadence_vs_high_start_ratio"] = (
            row["cadence_vs_high_start_ratio"] if is_low else float("nan")
        )
        row["low_mu_step_length_vs_high_start_ratio"] = (
            row["step_length_vs_high_start_ratio"] if is_low else float("nan")
        )
        row["low_mu_vx_vs_high_start_ratio"] = (
            row["vx_vs_high_start_ratio"] if is_low else float("nan")
        )
        if index == high_start_index and index == high_end_index:
            role = "HighStart=HighEnd"
        elif index == high_start_index:
            role = "HighStart"
        elif index == high_end_index:
            role = "HighEnd"
        elif is_low:
            role = "Low"
        elif float(row["mu"]) >= high_mu_min:
            role = "HighIntermediate"
        else:
            role = "Other"
        row["phase_role"] = role
        row["high_end_vx_recovery_ratio"] = (
            row["vx_vs_high_start_ratio"]
            if index == high_end_index
            else float("nan")
        )
        row["high_end_step_length_recovery_ratio"] = (
            row["step_length_vs_high_start_ratio"]
            if index == high_end_index
            else float("nan")
        )

    transitions: list[dict[str, object]] = []
    for transition_index, (before, after) in enumerate(
        zip(rows[:-1], rows[1:], strict=True)
    ):
        before_mu = float(before["mu"])
        after_mu = float(after["mu"])
        if before_mu >= high_mu_min and after_mu <= low_mu_max:
            transition_type = "HighToLow"
        elif before_mu <= low_mu_max and after_mu >= high_mu_min:
            transition_type = "LowToHigh"
        else:
            transition_type = "Other"
        before_vx = abs(float(before["steady_vx"]))
        after_vx = abs(float(after["steady_vx"]))
        before_cadence = float(before["step_frequency_hz"])
        after_cadence = float(after["step_frequency_hz"])
        before_step = float(before["mean_step_length_m"])
        after_step = float(after["mean_step_length_m"])
        before_stride = float(before["mean_stride_length_m"])
        after_stride = float(after["mean_stride_length_m"])
        transitions.append(
            {
                "transition": transition_index,
                "from_phase": int(before["phase"]),
                "to_phase": int(after["phase"]),
                "transition_type": transition_type,
                "from_mu": before_mu,
                "to_mu": after_mu,
                "from_vx_mps": before_vx,
                "to_vx_mps": after_vx,
                "vx_delta_mps": after_vx - before_vx,
                "vx_ratio": finite_ratio(after_vx, before_vx),
                "from_cadence_hz": before_cadence,
                "to_cadence_hz": after_cadence,
                "cadence_delta_hz": after_cadence - before_cadence,
                "cadence_ratio": finite_ratio(after_cadence, before_cadence),
                "from_step_length_m": before_step,
                "to_step_length_m": after_step,
                "step_length_delta_m": after_step - before_step,
                "step_length_ratio": finite_ratio(after_step, before_step),
                "from_stride_length_m": before_stride,
                "to_stride_length_m": after_stride,
                "stride_length_delta_m": after_stride - before_stride,
                "stride_length_ratio": finite_ratio(after_stride, before_stride),
                "from_kinematic_speed_estimate_mps": before[
                    "kinematic_speed_estimate_mps"
                ],
                "from_kinematic_closure_error_mps": before[
                    "kinematic_closure_error_mps"
                ],
                "to_kinematic_speed_estimate_mps": after[
                    "kinematic_speed_estimate_mps"
                ],
                "to_kinematic_closure_error_mps": after[
                    "kinematic_closure_error_mps"
                ],
                "kinematic_closure_error_delta_mps": (
                    float(after["kinematic_closure_error_mps"])
                    - float(before["kinematic_closure_error_mps"])
                ),
                "to_kinematic_closure_relative_error": after[
                    "kinematic_closure_relative_error"
                ],
                "to_cadence_vs_high_start_ratio": after[
                    "cadence_vs_high_start_ratio"
                ],
                "to_step_length_vs_high_start_ratio": after[
                    "step_length_vs_high_start_ratio"
                ],
                "to_vx_vs_high_start_ratio": after[
                    "vx_vs_high_start_ratio"
                ],
                "response_time_s": float(after["response_time_s"]),
                "high_start_recovery_response_time_s": float(
                    after.get(
                        "high_start_recovery_response_time_s", float("nan")
                    )
                ),
                "to_corrected_contact_point_slip_mps": float(
                    after.get(
                        "steady_contact_point_tangent_slip_mps",
                        after.get("steady_contact_slip", float("nan")),
                    )
                ),
                "to_steady_abs_vy_mps": float(after["steady_abs_vy"]),
                "to_steady_mean_tilt_deg": float(
                    after.get("steady_mean_tilt_deg", float("nan"))
                ),
                "to_max_at_risk_tilt_deg": float(
                    after.get("max_at_risk_tilt_deg", float("nan"))
                ),
                "to_fall_event_count": int(after["fall_event_count"]),
            }
        )

    high_end = rows[high_end_index]
    recovery_from_low = bool(
        high_end_index > 0
        and float(rows[high_end_index - 1]["mu"]) <= low_mu_max
    )
    recovery = {
        "observed": bool(high_end_index != high_start_index and recovery_from_low),
        "high_start_phase": int(high_start["phase"]),
        "high_end_phase": int(high_end["phase"]),
        "high_end_preceded_by_low": recovery_from_low,
        "vx_recovery_ratio": float(high_end["vx_vs_high_start_ratio"]),
        "step_length_recovery_ratio": float(
            high_end["step_length_vs_high_start_ratio"]
        ),
        "stride_length_recovery_ratio": float(
            high_end["stride_length_vs_high_start_ratio"]
        ),
        "cadence_recovery_ratio": float(
            high_end["cadence_vs_high_start_ratio"]
        ),
        "response_time_s": float(
            high_end.get(
                "high_start_recovery_response_time_s",
                high_end["response_time_s"],
            )
        ),
    }
    return rows, transitions, recovery


def _build_switch_acceptance_gates(
    phase_rows: list[dict[str, object]],
    recovery: dict[str, object],
    *,
    total_falls: int,
    min_vx_recovery_ratio: float,
    min_step_recovery_ratio: float,
    max_recovery_response_s: float,
    max_tilt_deg: float,
    max_steady_abs_vy_mps: float,
    max_contact_point_slip_mps: float | None,
) -> tuple[list[tuple[str, float, bool, str]], dict[str, object]]:
    """Build adaptive-gait gates without a fixed low-friction speed target."""

    if not phase_rows:
        raise ValueError("switch acceptance requires phase rows")
    positive_limits = {
        "min_vx_recovery_ratio": min_vx_recovery_ratio,
        "min_step_recovery_ratio": min_step_recovery_ratio,
        "max_recovery_response_s": max_recovery_response_s,
        "max_tilt_deg": max_tilt_deg,
        "max_steady_abs_vy_mps": max_steady_abs_vy_mps,
    }
    if any(
        not np.isfinite(float(value)) or float(value) <= 0.0
        for value in positive_limits.values()
    ):
        raise ValueError("switch acceptance thresholds must be finite and positive")
    if max_contact_point_slip_mps is not None and (
        not np.isfinite(float(max_contact_point_slip_mps))
        or float(max_contact_point_slip_mps) <= 0.0
    ):
        raise ValueError("contact-point slip threshold must be positive when set")

    def finite_max(field: str, *, require_all: bool) -> float:
        values = np.asarray(
            [float(row.get(field, float("nan"))) for row in phase_rows],
            dtype=np.float64,
        )
        finite = np.isfinite(values)
        if require_all and not bool(finite.all()):
            return float("nan")
        return float(values[finite].max()) if bool(finite.any()) else float("nan")

    max_tilt = finite_max("max_at_risk_tilt_deg", require_all=True)
    max_lateral = finite_max("steady_abs_vy", require_all=True)
    max_slip = finite_max(
        "steady_contact_point_tangent_slip_mps",
        require_all=max_contact_point_slip_mps is not None,
    )
    vx_recovery = float(recovery.get("vx_recovery_ratio", float("nan")))
    step_recovery = float(
        recovery.get("step_length_recovery_ratio", float("nan"))
    )
    response = float(recovery.get("response_time_s", float("nan")))
    recovery_observed = bool(recovery.get("observed", False))
    gates: list[tuple[str, float, bool, str]] = [
        ("全程无摔倒", float(total_falls), int(total_falls) == 0, "= 0"),
        (
            "最大基座倾角",
            max_tilt,
            bool(np.isfinite(max_tilt) and max_tilt <= max_tilt_deg),
            f"<= {max_tilt_deg:.1f} deg",
        ),
        (
            "稳态横向速度",
            max_lateral,
            bool(
                np.isfinite(max_lateral)
                and max_lateral <= max_steady_abs_vy_mps
            ),
            f"<= {max_steady_abs_vy_mps:.3f} m/s",
        ),
        (
            "HighEnd速度恢复比",
            vx_recovery,
            bool(
                recovery_observed
                and np.isfinite(vx_recovery)
                and vx_recovery >= min_vx_recovery_ratio
            ),
            f">= {min_vx_recovery_ratio:.3f} vs HighStart",
        ),
        (
            "HighEnd步长恢复比",
            step_recovery,
            bool(
                recovery_observed
                and np.isfinite(step_recovery)
                and step_recovery >= min_step_recovery_ratio
            ),
            f">= {min_step_recovery_ratio:.3f} vs HighStart",
        ),
        (
            "HighEnd恢复响应时间",
            response,
            bool(
                recovery_observed
                and np.isfinite(response)
                and response <= max_recovery_response_s
            ),
            f"<= {max_recovery_response_s:.2f} s",
        ),
    ]
    slip_gate_enabled = max_contact_point_slip_mps is not None
    if slip_gate_enabled:
        slip_limit = float(max_contact_point_slip_mps)
        gates.append(
            (
                "校正接触点切向滑移",
                max_slip,
                bool(np.isfinite(max_slip) and max_slip <= slip_limit),
                f"<= {slip_limit:.3f} m/s",
            )
        )
    slip_diagnostic = {
        "schema": CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
        "maximum_phase_steady_mps": max_slip,
        "gate_enabled": slip_gate_enabled,
        "threshold_mps": (
            float(max_contact_point_slip_mps)
            if max_contact_point_slip_mps is not None
            else None
        ),
        "status": "calibrated_gate" if slip_gate_enabled else "diagnostic_only",
    }
    return gates, slip_diagnostic


def _first_fall_masks(
    ever_failed: torch.Tensor,
    falls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fair survival masks for one managed-environment step.

    Isaac Lab resets terminated rows inside ``env.step``.  Consequently the
    state read after a fall is already a reset state and must not enter the
    primary locomotion statistics.  An environment is permanently censored
    after its first non-timeout termination for the remainder of the rollout.

    Returns, in order: at-risk-before-step, primary-sample, first-fall,
    repeated-post-reset fall, and the updated ever-failed mask.
    """

    ever_failed = ever_failed.bool()
    falls = falls.bool()
    if ever_failed.shape != falls.shape:
        raise ValueError(
            "ever_failed and falls must have the same shape, got "
            f"{ever_failed.shape} and {falls.shape}"
        )
    at_risk = ~ever_failed
    first_fall = falls & at_risk
    repeated_fall = falls & ever_failed
    # A falling row has already been reset by the time simulator tensors are
    # read, so the failure step itself is excluded from speed/slip statistics.
    primary_sample = at_risk & ~falls
    return (
        at_risk,
        primary_sample,
        first_fall,
        repeated_fall,
        ever_failed | falls,
    )


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean over a 1-D environment mask, preserving an empty set as NaN."""

    mask = mask.bool()
    if values.shape[0] != mask.shape[0]:
        raise ValueError(
            "masked mean requires matching environment dimension, got "
            f"{values.shape[0]} and {mask.shape[0]}"
        )
    if not bool(mask.any().item()):
        return float("nan")
    return float(values[mask].float().mean().item())


def _simulator_contact_slip_metrics(
    robot,
    foot_body_ids,
    net_contact_sensor,
    foot_sensor_ids,
    left_dedicated_contact_sensor,
    right_dedicated_contact_sensor,
):
    """Return corrected contact-point slip plus the historical proxy.

    The corrected result uses each dedicated left/right ContactSensor filter's
    world contact point and normal force.  The old link-origin XY speed is
    returned under an explicitly legacy name solely for offline comparisons.
    Neither quantity is inserted into the actor observation.
    """

    foot_com_pos_w = robot.data.body_com_pos_w[:, foot_body_ids, :]
    foot_com_lin_vel_w = robot.data.body_com_lin_vel_w[:, foot_body_ids, :]
    foot_com_ang_vel_w = robot.data.body_com_ang_vel_w[:, foot_body_ids, :]
    corrected = static_ground_contact_point_tangential_speed(
        foot_com_pos_w,
        foot_com_lin_vel_w,
        foot_com_ang_vel_w,
        (
            left_dedicated_contact_sensor.data.contact_pos_w,
            right_dedicated_contact_sensor.data.contact_pos_w,
        ),
        (
            left_dedicated_contact_sensor.data.force_matrix_w,
            right_dedicated_contact_sensor.data.force_matrix_w,
        ),
        min_normal_force_n=5.0,
    )
    legacy = legacy_link_origin_planar_speed(
        foot_com_lin_vel_w,
        net_contact_sensor.data.net_forces_w[:, foot_sensor_ids, :],
        min_vertical_force_n=5.0,
    )
    return corrected, legacy


def _causal_hall_packet_validity(policy, num_envs: int, device: torch.device) -> torch.Tensor:
    """Return the two-foot packet-health decision available to deployment.

    The last four deployable observation entries in the Motion task are
    ``[valid_left, valid_right, body_vy, relative_heading]``.  This helper
    intentionally inspects only the two validity flags that were delivered to
    the actor; it does not inspect simulator contact, material, force or slip
    state.  If a policy failed to retain its exact observation, fail closed so
    a command governor cannot accidentally release motion on unknown data.
    """

    observation = getattr(policy, "last_policy_observation", None)
    if (
        observation is None
        or observation.ndim != 2
        or observation.shape[0] != num_envs
        or observation.shape[1] < 1862
    ):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return (
        torch.isfinite(observation[:, 1860:1862])
        & (observation[:, 1860:1862] > 0.5)
    ).all(dim=1)


def _update_follow_camera(env, robot) -> None:
    """Keep the recording camera centered on env-0 without affecting physics."""
    if not args_cli.camera_follow:
        return
    uenv = env.unwrapped
    try:
        root = robot.data.root_pos_w[0].detach().cpu().numpy()
        uenv.sim.set_camera_view(
            eye=root + np.asarray([3.2, -4.0, 2.25]),
            target=root + np.asarray([0.65, 0.0, 0.75]),
        )
    except (AttributeError, IndexError, RuntimeError):
        # Video is optional; a camera update must never make an evaluation fail.
        return


def _run_switch_evaluation(
    env,
    policy,
    robot,
    contact_sensor,
    left_dedicated_contact_sensor,
    right_dedicated_contact_sensor,
    foot_body_ids,
    foot_sensor_ids,
    execution_teacher=None,
    hall_governor=None,
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
    command_vy = float(args_cli.vy)
    command_wz = float(args_cli.wz)
    uenv = env.unwrapped
    n = int(uenv.num_envs)
    dt = float(uenv.step_dt)

    terrain = getattr(uenv.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_generator_cfg = getattr(
        getattr(uenv.cfg.scene, "terrain", None), "terrain_generator", None
    )
    if terrain_types is None or terrain_generator_cfg is None:
        terrain_names = ["plane"]
        terrain_types = torch.zeros(
            n, dtype=torch.long, device=uenv.device
        )
        terrain_levels = torch.zeros_like(terrain_types)
    else:
        terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
        terrain_types = terrain_types.to(device=uenv.device, dtype=torch.long)
        if terrain_levels is None:
            terrain_levels = torch.zeros_like(terrain_types)
        else:
            terrain_levels = terrain_levels.to(
                device=uenv.device, dtype=torch.long
            )

    env.reset()
    if hall_governor is not None:
        hall_governor.reset()

    _force_mu(env, sequence[0])
    _force_command(env, command_vx, command_vy, command_wz)
    ramp_steps = (
        args_cli.warmup_steps
        if args_cli.command_ramp_steps < 0
        else min(args_cli.command_ramp_steps, args_cli.warmup_steps)
    )
    if ramp_steps > 0:
        _set_command_value(env, 0.0, 0.0, 0.0)
    obs = env.get_observations()
    initial_command = uenv.command_manager.get_term(
        "base_velocity"
    ).vel_command_b[:, :3].detach().clone()
    _synchronize_evaluator_command_observation(
        env,
        obs,
        initial_command,
        torch.ones(n, dtype=torch.bool, device=uenv.device),
    )

    phase_data = []
    for index, mu in enumerate(sequence):
        phase_data.append(
            {
                "phase": index,
                "mu": mu,
                # Primary gait statistics are first-fall censored.  The
                # reset-inclusive mirrors retain the historical evaluator
                # values for diagnosis and backwards comparison.
                "vx": [],
                "vx_including_resets": [],
                "applied_vx": [],
                "risk_probability": [],
                "low_state_fraction": [],
                "abs_vy": [],
                "abs_vy_including_resets": [],
                "slip": [],
                "slip_including_resets": [],
                "slip_valid_fraction": [],
                "tilt_deg": [],
                "max_at_risk_tilt_deg": [],
                "fn": [],
                "ft": [],
                # Simulator-only diagnostics for the magnetic forward model.
                # These values are never part of the deployable policy input;
                # they verify that the embedded-TPU model is actually driven
                # by both normal loading and local shear during evaluation.
                "hall_driver_normal": [],
                "hall_driver_shear": [],
                "early_slip": [],
                "steady_slip": [],
                "touchdowns": 0.0,
                "stride_sum": 0.0,
                "stride_count": 0,
                # ``falls`` remains as a compatibility alias for the phase's
                # complete fall-event count.
                "falls": 0,
                "fall_event_count": 0,
                "unique_env_first_fall_count": 0,
                "time_to_first_fall_s": float("nan"),
                "failure_free_exposure_s": 0.0,
                "steady_failure_free_exposure_s": 0.0,
                "post_reset_count": 0,
                "post_reset_sample_count": 0,
                "failure_free_sample_count": 0,
                "terrain_vx": {name: [] for name in terrain_names},
                "terrain_vx_including_resets": {
                    name: [] for name in terrain_names
                },
                "terrain_slip": {name: [] for name in terrain_names},
                "terrain_slip_including_resets": {
                    name: [] for name in terrain_names
                },
                "terrain_risk": {name: [] for name in terrain_names},
                "terrain_falls": {name: 0 for name in terrain_names},
                "terrain_first_falls": {name: 0 for name in terrain_names},
                "env_falls": np.zeros(n, dtype=np.int32),
                "env_first_fall": np.zeros(n, dtype=np.bool_),
            }
        )
    time_rows = []
    hall_trace_time = []
    hall_trace_phase = []
    hall_trace_mu = []
    hall_trace_raw = []
    hall_trace_delta = []
    hall_trace_baseline = []
    hall_trace_valid = []
    dagger_policy_obs = []
    dagger_teacher_obs = []
    dagger_mu = []
    dagger_cmd = []
    dagger_weight = []
    dagger_phase = []
    dagger_time_since_switch = []
    dagger_env_id = []
    dagger_seed = []
    dagger_terrain_type = []
    dagger_terrain_level = []
    # ``--collect_npz`` is also supported in switch mode.  It is deliberately
    # kept separate from the DAgger export above: it records only the exact
    # deployable Hall/proprioception input plus simulator-only labels for an
    # offline traction-state or future-slip estimator.  No contact force,
    # friction value, or slip label is inserted into the policy observation.
    switch_collected_obs = []
    switch_collected_mu = []
    switch_collected_cmd = []
    switch_collected_seed = []
    switch_collected_root_lin_vel_b = []
    switch_collected_root_ang_vel_b = []
    switch_collected_actor_command = []
    switch_collected_applied_command = []
    switch_collected_contact_slip = []
    switch_collected_contact_slip_valid = []
    switch_collected_legacy_link_origin_planar_slip = []
    switch_collected_valid = []
    switch_collected_env_id = []
    switch_collected_step = []
    switch_collected_fall = []
    switch_collected_done = []
    switch_collected_time_out = []
    switch_collected_hall_valid_lr = []
    switch_collected_phase = []
    switch_collected_time_since_switch = []
    switch_collected_rollout_id = []
    # Managed environments reset a fallen row inside ``env.step``.  Preserve
    # a per-row episode segment so future-slip labels never join samples from
    # before and after that reset.  Phase index is folded into the saved id
    # below, which also prevents a target horizon from looking through a
    # discontinuous material switch.
    switch_collection_episode_generation = torch.zeros(
        n, dtype=torch.int64, device=uenv.device
    )
    # Global first-fall survival accounting spans warm-up and every material
    # phase.  A warm-up failure therefore permanently removes that row from
    # the primary gait statistics instead of silently reintroducing its reset
    # state at phase zero.
    ever_failed = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    first_fall_time_by_env_s = torch.full(
        (n,), float("nan"), dtype=torch.float32, device=uenv.device
    )
    fall_event_count = 0
    unique_env_first_fall_count = 0
    failure_free_exposure_s = 0.0
    post_reset_count = 0
    post_reset_sample_count = 0
    warmup_fall_event_count = 0
    warmup_unique_env_first_fall_count = 0
    warmup_failure_free_exposure_s = 0.0

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
        command_fraction = 1.0
        if step < ramp_steps:
            command_fraction = float(step + 1) / float(ramp_steps)
        requested_vx = command_vx * command_fraction
        requested_vy = command_vy * command_fraction
        requested_wz = command_wz * command_fraction
        # Custom command generators may ignore cfg.ranges when an environment
        # resets or its resampling timer expires.  Reassert the evaluator's
        # exact request every step; normal observation history is repaired
        # only after env.step and only where the generator actually changed it.
        _set_command_value(
            env, requested_vx, requested_vy, requested_wz
        )
        requested_command_this_step = (
            uenv.command_manager.get_term("base_velocity")
            .vel_command_b[:, :3]
            .detach()
            .clone()
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

        _update_follow_camera(env, robot)

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
        exact_actor_observation = _exact_actor_policy_observation(
            policy, policy_obs
        ).detach().clone()
        actor_command_this_step = exact_actor_observation[:, 42:45].clone()
        governed_command = None
        low_probability = None
        governor_state = None
        if hall_governor is not None:
            low_probability = getattr(
                policy, "last_low_traction_probability", None
            )
            if low_probability is None:
                raise RuntimeError(
                    "Hall Student did not expose low-traction probability"
                )
            requested = torch.zeros((n, 3), device=uenv.device)
            requested[:, 0] = requested_vx
            requested[:, 1] = requested_vy
            requested[:, 2] = requested_wz
            hall_valid = _causal_hall_packet_validity(policy, n, uenv.device)
            governed_command, governor_state = hall_governor.update(
                requested, low_probability, valid=hall_valid
            )
            prebrake_active = hall_governor.prebrake_active
            if hasattr(policy, "recovery_mask"):
                policy.recovery_mask = (
                    (governor_state == LOW) & ~hall_governor.probing
                ).detach().clone()
            term = uenv.command_manager.get_term("base_velocity")
            term.is_standing_env[:] = False
            term.vel_command_b[:, :3] = governed_command
            # Optional hybrid action path.  It is deliberately downstream of
            # the causal governor update: the original actor is retained in
            # HIGH and whenever Hall packets are invalid; the bounded Hall
            # Recovery action is used only after a valid LOW decision.  The
            # optional *same-actor* pre-brake reflex is the sole HIGH-state
            # exception, and requires a calibrated Hall-risk jump relative
            # to the settled walking reference.  The
            # initial active-sensing probe deliberately retains the audited
            # original action by default: applying a low-traction expert
            # before the probe has classified a high-traction surface causes
            # an avoidable nominal-gait perturbation.  Packet loss is always
            # an exact proprioceptive fallback.
            recovery_action = getattr(policy, "last_recovery_action", None)
            base_action = getattr(policy, "last_base_action", None)
            if base_action is not None and (
                recovery_action is not None
                or args_cli.hall_governed_command_reflex
            ):
                actor_obs_for_validity = getattr(
                    policy, "last_policy_observation", None
                )
                if actor_obs_for_validity is None:
                    hall_valid = torch.zeros(
                        n, dtype=torch.bool, device=uenv.device
                    )
                else:
                    hall_valid = (
                        torch.isfinite(actor_obs_for_validity[:, 1860:1862])
                        & (actor_obs_for_validity[:, 1860:1862] > 0.5)
                    ).all(dim=1)
                # Blend by the risk probability instead of hard-switching
                # 29 joint targets at a material transition.  A hard action
                # discontinuity was the remaining source of isolated falls
                # in the paired μ=0.2 test.  Crucially, the probability is
                # *not* by itself an action-switch signal: only a causal LOW
                # decision can enable the recovery branch.  This preserves
                # the audited original actor throughout HIGH/UNKNOWN and
                # prevents the recovery action from changing Hall evidence
                # before the bounded probe has classified the surface.
                # Invalid/stale Hall is an exact baseline path.
                blend_start = float(args_cli.hall_hybrid_blend_start)
                blend_full = float(args_cli.hall_hybrid_blend_full)
                if not 0.0 <= blend_start < blend_full <= 1.0:
                    raise ValueError(
                        "hall hybrid blend thresholds must satisfy "
                        "0 <= start < full <= 1"
                    )
                low_blend_floor = float(args_cli.hall_recovery_low_blend_floor)
                if not 0.0 <= low_blend_floor <= 1.0:
                    raise ValueError(
                        "--hall_recovery_low_blend_floor must be in [0,1]"
                    )
                risk_alpha = (
                    (low_probability - blend_start)
                    / max(blend_full - blend_start, 1.0e-6)
                ).clamp(0.0, 1.0)
                low_state = governor_state == LOW
                prebrake_state = prebrake_active & ~hall_governor.probing
                # In default operation a raw Hall score never changes the
                # joint target while the state machine is HIGH or UNKNOWN.
                # A tested LOW decision retains enough recovery authority to
                # matter even if the score falls after braking begins.
                risk_alpha = torch.where(
                    low_state,
                    torch.maximum(
                        risk_alpha,
                        risk_alpha.new_full(risk_alpha.shape, low_blend_floor),
                    ),
                    torch.zeros_like(risk_alpha),
                )
                prebrake_reflex_floor = float(
                    args_cli.governor_prebrake_reflex_floor
                )
                if not 0.0 <= prebrake_reflex_floor <= 1.0:
                    raise ValueError(
                        "--governor_prebrake_reflex_floor must be in [0,1]"
                    )
                # Pre-brake keeps the learned recovery actor out of HIGH,
                # but lets the *same audited base actor* immediately see the
                # newly clipped command history.  This removes the history
                # lag during the brief causal early-brake window without
                # leaking force, friction, or slip labels into the actor.
                reflex_alpha = torch.where(
                    prebrake_state,
                    torch.maximum(
                        risk_alpha,
                        risk_alpha.new_full(
                            risk_alpha.shape, prebrake_reflex_floor
                        ),
                    ),
                    risk_alpha,
                )
                probe_recovery = hall_governor.probing & bool(
                    args_cli.hall_recovery_on_probe
                )
                risk_alpha = torch.where(
                    probe_recovery,
                    torch.ones_like(risk_alpha),
                    risk_alpha,
                )
                reflex_alpha = torch.where(
                    probe_recovery,
                    torch.ones_like(reflex_alpha),
                    reflex_alpha,
                )
                risk_alpha = torch.where(
                    hall_valid,
                    risk_alpha,
                    torch.zeros_like(risk_alpha),
                )
                reflex_alpha = torch.where(
                    hall_valid,
                    reflex_alpha,
                    torch.zeros_like(reflex_alpha),
                )
                # The optional command reflex is an immediate re-evaluation
                # of the same audited actor, not a learned force/friction
                # inverse.  It is gated by confirmed LOW/probe states, or by
                # the explicit structural Hall-only pre-brake; it cannot
                # modify ordinary nominal HIGH walking.
                reflex_action = None
                if (
                    args_cli.hall_governed_command_reflex
                    and torch.any(reflex_alpha > 0.0)
                ):
                    reflex_fn = getattr(
                        policy, "governed_command_reflex_action", None
                    )
                    if reflex_fn is None:
                        raise RuntimeError(
                            "--hall_governed_command_reflex requires the "
                            "anchored shared ONNX actor path"
                        )
                    reflex_action = reflex_fn(governed_command)
                # A Hall-conditioned posture residual can itself alter the
                # Hall trajectory.  Leaving it active indefinitely after a
                # real high-friction recovery can therefore create a feedback
                # loop: residual -> altered gait -> conservative risk ->
                # residual.  When requested, use it only as a short bounded
                # correction after entering LOW (or, only if explicitly
                # requested, during an expert probe),
                # while the independent command limiter remains conservative
                # for as long as risk requires.  This uses only the governor's
                # causal internal clock, never friction/contact truth.
                recovery_max_low_s = float(args_cli.hall_recovery_max_low_s)
                if recovery_max_low_s < 0.0:
                    raise ValueError("--hall_recovery_max_low_s must be >= 0")
                if recovery_max_low_s > 0.0:
                    recovery_window = probe_recovery | (
                        low_state
                        & (hall_governor.low_state_time_s <= recovery_max_low_s)
                    )
                    risk_alpha = torch.where(
                        recovery_window,
                        risk_alpha,
                        torch.zeros_like(risk_alpha),
                    )
                    reflex_alpha = torch.where(
                        recovery_window | prebrake_state,
                        reflex_alpha,
                        torch.zeros_like(reflex_alpha),
                    )
                actions = base_action
                if reflex_action is not None:
                    actions = torch.lerp(
                        actions, reflex_action, reflex_alpha[:, None]
                    )
                if recovery_action is not None:
                    # The recovery actor was trained at the governed
                    # low-traction command envelope.  Its ordinary input has
                    # a five-frame command history, so directly blending the
                    # action computed just before a high->low transition
                    # would feed it the stale high-speed request for roughly
                    # one history window.  Re-evaluate it below with the
                    # already-causal governor output, exactly as the optional
                    # original-actor reflex does.  This is a command-history
                    # correction only: Hall/proprioception remain the sole
                    # runtime evidence and no contact/friction/slip truth is
                    # introduced into either actor.
                    recovery_reflex_fn = getattr(
                        policy, "governed_command_recovery_action", None
                    )
                    if recovery_reflex_fn is not None and torch.any(
                        risk_alpha > 0.0
                    ):
                        recovery_action = recovery_reflex_fn(governed_command)
                actions = torch.lerp(
                    actions, recovery_action, risk_alpha[:, None]
                )
        applied_command_this_step = (
            uenv.command_manager.get_term("base_velocity")
            .vel_command_b[:, :3]
            .detach()
            .clone()
        )
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
            dagger_env_id.append(np.arange(n, dtype=np.int32))
            dagger_seed.append(np.full(n, args_cli.seed, dtype=np.int32))
            dagger_terrain_type.append(
                terrain_types.detach().cpu().numpy().astype(np.int16)
            )
            dagger_terrain_level.append(
                terrain_levels.detach().cpu().numpy().astype(np.int16)
            )
        switch_collection_pre_obs = None
        if (
            args_cli.collect_npz is not None
            and phase_index >= 0
            # The first observation after forcing a different material still
            # represents the previous physics step.  Start at local step one
            # so Hall history and the μ label are causally aligned.
            and local_step > 0
        ):
            exact_policy_obs = exact_actor_observation
            if exact_policy_obs.shape[1] != 1864:
                raise ValueError(
                    "switch --collect_npz requires the deployable 1864-D "
                    f"policy observation, got {exact_policy_obs.shape}"
                )
            # ``env.step`` may reset a terminated environment before it
            # returns, therefore the pre-step tensor is the only valid causal
            # input for a fall label generated by this physics step.
            switch_collection_pre_obs = exact_policy_obs.detach().clone()
        # Read attitude before stepping.  A terminating Isaac environment is
        # reset inside env.step(), so a post-step read would replace the
        # near-fall attitude with a fresh upright state and under-report risk.
        pre_step_tilt_deg = _projected_gravity_tilt_degrees(
            robot.data.projected_gravity_b
        ).detach().clone()
        obs, _, dones, extras = env.step(actions)

        # Capture after the physics/sensor update so every row describes the
        # same state as the recorded Isaac frame.  Hall observables stay in
        # their native contract: absolute raw B, auto-zeroed dB, baseline and
        # channel validity.  Contact force/slip truth is deliberately absent.
        if args_cli.hall_trace_npz is not None:
            trace_env_id = int(args_cli.hall_trace_env_id)
            if not 0 <= trace_env_id < n:
                raise ValueError(
                    f"--hall_trace_env_id={trace_env_id} outside [0,{n})"
                )
            hall_sensor = getattr(uenv, "_hall_foot_sensor", None)
            if hall_sensor is None:
                raise RuntimeError(
                    "--hall_trace_npz requested but the task has no HallFootSensor"
                )
            hall_debug = hall_sensor.get_debug_data()
            hall_trace_time.append((step + 1) * dt)
            hall_trace_phase.append(phase_index)
            hall_trace_mu.append(
                sequence[0] if phase_index < 0 else sequence[phase_index]
            )
            hall_trace_raw.append(
                hall_debug["raw_magnetic_field"][trace_env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            hall_trace_delta.append(
                hall_debug["magnetic_delta"][trace_env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            hall_trace_baseline.append(
                hall_debug["zero_load_baseline"][trace_env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            hall_trace_valid.append(
                hall_debug["valid_mask"][trace_env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        managed_resets = dones.bool()
        post_step_command = applied_command_this_step.clone()
        post_step_command[managed_resets] = requested_command_this_step[
            managed_resets
        ]
        _synchronize_evaluator_command_observation(
            env,
            obs,
            post_step_command,
            managed_resets,
        )
        timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
        if timeouts is None:
            timeout_mask = torch.zeros_like(managed_resets)
        else:
            timeout_mask = timeouts.to(device=dones.device).bool()
        falls = managed_resets & ~timeout_mask
        (
            at_risk_before_step,
            primary_sample_mask,
            first_falls,
            repeated_falls,
            ever_failed,
        ) = _first_fall_masks(ever_failed, falls)
        step_exposure_s = float(at_risk_before_step.sum().item()) * dt
        failure_free_exposure_s += step_exposure_s
        step_fall_events = int(falls.sum().item())
        step_first_falls = int(first_falls.sum().item())
        step_repeated_falls = int(repeated_falls.sum().item())
        fall_event_count += step_fall_events
        unique_env_first_fall_count += step_first_falls
        post_reset_count += step_repeated_falls
        # Isaac returns the managed-reset state on a falling step.  Count that
        # excluded row, plus every subsequent sample from an already failed
        # environment, so reset contamination remains directly auditable.
        post_reset_sample_count += int((~primary_sample_mask).sum().item())
        if first_falls.any():
            first_fall_time_by_env_s[first_falls] = float(step + 1) * dt
        if phase_index < 0:
            warmup_failure_free_exposure_s += step_exposure_s
            warmup_fall_event_count += step_fall_events
            warmup_unique_env_first_fall_count += step_first_falls
            # Warm-up failures count toward the release gate.  They are not
            # exported as learning samples, but the managed environment has
            # still reset that row.  Advance its segment so collection cannot
            # join histories across the discontinuity.
            if managed_resets.any():
                switch_collection_episode_generation[managed_resets] += 1
                # Keep the causal governor aligned with Isaac's managed reset.
                # Leaving an old HIGH/LOW state and slew-limited command on a
                # newly reset robot caused an avoidable first-phase transient.
                if hall_governor is not None:
                    hall_governor.reset(
                        torch.nonzero(managed_resets, as_tuple=False).flatten()
                    )
            continue
        data = phase_data[phase_index]
        data["falls"] += step_fall_events
        data["fall_event_count"] += step_fall_events
        data["unique_env_first_fall_count"] += step_first_falls
        data["failure_free_exposure_s"] += step_exposure_s
        data["post_reset_count"] += step_repeated_falls
        data["post_reset_sample_count"] += int(
            (~primary_sample_mask).sum().item()
        )
        data["failure_free_sample_count"] += int(
            primary_sample_mask.sum().item()
        )
        if step_first_falls > 0 and not np.isfinite(
            data["time_to_first_fall_s"]
        ):
            data["time_to_first_fall_s"] = float(local_step + 1) * dt
        data["env_falls"] += falls.detach().cpu().numpy().astype(np.int32)
        data["env_first_fall"] |= (
            first_falls.detach().cpu().numpy().astype(np.bool_)
        )
        for terrain_index, terrain_name in enumerate(terrain_names):
            terrain_mask = terrain_types == terrain_index
            data["terrain_falls"][terrain_name] += int(
                (falls & terrain_mask).sum().item()
            )
            data["terrain_first_falls"][terrain_name] += int(
                (first_falls & terrain_mask).sum().item()
            )
        if hall_governor is not None and managed_resets.any():
            hall_governor.reset(
                torch.nonzero(managed_resets, as_tuple=False).flatten()
            )

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
        corrected_slip, legacy_slip = _simulator_contact_slip_metrics(
            robot,
            foot_body_ids,
            contact_sensor,
            foot_sensor_ids,
            left_dedicated_contact_sensor,
            right_dedicated_contact_sensor,
        )
        contact_slip_per_env = corrected_slip.speed_per_env
        contact_slip_valid_per_env = corrected_slip.valid_per_env
        legacy_link_origin_planar_slip_per_env = legacy_slip.speed_per_env
        if switch_collection_pre_obs is not None:
            scheduled = local_step % max(args_cli.collect_stride, 1) == 0
            selected = (
                torch.arange(n, device=uenv.device)
                if scheduled
                else torch.nonzero(falls, as_tuple=False).flatten()
            )
            if selected.numel() > 0:
                selected_count = int(selected.numel())
                switch_collected_obs.append(
                    switch_collection_pre_obs.index_select(0, selected)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_mu.append(
                    np.full(selected_count, sequence[phase_index], dtype=np.float32)
                )
                switch_collected_cmd.append(
                    np.full(selected_count, command_vx, dtype=np.float32)
                )
                switch_collected_seed.append(
                    np.full(selected_count, args_cli.seed, dtype=np.int32)
                )
                switch_collected_root_lin_vel_b.append(
                    robot.data.root_lin_vel_b.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_root_ang_vel_b.append(
                    robot.data.root_ang_vel_b.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_actor_command.append(
                    actor_command_this_step.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_applied_command.append(
                    applied_command_this_step.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_contact_slip.append(
                    contact_slip_per_env.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_contact_slip_valid.append(
                    contact_slip_valid_per_env.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                )
                switch_collected_legacy_link_origin_planar_slip.append(
                    legacy_link_origin_planar_slip_per_env.index_select(
                        0, selected
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                # A fall describes the preceding observation, not a missing
                # Hall packet.  Retain it for causal supervision.
                switch_collected_valid.append(
                    np.ones(selected_count, dtype=np.bool_)
                )
                switch_collected_env_id.append(
                    selected.detach().cpu().numpy().astype(np.int32)
                )
                switch_collected_step.append(
                    np.full(selected_count, local_step, dtype=np.int32)
                )
                switch_collected_fall.append(
                    falls.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                )
                switch_collected_done.append(
                    managed_resets.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                )
                switch_collected_time_out.append(
                    timeout_mask.index_select(0, selected)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                )
                # Packet health is copied from the exact pre-step tensor
                # consumed by the actor, never reconstructed from simulator
                # sensor buffers after the physics step/reset.
                switch_collected_hall_valid_lr.append(
                    switch_collection_pre_obs.index_select(0, selected)[
                        :, 1860:1862
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                switch_collected_phase.append(
                    np.full(selected_count, phase_index, dtype=np.int16)
                )
                switch_collected_time_since_switch.append(
                    np.full(
                        selected_count,
                        (local_step + 1) * dt,
                        dtype=np.float32,
                    )
                )
                # Different material phases and reset segments are distinct
                # causal trajectories.  The training loader combines this id
                # with the env id, so it remains valid for batched rollout
                # collection as well.
                switch_collected_rollout_id.append(
                    (
                        phase_index * 1_000_000
                        + switch_collection_episode_generation.index_select(
                            0, selected
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
        mean_vx_including_resets = float(vel[:, 0].mean().item())
        mean_abs_vy_including_resets = float(
            torch.abs(vel[:, 1]).mean().item()
        )
        mean_slip_including_resets = _masked_tensor_mean(
            contact_slip_per_env, contact_slip_valid_per_env
        )
        mean_vx = _masked_tensor_mean(vel[:, 0], primary_sample_mask)
        mean_abs_vy = _masked_tensor_mean(
            torch.abs(vel[:, 1]), primary_sample_mask
        )
        mean_slip = _masked_tensor_mean(
            contact_slip_per_env,
            primary_sample_mask & contact_slip_valid_per_env,
        )
        primary_count = int(primary_sample_mask.sum().item())
        slip_valid_fraction = (
            float(
                (primary_sample_mask & contact_slip_valid_per_env)
                .sum()
                .item()
            )
            / primary_count
            if primary_count > 0
            else float("nan")
        )
        mean_tilt_deg = _masked_tensor_mean(
            pre_step_tilt_deg, primary_sample_mask
        )
        max_at_risk_tilt_deg = (
            float(pre_step_tilt_deg[at_risk_before_step].max().item())
            if bool(at_risk_before_step.any().item())
            else float("nan")
        )
        mean_fn = _masked_tensor_mean(fn.sum(dim=1), primary_sample_mask)
        mean_ft = _masked_tensor_mean(ft.sum(dim=1), primary_sample_mask)
        hall_sensor = getattr(uenv, "_hall_foot_sensor", None)
        if hall_sensor is None:
            mean_hall_driver_normal = float("nan")
            mean_hall_driver_shear = float("nan")
        else:
            hall_driver = hall_sensor.get_debug_data()[
                "mechanical_driver_force_privileged"
            ]
            # Local forces have already been distributed over the 15 Hall
            # regions.  Sum the regions first to recover the two foot loads,
            # then average across environments and feet.
            hall_driver_foot = hall_driver.sum(dim=2)
            mean_hall_driver_normal = _masked_tensor_mean(
                torch.abs(hall_driver_foot[..., 2]).mean(dim=1),
                primary_sample_mask,
            )
            mean_hall_driver_shear = _masked_tensor_mean(
                torch.linalg.vector_norm(
                    hall_driver_foot[..., :2], dim=-1
                ).mean(dim=1),
                primary_sample_mask,
            )
        data["vx"].append(mean_vx)
        data["vx_including_resets"].append(mean_vx_including_resets)
        if governed_command is not None:
            applied_vx = _masked_tensor_mean(
                governed_command[:, 0], primary_sample_mask
            )
        else:
            applied_vx = (
                float(requested_vx)
                if bool(primary_sample_mask.any().item())
                else float("nan")
            )
        data["applied_vx"].append(
            applied_vx
        )
        data["risk_probability"].append(
            _masked_tensor_mean(low_probability, primary_sample_mask)
            if low_probability is not None
            else float("nan")
        )
        data["low_state_fraction"].append(
            _masked_tensor_mean(
                (governor_state == LOW).float(), primary_sample_mask
            )
            if governor_state is not None
            else 0.0
        )
        data["abs_vy"].append(mean_abs_vy)
        data["abs_vy_including_resets"].append(
            mean_abs_vy_including_resets
        )
        data["slip"].append(mean_slip)
        data["slip_including_resets"].append(
            mean_slip_including_resets
        )
        data["slip_valid_fraction"].append(slip_valid_fraction)
        data["tilt_deg"].append(mean_tilt_deg)
        data["max_at_risk_tilt_deg"].append(max_at_risk_tilt_deg)
        data["fn"].append(mean_fn)
        data["ft"].append(mean_ft)
        data["hall_driver_normal"].append(mean_hall_driver_normal)
        data["hall_driver_shear"].append(mean_hall_driver_shear)
        for terrain_index, terrain_name in enumerate(terrain_names):
            terrain_mask = terrain_types == terrain_index
            if not terrain_mask.any():
                continue
            terrain_primary_mask = terrain_mask & primary_sample_mask
            data["terrain_vx"][terrain_name].append(
                _masked_tensor_mean(vel[:, 0], terrain_primary_mask)
            )
            data["terrain_vx_including_resets"][terrain_name].append(
                float(vel[terrain_mask, 0].mean().item())
            )
            data["terrain_slip"][terrain_name].append(
                _masked_tensor_mean(
                    contact_slip_per_env,
                    terrain_primary_mask & contact_slip_valid_per_env,
                )
            )
            data["terrain_slip_including_resets"][terrain_name].append(
                _masked_tensor_mean(
                    contact_slip_per_env,
                    terrain_mask & contact_slip_valid_per_env,
                )
            )
            data["terrain_risk"][terrain_name].append(
                _masked_tensor_mean(low_probability, terrain_primary_mask)
                if low_probability is not None
                else float("nan")
            )
        if local_step < max(int(round(0.50 / dt)), 1):
            data["early_slip"].append(mean_slip)
        if local_step >= args_cli.switch_settle_steps:
            data["steady_slip"].append(mean_slip)
            data["steady_failure_free_exposure_s"] += (
                float(primary_sample_mask.sum().item()) * dt
            )

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
                & primary_sample_mask[:, None]
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
                "cmd_vy": command_vy,
                "cmd_wz": command_wz,
                "applied_vx_command": data["applied_vx"][-1],
                "low_traction_probability": data["risk_probability"][-1],
                "low_state_fraction": data["low_state_fraction"][-1],
                "mean_vx": mean_vx,
                "mean_vx_including_resets": mean_vx_including_resets,
                "mean_abs_vy": mean_abs_vy,
                "mean_abs_vy_including_resets": (
                    mean_abs_vy_including_resets
                ),
                "mean_contact_slip": mean_slip,
                "mean_contact_slip_including_resets": (
                    mean_slip_including_resets
                ),
                "mean_contact_point_tangent_slip_mps": mean_slip,
                "contact_point_slip_valid_fraction": slip_valid_fraction,
                "mean_base_tilt_deg": mean_tilt_deg,
                "max_at_risk_base_tilt_deg": max_at_risk_tilt_deg,
                "mean_foot_fn": mean_fn,
                "mean_foot_ft": mean_ft,
                "mean_hall_driver_normal": mean_hall_driver_normal,
                "mean_hall_driver_shear": mean_hall_driver_shear,
                "failure_free_env_count": int(
                    primary_sample_mask.sum().item()
                ),
                "fall_event_count": fall_event_count,
                "unique_env_first_fall_count": (
                    unique_env_first_fall_count
                ),
                "failure_free_exposure_s": failure_free_exposure_s,
                "post_reset_count": post_reset_count,
                "post_reset_sample_count": post_reset_sample_count,
                "falls_cumulative": data["falls"],
            }
        )
        if managed_resets.any():
            switch_collection_episode_generation[managed_resets] += 1

    phase_rows = []
    previous_steady = None
    steady_start = args_cli.switch_settle_steps
    for data in phase_data:
        steady_vx_values = data["vx"][steady_start:]
        steady_vx = _finite_mean(steady_vx_values)
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
            data["steady_failure_free_exposure_s"], 1.0e-6
        )
        early_slip = _finite_mean(data["early_slip"])
        steady_slip = _finite_mean(data["steady_slip"])
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
                "steady_applied_vx_command": _finite_mean(
                    data["applied_vx"][steady_start:]
                ),
                "steady_low_traction_probability": _finite_mean(
                    data["risk_probability"][steady_start:]
                ),
                "steady_low_state_fraction": _finite_mean(
                    data["low_state_fraction"][steady_start:]
                ),
                "steady_vx": steady_vx,
                "steady_vx_including_resets": _finite_mean(
                    data["vx_including_resets"][steady_start:]
                ),
                "steady_abs_vy": _finite_mean(
                    data["abs_vy"][steady_start:]
                ),
                "steady_abs_vy_including_resets": _finite_mean(
                    data["abs_vy_including_resets"][steady_start:]
                ),
                "steady_contact_slip": steady_slip,
                # Compatibility field above remains unchanged; the explicit
                # name below identifies the corrected contact-point metric.
                "steady_contact_point_tangent_slip_mps": steady_slip,
                "steady_contact_point_tangent_slip_valid_fraction": (
                    _finite_mean(data["slip_valid_fraction"][steady_start:])
                ),
                "steady_contact_slip_including_resets": _finite_mean(
                    data["slip_including_resets"][steady_start:]
                ),
                "steady_mean_tilt_deg": _finite_mean(
                    data["tilt_deg"][steady_start:]
                ),
                "max_at_risk_tilt_deg": _finite_max(
                    data["max_at_risk_tilt_deg"]
                ),
                "steady_hall_driver_normal": _finite_mean(
                    data["hall_driver_normal"][steady_start:]
                ),
                "steady_hall_driver_shear": _finite_mean(
                    data["hall_driver_shear"][steady_start:]
                ),
                "early_contact_slip": early_slip,
                "slip_reduction_fraction": slip_reduction,
                "step_frequency_hz": step_frequency,
                "mean_stride_length_m": stride,
                "mean_step_length_m": 0.5 * stride,
                "response_time_s": response,
                "high_start_recovery_response_time_s": float("nan"),
                "falls": data["falls"],
                "fall_event_count": data["fall_event_count"],
                "unique_env_first_fall_count": data[
                    "unique_env_first_fall_count"
                ],
                "time_to_first_fall_s": data["time_to_first_fall_s"],
                "failure_free_exposure_s": data[
                    "failure_free_exposure_s"
                ],
                "post_reset_count": data["post_reset_count"],
                "post_reset_sample_count": data[
                    "post_reset_sample_count"
                ],
                "failure_free_sample_count": data[
                    "failure_free_sample_count"
                ],
            }
        )
        previous_steady = steady_vx

    high_phase_indices = [
        index for index, row in enumerate(phase_rows) if row["mu"] >= 0.75
    ]
    if high_phase_indices:
        high_start_index = high_phase_indices[0]
        high_end_index = high_phase_indices[-1]
        high_end_preceded_by_low = bool(
            high_end_index > 0 and phase_rows[high_end_index - 1]["mu"] <= 0.25
        )
        if high_end_index != high_start_index and high_end_preceded_by_low:
            phase_rows[high_end_index][
                "high_start_recovery_response_time_s"
            ] = _high_start_recovery_response_time(
                phase_data[high_end_index]["vx"],
                abs(float(phase_rows[high_start_index]["steady_vx"])),
                args_cli.switch_min_high_end_vx_recovery_ratio,
                dt,
                args_cli.switch_response_window_steps,
            )

    phase_rows, transition_rows, recovery = _build_switch_gait_diagnostics(
        phase_rows
    )

    terrain_rows = []
    terrain_env_rows = []
    for data in phase_data:
        for terrain_index, terrain_name in enumerate(terrain_names):
            mask = terrain_types == terrain_index
            if not mask.any():
                continue
            levels = terrain_levels[mask]
            terrain_rows.append(
                {
                    "phase": data["phase"],
                    "mu": data["mu"],
                    "terrain_type": terrain_name,
                    "num_envs": int(mask.sum().item()),
                    "mean_level": float(levels.float().mean().item()),
                    "steady_vx": _finite_mean(
                        data["terrain_vx"][terrain_name][steady_start:]
                    ),
                    "steady_vx_including_resets": _finite_mean(
                        data["terrain_vx_including_resets"][terrain_name][
                            steady_start:
                        ]
                    ),
                    "steady_contact_slip": _finite_mean(
                        data["terrain_slip"][terrain_name][steady_start:]
                    ),
                    "steady_contact_slip_including_resets": _finite_mean(
                        data["terrain_slip_including_resets"][terrain_name][
                            steady_start:
                        ]
                    ),
                    "steady_low_traction_probability": _finite_mean(
                        data["terrain_risk"][terrain_name][steady_start:]
                    ),
                    "falls": data["terrain_falls"][terrain_name],
                    "unique_env_first_fall_count": data[
                        "terrain_first_falls"
                    ][terrain_name],
                }
            )
            for env_id in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                terrain_env_rows.append(
                    {
                        "phase": data["phase"],
                        "mu": data["mu"],
                        "terrain_type": terrain_name,
                        "env_id": int(env_id),
                        "terrain_level": int(terrain_levels[env_id].item()),
                        "falls": int(data["env_falls"][env_id]),
                        "first_fall_in_phase": bool(
                            data["env_first_fall"][env_id]
                        ),
                        "first_fall_time_from_rollout_start_s": float(
                            first_fall_time_by_env_s[env_id].item()
                        ),
                    }
                )

    low_rows = [row for row in phase_rows if row["mu"] <= 0.25]
    high_rows = [row for row in phase_rows if row["mu"] >= 0.75]
    low_vx = _finite_mean([row["steady_vx"] for row in low_rows])
    high_vx = _finite_mean([row["steady_vx"] for row in high_rows])
    low_cadence = _finite_mean(
        [row["step_frequency_hz"] for row in low_rows]
    )
    high_cadence = _finite_mean(
        [row["step_frequency_hz"] for row in high_rows]
    )
    low_step = _finite_mean([row["mean_step_length_m"] for row in low_rows])
    high_step = _finite_mean([row["mean_step_length_m"] for row in high_rows])
    # This total deliberately includes warm-up failures, unlike the legacy
    # sum of per-phase rows.
    total_falls = fall_event_count
    finite_first_fall_times = first_fall_time_by_env_s[
        torch.isfinite(first_fall_time_by_env_s)
    ]
    time_to_first_fall_s = (
        float(finite_first_fall_times.min().item())
        if finite_first_fall_times.numel() > 0
        else float("nan")
    )
    gates, slip_diagnostic = _build_switch_acceptance_gates(
        phase_rows,
        recovery,
        total_falls=total_falls,
        min_vx_recovery_ratio=(
            args_cli.switch_min_high_end_vx_recovery_ratio
        ),
        min_step_recovery_ratio=(
            args_cli.switch_min_high_end_step_length_recovery_ratio
        ),
        max_recovery_response_s=args_cli.switch_max_response_s,
        max_tilt_deg=args_cli.switch_max_tilt_deg,
        max_steady_abs_vy_mps=args_cli.switch_max_steady_abs_vy_mps,
        max_contact_point_slip_mps=(
            args_cli.switch_max_contact_point_slip_mps
        ),
    )
    # Preserve the historical fixed-slowdown comparisons for longitudinal
    # studies, but do not use them as release gates.  A safe low-mu gait may
    # maintain speed through higher cadence and shorter steps.
    legacy_diagnostics = [
        ("历史：高-低稳态速度差", high_vx - low_vx, "diagnostic"),
        ("历史：低摩擦稳态速度", low_vx, "diagnostic"),
        ("历史：高-低步频差", high_cadence - low_cadence, "diagnostic"),
        ("历史：高-低步长差", high_step - low_step, "diagnostic"),
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
    transition_csv = output_csv.with_name(
        f"{output_csv.stem}.transitions{output_csv.suffix}"
    )
    with transition_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)
    time_csv = output_csv.with_name(
        f"{output_csv.stem}.timeseries{output_csv.suffix}"
    )
    with time_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(time_rows[0]))
        writer.writeheader()
        writer.writerows(time_rows)
    terrain_csv = output_csv.with_name(
        f"{output_csv.stem}.terrain{output_csv.suffix}"
    )
    with terrain_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(terrain_rows[0]))
        writer.writeheader()
        writer.writerows(terrain_rows)
    terrain_env_csv = output_csv.with_name(
        f"{output_csv.stem}.terrain_env{output_csv.suffix}"
    )
    with terrain_env_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(terrain_env_rows[0]))
        writer.writeheader()
        writer.writerows(terrain_env_rows)
    safety_csv = output_csv.with_name(
        f"{output_csv.stem}.safety{output_csv.suffix}"
    )
    gate_values = {name: value for name, value, _, _ in gates}
    safety_row = {
        "fall_event_count": fall_event_count,
        "unique_env_first_fall_count": unique_env_first_fall_count,
        "time_to_first_fall_s": time_to_first_fall_s,
        "failure_free_exposure_s": failure_free_exposure_s,
        "post_reset_count": post_reset_count,
        "post_reset_sample_count": post_reset_sample_count,
        "warmup_fall_event_count": warmup_fall_event_count,
        "warmup_unique_env_first_fall_count": (
            warmup_unique_env_first_fall_count
        ),
        "warmup_failure_free_exposure_s": warmup_failure_free_exposure_s,
        "num_envs": n,
        "seed": args_cli.seed,
        "hall_contact_distribution_mode": getattr(
            args_cli, "effective_hall_contact_distribution_mode", "unknown"
        ),
        "high_start_phase": recovery["high_start_phase"],
        "high_end_phase": recovery["high_end_phase"],
        "high_end_vx_recovery_ratio": recovery["vx_recovery_ratio"],
        "high_end_step_length_recovery_ratio": recovery[
            "step_length_recovery_ratio"
        ],
        "high_end_stride_length_recovery_ratio": recovery[
            "stride_length_recovery_ratio"
        ],
        "high_end_cadence_recovery_ratio": recovery[
            "cadence_recovery_ratio"
        ],
        "high_end_recovery_response_time_s": recovery["response_time_s"],
        "max_at_risk_tilt_deg": gate_values["最大基座倾角"],
        "max_phase_steady_abs_vy_mps": gate_values["稳态横向速度"],
        "max_phase_steady_contact_point_tangent_slip_mps": (
            slip_diagnostic["maximum_phase_steady_mps"]
        ),
        "contact_point_slip_gate_enabled": slip_diagnostic["gate_enabled"],
        "contact_point_slip_gate_threshold_mps": (
            slip_diagnostic["threshold_mps"]
        ),
        "primary_gate_pass": all(item[2] for item in gates),
    }
    with safety_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(safety_row))
        writer.writeheader()
        writer.writerow(safety_row)

    summary = args_cli.output_summary
    if summary is None:
        summary = output_csv.with_name(f"{output_csv.stem}.summary.md")
    primary_gate_pass = all(item[2] for item in gates)
    if not primary_gate_pass:
        overall = "NEEDS_TRAINING"
    elif slip_diagnostic["gate_enabled"]:
        overall = "PASS"
    else:
        overall = "PROVISIONAL_PASS_SLIP_DIAGNOSTIC_ONLY"
    lines = [
        "# Friction-switch adaptation",
        "",
        f"- Overall: **{overall}**",
        (
            "- Command: "
            f"`vx={command_vx:.3f} m/s, vy={command_vy:.3f} m/s, "
            f"wz={command_wz:.3f} rad/s` (unchanged across phases)"
        ),
        f"- Sequence: `{sequence}`",
        f"- Foot-sensor ablation: `{args_cli.ablate_foot_sensor}`",
        f"- Hall-risk governor: `{hall_governor is not None}`",
        (
            "- Hall contact distribution: `"
            f"{getattr(args_cli, 'effective_hall_contact_distribution_mode', 'unknown')}`"
        ),
        f"- Environments / seed: `{n}` / `{args_cli.seed}`",
        (
            "- Primary speed/slip statistics: `first-fall censored`; "
            "reset-inclusive values remain in CSV columns suffixed "
            "`_including_resets`"
        ),
        (
            "- Adaptive-gait rule: low μ may preserve forward speed with "
            "higher cadence and shorter steps; there is no fixed low-speed, "
            "high-minus-low speed, or cadence-difference release gate."
        ),
        (
            "- Slip metric: corrected rigid-body contact-point tangential "
            f"speed (`{slip_diagnostic['schema']}`); status="
            f"`{slip_diagnostic['status']}`."
        ),
        (
            "- Legacy warning: schema-v1 `contact_slip` based on ankle/body-COM "
            "planar speed is diagnostic-only and must not be used for a formal "
            "gate or training target."
        ),
        "",
        "## Safety accounting",
        "",
        "| metric | value | definition |",
        "|---|---:|---|",
        (
            f"| fall_event_count | {fall_event_count} | all non-timeout "
            "terminations, including warm-up and repeated resets |"
        ),
        (
            "| unique_env_first_fall_count | "
            f"{unique_env_first_fall_count} | distinct environment rows with "
            "a first failure |"
        ),
        (
            "| time_to_first_fall_s | "
            f"{time_to_first_fall_s:.3f} | from rollout start; n/a when no "
            "failure |"
            if np.isfinite(time_to_first_fall_s)
            else "| time_to_first_fall_s | n/a | no failure observed |"
        ),
        (
            "| failure_free_exposure_s | "
            f"{failure_free_exposure_s:.3f} | summed at-risk environment "
            "seconds until first failure/censoring |"
        ),
        (
            f"| post_reset_count | {post_reset_count} | repeated failure "
            "events after an environment's first failure |"
        ),
        (
            "| post_reset_sample_count | "
            f"{post_reset_sample_count} | reset/post-failure env-step samples "
            "excluded from primary gait statistics |"
        ),
        (
            "| warmup_fall_event_count | "
            f"{warmup_fall_event_count} | warm-up failures included in the "
            "release gate |"
        ),
        (
            "| warmup_unique_env_first_fall_count | "
            f"{warmup_unique_env_first_fall_count} | first failures occurring "
            "during warm-up |"
        ),
        (
            "| warmup_failure_free_exposure_s | "
            f"{warmup_failure_free_exposure_s:.3f} | warm-up at-risk "
            "environment seconds |"
        ),
        "",
        "## Per-phase behavior",
        "",
        "| phase/role | μ | vx | cadence | step | stride | cadence×step | closure | vx/HS | cadence/HS | step/HS | abs(vy) | tilt max | cp-slip | valid | response | falls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        response_text = (
            f"{row['response_time_s']:.3f}"
            if np.isfinite(row["response_time_s"])
            else "n/a"
        )
        lines.append(
            f"| {row['phase']} {row['phase_role']} | {row['mu']:.3f} | "
            f"{row['steady_vx']:.3f} | {row['step_frequency_hz']:.3f} | "
            f"{row['mean_step_length_m']:.3f} | "
            f"{row['mean_stride_length_m']:.3f} | "
            f"{row['kinematic_speed_estimate_mps']:.3f} | "
            f"{row['kinematic_closure_error_mps']:.3f} | "
            f"{row['vx_vs_high_start_ratio']:.3f} | "
            f"{row['cadence_vs_high_start_ratio']:.3f} | "
            f"{row['step_length_vs_high_start_ratio']:.3f} | "
            f"{row['steady_abs_vy']:.3f} | "
            f"{row['max_at_risk_tilt_deg']:.3f} | "
            f"{row['steady_contact_point_tangent_slip_mps']:.3f} | "
            f"{row['steady_contact_point_tangent_slip_valid_fraction']:.3f} | "
            f"{response_text} | {row['fall_event_count']} |"
        )
    lines += [
        "",
        "## Per-transition gait response",
        "",
        "| transition | type | μ | vx ratio | cadence ratio | step ratio | stride ratio | to vx/HS | to step/HS | closure | response | HS-recovery response | tilt max | falls |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in transition_rows:
        hs_response = row["high_start_recovery_response_time_s"]
        lines.append(
            f"| {row['transition']} | {row['transition_type']} | "
            f"{row['from_mu']:.3f}→{row['to_mu']:.3f} | "
            f"{row['vx_ratio']:.3f} | {row['cadence_ratio']:.3f} | "
            f"{row['step_length_ratio']:.3f} | "
            f"{row['stride_length_ratio']:.3f} | "
            f"{row['to_vx_vs_high_start_ratio']:.3f} | "
            f"{row['to_step_length_vs_high_start_ratio']:.3f} | "
            f"{row['to_kinematic_closure_error_mps']:.3f} | "
            f"{row['response_time_s']:.3f} | "
            f"{hs_response:.3f} | {row['to_max_at_risk_tilt_deg']:.3f} | "
            f"{row['to_fall_event_count']} |"
        )
    lines += [
        "",
        "## Per-terrain behavior",
        "",
        "| phase | μ | terrain | envs | level | vx | vx incl. reset | p(low) | slip | slip incl. reset | fall events | first-fall envs |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in terrain_rows:
        lines.append(
            f"| {row['phase']} | {row['mu']:.3f} | "
            f"{row['terrain_type']} | {row['num_envs']} | "
            f"{row['mean_level']:.2f} | {row['steady_vx']:.3f} | "
            f"{row['steady_vx_including_resets']:.3f} | "
            f"{row['steady_low_traction_probability']:.3f} | "
            f"{row['steady_contact_slip']:.3f} | "
            f"{row['steady_contact_slip_including_resets']:.3f} | "
            f"{row['falls']} | {row['unique_env_first_fall_count']} |"
        )
    lines += [
        "",
        "## Primary adaptive-gait gates",
        "",
        "| gate | value | result | target |",
        "|---|---:|:---:|---:|",
    ]
    for name, value, passed, target in gates:
        value_text = f"{value:.3f}" if np.isfinite(value) else "n/a"
        lines.append(
            f"| {name} | {value_text} | "
            f"{'PASS' if passed else 'WARN'} | {target} |"
        )
    lines += [
        "",
        "HighEnd recovery is the final high-μ phase divided by the first "
        "high-μ phase (HighStart); high phases are never averaged for the "
        "formal recovery gate.",
        "",
        "## Historical fixed-slowdown diagnostics (not gates)",
        "",
        "| metric | value | status |",
        "|---|---:|---|",
    ]
    for name, value, status in legacy_diagnostics:
        value_text = f"{value:.3f}" if np.isfinite(value) else "n/a"
        lines.append(f"| {name} | {value_text} | {status} |")
    lines.append("")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines), encoding="utf-8")

    if args_cli.collect_npz is not None:
        if not switch_collected_obs:
            raise RuntimeError(
                "no switch observations were collected; use a positive "
                "--collect_stride and at least two physics steps per phase"
            )
        args_cli.collect_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args_cli.collect_npz,
            obs=np.concatenate(switch_collected_obs, axis=0),
            mu=np.concatenate(switch_collected_mu, axis=0),
            cmd_vx=np.concatenate(switch_collected_cmd, axis=0),
            seed=np.concatenate(switch_collected_seed, axis=0),
            root_lin_vel_b=np.concatenate(switch_collected_root_lin_vel_b, axis=0),
            root_ang_vel_b=np.concatenate(switch_collected_root_ang_vel_b, axis=0),
            actor_command=np.concatenate(
                switch_collected_actor_command, axis=0
            ),
            applied_command=np.concatenate(switch_collected_applied_command, axis=0),
            contact_slip=np.concatenate(switch_collected_contact_slip, axis=0),
            contact_point_tangent_slip=np.concatenate(
                switch_collected_contact_slip, axis=0
            ),
            contact_point_tangent_slip_valid=np.concatenate(
                switch_collected_contact_slip_valid, axis=0
            ),
            legacy_link_origin_planar_slip=np.concatenate(
                switch_collected_legacy_link_origin_planar_slip, axis=0
            ),
            valid=np.concatenate(switch_collected_valid, axis=0),
            env_id=np.concatenate(switch_collected_env_id, axis=0),
            step=np.concatenate(switch_collected_step, axis=0),
            fall=np.concatenate(switch_collected_fall, axis=0),
            done=np.concatenate(switch_collected_done, axis=0),
            time_out=np.concatenate(switch_collected_time_out, axis=0),
            hall_valid_lr=np.concatenate(
                switch_collected_hall_valid_lr, axis=0
            ),
            phase=np.concatenate(switch_collected_phase, axis=0),
            time_since_switch_s=np.concatenate(
                switch_collected_time_since_switch, axis=0
            ),
            rollout_id=np.concatenate(switch_collected_rollout_id, axis=0),
            **_collection_metadata(
                dataset_kind="switch",
                task=args_cli.task,
                seed=args_cli.seed,
                policy_dt=dt,
                collect_stride=args_cli.collect_stride,
                actor_checkpoint=_selected_actor_checkpoint(args_cli)[0],
                actor_source=_selected_actor_checkpoint(args_cli)[1],
                hall_contact_distribution_mode=getattr(
                    args_cli,
                    "effective_hall_contact_distribution_mode",
                    "unknown",
                ),
            ),
        )
        print(
            f"[info] switch observation dataset: {args_cli.collect_npz} "
            f"shape={sum(len(item) for item in switch_collected_obs)}x1864"
        )

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
            env_id=np.concatenate(dagger_env_id, axis=0),
            seed=np.concatenate(dagger_seed, axis=0),
            terrain_type=np.concatenate(dagger_terrain_type, axis=0),
            terrain_level=np.concatenate(dagger_terrain_level, axis=0),
            hall_contact_distribution_mode=np.asarray(
                getattr(
                    args_cli,
                    "effective_hall_contact_distribution_mode",
                    "unknown",
                ),
                dtype=np.str_,
            ),
        )
        print(
            f"[info] switch DAgger dataset: {args_cli.collect_dagger_npz} "
            f"shape={sum(len(item) for item in dagger_policy_obs)}x1864"
        )

    if args_cli.hall_trace_npz is not None:
        if not hall_trace_delta:
            raise RuntimeError("no Hall trace samples were collected")
        trace_sensor = getattr(uenv, "_hall_foot_sensor", None)
        if trace_sensor is None:
            raise RuntimeError("Hall sensor disappeared before trace export")
        args_cli.hall_trace_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args_cli.hall_trace_npz,
            time_s=np.asarray(hall_trace_time, dtype=np.float32),
            phase=np.asarray(hall_trace_phase, dtype=np.int16),
            mu=np.asarray(hall_trace_mu, dtype=np.float32),
            raw_tesla=np.stack(hall_trace_raw),
            delta_tesla=np.stack(hall_trace_delta),
            baseline_tesla=np.stack(hall_trace_baseline),
            valid_mask=np.stack(hall_trace_valid),
            foot_order=np.asarray(["left", "right"]),
            sensor_order=np.asarray([f"P{i:02d}" for i in range(15)]),
            axis_order=np.asarray(["Bx", "By", "Bz"]),
            field_units=np.asarray("tesla"),
            hall_positions_normalized=np.asarray(
                trace_sensor.cfg.hall_positions_normalized,
                dtype=np.float32,
            ),
            mirror_right_y=np.asarray(
                trace_sensor.cfg.mirror_right_y, dtype=np.bool_
            ),
            foot_local_axis_description=np.asarray(
                "+x toe, +y robot-left, +z up"
            ),
            trace_env_id=np.asarray(args_cli.hall_trace_env_id, dtype=np.int32),
            seed=np.asarray(args_cli.seed, dtype=np.int32),
            hall_contact_distribution_mode=np.asarray(
                getattr(
                    args_cli,
                    "effective_hall_contact_distribution_mode",
                    "unknown",
                ),
                dtype=np.str_,
            ),
        )
        print(
            f"[info] frame-synchronous Hall trace: {args_cli.hall_trace_npz} "
            f"shape={np.stack(hall_trace_delta).shape} [time,foot,P,axis]"
        )

    print("\n".join(lines))
    print(f"[info] phase CSV: {output_csv}")
    print(f"[info] transition CSV: {transition_csv}")
    print(f"[info] time-series CSV: {time_csv}")
    print(f"[info] terrain CSV: {terrain_csv}")
    print(f"[info] terrain/env CSV: {terrain_env_csv}")
    print(f"[info] safety CSV: {safety_csv}")
    print(f"[info] summary: {summary}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key=(
            "play_env_cfg_entry_point" if args_cli.video else "env_cfg_entry_point"
        ),
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    _disable_eval_capture_gate_warmup(agent_cfg)
    env_cfg.seed = args_cli.seed

    # The default friction matrix isolates friction on a plane.  Dedicated
    # terrain generalization runs preserve their generator, but still disable
    # online level changes so an evaluation trajectory stays on one known
    # slope/stair tile after reset.
    if not args_cli.preserve_task_terrain:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    else:
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is None:
            raise ValueError("--preserve_task_terrain requires a terrain generator")
        generator.curriculum = False
        if args_cli.terrain_max_init_level >= 0:
            env_cfg.scene.terrain.max_init_terrain_level = min(
                args_cli.terrain_max_init_level,
                generator.num_rows - 1,
            )
    if hasattr(env_cfg.events, "physics_material_reset"):
        env_cfg.events.physics_material_reset = None
    if hasattr(env_cfg.events, "friction_switch"):
        # Both matrix cells and switch phases apply an exact synchronized
        # material value themselves.  An inherited interval event would
        # otherwise overwrite that value mid-rollout and corrupt the label.
        # This is an evaluation-only script; training keeps its event.
        env_cfg.events.friction_switch = None
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    if args_cli.video and hasattr(env_cfg, "hall_sensor_cfg"):
        env_cfg.hall_sensor_cfg.enable_debug_vis = True
        env_cfg.hall_sensor_cfg.debug_vis_max_envs = env_cfg.scene.num_envs
        # Give low/high-friction comparison clips visibly different opaque
        # floors.  This color is presentation metadata only; physics still
        # comes from the exact material coefficient applied below.
        visual_mu = (
            float(args_cli.mu_bins[0])
            if args_cli.switch_sequence is None and args_cli.mu_bins
            else float(args_cli.switch_sequence[0])
        )
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.23, 0.38, 0.58)
            if visual_mu <= 0.25
            else (0.34, 0.27, 0.16),
            roughness=0.88,
            opacity=1.0,
        )

    # This must run after all evaluator Hall overrides (including debug-vis)
    # and before ``gym.make`` deep-copies/constructs observation managers.
    # The synchronized term configs are then the sole source used by every
    # lazy Hall sensor construction and managed-reset resample.
    (
        effective_hall_contact_distribution_mode,
        detailed_hall_terms,
    ) = _configure_detailed_hall_contact_cfg(
        env_cfg,
        enabled=args_cli.detailed_hall_contact,
        sync_fn=sync_hall_sensor_cfg_to_policy_terms,
    )
    args_cli.effective_hall_contact_distribution_mode = (
        effective_hall_contact_distribution_mode
    )
    if detailed_hall_terms:
        print(
            "[info] detailed Hall contact distribution installed before env "
            "creation: "
            + ",".join(detailed_hall_terms)
        )
    nominal_hall_terms = _configure_nominal_hall_sensor_cfg(
        env_cfg,
        enabled=args_cli.nominal_magnetic_sensor,
        sync_fn=sync_hall_sensor_cfg_to_policy_terms,
    )
    if nominal_hall_terms:
        print(
            "[info] nominal Hall configuration installed before env creation: "
            + ",".join(nominal_hall_terms)
        )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        terrain_cfg = env_cfg.scene.terrain.terrain_generator
        terrain_names = (
            list(terrain_cfg.sub_terrains.keys())
            if terrain_cfg is not None
            else ["flat"]
        )
        terrain_index = (
            terrain_names.index(args_cli.camera_terrain)
            if args_cli.camera_terrain in terrain_names
            else 0
        )
        terrain = getattr(env.unwrapped.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        focus_env = 0
        if terrain_types is not None:
            candidates = torch.nonzero(
                terrain_types == terrain_index, as_tuple=False
            ).flatten()
            if candidates.numel() > 0:
                if terrain_levels is None:
                    focus_env = int(candidates[0].item())
                else:
                    focus_env = int(
                        candidates[
                            torch.argmax(terrain_levels[candidates])
                        ].item()
                    )
        origin = (
            env.unwrapped.scene.env_origins[focus_env]
            .detach()
            .cpu()
            .numpy()
        )
        env.unwrapped.sim.set_camera_view(
            eye=origin + np.asarray([3.2, -4.0, 2.25]),
            target=origin + np.asarray([0.65, 0.0, 0.75]),
        )
        args_cli.video_dir.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
            "video_folder": os.fspath(args_cli.video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": (
                f"hall_{args_cli.camera_terrain}_seed{args_cli.seed}"
            ),
            "disable_logger": True,
        }
        print(
            "[info] video camera: "
            f"terrain={args_cli.camera_terrain}, env={focus_env}, "
            f"level={int(terrain_levels[focus_env].item()) if terrain_levels is not None else 0}"
        )
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.shared_policy is not None and args_cli.shared_onnx is not None:
        raise ValueError("use only one of --shared_policy and --shared_onnx")

    forward_velocity_estimator = None
    forward_velocity_estimate = None
    if not 0.0 < args_cli.forward_velocity_filter_alpha <= 1.0:
        raise ValueError("--forward_velocity_filter_alpha must be in (0, 1]")
    if args_cli.forward_velocity_estimator is not None:
        from unitree_rl_lab.traction.forward_velocity_estimator import (
            build_forward_velocity_estimator,
        )

        estimator_payload = torch.load(
            args_cli.forward_velocity_estimator,
            map_location="cpu",
            weights_only=False,
        )
        if estimator_payload.get("input_dim") != 1864:
            raise ValueError(
                "forward velocity estimator must consume the deployable 1864-D "
                f"Hall observation, got {estimator_payload.get('input_dim')}"
            )
        forward_velocity_estimator = build_forward_velocity_estimator(
            estimator_payload
        ).to(env.unwrapped.device).eval()
        print(
            "[info] diagnostic forward-speed estimator "
            f"{args_cli.forward_velocity_estimator}"
        )

    def _update_forward_velocity_estimate(observation: torch.Tensor) -> None:
        nonlocal forward_velocity_estimate
        if forward_velocity_estimator is None:
            return
        with torch.inference_mode():
            current = forward_velocity_estimator(observation).reshape(-1)
        if forward_velocity_estimate is None:
            forward_velocity_estimate = current.detach().clone()
        else:
            alpha = float(args_cli.forward_velocity_filter_alpha)
            forward_velocity_estimate = (
                (1.0 - alpha) * forward_velocity_estimate + alpha * current
            )

    is_layout_student = False
    # A PPO/RSL actor can use exactly the same deployable Hall-risk head as a
    # geometry-aware Student.  Keeping the gait actor separate from the risk
    # head lets us safety-test a conservative command layer without changing
    # 29 joint targets or smuggling simulator labels into the actor.
    rsl_hall_risk_mode = False
    # ``anchored_hall_mode`` deliberately keeps the original locomotion actor
    # as the sole action source.  Hall data can therefore change the requested
    # command through the governor, but cannot inject an unvalidated residual
    # into the 29 joint targets.  This is also the exact sensor-loss fallback
    # used by the real-time deployment wrapper.
    anchored_hall_mode = bool(args_cli.anchored_hall_governor)
    if anchored_hall_mode and (
        args_cli.shared_onnx is None
        or args_cli.hall_risk_checkpoint is None
        or not args_cli.hall_traction_governor
    ):
        raise ValueError(
            "--anchored_hall_governor requires --shared_onnx, "
            "--hall_risk_checkpoint, and --hall_traction_governor"
        )
    if anchored_hall_mode and args_cli.shared_policy is not None:
        raise ValueError(
            "--anchored_hall_governor is only supported with --shared_onnx"
        )
    if args_cli.shared_onnx is not None:
        import onnxruntime as ort

        from unitree_rl_lab.traction.layout_magnetic_student import INPUT_DIM
        from unitree_rl_lab.traction.schema import legacy_actor_schema

        session = ort.InferenceSession(
            os.fspath(args_cli.shared_onnx),
            providers=["CPUExecutionProvider"],
        )
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("magnetic policy ONNX must have exactly one input and output")
        onnx_input = session.get_inputs()[0]
        onnx_output = session.get_outputs()[0]
        input_shape = onnx_input.shape
        output_shape = onnx_output.shape
        if input_shape[-1] != INPUT_DIM or output_shape[-1] != 29:
            raise ValueError(
                "magnetic ONNX schema mismatch: "
                f"input={input_shape}, output={output_shape}, expected [N,{INPUT_DIM}]->[N,29]"
            )
        command_history_slice = legacy_actor_schema(
            include_force=False
        ).term_slice("velocity_commands")
        if command_history_slice.stop > 480 or command_history_slice.stop - command_history_slice.start != 15:
            raise RuntimeError(
                "legacy actor command history must be a five-frame 3-D slice"
            )

        def _run_single_output_onnx(
            runtime_session,
            runtime_input,
            runtime_output,
            observation_np: np.ndarray,
        ) -> np.ndarray:
            """Run a compatible [N,1864] -> [N,29] deployable actor graph."""

            fixed_batch = (
                runtime_input.shape[0]
                if isinstance(runtime_input.shape[0], int)
                else None
            )
            if fixed_batch == 1 and observation_np.shape[0] != 1:
                return np.concatenate(
                    [
                        runtime_session.run(
                            [runtime_output.name],
                            {runtime_input.name: observation_np[index : index + 1]},
                        )[0]
                        for index in range(observation_np.shape[0])
                    ],
                    axis=0,
                )
            return runtime_session.run(
                [runtime_output.name], {runtime_input.name: observation_np}
            )[0]

        last_policy_observation = None
        lateral_estimator = None
        lateral_estimate = None
        hall_risk_estimator = None
        risk_session = None
        risk_input = None
        risk_output = None
        recovery_session = None
        recovery_input = None
        recovery_output = None
        if args_cli.hall_recovery_onnx is not None:
            if not anchored_hall_mode:
                raise ValueError(
                    "--hall_recovery_onnx requires --anchored_hall_governor"
                )
            recovery_session = ort.InferenceSession(
                os.fspath(args_cli.hall_recovery_onnx),
                providers=["CPUExecutionProvider"],
            )
            if (
                len(recovery_session.get_inputs()) != 1
                or len(recovery_session.get_outputs()) != 1
            ):
                raise ValueError(
                    "Hall recovery ONNX must have exactly one input and output"
                )
            recovery_input = recovery_session.get_inputs()[0]
            recovery_output = recovery_session.get_outputs()[0]
            if (
                recovery_input.shape[-1] != INPUT_DIM
                or recovery_output.shape[-1] != 29
            ):
                raise ValueError(
                    "Hall recovery ONNX schema mismatch: "
                    f"input={recovery_input.shape}, output={recovery_output.shape}, "
                    f"expected [N,{INPUT_DIM}]->[N,29]"
                )
            print(
                f"[info] load Hall recovery ONNX {args_cli.hall_recovery_onnx} "
                "(used only in LOW/probe states)"
            )
        if anchored_hall_mode:
            risk_path = args_cli.hall_risk_checkpoint
            assert risk_path is not None
            if risk_path.suffix.lower() == ".onnx":
                risk_session = ort.InferenceSession(
                    os.fspath(risk_path),
                    providers=["CPUExecutionProvider"],
                )
                if len(risk_session.get_inputs()) != 1 or len(risk_session.get_outputs()) != 1:
                    raise ValueError(
                        "anchored Hall risk ONNX must have exactly one input and output"
                    )
                risk_input = risk_session.get_inputs()[0]
                risk_output = risk_session.get_outputs()[0]
                risk_shape = risk_input.shape
                risk_out_shape = risk_output.shape
                if risk_shape[-1] != INPUT_DIM or risk_out_shape[-1] != 1:
                    raise ValueError(
                        "anchored Hall risk ONNX schema mismatch: "
                        f"input={risk_shape}, output={risk_out_shape}, "
                        f"expected [N,{INPUT_DIM}]->[N,1]"
                    )
                print(
                    f"[info] load anchored Hall risk ONNX {risk_path} "
                    "(input=1864, output=1, provider=CPUExecutionProvider)"
                )
            else:
                from unitree_rl_lab.traction.hall_risk_estimator import (
                    build_hall_risk_estimator,
                )

                risk_payload = torch.load(
                    risk_path,
                    map_location="cpu",
                    weights_only=False,
                )
                hall_risk_estimator = build_hall_risk_estimator(
                    risk_payload
                ).to(env.unwrapped.device).eval()
                print(f"[info] load anchored Hall risk checkpoint {risk_path}")

        def policy(observation):
            nonlocal last_policy_observation
            actor_observation = observation["policy"]
            if actor_observation.shape[1] != INPUT_DIM:
                raise ValueError(
                    f"environment policy observation is {actor_observation.shape[1]}, "
                    f"expected {INPUT_DIM}"
                )
            _update_forward_velocity_estimate(actor_observation)
            last_policy_observation = actor_observation.detach().clone()
            policy.last_policy_observation = last_policy_observation
            observation_np = actor_observation.detach().cpu().numpy()
            action_np = _run_single_output_onnx(
                session, onnx_input, onnx_output, observation_np
            )
            action = torch.as_tensor(
                action_np,
                device=actor_observation.device,
                dtype=actor_observation.dtype,
            )
            # Keep both action candidates.  The command governor is updated
            # after the policy call because the current Hall frame is the
            # causal evidence for the next control command.  The main loop
            # selects the recovery action only after that state transition;
            # HIGH therefore preserves the original actor exactly.
            policy.last_base_action = action.detach().clone()
            policy.last_recovery_action = None
            if recovery_session is not None:
                recovery_np = _run_single_output_onnx(
                    recovery_session,
                    recovery_input,
                    recovery_output,
                    observation_np,
                )
                policy.last_recovery_action = torch.as_tensor(
                    recovery_np,
                    device=actor_observation.device,
                    dtype=actor_observation.dtype,
                )
            if anchored_hall_mode:
                # The risk estimator sees precisely the deployable 1864-D
                # observation.  No simulator contact force, friction label, or
                # Hall-to-force inverse is available on this path.
                if risk_session is not None:
                    risk_np = actor_observation.detach().cpu().numpy()
                    risk_fixed_batch = (
                        risk_input.shape[0]
                        if isinstance(risk_input.shape[0], int)
                        else None
                    )
                    if risk_fixed_batch == 1 and risk_np.shape[0] != 1:
                        risk_np = np.concatenate(
                            [
                                risk_session.run(
                                    [risk_output.name],
                                    {risk_input.name: risk_np[index : index + 1]},
                                )[0]
                                for index in range(risk_np.shape[0])
                            ],
                            axis=0,
                        )
                    else:
                        risk_np = risk_session.run(
                            [risk_output.name],
                            {risk_input.name: risk_np},
                        )[0]
                    low_probability = torch.as_tensor(
                        risk_np,
                        device=actor_observation.device,
                        dtype=actor_observation.dtype,
                    ).reshape(-1)
                else:
                    assert hall_risk_estimator is not None
                    low_probability = hall_risk_estimator(actor_observation).reshape(-1)
                policy.last_low_traction_probability = (
                    low_probability.detach().clamp(0.0, 1.0)
                )
            return action

        def _governed_command_reflex_action(
            governed_command: torch.Tensor,
        ) -> torch.Tensor:
            """Evaluate the audited actor with a causal governed command.

            The usual observation has five frames of velocity-command history,
            so changing the command manager after the actor call delays the
            original actor's response by roughly that history length.  During
            a confirmed Hall LOW state we may instead replace only those
            command entries with the governor's already-causal output and run
            the *same* original actor once more.  Hall still provides only the
            risk decision; this helper consumes no contact/slip/friction truth.
            """

            if last_policy_observation is None:
                raise RuntimeError("governed-command reflex called before policy")
            if governed_command.shape != (last_policy_observation.shape[0], 3):
                raise ValueError(
                    "governed command must be [num_envs,3] for the actor reflex"
                )
            reflex_observation = last_policy_observation.clone()
            command_history = governed_command.to(
                device=reflex_observation.device,
                dtype=reflex_observation.dtype,
            )[:, None, :].expand(-1, 5, -1).reshape(len(reflex_observation), -1)
            reflex_observation[:, command_history_slice] = command_history
            reflex_np = _run_single_output_onnx(
                session,
                onnx_input,
                onnx_output,
                reflex_observation.detach().cpu().numpy(),
            )
            return torch.as_tensor(
                reflex_np,
                device=reflex_observation.device,
                dtype=reflex_observation.dtype,
            )

        def _governed_command_recovery_action(
            governed_command: torch.Tensor,
        ) -> torch.Tensor:
            """Evaluate the low-grip recovery actor at the causal command.

            This mirrors :func:`_governed_command_reflex_action`, but is
            intentionally restricted to the optional recovery graph.  It
            prevents a high-speed command history from being presented to an
            actor trained only on the low-grip envelope during the first
            Hall-confirmed LOW control step.  The caller retains the strict
            LOW/validity gate and blends its output continuously.
            """

            if recovery_session is None or recovery_input is None or recovery_output is None:
                raise RuntimeError("governed recovery action requested without a recovery ONNX")
            if last_policy_observation is None:
                raise RuntimeError("governed recovery called before policy")
            if governed_command.shape != (last_policy_observation.shape[0], 3):
                raise ValueError(
                    "governed command must be [num_envs,3] for the recovery actor"
                )
            recovery_observation = last_policy_observation.clone()
            command_history = governed_command.to(
                device=recovery_observation.device,
                dtype=recovery_observation.dtype,
            )[:, None, :].expand(-1, 5, -1).reshape(len(recovery_observation), -1)
            recovery_observation[:, command_history_slice] = command_history
            recovery_np = _run_single_output_onnx(
                recovery_session,
                recovery_input,
                recovery_output,
                recovery_observation.detach().cpu().numpy(),
            )
            return torch.as_tensor(
                recovery_np,
                device=recovery_observation.device,
                dtype=recovery_observation.dtype,
            )

        policy.recovery_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.unwrapped.device
        )
        policy.last_base_action = None
        policy.last_recovery_action = None
        policy.governed_command_reflex_action = _governed_command_reflex_action
        policy.governed_command_recovery_action = (
            _governed_command_recovery_action
            if recovery_session is not None
            else None
        )
        print(
            f"[info] load magnetic ONNX {args_cli.shared_onnx} "
            f"(input={INPUT_DIM}, output=29, provider=CPUExecutionProvider)"
            + (" [anchored Hall governor]" if anchored_hall_mode else "")
            + (" [Hall recovery hybrid]" if recovery_session is not None else "")
        )
    elif args_cli.shared_policy is None:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if args_cli.checkpoint:
            print(f"[info] load {args_cli.checkpoint}")
            runner.load(
                args_cli.checkpoint,
                load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG),
                strict=True,
            )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        rsl_hall_risk_mode = args_cli.hall_risk_checkpoint is not None
    else:
        shared_script_dir = Path(__file__).resolve().parents[3] / "scripts"
        sys.path.insert(0, str(shared_script_dir))
        payload = torch.load(args_cli.shared_policy, map_location="cpu", weights_only=False)
        state = payload.get("model")
        is_hall_recovery = payload.get("policy_type") == "hall_recovery_policy"
        recovery_activation_mode = payload.get(
            "activation_mode", "governor_low"
        )
        # Both the legacy shared actor and the newer geometry-aware Student
        # own a ``foot_encoder``.  Testing that broad prefix alone therefore
        # misclassified a valid shared policy as LayoutMagneticStudent and
        # made GPU/PyTorch evaluation fail.  The geometry-aware model has a
        # baseline actor plus an explicit residual head; those keys are its
        # stable schema signature.
        is_layout_student = is_hall_recovery or (
            isinstance(state, dict)
            and (
                "residual_limit" in state
                or "baseline_actor.mlp.0.weight" in state
                or "residual_head.weight" in state
            )
        )
        if is_hall_recovery:
            from unitree_rl_lab.traction.hall_recovery_policy import (
                HallRecoveryPolicy,
            )
            from unitree_rl_lab.traction.hall_risk_estimator import (
                BaselineInvariantHallTractionRiskEstimator,
                HallTractionRiskEstimator,
            )
            from unitree_rl_lab.traction.layout_magnetic_student import (
                INPUT_DIM,
                LayoutMagneticStudent,
                normalize_trailing_feature_mode,
            )

            base_policy = LayoutMagneticStudent(
                float(payload.get("base_residual_limit", 1.0)),
                trailing_feature_mode=normalize_trailing_feature_mode(
                    str(payload.get("trailing_feature_mode", "sensor_age"))
                ),
            )
            embedded_risk = (
                BaselineInvariantHallTractionRiskEstimator()
                if payload.get("risk_model_variant") == "baseline_invariant"
                else HallTractionRiskEstimator()
            )
            shared_model = HallRecoveryPolicy(
                base_policy,
                embedded_risk,
                correction_limit=float(payload.get("correction_limit", 0.25)),
                risk_gate_start=float(payload.get("risk_gate_start", 0.35)),
                risk_gate_full=float(payload.get("risk_gate_full", 0.75)),
            ).to(env.unwrapped.device).eval()
            shared_model.load_state_dict(state, strict=True)
            shared_description = "risk-gated Hall recovery Student"
        elif is_layout_student:
            from unitree_rl_lab.traction.layout_magnetic_student import (
                INPUT_DIM,
                LayoutMagneticStudent,
                normalize_trailing_feature_mode,
            )

            shared_model = LayoutMagneticStudent(
                float(payload.get("residual_limit", 1.0)),
                trailing_feature_mode=normalize_trailing_feature_mode(
                    str(payload.get("trailing_feature_mode", "sensor_age"))
                ),
            ).to(env.unwrapped.device).eval()
            shared_model.load_state_dict(state, strict=True)
            shared_description = "layout-aware direct Hall Student"
        else:
            from train_shared_magnetic_policy import INPUT_DIM, SharedMagneticPolicy

        if not is_layout_student and "model" in payload:
            shared_model = SharedMagneticPolicy().to(env.unwrapped.device).eval()
            shared_model.load_state_dict(payload["model"], strict=True)
            shared_description = "single shared magnetic actor"
        elif not is_layout_student and payload.get("policy_type") == "estimator_guided_magnetic_teacher":
            from export_estimator_guided_magnetic_teacher import build_runtime

            shared_model = build_runtime(payload).to(env.unwrapped.device).eval()
            shared_description = (
                "calibration-gated estimator-guided magnetic Teacher"
            )
        elif not is_layout_student and (
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
        elif not is_layout_student and "fast_checkpoint" in payload and "stable_checkpoint" in payload:
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
        elif not is_layout_student:
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
        hall_risk_estimator = None
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
        if args_cli.hall_risk_checkpoint is not None:
            if not is_layout_student:
                raise ValueError(
                    "--hall_risk_checkpoint requires a layout-aware Hall Student"
                )
            from unitree_rl_lab.traction.hall_risk_estimator import (
                build_hall_risk_estimator,
            )

            risk_payload = torch.load(
                args_cli.hall_risk_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            hall_risk_estimator = build_hall_risk_estimator(risk_payload).to(
                env.unwrapped.device
            ).eval()
            print(
                "[info] independent Hall risk estimator "
                f"{args_cli.hall_risk_checkpoint}"
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
            _update_forward_velocity_estimate(actor_observation)
            # Preserve the exact tensor consumed by the Student.  In
            # particular, channel 1862 must contain the deployable body-vy
            # estimate instead of the simulator's privileged ground truth.
            last_policy_observation = actor_observation.detach().clone()
            policy.last_policy_observation = last_policy_observation
            if is_layout_student:
                if is_hall_recovery:
                    action, base_action, _, _ = shared_model.recovery_outputs(
                        actor_observation
                    )
                    if recovery_activation_mode != "confidence_gated":
                        recovery_mask = getattr(
                            policy,
                            "recovery_mask",
                            torch.zeros(
                                action.shape[0],
                                dtype=torch.bool,
                                device=action.device,
                            ),
                        )
                        action = torch.where(
                            recovery_mask[:, None], action, base_action
                        )
                    (
                        _,
                        estimated_mu,
                        _,
                        confidence,
                        _,
                    ) = shared_model.base_policy.all_outputs(actor_observation)
                else:
                    (
                        action,
                        estimated_mu,
                        _,
                        confidence,
                        _,
                    ) = shared_model.all_outputs(actor_observation)
                if hall_risk_estimator is None:
                    low_probability = torch.sigmoid(
                        (0.25 - estimated_mu) / 0.05
                    )
                    # Missing/stale Hall is a conservative decision, not
                    # evidence of a high-traction surface.
                    low_probability = (
                        confidence * low_probability + (1.0 - confidence)
                    )
                else:
                    low_probability = hall_risk_estimator(actor_observation)
                policy.last_low_traction_probability = (
                    low_probability.detach().reshape(-1)
                )
                return action
            return shared_model(actor_observation)

        print(
            f"[info] load shared magnetic policy {args_cli.shared_policy} "
            f"({shared_description}, input={INPUT_DIM}, output=29)"
        )
        policy.recovery_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.unwrapped.device
        )

    if rsl_hall_risk_mode:
        # The RSL actor itself is a regular 1864-D Hall/proprioceptive policy.
        # Attach an independent future-slip risk head only at the command
        # interface.  The estimator was trained with simulator contact/fall
        # outcomes as offline labels, but it receives only the actor's exact
        # causal observation at runtime.
        from unitree_rl_lab.traction.hall_risk_estimator import (
            build_hall_risk_estimator,
        )
        from unitree_rl_lab.traction.layout_magnetic_student import INPUT_DIM

        risk_path = args_cli.hall_risk_checkpoint
        assert risk_path is not None
        rsl_risk_session = None
        rsl_risk_input = None
        rsl_risk_output = None
        rsl_hall_risk_estimator = None
        if risk_path.suffix.lower() == ".onnx":
            import onnxruntime as ort

            rsl_risk_session = ort.InferenceSession(
                os.fspath(risk_path), providers=["CPUExecutionProvider"]
            )
            if (
                len(rsl_risk_session.get_inputs()) != 1
                or len(rsl_risk_session.get_outputs()) != 1
            ):
                raise ValueError("RSL Hall risk ONNX must have one input and one output")
            rsl_risk_input = rsl_risk_session.get_inputs()[0]
            rsl_risk_output = rsl_risk_session.get_outputs()[0]
            if rsl_risk_input.shape[-1] != INPUT_DIM or rsl_risk_output.shape[-1] != 1:
                raise ValueError(
                    "RSL Hall risk ONNX schema mismatch: "
                    f"input={rsl_risk_input.shape}, output={rsl_risk_output.shape}, "
                    f"expected [N,{INPUT_DIM}]->[N,1]"
                )
        else:
            rsl_risk_payload = torch.load(
                risk_path, map_location="cpu", weights_only=False
            )
            rsl_hall_risk_estimator = build_hall_risk_estimator(
                rsl_risk_payload
            ).to(env.unwrapped.device).eval()

        rsl_base_policy = policy

        def policy(observation):
            action = rsl_base_policy(observation)
            # RSL-RL 3.x passes a TensorDict whose top-level ``policy`` key
            # can itself be a nested TensorDict (with batch shape [N]), rather
            # than the final flat tensor.  Build exactly the raw latent used
            # by the actor from its declared ordered observation groups.  This
            # is intentionally before the actor normalizer: the independently
            # trained risk head expects the same raw 1864-D deploy schema that
            # comes from the environment/Hall adapter.
            actor_groups = getattr(rsl_base_policy, "obs_groups", None)
            if actor_groups is not None:
                actor_observation = torch.cat(
                    [observation[group] for group in actor_groups], dim=-1
                )
            else:
                actor_observation = (
                    observation["policy"] if isinstance(observation, dict) else observation
                )
            if actor_observation.ndim != 2 or actor_observation.shape[1] != INPUT_DIM:
                raise ValueError(
                    "RSL Hall-risk wrapper requires the exact deployable "
                    f"[N,{INPUT_DIM}] actor observation, got {tuple(actor_observation.shape)}"
                )
            policy.last_policy_observation = actor_observation.detach().clone()
            if rsl_risk_session is not None:
                assert rsl_risk_input is not None and rsl_risk_output is not None
                risk_np = actor_observation.detach().cpu().numpy()
                risk_np = rsl_risk_session.run(
                    [rsl_risk_output.name], {rsl_risk_input.name: risk_np}
                )[0]
                risk = torch.as_tensor(
                    risk_np, device=actor_observation.device, dtype=actor_observation.dtype
                ).reshape(-1)
            else:
                assert rsl_hall_risk_estimator is not None
                risk = rsl_hall_risk_estimator(actor_observation).reshape(-1)
            policy.last_low_traction_probability = risk.detach().clamp(0.0, 1.0)
            return action

        policy.recovery_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.unwrapped.device
        )
        print(
            "[info] attach independent Hall-only risk head to RSL actor: "
            f"{risk_path}"
        )

    hall_governor = None
    if args_cli.hall_traction_governor:
        if not is_layout_student and not anchored_hall_mode and not rsl_hall_risk_mode:
            raise ValueError(
                "--hall_traction_governor requires a layout-aware Hall Student, "
                "--anchored_hall_governor, or an RSL --checkpoint with "
                "--hall_risk_checkpoint"
            )
        governor_cfg = HallTractionGovernorCfg(
            low_speed_limit=args_cli.governor_low_speed,
            high_speed_limit=args_cli.governor_high_speed,
            critical_speed_limit=args_cli.governor_critical_speed,
            probability_low_enter=args_cli.governor_low_probability,
            probability_high_enter=args_cli.governor_high_probability,
            probability_critical_enter=args_cli.governor_critical_probability,
            critical_hold_s=args_cli.governor_critical_hold,
            probability_ema_alpha=args_cli.governor_probability_alpha,
            state_reference_ema_alpha=args_cli.governor_reference_alpha,
            reference_settle_s=args_cli.governor_reference_settle_s,
            reference_settle_alpha=args_cli.governor_reference_settle_alpha,
            prebrake_probability=args_cli.governor_prebrake_probability,
            prebrake_relative_rise=args_cli.governor_prebrake_relative_rise,
            prebrake_speed_limit=args_cli.governor_prebrake_speed,
            relative_low_rise=args_cli.governor_relative_low_rise,
            relative_high_drop=args_cli.governor_relative_high_drop,
            relative_low_min_probability=(
                args_cli.governor_relative_low_min_probability
            ),
            allow_absolute_high_clear=(
                args_cli.governor_allow_absolute_high_clear
            ),
            probe_speed_limit=args_cli.governor_probe_speed,
            probe_duration_s=args_cli.governor_probe_duration,
            initial_probe_ignore_critical=(
                args_cli.governor_initial_probe_ignore_critical
            ),
            allow_critical_reprobe=args_cli.governor_allow_critical_reprobe,
            critical_reprobe_s=args_cli.governor_critical_reprobe,
            probe_relative_clear_drop=(
                args_cli.governor_probe_relative_clear_drop
            ),
            crawl_pulse_s=args_cli.governor_crawl_pulse,
            low_reprobe_s=args_cli.governor_low_reprobe,
            low_hold_s=args_cli.governor_low_hold,
            high_hold_s=args_cli.governor_high_hold,
            linear_accel_rate=args_cli.governor_accel_rate,
            linear_decel_rate=args_cli.governor_decel_rate,
        )
        hall_governor = HallTractionGovernor(
            env.unwrapped.num_envs,
            float(env.unwrapped.step_dt),
            env.unwrapped.device,
            governor_cfg,
        )
        print(
            "[info] Hall-risk governor enabled: "
            f"vx_low={governor_cfg.low_speed_limit:.3f}, "
            f"vx_high={governor_cfg.high_speed_limit:.3f}, "
            f"p_low_enter={governor_cfg.probability_low_enter:.2f}, "
            f"p_high_enter={governor_cfg.probability_high_enter:.2f}"
        )
    robot = env.unwrapped.scene["robot"]
    foot_body_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    try:
        left_dedicated_contact_sensor = env.unwrapped.scene[
            "left_hall_contact"
        ]
        right_dedicated_contact_sensor = env.unwrapped.scene[
            "right_hall_contact"
        ]
    except KeyError as exc:
        raise RuntimeError(
            "contact-point slip evaluation requires dedicated "
            "left_hall_contact/right_hall_contact ContactSensors with "
            "track_contact_points=True and a ground filter"
        ) from exc
    foot_sensor_ids = contact_sensor.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]

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
            "[info] DAgger trajectories execute Oracle Teacher "
            f"{args_cli.dagger_execution_teacher_onnx}"
        )

    if args_cli.switch_sequence is not None:
        _run_switch_evaluation(
            env,
            policy,
            robot,
            contact_sensor,
            left_dedicated_contact_sensor,
            right_dedicated_contact_sensor,
            foot_body_ids,
            foot_sensor_ids,
            execution_teacher,
            hall_governor,
        )
        env.close()
        simulation_app.close()
        return

    fields = [
        "mu",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "mean_applied_vx_command",
        "mean_applied_vy_command",
        "mean_applied_wz_command",
        "mean_low_traction_probability",
        "mean_p90_low_traction_probability",
        "mean_low_state_fraction",
        "mean_vx",
        "mean_vx_including_resets",
        "mean_vy",
        "mean_abs_vy",
        "mean_abs_vy_including_resets",
        "mean_abs_wz",
        "mean_forward_velocity_estimate",
        "mean_forward_velocity_abs_error",
        "mean_contact_slip",
        "mean_contact_slip_including_resets",
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
        "fall_event_count",
        "unique_env_first_fall_count",
        "time_to_first_fall_s",
        "failure_free_exposure_s",
        "post_reset_count",
        "post_reset_sample_count",
        "failure_free_sample_count",
        "warmup_fall_event_count",
        "warmup_unique_env_first_fall_count",
        "warmup_failure_free_exposure_s",
        "steps",
        "seed",
    ]
    fall_fields = [
        "mu",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
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
    # These labels are deliberately evaluation-only.  They make it possible
    # to train a causal forward-speed monitor from Hall history + proprioception
    # without inserting simulator velocity or true friction into the deployed
    # policy observation.
    collected_root_lin_vel_b = []
    collected_root_ang_vel_b = []
    collected_actor_command = []
    collected_applied_command = []
    collected_contact_slip = []
    collected_contact_slip_valid = []
    collected_legacy_link_origin_planar_slip = []
    collected_valid = []
    collected_env_id = []
    collected_step = []
    collected_fall = []
    collected_done = []
    collected_time_out = []
    collected_hall_valid_lr = []
    # Every matrix cell performs an explicit environment reset.  Persist the
    # cell boundary *and* every managed reset inside a cell so future-slip
    # labels cannot join equal env ids from different physical episodes.
    collected_rollout_id = []
    dagger_policy_obs = []
    dagger_teacher_obs = []
    dagger_mu = []
    dagger_cmd = []
    dagger_seed = []
    dagger_step = []
    dagger_fall = []
    dagger_recovery = []
    dagger_sample_weight = []
    dagger_env_id = []

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
    rollout_id = 0
    for mu in args_cli.mu_bins:
        for vx in args_cli.vx:
            rollout_id += 1
            env.reset()
            collection_episode_generation = torch.zeros(
                args_cli.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            )
            forward_velocity_estimate = None
            if hall_governor is not None:
                hall_governor.reset()
            if args_cli.shared_policy is not None:
                lateral_estimate = None
                last_policy_observation = None
            _force_mu(env, mu)
            command_vy = float(args_cli.vy)
            command_wz = float(args_cli.wz)
            _force_command(env, vx, command_vy, command_wz)
            ramp_steps = (
                args_cli.warmup_steps
                if args_cli.command_ramp_steps < 0
                else min(args_cli.command_ramp_steps, args_cli.warmup_steps)
            )
            if ramp_steps > 0:
                _set_command_value(env, 0.0, 0.0, 0.0)
            obs = env.get_observations()
            initial_command = uenv.command_manager.get_term(
                "base_velocity"
            ).vel_command_b[:, :3].detach().clone()
            _synchronize_evaluator_command_observation(
                env,
                obs,
                initial_command,
                torch.ones(
                    args_cli.num_envs,
                    dtype=torch.bool,
                    device=uenv.device,
                ),
            )
            vx_acc = []
            vx_including_resets_acc = []
            vy_signed_acc = []
            vy_acc = []
            vy_including_resets_acc = []
            wz_acc = []
            forward_velocity_estimate_acc = []
            forward_velocity_error_acc = []
            slip_acc = []
            slip_including_resets_acc = []
            fn_acc = []
            ft_acc = []
            force_ratio_acc = []
            lateral_pos_acc = []
            lateral_pos_signed_acc = []
            action_acc = []
            applied_command_acc = []
            applied_vy_command_acc = []
            applied_wz_command_acc = []
            risk_probability_acc = []
            risk_probability_p90_acc = []
            low_state_fraction_acc = []
            dones_total = 0
            falls_total = 0
            ever_failed = torch.zeros(
                args_cli.num_envs,
                dtype=torch.bool,
                device=env.unwrapped.device,
            )
            first_fall_time_by_env_s = torch.full(
                (args_cli.num_envs,),
                float("nan"),
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            fall_event_count = 0
            unique_env_first_fall_count = 0
            failure_free_exposure_s = 0.0
            post_reset_count = 0
            post_reset_sample_count = 0
            failure_free_sample_count = 0
            warmup_fall_event_count = 0
            warmup_unique_env_first_fall_count = 0
            warmup_failure_free_exposure_s = 0.0
            evaluation_dt = float(env.unwrapped.step_dt)
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
                command_fraction = 1.0
                if step < ramp_steps:
                    command_fraction = float(step + 1) / float(ramp_steps)
                requested_vx = vx * command_fraction
                requested_vy = command_vy * command_fraction
                requested_wz = command_wz * command_fraction
                _set_command_value(
                    env, requested_vx, requested_vy, requested_wz
                )
                requested_command_this_step = (
                    uenv.command_manager.get_term("base_velocity")
                    .vel_command_b[:, :3]
                    .detach()
                    .clone()
                )
                policy_obs = (
                    _ablate_foot_observation(obs)
                    if args_cli.ablate_foot_sensor
                    else obs
                )
                with torch.inference_mode():
                    actions = policy(policy_obs)
                    if execution_teacher is not None:
                        actions = execution_teacher(obs["teacher"])
                exact_actor_observation = _exact_actor_policy_observation(
                    policy, policy_obs
                ).detach().clone()
                actor_command_this_step = exact_actor_observation[
                    :, 42:45
                ].clone()
                governed_command = None
                low_probability = None
                governor_state = None
                if hall_governor is not None:
                    low_probability = getattr(
                        policy, "last_low_traction_probability", None
                    )
                    if low_probability is None:
                        raise RuntimeError(
                            "Hall Student did not expose low-traction probability"
                        )
                    requested = torch.zeros(
                        (args_cli.num_envs, 3),
                        device=env.unwrapped.device,
                    )
                    requested[:, 0] = requested_vx
                    requested[:, 1] = requested_vy
                    requested[:, 2] = requested_wz
                    hall_valid = _causal_hall_packet_validity(
                        policy, args_cli.num_envs, env.unwrapped.device
                    )
                    governed_command, governor_state = hall_governor.update(
                        requested, low_probability, valid=hall_valid
                    )
                    if hasattr(policy, "recovery_mask"):
                        policy.recovery_mask = (
                            (governor_state == LOW) & ~hall_governor.probing
                        ).detach().clone()
                    term = env.unwrapped.command_manager.get_term("base_velocity")
                    term.is_standing_env[:] = False
                    term.vel_command_b[:, :3] = governed_command
                applied_command_this_step = (
                    uenv.command_manager.get_term("base_velocity")
                    .vel_command_b[:, :3]
                    .detach()
                    .clone()
                )
                if args_cli.collect_dagger_npz is not None:
                    if last_policy_observation is None:
                        raise RuntimeError("exact Student policy input was not captured")
                    dagger_policy_pre = last_policy_observation
                    dagger_teacher_pre = obs["teacher"].detach().clone()
                    dagger_recovery_pre = recovery_remaining > 0
                # ``env.step`` resets a terminated environment before it
                # returns.  Keep the causal pre-step Hall/proprio input so a
                # future-fall label is never accidentally paired with a reset
                # state.  Prefer the exact tensor consumed by a wrapped
                # policy (for example after a deployable vy estimator).
                collection_pre_obs = None
                if args_cli.collect_npz is not None:
                    collection_pre_obs = exact_actor_observation
                    if collection_pre_obs.shape[1] != 1864:
                        raise ValueError(
                            "matrix --collect_npz requires the deployable "
                            "1864-D policy observation, got "
                            f"{collection_pre_obs.shape}"
                        )
                # Snapshot immediately before stepping because managed envs
                # reset a terminated robot inside env.step().
                pre_root_z = robot.data.root_pos_w[:, 2].clone()
                pre_tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
                pre_abs_wxy = torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)
                pre_abs_vxy = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
                pre_root_vx = robot.data.root_lin_vel_b[:, 0].clone()
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
                managed_resets = dones.bool()
                post_step_command = applied_command_this_step.clone()
                post_step_command[managed_resets] = requested_command_this_step[
                    managed_resets
                ]
                _synchronize_evaluator_command_observation(
                    env,
                    obs,
                    post_step_command,
                    managed_resets,
                )
                timeouts = extras.get("time_outs") if isinstance(extras, dict) else None
                if timeouts is None:
                    timeout_mask = torch.zeros_like(managed_resets)
                else:
                    timeout_mask = timeouts.to(device=dones.device).bool()
                falls = managed_resets & ~timeout_mask
                (
                    at_risk_before_step,
                    primary_sample_mask,
                    first_falls,
                    repeated_falls,
                    ever_failed,
                ) = _first_fall_masks(ever_failed, falls)
                step_fall_events = int(falls.sum().item())
                step_first_falls = int(first_falls.sum().item())
                step_repeated_falls = int(repeated_falls.sum().item())
                step_exposure_s = (
                    float(at_risk_before_step.sum().item()) * evaluation_dt
                )
                falls_total += step_fall_events
                fall_event_count += step_fall_events
                unique_env_first_fall_count += step_first_falls
                failure_free_exposure_s += step_exposure_s
                post_reset_count += step_repeated_falls
                post_reset_sample_count += int(
                    (~primary_sample_mask).sum().item()
                )
                if first_falls.any():
                    first_fall_time_by_env_s[first_falls] = (
                        float(step + 1) * evaluation_dt
                    )
                if step < args_cli.warmup_steps:
                    warmup_fall_event_count += step_fall_events
                    warmup_unique_env_first_fall_count += step_first_falls
                    warmup_failure_free_exposure_s += step_exposure_s
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
                                "cmd_vy": f"{command_vy:.3f}",
                                "cmd_wz": f"{command_wz:.3f}",
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
                        dagger_env_id.append(
                            selected.cpu().numpy().astype(np.int32)
                        )

                recovery_remaining = torch.clamp(
                    recovery_remaining - 1, min=0
                )
                if falls.any():
                    recovery_remaining[falls] = max(
                        args_cli.recovery_steps, 0
                    )
                if hall_governor is not None and managed_resets.any():
                    hall_governor.reset(
                        torch.nonzero(managed_resets, as_tuple=False).flatten()
                    )
                if step < args_cli.warmup_steps:
                    if step == args_cli.warmup_steps - 1:
                        reference_xy = robot.data.root_pos_w[:, :2].clone()
                        reference_yaw = _yaw_from_wxyz(robot.data.root_quat_w).clone()
                    # No warm-up sample is exported, but managed resets still
                    # delimit a causal trajectory for the measured segment.
                    if managed_resets.any():
                        collection_episode_generation[managed_resets] += 1
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
                failure_free_sample_count += int(
                    primary_sample_mask.sum().item()
                )
                vel = robot.data.root_lin_vel_b
                if forward_velocity_estimate is None:
                    forward_velocity_estimate_acc.append(float("nan"))
                    forward_velocity_error_acc.append(float("nan"))
                else:
                    estimate = forward_velocity_estimate.detach()
                    forward_velocity_estimate_acc.append(
                        _masked_tensor_mean(estimate, primary_sample_mask)
                    )
                    forward_velocity_error_acc.append(
                        _masked_tensor_mean(
                            torch.abs(estimate - pre_root_vx),
                            primary_sample_mask,
                        )
                    )
                vx_including_resets_acc.append(vel[:, 0].mean().item())
                vy_including_resets_acc.append(
                    torch.abs(vel[:, 1]).mean().item()
                )
                vx_acc.append(
                    _masked_tensor_mean(vel[:, 0], primary_sample_mask)
                )
                vy_signed_acc.append(
                    _masked_tensor_mean(vel[:, 1], primary_sample_mask)
                )
                vy_acc.append(
                    _masked_tensor_mean(
                        torch.abs(vel[:, 1]), primary_sample_mask
                    )
                )
                wz_acc.append(
                    _masked_tensor_mean(
                        torch.abs(robot.data.root_ang_vel_b[:, 2]),
                        primary_sample_mask,
                    )
                )
                foot_vel = robot.data.body_lin_vel_w[:, foot_body_ids, :2]
                foot_speed = torch.linalg.norm(foot_vel, dim=-1)
                fn = torch.abs(contact_sensor.data.net_forces_w[:, foot_sensor_ids, 2])
                ft = torch.linalg.norm(
                    contact_sensor.data.net_forces_w[:, foot_sensor_ids, :2], dim=-1
                )
                contact = fn > 5.0
                corrected_slip, legacy_slip = _simulator_contact_slip_metrics(
                    robot,
                    foot_body_ids,
                    contact_sensor,
                    foot_sensor_ids,
                    left_dedicated_contact_sensor,
                    right_dedicated_contact_sensor,
                )
                contact_slip_per_env = corrected_slip.speed_per_env
                contact_slip_valid_per_env = corrected_slip.valid_per_env
                legacy_link_origin_planar_slip_per_env = (
                    legacy_slip.speed_per_env
                )
                slip_including_resets_acc.append(
                    _masked_tensor_mean(
                        contact_slip_per_env,
                        contact_slip_valid_per_env,
                    )
                )
                slip_acc.append(
                    _masked_tensor_mean(
                        contact_slip_per_env,
                        primary_sample_mask & contact_slip_valid_per_env,
                    )
                )
                fn_acc.append(
                    _masked_tensor_mean(
                        fn.sum(dim=1), primary_sample_mask
                    )
                )
                ft_acc.append(
                    _masked_tensor_mean(
                        ft.sum(dim=1), primary_sample_mask
                    )
                )
                force_ratio_acc.append(
                    _masked_tensor_mean(
                        (ft / (fn + 5.0)).mean(dim=1),
                        primary_sample_mask,
                    )
                )
                displacement = robot.data.root_pos_w[:, :2] - reference_xy
                # Project world displacement onto the lateral axis at the start
                # of this measured path. This remains valid with randomized yaw.
                local_y = (
                    -torch.sin(reference_yaw) * displacement[:, 0]
                    + torch.cos(reference_yaw) * displacement[:, 1]
                )
                lateral_pos_acc.append(
                    _masked_tensor_mean(torch.abs(local_y), primary_sample_mask)
                )
                lateral_pos_signed_acc.append(
                    _masked_tensor_mean(local_y, primary_sample_mask)
                )
                action_acc.append(
                    _masked_tensor_mean(
                        torch.abs(actions).mean(dim=1), primary_sample_mask
                    )
                )
                if governed_command is not None:
                    applied_command_acc.append(
                        _masked_tensor_mean(
                            governed_command[:, 0], primary_sample_mask
                        )
                    )
                    applied_vy_command_acc.append(
                        _masked_tensor_mean(
                            governed_command[:, 1], primary_sample_mask
                        )
                    )
                    applied_wz_command_acc.append(
                        _masked_tensor_mean(
                            governed_command[:, 2], primary_sample_mask
                        )
                    )
                    risk_probability_acc.append(
                        _masked_tensor_mean(
                            low_probability, primary_sample_mask
                        )
                    )
                    if primary_sample_mask.any():
                        risk_probability_p90_acc.append(
                            torch.quantile(
                                low_probability[primary_sample_mask], 0.90
                            ).item()
                        )
                    else:
                        risk_probability_p90_acc.append(float("nan"))
                    low_state_fraction_acc.append(
                        _masked_tensor_mean(
                            (governor_state == LOW).float(),
                            primary_sample_mask,
                        )
                    )
                else:
                    has_primary = bool(primary_sample_mask.any().item())
                    applied_command_acc.append(
                        float(requested_vx) if has_primary else float("nan")
                    )
                    applied_vy_command_acc.append(
                        float(requested_vy) if has_primary else float("nan")
                    )
                    applied_wz_command_acc.append(
                        float(requested_wz) if has_primary else float("nan")
                    )
                    risk_probability_acc.append(float("nan"))
                    risk_probability_p90_acc.append(float("nan"))
                    low_state_fraction_acc.append(0.0)
                if args_cli.collect_npz is not None:
                    assert collection_pre_obs is not None
                    scheduled = step % max(args_cli.collect_stride, 1) == 0
                    # Include every scheduled state plus off-stride pre-fall
                    # states.  The latter are essential causal supervision for
                    # the risk head and were previously dropped.
                    selected = (
                        torch.arange(args_cli.num_envs, device=dones.device)
                        if scheduled
                        else torch.nonzero(falls, as_tuple=False).flatten()
                    )
                    if selected.numel() > 0:
                        selected_count = int(selected.numel())
                        collected_obs.append(
                            collection_pre_obs.index_select(0, selected)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_mu.append(
                            np.full(selected_count, mu, dtype=np.float32)
                        )
                        collected_cmd.append(
                            np.full(selected_count, vx, dtype=np.float32)
                        )
                        collected_seed.append(
                            np.full(selected_count, args_cli.seed, dtype=np.int32)
                        )
                        collected_root_lin_vel_b.append(
                            robot.data.root_lin_vel_b.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_root_ang_vel_b.append(
                            robot.data.root_ang_vel_b.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_actor_command.append(
                            actor_command_this_step.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_applied_command.append(
                            applied_command_this_step.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_contact_slip.append(
                            contact_slip_per_env.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_contact_slip_valid.append(
                            contact_slip_valid_per_env.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                        collected_legacy_link_origin_planar_slip.append(
                            legacy_link_origin_planar_slip_per_env.index_select(
                                0, selected
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        # A fall labels the *pre-step* observation; it is not
                        # an invalid Hall packet.  Keep it for prospective
                        # supervision instead of filtering it out on load.
                        collected_valid.append(
                            np.ones(selected_count, dtype=np.bool_)
                        )
                        collected_env_id.append(
                            selected.detach().cpu().numpy().astype(np.int32)
                        )
                        collected_step.append(
                            np.full(
                                selected_count,
                                step - args_cli.warmup_steps,
                                dtype=np.int32,
                            )
                        )
                        collected_fall.append(
                            falls.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                        collected_done.append(
                            managed_resets.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                        collected_time_out.append(
                            timeout_mask.index_select(0, selected)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                        collected_hall_valid_lr.append(
                            collection_pre_obs.index_select(0, selected)[
                                :, 1860:1862
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        collected_rollout_id.append(
                            (
                                rollout_id * 1_000_000
                                + collection_episode_generation.index_select(
                                    0, selected
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        )
                # Advance after exporting the causal pre-fall sample.  The
                # next managed-reset observation receives a new trajectory id
                # even though the evaluator's global step counter continues.
                if managed_resets.any():
                    collection_episode_generation[managed_resets] += 1
            finite_first_fall_times = first_fall_time_by_env_s[
                torch.isfinite(first_fall_time_by_env_s)
            ]
            time_to_first_fall_s = (
                float(finite_first_fall_times.min().item())
                if finite_first_fall_times.numel() > 0
                else float("nan")
            )
            mean = _finite_mean
            row = {
                "mu": f"{mu:.3f}",
                "cmd_vx": f"{vx:.3f}",
                "cmd_vy": f"{command_vy:.3f}",
                "cmd_wz": f"{command_wz:.3f}",
                "mean_applied_vx_command": f"{mean(applied_command_acc):.4f}",
                "mean_applied_vy_command": f"{mean(applied_vy_command_acc):.4f}",
                "mean_applied_wz_command": f"{mean(applied_wz_command_acc):.4f}",
                "mean_low_traction_probability": (
                    f"{mean(risk_probability_acc):.4f}"
                    if hall_governor is not None
                    else "nan"
                ),
                "mean_p90_low_traction_probability": (
                    f"{mean(risk_probability_p90_acc):.4f}"
                    if hall_governor is not None
                    else "nan"
                ),
                "mean_low_state_fraction": f"{mean(low_state_fraction_acc):.4f}",
                "mean_vx": f"{mean(vx_acc):.4f}",
                "mean_vx_including_resets": (
                    f"{mean(vx_including_resets_acc):.4f}"
                ),
                "mean_vy": f"{mean(vy_signed_acc):.4f}",
                "mean_abs_vy": f"{mean(vy_acc):.4f}",
                "mean_abs_vy_including_resets": (
                    f"{mean(vy_including_resets_acc):.4f}"
                ),
                "mean_abs_wz": f"{mean(wz_acc):.4f}",
                "mean_forward_velocity_estimate": f"{mean(forward_velocity_estimate_acc):.4f}",
                "mean_forward_velocity_abs_error": f"{mean(forward_velocity_error_acc):.4f}",
                "mean_contact_slip": f"{mean(slip_acc):.4f}",
                "mean_contact_slip_including_resets": (
                    f"{mean(slip_including_resets_acc):.4f}"
                ),
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
                "fall_event_count": fall_event_count,
                "unique_env_first_fall_count": (
                    unique_env_first_fall_count
                ),
                "time_to_first_fall_s": f"{time_to_first_fall_s:.4f}",
                "failure_free_exposure_s": (
                    f"{failure_free_exposure_s:.4f}"
                ),
                "post_reset_count": post_reset_count,
                "post_reset_sample_count": post_reset_sample_count,
                "failure_free_sample_count": failure_free_sample_count,
                "warmup_fall_event_count": warmup_fall_event_count,
                "warmup_unique_env_first_fall_count": (
                    warmup_unique_env_first_fall_count
                ),
                "warmup_failure_free_exposure_s": (
                    f"{warmup_failure_free_exposure_s:.4f}"
                ),
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
            root_lin_vel_b=np.concatenate(collected_root_lin_vel_b, axis=0),
            root_ang_vel_b=np.concatenate(collected_root_ang_vel_b, axis=0),
            actor_command=np.concatenate(collected_actor_command, axis=0),
            applied_command=np.concatenate(collected_applied_command, axis=0),
            contact_slip=np.concatenate(collected_contact_slip, axis=0),
            contact_point_tangent_slip=np.concatenate(
                collected_contact_slip, axis=0
            ),
            contact_point_tangent_slip_valid=np.concatenate(
                collected_contact_slip_valid, axis=0
            ),
            legacy_link_origin_planar_slip=np.concatenate(
                collected_legacy_link_origin_planar_slip, axis=0
            ),
            valid=np.concatenate(collected_valid, axis=0),
            env_id=np.concatenate(collected_env_id, axis=0),
            step=np.concatenate(collected_step, axis=0),
            fall=np.concatenate(collected_fall, axis=0),
            done=np.concatenate(collected_done, axis=0),
            time_out=np.concatenate(collected_time_out, axis=0),
            hall_valid_lr=np.concatenate(collected_hall_valid_lr, axis=0),
            rollout_id=np.concatenate(collected_rollout_id, axis=0),
            **_collection_metadata(
                dataset_kind="matrix",
                task=args_cli.task,
                seed=args_cli.seed,
                policy_dt=float(env.unwrapped.step_dt),
                collect_stride=args_cli.collect_stride,
                actor_checkpoint=_selected_actor_checkpoint(args_cli)[0],
                actor_source=_selected_actor_checkpoint(args_cli)[1],
                hall_contact_distribution_mode=getattr(
                    args_cli,
                    "effective_hall_contact_distribution_mode",
                    "unknown",
                ),
            ),
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
            env_id=np.concatenate(dagger_env_id, axis=0),
            hall_contact_distribution_mode=np.asarray(
                getattr(
                    args_cli,
                    "effective_hall_contact_distribution_mode",
                    "unknown",
                ),
                dtype=np.str_,
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
