#!/usr/bin/env python3
"""Evaluate HighEnd recovery on a leak-free validation state bank.

The actor input remains the audited 1864-D Hall Bx/By/Bz + proprioception
tensor.  Bank labels and termination state are evaluator-only.  Statistics
are censored permanently after each environment's first terminal event.

Run the same task/seed twice: omit ``--checkpoint`` for the frozen high-speed
backbone, then supply a recovery checkpoint for a paired comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import sys

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
    "SpatialFrictionCadenceStrideHighEndRecoveryExpert"
)
POLICY_DIM = 1864
ACTION_DIM = 29

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK)
parser.add_argument("--state_bank", type=Path, required=True)
parser.add_argument(
    "--state_bank_role", default="validation_high_end_state_bank"
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=540)
parser.add_argument("--summary_json", type=Path, required=True)
parser.add_argument("--fail_on_gate", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num_envs and --steps must be positive")
if args_cli.state_bank_role != "validation_high_end_state_bank":
    parser.error("formal screening requires validation_high_end_state_bank")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.traction.high_end_state_bank import (  # noqa: E402
    VALIDATION_ROLE,
    load_high_end_state_bank,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_tensor(observation) -> torch.Tensor:
    value = observation["policy"]
    if value.ndim != 2 or value.shape[1] != POLICY_DIM:
        raise RuntimeError(
            f"actor policy must be [N,{POLICY_DIM}], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise FloatingPointError("actor policy contains NaN/Inf")
    return value


def _fall_mask(done: torch.Tensor, extras) -> tuple[torch.Tensor, torch.Tensor]:
    timeout = extras.get("time_outs") if isinstance(extras, dict) else None
    timeout = (
        torch.zeros_like(done, dtype=torch.bool)
        if timeout is None
        else timeout.to(device=done.device, dtype=torch.bool)
    )
    return done.bool() & ~timeout, done.bool() & timeout


def _safe_mean(value: np.ndarray) -> float | None:
    return float(value.mean()) if value.size else None


def _rms(sum_square: torch.Tensor, count: torch.Tensor) -> np.ndarray:
    return torch.sqrt(sum_square / count.clamp_min(1)).cpu().numpy()


def _aggregate(mask: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, object]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "fall_count": int(values["fallen"][mask].sum()),
        "timeout_retention_count": int(values["timed_out"][mask].sum()),
        "retention_fraction": float(values["timed_out"][mask].mean()),
        "mean_survival_s": float(values["survival_s"][mask].mean()),
        "mean_body_vx_m_s": _safe_mean(values["mean_vx"][mask]),
        "heading_rms_rad": _safe_mean(values["heading_rms"][mask]),
        "body_vy_rms_m_s": _safe_mean(values["vy_rms"][mask]),
        "angular_velocity_rms_rad_s": _safe_mean(values["omega_rms"][mask]),
        "tilt_rms": _safe_mean(values["tilt_rms"][mask]),
        "action_slew_rms": _safe_mean(values["action_slew_rms"][mask]),
        "action_saturation_fraction": float(
            values["saturation_fraction"][mask].mean()
        ),
        "recovered_and_held_fraction": float(
            values["recovered_and_held"][mask].mean()
        ),
    }


def main() -> int:
    bank_path = args_cli.state_bank.expanduser().resolve()
    bank = load_high_end_state_bank(
        bank_path, device="cpu", allowed_roles=(VALIDATION_ROLE,)
    )
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.events.reset_base.params["state_bank_path"] = str(bank_path)
    env_cfg.events.reset_base.params["state_bank_required_role"] = VALIDATION_ROLE
    # Evaluation keeps the source-bank Hall electronics state, restored by the
    # specialized environment. No config-side sensor fault can overwrite it.
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    policy_kind = "frozen_high_speed_backbone"
    checkpoint_path: Path | None = None
    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        runner.load(
            str(checkpoint_path),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        policy_kind = "high_end_recovery_candidate"
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation = env.get_observations()
    policy_obs = _policy_tensor(observation)
    base = env.unwrapped
    if type(base).__name__ != "HighEndRecoveryRLEnv":
        raise RuntimeError("task did not instantiate HighEndRecoveryRLEnv")
    if not isinstance(getattr(base, "_high_end_recovery_last_audit", None), dict):
        raise RuntimeError("complete state-reset runtime audit did not execute")
    initial_reset_audit = json.loads(
        json.dumps(base._high_end_recovery_last_audit, allow_nan=False)
    )
    sample_ids = base._high_end_recovery_last_sample_ids.detach().clone()
    if bool((sample_ids < 0).any().item()):
        raise RuntimeError("initial validation-bank sample ids are incomplete")
    state_kind = bank.arrays["state_kind"].index_select(
        0, sample_ids.cpu()
    ).to(device=base.device)
    source_seed = bank.arrays["source_seed"].index_select(
        0, sample_ids.cpu()
    ).to(device=base.device)
    robot = base.scene["robot"]
    n = base.num_envs
    active = torch.ones(n, dtype=torch.bool, device=base.device)
    fallen = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    first_fall_step = torch.full((n,), -1, dtype=torch.long, device=base.device)
    terminal_step = torch.full_like(first_fall_step, -1)
    sums = {
        name: torch.zeros(n, device=base.device)
        for name in ("vx", "heading2", "vy2", "omega2", "tilt2", "slew2")
    }
    count = torch.zeros(n, dtype=torch.long, device=base.device)
    saturation_count = torch.zeros(n, device=base.device)
    action_element_count = torch.zeros(n, device=base.device)
    recovery_streak = torch.zeros(n, dtype=torch.long, device=base.device)
    recovered_and_held = torch.zeros(n, dtype=torch.bool, device=base.device)
    previous_action = policy_obs[:, 335:480].reshape(n, 5, ACTION_DIM)[:, -1]
    dt = float(base.step_dt)
    required_recovery_steps = max(int(round(1.0 / dt)), 1)

    for step in range(int(args_cli.steps)):
        policy_obs = _policy_tensor(observation)
        with torch.inference_mode():
            action = policy(observation)
        if action.shape != (n, ACTION_DIM) or not torch.isfinite(action).all():
            raise FloatingPointError("candidate action ABI/finite check failed")
        pre_active = active.clone()
        heading = policy_obs[:, 1863]
        vy = robot.data.root_lin_vel_b[:, 1]
        vx = robot.data.root_lin_vel_b[:, 0]
        omega = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=1)
        gravity = policy_obs[:, 27:30]
        tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        slew = torch.linalg.vector_norm(action - previous_action, dim=1)
        previous_action = action.detach().clone()
        for name, value in (
            ("vx", vx),
            ("heading2", heading.square()),
            ("vy2", vy.square()),
            ("omega2", omega.square()),
            ("tilt2", tilt.square()),
            ("slew2", slew.square()),
        ):
            sums[name] += torch.where(pre_active, value, torch.zeros_like(value))
        count += pre_active.long()
        saturation_count += torch.where(
            pre_active,
            (action.abs() >= 2.9).sum(dim=1).float(),
            torch.zeros(n, device=base.device),
        )
        action_element_count += pre_active.float() * ACTION_DIM
        recovered_now = (
            (heading.abs() < 0.20)
            & (vy.abs() < 0.25)
            & (omega < 1.0)
            & (tilt < 0.15)
            & (vx >= 0.69)
        )
        recovery_streak = torch.where(
            pre_active & recovered_now,
            recovery_streak + 1,
            torch.zeros_like(recovery_streak),
        )
        recovered_and_held |= recovery_streak >= required_recovery_steps
        observation, _, done, extras = env.step(action)
        fall, timeout = _fall_mask(done, extras)
        new_fall = pre_active & fall
        new_timeout = pre_active & timeout
        first_fall_step[new_fall] = step + 1
        terminal_step[new_fall | new_timeout] = step + 1
        fallen |= new_fall
        timed_out |= new_timeout
        active &= ~done.bool()
        # Managed reset samples are deliberately ignored for the rest of this
        # process.  Stop early only when every first episode has terminated.
        if not bool(active.any().item()):
            break

    terminal_step = torch.where(
        terminal_step >= 0,
        terminal_step,
        torch.full_like(terminal_step, int(args_cli.steps)),
    )
    mean_vx = (sums["vx"] / count.clamp_min(1)).cpu().numpy()
    arrays = {
        "fallen": fallen.cpu().numpy(),
        "timed_out": timed_out.cpu().numpy(),
        "survival_s": (terminal_step.float() * dt).cpu().numpy(),
        "mean_vx": mean_vx,
        "heading_rms": _rms(sums["heading2"], count),
        "vy_rms": _rms(sums["vy2"], count),
        "omega_rms": _rms(sums["omega2"], count),
        "tilt_rms": _rms(sums["tilt2"], count),
        "action_slew_rms": _rms(sums["slew2"], count),
        "saturation_fraction": (
            saturation_count / action_element_count.clamp_min(1)
        ).cpu().numpy(),
        "recovered_and_held": recovered_and_held.cpu().numpy(),
    }
    kinds = state_kind.cpu().numpy()
    overall = _aggregate(np.ones(n, dtype=bool), arrays)
    by_kind = {
        "nominal_retention": _aggregate(kinds == 0, arrays),
        "near_failure_recovery": _aggregate(kinds == 1, arrays),
    }
    gate_pass = bool(
        overall["fall_count"] == 0
        and overall["timeout_retention_count"] == n
        and overall["mean_body_vx_m_s"] is not None
        and overall["mean_body_vx_m_s"] >= 0.69
        and overall["action_saturation_fraction"] <= 0.05
    )
    report = {
        "format": "high-end-recovery-bank-eval-v1",
        "status": "PASS" if gate_pass else "FAIL",
        "diagnostic_until_paired_multi_seed_gate": True,
        "task": args_cli.task,
        "policy_kind": policy_kind,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "state_bank": str(bank_path),
        "state_bank_sha256": _sha256(bank_path),
        "state_bank_role": args_cli.state_bank_role,
        "source_seeds_sampled": sorted(set(source_seed.cpu().tolist())),
        "seed": int(args_cli.seed),
        "num_envs": n,
        "requested_steps": int(args_cli.steps),
        "step_dt_s": dt,
        "actor_boundary": {
            "dimension": POLICY_DIM,
            "measurement": "multi-frame Hall Bx/By/Bz + packet metadata + proprioception",
            "uses_force_contact_mu_slip_or_stage": False,
        },
        "censoring": "first episode only; terminal and every post-reset sample excluded",
        "complete_reset_audit": initial_reset_audit,
        "overall": overall,
        "by_state_kind": by_kind,
        "gate": {
            "zero_falls": True,
            "all_timeout_retention": True,
            "minimum_mean_vx_m_s": 0.69,
            "maximum_action_saturation_fraction": 0.05,
            "pass": gate_pass,
        },
        "per_env": [
            {
                "env_id": i,
                "bank_sample_id": int(sample_ids[i]),
                "source_seed": int(source_seed[i]),
                "state_kind": int(kinds[i]),
                "fall": bool(arrays["fallen"][i]),
                "timeout_retention": bool(arrays["timed_out"][i]),
                "survival_s": float(arrays["survival_s"][i]),
                "mean_vx_m_s": float(arrays["mean_vx"][i]),
                "heading_rms_rad": float(arrays["heading_rms"][i]),
                "body_vy_rms_m_s": float(arrays["vy_rms"][i]),
                "angular_velocity_rms_rad_s": float(arrays["omega_rms"][i]),
                "action_saturation_fraction": float(
                    arrays["saturation_fraction"][i]
                ),
                "recovered_and_held": bool(arrays["recovered_and_held"][i]),
            }
            for i in range(n)
        ],
    }
    output = args_cli.summary_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **overall}, indent=2))
    env.close()
    return 2 if args_cli.fail_on_gate and not gate_pass else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
