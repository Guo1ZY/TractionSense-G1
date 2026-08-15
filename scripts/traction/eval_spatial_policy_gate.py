#!/usr/bin/env python3
"""Run and rank a same-seed Hall H-L-H policy matrix.

This is an orchestration/selection layer around the Isaac evaluator.  It does
not import or modify the training environment.  Every candidate sees the same
task, seeds, number of environments, Hall profile and rollout length.

Example::

    python3 scripts/traction/eval_spatial_policy_gate.py \
      --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMild \
      --candidate speedboost=onnx:artifacts/hall_speed_demo/speedboost112_dynamic.onnx \
      --candidate s1_100=checkpoint:logs/rsl_rl/.../model_100.pt \
      --candidate safe6350=checkpoint:logs/rsl_rl/.../model_6350.pt \
      --seed 396 397 --num_envs 16 --steps 400 \
      --output artifacts/hall_speed_demo/same_seed_gate/report.json

Hardened Hall is the default.  Use ``--nominal_hall`` only for an explicitly
labelled diagnostic; nominal and hardened summaries are never merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"
DEFAULT_TASK = (
    "Unitree-G1-29dof-Velocity-Foot-"
    "TractionMagneticMotionStudent-SpatialFriction"
)


def _candidate(value: str) -> tuple[str, str, Path]:
    try:
        label, specification = value.split("=", 1)
        kind, raw_path = specification.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=onnx:PATH or LABEL=checkpoint:PATH"
        ) from exc
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"invalid candidate label: {label!r}")
    if kind not in ("onnx", "checkpoint", "torchscript"):
        raise argparse.ArgumentTypeError(f"unsupported candidate kind: {kind!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"candidate file does not exist: {path}")
    return label, kind, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _hardened_audit(summary: dict[str, Any], hardened: bool) -> tuple[bool, list[str]]:
    errors: list[str] = []
    profile = summary.get("hall_fault_profile", {})
    effective = summary.get("effective_hall_config", {})
    if bool(profile.get("requested_hardened")) != hardened:
        errors.append("requested_hardened mismatch")
    if bool(profile.get("domain_randomization_enabled")) != hardened:
        errors.append("domain_randomization_enabled mismatch")
    if bool(effective.get("enable_domain_randomization")) != hardened:
        errors.append("effective Hall DR mismatch")
    if hardened:
        expected = {
            "foot_dropout_probability": 0.10,
            "dead_channel_probability": 0.08,
            "maximum_packet_delay_steps": 5,
        }
        for field, value in expected.items():
            if profile.get(field) != value or effective.get(field) != value:
                errors.append(f"effective {field} != {value}")
        sampled = (
            summary.get("natural_rollout", {})
            .get("initial_hall_fault_state", {})
            .get("sampled_statistics", {})
        )
        if not sampled:
            errors.append("missing sampled initial Hall fault statistics")
        elif sampled.get("normal_stiffness_scale", {}).get("std", 0.0) <= 0.0:
            errors.append("hardened mechanical randomization did not vary")
    rollout = summary.get("natural_rollout", {})
    if not rollout.get("first_episode_only", False):
        errors.append("rollout is not first-episode-only")
    if rollout.get("transition_response", {}).get("definition") != "first-episode-causal-response-v1":
        errors.append("missing causal response metrics")
    return not errors, errors


def _candidate_aggregate(
    label: str,
    kind: str,
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    hardened: bool,
    thresholds: argparse.Namespace,
) -> dict[str, Any]:
    audits = [_hardened_audit(summary, hardened) for summary in summaries]
    rollouts = [summary["natural_rollout"] for summary in summaries]
    responses = [rollout["transition_response"] for rollout in rollouts]
    total_envs = sum(int(summary["num_envs"]) for summary in summaries)
    fall_envs = sum(int(rollout["fall_envs"]) for rollout in rollouts)
    fall_events = sum(int(rollout["fall_events"]) for rollout in rollouts)
    completed = sum(int(rollout["completed_hlh_envs"]) for rollout in rollouts)

    speeds = {
        region: _mean([rollout["mean_body_vx_m_s"].get(region) for rollout in rollouts])
        for region in ("high_start", "low", "high_end")
    }
    decel_05 = _mean(
        [
            response["deceleration_after_low_contact"]["0.5s"]["deceleration_m_s"]["mean"]
            for response in responses
        ]
    )
    low_vx_10 = _mean(
        [
            response["deceleration_after_low_contact"]["1s"]["vx_m_s"]["mean"]
            for response in responses
        ]
    )
    low_10_survival = _mean(
        [
            response["deceleration_after_low_contact"]["1s"]["survival_fraction"]
            for response in responses
        ]
    )
    recovery_fraction = _mean(
        [response["absolute_high_recovery"]["recovery_fraction"] for response in responses]
    )
    recovery_time = _mean(
        [response["absolute_high_recovery"]["time_s"]["mean"] for response in responses]
    )
    first_falls = [response["first_fall_time_s"] for response in responses]

    checks = {
        "effective_config_valid": all(valid for valid, _ in audits),
        "nan_free": all(not bool(rollout["nan_detected"]) for rollout in rollouts),
        "zero_fall": fall_envs == 0 and fall_events == 0,
        "hlh_completion": completed / max(total_envs, 1) >= thresholds.min_completion_fraction,
        "high_start_speed": speeds["high_start"] is not None
        and speeds["high_start"] >= thresholds.min_high_start_speed,
        "low_speed_by_1s": low_vx_10 is not None and low_vx_10 <= thresholds.max_low_speed_1s,
        "deceleration_by_0_5s": decel_05 is not None
        and decel_05 >= thresholds.min_deceleration_0_5s,
        "low_1s_survival": low_10_survival is not None
        and low_10_survival >= thresholds.min_low_1s_survival,
        "high_recovery_fraction": recovery_fraction is not None
        and recovery_fraction >= thresholds.min_recovery_fraction,
        "high_recovery_time": recovery_time is not None
        and recovery_time <= thresholds.max_recovery_time,
    }
    # Safety, configuration integrity and completion are hard selection gates.
    # Performance checks remain explicit so a candidate cannot hide behind a
    # single weighted score.
    formal_pass = all(checks.values())
    return {
        "label": label,
        "kind": kind,
        "path": str(path),
        "sha256": _sha256(path),
        "seeds": [int(summary["seed"]) for summary in summaries],
        "total_env_rollouts": total_envs,
        "fall_envs": fall_envs,
        "fall_events": fall_events,
        "first_fall_time_s_by_seed": first_falls,
        "completed_hlh_envs": completed,
        "completion_fraction": completed / max(total_envs, 1),
        "mean_body_vx_m_s": speeds,
        "mean_deceleration_0_5s_m_s": decel_05,
        "mean_low_vx_1_0s_m_s": low_vx_10,
        "low_1_0s_survival_fraction": low_10_survival,
        "absolute_high_recovery_fraction": recovery_fraction,
        "mean_absolute_high_recovery_time_s": recovery_time,
        "checks": checks,
        "formal_pass": formal_pass,
        "effective_config_errors": [error for _, errors in audits for error in errors],
        "hall_health_by_seed": [
            rollout["hall_health_performance"] for rollout in rollouts
        ],
    }


def _score(row: dict[str, Any]) -> tuple[float, ...]:
    speed = row["mean_body_vx_m_s"]
    return (
        float(row["formal_pass"]),
        float(row["checks"]["effective_config_valid"]),
        -float(row["fall_envs"]),
        float(row["completion_fraction"]),
        float(row["mean_deceleration_0_5s_m_s"] or -99.0),
        float(speed["high_start"] or -99.0),
        -float(row["mean_absolute_high_recovery_time_s"] or 99.0),
    )


def _format_metric(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Hall spatial same-seed gate",
        "",
        f"- Profile: `{report['hall_profile']}`",
        f"- Task: `{report['task']}`",
        f"- Seeds: `{report['seeds']}`; envs/seed: `{report['num_envs_per_seed']}`",
        f"- Formal pass: `{report['formal_pass_candidates']}`",
        "",
        "| rank | candidate | falls | H-L-H | H/L/H mean vx (m/s) | low vx @1s | decel @0.5s | recovery to 0.70 | gate |",
        "|---:|---|---:|---:|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(report["ranking"], start=1):
        speed = row["mean_body_vx_m_s"]
        recovery = (
            f"{_format_metric(row['mean_absolute_high_recovery_time_s'])} s / "
            f"{_format_metric(row['absolute_high_recovery_fraction'])}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(rank),
                    str(row["label"]),
                    f"{row['fall_envs']}/{row['total_env_rollouts']}",
                    f"{row['completed_hlh_envs']}/{row['total_env_rollouts']}",
                    "/".join(_format_metric(speed[name]) for name in ("high_start", "low", "high_end")),
                    _format_metric(row["mean_low_vx_1_0s_m_s"]),
                    _format_metric(row["mean_deceleration_0_5s_m_s"]),
                    recovery,
                    "PASS" if row["formal_pass"] else "FAIL",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "`decel @0.5s = low-entry vx - vx after 0.5 s`; negative values mean the actor accelerated.",
            "Recovery requires 0.70 m/s for the configured consecutive-frame hold and is meaningful only together with the low-speed gate.",
            "Every speed and response statistic is restricted to the first episode; post-reset frames are excluded.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--seed", nargs="+", type=int, default=[396])
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--command", type=float, default=0.80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--python_executable",
        type=Path,
        default=Path("/home/mosense/miniconda3/envs/isaaclab-v2/bin/python"),
        help="Python with Isaac Lab and onnxruntime installed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal_hall", action="store_true")
    parser.add_argument("--with_label_probe", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Reuse valid per-run JSON files.")
    parser.add_argument("--require_pass", action="store_true")
    parser.add_argument("--min_completion_fraction", type=float, default=1.0)
    parser.add_argument("--min_high_start_speed", type=float, default=0.60)
    parser.add_argument("--max_low_speed_1s", type=float, default=0.55)
    parser.add_argument("--min_deceleration_0_5s", type=float, default=0.05)
    parser.add_argument("--min_low_1s_survival", type=float, default=0.95)
    parser.add_argument("--min_recovery_fraction", type=float, default=0.80)
    parser.add_argument("--max_recovery_time", type=float, default=1.50)
    args = parser.parse_args()

    if args.num_envs <= 0 or args.steps <= 0 or not args.seed:
        parser.error("num_envs, steps and seed list must be non-empty/positive")
    if not args.python_executable.is_file():
        parser.error(f"Isaac Python does not exist: {args.python_executable}")
    labels = [candidate[0] for candidate in args.candidate]
    if len(set(labels)) != len(labels):
        parser.error("candidate labels must be unique")

    output = args.output.expanduser().resolve()
    run_dir = output.parent / f"{output.stem}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    hardened = not args.nominal_hall
    candidate_summaries: dict[str, list[dict[str, Any]]] = {
        label: [] for label in labels
    }
    for label, kind, path in args.candidate:
        for seed in args.seed:
            summary_path = run_dir / f"{label}_seed{seed}.json"
            log_path = run_dir / f"{label}_seed{seed}.log"
            reuse_ok = False
            if args.reuse and summary_path.is_file():
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                reuse_ok, _ = _hardened_audit(payload, hardened)
            if not reuse_ok:
                command = [
                    str(args.python_executable),
                    str(EVALUATOR),
                    "--headless",
                    "--task",
                    args.task,
                    "--num_envs",
                    str(args.num_envs),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(seed),
                    "--command",
                    str(args.command),
                    "--device",
                    args.device,
                    "--summary_json",
                    str(summary_path),
                    f"--{kind}",
                    str(path),
                ]
                if hardened:
                    command.append("--hardened_hall")
                if not args.with_label_probe:
                    command.append("--skip_label_probe")
                print(f"[gate] running {label} seed={seed} ({kind})", flush=True)
                child_env = os.environ.copy()
                source_path = str(ROOT / "source" / "unitree_rl_lab")
                existing_pythonpath = child_env.get("PYTHONPATH", "")
                child_env["PYTHONPATH"] = (
                    source_path
                    if not existing_pythonpath
                    else f"{source_path}:{existing_pythonpath}"
                )
                # Isaac Lab's shell wrapper calls `tabs`; a concrete terminal
                # type avoids its non-interactive `dumb`-terminal failure.
                child_env.setdefault("TERM", "xterm")
                if child_env["TERM"] == "dumb":
                    child_env["TERM"] = "xterm"
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0 or not summary_path.is_file():
                    print(f"[gate] evaluator failed; inspect {log_path}", file=sys.stderr)
                    return 2
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["seed"] = seed
            payload["candidate_label"] = label
            candidate_summaries[label].append(payload)

    aggregates = []
    for label, kind, path in args.candidate:
        aggregates.append(
            _candidate_aggregate(
                label,
                kind,
                path,
                candidate_summaries[label],
                hardened=hardened,
                thresholds=args,
            )
        )
    ranking = sorted(aggregates, key=_score, reverse=True)
    report = {
        "format": "hall-spatial-same-seed-selection-gate-v1",
        "task": args.task,
        "hall_profile": "nominal" if args.nominal_hall else "true_hardened",
        "seeds": args.seed,
        "num_envs_per_seed": args.num_envs,
        "steps": args.steps,
        "command_m_s": args.command,
        "thresholds": {
            key: getattr(args, key)
            for key in (
                "min_completion_fraction",
                "min_high_start_speed",
                "max_low_speed_1s",
                "min_deceleration_0_5s",
                "min_low_1s_survival",
                "min_recovery_fraction",
                "max_recovery_time",
            )
        },
        "formal_pass_candidates": [row["label"] for row in ranking if row["formal_pass"]],
        "ranking": ranking,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output.with_suffix(".md"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.require_pass and not report["formal_pass_candidates"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
