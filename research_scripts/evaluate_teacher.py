#!/usr/bin/env python3
"""Evaluate the newest privileged traction teacher and judge acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


LAB = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
EXPERIMENT_ROOT = (
    LAB / "logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher"
)
EVALUATOR = LAB / "scripts/rsl_rl/eval_friction_matrix.py"
EVALUATION_ROOT = LAB / "logs/evaluations/traction_teacher"
TASK = "Unitree-G1-29dof-Velocity-Foot-TractionTeacher"


@dataclass
class Gate:
    name: str
    value: float
    target: str
    passed: bool


class Palette:
    def __init__(self, enabled: bool):
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.green = "\033[38;5;82m" if enabled else ""
        self.red = "\033[38;5;196m" if enabled else ""
        self.yellow = "\033[38;5;220m" if enabled else ""
        self.cyan = "\033[38;5;45m" if enabled else ""
        self.gray = "\033[38;5;240m" if enabled else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a friction x speed matrix for the latest TractionTeacher checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint; default: newest complete model")
    parser.add_argument("--task", default=TASK, help=f"Isaac task id (default: {TASK})")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Checkpoint search root when --checkpoint is omitted",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=EVALUATION_ROOT,
        help="Directory for generated matrix reports",
    )
    parser.add_argument("--device", default="cpu", help="Isaac device (default: cpu, safe while training)")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--command-ramp-steps",
        type=int,
        default=-1,
        help="0 for an abrupt step; negative ramps across the full warmup",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mus", nargs="+", type=float, default=[0.08, 0.20, 0.40, 0.80, 1.20])
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--quick", action="store_true", help="Fast 2-friction x 2-speed smoke matrix")
    parser.add_argument("--strict", action="store_true", help="Exit 3 when any acceptance gate fails")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--report-only",
        type=Path,
        metavar="CSV",
        help="Skip Isaac and only assess an existing matrix CSV",
    )
    return parser.parse_args()


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(experiment_root: Path = EXPERIMENT_ROOT) -> Path:
    candidates = sorted(
        experiment_root.rglob("model_*.pt"),
        key=lambda path: (path.stat().st_mtime, checkpoint_iteration(path)),
        reverse=True,
    )
    for path in candidates:
        first = path.stat().st_size
        time.sleep(0.05)
        second = path.stat().st_size
        if first >= 1_000_000 and first == second:
            return path.resolve()
    raise FileNotFoundError(f"no complete checkpoint below {experiment_root}")


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"empty evaluation matrix: {path}")
    return rows


def cell(rows: list[dict[str, float]], mu: float, speed: float) -> dict[str, float]:
    candidates = [
        row for row in rows if abs(row["mu"] - mu) < 1.0e-4 and abs(row["cmd_vx"] - speed) < 1.0e-4
    ]
    if not candidates:
        raise KeyError(f"matrix is missing mu={mu}, cmd={speed}")
    return candidates[0]


def assess(rows: list[dict[str, float]]) -> list[Gate]:
    low_slow = cell(rows, 0.08, 0.5)
    low_fast = cell(rows, 0.08, 1.5)
    high_slow = cell(rows, 1.20, 0.5)
    high_fast = cell(rows, 1.20, 1.5)
    max_fall = max(row["fall_per_env"] for row in rows)
    max_vy = max(row["mean_abs_vy"] for row in rows)
    max_wz = max(row["mean_abs_wz"] for row in rows)
    separation = high_fast["mean_vx"] - low_fast["mean_vx"]
    return [
        Gate("全矩阵摔倒率", max_fall, "= 0", max_fall <= 1.0e-6),
        Gate("低摩擦 cmd=0.5 限速", low_slow["mean_vx"], "0.15..0.45 m/s", 0.15 <= low_slow["mean_vx"] <= 0.45),
        Gate("低摩擦 cmd=1.5 限速", low_fast["mean_vx"], "0.15..0.45 m/s", 0.15 <= low_fast["mean_vx"] <= 0.45),
        Gate("高摩擦 cmd=0.5 跟踪", high_slow["mean_vx"], "0.38..0.65 m/s", 0.38 <= high_slow["mean_vx"] <= 0.65),
        Gate("高摩擦 cmd=1.5 高速", high_fast["mean_vx"], ">= 1.00 m/s", high_fast["mean_vx"] >= 1.00),
        Gate("高低摩擦高速差", separation, ">= 0.65 m/s", separation >= 0.65),
        Gate("最大横向速度", max_vy, "<= 0.25 m/s", max_vy <= 0.25),
        Gate("最大偏航速度", max_wz, "<= 0.35 rad/s", max_wz <= 0.35),
    ]


def markdown(checkpoint: Path | None, rows: list[dict[str, float]], gates: list[Gate]) -> str:
    passed = all(gate.passed for gate in gates)
    lines = [
        "# TractionTeacher 自动评估",
        "",
        f"- Checkpoint: `{checkpoint or 'report-only'}`",
        f"- Overall: **{'PASS' if passed else 'NEEDS_WORK'}**",
        "",
        "## 摩擦 × 速度矩阵",
        "",
        "| μ | cmd vx | mean vx | |vy| | |wz| | slip | fall/env |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mu']:.2f} | {row['cmd_vx']:.2f} | {row['mean_vx']:.3f} | "
            f"{row['mean_abs_vy']:.3f} | {row['mean_abs_wz']:.3f} | "
            f"{row['mean_contact_slip']:.3f} | {row['fall_per_env']:.3f} |"
        )
    lines += ["", "## 达标门槛", "", "| Gate | Value | Result | Target |", "|---|---:|:---:|---:|"]
    for gate in gates:
        lines.append(
            f"| {gate.name} | {gate.value:.3f} | {'PASS' if gate.passed else 'FAIL'} | {gate.target} |"
        )
    return "\n".join(lines) + "\n"


def render(rows: list[dict[str, float]], gates: list[Gate], color: bool) -> str:
    p = Palette(color)
    lines = [
        f"{p.cyan}{p.bold}TRACTION TEACHER EVALUATION{p.reset}",
        "",
        "    μ   cmd    mean_vx    |vy|    |wz|   fall/env",
        f"{p.gray}{'─' * 56}{p.reset}",
    ]
    for row in rows:
        lines.append(
            f" {row['mu']:4.2f}  {row['cmd_vx']:4.2f}     {row['mean_vx']:6.3f}   "
            f"{row['mean_abs_vy']:5.3f}   {row['mean_abs_wz']:5.3f}    {row['fall_per_env']:6.3f}"
        )
    lines += ["", f"{p.bold}Acceptance gates{p.reset}"]
    for gate in gates:
        state = f"{p.green}PASS{p.reset}" if gate.passed else f"{p.red}FAIL{p.reset}"
        lines.append(f"  [{state}] {gate.name}: {gate.value:.3f}  ({gate.target})")
    overall = all(gate.passed for gate in gates)
    verdict_color = p.green if overall else p.yellow
    lines += ["", f"Overall: {verdict_color}{p.bold}{'PASS' if overall else 'NEEDS_WORK'}{p.reset}"]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    checkpoint: Path | None
    if args.report_only:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else None
        csv_path = args.report_only.expanduser().resolve()
        output_dir = args.output_dir or csv_path.parent
    else:
        checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint
            else latest_checkpoint(args.experiment_root.expanduser().resolve())
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.quick:
            args.mus = [0.08, 1.20]
            args.speeds = [0.5, 1.5]
            args.num_envs = min(args.num_envs, 16)
            args.max_steps = min(args.max_steps, 100)
            args.warmup_steps = min(args.warmup_steps, 30)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or args.evaluation_root / f"{stamp}_{checkpoint.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "matrix.csv"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--task",
            args.task,
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--num_envs",
            str(args.num_envs),
            "--max_steps",
            str(args.max_steps),
            "--warmup_steps",
            str(args.warmup_steps),
            "--command_ramp_steps",
            str(args.command_ramp_steps),
            "--seed",
            str(args.seed),
            "--mu_bins",
            *[str(value) for value in args.mus],
            "--vx",
            *[str(value) for value in args.speeds],
            "--output_csv",
            str(csv_path),
            "--headless",
        ]
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Output:     {output_dir}", flush=True)
        subprocess.run(command, cwd=LAB, check=True)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    gates = assess(rows)
    report = markdown(checkpoint, rows, gates)
    report_path = output_dir / "summary.md"
    json_path = output_dir / "gates.json"
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint) if checkpoint else None,
                "overall": "PASS" if all(gate.passed for gate in gates) else "NEEDS_WORK",
                "gates": [asdict(gate) for gate in gates],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + render(rows, gates, color))
    print(f"\nReport: {report_path}")
    if args.strict and not all(gate.passed for gate in gates):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
