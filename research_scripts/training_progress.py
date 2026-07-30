#!/usr/bin/env python3
"""Pretty progress display for the newest Unitree RL training run.

The exact iteration is read from TensorBoard events. If TensorBoard is not
available or the event file cannot be read, the newest model_N.pt checkpoint
is used as a safe fallback.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/rsl_rl"
ISAACLAB_PYTHON = Path(os.environ.get("ISAACLAB_PYTHON", sys.executable))


def ensure_tensorboard_python() -> None:
    """Re-exec in the Isaac Lab environment when launched from system Python."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        if ISAACLAB_PYTHON.exists() and Path(sys.executable) != ISAACLAB_PYTHON:
            os.execv(str(ISAACLAB_PYTHON), [str(ISAACLAB_PYTHON), *sys.argv])


ensure_tensorboard_python()


@dataclass
class Progress:
    run_dir: Path
    task: str
    current: int
    target: int
    start: int
    checkpoint: Path | None
    pid: int | None
    elapsed_seconds: float | None
    seconds_per_iteration: float | None
    source: str
    updated_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a pretty progress bar for the newest RL training run."
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Specific run directory; default: newest run under --log-root",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"RSL-RL log root (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh until Ctrl+C",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds for --watch (default: 5)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors",
    )
    parser.add_argument(
        "--target",
        type=int,
        help="Override the target iteration (useful for a resumed fine-tune)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting iteration used for stage-local percentage and ETA",
    )
    return parser.parse_args()


def newest_run(log_root: Path) -> Path:
    event_files = list(log_root.glob("*/*/events.out.tfevents.*"))
    if not event_files:
        event_files = list(log_root.rglob("events.out.tfevents.*"))
    if event_files:
        return max(event_files, key=lambda path: path.stat().st_mtime).parent

    checkpoints = list(log_root.rglob("model_*.pt"))
    if checkpoints:
        return max(checkpoints, key=lambda path: path.stat().st_mtime).parent
    raise FileNotFoundError(f"no TensorBoard event or model checkpoint under {log_root}")


def read_target(run_dir: Path) -> int:
    agent_cfg = run_dir / "params" / "agent.yaml"
    if agent_cfg.is_file():
        match = re.search(
            r"(?m)^\s*max_iterations\s*:\s*(\d+)\s*$",
            agent_cfg.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            return int(match.group(1))
    return 0


def checkpoint_iteration(run_dir: Path) -> tuple[int, Path | None]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])


def event_iteration(run_dir: Path) -> tuple[int | None, float | None, float]:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None, run_dir.stat().st_mtime
    event_file = max(event_files, key=lambda path: path.stat().st_mtime)

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 500}
        )
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        preferred = [
            "Train/mean_reward",
            "Loss/value",
            "Perf/total_fps",
        ]
        tag = next((name for name in preferred if name in tags), None)
        if tag is None and tags:
            tag = tags[0]
        if tag is None:
            return None, None, event_file.stat().st_mtime

        values = accumulator.Scalars(tag)
        if not values:
            return None, None, event_file.stat().st_mtime
        last = max(values, key=lambda value: value.step)
        recent = [value for value in values if value.step >= last.step - 200]
        first = min(recent, key=lambda value: value.step)
        seconds_per_iteration = None
        if last.step > first.step and last.wall_time > first.wall_time:
            seconds_per_iteration = (last.wall_time - first.wall_time) / (
                last.step - first.step
            )
        return int(last.step), seconds_per_iteration, float(last.wall_time)
    except Exception:
        return None, None, event_file.stat().st_mtime


def event_pid(run_dir: Path) -> int | None:
    files = sorted(
        run_dir.glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for event_file in files:
        match = re.search(r"\.(\d+)\.\d+$", event_file.name)
        if not match:
            continue
        pid = int(match.group(1))
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except OSError:
            continue
        if "train.py" in cmdline:
            return pid
    return None


def process_cmdline(pid: int | None) -> list[str]:
    if pid is None:
        return []
    try:
        return [
            value.decode(errors="replace")
            for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        return []


def option_value(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def process_elapsed(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        start_ticks = int(fields[21])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def collect(
    run_dir: Path,
    target_override: int | None = None,
    start: int = 0,
) -> Progress:
    checkpoint_step, checkpoint = checkpoint_iteration(run_dir)
    event_step, seconds_per_iteration, updated_at = event_iteration(run_dir)
    if event_step is None:
        current = checkpoint_step
        source = "checkpoint"
        if checkpoint is not None:
            updated_at = checkpoint.stat().st_mtime
    else:
        current = event_step
        source = "TensorBoard"

    pid = event_pid(run_dir)
    command = process_cmdline(pid)
    target_text = option_value(command, "--max_iterations")
    target = int(target_text) if target_text and target_text.isdigit() else read_target(run_dir)
    if target_override is not None:
        target = target_override
    task = option_value(command, "--task") or run_dir.parent.name
    return Progress(
        run_dir=run_dir,
        task=task,
        current=current,
        target=target,
        start=start,
        checkpoint=checkpoint,
        pid=pid,
        elapsed_seconds=process_elapsed(pid),
        seconds_per_iteration=seconds_per_iteration,
        source=source,
        updated_at=updated_at,
    )


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    total = int(seconds)
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class Palette:
    def __init__(self, enabled: bool):
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.green = "\033[38;5;82m" if enabled else ""
        self.yellow = "\033[38;5;220m" if enabled else ""
        self.cyan = "\033[38;5;45m" if enabled else ""
        self.gray = "\033[38;5;240m" if enabled else ""


def render(progress: Progress, color: bool) -> str:
    palette = Palette(color)
    terminal_width = shutil.get_terminal_size((100, 24)).columns
    bar_width = max(20, min(60, terminal_width - 28))
    stage_target = max(0, progress.target - progress.start)
    stage_current = max(0, progress.current - progress.start)
    ratio = stage_current / stage_target if stage_target else 0.0
    clamped = min(1.0, max(0.0, ratio))
    filled = int(clamped * bar_width)
    bar = (
        f"{palette.green}{'█' * filled}"
        f"{palette.gray}{'░' * (bar_width - filled)}{palette.reset}"
    )
    percent = ratio * 100.0

    if progress.pid is not None:
        state = f"{palette.green}● RUNNING{palette.reset}  PID {progress.pid}"
    else:
        state = f"{palette.yellow}● STOPPED{palette.reset}"

    remaining = max(0, stage_target - stage_current)
    eta_seconds = (
        remaining * progress.seconds_per_iteration
        if progress.seconds_per_iteration is not None
        else None
    )
    speed = (
        f"{progress.seconds_per_iteration:.2f} s/轮"
        if progress.seconds_per_iteration is not None
        else "--"
    )
    checkpoint = progress.checkpoint.name if progress.checkpoint else "--"
    age = max(0.0, time.time() - progress.updated_at)

    lines = [
        f"{palette.cyan}┌─{palette.bold} RL TRAINING PROGRESS {palette.reset}{palette.cyan}{'─' * 28}┐{palette.reset}",
        f"{palette.cyan}│{palette.reset} 状态       {state}",
        f"{palette.cyan}│{palette.reset} 任务       {progress.task}",
        f"{palette.cyan}│{palette.reset}",
        f"{palette.cyan}│{palette.reset} {bar}  {palette.bold}{percent:6.2f}%{palette.reset}",
        f"{palette.cyan}│{palette.reset} 轮次       {palette.bold}{progress.current:,}{palette.reset} / {progress.target:,}",
        f"{palette.cyan}│{palette.reset} 本阶段     +{stage_current:,} / +{stage_target:,}",
        f"{palette.cyan}│{palette.reset} Checkpoint {checkpoint}",
        f"{palette.cyan}│{palette.reset} 速度       {speed}    已运行 {duration(progress.elapsed_seconds)}",
        f"{palette.cyan}│{palette.reset} 预计剩余   {duration(eta_seconds)}",
        f"{palette.cyan}│{palette.reset} 数据来源   {progress.source}（{duration(age)} 前更新）",
        f"{palette.cyan}│{palette.reset} 日志目录   {progress.run_dir}",
        f"{palette.cyan}└{'─' * 52}┘{palette.reset}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        print("--interval must be greater than zero", file=sys.stderr)
        return 2
    try:
        run_dir = args.run.expanduser().resolve() if args.run else newest_run(args.log_root)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    if not run_dir.is_dir():
        print(f"[ERROR] run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    try:
        while True:
            progress = collect(run_dir, args.target, args.start)
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(progress, color), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监控。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
