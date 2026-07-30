#!/usr/bin/env python3
"""Show concise progress for the newest magnetic speed/lateral pipeline run."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DEFAULT_ROOT = REPO_ROOT / "logs/evaluations/traction_magnetic_speed_lateral"


def bar(done: int, total: int, width: int = 36) -> str:
    filled = min(width, int(width * done / max(total, 1)))
    return "\033[38;5;42m" + "█" * filled + "\033[38;5;240m" + "░" * (
        width - filled
    ) + "\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    if args.run is not None:
        run = args.run
    else:
        runs = sorted(
            (path for path in args.root.glob("*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            print("尚无训练流水线记录。")
            return 1
        run = runs[0]

    stages = [
        ("Teacher 微调", bool(list(run.glob("teacher/model_*/isaac_seed*.csv")))),
        ("Teacher 选模", (run / "teacher/selected_checkpoint.txt").is_file()),
        ("第 1 轮 DAgger", (run / "dagger/round1_seed271828.npz").is_file()),
        ("Student 候选训练", (run / "student_round1/selection.json").is_file()),
        ("第 2 轮 DAgger", (run / "dagger/round2_seed424242.npz").is_file()),
        ("Student 精修", bool(list(run.glob("student_round2/*/policy.onnx")))),
        ("Isaac+MuJoCo 评估", (run / "final_selection.json").is_file()),
        ("最佳模型封装", (run / "best_model/policy.onnx").is_file()),
    ]
    done = sum(complete for _, complete in stages)
    print(f"\033[1mRun: {run.name}\033[0m")
    print(f"{bar(done, len(stages))}  {done}/{len(stages)}")
    for name, complete in stages:
        marker = "\033[38;5;42m✓\033[0m" if complete else "\033[38;5;214m○\033[0m"
        print(f"  {marker} {name}")

    progress_files = sorted(
        run.glob("student_*/**/progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if progress_files:
        try:
            item = json.loads(progress_files[0].read_text(encoding="utf-8"))
            print(
                f"\nStudent: {item.get('epoch', 0)}/{item.get('epochs', 0)} "
                f"({item.get('percent', 0):.1f}%)"
            )
        except (OSError, json.JSONDecodeError):
            pass
    log = run / "pipeline.log"
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
        iterations = [int(item) for item in re.findall(r"Learning iteration (\\d+)", tail)]
        if iterations:
            print(f"Teacher 最新轮次: {iterations[-1]}")
        print(f"日志: {log}")
    if (run / "BEST_MODEL.txt").is_file():
        print(
            "最佳模型: "
            + (run / "BEST_MODEL.txt").read_text(encoding="utf-8").strip()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
