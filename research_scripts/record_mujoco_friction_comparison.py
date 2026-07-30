#!/usr/bin/env python3
"""Record and compose a low/mid/high-friction MuJoCo policy comparison."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT
MATRIX = ROOT / "research_scripts/mujoco_friction_speed_matrix.py"
SLOT = LAB / "deploy/robots/g1_29dof/config/policy/velocity/traction_student"
DEFAULT_CHECKPOINT = (
    LAB
    / "logs/evaluations/traction_student/20260721_model_7750_privileged_aux_dagger2"
    / "student_actor.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo friction video comparison")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--slot",
        default="traction_student",
        help="Preinstalled g1_ctrl velocity policy slot (for example v0)",
    )
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--mus", nargs=3, type=float, default=[0.08, 0.40, 1.20])
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--measure-sec", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--latest-path",
        type=Path,
        default=ROOT / "video/mujoco_friction_comparison_latest.mp4",
        help="Convenience copy path for the composed video",
    )
    parser.add_argument(
        "--allow-unstable",
        action="store_true",
        help="Finish the comparison when a policy falls or fails stability checks",
    )
    parser.add_argument(
        "--disable-command-slew",
        action="store_true",
        help="Give every policy the same immediate velocity step",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ROOT / "video" / f"mujoco_friction_comparison_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ("LOW GRIP", "MID GRIP", "HIGH GRIP")
    clips: list[Path] = []
    results: list[dict[str, str]] = []

    for label, mu in zip(labels, args.mus):
        key = label.lower().replace(" ", "_")
        case_dir = output_dir / key
        frames = case_dir / "frames"
        evaluation = case_dir / "evaluation"
        matrix_command = [
            "python3", str(MATRIX),
            "--profile", "adaptive",
            "--slot", args.slot,
            "--checkpoint", str(checkpoint),
            "--skip-export", "--skip-build",
            "--mus", f"{mu:.3f}",
            "--speeds", f"{args.speed:.3f}",
            "--stand-sec", "8",
            "--warmup-sec", f"{args.warmup_sec:.3f}",
            "--measure-sec", f"{args.measure_sec:.3f}",
            "--record-frames-dir", str(frames),
            "--record-fps", f"{args.fps:.3f}",
            "--output-dir", str(evaluation),
        ]
        if args.disable_command_slew:
            matrix_command.append("--disable-command-slew")
        run(matrix_command)
        with (evaluation / "matrix.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        if (
            not args.allow_unstable
            and (int(row["fall"]) != 0 or row["stable"] != "PASS")
        ):
            raise RuntimeError(f"Recorded case is unstable: mu={mu} row={row}")
        results.append(row)
        measured_vx = float(row["mean_vx"])
        clip = case_dir / "clip.mp4"
        outcome = (
            "FALL"
            if int(row["fall"]) != 0
            else ("ABNORMAL" if row["stable"] != "PASS" else "STABLE")
        )
        text = (
            f"{label} | mu {mu:.2f} | cmd {args.speed:.1f} | "
            f"vx {measured_vx:.3f} m/s | {outcome}"
        )
        filter_text = (
            "drawbox=x=0:y=0:w=iw:h=52:color=black@0.70:t=fill,"
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"text='{text}':fontcolor=white:fontsize=17:x=(w-text_w)/2:y=16"
        )
        run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", f"{args.fps:.3f}",
                "-i", str(frames / "frame_%06d.png"),
                "-vf", filter_text,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", str(clip),
            ]
        )
        clips.append(clip)

    comparison = output_dir / "mujoco_friction_comparison.mp4"
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clips[0]), "-i", str(clips[1]), "-i", str(clips[2]),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];"
            "[2:v]setpts=PTS-STARTPTS[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]",
            "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ]
    )
    latest = args.latest_path.expanduser().resolve()
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(comparison, latest)
    report = output_dir / "README.md"
    lines = [
        "# MuJoCo friction comparison",
        "",
        f"- Policy: `{checkpoint}`",
        f"- Deploy slot: `{args.slot}`",
        f"- Forward command: {args.speed:.2f} m/s",
        f"- Video: `{comparison}`",
        "",
        "| Surface | μ | measured vx | |vy| | fall |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label, row in zip(labels, results):
        lines.append(
            f"| {label} | {float(row['mu']):.2f} | {float(row['mean_vx']):.3f} | "
            f"{float(row['mean_abs_vy']):.3f} | {int(row['fall'])} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] comparison={comparison}")
    print(f"[DONE] latest={latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
