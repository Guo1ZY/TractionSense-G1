#!/usr/bin/env python3
"""Extract scalar training curves from one RSL-RL tensorboard event file.

Writes ``curves.json`` plus a compact CSV of the headline metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


HEADLINE = (
    "Train/mean_reward",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Policy/mean_std",
    "Train/mean_episode_length",
    "Episode_Termination/bad_orientation",
    "Episode_Termination/base_height",
    "Episode_Termination/time_out",
    "Episode_Termination/course_success",
    "Episode_Reward/track_lin_vel_xy",
    "Episode_Reward/contact_point_slip",
    "Episode_Reward/straight_heading_error",
    "Episode_Reward/low_stage_yaw_rate",
    "Episode_Reward/low_entry_heading_change",
    "Episode_Reward/transition_heading_retention",
    "Loss/learning_rate",
    "Perf/total_fps",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("."))
    args = parser.parse_args()

    accumulator = EventAccumulator(
        str(args.event_file), size_guidance={}
    )
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    series = {}
    for tag in sorted(tags):
        events = accumulator.Scalars(tag)
        series[tag] = [
            {"step": int(item.step), "value": float(item.value)}
            for item in events
        ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "curves.json").write_text(
        json.dumps(series, indent=2) + "\n"
    )

    iterations = sorted({item["step"] for item in series.get("Train/mean_reward", [])})
    with (args.out_dir / "headline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + list(HEADLINE))
        for iteration in iterations:
            row = [iteration]
            for tag in HEADLINE:
                value = next(
                    (item["value"] for item in series.get(tag, []) if item["step"] == iteration),
                    None,
                )
                row.append(value)
            writer.writerow(row)
    print(f"wrote {len(series)} tags / {len(iterations)} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
