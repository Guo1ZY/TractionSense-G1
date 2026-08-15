#!/usr/bin/env python3
"""Run a reproducible fixed-policy MuJoCo torque-traction smoke matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCENARIOS = {
    "high_friction": ("--friction", "0.8"),
    "low_friction": ("--friction", "0.1"),
    "very_low_friction": ("--friction", "0.05"),
    "abrupt_friction_drop": ("--friction", "0.8", "--transition_friction", "0.1"),
    "asymmetric_friction": ("--left_friction", "0.1", "--right_friction", "0.8"),
    "turning": ("--friction", "0.8", "--command", "0.35", "0.0", "0.6"),
    "lateral": ("--friction", "0.8", "--command", "0.2", "0.3", "0.0"),
    "combined_randomization": ("--friction", "0.35", "--randomization_stage", "5"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/traction_torque/matrix"))
    parser.add_argument("--duration_s", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=tuple(SCENARIOS))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_torque_traction_sim2sim.py")
    manifest = {"policy": str(args.policy.resolve()), "seed": args.seed, "duration_s": args.duration_s, "runs": []}
    for name in args.scenarios:
        output = args.output_dir / f"{name}_seed{args.seed}.npz"
        command = [sys.executable, str(runner), "--policy", str(args.policy), "--duration_s", str(args.duration_s), "--seed", str(args.seed), "--output", str(output), *SCENARIOS[name]]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        manifest["runs"].append({"scenario": name, "output": str(output.resolve()), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            raise RuntimeError(f"scenario {name} failed: {completed.stderr}")
    path = args.output_dir / "manifest.json"; path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(path.resolve()), "completed": len(manifest["runs"])}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

