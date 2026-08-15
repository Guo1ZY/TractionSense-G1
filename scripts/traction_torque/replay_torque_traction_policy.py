#!/usr/bin/env python3
"""Offline-only replay through Student heads and the native-signal governor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unitree_rl_lab.traction_torque.governor import TorqueTractionCommandGovernor
from unitree_rl_lab.traction_torque.schema import TORQUE_TRACTION_FRAME_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--disable_governor", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/offline_replay.npz"))
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    histories = np.asarray(data["student_history"][:, args.environment], dtype=np.float32)
    base_velocity = np.asarray(data["base_velocity_b"][:, args.environment], dtype=np.float32)
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    governor = TorqueTractionCommandGovernor(1, enabled=not args.disable_governor)
    slip_duration = torch.zeros(1, 2)
    records: dict[str, list[np.ndarray]] = {name: [] for name in ("action", "estimated_force", "contact_probability", "slip_probability", "traction_utilization", "traction_margin", "estimator_confidence", "raw_command", "adjusted_command", "speed_scale", "acceleration_limit", "push_off_scale", "governor_state", "safety_flags")}
    for history_np, velocity_np in zip(histories, base_velocity, strict=True):
        history = torch.from_numpy(history_np[None])
        with torch.inference_mode(): first = policy(history)
        action, force, contact, slip, utilization, margin, confidence = first
        slipping = (slip >= 0.6) & (contact >= 0.5)
        slip_duration = torch.where(slipping, slip_duration + 0.02, torch.zeros_like(slip_duration))
        latest = history[:, -1]
        raw_command = latest[:, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")]
        foot_velocity = latest[:, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("foot_planar_velocity")].reshape(1, 2, 2)
        governed = governor.update(raw_command=raw_command, slip_probability=slip, traction_utilization=utilization, traction_margin=margin, contact_probability=contact, estimator_confidence=confidence, foot_relative_velocity=foot_velocity, slip_duration=slip_duration, current_velocity=torch.from_numpy(velocity_np[None]))
        governed_history = history.clone(); governed_history[:, -1, TORQUE_TRACTION_FRAME_SCHEMA.term_slice("command")] = governed.adjusted_command
        with torch.inference_mode(): final = policy(governed_history)
        values = {
            "action": final[0], "estimated_force": final[1], "contact_probability": final[2],
            "slip_probability": final[3], "traction_utilization": final[4], "traction_margin": final[5],
            "estimator_confidence": final[6], "raw_command": raw_command,
            "adjusted_command": governed.adjusted_command, "speed_scale": governed.speed_scale,
            "acceleration_limit": governed.acceleration_limit, "push_off_scale": governed.push_off_scale,
            "governor_state": governed.state, "safety_flags": governed.safety_flags,
        }
        for name, value in values.items(): records[name].append(value.detach().cpu().numpy()[0])
    output = {name: np.asarray(values) for name, values in records.items()}
    nonfinite = sum(int((~np.isfinite(value)).sum()) for value in output.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output, metadata=np.asarray({"offline_only": True, "real_robot_control": False, "contact_truth_policy_input": False}, dtype=object))
    report = {"output": str(args.output.resolve()), "samples": len(histories), "nonfinite_count": nonfinite, "governor_activation_ratio": float((output["governor_state"] != 0).mean()), "mean_speed_scale": float(output["speed_scale"].mean())}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

