#!/usr/bin/env python3
"""Offline-only replay through force adapter, history, Student, and governor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction import (  # noqa: E402
    IsaacForceAdapter,
    ProprioceptiveState,
    TractionPolicyRuntime,
)
from unitree_rl_lab.traction.deployment import (  # noqa: E402
    DEFAULT_JOINT_POSITION,
    NOMINAL_ROBOT_MASS_KG,
)


def _load_environment(path: Path, environment_id: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "hall_xyz" in archive and "observed_force_normalized" not in archive:
            raise ValueError(
                "raw Hall replay cannot enter the policy: no measured "
                "Hall[15,3] -> calibrated force[3] model was found"
            )
        data = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata"}
    ids = data["environment_id"].reshape(-1).astype(int)
    indices = np.flatnonzero(ids == environment_id)
    if not indices.size:
        raise ValueError(f"environment_id {environment_id} is absent")
    order = np.argsort(data["timestamp_s"][indices].reshape(-1))
    return {key: value[indices[order]] for key, value in data.items()}


def _state(data: dict[str, np.ndarray], index: int) -> ProprioceptiveState:
    current = data["student_history"][index].reshape(15, 106)[-1]
    return ProprioceptiveState(
        timestamp=float(data["timestamp_s"][index].item()),
        base_angular_velocity=current[0:3] / 0.2,
        projected_gravity=current[3:6],
        joint_position=current[6:35] + np.asarray(DEFAULT_JOINT_POSITION),
        joint_velocity=current[35:64] / 0.05,
        previous_action=current[64:93],
        base_linear_velocity=data["base_velocity"][index],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--environment_id", type=int, default=0)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()
    data = _load_environment(args.dataset, args.environment_id)
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    runtime = TractionPolicyRuntime(policy)
    rows = []
    nonfinite = 0
    for index in range(len(data["timestamp_s"])):
        timestamp = float(data["timestamp_s"][index].item())
        force_n = (
            data["observed_force_normalized"][index]
            * (NOMINAL_ROBOT_MASS_KG * 9.81)
        )
        force = IsaacForceAdapter.adapt(
            timestamp,
            force_n,
            valid=data["sensor_valid"][index] > 0.5,
            age_s=data["sensor_age_s"][index],
        )
        command = data["command"][index]
        with torch.inference_mode():
            output = runtime.step(_state(data, index), force, command)
        values = (
            output.action,
            output.slip_probability,
            output.traction_score,
            output.sensor_confidence,
            output.governor.adjusted_command,
        )
        nonfinite += sum(
            int((~torch.isfinite(value)).sum().item()) for value in values
        )
        rows.append(
            {
                "timestamp_s": timestamp,
                "raw_vx": float(command[0]),
                "raw_vy": float(command[1]),
                "raw_yaw": float(command[2]),
                "adjusted_vx": output.governor.adjusted_command[0, 0].item(),
                "adjusted_vy": output.governor.adjusted_command[0, 1].item(),
                "adjusted_yaw": output.governor.adjusted_command[0, 2].item(),
                "acceleration_limit": output.governor.acceleration_limit.item(),
                "deceleration_limit": output.governor.deceleration_limit.item(),
                "yaw_limit": output.governor.yaw_limit.item(),
                "speed_scale": output.governor.speed_scale.item(),
                "push_off_scale": output.governor.push_off_scale.item(),
                "slip_probability_left": output.slip_probability[0, 0].item(),
                "slip_probability_right": output.slip_probability[0, 1].item(),
                "traction_score": output.traction_score.item(),
                "sensor_confidence": output.sensor_confidence.item(),
                "governor_state": output.governor.state.item(),
                "action_max_abs": output.action.abs().max().item(),
                "safety_flags": "|".join(output.safety_flags),
            }
        )
    if nonfinite:
        raise FloatingPointError(f"replay produced {nonfinite} nonfinite values")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "mode": "offline_only_no_robot_control",
        "samples": len(rows),
        "environment_id": args.environment_id,
        "nonfinite": nonfinite,
        "governor_states": sorted({int(row["governor_state"]) for row in rows}),
        "maximum_action_abs": max(float(row["action_max_abs"]) for row in rows),
        "output": str(args.output_csv.resolve()),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
