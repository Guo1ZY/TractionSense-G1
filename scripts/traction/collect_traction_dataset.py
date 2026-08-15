#!/usr/bin/env python3
"""Collect canonical Isaac traction transitions for distillation or DAgger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Unitree-G1-29dof-Velocity-TractionCanonicalStudent",
)
parser.add_argument("--teacher_checkpoint", type=Path, required=True)
parser.add_argument("--student_checkpoint", type=Path)
parser.add_argument(
    "--rollout_policy",
    choices=("teacher", "student"),
    default="teacher",
    help="student mode is DAgger: execute Student while querying Teacher labels.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=20260731)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion.agents.traction_rsl_cfg import (  # noqa: E402
    CanonicalTractionStudentPPORunnerCfg,
    CanonicalTractionTeacherPPORunnerCfg,
)
from unitree_rl_lab.traction.rsl_models import (  # noqa: E402
    CanonicalStudentRslModel,
    TEACHER_FRAME_DIM,
    TEACHER_FLAT_DIM,
    TractionTeacherRslModel,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _model_kwargs(config) -> dict:
    result = config.to_dict()
    result.pop("class_name", None)
    return result


def _teacher_model(observation: torch.Tensor) -> TractionTeacherRslModel:
    obs = TensorDict({"teacher": observation}, batch_size=[observation.shape[0]])
    model = TractionTeacherRslModel(
        obs,
        {"actor": ["teacher"]},
        "actor",
        29,
        **_model_kwargs(CanonicalTractionTeacherPPORunnerCfg().actor),
    ).to(observation.device)
    checkpoint = torch.load(
        args_cli.teacher_checkpoint,
        map_location=observation.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()
    return model


def _student_model(observation: torch.Tensor) -> CanonicalStudentRslModel:
    if args_cli.student_checkpoint is None:
        raise ValueError("--student_checkpoint is required for --rollout_policy student")
    obs = TensorDict({"student": observation}, batch_size=[observation.shape[0]])
    model = CanonicalStudentRslModel(
        obs,
        {"actor": ["student"]},
        "actor",
        29,
        **_model_kwargs(CanonicalTractionStudentPPORunnerCfg().actor),
    ).to(observation.device)
    checkpoint = torch.load(
        args_cli.student_checkpoint,
        map_location=observation.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()
    return model


def _append(storage: dict[str, list[np.ndarray]], name: str, value: torch.Tensor) -> None:
    storage.setdefault(name, []).append(value.detach().cpu().numpy())


def main() -> int:
    if args_cli.rollout_policy == "student" and args_cli.student_checkpoint is None:
        raise ValueError("DAgger collection requires --student_checkpoint")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    observations, _ = env.reset(seed=args_cli.seed)
    teacher = _teacher_model(observations["critic"])
    student = (
        _student_model(observations["policy"])
        if args_cli.student_checkpoint is not None
        else None
    )
    storage: dict[str, list[np.ndarray]] = {}
    nonfinite = 0

    for step in range(args_cli.steps):
        student_history = observations["policy"]
        teacher_observation = observations["critic"]
        if teacher_observation.shape[1] != TEACHER_FLAT_DIM:
            raise RuntimeError(
                f"Teacher history changed to {teacher_observation.shape[1]}, "
                f"expected {TEACHER_FLAT_DIM}"
            )
        teacher_frame = teacher_observation[:, -TEACHER_FRAME_DIM:]
        teacher_td = TensorDict(
            {"teacher": teacher_observation},
            batch_size=[args_cli.num_envs],
        )
        with torch.inference_mode():
            teacher_action = teacher(teacher_td)
            teacher_latent = teacher.latest_traction_latent
            if student is None:
                rollout_action = teacher_action
            else:
                student_td = TensorDict(
                    {"student": student_history},
                    batch_size=[args_cli.num_envs],
                )
                student_action = student(student_td)
                rollout_action = (
                    student_action
                    if args_cli.rollout_policy == "student"
                    else teacher_action
                )

        current_frame = student_history[:, -106:]
        # Latest Teacher frame = 96 current + 3 command + 135 privilege.
        ground_mu = teacher_frame[:, 195:197]
        ideal_force = teacher_frame[:, 197:203]
        force_normal = teacher_frame[:, 203:205]
        force_tangent = teacher_frame[:, 205:207]
        utilization = teacher_frame[:, 207:209]
        contact = teacher_frame[:, 209:211]
        tangent_velocity = teacher_frame[:, 211:215]
        slip_speed = teacher_frame[:, 215:217]
        slip_label = teacher_frame[:, 217:219]
        base_velocity = teacher_frame[:, 219:222]
        support_ratio = force_normal / force_normal.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        # Privileged auxiliary target is a bounded traction *margin*, not a
        # request for the Student to regress the exact friction coefficient.
        # Airborne feet do not reduce the score; an active slip forces zero.
        friction_demand_ratio = utilization / ground_mu.clamp_min(1.0e-3)
        foot_traction_margin = (1.0 - friction_demand_ratio).clamp(0.0, 1.0)
        foot_traction_margin = torch.where(
            contact > 0.5,
            foot_traction_margin,
            torch.ones_like(foot_traction_margin),
        )
        traction_target = foot_traction_margin.min(dim=1, keepdim=True).values
        traction_target = torch.where(
            slip_label.max(dim=1, keepdim=True).values > 0.5,
            torch.zeros_like(traction_target),
            traction_target,
        )

        timestamp = torch.full(
            (args_cli.num_envs, 1),
            step * float(env.unwrapped.step_dt),
            device=rollout_action.device,
        )
        environment_id = torch.arange(
            args_cli.num_envs, device=rollout_action.device
        )[:, None]
        _append(storage, "timestamp_s", timestamp)
        _append(storage, "environment_id", environment_id)
        _append(storage, "student_history", student_history)
        _append(storage, "teacher_observation", teacher_observation)
        _append(storage, "teacher_action", teacher_action)
        _append(storage, "teacher_latent", teacher_latent)
        _append(storage, "rollout_action", rollout_action)
        _append(storage, "command", current_frame[:, 93:96])
        _append(storage, "base_velocity", base_velocity)
        _append(
            storage,
            "base_yaw_rate",
            teacher_frame[:, 2:3] / 0.2,
        )
        _append(
            storage,
            "projected_gravity",
            teacher_frame[:, 3:6],
        )
        _append(storage, "joint_position", current_frame[:, 6:35])
        _append(storage, "joint_velocity", current_frame[:, 35:64])
        _append(storage, "joint_torque", env.unwrapped.scene["robot"].data.applied_torque)
        _append(storage, "previous_action", current_frame[:, 64:93])
        _append(storage, "ideal_force_xyz_n", ideal_force)
        _append(storage, "observed_force_normalized", current_frame[:, 96:102])
        _append(storage, "force_normal_n", force_normal)
        _append(storage, "force_tangent_n", force_tangent)
        _append(storage, "friction_utilization", utilization)
        _append(storage, "contact", contact)
        _append(storage, "foot_tangent_velocity_proxy", tangent_velocity)
        _append(storage, "slip_speed_proxy", slip_speed)
        _append(storage, "slip_label", slip_label)
        _append(storage, "support_load_ratio", support_ratio)
        _append(storage, "ground_friction_mu", ground_mu)
        _append(storage, "sensor_valid", current_frame[:, 102:104])
        _append(storage, "sensor_age_s", current_frame[:, 104:106])
        _append(storage, "traction_target", traction_target)
        if student is not None:
            _append(
                storage,
                "predicted_slip_probability",
                student.latest_slip_probability,
            )
            _append(
                storage,
                "predicted_traction_score",
                student.latest_traction_score,
            )
            _append(
                storage,
                "predicted_sensor_confidence",
                student.latest_sensor_confidence,
            )
            _append(
                storage,
                "student_latent",
                student.latest_traction_latent,
            )

        observations, reward, terminated, truncated, _ = env.step(rollout_action)
        done = terminated | truncated
        _append(storage, "reward", reward[:, None])
        _append(storage, "terminated", terminated[:, None])
        _append(storage, "truncated", truncated[:, None])
        _append(storage, "episode_done", done[:, None])
        nonfinite += int((~torch.isfinite(rollout_action)).sum().item())
        nonfinite += int((~torch.isfinite(reward)).sum().item())
        if nonfinite:
            raise FloatingPointError(f"nonfinite values detected at step {step}")

    arrays = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in storage.items()
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        **arrays,
        metadata=np.asarray(
            [
                f"seed={args_cli.seed}",
                f"task={args_cli.task}",
                f"rollout_policy={args_cli.rollout_policy}",
                f"teacher_checkpoint={args_cli.teacher_checkpoint.resolve()}",
                "foot_tangent_velocity_and_slip_speed_are_ankle_rigid_body_proxies",
                "traction_target=minimum_contact_traction_margin_with_slip_zero",
            ]
        ),
    )
    print(
        {
            "output": str(args_cli.output.resolve()),
            "samples": int(arrays["student_history"].shape[0]),
            "student_dimension": int(arrays["student_history"].shape[1]),
            "teacher_dimension": int(arrays["teacher_observation"].shape[1]),
            "action_dimension": int(arrays["teacher_action"].shape[1]),
            "nonfinite": nonfinite,
            "slip_positive_fraction": float(arrays["slip_label"].mean()),
        },
        flush=True,
    )
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
