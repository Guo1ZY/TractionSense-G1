#!/usr/bin/env python3
"""Short Isaac smoke for canonical Teacher/Student observation and RSL models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    choices=(
        "Unitree-G1-29dof-Velocity-TractionCanonicalTeacher",
        "Unitree-G1-29dof-Velocity-TractionCanonicalStudent",
        "Unitree-G1-29dof-Velocity-TractionCanonicalStudent-Ideal",
        "Unitree-G1-29dof-Velocity-TractionCanonicalStudent-Proprio",
    ),
    default="Unitree-G1-29dof-Velocity-TractionCanonicalStudent",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--build_runner", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion.agents.traction_rsl_cfg import (  # noqa: E402
    CanonicalTractionStudentPPORunnerCfg,
    CanonicalTractionTeacherPPORunnerCfg,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = 20260731
    env_cfg.scene.num_envs = args_cli.num_envs
    # Exercise at least one interval update during this short validation only;
    # the training task retains its slower 1.5--3.0 s transition schedule.
    env_cfg.events.friction_transition.interval_range_s = (0.20, 0.30)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    policy_dim = base_env.observation_manager.group_obs_dim["policy"][-1]
    critic_dim = base_env.observation_manager.group_obs_dim["critic"][-1]
    action_dim = base_env.action_manager.total_action_dim
    is_teacher = args_cli.task.endswith("Teacher")
    assert policy_dim == (234 if is_teacher else 1590), policy_dim
    assert critic_dim == 234, critic_dim
    assert action_dim == 29, action_dim
    friction_mu = base_env.ground_friction_mu_buf
    initial_friction_mu = friction_mu.clone()
    materials = (
        base_env.scene["robot"].root_physx_view.get_material_properties().to(
            base_env.device
        )
    )
    foot_materials = torch.stack(
        (
            materials[:, base_env._canonical_foot_material_slices[0], :2],
            materials[:, base_env._canonical_foot_material_slices[1], :2],
        ),
        dim=1,
    )
    material_mu_error = (
        foot_materials - friction_mu[:, :, None, None]
    ).abs().max().item()
    assert material_mu_error < 1.0e-6, material_mu_error
    assert friction_mu.min() >= 0.05 and friction_mu.max() <= 1.20

    env.reset()
    action = torch.zeros((base_env.num_envs, 29), device=base_env.device)
    nonfinite_observations = 0
    nonfinite_rewards = 0
    for _ in range(args_cli.steps):
        observation, reward, _, _, _ = env.step(action)
        nonfinite_observations += int(
            sum((~torch.isfinite(value)).sum().item() for value in observation.values())
        )
        nonfinite_rewards += int((~torch.isfinite(reward)).sum().item())
    materials = (
        base_env.scene["robot"].root_physx_view.get_material_properties().to(
            base_env.device
        )
    )
    foot_materials = torch.stack(
        (
            materials[:, base_env._canonical_foot_material_slices[0], :2],
            materials[:, base_env._canonical_foot_material_slices[1], :2],
        ),
        dim=1,
    )
    material_mu_error = (
        foot_materials - friction_mu[:, :, None, None]
    ).abs().max().item()
    assert material_mu_error < 1.0e-6, material_mu_error
    assert nonfinite_observations == 0
    assert nonfinite_rewards == 0
    assert torch.any(friction_mu != initial_friction_mu)
    observation = base_env.observation_manager.compute()
    if not is_teacher:
        current = observation["policy"][:, -106:]
        force = current[:, 96:102]
        valid = current[:, 102:104]
        age = current[:, 104:106]
        if args_cli.task.endswith("-Proprio"):
            assert torch.count_nonzero(force) == 0
            assert torch.count_nonzero(valid) == 0
        elif args_cli.task.endswith("-Ideal"):
            assert torch.all(valid == 1.0)
            assert torch.count_nonzero(age) == 0
        assert torch.all((valid == 0.0) | (valid == 1.0))
        assert torch.all(age >= 0.0)

    runner_model = "not_built"
    if args_cli.build_runner:
        wrapped = RslRlVecEnvWrapper(env)
        cfg = (
            CanonicalTractionTeacherPPORunnerCfg()
            if is_teacher
            else CanonicalTractionStudentPPORunnerCfg()
        )
        cfg = handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))
        runner = OnPolicyRunner(
            wrapped,
            cfg.to_dict(),
            log_dir=None,
            device=base_env.device,
        )
        runner_model = type(runner.alg.actor).__name__
        policy = runner.get_inference_policy(device=base_env.device)
        obs = wrapped.get_observations()
        with torch.inference_mode():
            action = policy(obs)
        assert action.shape == (base_env.num_envs, 29)
        assert torch.isfinite(action).all()
        wrapped.close()
    else:
        env.close()

    print(
        "[canonical-traction-smoke]",
        {
            "task": args_cli.task,
            "policy_dim": policy_dim,
            "critic_dim": critic_dim,
            "action_dim": action_dim,
            "nonfinite_observations": nonfinite_observations,
            "nonfinite_rewards": nonfinite_rewards,
            "runner_actor": runner_model,
            "ground_friction_mu_min": friction_mu.min().item(),
            "ground_friction_mu_max": friction_mu.max().item(),
            "asymmetric_environment_fraction": (
                (friction_mu[:, 0] != friction_mu[:, 1]).float().mean().item()
            ),
            "friction_transition_environment_fraction": (
                (friction_mu != initial_friction_mu).any(dim=1).float().mean().item()
            ),
            "material_mu_label_max_error": material_mu_error,
        },
        flush=True,
    )
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
