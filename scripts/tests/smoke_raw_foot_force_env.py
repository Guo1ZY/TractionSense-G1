#!/usr/bin/env python3
"""Short Isaac Lab smoke test for the G1-29DoF raw foot-force observation."""

from __future__ import annotations

import argparse
import pathlib
import sys
from importlib.metadata import version

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    choices=(
        "Unitree-G1-29dof-Velocity-RawFoot",
        "Unitree-G1-29dof-Velocity-RawFoot-Off",
    ),
    default="Unitree-G1-29dof-Velocity-RawFoot",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=150)
parser.add_argument("--walk_steps", type=int, default=300)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(pathlib.Path(__file__).resolve().parents[2] / "model/rl/model_49999.pt"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.tasks.locomotion import mdp  # noqa: E402
from unitree_rl_lab.tasks.locomotion.mdp.foot_sensor import FOOT_BODY_NAMES  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.utils.partial_checkpoint import load_partial_into_runner  # noqa: E402


def _resolved_foot_cfgs(env):
    left_sensor_cfg = SceneEntityCfg("left_raw_foot_contact")
    right_sensor_cfg = SceneEntityCfg("right_raw_foot_contact")
    asset_cfg = SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True)
    left_sensor_cfg.resolve(env.scene)
    right_sensor_cfg.resolve(env.scene)
    asset_cfg.resolve(env.scene)
    return left_sensor_cfg, right_sensor_cfg, asset_cfg


def _force_sample(env, left_sensor_cfg, right_sensor_cfg, asset_cfg):
    robot = env.scene[asset_cfg.name]
    force_w = mdp.raw_foot_force_world_n(env, left_sensor_cfg, right_sensor_cfg)
    force_l = mdp.raw_foot_force_local_n(
        env, left_sensor_cfg, right_sensor_cfg, asset_cfg
    ).reshape(env.num_envs, 2, 3)
    force_w_roundtrip = quat_apply(
        robot.data.body_quat_w[:, asset_cfg.body_ids, :], force_l
    )
    roundtrip_error = (force_w_roundtrip - force_w).abs().max()
    return force_w, force_l, roundtrip_error


def _first_linear(module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            return layer
    raise RuntimeError(f"No Linear layer found in {type(module).__name__}")


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    policy_dim = base_env.observation_manager.group_obs_dim["policy"][-1]
    critic_dim = base_env.observation_manager.group_obs_dim["critic"][-1]
    action_dim = base_env.action_manager.total_action_dim
    policy_terms = base_env.observation_manager.active_terms["policy"]
    critic_terms = base_env.observation_manager.active_terms["critic"]
    if args_cli.task.endswith("-Off"):
        assert policy_dim == 480, policy_dim
        assert critic_dim == 495, critic_dim
        assert action_dim == 29, action_dim
        assert "raw_foot_force" not in policy_terms
        assert "raw_foot_force" not in critic_terms
        print(
            "[raw-foot-smoke] toggle OFF dimensions:",
            {"policy": policy_dim, "critic": critic_dim, "action": action_dim},
            flush=True,
        )
        env.close()
        simulation_app.close()
        return 0

    left_sensor_cfg, right_sensor_cfg, asset_cfg = _resolved_foot_cfgs(base_env)
    assert policy_dim == 510, policy_dim
    assert critic_dim == 525, critic_dim
    assert action_dim == 29, action_dim
    assert policy_terms[-1] == "raw_foot_force", policy_terms
    assert critic_terms[-1] == "raw_foot_force", critic_terms

    wrapped_env = RslRlVecEnvWrapper(env)
    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

    runner_cfg = handle_deprecated_rsl_rl_cfg(
        BasePPORunnerCfg(), version("rsl-rl-lib")
    )
    runner = OnPolicyRunner(
        wrapped_env, runner_cfg.to_dict(), log_dir=None, device=base_env.device
    )

    actor_first = _first_linear(runner.alg.actor)
    critic_first = _first_linear(runner.alg.critic)
    stats = load_partial_into_runner(
        runner, str(pathlib.Path(args_cli.checkpoint).resolve()), device=base_env.device
    )
    checkpoint = torch.load(
        pathlib.Path(args_cli.checkpoint).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    assert tuple(actor_first.weight.shape) == (512, 510)
    assert tuple(critic_first.weight.shape) == (512, 525)
    assert torch.count_nonzero(actor_first.weight[:, 480:]).item() == 0
    assert torch.count_nonzero(critic_first.weight[:, 495:]).item() == 0
    for key, value in checkpoint["actor_state_dict"].items():
        loaded = runner.alg.actor.state_dict()[key].detach().cpu()
        if key == "mlp.0.weight":
            assert torch.equal(loaded[:, :480], value)
        else:
            assert torch.equal(loaded, value), key
    for key, value in checkpoint["critic_state_dict"].items():
        loaded = runner.alg.critic.state_dict()[key].detach().cpu()
        if key == "mlp.0.weight":
            assert torch.equal(loaded[:, :495], value)
        else:
            assert torch.equal(loaded, value), key

    # Phase 1: zero action holds the original default pose and measures support load.
    standing_ratios = []
    standing_local_z = []
    zero_actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
    for step in range(args_cli.settle_steps):
        wrapped_env.step(zero_actions)
        if step >= args_cli.settle_steps // 2:
            force_w, force_l, _ = _force_sample(
                base_env, left_sensor_cfg, right_sensor_cfg, asset_cfg
            )
            mass = (
                base_env.scene["robot"]
                .root_physx_view.get_masses()
                .sum(dim=1)
                .to(force_w.device)
            )
            standing_ratios.append(force_w[:, :, 2].sum(dim=1) / (mass * 9.81))
            standing_local_z.append(force_l[:, :, 2])
    standing_ratio = torch.cat(standing_ratios)
    local_z = torch.cat(standing_local_z)

    # Phase 2: the zero-column partial load is initially behavior-identical to
    # the old actor; a short inference rollout should expose true swing samples.
    policy = runner.get_inference_policy(device=base_env.device)
    obs = wrapped_env.get_observations()
    swing_force_norms = []
    roundtrip_errors = []
    local_force_samples = []
    for _ in range(args_cli.walk_steps):
        with torch.inference_mode():
            obs, _, _, _ = wrapped_env.step(policy(obs))
        force_w, force_l, roundtrip_error = _force_sample(
            base_env, left_sensor_cfg, right_sensor_cfg, asset_cfg
        )
        roundtrip_errors.append(roundtrip_error)
        local_force_samples.append(force_l)
        norms = torch.linalg.norm(force_l, dim=-1)
        left_sensor = base_env.scene.sensors[left_sensor_cfg.name]
        right_sensor = base_env.scene.sensors[right_sensor_cfg.name]
        air_time = torch.stack(
            (
                left_sensor.data.current_air_time[:, 0],
                right_sensor.data.current_air_time[:, 0],
            ),
            dim=1,
        )
        contact_time = torch.stack(
            (
                left_sensor.data.current_contact_time[:, 0],
                right_sensor.data.current_contact_time[:, 0],
            ),
            dim=1,
        )
        for side in range(2):
            other = 1 - side
            mask = (air_time[:, side] > 0.02) & (contact_time[:, other] > 0.02)
            if mask.any():
                swing_force_norms.append(norms[mask, side])

    obs = wrapped_env.get_observations()
    assert obs["policy"].shape == (base_env.num_envs, 510)
    assert obs["critic"].shape == (base_env.num_envs, 525)
    normalized_current = mdp.normalized_raw_foot_force_local(
        base_env,
        left_sensor_cfg,
        right_sensor_cfg,
        asset_cfg,
        robot_mass_kg=env_cfg.raw_foot_force_robot_mass_kg,
        gravity_m_s2=9.81,
    ).clip(*env_cfg.raw_foot_force_clip)
    policy_force_error = (obs["policy"][:, -6:] - normalized_current).abs().max().item()
    critic_force_error = (obs["critic"][:, -6:] - normalized_current).abs().max().item()
    assert policy_force_error < 1.0e-6, policy_force_error
    assert critic_force_error < 1.0e-6, critic_force_error
    max_roundtrip_error = torch.stack(roundtrip_errors).max().item()
    walk_force_local = torch.cat(local_force_samples, dim=0).reshape(-1, 6)
    swing_max = (
        torch.cat(swing_force_norms).max().item() if swing_force_norms else float("nan")
    )

    print("[raw-foot-smoke] task:", args_cli.task)
    print("[raw-foot-smoke] body order:", list(FOOT_BODY_NAMES))
    print(
        "[raw-foot-smoke] per-frame force order:",
        ["left_Fx", "left_Fy", "left_Fz", "right_Fx", "right_Fy", "right_Fz"],
    )
    print("[raw-foot-smoke] policy terms:", policy_terms)
    print("[raw-foot-smoke] critic terms:", critic_terms)
    print(
        "[raw-foot-smoke] dimensions:",
        {"policy": policy_dim, "critic": critic_dim, "action": action_dim},
    )
    print(
        "[raw-foot-smoke] checkpoint expansion:",
        {
            "actor": tuple(actor_first.weight.shape),
            "critic": tuple(critic_first.weight.shape),
            "actor_new_nonzero": torch.count_nonzero(actor_first.weight[:, 480:]).item(),
            "critic_new_nonzero": torch.count_nonzero(critic_first.weight[:, 495:]).item(),
            "actor_copied": len(stats["actor"]["copied"]),
            "critic_copied": len(stats["critic"]["copied"]),
        },
    )
    print(
        "[raw-foot-smoke] standing world-Fz/weight:",
        {
            "median": standing_ratio.median().item(),
            "mean": standing_ratio.mean().item(),
            "min": standing_ratio.min().item(),
            "max": standing_ratio.max().item(),
        },
    )
    print(
        "[raw-foot-smoke] contacting local-Fz signs:",
        {
            "left_positive_fraction": (local_z[:, 0] > 5.0).float().mean().item(),
            "right_positive_fraction": (local_z[:, 1] > 5.0).float().mean().item(),
            "world_local_roundtrip_max_N": max_roundtrip_error,
        },
    )
    print(
        "[raw-foot-smoke] signed local force range N:",
        {
            "min": walk_force_local.min(dim=0).values.tolist(),
            "max": walk_force_local.max(dim=0).values.tolist(),
            "policy_normalized_tail_max_error": policy_force_error,
            "critic_normalized_tail_max_error": critic_force_error,
        },
    )
    print(
        "[raw-foot-smoke] swing/no-contact:",
        {"samples": sum(x.numel() for x in swing_force_norms), "max_force_N": swing_max},
    )

    sys.stdout.flush()
    wrapped_env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
