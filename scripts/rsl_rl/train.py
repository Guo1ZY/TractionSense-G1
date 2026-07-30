# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _wrap_env_nan_guard(env):
    """Sanitize actions in / rewards+obs out so NaNs cannot enter the rollout buffer."""
    import torch

    class _NanGuardEnv:
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            return getattr(self._base, name)

        def get_observations(self):
            obs = self._base.get_observations()
            return _nan_obs(obs)

        def step(self, actions):
            if torch.is_tensor(actions):
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
                actions = torch.clamp(actions, -10.0, 10.0)
            result = self._base.step(actions)
            # rsl-rl wrappers: (obs, rew, dones, extras) or similar
            if isinstance(result, tuple) and len(result) >= 2:
                obs = _nan_obs(result[0])
                rew = result[1]
                if torch.is_tensor(rew):
                    rew = torch.nan_to_num(rew, nan=0.0, posinf=10.0, neginf=-10.0)
                    rew = torch.clamp(rew, -50.0, 50.0)
                return (obs, rew, *result[2:])
            return result

    def _nan_obs(obs):
        if torch.is_tensor(obs):
            return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        # TensorDict / dict-like
        try:
            from tensordict import TensorDict

            if isinstance(obs, TensorDict):
                for k in obs.keys():
                    if torch.is_tensor(obs[k]):
                        obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=10.0, neginf=-10.0)
                return obs
        except Exception:
            pass
        if isinstance(obs, dict):
            for k, v in obs.items():
                if torch.is_tensor(v):
                    obs[k] = torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    print("[INFO] Env NaN guard ON (obs/rew/actions sanitized each step)")
    return _NanGuardEnv(env)


def _install_actor_std_guard(runner, min_std: float = 1e-3, max_std: float = 1.0) -> None:
    """Hard-guard Gaussian std so PPO never samples with std<=0 / NaN.

    Root cause of:
      RuntimeError: normal expects all elements of std >= 0.0
    rsl-rl ``scalar`` std is unconstrained; mid-update optimizer steps can set
    std_param to NaN/negative *between* mini-batches. Clamping only before/after
    ``alg.update()`` is too late — must sanitize:
      1) distribution.update() every sample
      2) grads before optimizer.step
      3) params after optimizer.step
    """
    import math
    import types

    import torch
    from torch.distributions import Normal

    alg = runner.alg
    actor = getattr(alg, "actor", None) or getattr(alg, "policy", None)
    critic = getattr(alg, "critic", None)
    if actor is None:
        print("[WARN] std guard: no actor found")
        return
    dist = getattr(actor, "distribution", None)
    if dist is None:
        print("[WARN] std guard: no distribution on actor")
        return

    warn_budget = {"n": 0}

    def _sanitize_params(module, name: str) -> None:
        if module is None:
            return
        with torch.no_grad():
            for p in module.parameters():
                if p is None or not torch.is_tensor(p):
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    if warn_budget["n"] < 5:
                        print(f"[WARN] {name} has NaN/Inf params — nan_to_num", flush=True)
                        warn_budget["n"] += 1
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def _clamp_std_param() -> None:
        with torch.no_grad():
            if hasattr(dist, "std_param") and dist.std_param is not None:
                p = dist.std_param
                bad = bool(torch.isnan(p).any() or torch.isinf(p).any() or (p <= 0).any())
                if bad and warn_budget["n"] < 8:
                    try:
                        mn = float(torch.nan_to_num(p.detach(), nan=-999.0).min())
                    except Exception:
                        mn = float("nan")
                    print(f"[WARN] actor std invalid (min={mn}); clamping", flush=True)
                    warn_budget["n"] += 1
                p.data = torch.nan_to_num(p.data, nan=0.45, posinf=max_std, neginf=min_std)
                p.data.clamp_(min=min_std, max=max_std)
            if hasattr(dist, "log_std_param") and dist.log_std_param is not None:
                lo, hi = math.log(min_std), math.log(max_std)
                p = dist.log_std_param
                p.data = torch.nan_to_num(p.data, nan=math.log(0.45), posinf=hi, neginf=lo)
                p.data.clamp_(lo, hi)

    def _safe_dist_update(self, mlp_output: torch.Tensor) -> None:
        # Always positive std for Normal(); sanitize mean so log_prob stays finite.
        mean = torch.nan_to_num(mlp_output, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = torch.clamp(mean, -10.0, 10.0)
        if getattr(self, "std_type", "scalar") == "scalar" and hasattr(self, "std_param"):
            # Detach-safe positive std for sampling; keep param clamped for next step.
            with torch.no_grad():
                self.std_param.data = torch.nan_to_num(
                    self.std_param.data, nan=0.45, posinf=max_std, neginf=min_std
                ).clamp_(min_std, max_std)
            std = torch.clamp(self.std_param, min=min_std, max=max_std).expand_as(mean)
        elif hasattr(self, "log_std_param"):
            lo, hi = math.log(min_std), math.log(max_std)
            with torch.no_grad():
                self.log_std_param.data = torch.nan_to_num(
                    self.log_std_param.data, nan=math.log(0.45), posinf=hi, neginf=lo
                ).clamp_(lo, hi)
            std = torch.exp(torch.clamp(self.log_std_param, lo, hi)).expand_as(mean)
        else:
            std = torch.full_like(mean, 0.45)
        # Avoid validate_args cost; ensure scale > 0
        std = torch.clamp(std, min=min_std, max=max_std)
        self._distribution = Normal(mean, std)

    # Patch every sample path inside PPO mini-batches
    dist.update = types.MethodType(_safe_dist_update, dist)  # type: ignore[method-assign]

    # Sanitize grads + params on every optimizer step (fires many times per alg.update)
    opt = getattr(alg, "optimizer", None)
    if opt is not None:
        orig_step = opt.step

        def _safe_step(closure=None):
            modules = [m for m in (actor, critic) if m is not None]
            for mod in modules:
                for p in mod.parameters():
                    if p.grad is not None:
                        p.grad.data = torch.nan_to_num(
                            p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                        )
            out = orig_step(closure=closure)
            for mod, name in ((actor, "actor"), (critic, "critic")):
                _sanitize_params(mod, name)
            _clamp_std_param()
            return out

        opt.step = _safe_step  # type: ignore[method-assign]

    # Also wrap alg.update for storage/advantage NaNs if present
    orig_update = alg.update

    def _safe_update(*args, **kwargs):
        _clamp_std_param()
        # storage advantages / returns can poison the whole update
        storage = getattr(alg, "storage", None)
        if storage is not None:
            for attr in ("advantages", "returns", "values", "rewards"):
                t = getattr(storage, attr, None)
                if torch.is_tensor(t) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    print(f"[WARN] storage.{attr} has NaN/Inf — sanitizing", flush=True)
                    setattr(
                        storage,
                        attr,
                        torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0),
                    )
        try:
            return orig_update(*args, **kwargs)
        finally:
            _clamp_std_param()
            _sanitize_params(actor, "actor")
            _sanitize_params(critic, "critic")

    alg.update = _safe_update  # type: ignore[method-assign]
    _clamp_std_param()
    print(
        f"[INFO] Actor std HARD guard ON "
        f"(dist.update clamp + optimizer.step nan-clean, std∈[{min_std},{max_std}])"
    )

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # rsl-rl >= 4.0 / 5.0: convert legacy policy cfg -> actor/critic
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    # ensure log folder name is set (cfg default is empty string)
    if not agent_cfg.experiment_name:
        agent_cfg.experiment_name = args_cli.task.lower().replace("-", "_").removesuffix("_play")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Kill NaN rewards / obs that poison advantages → std/mean NaN mid-PPO
    env = _wrap_env_nan_guard(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    # strict resume from an absolute checkpoint path (same obs dim; e.g. after NaN crash)
    elif getattr(args_cli, "resume_checkpoint", None):
        resume_abs = os.path.abspath(args_cli.resume_checkpoint)
        load_opt = bool(getattr(args_cli, "load_optimizer", False))
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": load_opt,
            "iteration": True,
            "rnd": False,
        }
        print(f"[INFO]: Strict resume from: {resume_abs} (optimizer={load_opt})")
        runner.load(resume_abs, load_cfg=load_cfg, strict=True)
    # warm-start from a baseline checkpoint when obs dim changed (foot-sensor fine-tune)
    elif getattr(args_cli, "partial_checkpoint", None):
        from unitree_rl_lab.utils.partial_checkpoint import load_partial_into_runner

        partial_path = os.path.abspath(args_cli.partial_checkpoint)
        print(f"[INFO]: Partial warm-start from: {partial_path}")
        load_partial_into_runner(runner, partial_path, device=agent_cfg.device, verbose=True)

    # rsl-rl scalar std is unconstrained; grads can push it <0 or NaN → Normal.sample crash.
    # Clamp after load and after every PPO update.
    _install_actor_std_guard(runner, min_std=1e-3, max_std=1.0)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
