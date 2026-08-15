# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


CHECKPOINT_LOAD_MODE_FRESH = "fresh"
CHECKPOINT_LOAD_MODE_REGISTRY_RESUME = "registry_resume"
CHECKPOINT_LOAD_MODE_STRICT_RESUME = "strict_resume"
CHECKPOINT_LOAD_MODE_PARTIAL = "partial_checkpoint"


def validate_checkpoint_load_args(args_cli: argparse.Namespace) -> str:
    """Return the unambiguous checkpoint load mode or fail closed.

    RSL-RL's registry resume, the project-local strict same-schema resume and
    the dimension-expanding partial warm-start have deliberately different
    state semantics.  Accepting more than one silently made the first branch
    in ``train.py`` win, so a command could look like a strict continuation
    while actually loading another run (or vice versa).
    """

    selected = []
    if bool(getattr(args_cli, "resume", False)):
        selected.append(CHECKPOINT_LOAD_MODE_REGISTRY_RESUME)
    if getattr(args_cli, "resume_checkpoint", None):
        selected.append(CHECKPOINT_LOAD_MODE_STRICT_RESUME)
    if getattr(args_cli, "partial_checkpoint", None):
        selected.append(CHECKPOINT_LOAD_MODE_PARTIAL)
    if len(selected) > 1:
        raise ValueError(
            "--resume, --resume_checkpoint and --partial_checkpoint are "
            "mutually exclusive"
        )

    mode = selected[0] if selected else CHECKPOINT_LOAD_MODE_FRESH
    if bool(getattr(args_cli, "partial_checkpoint_critic_only", False)) and mode != CHECKPOINT_LOAD_MODE_PARTIAL:
        raise ValueError(
            "--partial_checkpoint_critic_only requires --partial_checkpoint PATH"
        )
    if bool(getattr(args_cli, "load_optimizer", False)) and mode != CHECKPOINT_LOAD_MODE_STRICT_RESUME:
        raise ValueError(
            "--load_optimizer is valid only with --resume_checkpoint PATH"
        )
    return mode


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--partial_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a baseline .pt to warm-start from when observation dims grew "
            "(e.g. model_49999.pt → foot-sensor task). Copies matching weights and "
            "expands the first actor/critic Linear input columns; does not load optimizer."
        ),
    )
    arg_group.add_argument(
        "--partial_checkpoint_critic_only",
        action="store_true",
        default=False,
        help=(
            "With --partial_checkpoint, load only critic tensors. This is the "
            "safe FastBase gate-warmup mode: the actor mean, gate, residual and "
            "exploration distribution all remain freshly initialized."
        ),
    )
    arg_group.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Absolute/relative path to a same-obs-dim checkpoint for strict resume "
            "(e.g. foot_ft/model_700.pt after a NaN crash). Loads actor+critic; "
            "optimizer load controlled by --load_optimizer."
        ),
    )
    arg_group.add_argument(
        "--load_optimizer",
        action="store_true",
        default=False,
        help="When using --resume_checkpoint, also load optimizer state (default: off, safer after instability).",
    )
    arg_group.add_argument(
        "--learning_rate", type=float, default=None,
        help="Optional PPO learning-rate override for safe continuation runs.",
    )
    arg_group.add_argument(
        "--anchor_loss_coef",
        type=float,
        default=None,
        help="Override the HIGH-only frozen-Teacher anchor coefficient (lambda).",
    )
    arg_group.add_argument(
        "--anchor_delta_cap",
        type=float,
        default=None,
        help="Override the per-joint cached Teacher target delta cap.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if getattr(args_cli, "learning_rate", None) is not None:
        if args_cli.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        agent_cfg.algorithm.learning_rate = float(args_cli.learning_rate)
    if getattr(args_cli, "anchor_loss_coef", None) is not None:
        if not hasattr(agent_cfg.algorithm, "anchor_loss_coef"):
            raise ValueError("--anchor_loss_coef requires an AnchoredPPO task")
        if args_cli.anchor_loss_coef < 0.0:
            raise ValueError("anchor_loss_coef must be non-negative")
        agent_cfg.algorithm.anchor_loss_coef = float(args_cli.anchor_loss_coef)
    if getattr(args_cli, "anchor_delta_cap", None) is not None:
        if not hasattr(agent_cfg.algorithm, "anchor_delta_cap"):
            raise ValueError("--anchor_delta_cap requires an AnchoredPPO task")
        if args_cli.anchor_delta_cap <= 0.0:
            raise ValueError("anchor_delta_cap must be positive")
        agent_cfg.algorithm.anchor_delta_cap = float(args_cli.anchor_delta_cap)
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg
