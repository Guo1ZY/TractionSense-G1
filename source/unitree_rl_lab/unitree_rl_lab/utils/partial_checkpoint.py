# Copyright (c) 2025 local foot-sensor extension.
# SPDX-License-Identifier: BSD-3-Clause
"""Partial checkpoint loading for fine-tuning when observation dims grow.

Supports:
  * rsl-rl ≥ 5: ``actor_state_dict`` / ``critic_state_dict`` with ``mlp.0.weight``
  * legacy: ``model_state_dict`` (single ActorCritic module)

Typical use: warm-start ``Unitree-G1-29dof-Velocity-Foot`` from ``model_49999.pt``.

Strategy:
  * Matching tensors are copied as-is.
  * Linear ``weight`` ``(out, in_new)`` vs ``(out, in_old)`` with ``in_new > in_old``:
    copy old columns, leave new columns as current init.
  * Optimizer state is NOT loaded (obs layout changed → Adam moments invalid).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _expand_linear_weight(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor | None:
    """Return expanded weight if only the input dimension grew, else None."""
    if dst.ndim != 2 or src.ndim != 2:
        return None
    if dst.shape[0] != src.shape[0]:
        return None
    if dst.shape[1] < src.shape[1]:
        return None
    if dst.shape[1] == src.shape[1]:
        return src.clone()
    out = dst.clone()
    out[:, : src.shape[1]] = src
    return out


def merge_state_dicts(
    current: dict[str, torch.Tensor],
    loaded: dict[str, torch.Tensor],
    *,
    verbose: bool = True,
    label: str = "",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Merge ``loaded`` into ``current`` with input-dim expansion for Linear layers."""
    merged = {k: v.clone() if torch.is_tensor(v) else v for k, v in current.items()}
    stats = {
        "copied": [],
        "expanded": [],
        "skipped_missing": [],
        "skipped_shape": [],
    }

    for key, src in loaded.items():
        if key not in merged:
            stats["skipped_missing"].append(key)
            continue
        dst = merged[key]
        if not torch.is_tensor(src) or not torch.is_tensor(dst):
            continue
        if src.shape == dst.shape:
            merged[key] = src.to(dtype=dst.dtype, device=dst.device)
            stats["copied"].append(key)
            continue
        expanded = _expand_linear_weight(dst, src.to(dtype=dst.dtype, device=dst.device))
        if expanded is not None:
            merged[key] = expanded
            stats["expanded"].append(f"{key}: {tuple(src.shape)} -> {tuple(expanded.shape)}")
            continue
        stats["skipped_shape"].append(f"{key}: ckpt{tuple(src.shape)} vs model{tuple(dst.shape)}")

    if verbose:
        tag = f"[partial_checkpoint{('/' + label) if label else ''}]"
        print(f"{tag} copied: {len(stats['copied'])}")
        print(f"{tag} expanded: {len(stats['expanded'])}")
        for line in stats["expanded"]:
            print("   ", line)
        if stats["skipped_shape"]:
            print(f"{tag} skipped (shape): {len(stats['skipped_shape'])}")
            for line in stats["skipped_shape"][:12]:
                print("   ", line)
        if stats["skipped_missing"]:
            print(f"{tag} skipped (missing in model): {len(stats['skipped_missing'])}")

    return merged, stats


def load_partial_into_module(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    label: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """Merge a state_dict into a module with possible first-layer expansion."""
    current = module.state_dict()
    merged, stats = merge_state_dicts(current, state_dict, verbose=verbose, label=label)
    module.load_state_dict(merged, strict=True)
    return stats


def load_partial_into_runner(
    runner: Any,
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Load a baseline checkpoint into a runner whose obs dims may be larger.

    Handles rsl-rl 5.x (``runner.alg.actor`` / ``runner.alg.critic``) and legacy
    (``runner.alg.policy``).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    alg = runner.alg
    all_stats: dict[str, Any] = {"path": checkpoint_path}

    # --- rsl-rl ≥ 5 format ---
    if "actor_state_dict" in ckpt and hasattr(alg, "actor"):
        all_stats["actor"] = load_partial_into_module(
            alg.actor, ckpt["actor_state_dict"], label="actor", verbose=verbose
        )
        if "critic_state_dict" in ckpt and hasattr(alg, "critic"):
            all_stats["critic"] = load_partial_into_module(
                alg.critic, ckpt["critic_state_dict"], label="critic", verbose=verbose
            )
        if verbose:
            print(f"[partial_checkpoint] loaded rsl-rl5 actor/critic from {checkpoint_path}")
        return all_stats

    # --- legacy single policy module ---
    if "model_state_dict" in ckpt:
        policy = getattr(alg, "policy", None) or getattr(alg, "actor_critic", None)
        if policy is None:
            raise RuntimeError("Checkpoint has model_state_dict but runner has no policy module")
        all_stats["policy"] = load_partial_into_module(
            policy, ckpt["model_state_dict"], label="policy", verbose=verbose
        )
        if verbose:
            print(f"[partial_checkpoint] loaded legacy policy from {checkpoint_path}")
        return all_stats

    raise KeyError(
        f"Unsupported checkpoint format at {checkpoint_path}. "
        f"Keys: {list(ckpt.keys())}"
    )


# Backward-compatible name used in early drafts
def load_partial_into_policy(
    policy: nn.Module,
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Load into a single module (legacy helper). Prefer :func:`load_partial_into_runner`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "actor_state_dict" in ckpt:
        # If only actor is provided to a single module, load actor weights
        state = ckpt["actor_state_dict"]
    else:
        state = ckpt
    return load_partial_into_module(policy, state, label="policy", verbose=verbose)
