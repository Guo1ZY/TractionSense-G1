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
    copy old columns and initialize every new input column to zero.
  * Optimizer state is NOT loaded (obs layout changed → Adam moments invalid).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from unitree_rl_lab.traction.schema import (
    legacy_actor_schema,
    legacy_critic_schema,
    old_to_new_flat_index,
)


def _known_observation_mapping(old_dim: int, new_dim: int) -> list[int] | None:
    """Return a semantic mapping for audited G1 actor/critic expansions."""
    # Motion-Hall policy/critic schemas append deployable Hall history and
    # privileged critic terms after exact legacy prefixes.  Make this an
    # explicit audited contract instead of relying on the generic unknown-
    # dimension prefix fallback: model_49999 actor[480] -> policy[1864] and
    # critic[495] -> critic[570].
    if (old_dim, new_dim) in ((480, 482), (480, 1864), (495, 570)):
        return list(range(old_dim))
    schema_pairs = (
        (
            legacy_actor_schema(include_force=False),
            legacy_actor_schema(include_force=True),
        ),
        (
            legacy_critic_schema(include_force=False),
            legacy_critic_schema(include_force=True),
        ),
    )
    for old_schema, new_schema in schema_pairs:
        if (
            old_schema.flat_dimension == old_dim
            and new_schema.flat_dimension == new_dim
        ):
            return old_to_new_flat_index(old_schema, new_schema).tolist()
    return None


def _expand_linear_weight(
    dst: torch.Tensor,
    src: torch.Tensor,
    input_mapping: Sequence[int] | None = None,
) -> torch.Tensor | None:
    """Expand an input weight using explicit old-to-new observation columns.

    If no mapping is supplied, the two audited G1 expansions (480→510 and
    495→525) obtain a semantic mapping from the canonical observation schema.
    Unknown legacy expansions retain prefix-copy behavior for compatibility.
    """
    if dst.ndim != 2 or src.ndim != 2:
        return None
    if dst.shape[0] != src.shape[0]:
        return None
    if dst.shape[1] < src.shape[1]:
        return None
    if dst.shape[1] == src.shape[1]:
        return src.clone()
    if input_mapping is None:
        input_mapping = _known_observation_mapping(src.shape[1], dst.shape[1])
    if input_mapping is None:
        input_mapping = range(src.shape[1])
    mapping = torch.as_tensor(input_mapping, dtype=torch.long, device=dst.device)
    if mapping.ndim != 1 or mapping.numel() != src.shape[1]:
        raise ValueError(
            f"input mapping must contain {src.shape[1]} columns, "
            f"got shape {tuple(mapping.shape)}"
        )
    if mapping.numel() and (
        int(mapping.min()) < 0 or int(mapping.max()) >= dst.shape[1]
    ):
        raise ValueError(
            f"input mapping range is outside new input dimension {dst.shape[1]}"
        )
    if torch.unique(mapping).numel() != mapping.numel():
        raise ValueError("input mapping must be one-to-one")

    # New input columns start at zero so missing/new force inputs cannot perturb
    # the warm-started policy or value before fine-tuning.
    out = torch.zeros_like(dst)
    out[:, mapping] = src
    return out


def merge_state_dicts(
    current: dict[str, torch.Tensor],
    loaded: dict[str, torch.Tensor],
    *,
    verbose: bool = True,
    label: str = "",
    input_mappings: dict[str, Sequence[int]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Merge ``loaded`` into ``current`` with input-dim expansion for Linear layers."""
    merged = {k: v.clone() if torch.is_tensor(v) else v for k, v in current.items()}
    stats = {
        "copied": [],
        "expanded": [],
        "mapping": [],
        "skipped_missing": [],
        "skipped_shape": [],
        "new_initialized": [],
        "not_in_checkpoint": sorted(set(current) - set(loaded)),
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
        mapping = None if input_mappings is None else input_mappings.get(key)
        known_mapping = _known_observation_mapping(src.shape[-1], dst.shape[-1])
        expanded = _expand_linear_weight(
            dst,
            src.to(dtype=dst.dtype, device=dst.device),
            input_mapping=mapping,
        )
        if expanded is not None:
            merged[key] = expanded
            stats["expanded"].append(f"{key}: {tuple(src.shape)} -> {tuple(expanded.shape)}")
            if mapping is not None:
                mapping_source = "caller"
                used_mapping = list(mapping)
            elif known_mapping is not None:
                mapping_source = "canonical_schema"
                used_mapping = known_mapping
            else:
                mapping_source = "legacy_prefix_fallback"
                used_mapping = list(range(src.shape[1]))
            new_columns = sorted(set(range(dst.shape[1])) - set(used_mapping))
            stats["mapping"].append(
                {
                    "key": key,
                    "source": mapping_source,
                    "old_columns": src.shape[1],
                    "new_columns": dst.shape[1],
                }
            )
            stats["new_initialized"].append(
                {
                    "key": key,
                    "count": len(new_columns),
                    "initialization": "zero",
                }
            )
            continue
        stats["skipped_shape"].append(f"{key}: ckpt{tuple(src.shape)} vs model{tuple(dst.shape)}")

    if verbose:
        tag = f"[partial_checkpoint{('/' + label) if label else ''}]"
        print(f"{tag} copied: {len(stats['copied'])}")
        print(f"{tag} expanded: {len(stats['expanded'])}")
        for line in stats["expanded"]:
            print("   ", line)
        for entry in stats["mapping"]:
            print(
                f"    {entry['key']} mapping={entry['source']} "
                f"({entry['old_columns']} -> {entry['new_columns']})"
            )
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
    input_mappings: dict[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Merge a state_dict into a module with possible first-layer expansion."""
    current = module.state_dict()
    merged, stats = merge_state_dicts(
        current,
        state_dict,
        verbose=verbose,
        label=label,
        input_mappings=input_mappings,
    )
    module.load_state_dict(merged, strict=True)
    return stats


def load_partial_into_runner(
    runner: Any,
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    verbose: bool = True,
    actor_input_mappings: dict[str, Sequence[int]] | None = None,
    critic_input_mappings: dict[str, Sequence[int]] | None = None,
    load_actor: bool = True,
    load_critic: bool = True,
) -> dict[str, Any]:
    """Load a baseline checkpoint into a runner whose obs dims may be larger.

    Handles rsl-rl 5.x (``runner.alg.actor`` / ``runner.alg.critic``) and legacy
    (``runner.alg.policy``).
    """
    if not load_actor and not load_critic:
        raise ValueError("partial checkpoint must load actor, critic, or both")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    alg = runner.alg
    all_stats: dict[str, Any] = {"path": checkpoint_path}

    # --- rsl-rl ≥ 5 format ---
    if "actor_state_dict" in ckpt and hasattr(alg, "actor"):
        if load_actor:
            all_stats["actor"] = load_partial_into_module(
                alg.actor,
                ckpt["actor_state_dict"],
                label="actor",
                verbose=verbose,
                input_mappings=actor_input_mappings,
            )
        if load_critic:
            if "critic_state_dict" not in ckpt or not hasattr(alg, "critic"):
                raise KeyError(
                    "critic-only partial warm-start requested but checkpoint/runner "
                    "does not provide critic_state_dict and alg.critic"
                )
            all_stats["critic"] = load_partial_into_module(
                alg.critic,
                ckpt["critic_state_dict"],
                label="critic",
                verbose=verbose,
                input_mappings=critic_input_mappings,
            )
        if verbose:
            loaded_names = "/".join(name for name in ("actor", "critic") if name in all_stats)
            print(
                f"[partial_checkpoint] loaded rsl-rl5 {loaded_names} "
                f"from {checkpoint_path}"
            )
        return all_stats

    # --- legacy single policy module ---
    if "model_state_dict" in ckpt:
        if not load_actor:
            raise ValueError(
                "critic-only partial warm-start requires an rsl-rl5 checkpoint "
                "with a separate critic_state_dict"
            )
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
