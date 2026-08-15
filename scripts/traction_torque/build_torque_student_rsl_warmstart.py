#!/usr/bin/env python3
"""Merge baseline locomotion and distilled Student into an RSL PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from unitree_rl_lab.traction_torque.networks import TorqueTractionStudentPolicy, torque_history_to_legacy_proprio


def _mlp(state: dict[str, torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    for index in (0, 2, 4):
        value = functional.elu(functional.linear(value, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
    return functional.linear(value, state["mlp.6.weight"], state["mlp.6.bias"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="Any matching torque Student RSL checkpoint; weights are replaced.")
    parser.add_argument("--baseline", type=Path, default=Path("model/rl/model_49999.pt"))
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/torque_student_rsl_warmstart.pt"))
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(); torch.manual_seed(args.seed)
    template = torch.load(args.template, map_location="cpu", weights_only=False)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    distilled = torch.load(args.distilled, map_location="cpu", weights_only=False)["student_state_dict"]
    actor, critic = template["actor_state_dict"], template["critic_state_dict"]
    baseline_actor, baseline_critic = baseline["actor_state_dict"], baseline["critic_state_dict"]
    for index in (0, 2, 4, 6):
        source_weight, target_weight = baseline_actor[f"mlp.{index}.weight"], actor[f"mlp.{index}.weight"]
        if index == 0:
            target_weight.zero_(); target_weight[:, :480].copy_(source_weight)
        else:
            target_weight.copy_(source_weight)
        actor[f"mlp.{index}.bias"].copy_(baseline_actor[f"mlp.{index}.bias"])
    if "distribution.std_param" in baseline_actor:
        actor["distribution.std_param"].copy_(baseline_actor["distribution.std_param"])
    missing = []
    for name, value in distilled.items():
        target = f"student_policy.{name}"
        if target not in actor or actor[target].shape != value.shape:
            missing.append(target)
        else:
            actor[target].copy_(value)
    if missing:
        raise KeyError(f"template cannot accept distilled parameters: {missing}")
    for name, value in baseline_critic.items():
        if name in critic and critic[name].shape == value.shape:
            critic[name].copy_(value)
    old_obs = torch.randn(32, 480)
    new_obs = torch.cat((old_obs, torch.randn(32, 19)), dim=-1)
    error = (_mlp(actor, new_obs) - _mlp(baseline_actor, old_obs)).abs()
    student = TorqueTractionStudentPolicy().eval()
    student.load_state_dict(distilled, strict=True)
    history = torch.randn(32, 15, 125)
    with torch.inference_mode():
        student_output = student(history)
        student_baseline = student.baseline_actor(torque_history_to_legacy_proprio(history))
    student_residual = (student_output.action - student_baseline).abs()
    output = {"actor_state_dict": actor, "critic_state_dict": critic, "iter": 0, "infos": {"source_baseline": str(args.baseline.resolve()), "source_distilled": str(args.distilled.resolve()), "new_actor_columns_zero": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(output, args.output)
    report = {"output": str(args.output.resolve()), "locomotion_head_baseline_action_max_abs_error": float(error.max()), "locomotion_head_baseline_action_mean_abs_error": float(error.mean()), "distilled_residual_max_abs": float(student_residual.max()), "distilled_residual_mean_abs": float(student_residual.mean()), "combined_initial_action_is_exact_baseline": bool(student_residual.max() == 0), "distilled_parameters_loaded": len(distilled), "missing_parameters": missing, "new_rsl_input_columns": 19, "new_columns_initialized_to_zero": True}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
