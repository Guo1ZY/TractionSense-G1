#!/usr/bin/env python3
"""Train causal analytical-force correction using deployment-only histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from unitree_rl_lab.traction_torque.networks import TemporalForceCorrector, TemporalForceCorrectorCfg


def metrics(estimate: torch.Tensor, truth: torch.Tensor) -> dict[str, list[float]]:
    error = estimate - truth
    return {
        "mae_n": error.abs().mean(dim=0).cpu().tolist(),
        "rmse_n": torch.sqrt(torch.square(error).mean(dim=0)).cpu().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/temporal_force_corrector.pt"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=True)
    if "student_history" not in data:
        raise KeyError("dataset lacks student_history; recollect with the current collector")
    history_np = data["student_history"]
    analytical_np = data["estimated_force_local_n"]
    truth_np = data["true_force_local_n"]
    contact_np = data["true_contact"]
    steps, envs = history_np.shape[:2]
    # Environment-disjoint validation prevents adjacent time windows from leaking.
    validation_envs = max(1, envs // 4)
    train_env = np.arange(0, envs - validation_envs)
    validation_env = np.arange(envs - validation_envs, envs)

    def select(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array[:, indices].reshape(-1, *array.shape[2:]), dtype=torch.float32, device=args.device)

    train_history, val_history = select(history_np, train_env), select(history_np, validation_env)
    train_analytical, val_analytical = select(analytical_np, train_env), select(analytical_np, validation_env)
    train_truth, val_truth = select(truth_np, train_env), select(truth_np, validation_env)
    train_contact = select(contact_np.astype(np.float32), train_env)
    model = TemporalForceCorrector(TemporalForceCorrectorCfg(maximum_correction_n=350.0)).to(args.device)
    # The module is zero-impact at construction. Offline supervised training
    # may safely open this gate because no locomotion action is being applied.
    model.gate_logit.data.fill_(-1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    best, best_state = float("inf"), None
    curve: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        permutation = torch.randperm(train_history.shape[0], generator=generator, device=args.device)
        model.train()
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            index = permutation[start : start + args.batch_size]
            output = model(train_history[index], train_analytical[index])
            target = train_truth[index]
            contact = train_contact[index].repeat_interleave(3, dim=1)
            force_loss = functional.smooth_l1_loss(output.corrected_force_n, target, beta=20.0)
            swing_loss = (torch.square(output.corrected_force_n) * (1.0 - contact)).mean() / 1000.0
            confidence_target = torch.exp(-torch.abs(output.corrected_force_n.detach() - target).reshape(-1, 2, 3).mean(dim=-1) / 80.0)
            confidence_loss = functional.mse_loss(output.confidence, confidence_target)
            loss = force_loss + 0.15 * swing_loss + 0.20 * confidence_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.detach())
        model.eval()
        with torch.inference_mode():
            validation = model(val_history, val_analytical)
            validation_loss = functional.smooth_l1_loss(validation.corrected_force_n, val_truth, beta=20.0).item()
        curve.append({"epoch": epoch + 1, "train_loss": torch.stack(losses).mean().item(), "validation_loss": validation_loss})
        if validation_loss < best:
            best = validation_loss
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        corrected = model(val_history, val_analytical).corrected_force_n
    report = {
        "dataset": str(args.dataset.resolve()), "seed": args.seed,
        "split": {"train_environments": train_env.tolist(), "validation_environments": validation_env.tolist()},
        "validation_samples": int(val_history.shape[0]), "best_validation_loss": best,
        "analytical": metrics(val_analytical, val_truth), "analytical_plus_temporal_correction": metrics(corrected, val_truth),
        "correction_inputs_privileged": False, "trained_on_ground_truth_labels": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "config": model.cfg.__dict__, "report": report, "curve": curve}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
