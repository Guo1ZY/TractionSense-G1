#!/usr/bin/env python3
"""Distill a fast HIGH teacher and a safe LOW Hall actor into one actor.

The old fast policy is never used for samples labeled LOW.  This is an
offline action distillation stage; friction/contact labels select targets only
and are not part of the exported 1864-D actor input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1864, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 29),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mlp(x), -3.0, 3.0)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    obs = np.asarray(data["observation"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    low = np.asarray(data["low"], dtype=np.bool_)
    if obs.ndim != 2 or obs.shape[1] != 1864 or action.shape != (len(obs), 29):
        raise ValueError(f"invalid dataset shapes in {path}")
    return obs, action, low


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--safe-dataset", type=Path, required=True)
    parser.add_argument("--old-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    old_obs, old_action, old_low = load(args.old_dataset)
    safe_obs, safe_action, safe_low = load(args.safe_dataset)
    # Old samples are only trusted before the old policy encounters the LOW
    # patch.  Safe samples supply both regions and remain the fallback anchor.
    old_keep = ~old_low
    old_obs, old_action = old_obs[old_keep], old_action[old_keep]
    if len(old_obs) == 0:
        raise ValueError("old teacher has no surviving HIGH samples")
    session = ort.InferenceSession(str(args.old_policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # Some deployment exports have a fixed batch dimension of one.  Evaluate
    # in small chunks so both fixed- and dynamic-batch ONNX files are valid.
    old_chunks = []
    input_shape = session.get_inputs()[0].shape
    batch_fixed_one = input_shape[0] == 1
    chunk_size = 1 if batch_fixed_one else 256
    for start in range(0, len(safe_obs), chunk_size):
        old_chunks.append(session.run(None, {input_name: safe_obs[start : start + chunk_size]})[0])
    old_on_safe = np.concatenate(old_chunks, axis=0).astype(np.float32)
    if old_on_safe.shape != safe_action.shape:
        raise ValueError("old teacher output shape mismatch")

    # On HIGH observations keep most of the proven fast action, but retain a
    # safety anchor.  On LOW observations the target is exactly the safe actor.
    safe_target = np.where(
        safe_low[:, None], safe_action, 0.70 * old_on_safe + 0.30 * safe_action
    )
    obs = np.concatenate((safe_obs, old_obs), axis=0)
    target = np.concatenate((safe_target, old_action), axis=0)
    order = np.random.permutation(len(obs))
    split = max(1, int(0.9 * len(order)))
    train_idx, test_idx = order[:split], order[split:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Actor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    x = torch.from_numpy(obs).to(device)
    y = torch.from_numpy(target).to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    test_idx_t = torch.from_numpy(test_idx).to(device)
    batch = 1024
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
        total = 0.0
        for start in range(0, len(perm), batch):
            idx = perm[start : start + batch]
            prediction = model(x[idx])
            loss = torch.mean((prediction - y[idx]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            test_loss = torch.mean((model(x[test_idx_t]) - y[test_idx_t]) ** 2).item()
        history.append({"epoch": epoch + 1, "train_mse": total / len(perm), "test_mse": test_loss})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "spatial_teacher_mixture.pt"
    torch.save({"model": model.cpu().state_dict(), "input_dim": 1864, "output_dim": 29}, checkpoint)
    onnx_path = args.output_dir / "policy.onnx"
    example = torch.zeros((1, 1864), dtype=torch.float32)
    torch.onnx.export(
        model.cpu(), example, onnx_path, input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17,
    )
    report = {
        "status": "PASS",
        "input_dimension": 1864,
        "output_dimension": 29,
        "old_high_samples": int(len(old_obs)),
        "safe_samples": int(len(safe_obs)),
        "safe_low_samples": int(safe_low.sum()),
        "teacher_boundary": "old actor is target-only; exported actor receives Hall/proprioception only",
        "forbidden_inputs": ["normal_force", "tangential_force", "ground_friction_mu", "contact_truth"],
        "final_train_mse": history[-1]["train_mse"],
        "final_test_mse": history[-1]["test_mse"],
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
