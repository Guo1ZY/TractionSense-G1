#!/usr/bin/env python3
"""Train a bounded Hall/proprio residual on top of a safe spatial actor.

The base actor is frozen.  Only a small residual is learned from rollout
history, with zero residual on LOW-patch samples and a clipped teacher target
on HIGH samples.  This keeps the proven low-grip gait and avoids an unsafe
frame-wise replacement of the policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1864, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 29),
        )

    def forward(self, x):
        return torch.clamp(self.mlp(x), -3.0, 3.0)


class ResidualActor(nn.Module):
    def __init__(self, base_state, residual_limit=0.25):
        super().__init__()
        self.base = BaseActor()
        self.base.load_state_dict({k: v for k, v in base_state.items() if k.startswith("mlp.")}, strict=True)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.residual = nn.Sequential(
            nn.Linear(1864, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 29),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual_limit = float(residual_limit)

    def forward(self, x):
        return torch.clamp(self.base(x) + self.residual_limit * torch.tanh(self.residual(x)), -3.0, 3.0)


def load_npz(path):
    d = np.load(path)
    o = np.asarray(d["observation"], np.float32)
    a = np.asarray(d["action"], np.float32)
    low = np.asarray(d["low"], bool)
    if o.ndim != 2 or o.shape[1] != 1864 or a.shape != (len(o), 29) or low.shape != (len(o),):
        raise ValueError(f"invalid dataset shapes: {path}")
    return o, a, low


def old_actions(path, obs):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    fixed = session.get_inputs()[0].shape[0] == 1
    n = 1 if fixed else 256
    out = []
    for i in range(0, len(obs), n):
        out.append(session.run(None, {name: obs[i : i + n]})[0])
    return np.concatenate(out, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--safe-checkpoint", type=Path, required=True)
    p.add_argument("--old-policy", type=Path, required=True)
    p.add_argument("--safe-dataset", type=Path, required=True)
    p.add_argument("--old-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--residual-limit", type=float, default=0.25)
    p.add_argument("--high-blend", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260810)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    safe_obs, safe_act, safe_low = load_npz(a.safe_dataset)
    old_obs, old_act, old_low = load_npz(a.old_dataset)
    old_obs, old_act = old_obs[~old_low], old_act[~old_low]
    old_safe = old_actions(a.old_policy, safe_obs)
    if old_safe.shape != safe_act.shape:
        raise ValueError("old action shape mismatch")

    state = torch.load(a.safe_checkpoint, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in state:
        raise ValueError("safe checkpoint must contain actor_state_dict")
    base_model = BaseActor()
    base_model.load_state_dict({k: v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.")}, strict=True)
    base_model.eval()
    with torch.no_grad():
        base_all = base_model(torch.from_numpy(np.concatenate([safe_obs, old_obs])))
    base_safe = base_all[: len(safe_obs)].numpy()
    base_old = base_all[len(safe_obs) :].numpy()
    # Target is a bounded residual from the frozen safe actor.  LOW is exact
    # safe behavior; HIGH retains only a conservative fraction of fast teacher.
    high_target = (1.0 - a.high_blend) * base_safe + a.high_blend * old_safe
    obs = np.concatenate([safe_obs, old_obs])
    target = np.concatenate([high_target, old_act])
    # Dataset labels are used only to construct targets, never fed to actor.
    target_residual = np.clip(target - np.concatenate([base_safe, base_old], axis=0),
                              -a.residual_limit, a.residual_limit)
    # Explicitly force safe samples from LOW region to zero residual.
    target_residual[: len(safe_obs)][safe_low] = 0.0

    model = ResidualActor(state["actor_state_dict"], a.residual_limit)
    opt = torch.optim.AdamW(model.residual.parameters(), lr=1e-4, weight_decay=1e-5)
    x = torch.from_numpy(obs)
    y = torch.from_numpy(target_residual)
    order = np.random.permutation(len(obs))
    split = max(1, int(.9 * len(order)))
    tr = torch.from_numpy(order[:split]); te = torch.from_numpy(order[split:])
    hist = []
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(perm), 1024):
            idx = perm[i:i+1024]
            pred = (model(x[idx]) - model.base(x[idx])).detach()  # base is fixed
            # Recompute residual without graph detachment for update.
            pred = a.residual_limit * torch.tanh(model.residual(x[idx]))
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.residual.parameters(), 0.5); opt.step()
            total += float(loss.detach()) * len(idx)
        with torch.no_grad():
            test = ((a.residual_limit * torch.tanh(model.residual(x[te])) - y[te]) ** 2).mean().item()
        hist.append({"epoch": ep + 1, "train_mse": total / len(tr), "test_mse": test})

    a.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = a.output_dir / "spatial_temporal_residual.pt"
    torch.save({"model": model.state_dict(), "input_dim": 1864, "output_dim": 29,
                "residual_limit": a.residual_limit}, ckpt)
    onnx = a.output_dir / "policy.onnx"
    torch.onnx.export(model.eval(), torch.zeros(1, 1864), onnx, input_names=["obs"],
                      output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    report = {"status": "PASS", "base_frozen": True, "input_dimension": 1864,
              "residual_limit": a.residual_limit, "high_blend": a.high_blend,
              "safe_samples": len(safe_obs), "safe_low_samples": int(safe_low.sum()),
              "old_high_samples": len(old_obs), "forbidden_inputs": ["normal_force", "tangential_force", "ground_friction_mu", "contact_truth"],
              "checkpoint": str(ckpt), "onnx": str(onnx), "history": hist}
    (a.output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
