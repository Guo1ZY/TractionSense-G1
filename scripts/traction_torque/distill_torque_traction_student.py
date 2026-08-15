#!/usr/bin/env python3
"""Offline privileged-Teacher auxiliary training and torque Student distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torch.nn as nn

from unitree_rl_lab.traction_torque.networks import (
    TorqueTractionStudentCfg,
    TorqueTractionStudentPolicy,
    TorqueTractionTeacherCfg,
    TorqueTractionTeacherPolicy,
    torque_history_to_legacy_proprio,
)
from unitree_rl_lab.traction_torque.rsl_models import torque_teacher_history_to_legacy
from unitree_rl_lab.traction_torque.teacher_schema import (
    TORQUE_TEACHER_FRAME_DIM,
    TORQUE_TEACHER_HISTORY_FRAMES,
    TORQUE_TEACHER_PRIVILEGED_DIM,
)


class RslTeacherActionLabeler(nn.Module):
    """Deterministic PPO Teacher mean and latent for offline DAgger labels."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.encoder = nn.Sequential(
            nn.Linear(TORQUE_TEACHER_PRIVILEGED_DIM, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 16),
        )
        self.actor = nn.Sequential(
            nn.Linear(496, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29),
        )
        self.encoder.load_state_dict(
            {name.removeprefix("privileged_encoder."): value for name, value in state.items() if name.startswith("privileged_encoder.")},
            strict=True,
        )
        self.actor.load_state_dict(
            {name.removeprefix("mlp."): value for name, value in state.items() if name.startswith("mlp.")},
            strict=True,
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[-2:] != (TORQUE_TEACHER_HISTORY_FRAMES, TORQUE_TEACHER_FRAME_DIM):
            raise ValueError("PPO Teacher history must be [batch,5,248]")
        latent = self.encoder(history[:, -1, 99:])
        legacy = torque_teacher_history_to_legacy(history, critic=False)
        return self.actor(torch.cat((legacy, latent), dim=-1)), latent


def load_baseline(module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    module.load_state_dict({name: state[name] for name in module.state_dict()}, strict=True)


def auc_score(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(score) + 1)
    positive, negative = label.astype(bool), ~label.astype(bool)
    if not positive.any() or not negative.any(): return float("nan")
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def classification(score: np.ndarray, label: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred, target = score >= threshold, label.astype(bool)
    tp, fp, fn = (np.logical_and(pred, target).sum(), np.logical_and(pred, ~target).sum(), np.logical_and(~pred, target).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / max(precision + recall, 1e-12)), "auc": auc_score(score.reshape(-1), target.reshape(-1))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--baseline_checkpoint", type=Path, default=Path("model/rl/model_49999.pt"))
    parser.add_argument("--teacher_checkpoint", type=Path, help="PPO Teacher used for action/latent labels on Student-visited states")
    parser.add_argument("--teacher_epochs", type=int, default=120)
    parser.add_argument("--student_epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=Path("artifacts/traction_torque/torque_student_distilled.pt"))
    args = parser.parse_args(); torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=True)
    history_np, truth_force_np, analytical_np = data["student_history"], data["true_force_local_n"], data["estimated_force_local_n"]
    contact_np, mu_np = data["true_contact"].astype(np.float32), data["ground_friction_mu"]
    foot_velocity_np, base_velocity_np = data["foot_velocity_w_m_s"], data["base_velocity_b"]
    residual_np, condition_np = data["residual_norm_nm"], data["condition_score"]
    teacher_history_np = data["teacher_history"] if "teacher_history" in data.files else None
    if args.teacher_checkpoint is not None and teacher_history_np is None:
        raise KeyError("--teacher_checkpoint requires teacher_history in the dataset")
    steps, envs = history_np.shape[:2]; validation_count = max(1, envs // 4)
    train_env, val_env = np.arange(envs - validation_count), np.arange(envs - validation_count, envs)
    mass_gravity = 35.2793 * 9.81

    def tensor(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array[:, indices].reshape(-1, *array.shape[2:]), dtype=torch.float32, device=args.device)

    def bundle(indices: np.ndarray) -> dict[str, torch.Tensor]:
        history = tensor(history_np, indices)
        true_force = tensor(truth_force_np, indices)
        analytical = tensor(analytical_np, indices)
        contact = tensor(contact_np, indices)
        mu = tensor(mu_np, indices)
        foot_velocity = tensor(foot_velocity_np, indices)
        base_velocity = tensor(base_velocity_np, indices)
        residual = tensor(residual_np, indices); condition = tensor(condition_np, indices)
        teacher_history = tensor(teacher_history_np, indices) if teacher_history_np is not None else None
        truth_foot = true_force.reshape(-1, 2, 3)
        utilization = torch.linalg.vector_norm(truth_foot[..., :2], dim=-1) / (truth_foot[..., 2].abs() + 1.0)
        planar_speed = torch.linalg.vector_norm(foot_velocity[..., :2], dim=-1)
        # Exact contact combined with ankle rigid-body speed: a supervised slip
        # proxy, not contact-point slip truth.
        slip = contact * (planar_speed > 0.12).float()
        contact_sequence = contact_np[:, indices]
        contact_transition_np = np.zeros_like(contact_sequence, dtype=np.float32)
        contact_transition_np[1:] = (contact_sequence[1:] != contact_sequence[:-1]).astype(np.float32)
        contact_transition = torch.as_tensor(
            contact_transition_np.reshape(-1, 2), dtype=torch.float32, device=args.device
        )
        margin = (mu - utilization).clamp(-2, 2)
        confidence = torch.exp(-torch.abs(analytical - true_force).reshape(-1, 2, 3).mean(dim=-1) / 80.0)
        privilege = torch.cat((true_force / mass_gravity, contact, mu, foot_velocity.reshape(-1, 6), base_velocity, residual / 50.0, condition, (true_force - analytical) / mass_gravity, history[:, -1, 6:9]), dim=-1)
        assert privilege.shape[-1] == 32
        sample_weight = 1.0 + 4.0 * slip.max(dim=1).values + 2.0 * contact_transition.max(dim=1).values + (utilization.max(dim=1).values > 0.55).float()
        return {"history": history, "teacher_history": teacher_history, "true_force": true_force / mass_gravity, "analytical": analytical / mass_gravity, "contact": contact, "mu": mu, "utilization": utilization, "slip": slip, "margin": margin, "confidence": confidence, "privilege": privilege, "sample_weight": sample_weight}

    train, val = bundle(train_env), bundle(val_env)
    teacher = TorqueTractionTeacherPolicy(TorqueTractionTeacherCfg(privileged_input_dim=32)).to(args.device)
    load_baseline(teacher.baseline_actor, args.baseline_checkpoint)
    for parameter in teacher.baseline_actor.parameters(): parameter.requires_grad_(False)
    for parameter in teacher.actor_residual.parameters(): parameter.requires_grad_(False)
    teacher_optimizer = torch.optim.AdamW((p for p in teacher.parameters() if p.requires_grad), lr=3e-4)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for _ in range(args.teacher_epochs):
        balanced = torch.multinomial(train["sample_weight"], train["history"].shape[0], replacement=True, generator=generator)
        for index in balanced.split(args.batch_size):
            baseline = torque_history_to_legacy_proprio(train["history"][index]); command = train["history"][index, -1, 6:9]
            output = teacher(baseline, command, train["privilege"][index])
            loss = functional.binary_cross_entropy(output.slip_probability, train["slip"][index]) + 0.5 * functional.smooth_l1_loss(output.traction_margin, train["margin"][index]) + 0.5 * functional.binary_cross_entropy(output.contact_probability, train["contact"][index]) + 0.5 * functional.smooth_l1_loss(output.force_correction_target, train["true_force"][index] - train["analytical"][index])
            teacher_optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); teacher_optimizer.step()
    teacher.eval()
    teacher_labeler = None
    if args.teacher_checkpoint is not None:
        teacher_labeler = RslTeacherActionLabeler(args.teacher_checkpoint).to(args.device).eval()
        for parameter in teacher_labeler.parameters():
            parameter.requires_grad_(False)
    student = TorqueTractionStudentPolicy(TorqueTractionStudentCfg(freeze_baseline=True)).to(args.device)
    load_baseline(student.baseline_actor, args.baseline_checkpoint)
    student.force_gate_logit.data.fill_(-1.0)
    if teacher_labeler is None:
        for parameter in student.residual_actor.parameters(): parameter.requires_grad_(False)
        student.residual_gate_logit.requires_grad_(False)
    else:
        # The final residual layer is initialized to zero, so this still gives
        # an exact baseline action while preserving a useful action gradient.
        student.residual_gate_logit.data.fill_(-2.0)
    student_optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-5)
    curve = []
    for epoch in range(args.student_epochs):
        losses = []
        balanced = torch.multinomial(train["sample_weight"], train["history"].shape[0], replacement=True, generator=generator)
        for index in balanced.split(args.batch_size):
            with torch.no_grad():
                if teacher_labeler is None:
                    teacher_output = teacher(torque_history_to_legacy_proprio(train["history"][index]), train["history"][index, -1, 6:9], train["privilege"][index])
                    teacher_action, teacher_latent = teacher_output.action, teacher_output.traction_latent
                else:
                    teacher_action, teacher_latent = teacher_labeler(train["teacher_history"][index])
            output = student(train["history"][index])
            previous_history = torch.cat((train["history"][index, :1], train["history"][index, :-1]), dim=1)
            previous_output = student(previous_history)
            loss = (
                functional.mse_loss(output.traction_latent, teacher_latent)
                + functional.binary_cross_entropy(output.slip_probability, train["slip"][index])
                + 0.4 * functional.smooth_l1_loss(output.traction_utilization, train["utilization"][index])
                + 0.4 * functional.smooth_l1_loss(output.traction_margin, train["margin"][index])
                + 0.5 * functional.smooth_l1_loss(output.estimated_force, train["true_force"][index])
                + 0.2 * functional.mse_loss(output.estimator_confidence, train["confidence"][index])
                + 0.1 * functional.binary_cross_entropy(output.contact_probability.clamp(1e-5, 1 - 1e-5), train["contact"][index])
                + (functional.smooth_l1_loss(output.action, teacher_action) if teacher_labeler is not None else 0.0)
                + 0.02 * functional.smooth_l1_loss(output.traction_latent, previous_output.traction_latent)
                + 0.01 * functional.smooth_l1_loss(output.estimated_force, previous_output.estimated_force)
            )
            student_optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); student_optimizer.step(); losses.append(loss.detach())
        if (epoch + 1) % 10 == 0: curve.append({"epoch": epoch + 1, "loss": torch.stack(losses).mean().item()})
    student.eval()
    with torch.inference_mode():
        output = student(val["history"])
        baseline = student.baseline_actor(torque_history_to_legacy_proprio(val["history"]))
        if teacher_labeler is not None:
            validation_teacher_action, validation_teacher_latent = teacher_labeler(val["teacher_history"])
        else:
            validation_teacher_action, validation_teacher_latent = baseline, output.traction_latent
    action_error = (output.action - baseline).abs()
    report = {
        "dataset": str(args.dataset.resolve()), "seed": args.seed,
        "split": {"train_environments": train_env.tolist(), "validation_environments": val_env.tolist()},
        "validation_samples": int(val["history"].shape[0]),
        "slip_proxy_metrics": classification(output.slip_probability.cpu().numpy(), val["slip"].cpu().numpy()),
        "force_normalized_mae": (output.estimated_force - val["true_force"]).abs().mean(dim=0).cpu().tolist(),
        "traction_utilization_mae": float((output.traction_utilization - val["utilization"]).abs().mean()),
        "traction_margin_mae": float((output.traction_margin - val["margin"]).abs().mean()),
        "teacher_action_mae": float((output.action - validation_teacher_action).abs().mean()),
        "teacher_latent_mse": float(functional.mse_loss(output.traction_latent, validation_teacher_latent)),
        "baseline_action_max_abs_error": float(action_error.max()), "baseline_action_mean_abs_error": float(action_error.mean()),
        "teacher_training": "ppo_teacher_action_latent_labels" if teacher_labeler is not None else "offline_privileged_auxiliary_candidate_not_PPO",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()) if args.teacher_checkpoint is not None else None,
        "student_training": "dagger_style_student_visited_offline_distillation" if teacher_labeler is not None else "offline_distillation_candidate_not_on_policy",
        "slip_label": "exact contact AND ankle rigid-body planar-speed proxy > 0.12 m/s",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student_state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()}, "teacher_state_dict": {k: v.detach().cpu() for k, v in teacher.state_dict().items()}, "report": report, "curve": curve}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
