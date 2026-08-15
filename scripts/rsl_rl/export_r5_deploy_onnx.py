#!/usr/bin/env python3
"""Export the R5 actor mean to a dynamic-batch ONNX without Isaac/Kit.

The R5 deployment graph is the deterministic FastBase composition:

    action = fastbase + effective_gate * bounded_hall_residual
             + stability_authority * bounded_stability_residual

The actor consumes the deployable 1864-D Hall/proprio observation and returns
29 JointPositionAction units.  The deployment side applies the audited 0.25
rad action scale and the trained default-position offset; the PI heading hold
is a separate velocity-command module and is intentionally not exported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unitree_rl_lab.traction.fastbase_capture_residual import (
    FastBaseHallCaptureStabilityResidual,
)
from unitree_rl_lab.traction.frozen_speedboost_teacher import (
    load_frozen_speedboost_teacher,
)


INPUT_DIM = 1864
ACTION_DIM = 29
ACTION_SCALE = 0.25

# Frozen by the R5 freeze contract; they must match the training runner.
R5_STABILITY = {
    "stability_limit": 0.75,
    "stability_heading_start": 0.02,
    "stability_heading_full": 0.18,
    "stability_tilt_start": 0.12,
    "stability_tilt_full": 0.30,
    "stability_omega_start": 0.35,
    "stability_omega_full": 1.20,
    "stability_turning_yaw_threshold": 0.05,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mlp(checkpoint: Path, teacher_checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError(f"{checkpoint}: missing actor_state_dict")
    teacher = load_frozen_speedboost_teacher(teacher_checkpoint, device="cpu")
    mlp = FastBaseHallCaptureStabilityResidual(
        teacher,
        residual_limit=0.55,
        gate_power=1.0,
        gate_logit_scale=2.75,
        gate_logit_bias=-3.2,
        teacher_trailing_mode="assume_fresh",
        structured_features=True,
        **R5_STABILITY,
    ).eval()
    mlp_state = {
        key[len("mlp."):]: value
        for key, value in actor_state.items()
        if key.startswith("mlp.")
    }
    mlp.load_state_dict(mlp_state, strict=True)
    for name, expected in R5_STABILITY.items():
        actual = float(getattr(mlp, name).item())
        if abs(actual - float(expected)) > 1.0e-6:
            raise RuntimeError(
                f"checkpoint stability buffer {name}={actual}, expected {expected}"
            )
    return mlp


class _DeployMean(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy_checkpoint", required=True, type=Path,
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=Path,
        default=Path(
            "/home/mosense/guo/unitree_rl_lab/artifacts/hall_speed_demo/"
            "speedboost112_frozen_teacher.pt"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    if not 0.0 < args.atol < 1.0:
        parser.error("--atol must be in (0, 1)")
    checkpoint = args.policy_checkpoint.expanduser().resolve()
    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    mlp = _load_mlp(checkpoint, teacher_path)
    wrapper = _DeployMean(mlp).eval()
    generator = torch.Generator().manual_seed(int(args.seed))
    sample = torch.randn(3, INPUT_DIM, generator=generator) * 0.15
    sample[:, 1830:1860] = 0.02
    sample[:, 1860:1862] = 1.0
    with torch.inference_mode():
        reference = wrapper(sample).numpy()
    if reference.shape != (3, ACTION_DIM) or not np.isfinite(reference).all():
        raise RuntimeError(
            f"actor mean invalid: shape={reference.shape}, finite={np.isfinite(reference).all()}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample[:1],
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(args.output))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: sample.numpy()},
    )[0]
    parity = float(np.max(np.abs(reference - actual)))
    if not np.isfinite(actual).all() or parity > float(args.atol):
        raise RuntimeError(f"ONNX parity/finiteness failed: max_abs={parity:.9g}")

    manifest = {
        "format": "r5-transition-retention-deploy-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "teacher_checkpoint": str(teacher_path),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "input_dim": INPUT_DIM,
        "output_dim": ACTION_DIM,
        "action_scale_rad": ACTION_SCALE,
        "deployment_formula": (
            "joint_target = default_joint_position + ACTION_SCALE * onnx(obs); "
            "obs is the deployable 1864-D Hall/proprio observation"
        ),
        "stability_buffers": dict(R5_STABILITY),
        "pi_heading_hold": "external velocity-command module, not exported",
        "onnx_parity_max_abs_error": parity,
        "finite": True,
        "status": "PASS",
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
