#!/usr/bin/env python3
"""Export a fixed canonical Student checkpoint with its complete deploy schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.deployment import (  # noqa: E402
    DeploymentObservationCfg,
    deployment_metadata,
)
from unitree_rl_lab.traction.governor import (  # noqa: E402
    TractionAwareCommandGovernorCfg,
)
from unitree_rl_lab.traction.networks import (  # noqa: E402
    GatedTractionPolicy,
    temporal_history_to_legacy_proprio,
)
from unitree_rl_lab.traction.rsl_models import (  # noqa: E402
    CanonicalStudentRslModel,
)
from unitree_rl_lab.traction.schema import (  # noqa: E402
    TEMPORAL_STUDENT_FRAME_SCHEMA,
)
from unitree_rl_lab.traction.tactile import (  # noqa: E402
    TactileDomainRandomizationCfg,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _GatedExportModel(torch.nn.Module):
    def __init__(self, policy: GatedTractionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history = observation.reshape(
            -1,
            TEMPORAL_STUDENT_FRAME_SCHEMA.history_frames,
            TEMPORAL_STUDENT_FRAME_SCHEMA.frame_dimension,
        )
        output = self.policy(
            temporal_history_to_legacy_proprio(history),
            history,
            history[:, -1, 93:96],
        )
        return (
            output.action_mean,
            output.slip_probability,
            output.traction_score,
            output.sensor_confidence,
        )


def _student(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, dict, str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "student_policy_state_dict" in checkpoint:
        policy = GatedTractionPolicy()
        policy.load_state_dict(
            checkpoint["student_policy_state_dict"],
            strict=True,
        )
        policy.eval()
        return _GatedExportModel(policy).eval(), checkpoint, "gated_baseline_residual"
    if "actor_state_dict" not in checkpoint:
        raise ValueError(
            "checkpoint must contain actor_state_dict or student_policy_state_dict"
        )
    dimension = TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension
    example = TensorDict(
        {"student": torch.zeros((1, dimension))},
        batch_size=[1],
    )
    model = CanonicalStudentRslModel(
        example,
        {"actor": ["student"]},
        "actor",
        29,
        latent_dim=16,
        temporal_variant="gru",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.45,
            "std_type": "scalar",
        },
    )
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()
    return model.as_onnx().eval(), checkpoint, "rsl_student"


def _sample(path: Path | None) -> torch.Tensor:
    dimension = TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension
    if path is None:
        result = torch.zeros((1, dimension), dtype=torch.float32)
        result[:, -4:-2] = 1.0
        return result
    with np.load(path, allow_pickle=False) as archive:
        history = np.asarray(archive["student_history"][0:1], dtype=np.float32)
    if history.shape != (1, dimension):
        raise ValueError(f"sample history shape {history.shape}, expected {(1, dimension)}")
    return torch.from_numpy(history)


def _maximum_errors(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[np.ndarray, ...],
) -> dict[str, float]:
    names = ("action", "slip_probability", "traction_score", "sensor_confidence")
    return {
        f"{name}_max_abs_error": float(
            np.max(np.abs(expected.detach().numpy() - actual))
        )
        for name, expected, actual in zip(names, reference, candidate, strict=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_dataset", type=Path)
    parser.add_argument(
        "--training_status",
        choices=(
            "smoke_untrained",
            "baseline_reference",
            "trained_candidate",
            "validated",
        ),
        default="smoke_untrained",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, architecture = _student(args.checkpoint)
    sample = _sample(args.sample_dataset)
    with torch.inference_mode():
        reference = model(sample)

    torchscript_path = args.output_dir / "traction_student.ts"
    traced = torch.jit.trace(model, sample, strict=False)
    traced.save(str(torchscript_path))
    with torch.inference_mode():
        jit_output = tuple(value.numpy() for value in traced(sample))

    onnx_path = args.output_dir / "traction_student.onnx"
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["student_history"],
        output_names=[
            "action",
            "slip_probability",
            "traction_score",
            "sensor_confidence",
        ],
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = tuple(
        session.run(None, {"student_history": sample.numpy()})
    )
    jit_errors = _maximum_errors(reference, jit_output)
    onnx_errors = _maximum_errors(reference, onnx_output)

    observation_cfg = DeploymentObservationCfg()
    metadata = deployment_metadata(observation_cfg)
    metadata.update(
        {
            "artifact_version": "traction_deploy_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_iteration": int(
                checkpoint.get("iter", checkpoint.get("epoch", -1))
            ),
            "policy_architecture": architecture,
            "training_status": args.training_status,
            "policy_input": {
                "name": "student_history",
                "shape": [1, TEMPORAL_STUDENT_FRAME_SCHEMA.flat_dimension],
                "dtype": "float32",
            },
            "policy_outputs": {
                "action": [1, 29],
                "slip_probability": [1, 2],
                "traction_score": [1, 1],
                "sensor_confidence": [1, 1],
            },
            "export_validation": {
                "torchscript": jit_errors,
                "onnxruntime": onnx_errors,
            },
        }
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    TEMPORAL_STUDENT_FRAME_SCHEMA.write_json(
        args.output_dir / "observation_schema.json"
    )
    (args.output_dir / "tactile_randomization.json").write_text(
        json.dumps(
            asdict(TactileDomainRandomizationCfg()),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command_governor.json").write_text(
        json.dumps(
            asdict(TractionAwareCommandGovernorCfg()),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Canonical traction policy export\n\n"
        "This package contains a fixed 29-action Student policy. Input is the "
        "15×106 time-major canonical history. Force is signed N in each "
        "ankle-roll local frame before normalization by robot weight. The "
        "runtime estimates slip/traction/confidence, applies the command "
        "governor, and performs a second fixed-policy pass with the adjusted "
        "command. This package does not authorize real G1 actuation.\n",
        encoding="utf-8",
    )
    result = {
        "output_dir": str(args.output_dir.resolve()),
        "onnx_sha256": _sha256(onnx_path),
        "torchscript_sha256": _sha256(torchscript_path),
        "onnxruntime": onnx_errors,
        "torchscript": jit_errors,
        "training_status": args.training_status,
    }
    print(json.dumps(result, indent=2))
    tolerance = 1.0e-4
    if max((*onnx_errors.values(), *jit_errors.values())) > tolerance:
        raise RuntimeError(f"export validation exceeded {tolerance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
