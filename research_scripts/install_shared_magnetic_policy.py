#!/usr/bin/env python3
"""Install a validated 1864->29 shared-magnetic ONNX into g1_ctrl.

This prepares a separate policy slot.  It deliberately does not activate the
slot in the robot FSM config; activation is a later, explicit validation step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort


INPUT_DIM = 1864
OUTPUT_DIM = 29
MAGNETIC_VALUES = 90
REPO_ROOT = Path(
    os.environ.get("TRACTIONSENSE_ROOT", Path(__file__).resolve().parents[1])
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=REPO_ROOT / "deploy/robots/g1_29dof",
    )
    parser.add_argument("--slot", default="traction_magnetic_lateral_guard")
    parser.add_argument(
        "--motion-feedback",
        action="store_true",
        help="Use [valid_lr, body_vy, relative_heading] in the final four slots",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional 640-D student deploy.yaml used for the common 480-D prefix.",
    )
    return parser.parse_args()


def validate_onnx(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    input_tail = model_input.shape[-1]
    output_tail = model_output.shape[-1]
    if input_tail != INPUT_DIM or output_tail != OUTPUT_DIM:
        raise ValueError(
            f"ONNX shape mismatch: input={model_input.shape}, "
            f"output={model_output.shape}"
        )
    output = session.run(
        [model_output.name],
        {model_input.name: np.zeros((1, INPUT_DIM), dtype=np.float32)},
    )[0]
    if output.shape != (1, OUTPUT_DIM) or not np.isfinite(output).all():
        raise ValueError(f"ONNX dry-run failed: shape={output.shape}")
    return {
        "input_name": model_input.name,
        "input_dim": INPUT_DIM,
        "output_name": model_output.name,
        "output_dim": OUTPUT_DIM,
        "zero_action_max_abs": float(np.max(np.abs(output))),
    }


def magnetic_observation_yaml(motion_feedback: bool = False) -> str:
    final_term = """  lateral_motion_feedback:
    params: {asset_name: robot, lateral_velocity_clip: 1.5, heading_error_clip: 1.0}
    clip: [-1.5, 1.5]
    scale: null
    history_length: 1
""" if motion_feedback else """  foot_sensor_age_lr:
    params: {}
    clip: [0.0, 1.0]
    scale: null
    history_length: 1
"""
    return f"""  foot_magnetic_array:
    params: {{}}
    clip: [-6.0, 6.0]
    scale: null
    history_length: 15
  foot_sample_period_lr:
    params: {{}}
    clip: [0.001, 0.25]
    scale: null
    history_length: 15
  foot_sensor_valid_lr:
    params: {{}}
    clip: [0.0, 1.0]
    scale: null
    history_length: 1
{final_term}"""


def prepare_template(text: str) -> str:
    """Apply the final deploy command/safety limits before replacing foot terms."""

    text, command_count = re.subn(
        r"(lin_vel_x:\s*)\[[^\]]+\]",
        r"\1[-0.3, 1.0]",
        text,
        count=1,
    )
    if command_count != 1:
        raise RuntimeError("template has no lin_vel_x command range")
    text, clip_count = re.subn(
        r"(raw_clip:\s*)(?:null|\[[^\]]+\])",
        r"\1[-4.0, 4.0]",
        text,
        count=1,
    )
    if clip_count != 1:
        raise RuntimeError("template has no action raw_clip field")
    return text


def main() -> int:
    args = parse_args()
    source_onnx = args.model_dir / "policy.onnx"
    source_metrics = args.model_dir / "metrics.json"
    if not source_onnx.is_file():
        raise FileNotFoundError(source_onnx)
    validation = validate_onnx(source_onnx)

    template = args.template
    if template is None:
        candidates = (
            args.deploy_root
            / "config/policy/velocity/traction_student_7989/params/deploy.yaml",
            args.deploy_root
            / "config/policy/velocity/traction_student/params/deploy.yaml",
        )
        template = next((path for path in candidates if path.is_file()), None)
    if template is None or not template.is_file():
        raise FileNotFoundError("no compatible traction Student deploy template")
    text = prepare_template(template.read_text(encoding="utf-8"))
    marker = "\n  foot_contact:\n"
    if marker not in text:
        raise RuntimeError(f"foot observation marker missing in {template}")
    deploy_yaml = (
        text.split(marker, 1)[0]
        + "\n"
        + magnetic_observation_yaml(args.motion_feedback)
    )

    slot = args.deploy_root / "config/policy/velocity" / args.slot
    exported = slot / "exported"
    params = slot / "params"
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_onnx, exported / "policy.onnx")
    (params / "deploy.yaml").write_text(deploy_yaml, encoding="utf-8")
    if source_metrics.is_file():
        shutil.copy2(source_metrics, slot / "distillation_metrics.json")
    source_checkpoint = args.model_dir / "shared_magnetic_policy.pt"
    if source_checkpoint.is_file():
        (slot / "checkpoint.txt").write_text(
            str(source_checkpoint.resolve()) + "\n", encoding="utf-8"
        )

    manifest = {
        "source_model_dir": str(args.model_dir.resolve()),
        "slot": str(slot.resolve()),
        "activated": False,
        "observation_layout": {
            "proprio_history": 480,
            "magnetic_history": 15 * 2 * 15 * 3,
            "period_history": 15 * 2,
            "health": 4,
            "health_semantics": (
                "valid_lr + body_vy + relative_heading"
                if args.motion_feedback
                else "valid_lr + age_lr"
            ),
            "total": INPUT_DIM,
        },
        "onnx": validation,
    }
    (slot / "install_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
