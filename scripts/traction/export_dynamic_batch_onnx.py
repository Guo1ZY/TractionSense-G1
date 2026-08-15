#!/usr/bin/env python3
"""Convert a fixed-batch deploy ONNX to dynamic batch and verify parity.

Only the leading input/output dimensions are changed.  The script refuses to
write the result unless ONNX checker validation, batch-1 parity, and repeated
batch parity all pass.  This keeps the original deploy graph untouched while
making Isaac Lab matrix evaluation efficient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    return parser.parse_args()


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def main() -> int:
    args = parse_args()
    if args.batch < 2:
        raise ValueError("--batch must be at least 2")
    if args.atol <= 0.0:
        raise ValueError("--atol must be positive")

    model = onnx.load(str(args.input))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("expected exactly one graph input and output")
    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    input_shape = input_value.type.tensor_type.shape
    output_shape = output_value.type.tensor_type.shape
    if len(input_shape.dim) != 2 or len(output_shape.dim) != 2:
        raise ValueError("expected rank-2 policy input and output")
    input_dim = int(input_shape.dim[1].dim_value)
    output_dim = int(output_shape.dim[1].dim_value)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("policy feature dimensions must be static and positive")

    for value in (input_value, output_value):
        leading = value.type.tensor_type.shape.dim[0]
        leading.ClearField("dim_value")
        leading.dim_param = "batch"

    # PyTorch exports ``tensor.reshape(batch, history, -1)`` with the example
    # batch baked into a Constant when ``dynamic_axes`` was omitted.  Merely
    # changing graph metadata therefore fails later at Concat.  Rewrite this
    # specific, auditable encoder shape to infer batch and retain the known
    # 15 sensors x 16 point features = 240 flattened features.  Do not touch
    # scalar Slice/Unsqueeze constants that also contain the value one.
    rewritten_shapes = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value":
                continue
            value = numpy_helper.to_array(attribute.t)
            if value.shape == (3,) and np.array_equal(
                value, np.asarray([1, 15, -1], dtype=value.dtype)
            ):
                replacement = np.asarray([-1, 15, 240], dtype=value.dtype)
                attribute.t.CopyFrom(
                    numpy_helper.from_array(replacement, name=attribute.t.name)
                )
                rewritten_shapes += 1
    if rewritten_shapes == 0:
        raise ValueError(
            "no fixed encoder reshape constants were found; refusing an "
            "unverified metadata-only conversion"
        )
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))

    original = _session(args.input)
    dynamic = _session(args.output)
    rng = np.random.default_rng(args.seed)
    sample = rng.normal(0.0, 0.25, size=(1, input_dim)).astype(np.float32)
    original_output = original.run(
        None, {original.get_inputs()[0].name: sample}
    )[0]
    dynamic_output = dynamic.run(
        None, {dynamic.get_inputs()[0].name: sample}
    )[0]
    batch = np.repeat(sample, args.batch, axis=0)
    batch_reference = np.repeat(original_output, args.batch, axis=0)
    batch1_error = float(np.max(np.abs(original_output - dynamic_output)))
    batch_error: str | None = None
    try:
        batch_output = dynamic.run(
            None, {dynamic.get_inputs()[0].name: batch}
        )[0]
        repeated_error = float(np.max(np.abs(batch_reference - batch_output)))
        finite = bool(
            np.isfinite(dynamic_output).all() and np.isfinite(batch_output).all()
        )
    except Exception as error:  # ONNX Runtime supplies the actionable node name.
        repeated_error = float("inf")
        finite = False
        batch_error = f"{type(error).__name__}: {error}"
    passed = finite and max(batch1_error, repeated_error) <= args.atol
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "schema": {"input_dim": input_dim, "output_dim": output_dim},
        "rewritten_encoder_shapes": rewritten_shapes,
        "validation_batch": args.batch,
        "batch1_max_abs_error": batch1_error,
        "repeated_batch_max_abs_error": repeated_error,
        "finite": finite,
        "tolerance": args.atol,
        "status": "PASS" if passed else "FAIL",
    }
    if batch_error is not None:
        report["batch_error"] = batch_error
    report_path = args.output.with_suffix(".dynamic_batch.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        args.output.unlink(missing_ok=True)
        raise RuntimeError("dynamic-batch ONNX parity validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
