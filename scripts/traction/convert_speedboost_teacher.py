#!/usr/bin/env python3
"""Convert the canonical speedboost112 ONNX composite into a frozen PyTorch Teacher.

This is a deliberately fail-closed converter for the audited ONNX artifact.  It
does not try to translate arbitrary ONNX graphs.  Instead it reconstructs the
known source architecture, maps all 82 graph initializers (80 model/buffer
values plus two validated constant-folded lerp tensors), and verifies the full
417-node graph against ONNX Runtime before writing the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import torch
from onnx import numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.traction.frozen_speedboost_teacher import (  # noqa: E402
    INPUT_DIM,
    KNOWN_SPEEDBOOST112_SHA256,
    OUTPUT_DIM,
    FrozenSpeedBoostTeacher,
    SpeedBoostTeacherConfig,
    save_frozen_speedboost_teacher,
)


EXPECTED_IR_VERSION = 8
EXPECTED_OPSET = 17
EXPECTED_NODE_COUNT = 417
EXPECTED_INITIALIZER_COUNT = 82
EXPECTED_OP_COUNTS = {
    "Abs": 2,
    "Add": 44,
    "Clip": 2,
    "Concat": 12,
    "Constant": 82,
    "Conv": 24,
    "Div": 2,
    "Elu": 76,
    "Gather": 17,
    "Gemm": 24,
    "Identity": 20,
    "Less": 1,
    "MatMul": 32,
    "Mul": 18,
    "ReduceMax": 1,
    "ReduceMean": 4,
    "ReduceMin": 1,
    "Reshape": 12,
    "Sigmoid": 5,
    "Slice": 10,
    "Squeeze": 1,
    "Sub": 11,
    "Transpose": 8,
    "Unsqueeze": 6,
    "Where": 2,
}

# torch.onnx exports the five Linear/embedding weights below as MatMul/Add
# initializers rather than preserving their state_dict names.  These names are
# tied to the canonical source SHA and must never be guessed for a new graph.
SPECIAL_INITIALIZERS = {
    "fast": (
        "onnx::MatMul_663",
        "onnx::MatMul_664",
        "onnx::Add_665",
        "onnx::MatMul_674",
        "onnx::MatMul_675",
    ),
    "stable": (
        "onnx::MatMul_731",
        "onnx::MatMul_732",
        "onnx::Add_733",
        "onnx::MatMul_734",
        "onnx::MatMul_735",
    ),
    "safe": (
        "onnx::MatMul_742",
        "onnx::MatMul_743",
        "onnx::Add_744",
        "onnx::MatMul_745",
        "onnx::MatMul_746",
    ),
}

SPECIAL_STATE_SUFFIXES = (
    "foot_encoder.point_mlp.0.weight",
    "foot_encoder.point_mlp.2.weight",
    "foot_encoder.sensor_embedding",
    "foot_encoder.frame_mlp.0.weight",
    "foot_encoder.frame_mlp.2.weight",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        dimensions.append(dimension.dim_param if dimension.dim_param else int(dimension.dim_value))
    return dimensions


def graph_signature(model: onnx.ModelProto) -> dict[str, Any]:
    opsets = {entry.domain: int(entry.version) for entry in model.opset_import}
    return {
        "ir_version": int(model.ir_version),
        "opset": int(opsets.get("", -1)),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "inputs": [{"name": value.name, "shape": _shape(value)} for value in model.graph.input],
        "outputs": [{"name": value.name, "shape": _shape(value)} for value in model.graph.output],
    }


def validate_canonical_graph(model: onnx.ModelProto, source_sha256: str) -> dict[str, Any]:
    if source_sha256.lower() != KNOWN_SPEEDBOOST112_SHA256:
        raise ValueError(
            "refusing numeric initializer mapping for an unaudited ONNX graph: "
            f"expected SHA256 {KNOWN_SPEEDBOOST112_SHA256}, got {source_sha256.lower()}"
        )
    onnx.checker.check_model(model)
    signature = graph_signature(model)
    expected = {
        "ir_version": EXPECTED_IR_VERSION,
        "opset": EXPECTED_OPSET,
        "node_count": EXPECTED_NODE_COUNT,
        "initializer_count": EXPECTED_INITIALIZER_COUNT,
        "op_counts": EXPECTED_OP_COUNTS,
        "inputs": [{"name": "obs", "shape": ["batch", INPUT_DIM]}],
        "outputs": [{"name": "actions", "shape": ["batch", OUTPUT_DIM]}],
    }
    if signature != expected:
        raise ValueError(
            "canonical speedboost112 graph signature mismatch:\n"
            + json.dumps({"expected": expected, "actual": signature}, indent=2, sort_keys=True)
        )
    return signature


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        if initializer.name in result:
            raise ValueError(f"duplicate initializer {initializer.name!r}")
        # Copy avoids read-only NumPy buffers and makes transpose/reshape safe.
        result[initializer.name] = np.array(numpy_helper.to_array(initializer), copy=True)
    return result


def _constants(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
        if len(tensor_attributes) != 1:
            raise ValueError(f"Constant node {node.name!r} does not contain exactly one tensor value")
        result[node.output[0]] = np.array(numpy_helper.to_array(tensor_attributes[0].t), copy=True)
    return result


def _scalar(constants: dict[str, np.ndarray], name: str) -> float:
    if name not in constants:
        raise KeyError(f"missing ONNX Constant output {name!r}")
    value = constants[name]
    if value.size != 1:
        raise ValueError(f"ONNX Constant {name!r} must be scalar, got shape {value.shape}")
    return float(value.reshape(()))


def recover_config(model: onnx.ModelProto) -> SpeedBoostTeacherConfig:
    constants = _constants(model)
    config = SpeedBoostTeacherConfig(
        residual_center=_scalar(constants, "/Constant_8_output_0"),
        residual_sharpness=_scalar(constants, "/Constant_9_output_0"),
        evidence_center=_scalar(constants, "/Constant_10_output_0"),
        evidence_sharpness=_scalar(constants, "/Constant_11_output_0"),
        mu_max=_scalar(constants, "/Constant_37_output_0"),
        traction_center=_scalar(constants, "/Constant_38_output_0"),
        traction_sharpness=_scalar(constants, "/Constant_39_output_0"),
        command_center=_scalar(constants, "/Constant_40_output_0"),
        command_sharpness=_scalar(constants, "/Constant_41_output_0"),
        boost_factor=1.0 + _scalar(constants, "/Constant_44_output_0"),
        stable_uses_boosted_command=True,
    )
    config.validate()
    command_indices = tuple(int(value) for value in constants["onnx::Gather_310"].reshape(-1))
    if command_indices != (30, 33, 36, 39, 42):
        raise ValueError(f"unexpected command-vx indices in ONNX graph: {command_indices}")
    return config


def _tensor_for_state(array: np.ndarray, expected: torch.Tensor, state_name: str) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(array))
    if value.shape != expected.shape:
        raise ValueError(
            f"shape mismatch for {state_name}: recovered {tuple(value.shape)}, expected {tuple(expected.shape)}"
        )
    if value.dtype != expected.dtype:
        raise ValueError(f"dtype mismatch for {state_name}: recovered {value.dtype}, expected {expected.dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"non-finite ONNX initializer mapped to {state_name}")
    return value


def recover_model(model: onnx.ModelProto) -> tuple[FrozenSpeedBoostTeacher, dict[str, Any]]:
    initializers = _initializers(model)
    teacher = FrozenSpeedBoostTeacher(recover_config(model))
    template = teacher.state_dict()
    recovered: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    # Values whose ONNX names are stable state_dict/buffer names.
    for state_name in template:
        if state_name in initializers:
            recovered[state_name] = _tensor_for_state(initializers[state_name], template[state_name], state_name)
            consumed.add(state_name)

    command_initializer = "onnx::Mul_690"
    recovered["command_mask"] = _tensor_for_state(
        initializers[command_initializer].reshape(INPUT_DIM), template["command_mask"], "command_mask"
    )
    consumed.add(command_initializer)

    for branch, initializer_names in SPECIAL_INITIALIZERS.items():
        for index, (initializer_name, state_suffix) in enumerate(zip(initializer_names, SPECIAL_STATE_SUFFIXES)):
            state_name = f"{branch}.{state_suffix}"
            value = initializers[initializer_name]
            if index in (0, 1, 3, 4):
                value = np.ascontiguousarray(value.T)
            elif index == 2:
                value = value.reshape(15, 16)
            recovered[state_name] = _tensor_for_state(value, template[state_name], state_name)
            consumed.add(initializer_name)

    missing_state = sorted(set(template) - set(recovered))
    if missing_state:
        raise ValueError(f"ONNX reconstruction left unmapped PyTorch state: {missing_state}")

    # torch.lerp with a constant joint mask was folded into these two graph
    # initializers.  They are derived from stable_weight, not independent state.
    stable_weight = recovered["stable_weight"].numpy()
    expected_where = stable_weight == 0.0
    expected_multiplier = 1.0 - stable_weight
    if not np.array_equal(initializers["onnx::Where_740"], expected_where):
        raise ValueError("onnx::Where_740 is not the expected stable_weight == 0 folded constant")
    if not np.array_equal(initializers["onnx::Mul_741"], expected_multiplier):
        raise ValueError("onnx::Mul_741 is not the expected 1 - stable_weight folded constant")
    consumed.update(("onnx::Where_740", "onnx::Mul_741"))

    unused = sorted(set(initializers) - consumed)
    if unused:
        raise ValueError(f"ONNX reconstruction did not account for every initializer: {unused}")
    if len(consumed) != EXPECTED_INITIALIZER_COUNT:
        raise ValueError(f"expected to account for {EXPECTED_INITIALIZER_COUNT} initializers, got {len(consumed)}")

    teacher.load_state_dict(recovered, strict=True)
    teacher.freeze()
    audit = {
        "state_tensor_count": len(recovered),
        "initializer_count": len(consumed),
        "direct_or_mapped_state_initializers": len(recovered),
        "derived_initializers": ["onnx::Where_740", "onnx::Mul_741"],
        "all_initializers_accounted": True,
    }
    return teacher, audit


def compare_initializer_payloads(model: onnx.ModelProto, reference_path: Path) -> dict[str, Any]:
    reference = onnx.load(reference_path, load_external_data=True)
    left = _initializers(model)
    right = _initializers(reference)
    if set(left) != set(right):
        raise ValueError("reference ONNX initializer names differ from the canonical dynamic graph")
    mismatched = [name for name in sorted(left) if not np.array_equal(left[name], right[name])]
    if mismatched:
        raise ValueError(f"reference ONNX contains different initializer values: {mismatched}")
    return {
        "path": str(reference_path.resolve()),
        "sha256": sha256_file(reference_path),
        "initializer_count": len(left),
        "bitwise_equal": True,
    }


def _realistic_random_observation(rng: np.random.Generator, batch: int) -> np.ndarray:
    observation = rng.normal(0.0, 0.35, size=(batch, INPUT_DIM)).astype(np.float32)
    observation[:, 1830:1860] = rng.uniform(0.01, 0.08, size=(batch, 30)).astype(np.float32)
    observation[:, 1860:1862] = 1.0
    observation[:, 1862:1864] = rng.uniform(0.0, 0.25, size=(batch, 2)).astype(np.float32)
    observation[:, [30, 33, 36, 39, 42]] = rng.uniform(0.65, 1.0, size=(batch, 5)).astype(np.float32)
    return observation


def load_recorded_observations(path: Path, key: str | None, limit: int) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        observation = loaded
    else:
        try:
            if key is None:
                candidates = [name for name in ("obs", "observation", "observations", "policy_obs") if name in loaded]
                if len(candidates) != 1:
                    raise ValueError(
                        f"could not choose an observation array from {path}; "
                        f"keys={list(loaded.keys())}, pass --recorded-key"
                    )
                key = candidates[0]
            observation = loaded[key]
        finally:
            loaded.close()
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim != 2 or observation.shape[1] != INPUT_DIM:
        raise ValueError(f"recorded observations must be [N,{INPUT_DIM}], got {observation.shape}")
    return np.ascontiguousarray(observation[:limit])


def verify_runtime_parity(
    teacher: FrozenSpeedBoostTeacher,
    onnx_path: Path,
    *,
    batch_sizes: Iterable[int] = (1, 7, 32),
    seed: int = 112,
    tolerance: float = 2.0e-5,
    recorded_observations: np.ndarray | None = None,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for full-graph parity verification") from error

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["obs"] or [item.name for item in session.get_outputs()] != [
        "actions"
    ]:
        raise ValueError("ONNX Runtime input/output names do not match obs -> actions")
    teacher = teacher.cpu().freeze()
    rng = np.random.default_rng(seed)
    cases: list[tuple[str, np.ndarray]] = [("zeros", np.zeros((1, INPUT_DIM), dtype=np.float32))]
    for batch in batch_sizes:
        if batch <= 0:
            raise ValueError(f"parity batch sizes must be positive, got {batch}")
        cases.append((f"random_batch_{batch}", _realistic_random_observation(rng, int(batch))))
    repeated_row = _realistic_random_observation(rng, 1)
    cases.append(("repeated_row_batch_8", np.repeat(repeated_row, 8, axis=0)))
    if recorded_observations is not None and len(recorded_observations):
        cases.append((f"recorded_batch_{len(recorded_observations)}", recorded_observations))

    case_reports: list[dict[str, Any]] = []
    global_max = 0.0
    with torch.no_grad():
        for name, observation in cases:
            ort_action = session.run(["actions"], {"obs": observation})[0]
            torch_action = teacher(torch.from_numpy(observation)).cpu().numpy()
            if ort_action.shape != (len(observation), OUTPUT_DIM) or torch_action.shape != ort_action.shape:
                raise ValueError(
                    f"parity case {name} returned invalid shapes ORT={ort_action.shape}, PyTorch={torch_action.shape}"
                )
            if not np.isfinite(ort_action).all() or not np.isfinite(torch_action).all():
                raise ValueError(f"parity case {name} produced non-finite actions")
            difference = np.abs(torch_action - ort_action)
            max_abs = float(difference.max(initial=0.0))
            mean_abs = float(difference.mean())
            global_max = max(global_max, max_abs)
            case_reports.append(
                {
                    "name": name,
                    "batch": len(observation),
                    "max_abs_error": max_abs,
                    "mean_abs_error": mean_abs,
                    "max_abs_action": float(np.abs(ort_action).max(initial=0.0)),
                }
            )
    if global_max > tolerance:
        raise ValueError(f"PyTorch reconstruction parity failed: max_abs_error={global_max} > tolerance={tolerance}")
    return {
        "provider": "CPUExecutionProvider",
        "tolerance": float(tolerance),
        "max_abs_error": global_max,
        "cases": case_reports,
        "passed": True,
    }


def convert_speedboost_teacher(
    onnx_path: Path,
    output_path: Path,
    *,
    parity_batch_sizes: Iterable[int] = (1, 7, 32),
    parity_seed: int = 112,
    parity_tolerance: float = 2.0e-5,
    recorded_observations: np.ndarray | None = None,
    recorded_observations_provenance: dict[str, Any] | None = None,
    reference_onnx: Path | None = None,
) -> dict[str, Any]:
    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    source_sha256 = sha256_file(onnx_path)
    model_proto = onnx.load(onnx_path, load_external_data=True)
    signature = validate_canonical_graph(model_proto, source_sha256)
    teacher, initializer_audit = recover_model(model_proto)
    parity = verify_runtime_parity(
        teacher,
        onnx_path,
        batch_sizes=parity_batch_sizes,
        seed=parity_seed,
        tolerance=parity_tolerance,
        recorded_observations=recorded_observations,
    )
    reference_report = compare_initializer_payloads(model_proto, reference_onnx) if reference_onnx else None
    provenance = {
        "converter": str(Path(__file__).resolve()),
        "source_onnx": str(onnx_path),
        "architecture_sources": [
            "/home/mosense/guo/scripts/train_shared_magnetic_policy.py",
            "/home/mosense/guo/scripts/export_confidence_gated_magnetic_policy.py",
            "/home/mosense/guo/scripts/export_high_friction_speed_boost_policy.py",
            "/home/mosense/guo/scripts/export_jointwise_magnetic_ensemble.py",
        ],
        "initializer_audit": initializer_audit,
    }
    if recorded_observations_provenance is not None:
        provenance["recorded_observations"] = dict(recorded_observations_provenance)
    if reference_report is not None:
        provenance["reference_initializer_comparison"] = reference_report
    save_frozen_speedboost_teacher(
        output_path,
        teacher,
        source_onnx_sha256=source_sha256,
        source_graph=signature,
        parity=parity,
        provenance=provenance,
    )
    report = {
        "output": str(output_path.resolve()),
        "source_sha256": source_sha256,
        "source_graph": signature,
        "config": teacher.config.__dict__,
        "initializer_audit": initializer_audit,
        "parity": parity,
        "reference_initializer_comparison": reference_report,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_batches(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batch list must contain positive comma-separated integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_dynamic.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "hall_speed_demo" / "speedboost112_frozen_teacher.pt",
    )
    parser.add_argument("--parity-batches", type=_parse_batches, default=(1, 7, 32))
    parser.add_argument("--parity-seed", type=int, default=112)
    parser.add_argument("--parity-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--recorded-observations", type=Path)
    parser.add_argument("--recorded-key")
    parser.add_argument("--recorded-limit", type=int, default=128)
    parser.add_argument(
        "--reference-onnx",
        type=Path,
        help="Optional fixed-batch export whose 82 initializers must be bitwise identical",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.parity_tolerance <= 0.0:
        raise ValueError("--parity-tolerance must be positive")
    if args.recorded_limit <= 0:
        raise ValueError("--recorded-limit must be positive")
    recorded = None
    recorded_provenance = None
    if args.recorded_observations is not None:
        recorded = load_recorded_observations(args.recorded_observations, args.recorded_key, args.recorded_limit)
        recorded_provenance = {
            "path": str(args.recorded_observations.resolve()),
            "sha256": sha256_file(args.recorded_observations),
            "requested_key": args.recorded_key,
            "loaded_shape": list(recorded.shape),
        }
    report = convert_speedboost_teacher(
        args.onnx,
        args.output,
        parity_batch_sizes=args.parity_batches,
        parity_seed=args.parity_seed,
        parity_tolerance=args.parity_tolerance,
        recorded_observations=recorded,
        recorded_observations_provenance=recorded_provenance,
        reference_onnx=args.reference_onnx,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
