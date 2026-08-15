"""CPU/AST contract tests for optional FastBase evaluator diagnostics."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"
UTIL_PATH = ROOT / "scripts" / "traction" / "spatial_friction_eval_utils.py"


def _load_utils():
    spec = importlib.util.spec_from_file_location(
        "spatial_friction_eval_utils_capture_test", UTIL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name}, found {len(matches)}"
    return matches[0]


def test_capture_summary_reports_stage_and_causal_gate_timing() -> None:
    module = _load_utils()
    # Flattened in the evaluator's [step, alive-env] order.  Environment zero
    # activates on its second LOW frame and releases after three stable
    # HIGH_END frames.  Environment one never activates/releases.
    env_id = []
    rollout_step = []
    stage = []
    probability = []
    effective = []
    delta = []
    gates = {
        0: [0.1, 0.2, 0.8, 0.7, 0.2, 0.08, 0.07, 0.06],
        1: [0.1, 0.2, 0.3, 0.3, 0.4, 0.3, 0.2, 0.2],
    }
    stages = [0, 1, 1, 1, 2, 2, 2, 2]
    for step in range(8):
        for current_env in range(2):
            env_id.append(current_env)
            rollout_step.append(step)
            stage.append(stages[step])
            probability.append(gates[current_env][step] + 0.05)
            effective.append(gates[current_env][step])
            delta.append(2.0 * gates[current_env][step])

    report = module.summarize_fastbase_capture_diagnostics(
        raw_capture_probability=probability,
        capture_probability=probability,
        effective_gate=effective,
        delta_l2=delta,
        course_stage=stage,
        rollout_step=rollout_step,
        env_id=env_id,
        step_dt_s=0.1,
    )

    assert report["definition"] == "fastbase-capture-observation-only-diagnostics-v2"
    assert report["sample_count"] == 16
    assert report["stage_encoding"] == {"HIGH_START": 0, "LOW": 1, "HIGH_END": 2}
    assert report["by_stage"]["HIGH_START"]["samples"] == 2
    assert report["by_stage"]["LOW"]["samples"] == 6
    assert report["by_stage"]["HIGH_END"]["samples"] == 8
    assert report["by_stage"]["HIGH_START"]["effective_gate"]["p95"] is not None
    assert report["by_stage"]["LOW"]["raw_capture_probability"]["median"] is not None
    assert 0.0 <= report["low_vs_high_auc"]["raw_capture_probability"] <= 1.0
    assert 0.0 <= report["low_vs_high_auc"]["effective_gate"] <= 1.0
    assert report["low_activation"]["entered_envs"] == 2
    assert report["low_activation"]["activated_envs"] == 1
    assert report["low_activation"]["latency_s"]["mean"] == pytest.approx(0.1)
    assert report["high_end_release"]["entered_envs"] == 2
    assert report["high_end_release"]["released_envs"] == 1
    assert report["high_end_release"]["latency_s"]["mean"] == pytest.approx(0.1)


def test_capture_summary_rejects_misaligned_or_nonfinite_diagnostics() -> None:
    module = _load_utils()
    common = dict(
        capture_probability=[0.2],
        effective_gate=[0.1],
        delta_l2=[0.3],
        course_stage=[0],
        rollout_step=[0],
        env_id=[0],
        step_dt_s=0.02,
    )
    with pytest.raises(ValueError, match="not aligned"):
        module.summarize_fastbase_capture_diagnostics(
            **{**common, "effective_gate": [0.1, 0.2]}
        )
    with pytest.raises(ValueError, match="finite"):
        module.summarize_fastbase_capture_diagnostics(
            **{**common, "delta_l2": [float("nan")]}
        )
    with pytest.raises(ValueError, match="course stage"):
        module.summarize_fastbase_capture_diagnostics(
            **{**common, "course_stage": [3]}
        )


def test_capture_summary_keeps_sustained_high_end_release_window() -> None:
    module = _load_utils()
    # HIGH_END begins at step 2, while the actual episode terminal would occur
    # after step 11.  All ten post-entry frames must remain available; causal
    # H-L-H completion at step 2 is not an episode terminal.
    stage = [0, 1] + [2] * 10
    gate = [0.1, 0.8, 0.7, 0.5, 0.3, 0.2, 0.09, 0.08, 0.07, 0.04, 0.03, 0.02]
    report = module.summarize_fastbase_capture_diagnostics(
        capture_probability=gate,
        effective_gate=gate,
        delta_l2=[2.0 * value for value in gate],
        course_stage=stage,
        rollout_step=list(range(len(stage))),
        env_id=[0] * len(stage),
        step_dt_s=0.1,
    )
    assert report["by_stage"]["HIGH_END"]["samples"] == 10
    assert report["high_end_release"]["released_envs"] == 1
    assert report["high_end_release"]["latency_s"]["mean"] == pytest.approx(0.4)


def test_evaluator_capture_reader_has_no_privileged_actor_inputs() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    method = _function(tree, "capture_diagnostics")
    arguments = [argument.arg for argument in method.args.args]
    assert arguments == ["self", "observation"]
    loaded_names = {
        node.id
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for forbidden in ("stage", "local_x", "low", "contact_patch", "friction", "ground_mu"):
        assert forbidden not in loaded_names

    rollout = _function(tree, "_run_rollout")
    reads = [
        node
        for node in ast.walk(rollout)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_read_fastbase_capture_diagnostics"
    ]
    assert len(reads) == 1
    assert [argument.id for argument in reads[0].args if isinstance(argument, ast.Name)] == [
        "policy",
        "observation",
    ]
    assert not reads[0].keywords


def test_fastbase_dataset_fields_are_optional_and_legacy_keys_remain() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    for key in (
        "fastbase_raw_capture_probability",
        "fastbase_capture_probability",
        "fastbase_effective_gate",
        "fastbase_capture_delta_l2",
        "fastbase_stability_authority",
        "fastbase_stability_delta_l2",
        "fastbase_stability_delta_abs_max",
        "fastbase_course_stage",
        "fastbase_rollout_step",
        "fastbase_time_s",
        "fastbase_env_id",
        'report["fastbase_capture_diagnostics"]',
        'report["fastbase_stability_residual_diagnostics"]',
    ):
        assert key in source
    assert '"observation": trace["dataset_observation"]' in source
    assert '"action": trace["dataset_action"]' in source
    assert '"low": trace["dataset_low"]' in source
    assert "if name in trace:" in source
    assert "dataset_payload[name] = trace[name]" in source


def test_stability_reader_is_bounded_observation_only_and_retention_geometry_is_explicit() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    method = ast.unparse(
        _function(ast.parse(source), "capture_diagnostics")
    )
    assert "stability_authority" in method
    assert "stability_delta" in method
    assert "stability_limit" in method
    assert "course_stage" not in method
    assert "contact_patch" not in method
    assert "ground_mu" not in method
    assert '"CadenceStrideRetention" in args_cli.task' in source
    assert '"low_end_x_m": 2.0' in source
    assert '"success_x_m": 9.5' in source


def test_hlh_completion_does_not_truncate_the_first_episode_mask() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    assert "alive_before = ~(fallen | completed)" not in source
    assert "first_episode_active = torch.ones(" in source
    assert "first_episode_active_before = first_episode_active.clone()" in source
    assert "first_episode_terminal = dones.bool() & first_episode_active_before" in source
    assert "first_episode_active &= ~first_episode_terminal" in source
    assert (
        'base_env.termination_manager.get_term("course_success").bool()\n'
        "            & first_episode_active_before"
    ) in source
    # Capture, dataset, region-speed and state-machine accounting all use the
    # terminal mask, while ``completed`` remains the legacy H-L-H definition.
    assert source.count("first_episode_active_before") >= 10
    assert "completed |= course_success" in source


def test_eval_disables_training_only_gate_warmup_before_runner_construction() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    helper = _function(tree, "_disable_eval_capture_gate_warmup")
    helper_source = ast.unparse(helper)
    assert "capture_gate_warmup_updates" in helper_source
    assert "algorithm_cfg.capture_gate_warmup_updates = 0" in helper_source

    constructor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
        and any(
            isinstance(parent, ast.ClassDef) and parent.name == "_RslPolicy"
            for parent in tree.body
            if isinstance(parent, ast.ClassDef) and node in parent.body
        )
    )
    calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_disable_eval_capture_gate_warmup"
    ]
    assert len(calls) == 1
    disable_line = calls[0].lineno
    runner_lines = [
        node.lineno
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "runner_class"
    ]
    assert len(runner_lines) == 1
    assert disable_line < runner_lines[0]


def test_eval_checkpoint_load_is_actor_only_without_training_state() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EVAL_ACTOR_ONLY_LOAD_CFG"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
    }

    constructor = next(
        node
        for parent in tree.body
        if isinstance(parent, ast.ClassDef) and parent.name == "_RslPolicy"
        for node in parent.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    loads = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runner"
    ]
    assert len(loads) == 1
    keywords = {keyword.arg: keyword.value for keyword in loads[0].keywords}
    assert set(keywords) == {"load_cfg", "strict"}
    assert isinstance(keywords["strict"], ast.Constant)
    assert keywords["strict"].value is True
    load_cfg = keywords["load_cfg"]
    assert isinstance(load_cfg, ast.Call) and isinstance(load_cfg.func, ast.Name)
    assert load_cfg.func.id == "dict"
    assert isinstance(load_cfg.args[0], ast.Name)
    assert load_cfg.args[0].id == "EVAL_ACTOR_ONLY_LOAD_CFG"
