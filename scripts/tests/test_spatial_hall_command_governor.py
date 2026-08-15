"""CPU/AST contracts for the strict single-PT spatial Hall governor path."""

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
        "spatial_friction_eval_utils_hall_governor_test", UTIL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _motion_payload(**updates):
    payload = {
        "input_dim": 1864,
        "trailing_feature_mode": "motion_feedback",
        "measurement_boundary": (
            "runtime input is Hall Bx/By/Bz history + proprioception only; "
            "contact slip/falls are offline simulator labels, not inputs"
        ),
        "risk_target": "prospective contact-point slip/fall",
        "model_variant": "slip_aware_invariant",
        "model": {"weight": object()},
    }
    payload.update(updates)
    return payload


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name}, got {len(matches)}"
    return matches[0]


def test_motion_risk_metadata_accepts_explicit_contract_and_rejects_old_060() -> None:
    module = _load_utils()
    report = module.validate_motion_hall_risk_metadata(_motion_payload())
    assert report["input_dim"] == 1864
    assert report["trailing_feature_mode"] == "motion_feedback"
    assert report["risk_target"] == "prospective contact-point slip/fall"

    with pytest.raises(ValueError, match="sensor_age artifacts are incompatible"):
        module.validate_motion_hall_risk_metadata(
            _motion_payload(trailing_feature_mode="sensor_age")
        )
    with pytest.raises(ValueError, match="exactly 1864"):
        module.validate_motion_hall_risk_metadata(_motion_payload(input_dim=1862))
    with pytest.raises(ValueError, match="measurement boundary"):
        module.validate_motion_hall_risk_metadata(
            _motion_payload(measurement_boundary="uses hidden simulator labels")
        )


def test_governor_summary_reports_internal_hlh_and_command_latencies() -> None:
    module = _load_utils()
    states = [0, 2, 2, 1, 1, 1, 2, 2, 2]
    count = len(states)
    report = module.summarize_hall_command_governor_trace(
        risk_probability=[0.2, 0.2, 0.2, 0.9, 0.9, 0.7, 0.2, 0.2, 0.2],
        filtered_probability=[0.2, 0.2, 0.2, 0.7, 0.8, 0.75, 0.4, 0.3, 0.2],
        state=states,
        requested_vx=[0.8] * count,
        upstream_vx=[0.8] * count,
        effective_vx=[0.0, 0.1, 0.5, 0.5, 0.3, 0.1, 0.1, 0.3, 0.72],
        valid=[True] * count,
        probing=[False, True, False, False, False, False, False, False, False],
        prebrake=[False] * count,
        rollout_step=list(range(count)),
        env_id=[0] * count,
        step_dt_s=0.1,
        low_speed_limit_m_s=0.1,
        high_speed_limit_m_s=0.9,
    )
    assert report["definition"] == "hall-only-command-governor-response-v1"
    assert report["completed_internal_hlh_envs"] == 1
    assert report["per_env"][0]["compressed_state_sequence"] == [0, 2, 1, 2]
    assert report["low_state_to_command_limit_s"]["mean"] == pytest.approx(0.2)
    assert report["low_state_to_recovered_high_state_s"]["mean"] == pytest.approx(0.3)
    assert report["recovered_high_state_to_command_s"]["mean"] == pytest.approx(0.2)
    assert "no friction/contact/force/course-stage truth" in report["input_contract"]


def test_strict_rollout_order_is_risk_health_governor_apply_rewrite_actor() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    rollout = _function(tree, "_run_rollout")
    calls = list(ast.walk(rollout))

    def line_for(attribute: str, owner: str | None = None) -> int:
        matches = []
        for node in calls:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == attribute:
                if owner is None or (
                    isinstance(func.value, ast.Name) and func.value.id == owner
                ):
                    matches.append(node.lineno)
            elif owner is None and isinstance(func, ast.Name) and func.id == attribute:
                matches.append(node.lineno)
        assert matches, f"missing call {owner}.{attribute}"
        return min(matches)

    predict = line_for("predict", "hall_command_governor")
    health = line_for("update", "health_envelope")
    governor = line_for("update", "hall_command_governor")
    apply_command = line_for("_apply_effective_command")
    rewrite = line_for("_rewrite_actor_command_history")
    actor = line_for("policy")
    assert predict < health < governor < apply_command < rewrite < actor

    rollout_source = ast.unparse(rollout)
    assert "raw_governor_observation = _policy_tensor(observation)" in rollout_source
    assert "health_output.foot_healthy.all(dim=1)" in rollout_source
    assert "for command_index in RECOVERY_COMMAND_VX_INDICES" in rollout_source
    assert "synchronized[:, command_index:command_index + 3]" in rollout_source


def test_risk_predictor_has_no_privileged_runtime_arguments() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    predictor = _function(tree, "predict")
    assert [argument.arg for argument in predictor.args.args] == [
        "self",
        "raw_policy_observation",
    ]
    loaded_names = {
        node.id
        for node in ast.walk(predictor)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for forbidden in (
        "mu",
        "contact",
        "force",
        "slip",
        "stage",
        "local_x",
        "spatial_low_contact_buf",
    ):
        assert forbidden not in loaded_names


def test_governor_cli_is_independent_default_off_and_trace_is_complete() -> None:
    source = EVAL_PATH.read_text(encoding="utf-8")
    assert '"--hall_command_governor"' in source
    assert '"--hall_command_risk_checkpoint"' in source
    assert "action=\"store_true\"" in source
    assert "--hall_command_governor requires --checkpoint" in source
    assert "be combined with ONNX/TorchScript or the three-actor hybrid" in source
    assert "diagnostic_only_not_actor_specific_acceptance" in source
    for field in (
        "hall_governor_risk_probability",
        "hall_governor_filtered_probability",
        "hall_governor_state",
        "hall_governor_requested_command",
        "hall_governor_health_bounded_command",
        "hall_governor_effective_command",
        "hall_governor_valid",
        "hall_governor_rollout_step",
        "hall_governor_time_s",
        "hall_governor_env_id",
    ):
        assert field in source


def test_governor_is_reset_for_managed_environment_terminations() -> None:
    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    rollout_source = ast.unparse(_function(tree, "_run_rollout"))
    assert "hall_command_governor.reset(torch.nonzero(dones.bool()" in rollout_source
