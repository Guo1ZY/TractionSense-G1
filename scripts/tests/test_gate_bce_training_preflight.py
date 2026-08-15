from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unitree_rl_lab.traction.anchored_ppo import (
    CAPTURE_GATE_OPTIMIZER_ROLE,
    CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
    OPTIMIZER_ROLE_KEY,
    PPO_OPTIMIZER_ROLE,
    STRICT_ACTOR_CRITIC_RESUME_CFG,
    TRAINING_PROVENANCE_FORMAT,
    TRAINING_PROVENANCE_FORMAT_VERSION,
    AnchoredOnPolicyRunner,
    checkpoint_sha256,
    validate_bounded_new_updates,
    validate_fail_closed_gate_training_start,
    validate_hall_randomization_seed,
)


ROOT = Path(__file__).resolve().parents[2]
CLI_ARGS_PATH = ROOT / "scripts" / "rsl_rl" / "cli_args.py"
RUNNER_CFG_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "tasks"
    / "locomotion"
    / "agents"
    / "rsl_rl_ppo_cfg.py"
)
MODEL49 = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "unitree_g1_29dof_velocity_foot_traction_hall_spatial_fastbase_capture"
    / "2026-08-10_20-02-07_fastbase_gate_warmup_medium_r3"
    / "model_49.pt"
)
MODEL49_SHA256 = "beb4574037f5ab342cca01d67cfdca9b802d15bcdc85fa6d0d019d57d43f4955"


def _load_cli_args_module():
    spec = importlib.util.spec_from_file_location("gate_bce_cli_args", CLI_ARGS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli(**overrides):
    values = {
        "resume": False,
        "resume_checkpoint": None,
        "partial_checkpoint": None,
        "partial_checkpoint_critic_only": False,
        "load_optimizer": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_checkpoint_cli_modes_are_mutually_exclusive_and_optimizer_is_strict_only() -> None:
    cli_args = _load_cli_args_module()
    assert cli_args.validate_checkpoint_load_args(_cli()) == "fresh"
    assert (
        cli_args.validate_checkpoint_load_args(
            _cli(resume_checkpoint="model_49.pt")
        )
        == "strict_resume"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli_args.validate_checkpoint_load_args(
            _cli(resume=True, resume_checkpoint="model_49.pt")
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli_args.validate_checkpoint_load_args(
            _cli(
                resume_checkpoint="model_49.pt",
                partial_checkpoint="baseline.pt",
            )
        )
    with pytest.raises(ValueError, match="valid only"):
        cli_args.validate_checkpoint_load_args(_cli(load_optimizer=True))
    with pytest.raises(ValueError, match="valid only"):
        cli_args.validate_checkpoint_load_args(
            _cli(resume=True, load_optimizer=True)
        )
    assert (
        cli_args.validate_checkpoint_load_args(
            _cli(resume_checkpoint="model_49.pt", load_optimizer=True)
        )
        == "strict_resume"
    )


def test_released_model49_identity_and_saved_phase_are_exact() -> None:
    assert checkpoint_sha256(MODEL49) == MODEL49_SHA256
    payload = torch.load(MODEL49, weights_only=False, map_location="cpu")
    assert payload["iter"] == 49
    assert payload["capture_gate_warmup"] == {
        "configured_updates": 50,
        "completed_updates": 50,
        "active": False,
        "warmup_learning_rate": 1.0e-4,
        "released_learning_rate": 1.0e-5,
        "current_learning_rate": 1.0e-5,
        "max_grad_norm": 1.0,
    }


class _LoadOnlyAlgorithm:
    def __init__(self) -> None:
        self.calls = []

    def load(self, loaded_dict, load_cfg, strict):
        self.calls.append((loaded_dict, load_cfg, strict))
        return bool(load_cfg["iteration"])


def test_completed_model49_resume_labels_two_and_twelve_update_runs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_49.pt"
    torch.save(
        {
            "iter": 49,
            "infos": {"source": "test"},
            "capture_gate_warmup": {
                "configured_updates": 50,
                "completed_updates": 50,
            },
        },
        checkpoint,
    )
    runner = SimpleNamespace(
        alg=_LoadOnlyAlgorithm(),
        current_learning_iteration=0,
        _checkpoint_load_audit=None,
        _last_completed_iteration=None,
    )
    infos = AnchoredOnPolicyRunner.load(
        runner,
        str(checkpoint),
        load_cfg=dict(STRICT_ACTOR_CRITIC_RESUME_CFG),
        strict=True,
        map_location="cpu",
    )
    assert infos == {"source": "test"}
    assert runner.current_learning_iteration == 50
    assert runner._last_completed_iteration == 49
    completed_iterations = list(
        range(runner.current_learning_iteration, runner.current_learning_iteration + 12)
    )
    assert completed_iterations == list(range(50, 62))
    assert completed_iterations[-1] == 61
    assert 50 + len(completed_iterations) == 62
    smoke_iterations = list(
        range(runner.current_learning_iteration, runner.current_learning_iteration + 2)
    )
    assert smoke_iterations == [50, 51]
    assert smoke_iterations[-1] == 51
    assert 50 + len(smoke_iterations) == 52


def test_gate_bce_update_budget_allows_smoke_and_default_but_not_long_run() -> None:
    assert validate_bounded_new_updates(2, 12) == 2
    assert validate_bounded_new_updates(12, 12) == 12
    with pytest.raises(RuntimeError, match="1 <= updates <= 12"):
        validate_bounded_new_updates(0, 12)
    with pytest.raises(RuntimeError, match="1 <= updates <= 12"):
        validate_bounded_new_updates(100, 12)


def test_gate_bce_runner_explicitly_separates_actor_and_critic_observation_groups() -> None:
    """Do not rely on RSL-RL's deprecated observation-group inference."""

    source = RUNNER_CFG_PATH.read_text(encoding="utf-8")
    class_start = source.index(
        "class FootTractionHallSpatialCalibratedFastBaseExpertGateBceOnlyPPORunnerCfg"
    )
    class_end = source.index("\n\n@configclass", class_start)
    class_source = source[class_start:class_end]
    assert 'obs_groups = {"actor": ["policy"], "critic": ["critic"]}' in class_source


def test_hall_randomization_seed_matches_effective_env_seed_and_lazy_init() -> None:
    environment = SimpleNamespace(_hall_foot_sensor_seed=443)
    assert validate_hall_randomization_seed(environment, 443) == {
        "environment_seed": 443,
        "hall_foot_sensor_seed": 443,
        "match": True,
        "initialized_by_read_only_observation": False,
    }
    with pytest.raises(RuntimeError, match="seed mismatch"):
        validate_hall_randomization_seed(environment, 444)

    lazy_environment = SimpleNamespace()
    calls = []

    def initialize_from_observation():
        calls.append(True)
        lazy_environment._hall_foot_sensor_seed = 445

    audit = validate_hall_randomization_seed(
        lazy_environment,
        445,
        observation_reader=initialize_from_observation,
    )
    assert calls == [True]
    assert audit["initialized_by_read_only_observation"] is True


def _valid_runner():
    ppo_parameter = torch.nn.Parameter(torch.tensor(0.0))
    gate_parameter = torch.nn.Parameter(torch.tensor(0.0))
    residual = torch.nn.Sequential(torch.nn.Linear(3, 29))
    residual_parameters = list(residual.parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": [ppo_parameter],
                "lr": 5.0e-6,
                OPTIMIZER_ROLE_KEY: PPO_OPTIMIZER_ROLE,
            },
            {
                "params": [gate_parameter],
                "lr": 1.0e-5,
                OPTIMIZER_ROLE_KEY: CAPTURE_GATE_OPTIMIZER_ROLE,
            },
            {
                "params": residual_parameters,
                "lr": 1.0e-4,
                OPTIMIZER_ROLE_KEY: CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
            },
        ]
    )
    actor = SimpleNamespace(
        obs_dim=1864,
        obs_groups=["policy"],
        distribution=SimpleNamespace(output_dim=29),
        mlp=SimpleNamespace(
            gate_logit_scale=torch.tensor(2.75),
            gate_logit_bias=torch.tensor(-3.2),
            loaded_legacy_calibration=True,
        ),
    )
    algorithm = SimpleNamespace(
        actor=actor,
        critic=SimpleNamespace(obs_dim=570, obs_groups=["critic"]),
        optimizer=optimizer,
        capture_gate_updates_completed=50,
        capture_gate_warmup_updates=50,
        capture_gate_warmup_active=False,
        capture_gate_gradient_mode="stage_bce_only",
        low_expert_residual_gradient_mode="supervised_only",
        capture_residual=residual,
        capture_residual_parameters=residual_parameters,
        capture_residual_current_learning_rate=1.0e-4,
    )
    return SimpleNamespace(
        alg=algorithm,
        env=SimpleNamespace(num_actions=29),
        current_learning_iteration=50,
        checkpoint_load_audit={
            "strict": True,
            "load_cfg": dict(STRICT_ACTOR_CRITIC_RESUME_CFG),
            "saved_completed_iteration": 49,
            "next_learning_iteration": 50,
            "capture_gate_updates_completed": 50,
        },
    )


def _validate(runner, **overrides):
    values = {
        "load_mode": "strict_resume",
        "source_checkpoint_sha256": MODEL49_SHA256,
        "required_checkpoint_sha256": MODEL49_SHA256,
        "required_checkpoint_iteration": 49,
        "required_capture_gate_completed_updates": 50,
        "required_actor_observation_dim": 1864,
        "required_critic_observation_dim": 570,
        "required_action_dim": 29,
        "required_gate_logit_scale": 2.75,
        "required_gate_logit_bias": -3.2,
    }
    values.update(overrides)
    return validate_fail_closed_gate_training_start(runner, **values)


def test_gate_bce_start_audit_accepts_only_released_trainable_residual() -> None:
    runner = _valid_runner()
    audit = _validate(runner)
    assert audit["dimensions"] == {
        "actor_observation": 1864,
        "critic_observation": 570,
        "action": 29,
    }
    assert audit["capture_gate_updates_completed"] == 50
    assert audit["capture_residual_learning_rate"] == pytest.approx(1.0e-4)
    assert [entry["role"] for entry in audit["optimizer_roles"]] == [
        PPO_OPTIMIZER_ROLE,
        CAPTURE_GATE_OPTIMIZER_ROLE,
        CAPTURE_RESIDUAL_OPTIMIZER_ROLE,
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runner: setattr(runner, "current_learning_iteration", 49), "resume iteration"),
        (lambda runner: setattr(runner.alg.actor, "obs_dim", 1863), "actor observation"),
        (lambda runner: setattr(runner.alg.critic, "obs_dim", 569), "critic observation"),
        (lambda runner: setattr(runner.env, "num_actions", 28), "action dimension"),
        (
            lambda runner: setattr(runner.alg, "capture_gate_updates_completed", 49),
            "not released",
        ),
        (lambda runner: setattr(runner.alg, "capture_gate_warmup_active", True), "still active"),
        (
            lambda runner: setattr(
                runner.alg, "capture_residual_current_learning_rate", 0.0
            ),
            "learning rate must be positive",
        ),
        (
            lambda runner: runner.alg.capture_residual_parameters[0].requires_grad_(False),
            "not all trainable",
        ),
        (
            lambda runner: setattr(
                runner.alg.actor.mlp, "gate_logit_bias", torch.tensor(-2.0)
            ),
            "calibration mismatch",
        ),
    ],
)
def test_gate_bce_start_audit_rejects_low_level_wiring_errors(mutation, message) -> None:
    runner = _valid_runner()
    mutation(runner)
    with pytest.raises(RuntimeError, match=message):
        _validate(runner)


def test_gate_bce_start_audit_rejects_wrong_source_and_optimizer_load() -> None:
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _validate(_valid_runner(), source_checkpoint_sha256="0" * 64)
    runner = _valid_runner()
    runner.checkpoint_load_audit["load_cfg"]["optimizer"] = True
    with pytest.raises(RuntimeError, match="optimizer=false"):
        _validate(runner)


class _SaveOnlyAlgorithm:
    capture_gate_updates_completed = 62

    def save(self):
        return {"actor_state_dict": {"weight": torch.ones(1)}}

    def anchor_manifest(self):
        return {"format": "test-anchor"}


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls = []

    def save_model(self, path, iteration):
        self.calls.append((path, iteration))


def test_training_provenance_is_atomic_and_embedded_in_checkpoint_and_manifest(
    tmp_path: Path,
) -> None:
    runner = AnchoredOnPolicyRunner.__new__(AnchoredOnPolicyRunner)
    runner.alg = _SaveOnlyAlgorithm()
    runner.logger = _RecordingLogger()
    runner.current_learning_iteration = 61
    runner._checkpoint_load_audit = None
    runner._training_provenance = None
    runner._last_completed_iteration = 49
    runner._training_provenance_path = tmp_path / "params" / "training_provenance.json"
    runner._anchor_manifest_path = tmp_path / "params" / "anchor.json"
    provenance = {
        "format": TRAINING_PROVENANCE_FORMAT,
        "format_version": TRAINING_PROVENANCE_FORMAT_VERSION,
        "task": "GateBceOnly",
        "training_schedule": {
            "requested_update_count": 12,
            "first_iteration": 50,
            "expected_last_completed_iteration": 61,
            "expected_final_capture_gate_updates_completed": 62,
        },
    }
    runner.attach_training_provenance(provenance)
    assert json.loads(runner._training_provenance_path.read_text()) == provenance
    checkpoint = tmp_path / "model_61.pt"
    runner.save(str(checkpoint), infos={"existing": True})
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    embedded = saved["infos"]["training_provenance"]
    assert saved["iter"] == 61
    assert saved["infos"]["existing"] is True
    assert embedded["checkpoint_state"] == {
        "completed_iteration": 61,
        "capture_gate_updates_completed": 62,
    }
    manifest = json.loads(runner._anchor_manifest_path.read_text())
    assert manifest["training_provenance"] == embedded
    assert runner.logger.calls == [(str(checkpoint), 61)]
