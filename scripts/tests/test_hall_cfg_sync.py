#!/usr/bin/env python3
"""Pure-Python regression tests for Hall policy configuration synchronization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/sensors/hall_sensor_config.py"
)
EVALUATOR = ROOT / "scripts/rsl_rl/eval_hall_handoff_impulse.py"
ENV_CFG = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_foot_env_cfg.py"
)
FOOT_SENSOR = (
    ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/foot_sensor.py"
)

spec = importlib.util.spec_from_file_location("hall_sensor_config", CONFIG)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _observations(*, motion_tail: bool = False):
    terms = {}
    for name in module.HALL_POLICY_OBSERVATION_TERM_NAMES:
        if motion_tail and name == "foot_sensor_age_lr":
            terms[name] = SimpleNamespace(params={"age_scale": 0.25})
        else:
            terms[name] = SimpleNamespace(params={"hall_cfg": module.HallFootSensorCfg()})
    return SimpleNamespace(policy=SimpleNamespace(**terms))


def test_sync_copies_all_four_standard_hall_terms_without_aliases() -> None:
    observations = _observations()
    cfg = module.HallFootSensorCfg(
        enable_domain_randomization=True,
        foot_dropout_probability=0.10,
        dead_channel_probability=0.08,
        maximum_packet_delay_steps=5,
    )
    names = module.sync_hall_sensor_cfg_to_policy_terms(observations, cfg)
    assert names == module.HALL_POLICY_OBSERVATION_TERM_NAMES

    copies = [getattr(observations.policy, name).params["hall_cfg"] for name in names]
    assert all(item == cfg and item is not cfg for item in copies)
    assert len({id(item) for item in copies}) == 4

    cfg.foot_dropout_probability = 0.33
    assert all(item.foot_dropout_probability == 0.10 for item in copies)


def test_motion_tail_syncs_only_terms_that_really_accept_hall_cfg() -> None:
    observations = _observations(motion_tail=True)
    cfg = module.HallFootSensorCfg(enable_domain_randomization=True)
    names = module.sync_hall_sensor_cfg_to_policy_terms(observations, cfg)
    assert names == module.HALL_POLICY_OBSERVATION_TERM_NAMES[:3]
    assert observations.policy.foot_sensor_age_lr.params == {"age_scale": 0.25}


def test_nominal_override_is_copied_into_every_effective_term() -> None:
    observations = _observations()
    cfg = module.HallFootSensorCfg(enable_domain_randomization=True)
    module.sync_hall_sensor_cfg_to_policy_terms(observations, cfg)
    cfg.enable_domain_randomization = False
    module.sync_hall_sensor_cfg_to_policy_terms(observations, cfg)
    module.audit_hall_sensor_cfg_policy_terms(observations, cfg)
    assert all(
        not getattr(observations.policy, name).params["hall_cfg"].enable_domain_randomization
        for name in module.HALL_POLICY_OBSERVATION_TERM_NAMES
    )


def test_stage7_and_evaluator_call_central_sync_after_overrides() -> None:
    env_source = ENV_CFG.read_text(encoding="utf-8")
    hardening = env_source.split(
        "class RobotFootTractionMagneticMotionSwitchFaultHardeningEnvCfg", 1
    )[1].split("@configclass", 1)[0]
    assert hardening.index("foot_dropout_probability = 0.10") < hardening.index(
        "sync_hall_sensor_cfg_to_policy_terms"
    )

    evaluator = EVALUATOR.read_text(encoding="utf-8")
    main = evaluator.split("def main()", 1)[1]
    assert main.index("env_cfg.hall_sensor_cfg.enable_domain_randomization = False") < main.index(
        "sync_hall_sensor_cfg_to_policy_terms"
    )
    assert '"effective_hall_cfg": hall_cfg_audit' in evaluator


def test_hall_randomization_uses_effective_environment_seed() -> None:
    source = FOOT_SENSOR.read_text(encoding="utf-8")
    packet = source.split("def _hall_foot_packet(", 1)[1].split(
        "def hall_magnetic_array", 1
    )[0]
    assert "20260807" not in packet
    assert 'getattr(getattr(env, "cfg", None), "seed", None)' in packet
    assert "hall_seed = operator.index(configured_seed)" in packet
    assert "seed=hall_seed" in packet
    assert "env._hall_foot_sensor_seed = hall_seed" in packet
