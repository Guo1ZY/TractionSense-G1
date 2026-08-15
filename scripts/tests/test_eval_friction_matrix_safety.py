#!/usr/bin/env python3
"""Source-level safety invariants for the Isaac friction matrix evaluator."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/rsl_rl/eval_friction_matrix.py"
SOURCE = ROOT / "source" / "unitree_rl_lab"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from unitree_rl_lab.traction.contact_slip import (  # noqa: E402
    CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
    CONTACT_POINT_TANGENTIAL_SLIP_KEY,
    CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
    CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
    LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
    static_ground_contact_point_tangential_speed,
)


def _load_pure_helper(name: str):
    """Load one Isaac-independent helper without importing/launching Isaac."""

    tree = ast.parse(EVALUATOR.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "np": np,
        "CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA": (
            CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
        ),
    }
    exec(compile(module, str(EVALUATOR), "exec"), namespace)
    return namespace[name]


def _load_collection_metadata_helper():
    """Load the pickle-free metadata builder without importing Isaac Sim."""

    tree = ast.parse(EVALUATOR.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_collection_metadata"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "os": os,
        "CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA": (
            CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
        ),
        "CONTACT_POINT_TANGENTIAL_SLIP_KEY": (
            CONTACT_POINT_TANGENTIAL_SLIP_KEY
        ),
        "CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY": (
            CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY
        ),
        "CONTACT_POINT_TANGENTIAL_SLIP_FORMULA": (
            CONTACT_POINT_TANGENTIAL_SLIP_FORMULA
        ),
        "LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY": (
            LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY
        ),
    }
    exec(compile(module, str(EVALUATOR), "exec"), namespace)
    return namespace["_collection_metadata"]


def _dedicated_pair(
    positions: torch.Tensor, forces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return positions.reshape(positions.shape[0], 1, -1, 3), forces.reshape(
        forces.shape[0], 1, -1, 3
    )


def test_contact_point_slip_translation_and_rotation_formula_cpu() -> None:
    body_pos = torch.zeros((2, 2, 3), dtype=torch.float64)
    body_lin = torch.zeros_like(body_pos)
    body_ang = torch.zeros_like(body_pos)
    # env 0: pure COM translation; vertical velocity is projected out.
    body_lin[0, 0] = torch.tensor((1.0, 2.0, 3.0))
    # env 1: omega x r cancels COM motion exactly at r=(1,0,0).
    body_lin[1, 0] = torch.tensor((0.0, -2.0, 0.0))
    body_ang[1, 0] = torch.tensor((0.0, 0.0, 2.0))
    left_pos, left_force = _dedicated_pair(
        torch.tensor(((0.2, 0.0, -0.1), (1.0, 0.0, 0.0))),
        torch.tensor(((0.0, 0.0, 20.0), (0.0, 0.0, 20.0))),
    )
    right_pos = torch.full((2, 1, 1, 3), torch.nan, dtype=torch.float64)
    right_force = torch.zeros_like(right_pos)

    result = static_ground_contact_point_tangential_speed(
        body_pos,
        body_lin,
        body_ang,
        (left_pos.to(torch.float64), right_pos),
        (left_force.to(torch.float64), right_force),
    )

    torch.testing.assert_close(
        result.speed_per_env,
        torch.tensor((5.0**0.5, 0.0), dtype=torch.float64),
    )
    assert torch.equal(result.valid_per_env, torch.tensor((True, True)))
    assert torch.equal(
        result.valid_per_foot,
        torch.tensor(((True, False), (True, False))),
    )


def test_contact_point_slip_multifilter_force_weighting_and_surface_tangent() -> None:
    body_pos = torch.zeros((1, 2, 3), dtype=torch.float64)
    body_lin = torch.zeros_like(body_pos)
    body_ang = torch.zeros_like(body_pos)
    body_ang[0, 0, 2] = 1.0
    body_lin[0, 1, 0] = 3.0
    # Left filter speeds are 1 and 2 m/s with loads 10 and 30 N -> 1.75.
    left = _dedicated_pair(
        torch.tensor(
            (((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),), dtype=torch.float64
        ),
        torch.tensor(
            (((0.0, 0.0, 10.0), (0.0, 0.0, 30.0)),), dtype=torch.float64
        ),
    )
    # Right has one valid 20-N slope-normal filter.  Projecting (3,0,0)
    # against n=(1,0,1)/sqrt(2) leaves speed 3/sqrt(2).  The second filter's
    # NaN contact point must be ignored despite a nonzero force.
    right = _dedicated_pair(
        torch.tensor(
            (((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)),),
            dtype=torch.float64,
        ),
        torch.tensor(
            (((20.0, 0.0, 20.0), (0.0, 0.0, 100.0)),),
            dtype=torch.float64,
        ),
    )
    result = static_ground_contact_point_tangential_speed(
        body_pos, body_lin, body_ang, (left[0], right[0]), (left[1], right[1])
    )
    right_load = 20.0 * 2.0**0.5
    expected = (1.75 * 40.0 + (3.0 / 2.0**0.5) * right_load) / (
        40.0 + right_load
    )
    assert result.speed_per_foot[0, 0].item() == pytest.approx(1.75)
    assert result.speed_per_foot[0, 1].item() == pytest.approx(3.0 / 2.0**0.5)
    assert result.speed_per_env.item() == pytest.approx(expected)


def test_contact_point_slip_no_contact_is_invalid_and_bad_shape_fails_closed() -> None:
    body = torch.zeros((3, 2, 3))
    no_pos = torch.full((3, 1, 2, 3), torch.nan)
    no_force = torch.zeros_like(no_pos)
    result = static_ground_contact_point_tangential_speed(
        body, body, body, (no_pos, no_pos), (no_force, no_force)
    )
    assert not bool(result.valid_per_env.any())
    assert torch.equal(result.speed_per_env, torch.zeros(3))

    with pytest.raises(ValueError, match="shape"):
        static_ground_contact_point_tangential_speed(
            body,
            body,
            body,
            (no_pos[:, 0], no_pos),
            (no_force[:, 0], no_force),
        )


def test_fixed_command_sync_preserves_nonreset_ramp_history_and_fills_reset_rows() -> None:
    synchronize = _load_pure_helper(
        "_synchronize_evaluator_command_observation"
    )

    class History:
        max_length = 5
        _pointer = 2

        def __init__(self):
            self._buffer = torch.arange(5 * 3 * 3, dtype=torch.float32).reshape(
                5, 3, 3
            )

    class Term:
        vel_command_b = torch.full((3, 3), 99.0)
        is_standing_env = torch.ones(3, dtype=torch.bool)
        is_spin_env = torch.ones(3, dtype=torch.bool)
        is_heading_env = torch.ones(3, dtype=torch.bool)

    history = History()
    term = Term()

    class CommandManager:
        @staticmethod
        def get_term(name):
            assert name == "base_velocity"
            return term

    class ObservationManager:
        _group_obs_term_history_buffer = {
            "policy": {"velocity_commands": history}
        }

    class Unwrapped:
        num_envs = 3
        command_manager = CommandManager()
        observation_manager = ObservationManager()

    class Env:
        unwrapped = Unwrapped()

    policy = torch.arange(3 * 60, dtype=torch.float32).reshape(3, 60)
    observation = {"policy": policy}
    expected = torch.tensor(
        ((0.2, 0.0, 0.0), (0.8, 0.1, 0.0), (0.5, 0.0, -0.2))
    )
    reset = torch.tensor((False, True, False))
    old_storage = history._buffer.clone()
    old_policy = policy.clone()

    synchronize(Env(), observation, expected, reset)

    torch.testing.assert_close(term.vel_command_b, expected)
    assert not bool(term.is_standing_env.any())
    assert not bool(term.is_spin_env.any())
    assert not bool(term.is_heading_env.any())
    # Ordinary rows retain all four older causal frames; only newest changes.
    torch.testing.assert_close(policy[[0, 2], 30:42], old_policy[[0, 2], 30:42])
    torch.testing.assert_close(policy[[0, 2], 42:45], expected[[0, 2]])
    for storage_index in (0, 1, 3, 4):
        torch.testing.assert_close(
            history._buffer[storage_index, [0, 2]],
            old_storage[storage_index, [0, 2]],
        )
    # A managed-reset row begins a fresh physical segment, so all five frames
    # are the true evaluator command instead of a teacher resample.
    torch.testing.assert_close(
        policy[1, 30:45], expected[1].repeat(5)
    )
    torch.testing.assert_close(
        history._buffer[:, 1], expected[1].repeat(5, 1)
    )


def test_first_fall_masks_censor_managed_reset_and_all_later_samples() -> None:
    first_fall_masks = _load_pure_helper("_first_fall_masks")
    ever_failed = torch.tensor([False, False, True, False])
    falls = torch.tensor([True, False, True, False])

    at_risk, primary, first, repeated, updated = first_fall_masks(
        ever_failed, falls
    )

    assert torch.equal(at_risk, torch.tensor([True, True, False, True]))
    # Row zero is already a managed-reset state on return from env.step(); row
    # two belongs to an environment that failed previously.
    assert torch.equal(primary, torch.tensor([False, True, False, True]))
    assert torch.equal(first, torch.tensor([True, False, False, False]))
    assert torch.equal(repeated, torch.tensor([False, False, True, False]))
    assert torch.equal(updated, torch.tensor([True, False, True, False]))

    _, second_primary, second_first, second_repeated, _ = first_fall_masks(
        updated, torch.tensor([False, True, False, False])
    )
    assert torch.equal(second_primary, torch.tensor([False, False, False, True]))
    assert torch.equal(second_first, torch.tensor([False, True, False, False]))
    assert not bool(second_repeated.any())


def test_first_fall_masks_reject_shape_mismatch() -> None:
    first_fall_masks = _load_pure_helper("_first_fall_masks")

    with pytest.raises(ValueError, match="same shape"):
        first_fall_masks(torch.zeros(2), torch.zeros(3))


def test_masked_mean_excludes_post_reset_outliers_and_preserves_empty_nan() -> None:
    masked_mean = _load_pure_helper("_masked_tensor_mean")

    assert masked_mean(
        torch.tensor([1.0, 100.0, -200.0]),
        torch.tensor([True, False, False]),
    ) == pytest.approx(1.0)
    assert torch.isnan(
        torch.tensor(masked_mean(torch.ones(3), torch.zeros(3, dtype=torch.bool)))
    )


def test_warmup_falls_are_counted_before_measurement_continue() -> None:
    source = EVALUATOR.read_text()
    after_step = source.split("obs, rew, dones, extras = env.step(actions)", 1)[1]
    fall_count = after_step.index("falls_total +=")
    warmup_continue = after_step.index("if step < args_cli.warmup_steps:")

    assert fall_count < warmup_continue
    assert source.count("falls_total +=") == 1


def test_switch_warmup_failures_reach_global_zero_fall_gate() -> None:
    source = EVALUATOR.read_text()
    after_step = source.split("obs, _, dones, extras = env.step(actions)", 1)[1]

    assert after_step.index("fall_event_count +=") < after_step.index(
        "if phase_index < 0:"
    )
    assert "total_falls = fall_event_count" in source
    assert "warmup_fall_event_count" in source


def test_primary_metrics_are_first_fall_censored_with_raw_mirrors() -> None:
    source = EVALUATOR.read_text()

    assert "primary_sample = at_risk & ~falls" in source
    assert "_masked_tensor_mean(vel[:, 0], primary_sample_mask)" in source
    assert source.count("corrected_slip.speed_per_env") >= 2
    assert source.count(
        "primary_sample_mask & contact_slip_valid_per_env"
    ) >= 2
    assert '"mean_vx_including_resets"' in source
    assert '"mean_contact_slip_including_resets"' in source
    assert "post_reset_sample_count" in source


def test_machine_readable_safety_fields_are_emitted() -> None:
    source = EVALUATOR.read_text()

    for field in (
        "fall_event_count",
        "unique_env_first_fall_count",
        "time_to_first_fall_s",
        "failure_free_exposure_s",
        "post_reset_count",
    ):
        assert f'"{field}"' in source
    assert 'f"{output_csv.stem}.safety{output_csv.suffix}"' in source


def test_fall_diagnostics_identify_warmup_or_measurement_phase() -> None:
    source = EVALUATOR.read_text()

    assert '"phase",' in source
    assert 'phase = "warmup" if step < args_cli.warmup_steps else "measure"' in source
    assert '"phase": phase' in source


def test_ablation_is_applied_in_switch_and_matrix_loops() -> None:
    source = EVALUATOR.read_text()
    # Both evaluation loops must feed the explicitly ablated mapping to the
    # policy.  Otherwise a matrix labelled as a Hall-loss test is nominal.
    assert source.count("actions = policy(policy_obs)") >= 2


def test_exact_actor_capture_preserves_native_rsl_ablation() -> None:
    exact_input = _load_pure_helper("_exact_actor_policy_observation")

    class NativePolicy:
        pass

    submitted = {"policy": torch.arange(12, dtype=torch.float32).reshape(2, 6)}
    captured = exact_input(NativePolicy(), submitted)
    assert captured is submitted["policy"]

    class WrappedPolicy:
        last_policy_observation = torch.full((2, 6), 7.0)

    wrapped = exact_input(WrappedPolicy(), submitted)
    assert wrapped is WrappedPolicy.last_policy_observation

    with pytest.raises(ValueError, match="rank-2"):
        exact_input(NativePolicy(), {"policy": torch.zeros(6)})


def test_collector_segments_every_managed_reset_not_only_falls() -> None:
    source = EVALUATOR.read_text()

    assert "switch_collection_episode_generation[falls] += 1" not in source
    assert "collection_episode_generation[falls] += 1" not in source
    assert source.count(
        "switch_collection_episode_generation[managed_resets] += 1"
    ) >= 2
    assert source.count("collection_episode_generation[managed_resets] += 1") >= 2
    assert "managed_resets = dones.bool()" in source


def test_collect_npz_exports_reset_and_exact_hall_health_audit_fields() -> None:
    source = EVALUATOR.read_text()

    # Both switch and matrix collectors align these post-step outcomes with
    # the exact causal pre-step observation saved on the same row.
    assert source.count("done=np.concatenate(") >= 2
    assert source.count("time_out=np.concatenate(") >= 2
    assert source.count("hall_valid_lr=np.concatenate(") >= 2
    assert "switch_collection_pre_obs.index_select(0, selected)[" in source
    assert "collection_pre_obs.index_select(0, selected)[" in source
    assert source.count(":, 1860:1862") >= 2
    assert source.count("falls = managed_resets & ~timeout_mask") >= 2


def test_collection_metadata_is_pickle_free_and_hashes_absolute_actor_path(
    tmp_path: Path,
) -> None:
    collection_metadata = _load_collection_metadata_helper()
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"audited actor bytes")

    metadata = collection_metadata(
        dataset_kind="switch",
        task="Test-Hall-Task",
        seed=471,
        policy_dt=0.02,
        collect_stride=5,
        actor_checkpoint=actor,
        actor_source="rsl_checkpoint",
    )
    destination = tmp_path / "audit.npz"
    np.savez_compressed(destination, seed=np.asarray([471]), **metadata)

    with np.load(destination, allow_pickle=False) as archive:
        # Access every member so an accidental object array fails this test.
        for key in archive.files:
            assert archive[key].dtype.kind != "O"
        assert archive["task"].item() == "Test-Hall-Task"
        assert archive["policy_dt"].item() == pytest.approx(0.02)
        assert archive["collect_stride"].item() == 5
        assert archive["actor_checkpoint"].item() == str(actor.resolve())
        assert archive["actor_checkpoint_sha256"].item() == hashlib.sha256(
            actor.read_bytes()
        ).hexdigest()
        manifest = json.loads(archive["metadata_json"].item())
        assert manifest["seed"] == 471
        assert manifest["prospective_steps_contiguous"] is False
        assert manifest["hall_valid_lr_source"] == (
            "exact_actor_obs[:,1860:1862]"
        )
        assert manifest["schema_version"] == 2
        assert manifest["contact_slip_schema"] == (
            CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
        )
        assert manifest["contact_slip_metric_key"] == (
            CONTACT_POINT_TANGENTIAL_SLIP_KEY
        )
        assert manifest["contact_slip_valid_key"] == (
            CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY
        )
        assert manifest["contact_slip_formula"] == (
            CONTACT_POINT_TANGENTIAL_SLIP_FORMULA
        )
        assert manifest["legacy_contact_slip_metric_key"] == (
            LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY
        )
        assert manifest["actor_command_source"].endswith("[:,42:45]")
        assert "immediately before env.step" in manifest[
            "applied_command_source"
        ]


def test_collect_stride_metadata_records_effective_and_requested_values() -> None:
    collection_metadata = _load_collection_metadata_helper()

    metadata = collection_metadata(
        dataset_kind="matrix",
        task="Task",
        seed=7,
        policy_dt=0.02,
        collect_stride=0,
        actor_checkpoint=None,
        actor_source="unresolved",
    )

    assert metadata["collect_stride"].item() == 1
    assert metadata["collect_stride_requested"].item() == 0
    manifest = json.loads(metadata["metadata_json"].item())
    assert manifest["prospective_steps_contiguous"] is True


def test_both_collectors_export_corrected_and_explicit_legacy_slip_fields() -> None:
    source = EVALUATOR.read_text()

    assert source.count("contact_point_tangent_slip=np.concatenate(") == 2
    assert source.count(
        "contact_point_tangent_slip_valid=np.concatenate("
    ) == 2
    assert source.count("legacy_link_origin_planar_slip=np.concatenate(") == 2
    assert source.count("actor_command=np.concatenate(") == 2
    assert source.count("applied_command=np.concatenate(") == 2
    assert source.count("_simulator_contact_slip_metrics(") == 3  # def + two paths


def test_fixed_command_is_reasserted_and_reset_history_repaired_in_both_loops() -> None:
    source = EVALUATOR.read_text()

    # One initialization and one post-step repair in each collector path.
    assert source.count("_synchronize_evaluator_command_observation(") == 5
    assert source.count(
        "env, requested_vx, requested_vy, requested_wz"
    ) == 2
    assert source.count("actor_command_this_step =") == 2
    assert source.count("applied_command_this_step =") == 2
    assert "policy_observation[reset_mask, 30:45]" in source
    assert "policy_observation[:, 42:45] = expected_command" in source


def test_nominal_hall_is_installed_in_top_level_and_every_term_before_env() -> None:
    configure_nominal = _load_pure_helper(
        "_configure_nominal_hall_sensor_cfg"
    )

    @dataclass
    class HallCfg:
        enable_domain_randomization: bool = True
        enable_debug_vis: bool = True
        foot_dropout_probability: float = 0.15

    class Term:
        def __init__(self, cfg: HallCfg):
            self.params = {"hall_cfg": cfg}

    class Policy:
        foot_magnetic_array = Term(HallCfg())
        foot_sample_period_lr = Term(HallCfg())
        foot_sensor_valid_lr = Term(HallCfg())

    class Observations:
        policy = Policy()

    class EnvCfg:
        hall_sensor_cfg = HallCfg()
        observations = Observations()

    synchronized: list[str] = []

    def fake_sync(observations, top_level_cfg):
        assert top_level_cfg.enable_domain_randomization is False
        for name in (
            "foot_magnetic_array",
            "foot_sample_period_lr",
            "foot_sensor_valid_lr",
        ):
            term = getattr(observations.policy, name)
            term.params["hall_cfg"] = deepcopy(top_level_cfg)
            synchronized.append(name)
        return tuple(synchronized)

    env_cfg = EnvCfg()
    terms = configure_nominal(env_cfg, enabled=True, sync_fn=fake_sync)

    assert terms == tuple(synchronized)
    assert env_cfg.hall_sensor_cfg.enable_domain_randomization is False
    term_cfgs = [
        getattr(env_cfg.observations.policy, name).params["hall_cfg"]
        for name in terms
    ]
    assert all(cfg == env_cfg.hall_sensor_cfg for cfg in term_cfgs)
    assert all(cfg is not env_cfg.hall_sensor_cfg for cfg in term_cfgs)
    assert len({id(cfg) for cfg in term_cfgs}) == len(term_cfgs)

    source = EVALUATOR.read_text()
    assert source.index("nominal_hall_terms = _configure_nominal_hall_sensor_cfg(") < (
        source.index("env = gym.make(")
    )
    assert "sync_fn=sync_hall_sensor_cfg_to_policy_terms" in source


def test_default_hardened_hall_configuration_is_an_exact_noop() -> None:
    configure_nominal = _load_pure_helper(
        "_configure_nominal_hall_sensor_cfg"
    )

    class HallCfg:
        enable_domain_randomization = True

    class EnvCfg:
        hall_sensor_cfg = HallCfg()
        observations = object()

    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("disabled nominal path must not synchronize or mutate")

    env_cfg = EnvCfg()
    assert configure_nominal(env_cfg, enabled=False, sync_fn=forbidden_sync) == ()
    assert env_cfg.hall_sensor_cfg.enable_domain_randomization is True


def test_nominal_hall_needs_no_post_reset_runtime_buffer_patch() -> None:
    source = EVALUATOR.read_text()
    post_construction = source.split("env = gym.make(", 1)[1]

    # A managed reset now calls HallFootSensor.reset() with the already
    # nominal synchronized cfg.  The evaluator must not reach into private
    # packet/dropout buffers after either full or partial resets.
    assert "args_cli.nominal_magnetic_sensor" not in post_construction
    assert "_policy_foot_keep.fill_" not in source
    assert "_policy_channel_keep.fill_" not in source
    assert "_policy_delay_steps.zero_" not in source
    assert "magnetic_episode_valid_lr_buf" not in source
    assert "structured_foot_current_valid_buf" not in source


def test_collection_uses_exact_submitted_actor_tensor_and_disables_interval_mu() -> None:
    source = EVALUATOR.read_text()

    assert source.count(
        "exact_actor_observation = _exact_actor_policy_observation("
    ) >= 2
    assert "exact_policy_obs = exact_actor_observation" in source
    assert "collection_pre_obs = exact_actor_observation" in source
    # Exact matrix and switch labels must not be overwritten by an inherited
    # interval friction event, regardless of which evaluation mode is active.
    assert 'if hasattr(env_cfg.events, "friction_switch"):' in source
    assert 'args_cli.switch_sequence is not None and hasattr(\n        env_cfg.events, "friction_switch"' not in source


def test_recovery_expert_is_not_enabled_during_initial_probe_by_default() -> None:
    source = EVALUATOR.read_text()
    assert '"--hall_recovery_on_probe"' in source
    assert "probe_recovery = hall_governor.probing & bool(" in source
    assert "probe_recovery | (" in source


def test_recovery_expert_requires_confirmed_low_by_default() -> None:
    source = EVALUATOR.read_text()
    assert '"--hall_recovery_low_blend_floor"' in source
    assert "only a causal LOW" in source
    assert "torch.zeros_like(risk_alpha)" in source


def test_governed_command_reflex_is_hall_gated_and_has_no_truth_input() -> None:
    source = EVALUATOR.read_text()
    assert '"--hall_governed_command_reflex"' in source
    assert "legacy_actor_schema(" in source
    assert "governed_command_reflex_action" in source
    assert "no contact/slip/friction truth" in source


def test_rsl_actor_can_use_an_independent_hall_risk_governor() -> None:
    source = EVALUATOR.read_text()

    assert "rsl_hall_risk_mode" in source
    assert "attach independent Hall-only risk head to RSL actor" in source
    assert "RSL Hall-risk wrapper requires the exact deployable" in source
    assert "actor_groups = getattr(rsl_base_policy, \"obs_groups\", None)" in source
    assert "[observation[group] for group in actor_groups]" in source
    assert "_causal_hall_packet_validity" in source
    assert "valid=hall_valid" in source
    assert "contact, material, force or slip" in source


def test_rsl_evaluation_strictly_loads_actor_without_training_state() -> None:
    source = EVALUATOR.read_text()

    assert '"actor": True' in source
    assert '"critic": False' in source
    assert '"optimizer": False' in source
    assert '"iteration": False' in source
    assert "load_cfg=dict(EVAL_ACTOR_ONLY_LOAD_CFG)" in source
    assert "strict=True" in source
    assert "_disable_eval_capture_gate_warmup(agent_cfg)" in source


def test_eval_capture_warmup_disable_is_narrow_and_weight_preserving() -> None:
    disable = _load_pure_helper("_disable_eval_capture_gate_warmup")

    class Algorithm:
        capture_gate_warmup_updates = 50

    class Agent:
        algorithm = Algorithm()

    cfg = Agent()
    disable(cfg)
    assert cfg.algorithm.capture_gate_warmup_updates == 0

    # Ordinary PPO configs have no such field and remain valid.
    class OrdinaryAgent:
        algorithm = object()

    disable(OrdinaryAgent())


def _switch_phase_row(
    phase: int,
    mu: float,
    vx: float,
    cadence: float,
    step_length: float,
    *,
    response: float = 0.4,
    tilt: float = 8.0,
    lateral: float = 0.04,
    slip: float = 0.03,
    falls: int = 0,
) -> dict[str, object]:
    return {
        "phase": phase,
        "mu": mu,
        "steady_vx": vx,
        "step_frequency_hz": cadence,
        "mean_step_length_m": step_length,
        "mean_stride_length_m": 2.0 * step_length,
        "response_time_s": response,
        "high_start_recovery_response_time_s": response,
        "steady_abs_vy": lateral,
        "steady_mean_tilt_deg": 0.8 * tilt,
        "max_at_risk_tilt_deg": tilt,
        "steady_contact_point_tangent_slip_mps": slip,
        "fall_event_count": falls,
    }


def test_switch_gait_diagnostics_allow_fast_short_low_mu_gait() -> None:
    build = _load_pure_helper("_build_switch_gait_diagnostics")
    rows, transitions, recovery = build(
        [
            _switch_phase_row(0, 1.0, 0.80, 2.0, 0.40),
            # Same speed via higher cadence and shorter steps: this is valid
            # adaptation, not a failed fixed-slowdown target.
            _switch_phase_row(1, 0.12, 0.80, 4.0, 0.20),
            _switch_phase_row(2, 1.0, 0.784, 2.1, 0.38),
        ]
    )

    assert rows[1]["kinematic_speed_estimate_mps"] == pytest.approx(0.80)
    assert rows[1]["kinematic_closure_error_mps"] == pytest.approx(0.0)
    assert rows[1]["low_mu_cadence_vs_high_start_ratio"] == pytest.approx(2.0)
    assert rows[1]["low_mu_step_length_vs_high_start_ratio"] == pytest.approx(0.5)
    assert rows[1]["low_mu_vx_vs_high_start_ratio"] == pytest.approx(1.0)
    assert [row["transition_type"] for row in transitions] == [
        "HighToLow",
        "LowToHigh",
    ]
    assert recovery["high_start_phase"] == 0
    assert recovery["high_end_phase"] == 2
    assert recovery["vx_recovery_ratio"] == pytest.approx(0.98)
    assert recovery["step_length_recovery_ratio"] == pytest.approx(0.95)


def test_switch_acceptance_uses_high_end_not_high_phase_average() -> None:
    build = _load_pure_helper("_build_switch_gait_diagnostics")
    gate_builder = _load_pure_helper("_build_switch_acceptance_gates")
    rows, _, recovery = build(
        [
            _switch_phase_row(0, 1.0, 1.00, 2.0, 0.50),
            _switch_phase_row(1, 0.10, 0.75, 3.0, 0.25),
            # Formal recovery is 0.86/0.90, not an average of the two highs.
            _switch_phase_row(2, 1.0, 0.86, 2.0, 0.45),
        ]
    )
    gates, slip = gate_builder(
        rows,
        recovery,
        total_falls=0,
        min_vx_recovery_ratio=0.85,
        min_step_recovery_ratio=0.85,
        max_recovery_response_s=1.0,
        max_tilt_deg=20.0,
        max_steady_abs_vy_mps=0.25,
        max_contact_point_slip_mps=None,
    )
    names = {name for name, _, _, _ in gates}
    assert recovery["vx_recovery_ratio"] == pytest.approx(0.86)
    assert recovery["step_length_recovery_ratio"] == pytest.approx(0.90)
    assert all(passed for _, _, passed, _ in gates)
    assert slip["status"] == "diagnostic_only"
    assert "低摩擦稳态限速" not in names
    assert "高低摩擦稳态速度差" not in names
    assert "高摩擦步频恢复" not in names


def test_high_start_recovery_response_targets_original_high_speed() -> None:
    response = _load_pure_helper("_high_start_recovery_response_time")
    assert response(
        [0.20, 0.40, 0.80, 0.90, 0.90],
        high_start_speed=1.0,
        recovery_ratio=0.85,
        dt=0.10,
        window_steps=2,
    ) == pytest.approx(0.40)
    assert np.isnan(
        response(
            [0.20, 0.40, 0.60],
            high_start_speed=1.0,
            recovery_ratio=0.85,
            dt=0.10,
            window_steps=1,
        )
    )


def test_switch_optional_corrected_slip_and_safety_gates_fail_closed() -> None:
    build = _load_pure_helper("_build_switch_gait_diagnostics")
    gate_builder = _load_pure_helper("_build_switch_acceptance_gates")
    rows, _, recovery = build(
        [
            _switch_phase_row(0, 1.0, 1.0, 2.0, 0.5),
            _switch_phase_row(1, 0.1, 0.8, 3.2, 0.25, slip=0.13),
            _switch_phase_row(2, 1.0, 0.90, 2.0, 0.44),
        ]
    )
    gates, slip = gate_builder(
        rows,
        recovery,
        total_falls=0,
        min_vx_recovery_ratio=0.85,
        min_step_recovery_ratio=0.85,
        max_recovery_response_s=1.0,
        max_tilt_deg=20.0,
        max_steady_abs_vy_mps=0.25,
        max_contact_point_slip_mps=0.10,
    )
    result = {name: passed for name, _, passed, _ in gates}
    assert result["校正接触点切向滑移"] is False
    assert slip["schema"] == CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
    assert slip["gate_enabled"] is True


def test_projected_gravity_tilt_preserves_inverted_pose() -> None:
    tilt = _load_pure_helper("_projected_gravity_tilt_degrees")
    gravity = torch.tensor(
        [[0.0, 0.0, -1.0], [0.5, 0.0, -(3.0**0.5) / 2.0], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(tilt(gravity), torch.tensor([0.0, 30.0, 180.0]))


def test_detailed_hall_contact_is_preconstruction_fail_closed_and_provenanced() -> None:
    configure = _load_pure_helper("_configure_detailed_hall_contact_cfg")

    @dataclass
    class HallCfg:
        contact_distribution_mode: str = "aggregate"

    class EnvCfg:
        hall_sensor_cfg = HallCfg()
        observations = object()

    sync_calls = []

    def sync_fn(observations, hall_cfg):
        assert observations is EnvCfg.observations
        assert hall_cfg.contact_distribution_mode == "detailed"
        sync_calls.append(True)
        return ("foot_magnetic_array", "foot_sensor_valid_lr")

    env_cfg = EnvCfg()
    mode, terms = configure(env_cfg, enabled=False, sync_fn=sync_fn)
    assert (mode, terms) == ("aggregate", ())
    assert not sync_calls
    mode, terms = configure(env_cfg, enabled=True, sync_fn=sync_fn)
    assert mode == "detailed"
    assert terms == ("foot_magnetic_array", "foot_sensor_valid_lr")

    class MissingFieldCfg:
        hall_sensor_cfg = object()

    with pytest.raises(ValueError, match="silent fallback"):
        configure(MissingFieldCfg(), enabled=True, sync_fn=sync_fn)

    source = EVALUATOR.read_text()
    assert '"--detailed_hall_contact"' in source
    assert source.index("_configure_detailed_hall_contact_cfg(", source.index("def main")) < source.index(
        "env = gym.make("
    )
    assert source.count("hall_contact_distribution_mode=np.asarray(") >= 3
