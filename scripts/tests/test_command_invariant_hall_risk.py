from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

from unitree_rl_lab.traction.hall_risk_estimator import (
    COMMAND_HISTORY_SLICE,
    COMMAND_MASKED_MODEL_VARIANT,
    COMMAND_MASKED_TRAILING_FEATURE_MODE,
    CommandMaskedHallRiskEstimator,
    build_hall_risk_estimator,
    command_masked_risk_schema,
    command_masked_risk_schema_sha256,
)
from unitree_rl_lab.traction.layout_magnetic_student import INPUT_DIM, VALID_SLICE
from unitree_rl_lab.traction.contact_slip import (
    CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
    CONTACT_POINT_TANGENTIAL_SLIP_KEY,
    CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
    CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
    LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
)


ROOT = Path(__file__).resolve().parents[2]
TRACTION_SCRIPTS = ROOT / "scripts" / "traction"
if str(TRACTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRACTION_SCRIPTS))

from train_command_invariant_hall_risk_estimator import (  # noqa: E402
    OfflinePart,
    PROSPECTIVE_RISK_TARGET,
    RUNTIME_MEASUREMENT_BOUNDARY,
    grouped_evidence_permutation,
    load_spatial_part,
    load_switch_part,
    offline_research_gate,
    prospective_future_event_target,
    source_class_balanced_weights,
    validate_strict_heldout,
    weight_mass_report,
)
from spatial_friction_eval_utils import validate_motion_hall_risk_metadata  # noqa: E402


def _observation(rows: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    result = rng.normal(size=(rows, INPUT_DIM)).astype(np.float32)
    result[:, VALID_SLICE] = 1.0
    return result


def _checkpoint_payload(model: CommandMaskedHallRiskEstimator) -> dict[str, object]:
    schema = command_masked_risk_schema()
    schema_sha = command_masked_risk_schema_sha256()
    return {
        "model": model.state_dict(),
        "model_variant": COMMAND_MASKED_MODEL_VARIANT,
        "input_dim": INPUT_DIM,
        "trailing_feature_mode": COMMAND_MASKED_TRAILING_FEATURE_MODE,
        "masked_input_slices": {
            "command_history": [COMMAND_HISTORY_SLICE.start, COMMAND_HISTORY_SLICE.stop]
        },
        "observation_schema": schema,
        "observation_schema_sha256": schema_sha,
        "schema_sha256": schema_sha,
    }


def _contact_slip_provenance(
    observation: np.ndarray,
    slip: np.ndarray,
    slip_valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if slip_valid is None:
        slip_valid = np.ones(len(slip), dtype=bool)
    actor_command = observation[
        :, COMMAND_HISTORY_SLICE.stop - 3 : COMMAND_HISTORY_SLICE.stop
    ].copy()
    manifest = {
        "contact_slip_schema": CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA,
        "contact_slip_metric_key": CONTACT_POINT_TANGENTIAL_SLIP_KEY,
        "contact_slip_valid_key": CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY,
        "contact_slip_formula": CONTACT_POINT_TANGENTIAL_SLIP_FORMULA,
        "legacy_contact_slip_metric_key": LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY,
        "actor_command_key": "actor_command",
        "actor_command_source": (
            "exact pre-step actor observation newest command frame [:,42:45]"
        ),
        "applied_command_key": "applied_command",
        "applied_command_source": (
            "base_velocity.vel_command_b[:,0:3] snapshot immediately before env.step"
        ),
    }
    return {
        "contact_slip": np.asarray(slip, dtype=np.float32),
        CONTACT_POINT_TANGENTIAL_SLIP_KEY: np.asarray(slip, dtype=np.float32),
        CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY: np.asarray(
            slip_valid, dtype=bool
        ),
        LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY: np.zeros(
            len(slip), dtype=np.float32
        ),
        "actor_command": actor_command,
        "applied_command": actor_command.copy(),
        "contact_slip_schema": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_SCHEMA
        ),
        "contact_slip_metric_key": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_KEY
        ),
        "contact_slip_valid_key": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_VALID_KEY
        ),
        "contact_slip_formula": np.asarray(
            CONTACT_POINT_TANGENTIAL_SLIP_FORMULA
        ),
        "legacy_contact_slip_metric_key": np.asarray(
            LEGACY_LINK_ORIGIN_PLANAR_SLIP_KEY
        ),
        "metadata_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }


def test_command_mask_is_internal_and_probability_is_exactly_invariant():
    torch.manual_seed(3)
    model = CommandMaskedHallRiskEstimator().eval()
    first = torch.from_numpy(_observation(8))
    second = first.clone()
    second[:, COMMAND_HISTORY_SLICE] = torch.linspace(
        -100.0,
        100.0,
        steps=8 * (COMMAND_HISTORY_SLICE.stop - COMMAND_HISTORY_SLICE.start),
    ).reshape(8, -1)
    with torch.inference_mode():
        feature_first, _ = model.raw_features(first)
        feature_second, _ = model.raw_features(second)
        risk_first = model(first)
        risk_second = model(second)
    assert torch.equal(feature_first, feature_second)
    assert torch.equal(risk_first, risk_second)


def test_grouped_hall_permutation_changes_only_hall_and_period() -> None:
    observation = _observation(12)
    # Make every row unique and use two sufficiently large independent groups.
    observation += np.arange(12, dtype=np.float32)[:, None]
    source = np.asarray(["a"] * 6 + ["b"] * 6)
    phase = np.asarray([0, 0, 0, 1, 1, 1] * 2, dtype=np.int64)
    first = grouped_evidence_permutation(
        observation, source, phase, evidence="hall", seed=41
    )
    second = grouped_evidence_permutation(
        observation, source, phase, evidence="hall", seed=41
    )

    assert np.array_equal(first, second)
    # Hall+period is [480:1860); command/proprio, valid and motion feedback are
    # bit-identical to the submitted actor packet.
    assert np.array_equal(first[:, :480], observation[:, :480])
    assert np.array_equal(first[:, 1860:], observation[:, 1860:])
    # A grouped permutation preserves every group's marginal evidence exactly.
    for source_value in np.unique(source):
        for phase_value in np.unique(phase[source == source_value]):
            rows = (source == source_value) & (phase == phase_value)
            before = np.sort(observation[rows, 480:1860], axis=0)
            after = np.sort(first[rows, 480:1860], axis=0)
            assert np.array_equal(before, after)


def test_offline_gate_rejects_proprio_shortcut_and_reports_primary_sources() -> None:
    target = np.asarray((0, 0, 1, 1, 0, 1, 0, 1), dtype=np.float32)
    prediction = np.asarray((0.1, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2, 0.9), dtype=np.float32)
    data = {
        "target": target,
        "source_id": np.asarray(["high"] * 4 + ["slow"] * 4),
        "command_vx": np.asarray([0.8] * 4 + [0.32] * 4, dtype=np.float32),
    }
    failed = offline_research_gate(
        data,
        prediction,
        {"hall": {"auc_drop": 0.0}},
        threshold=0.5,
        primary_command_min=0.5,
        min_primary_auc=0.85,
        min_primary_recall=0.8,
        max_primary_false_alarm_rate=0.1,
        min_hall_auc_drop=0.03,
        command_invariance_exact=True,
        strict_restore_delta=5.0e-8,
    )
    assert failed["passed"] is False
    assert any("Hall permutation" in reason for reason in failed["failures"])
    assert failed["per_source"]["high"]["primary_operating_source"] is True
    assert failed["per_source"]["slow"]["primary_operating_source"] is False

    passed = offline_research_gate(
        data,
        prediction,
        {"hall": {"auc_drop": 0.2}},
        threshold=0.5,
        primary_command_min=0.5,
        min_primary_auc=0.85,
        min_primary_recall=0.8,
        max_primary_false_alarm_rate=0.1,
        min_hall_auc_drop=0.03,
        command_invariance_exact=True,
        strict_restore_delta=5.0e-8,
    )
    assert passed["passed"] is True


def test_strict_factory_accepts_complete_schema_and_rejects_semantic_drift():
    model = CommandMaskedHallRiskEstimator().eval()
    payload = _checkpoint_payload(model)
    restored = build_hall_risk_estimator(payload).eval()
    observation = torch.from_numpy(_observation(3))
    with torch.inference_mode():
        torch.testing.assert_close(model(observation), restored(observation))

    for key in (
        "input_dim",
        "trailing_feature_mode",
        "masked_input_slices",
        "observation_schema",
        "observation_schema_sha256",
        "schema_sha256",
    ):
        broken = dict(payload)
        broken.pop(key)
        with pytest.raises(ValueError):
            build_hall_risk_estimator(broken)

    broken = dict(payload)
    broken["masked_input_slices"] = {"command_history": [31, 45]}
    with pytest.raises(ValueError, match="masked_input_slices"):
        build_hall_risk_estimator(broken)


def test_spatial_loader_applies_causal_washout_and_both_valid_mask(tmp_path: Path):
    path = tmp_path / "spatial_seed450.npz"
    observation = _observation(7)
    observation[1, VALID_SLICE.start] = 0.0
    np.savez_compressed(
        path,
        observation=observation,
        fastbase_course_stage=np.asarray((0, 0, 1, 1, 1, 2, 2), dtype=np.uint8),
        fastbase_env_id=np.zeros(7, dtype=np.int32),
        fastbase_rollout_step=np.arange(7, dtype=np.int32),
    )
    part = load_spatial_part(path, transition_washout_steps=2, course_pattern="HLH")
    # step 1 is invalid; steps 2,3 and 5,6 are transition washout.
    np.testing.assert_array_equal(part.step, np.asarray((0, 4)))
    np.testing.assert_array_equal(part.target, np.asarray((0.0, 1.0)))
    assert part.seed == 450
    assert part.audit["not_both_valid_removed"] == 1
    assert part.audit["transition_washout_removed"] == 4


def test_spatial_loader_explicit_lhl_pattern_inverts_the_stage_mapping(tmp_path: Path):
    path = tmp_path / "spatial_seed466.npz"
    observation = _observation(6)
    np.savez_compressed(
        path,
        observation=observation,
        fastbase_course_stage=np.asarray((0, 0, 1, 1, 2, 2), dtype=np.uint8),
        fastbase_env_id=np.zeros(6, dtype=np.int32),
        fastbase_rollout_step=np.arange(6, dtype=np.int32),
    )
    hlh = load_spatial_part(path, transition_washout_steps=0, course_pattern="HLH")
    lhl = load_spatial_part(path, transition_washout_steps=0, course_pattern="LHL")
    np.testing.assert_array_equal(hlh.target, np.asarray((0, 0, 1, 1, 0, 0)))
    np.testing.assert_array_equal(lhl.target, 1.0 - hlh.target)
    assert "spatial-LHL" in lhl.source_id
    with pytest.raises(ValueError, match="explicitly"):
        load_spatial_part(path, transition_washout_steps=0, course_pattern="unknown")


def test_switch_loader_uses_mu_only_as_offline_label_and_washes_transition(
    tmp_path: Path,
):
    path = tmp_path / "switch_seed370.npz"
    observation = _observation(6)
    observation[4, VALID_SLICE.stop - 1] = 0.0
    np.savez_compressed(
        path,
        obs=observation,
        mu=np.asarray((0.8, 0.15, 0.5, 0.15, 0.8, 0.8), dtype=np.float32),
        env_id=np.zeros(6, dtype=np.int32),
        step=np.arange(6, dtype=np.int32),
        # t=.30 still contains one pre-switch Hall frame; t=.32 is the first
        # fully refreshed 15-frame window and must be retained.
        time_since_switch_s=np.asarray((0.30, 0.32, 1.0, 0.8, 0.9, 0.7), dtype=np.float32),
        seed=np.full(6, 370, dtype=np.int32),
        valid=np.asarray((True, True, True, True, True, False)),
    )
    part = load_switch_part(path, 0.32, low_mu_max=0.25, high_mu_min=0.75)
    np.testing.assert_array_equal(part.step, np.asarray((1, 3)))
    np.testing.assert_array_equal(part.target, np.ones(2, dtype=np.float32))
    assert part.audit["transition_washout_removed"] == 1
    assert part.audit["ambiguous_mu_removed"] == 1


def test_prospective_target_includes_aligned_current_outcome_and_requires_complete_future():
    slip = np.asarray((0.10, 0.0, 0.10, 0.0, 0.0), dtype=np.float32)
    fall = np.asarray((False, False, True, False, True))
    target, valid = prospective_future_event_target(
        slip,
        fall,
        rollout_id=np.zeros(5, dtype=np.int64),
        env_id=np.zeros(5, dtype=np.int64),
        phase=np.zeros(5, dtype=np.int64),
        step=np.arange(5, dtype=np.int64),
        horizon_steps=2,
        contact_slip_threshold=0.045,
        future_slip_quantile=1.0,
    )
    # Each obs row is pre-step and its slip/fall is the aligned post-step
    # outcome.  With a two-step window, t=0 includes outcomes 0 and 1; t=1
    # includes outcomes 1 and 2; t=3 includes outcomes 3 and 4 (the fall).
    # Only the final row lacks a complete two-outcome horizon.
    np.testing.assert_array_equal(target, np.asarray((1, 1, 1, 1, 0)))
    np.testing.assert_array_equal(valid, np.asarray((True, True, True, True, False)))


def test_prospective_target_uses_sustained_quantile_not_single_solver_spike():
    count = 25
    single_spike = np.zeros(count, dtype=np.float32)
    single_spike[5] = 1.0
    target, valid = prospective_future_event_target(
        single_spike,
        np.zeros(count, dtype=bool),
        rollout_id=np.zeros(count, dtype=np.int64),
        env_id=np.zeros(count, dtype=np.int64),
        phase=np.zeros(count, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        horizon_steps=12,
        contact_slip_threshold=0.045,
        future_slip_quantile=0.75,
    )
    assert valid[0]
    assert target[0] == 0.0

    sustained = np.zeros(count, dtype=np.float32)
    sustained[3:7] = 0.10
    target, _ = prospective_future_event_target(
        sustained,
        np.zeros(count, dtype=bool),
        rollout_id=np.zeros(count, dtype=np.int64),
        env_id=np.zeros(count, dtype=np.int64),
        phase=np.zeros(count, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        horizon_steps=12,
        contact_slip_threshold=0.045,
        future_slip_quantile=0.75,
    )
    assert target[0] == 1.0


def test_prospective_default_quantile_requires_four_of_twelve_frames():
    count = 24
    three_frames = np.zeros(count, dtype=np.float32)
    three_frames[:3] = 0.10
    target, valid = prospective_future_event_target(
        three_frames,
        np.zeros(count, dtype=bool),
        rollout_id=np.zeros(count, dtype=np.int64),
        env_id=np.zeros(count, dtype=np.int64),
        phase=np.zeros(count, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        horizon_steps=12,
        contact_slip_threshold=0.045,
        future_slip_quantile=0.75,
    )
    assert valid[0]
    assert target[0] == 0.0

    four_frames = three_frames.copy()
    four_frames[3] = 0.10
    target, _ = prospective_future_event_target(
        four_frames,
        np.zeros(count, dtype=bool),
        rollout_id=np.zeros(count, dtype=np.int64),
        env_id=np.zeros(count, dtype=np.int64),
        phase=np.zeros(count, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        horizon_steps=12,
        contact_slip_threshold=0.045,
        future_slip_quantile=0.75,
    )
    assert target[0] == 1.0


def test_prospective_target_never_crosses_phase_rollout_or_reset_and_handles_four_phases():
    phase = np.repeat(np.arange(4, dtype=np.int64), 100)
    step = np.tile(np.arange(100, dtype=np.int64), 4)
    rollout = phase * 1_000_000
    slip = np.zeros(400, dtype=np.float32)
    # First row of the new phase must not leak into the preceding phase.
    slip[100] = 0.2
    target, valid = prospective_future_event_target(
        slip,
        np.zeros(400, dtype=bool),
        rollout_id=rollout,
        env_id=np.zeros(400, dtype=np.int64),
        phase=phase,
        step=step,
        horizon_steps=12,
        contact_slip_threshold=0.045,
    )
    assert int(valid.sum()) == 4 * (100 - 11)
    assert not target[:100].any()
    # A single solver spike is insufficient under the sustained-slip rule.
    assert target[100] == 0.0
    for phase_id in range(4):
        tail = (phase == phase_id) & (step >= 89)
        assert not valid[tail].any()


def test_prospective_target_does_not_cross_rollout_reset_with_same_phase_id():
    step = np.tile(np.arange(4, dtype=np.int64), 2)
    rollout = np.repeat(np.asarray((10, 11), dtype=np.int64), 4)
    slip = np.zeros(8, dtype=np.float32)
    slip[4] = 0.2
    target, valid = prospective_future_event_target(
        slip,
        np.zeros(8, dtype=bool),
        rollout_id=rollout,
        env_id=np.zeros(8, dtype=np.int64),
        phase=np.zeros(8, dtype=np.int64),
        step=step,
        horizon_steps=2,
        contact_slip_threshold=0.045,
        future_slip_quantile=0.75,
    )
    assert not target[:4].any()
    np.testing.assert_array_equal(
        valid, np.asarray((True, True, True, False, True, True, True, False))
    )


def test_prospective_target_censors_no_contact_negatives_but_keeps_falls() -> None:
    slip = np.zeros(5, dtype=np.float32)
    slip_valid = np.asarray((True, False, True, True, True))
    kwargs = dict(
        contact_slip=slip,
        rollout_id=np.zeros(5, dtype=np.int64),
        env_id=np.zeros(5, dtype=np.int64),
        phase=np.zeros(5, dtype=np.int64),
        step=np.arange(5, dtype=np.int64),
        horizon_steps=2,
        contact_slip_threshold=0.045,
        contact_slip_valid=slip_valid,
    )
    target, valid = prospective_future_event_target(
        fall=np.zeros(5, dtype=bool), **kwargs
    )
    assert not valid[0]  # window includes undefined no-contact outcome 1
    assert not target[0]

    fall = np.zeros(5, dtype=bool)
    fall[1] = True
    target, valid = prospective_future_event_target(fall=fall, **kwargs)
    assert valid[0]
    assert target[0] == 1.0


def test_prospective_loader_keeps_real_early_precursors_despite_optional_washout(
    tmp_path: Path,
):
    path = tmp_path / "prospective_seed461.npz"
    count = 8
    observation = _observation(count)
    slip = np.zeros(count, dtype=np.float32)
    slip[:3] = 0.10
    np.savez_compressed(
        path,
        obs=observation,
        mu=np.full(count, 0.28, dtype=np.float32),
        env_id=np.zeros(count, dtype=np.int32),
        step=np.arange(count, dtype=np.int32),
        time_since_switch_s=np.arange(1, count + 1, dtype=np.float32) * 0.02,
        seed=np.full(count, 461, dtype=np.int32),
        valid=np.ones(count, dtype=bool),
        phase=np.ones(count, dtype=np.int16),
        rollout_id=np.full(count, 1_000_000, dtype=np.int64),
        fall=np.zeros(count, dtype=bool),
        done=np.zeros(count, dtype=bool),
        time_out=np.zeros(count, dtype=bool),
        hall_valid_lr=observation[:, VALID_SLICE],
        policy_dt=np.asarray(0.02, dtype=np.float64),
        collect_stride=np.asarray(1, dtype=np.int32),
        dataset_kind=np.asarray("switch"),
        cmd_vx=np.full(count, 0.8, dtype=np.float32),
        **_contact_slip_provenance(observation, slip),
    )
    part = load_switch_part(
        path,
        transition_washout_s=0.32,
        low_mu_max=0.30,
        high_mu_min=0.75,
        label_mode="prospective_slip_fall",
        future_horizon_s=0.04,
        policy_dt_s=0.02,
        contact_slip_threshold=0.045,
        prospective_transition_washout_s=0.32,
    )
    # t=0 and t=1 each have two sustained aligned outcomes and are retained
    # even though they precede the optional .32-s negative washout boundary.
    np.testing.assert_array_equal(part.step, np.asarray((0, 1)))
    np.testing.assert_array_equal(part.target, np.ones(2, dtype=np.float32))
    assert part.audit["positive_transition_rows_rescued"] == 2
    assert part.audit["right_censored_horizon_removed"] == 1
    assert part.audit["future_slip_quantile"] == 0.75
    assert part.audit["hall_valid_lr_cross_checked"] is True
    assert part.audit["managed_reset_segmentation_checked"] is True
    assert part.audit["contact_slip_source"] == (
        CONTACT_POINT_TANGENTIAL_SLIP_KEY
    )
    assert part.audit["research_legacy_link_origin_slip"] is False


def test_prospective_loader_rejects_old_slip_by_default_and_requires_explicit_research_override(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_seed489.npz"
    count = 7
    observation = _observation(count)
    legacy_slip = np.asarray((0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    np.savez_compressed(
        path,
        obs=observation,
        mu=np.full(count, 0.2, dtype=np.float32),
        env_id=np.zeros(count, dtype=np.int32),
        step=np.arange(count, dtype=np.int32),
        time_since_switch_s=np.arange(1, count + 1, dtype=np.float32) * 0.02,
        seed=np.full(count, 489, dtype=np.int32),
        valid=np.ones(count, dtype=bool),
        phase=np.zeros(count, dtype=np.int16),
        rollout_id=np.zeros(count, dtype=np.int64),
        contact_slip=legacy_slip,
        fall=np.zeros(count, dtype=bool),
        done=np.zeros(count, dtype=bool),
        time_out=np.zeros(count, dtype=bool),
        hall_valid_lr=observation[:, VALID_SLICE],
        policy_dt=np.asarray(0.02, dtype=np.float64),
        collect_stride=np.asarray(1, dtype=np.int32),
        dataset_kind=np.asarray("switch"),
    )
    with pytest.raises(ValueError, match="rejected by default"):
        load_switch_part(
            path,
            0.0,
            0.3,
            0.75,
            label_mode="prospective_slip_fall",
            future_horizon_s=0.04,
            policy_dt_s=0.02,
        )

    research = load_switch_part(
        path,
        0.0,
        0.3,
        0.75,
        label_mode="prospective_slip_fall",
        future_horizon_s=0.04,
        policy_dt_s=0.02,
        allow_research_legacy_link_origin_slip=True,
    )
    assert research.audit["research_legacy_link_origin_slip"] is True
    assert research.audit["contact_slip_source"] == (
        "unversioned_contact_slip_legacy_assumed"
    )


def test_prospective_loader_rejects_noncontiguous_metadata_and_bad_hall_copy(
    tmp_path: Path,
) -> None:
    count = 5
    observation = _observation(count)

    def save(
        path: Path,
        *,
        stride: int,
        corrupt_hall: bool = False,
        corrupt_actor_command: bool = False,
    ) -> None:
        hall_valid = observation[:, VALID_SLICE].copy()
        if corrupt_hall:
            hall_valid[0, 0] = 0.0
        provenance = _contact_slip_provenance(
            observation, np.zeros(count, dtype=np.float32)
        )
        if corrupt_actor_command:
            provenance["actor_command"] = provenance["actor_command"].copy()
            provenance["actor_command"][0, 0] += 0.1
        np.savez_compressed(
            path,
            obs=observation,
            mu=np.full(count, 0.28, dtype=np.float32),
            env_id=np.zeros(count, dtype=np.int32),
            step=np.arange(count, dtype=np.int32),
            time_since_switch_s=np.arange(1, count + 1, dtype=np.float32) * 0.02,
            seed=np.full(count, 490, dtype=np.int32),
            valid=np.ones(count, dtype=bool),
            phase=np.ones(count, dtype=np.int16),
            rollout_id=np.zeros(count, dtype=np.int64),
            fall=np.zeros(count, dtype=bool),
            done=np.zeros(count, dtype=bool),
            time_out=np.zeros(count, dtype=bool),
            hall_valid_lr=hall_valid,
            policy_dt=np.asarray(0.02, dtype=np.float64),
            collect_stride=np.asarray(stride, dtype=np.int32),
            dataset_kind=np.asarray("switch"),
            cmd_vx=np.full(count, 0.8, dtype=np.float32),
            **provenance,
        )

    stride_path = tmp_path / "stride_seed490.npz"
    save(stride_path, stride=2)
    with pytest.raises(ValueError, match="collect_stride=1"):
        load_switch_part(
            stride_path,
            0.0,
            0.30,
            0.75,
            label_mode="prospective_slip_fall",
            future_horizon_s=0.04,
            policy_dt_s=0.02,
        )

    hall_path = tmp_path / "hall_seed490.npz"
    save(hall_path, stride=1, corrupt_hall=True)
    with pytest.raises(ValueError, match="hall_valid_lr does not match"):
        load_switch_part(
            hall_path,
            0.0,
            0.30,
            0.75,
            label_mode="prospective_slip_fall",
            future_horizon_s=0.04,
            policy_dt_s=0.02,
        )

    command_path = tmp_path / "command_seed490.npz"
    save(command_path, stride=1, corrupt_actor_command=True)
    with pytest.raises(ValueError, match="actor_command does not match"):
        load_switch_part(
            command_path,
            0.0,
            0.30,
            0.75,
            label_mode="prospective_slip_fall",
            future_horizon_s=0.04,
            policy_dt_s=0.02,
        )


def test_prospective_checkpoint_metadata_is_accepted_by_strict_spatial_governor():
    report = validate_motion_hall_risk_metadata(
        {
            "input_dim": INPUT_DIM,
            "trailing_feature_mode": COMMAND_MASKED_TRAILING_FEATURE_MODE,
            "measurement_boundary": RUNTIME_MEASUREMENT_BOUNDARY,
            "risk_target": PROSPECTIVE_RISK_TARGET,
            "model_variant": COMMAND_MASKED_MODEL_VARIANT,
            "model": {"network.0.weight": object()},
        }
    )
    assert report["risk_target"] == "prospective contact-point slip/fall"
    assert report["model_variant"] == COMMAND_MASKED_MODEL_VARIANT


def _part(path: Path, kind: str, seed: int, target: tuple[float, ...]) -> OfflinePart:
    count = len(target)
    return OfflinePart(
        observation=_observation(count),
        target=np.asarray(target, dtype=np.float32),
        env_id=np.zeros(count, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        command_vx=np.zeros(count, dtype=np.float32),
        phase=np.zeros(count, dtype=np.int64),
        source_kind=kind,
        source_id=f"{kind}:{path.name}:seed{seed}",
        seed=seed,
        path=path.resolve(),
        audit={},
    )


def test_strict_split_rejects_any_whole_seed_overlap(tmp_path: Path):
    train = [_part(tmp_path / "a.npz", "spatial", 450, (0.0, 1.0))]
    heldout = [_part(tmp_path / "b.npz", "spatial", 450, (0.0, 1.0))]
    with pytest.raises(ValueError, match="seed overlap"):
        validate_strict_heldout(train, heldout)


def test_source_class_weighting_assigns_equal_mass_per_kind_and_class():
    kind = np.asarray(
        ("spatial",) * 6 + ("switch",) * 8,
        dtype="U16",
    )
    source = np.asarray(
        ("s450",) * 2 + ("s451",) * 4 + ("g370",) * 3 + ("g371",) * 5,
        dtype="U16",
    )
    target = np.asarray(
        (0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1),
        dtype=np.float32,
    )
    weight = source_class_balanced_weights(kind, source, target)
    report = weight_mass_report(kind, source, target, weight)
    for key in ("spatial/safe", "spatial/risk", "switch/safe", "switch/risk"):
        assert report[key] == pytest.approx(0.25)
    assert np.isfinite(weight).all()
    assert np.all(weight > 0.0)
