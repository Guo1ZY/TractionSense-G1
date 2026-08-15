"""Fast regression tests for the physical Hall H-L-H evaluator.

The Isaac process is exercised by ``eval_spatial_friction_course.py`` itself.
These tests intentionally stay Kit-free so they can run on every edit and
protect the causal transition accounting plus the runtime smoke contract.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "scripts" / "rsl_rl" / "eval_spatial_friction_course.py"
UTIL_PATH = ROOT / "scripts" / "traction" / "spatial_friction_eval_utils.py"
CFG_PATH = (
    ROOT
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "tasks"
    / "locomotion"
    / "robots"
    / "g1"
    / "29dof"
    / "velocity_foot_env_cfg.py"
)


def _load_utils():
    spec = importlib.util.spec_from_file_location("spatial_friction_eval_utils", UTIL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves its module through sys.modules while decorating.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name!r} not found in {path}")


def _load_eval_command_rewrite_helper():
    """Load the tensor-only helper without importing Isaac or parsing CLI."""

    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    required_assignments = {
        "RECOVERY_COMMAND_VX_INDICES",
        "RECOVERY_COMMAND_VY_INDICES",
        "RECOVERY_COMMAND_YAW_INDICES",
        "RECOVERY_POLICY_OBSERVATION_DIM",
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in required_assignments
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_with_recovery_command_history"
        )
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, str(EVAL_PATH), "exec"), namespace)
    return namespace


def _load_course_geometry_helpers(task: str):
    """Load only the task-to-course mapping without Isaac or CLI parsing."""

    tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))
    names = {
        "_runtime_course_geometry",
        "_runtime_patches",
        "_course_stage_from_local_x",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "args_cli": SimpleNamespace(task=task, expected_low_mu=None),
        "torch": torch,
    }
    exec(compile(module, str(EVAL_PATH), "exec"), namespace)
    return namespace


def test_hlh_tracker_requires_one_uninterrupted_episode():
    module = _load_utils()
    stage = module.WAIT_HIGH_START
    for sample in (
        module.SpatialTransitionSample(-0.5, False),
        module.SpatialTransitionSample(0.5, True),
        module.SpatialTransitionSample(1.5, False),
    ):
        stage = module.advance_high_low_high_stage(stage, sample)
    assert stage == module.COMPLETE

    # A fall/reset between low and the final high must not become a fake
    # recovery completion after the robot respawns on the first blue patch.
    stage = module.WAIT_HIGH_START
    for sample in (
        module.SpatialTransitionSample(-0.5, False),
        module.SpatialTransitionSample(0.5, True),
        module.SpatialTransitionSample(0.6, True, done=True),
        module.SpatialTransitionSample(-0.5, False),
        module.SpatialTransitionSample(1.5, False),
    ):
        stage = module.advance_high_low_high_stage(stage, sample)
    assert stage == module.WAIT_LOW

    # Crossing x=1 while airborne is not a physical high-patch recovery.
    stage = module.WAIT_HIGH_END
    stage = module.advance_high_low_high_stage(
        stage,
        module.SpatialTransitionSample(
            1.5,
            False,
            high_end_contact=False,
        ),
    )
    assert stage == module.WAIT_HIGH_END


def test_label_compression_and_invalid_boundaries():
    module = _load_utils()
    assert module.compress_contact_labels([False, False, True, True, False]) == [
        "HIGH",
        "LOW",
        "HIGH",
    ]
    try:
        module.advance_high_low_high_stage(
            module.WAIT_HIGH_START,
            module.SpatialTransitionSample(0.0, False),
            low_start_x=1.0,
            high_end_x=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid H-L-H boundaries were accepted")


def test_causal_response_excludes_post_reset_and_reports_missing_horizons():
    module = _load_utils()
    steps = 20
    vx = [[1.0, 0.8] for _ in range(steps)]
    low = [[False, False] for _ in range(steps)]
    high_end = [[False, False] for _ in range(steps)]
    falls = [[False, False] for _ in range(steps)]
    dones = [[False, False] for _ in range(steps)]
    for step in range(2, 13):
        low[step] = [True, True]
    vx[7][0] = 0.50
    vx[12][0] = 0.30
    for step in range(13, steps):
        high_end[step][0] = True
    vx[13][0] = 0.40
    vx[14][0] = 0.60
    for step in range(15, steps):
        vx[step][0] = 0.75
    falls[4][1] = True
    dones[4][1] = True
    # Deliberately fast post-reset values must not become horizon samples.
    for step in range(5, steps):
        vx[step][1] = 9.0

    report = module.analyze_transition_response(
        body_vx=vx,
        low_contact=low,
        high_end_contact=high_end,
        falls=falls,
        dones=dones,
        step_dt_s=0.1,
        low_speed_target_m_s=0.45,
        high_recovery_speed_m_s=0.70,
        recovery_stable_steps=3,
    )
    assert report["first_fall_step"] == 4
    assert report["first_fall_time_s"] == 0.5
    assert report["deceleration_after_low_contact"]["0.5s"]["sampled_envs"] == 1
    assert report["deceleration_after_low_contact"]["0.5s"]["deceleration_m_s"]["mean"] == 0.5
    assert report["deceleration_after_low_contact"]["1s"]["vx_m_s"]["mean"] == 0.3
    assert report["absolute_high_recovery"]["recovered_envs"] == 1
    assert abs(report["absolute_high_recovery"]["time_s"]["mean"] - 0.2) < 1.0e-9


def test_hall_health_groups_separate_channel_and_foot_faults():
    module = _load_utils()
    healthy = [[True] * 15, [True] * 15]
    assert module.classify_hall_health(healthy, [True, True]) == "fully_healthy"
    degraded = [healthy[0].copy(), healthy[1].copy()]
    degraded[1][4] = False
    assert module.classify_hall_health(degraded, [True, True]) == "channel_degraded"
    assert module.classify_hall_health(healthy, [True, False]) == "single_foot_offline"
    assert module.classify_hall_health(healthy, [False, False]) == "both_feet_offline"


def test_eval_declares_three_ordered_physical_patch_contracts():
    assert _literal_assignment(EVAL_PATH, "PATCHES") == (
        ("FrictionHighStart", 0.90, -0.50, 0),
        ("FrictionLow", 0.16, 0.50, 1),
        ("FrictionHighEnd", 0.90, 1.50, 2),
    )
    source = EVAL_PATH.read_text(encoding="utf-8")
    # Live USD schema/material checks: config-text inspection alone is not a
    # sufficient smoke test for cloned PhysX colliders.
    for token in (
        'f"{root_path}/geometry/mesh"',
        "mesh_prim.HasAPI(UsdPhysics.CollisionAPI)",
        "root_prim.HasAPI(UsdPhysics.RigidBodyAPI)",
        "mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI)",
        "GetStaticFrictionAttr().Get()",
        "GetDynamicFrictionAttr().Get()",
        "GetFrictionCombineModeAttr().Get()",
        'combine != "multiply"',
    ):
        assert token in source


def test_runtime_geometry_distinguishes_short_training_and_long_demo_course():
    ordinary = _load_course_geometry_helpers(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-SpatialFrictionMedium"
    )
    assert ordinary["_runtime_course_geometry"]() == {
        "long_course": False,
        "low_start_x_m": 0.0,
        "low_end_x_m": 1.0,
        "success_x_m": 2.60,
        "high_start_probe_x_m": -0.50,
        "low_probe_x_m": 0.50,
        "high_end_probe_x_m": 1.50,
    }
    assert ordinary["_runtime_patches"]() == (
        ("FrictionHighStart", 0.90, -0.50, 0),
        ("FrictionLow", 0.28, 0.50, 1),
        ("FrictionHighEnd", 0.90, 1.50, 2),
    )

    short_cadence = _load_course_geometry_helpers(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride"
    )
    assert short_cadence["_runtime_course_geometry"]()["long_course"] is False

    long_demo = _load_course_geometry_helpers(
        "Unitree-G1-29dof-Velocity-Foot-"
        "TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo"
    )
    geometry = long_demo["_runtime_course_geometry"]()
    assert geometry["long_course"] is True
    assert geometry["low_start_x_m"] == 0.0
    assert geometry["low_end_x_m"] == 6.0
    assert geometry["success_x_m"] == 17.5
    assert long_demo["_runtime_patches"]() == (
        ("FrictionHighStart", 0.90, -3.0, 0),
        ("FrictionLow", 0.28, 3.0, 1),
        ("FrictionHighEnd", 0.90, 12.0, 2),
    )
    stages = long_demo["_course_stage_from_local_x"](
        torch.tensor([-0.1, 0.0, 5.99, 6.0])
    )
    assert torch.equal(stages, torch.tensor([0, 1, 1, 2]))


def test_eval_runtime_checks_contact_hall_privileged_label_and_actor_boundary():
    source = EVAL_PATH.read_text(encoding="utf-8")
    for token in (
        "sensor.contact_physx_view.filter_count != 3",
        "expected_shape = (base_env.num_envs, 1, 3, 3)",
        "expected_hall_shape = (base_env.num_envs, 2, 15, 3)",
        "torch.isfinite(raw).all()",
        'hasattr(base_env, "spatial_low_contact_buf")',
        'compressed != ["HIGH", "LOW", "HIGH"]',
        "FORBIDDEN_POLICY_TOKENS",
        "policy_dim != 1864",
        "EXPECTED_POLICY_FUNCTIONS",
        "EXPECTED_POLICY_TERM_DIMS",
        "EXPECTED_POLICY_HISTORY_LENGTHS",
        "EXPECTED_POLICY_SLICES",
        '"lateral_motion_feedback"',
        '"motion feedback [body_vy,relative_heading]"',
        '"latest Hall frame [left,right,P00..P14,XYZ]"',
        '"latest Hall sample period [left,right]"',
        '"Hall valid [left,right]"',
        "history_buffers[term].buffer.reshape",
        "preserve_order",
        "EXPECTED_JOINT_IDS_MAP",
        'action_terms != ("JointPositionAction",)',
        "action_term.action_dim) != 29",
        "JointPositionAction scale must be 0.25",
        "actor_to_sdk_joint_ids_map",
        '"hall_sensor_rng_seed"',
        '"hall_randomization_probe"',
        "advance_high_low_high_stage",
        "done=bool(dones[env_id].item())",
    ):
        assert token in source


def test_actor_abi_constants_cover_exact_1864_term_major_layout():
    assert _literal_assignment(EVAL_PATH, "EXPECTED_POLICY_FUNCTIONS") == (
        "base_ang_vel",
        "projected_gravity",
        "generated_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "hall_magnetic_array",
        "hall_sample_period_lr",
        "hall_sensor_valid_lr",
        "lateral_motion_feedback",
    )
    assert _literal_assignment(EVAL_PATH, "EXPECTED_POLICY_TERM_DIMS") == (
        (15,),
        (15,),
        (15,),
        (145,),
        (145,),
        (145,),
        (1350,),
        (30,),
        (2,),
        (2,),
    )
    assert _literal_assignment(EVAL_PATH, "EXPECTED_POLICY_HISTORY_LENGTHS") == (
        5,
        5,
        5,
        5,
        5,
        5,
        15,
        15,
        0,
        0,
    )
    slices = _literal_assignment(EVAL_PATH, "EXPECTED_POLICY_SLICES")
    assert slices == (
        (0, 15),
        (15, 30),
        (30, 45),
        (45, 190),
        (190, 335),
        (335, 480),
        (480, 1830),
        (1830, 1860),
        (1860, 1862),
        (1862, 1864),
    )
    assert slices[0][0] == 0
    assert slices[-1][1] == 1864
    assert all(left[1] == right[0] for left, right in zip(slices, slices[1:]))
    dims = _literal_assignment(EVAL_PATH, "EXPECTED_POLICY_TERM_DIMS")
    assert tuple(stop - start for start, stop in slices) == tuple(
        math.prod(shape) for shape in dims
    )
    joint_map = _literal_assignment(EVAL_PATH, "EXPECTED_JOINT_IDS_MAP")
    assert joint_map == (
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5,
        11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    )
    assert sorted(joint_map) == list(range(29))


def test_eval_cli_supports_native_exported_policies_video_and_artifacts():
    source = EVAL_PATH.read_text(encoding="utf-8")
    for flag in (
        '"--onnx"',
        '"--torchscript"',
        '"--video"',
        '"--video_dir"',
        '"--summary_json"',
        '"--trace_npz"',
        '"--require_rollout_hlh"',
        '"--skip_label_probe"',
        '"--expected_low_mu"',
        '"--low_speed_target"',
        '"--high_recovery_speed"',
    ):
        assert flag in source
    # --checkpoint is supplied by the shared RSL-RL argument group.
    assert "cli_args.add_rsl_rl_args(parser)" in source
    assert "args_cli.checkpoint" in source
    assert "gym.wrappers.RecordVideo" in source
    assert "np.savez_compressed" in source
    assert "analyze_transition_response" in source
    assert '"first_episode_only": True' in source


def test_hardened_hall_really_enables_randomization_and_is_audited():
    source = EVAL_PATH.read_text(encoding="utf-8")
    assert "hall_cfg.enable_domain_randomization = True" in source
    assert '"domain_randomization_enabled"' in source
    assert '"foot_dropout_probability"' in source
    assert '"dead_channel_probability"' in source
    assert '"maximum_packet_delay_steps"' in source
    assert 'args_cli.task.endswith("SpatialFrictionMild")' in source
    assert 'args_cli.task.endswith("SpatialFrictionMedium")' in source


def test_eval_causal_hybrid_is_hall_only_and_bounded():
    source = EVAL_PATH.read_text(encoding="utf-8")
    for token in (
        '"--hybrid_baseline_onnx"',
        '"--hybrid_recovery_onnx"',
        '"--hall_risk_checkpoint"',
        '"--hybrid_on_steps"',
        '"--hybrid_off_steps"',
        '"--hybrid_max_active_steps"',
        "class _CausalHallHybridPolicy",
        "VALID_SLICE",
        "~healthy",
        "_with_recovery_command_history(",
    ):
        assert token in source


def test_recovery_command_rewrite_is_term_major_and_preserves_proprioception():
    namespace = _load_eval_command_rewrite_helper()
    rewrite = namespace["_with_recovery_command_history"]
    vx = namespace["RECOVERY_COMMAND_VX_INDICES"]
    vy = namespace["RECOVERY_COMMAND_VY_INDICES"]
    yaw = namespace["RECOVERY_COMMAND_YAW_INDICES"]
    assert vx == (30, 33, 36, 39, 42)
    assert vy == (31, 34, 37, 40, 43)
    assert yaw == (32, 35, 38, 41, 44)

    original = torch.arange(2 * 1864, dtype=torch.float32).reshape(2, 1864)
    before = original.clone()
    rewritten = rewrite(original, 0.16)

    assert torch.equal(original, before), "helper mutated the baseline observation"
    assert rewritten.data_ptr() != original.data_ptr()
    expected_vx = torch.full_like(rewritten[:, list(vx)], 0.16)
    torch.testing.assert_close(
        rewritten[:, list(vx)], expected_vx, rtol=0.0, atol=1.0e-7
    )
    assert torch.count_nonzero(rewritten[:, list(vy)]) == 0
    assert torch.count_nonzero(rewritten[:, list(yaw)]) == 0

    command_columns = set(vx) | set(vy) | set(yaw)
    assert command_columns == set(range(30, 45))
    untouched = [index for index in range(1864) if index not in command_columns]
    assert torch.equal(rewritten[:, untouched], original[:, untouched])
    # Regression for the old frame-major bug: every formerly targeted body or
    # joint-history column must now remain bit-identical.
    assert torch.equal(
        rewritten[:, [6, 102, 198, 294, 390]],
        original[:, [6, 102, 198, 294, 390]],
    )

    with pytest.raises(ValueError, match=r"\[N,1864\]"):
        rewrite(torch.zeros(2, 480), 0.16)
    with pytest.raises(ValueError, match="finite"):
        rewrite(torch.zeros(2, 1864), float("nan"))


def test_scene_uses_static_assetbase_cuboids_and_multiply_materials():
    source = CFG_PATH.read_text(encoding="utf-8")
    for token in (
        "def _friction_patch_cfg(",
        "AssetBaseCfg(",
        "sim_utils.CuboidCfg(",
        "collision_props=sim_utils.CollisionPropertiesCfg(",
        'friction_combine_mode="multiply"',
        "collision_group=0",
        '"FrictionHighStart"',
        '"FrictionLow"',
        '"FrictionHighEnd"',
    ):
        assert token in source
    # Static AssetBase ground must not accidentally acquire rigid-body/mass
    # configuration in the helper block.
    helper = source[source.index("def _friction_patch_cfg(") : source.index("@configclass\nclass HallSpatialFrictionSceneCfg")]
    assert "RigidBodyCfg(" not in helper
    assert "MassPropertiesCfg(" not in helper
