import numpy as np
import math
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    """Convert a config tree to YAML-safe primitives.

    Isaac Lab config dictionaries contain tuples, NumPy scalars and
    ``slice(None)`` selectors.  ``yaml.dump`` serializes those with Python
    object tags which yaml-cpp cannot consume and which ``yaml.safe_load``
    deliberately rejects.  A full slice means "all indices" throughout the
    deploy runtime and is represented by an empty sequence.  Any partial
    slice is ambiguous without the source dimension, so fail closed instead
    of silently treating its bounds as joint indices.
    """

    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, np.generic):
        return format_value(x.item())
    if isinstance(x, np.ndarray):
        return format_value(x.tolist())
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"non-finite numeric value in deploy config: {x!r}")
        return float(f"{x:.3g}")
    if isinstance(x, slice):
        if x == slice(None):
            return []
        raise ValueError(
            "partial slice selectors cannot be exported safely without an "
            f"explicit source dimension: {x!r}"
        )
    if isinstance(x, (list, tuple)):
        return [format_value(i) for i in x]
    if isinstance(x, dict):
        if not all(isinstance(key, str) for key in x):
            raise TypeError("deploy YAML mapping keys must all be strings")
        return {k: format_value(v) for k, v in x.items()}
    raise TypeError(f"unsupported deploy YAML value type {type(x).__name__}: {x!r}")


def deploy_observation_name(config_name: str, func) -> str:
    """Return the runtime observation registry name for deployment.

    Isaac Lab uses the config attribute as the manager term name.  The Motion
    Hall task deliberately retained the legacy ``foot_sensor_age_lr``
    attribute to keep the 1864-D checkpoint column order stable, while its
    callable now supplies ``[body_vy, relative_heading]``.  Exporting the
    attribute name would silently make the C++ runtime feed packet age into a
    motion checkpoint.  Resolve this one compatibility alias by callable
    semantics and fail closed if a future alias is ambiguous.
    """

    function_name = str(getattr(func, "__name__", type(func).__name__))
    if config_name == "foot_sensor_age_lr":
        if function_name == "lateral_motion_feedback":
            return "lateral_motion_feedback"
        if function_name not in ("hall_sensor_age_lr", "magnetic_sensor_age_lr"):
            raise ValueError(
                "foot_sensor_age_lr has unsupported deployment callable "
                f"{function_name!r}"
            )
    return config_name


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        deploy_name = deploy_observation_name(obs_name, obs_cfg.func)
        if deploy_name in cfg["observations"]:
            raise ValueError(f"duplicate deployment observation term {deploy_name!r}")
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][deploy_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    # ``safe_dump`` is intentional: generated deployment configs must be
    # consumable by both yaml.safe_load and yaml-cpp without Python tags.
    with open(filename, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
