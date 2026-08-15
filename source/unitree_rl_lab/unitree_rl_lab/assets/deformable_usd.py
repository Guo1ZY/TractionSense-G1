"""Current Isaac Sim 5.1 spawner for a deformable mesh stored in USD.

Isaac Lab 2.3.2's :class:`UsdFileCfg` can modify an existing deformable API,
but a generic mesh-converter USD has neither that API nor a deformable material
field.  This small version-local wrapper defines both after referencing the
mesh.  It intentionally uses the current PhysX deformable schema and contains
no legacy ``isaacsim.core.api`` calls.
"""

from __future__ import annotations

from collections.abc import Callable

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils import configclass


@clone
def spawn_deformable_from_usd(
    prim_path: str,
    cfg: "DeformableUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a one-mesh USD, define the soft body, and bind its material."""
    del kwargs
    # The referenced converter USD is a plain mesh, so the generic spawner
    # cannot "modify" a deformable API yet.  Suppress that invalid pre-pass;
    # the API is defined exactly once immediately below.
    reference_cfg = cfg.replace(deformable_props=None)
    prim = from_files._spawn_from_usd_file(
        prim_path,
        cfg.usd_path,
        reference_cfg,
        translation,
        orientation,
    )
    if cfg.deformable_props is None:
        raise ValueError("DeformableUsdFileCfg.deformable_props cannot be None")
    if cfg.physics_material is None:
        raise ValueError("DeformableUsdFileCfg.physics_material cannot be None")
    schemas.define_deformable_body_properties(prim_path, cfg.deformable_props)
    if cfg.physics_material_path.startswith("/"):
        material_path = cfg.physics_material_path
    else:
        material_path = f"{prim_path}/{cfg.physics_material_path}"
    cfg.physics_material.func(material_path, cfg.physics_material)
    bind_physics_material(prim_path, material_path)
    return prim


@configclass
class DeformableUsdFileCfg(sim_utils.UsdFileCfg):
    """USD-file config extended with a deformable-body material."""

    func: Callable = spawn_deformable_from_usd
    physics_material_path: str = "deformableMaterial"
    physics_material: sim_utils.DeformableBodyMaterialCfg | None = None
