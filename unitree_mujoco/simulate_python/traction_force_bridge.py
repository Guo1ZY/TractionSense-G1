"""Canonical signed local-foot force bridge for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FORCE_ORDER = ("L_Fx", "L_Fy", "L_Fz", "R_Fx", "R_Fy", "R_Fz")
FORCE_FRAME = "matching_ankle_roll_link_local"


@dataclass(frozen=True)
class FootContactDiagnostics:
    force_local_n: np.ndarray
    contact_count: tuple[int, int]


class MujocoFootForceBridge:
    """Aggregate ground contacts and transform world force to ankle local.

    MuJoCo's contact frame rows are world-space contact axes.
    ``mj_contactForce`` returns the wrench applied to geom2 by geom1. Thus
    ``world_force = contact.frame.T @ contact_force[:3]`` and the sign is
    positive for a foot on geom2, negative for a foot on geom1.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        left_body_name: str = "left_ankle_roll_link",
        right_body_name: str = "right_ankle_roll_link",
        ground_body_ids: tuple[int, ...] = (0,),
    ) -> None:
        self.model = model
        self.foot_body_ids = (
            self._body_id(left_body_name),
            self._body_id(right_body_name),
        )
        self.ground_body_ids = frozenset(ground_body_ids)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo body {name!r} was not found")
        return body_id

    def _is_descendant(self, body_id: int, root_id: int) -> bool:
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return body_id == root_id

    @staticmethod
    def contact_force_world(
        contact: mujoco.MjContact,
        contact_force: np.ndarray,
    ) -> np.ndarray:
        frame = np.asarray(contact.frame).reshape(3, 3)
        return frame.T @ np.asarray(contact_force[:3])

    @staticmethod
    def world_to_body_local(
        data: mujoco.MjData,
        body_id: int,
        force_world: np.ndarray,
    ) -> np.ndarray:
        local_to_world = np.asarray(data.xmat[body_id]).reshape(3, 3)
        return local_to_world.T @ force_world

    def read(
        self,
        data: mujoco.MjData,
    ) -> FootContactDiagnostics:
        force_world = np.zeros((2, 3), dtype=np.float64)
        count = [0, 0]
        wrench = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1_body = int(self.model.geom_bodyid[contact.geom[0]])
            geom2_body = int(self.model.geom_bodyid[contact.geom[1]])
            for foot_index, foot_root in enumerate(self.foot_body_ids):
                geom1_is_foot = self._is_descendant(geom1_body, foot_root)
                geom2_is_foot = self._is_descendant(geom2_body, foot_root)
                if geom1_is_foot == geom2_is_foot:
                    continue
                other_body = geom2_body if geom1_is_foot else geom1_body
                if other_body not in self.ground_body_ids:
                    continue
                wrench.fill(0.0)
                mujoco.mj_contactForce(
                    self.model,
                    data,
                    contact_index,
                    wrench,
                )
                on_geom2_world = self.contact_force_world(contact, wrench)
                force_world[foot_index] += (
                    on_geom2_world if geom2_is_foot else -on_geom2_world
                )
                count[foot_index] += 1
        force_local = np.stack(
            (
                self.world_to_body_local(
                    data,
                    self.foot_body_ids[0],
                    force_world[0],
                ),
                self.world_to_body_local(
                    data,
                    self.foot_body_ids[1],
                    force_world[1],
                ),
            )
        )
        force_local = np.nan_to_num(force_local).astype(np.float32)
        return FootContactDiagnostics(
            force_local.reshape(6),
            (count[0], count[1]),
        )
