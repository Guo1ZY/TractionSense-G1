"""MuJoCo contact truth for metrics only; never imported by the estimator."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from traction_force_bridge import MujocoFootForceBridge


@dataclass(frozen=True)
class MujocoTorqueContactTruth:
    force_local_n: np.ndarray
    contact_count: np.ndarray
    contact_point_slip_speed_m_s: np.ndarray


class MujocoTorqueTruthBridge:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.force_bridge = MujocoFootForceBridge(model)

    def read(self, data: mujoco.MjData) -> MujocoTorqueContactTruth:
        force = self.force_bridge.read(data)
        slip = np.zeros(2, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            bodies = tuple(int(self.model.geom_bodyid[contact.geom[index]]) for index in (0, 1))
            for foot, root in enumerate(self.force_bridge.foot_body_ids):
                foot_side = 0 if self.force_bridge._is_descendant(bodies[0], root) else (1 if self.force_bridge._is_descendant(bodies[1], root) else -1)
                if foot_side < 0:
                    continue
                other = bodies[1 - foot_side]
                if other not in self.force_bridge.ground_body_ids:
                    continue
                jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jacp, jacr, contact.pos, bodies[foot_side])
                foot_velocity = jacp @ data.qvel
                other_velocity = np.zeros(3)
                if other != 0:
                    mujoco.mj_jac(self.model, data, jacp, jacr, contact.pos, other)
                    other_velocity = jacp @ data.qvel
                relative = foot_velocity - other_velocity
                frame = np.asarray(contact.frame).reshape(3, 3)
                tangent = frame[1:] @ relative
                slip[foot] = max(slip[foot], float(np.linalg.norm(tangent)))
        return MujocoTorqueContactTruth(force.force_local_n, np.asarray(force.contact_count), slip.astype(np.float32))

