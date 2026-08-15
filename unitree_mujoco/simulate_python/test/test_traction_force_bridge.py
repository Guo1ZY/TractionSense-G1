from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMULATE_PYTHON = ROOT / "simulate_python"
if str(SIMULATE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SIMULATE_PYTHON))

from traction_force_bridge import MujocoFootForceBridge  # noqa: E402


TWO_FEET_XML = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="left_ankle_roll_link" pos="-0.2 0 0.1">
      <freejoint/>
      <geom name="left_foot" type="box" size=".1 .08 .1" mass="1"/>
    </body>
    <body name="right_ankle_roll_link" pos="0.2 0 0.1">
      <freejoint/>
      <geom name="right_foot" type="box" size=".1 .08 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_two_grounded_feet_have_signed_upward_local_force() -> None:
    model = mujoco.MjModel.from_xml_string(TWO_FEET_XML)
    data = mujoco.MjData(model)
    bridge = MujocoFootForceBridge(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    sample = bridge.read(data)
    force = sample.force_local_n.reshape(2, 3)
    assert sample.contact_count == (4, 4)
    assert np.allclose(force[:, :2], 0.0, atol=1.0e-5)
    assert np.allclose(force[:, 2], 9.81, rtol=1.0e-4)


def test_airborne_feet_return_zero_without_nonfinite_values() -> None:
    model = mujoco.MjModel.from_xml_string(TWO_FEET_XML)
    data = mujoco.MjData(model)
    data.qpos[2] = 2.0
    data.qpos[9] = 2.0
    mujoco.mj_forward(model, data)
    sample = MujocoFootForceBridge(model).read(data)
    assert sample.contact_count == (0, 0)
    assert np.array_equal(sample.force_local_n, np.zeros(6, dtype=np.float32))
    assert np.isfinite(sample.force_local_n).all()


def test_contact_frame_transpose_matches_mujoco_cfrc_external_force() -> None:
    model = mujoco.MjModel.from_xml_string(TWO_FEET_XML)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    bridge = MujocoFootForceBridge(model)
    sample = bridge.read(data)
    left_body = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "left_ankle_roll_link",
    )
    local_from_cfrc = (
        data.xmat[left_body].reshape(3, 3).T
        @ data.cfrc_ext[left_body, 3:6]
    )
    assert np.allclose(sample.force_local_n[:3], local_from_cfrc, atol=1.0e-5)


def test_actual_g1_body_names_and_canonical_output_shape() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "unitree_robots/g1/scene_29dof.xml")
    )
    data = mujoco.MjData(model)
    bridge = MujocoFootForceBridge(model)
    mujoco.mj_forward(model, data)
    sample = bridge.read(data)
    assert sample.force_local_n.shape == (6,)
    assert sample.force_local_n.dtype == np.float32
    assert np.isfinite(sample.force_local_n).all()
