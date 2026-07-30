"""Sagittal-plane symmetry transforms for the Unitree G1 29-DoF policy.

The observation layout matches the 641-D traction Teacher:

* 480-D proprioceptive prefix (five-frame histories)
* five 15-frame, two-foot scalar histories
* two five-frame scalar sensor-health histories
* one privileged friction scalar

The transform returns the original samples first and their left/right mirrors
second, as required by the RSL-RL symmetry extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TEACHER_OBSERVATION_DIM = 641
JOINT_COUNT = 29
PROPRIO_HISTORY = 5
FOOT_HISTORY = 15
FOOT_COUNT = 2

# Isaac articulation order, not the Unitree SDK order.
JOINT_MIRROR_INDEX = (
    2, 3, 0, 1, 4, 26, 8, 9, 6, 7, 10, 20, 14, 15, 12,
    13, 25, 27, 28, 23, 11, 24, 22, 19, 21, 16, 5, 17, 18,
)
JOINT_MIRROR_SIGN = (
    1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1,
    -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1,
)


def mirror_g1_29dof_joints(values: torch.Tensor) -> torch.Tensor:
    """Mirror joint vectors expressed in Isaac articulation order."""

    index = torch.as_tensor(JOINT_MIRROR_INDEX, device=values.device)
    sign = values.new_tensor(JOINT_MIRROR_SIGN)
    return values.index_select(-1, index) * sign


def mirror_traction_teacher_observation(observation: torch.Tensor) -> torch.Tensor:
    """Reflect an ``N x 641`` traction-Teacher observation left-to-right."""

    if observation.ndim != 2 or observation.shape[1] != TEACHER_OBSERVATION_DIM:
        raise ValueError(
            f"expected Nx{TEACHER_OBSERVATION_DIM}, got {tuple(observation.shape)}"
        )

    mirrored = observation.clone()

    # Angular velocity is an axial vector; gravity and linear commands are
    # polar vectors under reflection across the sagittal (x-z) plane.
    mirrored[:, 0:15] = (
        observation[:, 0:15].reshape(-1, PROPRIO_HISTORY, 3)
        * observation.new_tensor((-1.0, 1.0, -1.0))
    ).reshape(-1, 15)
    mirrored[:, 15:30] = (
        observation[:, 15:30].reshape(-1, PROPRIO_HISTORY, 3)
        * observation.new_tensor((1.0, -1.0, 1.0))
    ).reshape(-1, 15)
    mirrored[:, 30:45] = (
        observation[:, 30:45].reshape(-1, PROPRIO_HISTORY, 3)
        * observation.new_tensor((1.0, -1.0, -1.0))
    ).reshape(-1, 15)

    for start in (45, 190, 335):
        history = observation[
            :, start : start + PROPRIO_HISTORY * JOINT_COUNT
        ].reshape(-1, PROPRIO_HISTORY, JOINT_COUNT)
        mirrored[:, start : start + PROPRIO_HISTORY * JOINT_COUNT] = (
            mirror_g1_29dof_joints(history).reshape(
                -1, PROPRIO_HISTORY * JOINT_COUNT
            )
        )

    # Contact, Fn, Ft magnitude, friction ratio, and load are foot scalars.
    for start in (480, 510, 540, 570, 600):
        feet = observation[
            :, start : start + FOOT_HISTORY * FOOT_COUNT
        ].reshape(-1, FOOT_HISTORY, FOOT_COUNT)
        mirrored[:, start : start + FOOT_HISTORY * FOOT_COUNT] = (
            feet[:, :, (1, 0)].reshape(-1, FOOT_HISTORY * FOOT_COUNT)
        )

    # Sensor-validity/age histories (630:640) are already reduced to global
    # scalars. The final effective-mu scalar is invariant.
    return mirrored


def mirror_traction_motion_teacher_observation(
    observation: torch.Tensor,
) -> torch.Tensor:
    """Mirror the motion-feedback variant of the 641-D Teacher observation."""

    mirrored = mirror_traction_teacher_observation(observation)
    # Five frames of [body vy, relative heading error] replace the invariant
    # sensor-validity/age histories. Both terms change sign under reflection.
    mirrored[:, 630:640] = -observation[:, 630:640]
    return mirrored


@torch.no_grad()
def compute_g1_29dof_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and sagittal-mirrored policy samples for RSL-RL."""

    del env
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = mirror_traction_teacher_observation(
            obs["policy"]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            batch_size * 2,
            actions.shape[1],
            device=actions.device,
            dtype=actions.dtype,
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = mirror_g1_29dof_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


@torch.no_grad()
def compute_g1_29dof_motion_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Symmetry augmentation for the motion-feedback traction Teacher."""

    del env
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = (
            mirror_traction_motion_teacher_observation(obs["policy"])
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty(
            batch_size * 2,
            actions.shape[1],
            device=actions.device,
            dtype=actions.dtype,
        )
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = mirror_g1_29dof_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
