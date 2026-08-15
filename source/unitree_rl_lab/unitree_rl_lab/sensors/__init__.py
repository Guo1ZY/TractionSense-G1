"""Custom vectorized sensors used by Unitree RL Lab."""

from .hall_contact_distribution import (
    distribute_point_forces_to_hall_sites,
    indexed_buffer_indices,
    sum_vectors_by_index,
)
from .hall_foot_sensor import (
    DeformableMagnetPoseSample,
    DeformableNodeMagnetPoseProvider,
    HallFootSensor,
    MagnetPoseProvider,
)
from .hall_sensor_config import (
    DEFAULT_HALL_AXIS_YAW_DEG,
    DEFAULT_HALL_POSITIONS_IMAGE_PX,
    DEFAULT_HALL_POSITIONS_NORMALIZED,
    HALL_LAYOUT_IMAGE_SIZE_PX,
    HALL_LAYOUT_SOLE_BOUNDS_PX,
    HALL_LAYOUT_SOURCE_IMAGE,
    HALL_POLICY_OBSERVATION_TERM_NAMES,
    HallFootSensorCfg,
    audit_hall_sensor_cfg_policy_terms,
    sync_hall_sensor_cfg_to_policy_terms,
)
from .magnetic_field_model import (
    CalibratedMagneticFieldModel,
    DipoleMagneticFieldModel,
    MagneticFieldModel,
)

__all__ = [
    "CalibratedMagneticFieldModel",
    "DEFAULT_HALL_AXIS_YAW_DEG",
    "DEFAULT_HALL_POSITIONS_IMAGE_PX",
    "DEFAULT_HALL_POSITIONS_NORMALIZED",
    "DeformableMagnetPoseSample",
    "DeformableNodeMagnetPoseProvider",
    "DipoleMagneticFieldModel",
    "distribute_point_forces_to_hall_sites",
    "HallFootSensor",
    "HallFootSensorCfg",
    "HALL_LAYOUT_IMAGE_SIZE_PX",
    "HALL_LAYOUT_SOLE_BOUNDS_PX",
    "HALL_LAYOUT_SOURCE_IMAGE",
    "HALL_POLICY_OBSERVATION_TERM_NAMES",
    "MagneticFieldModel",
    "MagnetPoseProvider",
    "audit_hall_sensor_cfg_policy_terms",
    "indexed_buffer_indices",
    "sum_vectors_by_index",
    "sync_hall_sensor_cfg_to_policy_terms",
]
