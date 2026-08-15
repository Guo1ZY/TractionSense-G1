"""Canonical traction-adaptive locomotion interfaces.

This package is deliberately independent from Isaac Lab.  Isaac, MuJoCo,
offline replay, policy export, and future robot-side adapters import the same
schemas and stateful preprocessing implementations from here.
"""

from .ble import BleFrameParser, FootSensorFrame, Int16Unwrapper
from .diagnostics import (
    TractionDiagnostics,
    TractionDiagnosticsCfg,
    TractionDiagnosticsState,
)
from .deployment import (
    CanonicalObservationBuilder,
    DeploymentObservationCfg,
    IsaacForceAdapter,
    OfflineRecordedForceAdapter,
    PolicyRuntimeOutput,
    ProprioceptiveState,
    TractionPolicyRuntime,
)
from .history import TemporalHistoryBuffer
from .forward_velocity_estimator import (
    ForwardVelocityEstimator,
    NormalizedForwardVelocityEstimator,
    build_forward_velocity_estimator,
)
from .governor import (
    GovernorOutput,
    TractionAwareCommandGovernor,
    TractionAwareCommandGovernorCfg,
)
from .health_envelope import (
    HallHealthEnvelope,
    HealthEnvelope,
    HealthEnvelopeCfg,
    HealthEnvelopeOutput,
    rewrite_command_history,
    summarize_health_envelope_trace,
)
from .high_speed_stability_envelope import (
    HighSpeedStabilityEnvelope,
    HighSpeedStabilityEnvelopeCfg,
    HighSpeedStabilityEnvelopeOutput,
    summarize_high_speed_stability_trace,
)
from .stability_recovery_blend import (
    FrozenStage7RecoveryActor,
    StabilityRecoveryBlend,
    StabilityRecoveryBlendCfg,
    StabilityRecoveryBlendOutput,
    rewrite_recovery_observation,
)
from .networks import (
    DistillationLossCfg,
    GatedTractionPolicy,
    LegacyLocomotionActor,
    PrivilegedTractionEncoder,
    PrivilegedTractionEncoderCfg,
    StudentEncoderOutput,
    StudentPolicyOutput,
    TeacherPolicyOutput,
    TeacherTractionPolicy,
    TemporalStudentEncoderCfg,
    TemporalTactileProprioceptiveStudentEncoder,
    teacher_student_loss,
    temporal_history_to_legacy_proprio,
)
from .schema import (
    ACTION_DIM,
    FORCE_FRAME,
    FORCE_ORDER,
    G1_29DOF_JOINT_ORDER,
    POLICY_DT_S,
    PRIVILEGED_TRACTION_SCHEMA,
    TEMPORAL_STUDENT_FRAME_SCHEMA,
    legacy_actor_schema,
    legacy_critic_schema,
    old_to_new_flat_index,
)
from .sensor_layout import (
    PROVISIONAL_NORMALIZED_LAYOUT,
    DualFootForceInput,
    DualFootMagneticInput,
    DualFootSensorAggregator,
    FootSensorLayoutCfg,
    SingleFootSensorAdapter,
)
from .tactile import (
    TactileDomainRandomizationCfg,
    TactileObservation,
    TactileObservationModel,
)

__all__ = [
    "ACTION_DIM",
    "BleFrameParser",
    "CanonicalObservationBuilder",
    "DualFootForceInput",
    "DualFootMagneticInput",
    "DualFootSensorAggregator",
    "FORCE_FRAME",
    "FORCE_ORDER",
    "ForwardVelocityEstimator",
    "FootSensorFrame",
    "FootSensorLayoutCfg",
    "GatedTractionPolicy",
    "G1_29DOF_JOINT_ORDER",
    "HallHealthEnvelope",
    "HealthEnvelope",
    "HealthEnvelopeCfg",
    "HealthEnvelopeOutput",
    "HighSpeedStabilityEnvelope",
    "HighSpeedStabilityEnvelopeCfg",
    "HighSpeedStabilityEnvelopeOutput",
    "FrozenStage7RecoveryActor",
    "Int16Unwrapper",
    "IsaacForceAdapter",
    "LegacyLocomotionActor",
    "POLICY_DT_S",
    "PRIVILEGED_TRACTION_SCHEMA",
    "PROVISIONAL_NORMALIZED_LAYOUT",
    "PrivilegedTractionEncoder",
    "PrivilegedTractionEncoderCfg",
    "SingleFootSensorAdapter",
    "OfflineRecordedForceAdapter",
    "NormalizedForwardVelocityEstimator",
    "PolicyRuntimeOutput",
    "ProprioceptiveState",
    "TactileDomainRandomizationCfg",
    "TactileObservation",
    "TactileObservationModel",
    "TEMPORAL_STUDENT_FRAME_SCHEMA",
    "TemporalHistoryBuffer",
    "TemporalStudentEncoderCfg",
    "TemporalTactileProprioceptiveStudentEncoder",
    "TractionAwareCommandGovernor",
    "TractionAwareCommandGovernorCfg",
    "TractionDiagnostics",
    "TractionDiagnosticsCfg",
    "TractionDiagnosticsState",
    "TractionPolicyRuntime",
    "DeploymentObservationCfg",
    "DistillationLossCfg",
    "GovernorOutput",
    "StudentEncoderOutput",
    "StudentPolicyOutput",
    "TeacherPolicyOutput",
    "TeacherTractionPolicy",
    "StabilityRecoveryBlend",
    "StabilityRecoveryBlendCfg",
    "StabilityRecoveryBlendOutput",
    "legacy_actor_schema",
    "legacy_critic_schema",
    "old_to_new_flat_index",
    "rewrite_command_history",
    "rewrite_recovery_observation",
    "summarize_health_envelope_trace",
    "summarize_high_speed_stability_trace",
    "teacher_student_loss",
    "temporal_history_to_legacy_proprio",
    "build_forward_velocity_estimator",
]
