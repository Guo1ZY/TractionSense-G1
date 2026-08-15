"""Configuration and the single source of truth for the dual-foot Hall layout.

All dimensions are SI.  Normalized Hall coordinates use the canonical G1 foot
frame: ``+x`` points to the toe, ``+y`` points to the robot's left, and ``+z``
points upward.  Coordinates are centered and scaled by ``sole_length`` and
``sole_width``.  The order is toe-to-heel; sites on the same row are ordered
from the left side of the supplied image to the right side.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal


# Digitized from the supplied 1:1 A4 drawing
# /home/mosense/guo_1/vola_sensor/2.png (1280 x 1793 px).  Pixel coordinates
# are (u right, v heelward).  The ink outline spans u=7..508 and v=488..1780.
# Its centre maps to the foot-local origin; +x points upward/toe in the image
# and +y points toward image-left.  Every group is ordered top, left, centre,
# right, bottom, while the three groups run forefoot, midfoot, heel.
HALL_LAYOUT_SOURCE_IMAGE: str = "/home/mosense/guo_1/vola_sensor/2.png"
HALL_LAYOUT_IMAGE_SIZE_PX: tuple[int, int] = (1280, 1793)
HALL_LAYOUT_SOLE_BOUNDS_PX: tuple[float, float, float, float] = (7.0, 488.0, 508.0, 1780.0)
DEFAULT_HALL_POSITIONS_IMAGE_PX: tuple[tuple[float, float], ...] = (
    (260.0, 653.0),   # P00 forefoot top
    (208.5, 700.0),   # P01 forefoot image-left
    (259.0, 703.0),   # P02 forefoot centre
    (310.5, 701.5),   # P03 forefoot image-right
    (260.0, 750.5),   # P04 forefoot bottom
    (264.0, 1076.0),  # P05 midfoot top
    (209.5, 1123.0),  # P06 midfoot image-left
    (262.0, 1125.0),  # P07 midfoot centre
    (316.0, 1121.5),  # P08 midfoot image-right
    (263.0, 1171.5),  # P09 midfoot bottom
    (259.0, 1497.5),  # P10 heel top
    (206.0, 1546.0),  # P11 heel image-left
    (260.5, 1547.0),  # P12 heel centre
    (309.0, 1546.5),  # P13 heel image-right
    (256.0, 1593.0),  # P14 heel bottom
)


def _image_pixels_to_normalized_foot(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    u_min, v_min, u_max, v_max = HALL_LAYOUT_SOLE_BOUNDS_PX
    u_origin = 0.5 * (u_min + u_max)
    v_origin = 0.5 * (v_min + v_max)
    width = u_max - u_min
    length = v_max - v_min
    return tuple(((v_origin - v) / length, (u_origin - u) / width) for u, v in points)


DEFAULT_HALL_POSITIONS_NORMALIZED: tuple[tuple[float, float], ...] = (
    _image_pixels_to_normalized_foot(DEFAULT_HALL_POSITIONS_IMAGE_PX)
)

# Provisional chip-axis alignment already used by the real BLE/dashboard data
# path.  Zero means chip X/Y are aligned to foot X/Y.  Replace after a signed
# three-axis calibration if the PCB assembly revision differs.
DEFAULT_HALL_AXIS_YAW_DEG: tuple[float, ...] = (
    -90.0,
    -90.0,
    0.0,
    90.0,
    0.0,
    -90.0,
    -90.0,
    180.0,
    90.0,
    -90.0,
    0.0,
    -90.0,
    180.0,
    90.0,
    0.0,
)


# These are the only policy terms allowed to own a Hall forward-model
# configuration.  The Motion actor intentionally replaces ``foot_sensor_age_lr``
# with proprioceptive lateral-motion feedback, so a particular policy may have
# three rather than four Hall-configured terms.
HALL_POLICY_OBSERVATION_TERM_NAMES: tuple[str, ...] = (
    "foot_magnetic_array",
    "foot_sample_period_lr",
    "foot_sensor_valid_lr",
    "foot_sensor_age_lr",
)


def _hall_policy_terms_with_cfg(observations: Any) -> list[tuple[str, Any]]:
    """Return policy observation terms that explicitly accept ``hall_cfg``."""

    policy = getattr(observations, "policy", None)
    if policy is None:
        raise ValueError("observations must contain a policy observation group")
    terms: list[tuple[str, Any]] = []
    for name in HALL_POLICY_OBSERVATION_TERM_NAMES:
        term = getattr(policy, name, None)
        params = getattr(term, "params", None) if term is not None else None
        if isinstance(params, dict) and "hall_cfg" in params:
            terms.append((name, term))
    return terms


def audit_hall_sensor_cfg_policy_terms(
    observations: Any,
    hall_sensor_cfg: "HallFootSensorCfg",
) -> tuple[str, ...]:
    """Verify value equality and independent ownership for every Hall term.

    Isaac Lab ``configclass`` deep-copies top-level fields after each inherited
    ``__post_init__``.  Consequently, relying on a shared mutable reference
    between ``env_cfg.hall_sensor_cfg`` and observation-term ``hall_cfg`` is
    unsafe.  This audit requires each Hall term to own an independent but
    value-identical copy.
    """

    terms = _hall_policy_terms_with_cfg(observations)
    if not terms:
        raise ValueError("policy contains no Hall observation term with a hall_cfg parameter")

    mismatched: list[str] = []
    shared: list[str] = []
    term_cfg_ids: list[int] = []
    for name, term in terms:
        term_cfg = term.params["hall_cfg"]
        if term_cfg != hall_sensor_cfg:
            mismatched.append(name)
        if term_cfg is hall_sensor_cfg:
            shared.append(name)
        term_cfg_ids.append(id(term_cfg))
    if len(set(term_cfg_ids)) != len(term_cfg_ids):
        shared.append("duplicate_term_hall_cfg")
    if mismatched or shared:
        raise RuntimeError(
            "Hall policy configuration audit failed: "
            f"value_mismatch={mismatched}, shared_mutable_cfg={shared}"
        )
    return tuple(name for name, _ in terms)


def sync_hall_sensor_cfg_to_policy_terms(
    observations: Any,
    hall_sensor_cfg: "HallFootSensorCfg",
) -> tuple[str, ...]:
    """Copy the environment Hall config into every Hall policy term and audit it.

    A separate deep copy is installed in each term.  Call this after the last
    environment-specific Hall override and again after evaluator overrides
    such as ``--nominal_hall``.
    """

    terms = _hall_policy_terms_with_cfg(observations)
    if not terms:
        raise ValueError("policy contains no Hall observation term with a hall_cfg parameter")
    for _, term in terms:
        term.params["hall_cfg"] = deepcopy(hall_sensor_cfg)
    return audit_hall_sensor_cfg_policy_terms(observations, hall_sensor_cfg)


@dataclass
class HallFootSensorCfg:
    """Configuration for two flexible magnetic soles.

    Scheme A (``approximate``) is the default vectorized Kelvin--Voigt local
    compliance model.  Scheme B (``deformable``) requires a current Isaac Lab
    deformable-node-to-magnet pose provider supplied at initialization.
    """

    enabled: bool = True
    implementation_mode: Literal["approximate", "deformable"] = "approximate"
    left_foot_prim_path: str = "{ENV_REGEX_NS}/Robot/left_ankle_roll_link"
    right_foot_prim_path: str = "{ENV_REGEX_NS}/Robot/right_ankle_roll_link"

    # Foot-local layout mapping.  There is deliberately no intermediate
    # connector/compliance layer in the physical stack:
    # rigid robot sole -> rigid PCB enclosure (PCB inside) -> magnetized TPU.
    # The 9.4 mm enclosure is rigid and the 10 mm magnetized layer is the only
    # deformable object.  Mechanical source envelope: 215.02 x 80.04 mm.
    sole_length: float = 0.21502
    sole_width: float = 0.08004
    sole_thickness: float = 0.01050
    pcb_enclosure_thickness: float = 0.00940
    sole_origin: tuple[float, float, float] = (0.035, 0.0, 0.0)
    sole_yaw_deg: float = 0.0
    right_sole_yaw_deg: float = 0.0
    mirror_right_y: bool = True
    right_hall_axis_sign: tuple[float, float, float] = (1.0, -1.0, 1.0)
    hall_positions_normalized: tuple[tuple[float, float], ...] = DEFAULT_HALL_POSITIONS_NORMALIZED
    hall_axis_yaw_deg: tuple[float, ...] = DEFAULT_HALL_AXIS_YAW_DEG
    hall_height: float = -0.0420
    # Square Hall package shown in the supplied A4 drawing.  This affects
    # debug geometry only; the magnetic sample remains at its centre point.
    hall_package_size: tuple[float, float, float] = (0.0040, 0.0040, 0.0010)

    # Four round magnets under every Hall IC. ``magnet_size`` is diameter.
    magnets_per_hall: int = 4
    magnet_layout: Literal["square_2x2"] = "square_2x2"
    magnet_spacing_x: float = 0.0060
    magnet_spacing_y: float = 0.0060
    magnet_size: float = 0.0040
    magnet_thickness: float = 0.0015
    magnetization_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    magnetic_moment: float = 0.0100
    remanence_strength: float = 0.80
    # Depth of each magnet centre below the TPU upper surface.  The complete
    # round disc must remain inside the TPU layer; magnets are embedded
    # inclusions, never independent contact bodies.
    magnet_embedding_depth: float = 0.0040
    initial_hall_magnet_distance: float = 0.0060
    dipole_min_distance: float = 5.0e-4

    # Magnetized TPU and Scheme-A local compliance.  The material is FusRock
    # TPU-Aero, nominal solid Shore A70, printed at 220--240 C with 35% Grid
    # infill.  Scheme B uses an effective homogeneous Shore-A40 equivalent
    # until compression-coupon/indentation data are available.  If the Grid
    # cells are modelled explicitly, use the solid-wall modulus instead.
    tpu_thickness: float = 0.0100
    tpu_effective_shore_a: float = 40.0
    tpu_youngs_modulus: float = 1.7e6
    tpu_solid_youngs_modulus: float = 5.5e6
    tpu_poisson_ratio: float = 0.30
    tpu_density: float = 400.0
    tpu_dynamic_friction: float = 0.80
    tpu_damping: float = 0.08
    tpu_contact_offset: float = 8.0e-4
    tpu_rest_offset: float = 0.0
    tpu_solver_position_iteration_count: int = 16
    tpu_simulation_hexahedral_resolution: int = 36
    tpu_self_collision: bool = False
    # Thickness of the TPU top face tied directly to the rigid PCB enclosure.
    # This is a boundary-condition selection depth, not a physical layer.
    tpu_top_anchor_depth: float = 1.2e-3
    tpu_top_anchor_grid_size: float = 6.0e-3
    deformable_embedding_neighbors: int = 8
    deformable_frame_sample_radius: float = 2.0e-3
    # Isaac Sim 5.1's automatic hexahedral cooker can thicken this very thin,
    # curved 10 mm volume.  Training must stop if the cooked volume no longer
    # represents the measured magnetized layer.  A GUI/diagnostic scene may
    # explicitly disable the strict check, but its results are not quantitative.
    deformable_strict_geometry_check: bool = True
    deformable_max_cooked_thickness_ratio: float = 1.35
    local_normal_stiffness: float = 1.5e5
    local_shear_stiffness: float = 5.0e4
    local_damping: float = 3.0e3
    # Scheme-A mechanical input.  ``aggregate`` preserves the legacy low-cost
    # total-force + mean-point Gaussian.  ``detailed`` reads current PhysX
    # normal contact patches and friction anchors and distributes each one at
    # its own position.  Neither mode exposes force to the actor.
    contact_distribution_mode: Literal["aggregate", "detailed"] = "aggregate"
    contact_spread_sigma: float = 0.035
    # Detailed mode audits the raw-buffer sums against ContactSensor's
    # pairwise aggregates before they can drive the magnetic model.
    detailed_contact_force_atol: float = 1.0e-3
    detailed_contact_force_rtol: float = 1.0e-4
    detailed_contact_fail_on_buffer_saturation: bool = True
    detailed_contact_fail_on_audit_mismatch: bool = True
    max_normal_compression: float = 0.0040
    max_shear_displacement: float = 0.0040
    max_local_rotation: float = 0.35
    # Aggregate-mode whole-foot threshold.  Detailed raw buffers already
    # encode contact existence and preserve sub-threshold point forces.
    contact_force_threshold: float = 1.0

    # Simulated Hall electronics.  Magnetic quantities use tesla.  The real
    # FootSensor15 wire protocol exposes raw counts instead; no counts/T or
    # Hall/force conversion is implied by these simulation parameters.
    sensor_sample_rate: float = 50.0
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise_std: tuple[float, float, float] = (2.0e-5, 2.0e-5, 2.0e-5)
    resolution: float = 1.0e-6
    saturation_min: tuple[float, float, float] = (-0.20, -0.20, -0.20)
    saturation_max: tuple[float, float, float] = (0.20, 0.20, 0.20)
    low_pass_cutoff: float = 20.0
    drift_std_per_sqrt_s: float = 2.0e-6
    drift_limit: float = 2.0e-4
    temperature_coefficient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reference_temperature_c: float = 25.0
    auto_zero: bool = True
    auto_zero_samples: int = 8
    output_raw: bool = True
    output_processed: bool = True

    # RL adapter: simulated APIs stay in tesla; only the observation is
    # normalized.  Real counts require their own measured per-channel
    # baseline/temperature/scale normalization before sharing this policy range.
    observation_scale_t: float = 0.010
    observation_clip: tuple[float, float] = (-6.0, 6.0)
    loading_history_length: int = 16

    # Episode-correlated Sim2Real randomization for an inaccurate flexible
    # magnetic sole.  These parameters alter only the mechanical/magnetic
    # forward path and the normalized Hall observation.  They never create a
    # force observation.  The core class defaults to deterministic behavior;
    # the training environment explicitly enables this block.
    enable_domain_randomization: bool = False
    normal_stiffness_scale_range: tuple[float, float] = (0.50, 1.80)
    shear_stiffness_scale_range: tuple[float, float] = (0.45, 2.00)
    damping_scale_range: tuple[float, float] = (0.55, 1.80)
    contact_spread_scale_range: tuple[float, float] = (0.65, 1.45)
    magnetic_moment_scale_range: tuple[float, float] = (0.60, 1.40)
    magnet_position_jitter_std: float = 4.0e-4
    observation_sensor_gain_range: tuple[float, float] = (0.65, 1.35)
    observation_axis_gain_range: tuple[float, float] = (0.70, 1.30)
    observation_cross_axis_std: float = 0.08
    observation_zero_residual_std: float = 0.08
    dead_channel_probability: float = 0.025
    foot_dropout_probability: float = 0.015
    reported_sample_period_range: tuple[float, float] = (0.0125, 0.0500)
    maximum_packet_delay_steps: int = 3

    enable_debug_vis: bool = False
    debug_vis_max_envs: int = 1
    debug_field_scale: float = 0.50  # rendered metres per tesla
    debug_field_arrow_width: float = 0.0020
    debug_field_arrow_min_length: float = 0.0030
    debug_field_arrow_max_length: float = 0.0500

    def __post_init__(self) -> None:
        if self.implementation_mode not in ("approximate", "deformable"):
            raise ValueError(f"unsupported implementation_mode={self.implementation_mode!r}")
        if self.magnets_per_hall != 4 or self.magnet_layout != "square_2x2":
            raise ValueError("the current physical generator supports exactly four magnets in square_2x2 layout")
        if not self.hall_positions_normalized:
            raise ValueError("hall_positions_normalized cannot be empty")
        if len(self.hall_axis_yaw_deg) not in (1, len(self.hall_positions_normalized)):
            raise ValueError("hall_axis_yaw_deg must have one value or one value per Hall site")
        if len(self.hall_package_size) != 3 or min(self.hall_package_size) <= 0.0:
            raise ValueError("hall_package_size must be three positive SI dimensions")
        if min(self.sole_length, self.sole_width, self.tpu_thickness, self.pcb_enclosure_thickness) <= 0.0:
            raise ValueError("sole_length, sole_width, and tpu_thickness must be positive")
        if min(self.magnet_spacing_x, self.magnet_spacing_y, self.magnet_size, self.magnet_thickness) <= 0.0:
            raise ValueError("magnet geometry must be positive")
        if self.initial_hall_magnet_distance <= self.dipole_min_distance:
            raise ValueError("initial Hall-to-magnet distance must exceed dipole_min_distance")
        half_magnet = 0.5 * self.magnet_thickness
        if not half_magnet <= self.magnet_embedding_depth <= self.tpu_thickness - half_magnet:
            raise ValueError(
                "magnet_embedding_depth must keep the complete magnet inside the TPU layer"
            )
        if self.initial_hall_magnet_distance < self.magnet_embedding_depth:
            raise ValueError(
                "initial_hall_magnet_distance must be >= magnet_embedding_depth"
            )
        if min(self.local_normal_stiffness, self.local_shear_stiffness, self.local_damping) <= 0.0:
            raise ValueError("local stiffness and damping values must be positive")
        if self.contact_distribution_mode not in ("aggregate", "detailed"):
            raise ValueError(
                "contact_distribution_mode must be 'aggregate' or 'detailed'"
            )
        if self.contact_spread_sigma <= 0.0:
            raise ValueError("contact_spread_sigma must be positive")
        if self.detailed_contact_force_atol < 0.0 or self.detailed_contact_force_rtol < 0.0:
            raise ValueError("detailed contact audit tolerances must be non-negative")
        if not 0.0 <= self.tpu_poisson_ratio < 0.5:
            raise ValueError("tpu_poisson_ratio must be in [0, 0.5)")
        if min(self.tpu_youngs_modulus, self.tpu_solid_youngs_modulus, self.tpu_density) <= 0.0:
            raise ValueError("TPU modulus and density values must be positive")
        if not 0.0 < self.tpu_top_anchor_depth < self.tpu_thickness:
            raise ValueError("tpu_top_anchor_depth must be inside the magnetized TPU thickness")
        if self.tpu_top_anchor_grid_size <= 0.0:
            raise ValueError("tpu_top_anchor_grid_size must be positive")
        if self.deformable_embedding_neighbors < 1 or self.deformable_frame_sample_radius <= 0.0:
            raise ValueError("invalid deformable embedding settings")
        if self.deformable_max_cooked_thickness_ratio < 1.0:
            raise ValueError("deformable_max_cooked_thickness_ratio must be >= 1")
        if self.tpu_solver_position_iteration_count < 1 or self.tpu_simulation_hexahedral_resolution < 4:
            raise ValueError("invalid deformable TPU solver settings")
        if self.sensor_sample_rate <= 0.0 or self.auto_zero_samples < 1:
            raise ValueError("invalid sensor sample rate or auto-zero sample count")
        if self.resolution < 0.0 or self.low_pass_cutoff < 0.0:
            raise ValueError("resolution and low_pass_cutoff must be non-negative")
        if self.observation_scale_t <= 0.0 or self.loading_history_length < 1:
            raise ValueError("invalid observation scale or loading history length")
        for name in (
            "normal_stiffness_scale_range",
            "shear_stiffness_scale_range",
            "damping_scale_range",
            "contact_spread_scale_range",
            "magnetic_moment_scale_range",
            "observation_sensor_gain_range",
            "observation_axis_gain_range",
            "reported_sample_period_range",
        ):
            lower, upper = getattr(self, name)
            if lower <= 0.0 or upper < lower:
                raise ValueError(f"invalid {name}={getattr(self, name)!r}")
        if self.magnet_position_jitter_std < 0.0:
            raise ValueError("magnet_position_jitter_std must be non-negative")
        if self.observation_cross_axis_std < 0.0 or self.observation_zero_residual_std < 0.0:
            raise ValueError("observation cross-axis and residual standard deviations must be non-negative")
        if not 0.0 <= self.dead_channel_probability <= 1.0:
            raise ValueError("dead_channel_probability must be in [0,1]")
        if not 0.0 <= self.foot_dropout_probability <= 1.0:
            raise ValueError("foot_dropout_probability must be in [0,1]")
        if self.maximum_packet_delay_steps < 0:
            raise ValueError("maximum_packet_delay_steps must be non-negative")
        if self.debug_vis_max_envs < 1:
            raise ValueError("debug_vis_max_envs must be positive")
        if min(
            self.debug_field_scale,
            self.debug_field_arrow_width,
            self.debug_field_arrow_min_length,
            self.debug_field_arrow_max_length,
        ) <= 0.0:
            raise ValueError("debug field-arrow dimensions must be positive")
        if self.debug_field_arrow_max_length < self.debug_field_arrow_min_length:
            raise ValueError("debug field-arrow maximum length must be >= minimum length")
        direction_norm = math.sqrt(sum(value * value for value in self.magnetization_direction))
        if direction_norm <= 1.0e-12 or self.magnetic_moment <= 0.0:
            raise ValueError("magnetization direction and magnetic moment must be non-zero")

    @property
    def num_hall_sensors(self) -> int:
        return len(self.hall_positions_normalized)

    @property
    def sensor_period(self) -> float:
        return 1.0 / self.sensor_sample_rate

    @property
    def remanence_implied_moment(self) -> float:
        """Dipole moment of a uniformly magnetized round disc, in A m^2."""
        mu0 = 4.0 * math.pi * 1.0e-7
        volume = math.pi * (0.5 * self.magnet_size) ** 2 * self.magnet_thickness
        return self.remanence_strength * volume / mu0

    @property
    def hall_to_tpu_top_distance(self) -> float:
        """Vertical gap from the Hall sample plane to the TPU upper surface."""

        return self.initial_hall_magnet_distance - self.magnet_embedding_depth
