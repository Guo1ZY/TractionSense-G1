"""Reproducible experiment and ablation registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TractionExperimentCfg:
    identifier: str
    description: str
    policy: str
    force_mode: str = "randomized_tactile_force"
    temporal_encoder: str = "gru"
    history_seconds: float = 0.30
    latent_dim: int = 16
    use_teacher: bool = True
    use_dagger: bool = False
    use_on_policy_finetune: bool = False
    use_governor: bool = True
    domain_randomization_stage: int = 5
    perturbation: str = "none"
    matrix_values: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (20260731, 20260732, 20260733)

    def __post_init__(self) -> None:
        if self.force_mode not in (
            "proprio_only",
            "ideal_raw_force",
            "randomized_tactile_force",
        ):
            raise ValueError(self.force_mode)
        if self.temporal_encoder not in ("none", "gru", "tcn"):
            raise ValueError(self.temporal_encoder)
        if self.latent_dim not in (8, 16):
            raise ValueError("latent dimension must be 8 or 16")
        if self.domain_randomization_stage not in range(6):
            raise ValueError("domain randomization stage must be 0..5")


EXPERIMENTS = (
    TractionExperimentCfg(
        "proprio_baseline",
        "Proprio Baseline",
        "baseline",
        force_mode="proprio_only",
        temporal_encoder="none",
        use_teacher=False,
        use_governor=False,
        domain_randomization_stage=0,
    ),
    TractionExperimentCfg(
        "ideal_raw_force",
        "Ideal Raw Force",
        "raw_force_mlp",
        force_mode="ideal_raw_force",
        temporal_encoder="none",
        use_teacher=False,
        use_governor=False,
        domain_randomization_stage=0,
    ),
    TractionExperimentCfg(
        "raw_force_history",
        "Raw Force History",
        "temporal_student",
        force_mode="ideal_raw_force",
        use_teacher=False,
        use_governor=False,
        domain_randomization_stage=0,
    ),
    TractionExperimentCfg(
        "randomized_tactile_force",
        "Randomized Tactile Force",
        "temporal_student",
        use_teacher=False,
        use_governor=False,
    ),
    TractionExperimentCfg(
        "teacher_oracle",
        "Teacher Oracle",
        "teacher",
        force_mode="ideal_raw_force",
        temporal_encoder="none",
        use_governor=False,
        domain_randomization_stage=0,
    ),
    TractionExperimentCfg(
        "student_distillation",
        "Student Distillation",
        "temporal_student",
        use_governor=False,
    ),
    TractionExperimentCfg(
        "student_dagger",
        "Student + DAgger",
        "temporal_student",
        use_dagger=True,
        use_governor=False,
    ),
    TractionExperimentCfg(
        "student_on_policy",
        "Student + On-policy Fine-tuning",
        "temporal_student",
        use_dagger=True,
        use_on_policy_finetune=True,
        use_governor=False,
    ),
    TractionExperimentCfg(
        "full_method",
        "Full Method",
        "temporal_student",
        use_dagger=True,
        use_on_policy_finetune=True,
    ),
    TractionExperimentCfg(
        "no_teacher",
        "No Teacher",
        "temporal_student",
        use_teacher=False,
    ),
    TractionExperimentCfg(
        "no_temporal_history",
        "No Temporal History",
        "temporal_student",
        temporal_encoder="none",
        history_seconds=0.02,
    ),
    TractionExperimentCfg(
        "no_tactile_force",
        "No Tactile Force",
        "temporal_student",
        force_mode="proprio_only",
    ),
    TractionExperimentCfg(
        "no_governor",
        "No Governor",
        "temporal_student",
        use_governor=False,
    ),
    TractionExperimentCfg(
        "fixed_speed_scaling",
        "Fixed Speed Scaling",
        "temporal_student",
        use_governor=False,
        perturbation="fixed_speed_scale_0.5",
    ),
    TractionExperimentCfg(
        "proprio_only_governor",
        "Proprio-only Governor",
        "temporal_student",
        force_mode="proprio_only",
        perturbation="proprio_risk_estimator",
    ),
    TractionExperimentCfg(
        "no_domain_randomization",
        "No Domain Randomization",
        "temporal_student",
        domain_randomization_stage=0,
    ),
    TractionExperimentCfg(
        "no_delay_randomization",
        "No Delay Randomization",
        "temporal_student",
        perturbation="disable_delay",
    ),
    TractionExperimentCfg(
        "no_dropout_randomization",
        "No Dropout Randomization",
        "temporal_student",
        perturbation="disable_dropout",
    ),
    TractionExperimentCfg(
        "sensor_dropout",
        "Sensor Dropout",
        "temporal_student",
        perturbation="dropout",
    ),
    TractionExperimentCfg(
        "sensor_bias",
        "Sensor Bias",
        "temporal_student",
        perturbation="bias",
    ),
    TractionExperimentCfg(
        "sensor_delay",
        "Sensor Delay",
        "temporal_student",
        perturbation="delay",
    ),
    TractionExperimentCfg(
        "cross_axis_perturbation",
        "Cross-axis Perturbation",
        "temporal_student",
        perturbation="cross_axis",
    ),
    TractionExperimentCfg(
        "history_length_sweep",
        "Different history lengths",
        "temporal_student",
        matrix_values=("0.10", "0.20", "0.30", "0.40", "0.60"),
    ),
    TractionExperimentCfg(
        "gru_vs_tcn",
        "GRU vs TCN",
        "temporal_student",
        matrix_values=("gru", "tcn"),
    ),
    TractionExperimentCfg(
        "latent_dimension_sweep",
        "Different latent dimensions",
        "temporal_student",
        matrix_values=("8", "16"),
    ),
)


def experiment_by_id(identifier: str) -> TractionExperimentCfg:
    for experiment in EXPERIMENTS:
        if experiment.identifier == identifier:
            return experiment
    raise KeyError(identifier)


def write_experiment_registry(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            [asdict(experiment) for experiment in EXPERIMENTS],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
