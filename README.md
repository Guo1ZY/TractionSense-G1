# TractionSense-G1

**Traction-conditioned humanoid locomotion via privileged learning and
sensor-robust distillation.**

TractionSense-G1 is a research extension of
[Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) for the
Unitree G1 humanoid. It studies a simple behavior objective: under the same
forward command, the robot should self-limit on low-traction ground and retain
the requested speed on high-traction ground.

The repository contains the complete research path rather than only the final
policy: friction-conditioned PPO, a privileged Oracle Teacher, standalone and
shared-encoder Students, DAgger, Hall-foot experiments, sensor-noise
randomization, Isaac-to-MuJoCo evaluation and conservative deployment tools.

> **Research status:** the reported adaptive policies are simulation and
> Sim2Sim candidates. No included checkpoint is approved for untethered
> hardware operation. Plantar Hall sensing remains experimental; the 480-D
> proprioceptive two-surface controller is provided as an explicit sensor
> ablation, not as evidence that noisy Hall signals measure friction.

## Method

```text
effective friction μ ──► privileged Oracle Teacher (641 → 29)
                              │ actions / latent targets
                              ▼
proprio history + Hall history ──► shared-foot Student (1864 → 548 → 29)
             │                         │
             ├── sensor noise/dropout  └── DAgger + trust-region fine-tuning
             ▼
Isaac Lab ──► MuJoCo ──► guarded deployment ──► Real2Sim calibration loop
```

Two deployment tracks are retained:

1. **Paper pipeline:** privileged Teacher, Hall/proprio history, friction
   estimator and shared dual-foot encoder.
2. **Sensor-ablation pipeline:** ordinary 480-D G1 proprioceptive history,
   binary traction-state classification and a `0.20/0.35 m/s` hysteretic
   speed governor.

## Representative results

### 640-D standalone Student, MuJoCo

The DAgger-2 Student completed the full five-friction by three-speed matrix
without a fall.

| friction | command 0.5 | command 1.0 | command 1.5 |
|---:|---:|---:|---:|
| 0.08 | 0.164 | 0.189 | 0.175 |
| 0.20 | 0.252 | 0.243 | 0.225 |
| 0.40 | 0.388 | 0.380 | 0.326 |
| 0.80 | 0.496 | 0.859 | 0.970 |
| 1.20 | 0.539 | 0.954 | 1.149 |

Values are mean forward velocity in m/s. The full report and the remaining
Isaac randomized-tail limitation are documented in
[the robust pipeline report](docs/results/ROBUST_TRACTION_PIPELINE_RESULT_20260722.md).

### 480-D sensor ablation, MuJoCo

With the same raw command of `0.8 m/s`, the automatic governor selected
different caps:

| friction | state | measured velocity | falls |
|---:|:---:|---:|---:|
| 0.15 | LOW | 0.146 m/s | 0 |
| 1.20 | HIGH | 0.325 m/s | 0 |

This classifier is simulator-specific. A real robot must collect labeled
proprioception on the exact two floors before AUTO mode can be installed.

## Repository layout

```text
source/.../velocity_foot_env_cfg.py    traction-aware Isaac Lab tasks
source/.../mdp/                        observations, rewards, events, symmetry
scripts/rsl_rl/                        PPO training and friction evaluation
research_scripts/                      paper experiments and pipelines
deploy/robots/g1_29dof/                MuJoCo/real deployment integration
checkpoints/                           curated representative artifacts
docs/results/                          experiment decisions and limitations
assets/demo/                           compact qualitative comparisons
```

See [release scope](docs/RELEASE_SCOPE.md) for the exact inclusion/exclusion
policy and [checkpoint documentation](checkpoints/README.md) for dimensions,
status and SHA-256 hashes.

## Installation

The base installation follows Unitree RL Lab and Isaac Lab. The upstream
instructions are preserved in
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

```bash
git clone https://github.com/Guo1ZY/TractionSense-G1.git
cd TractionSense-G1

conda create -n traction-g1 python=3.11
conda activate traction-g1
pip install -e source/unitree_rl_lab
```

Isaac Sim/Isaac Lab, Unitree SDK2, MuJoCo and ONNX Runtime must be installed
according to their own licenses and platform requirements.

## Training

Baseline training:

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity \
  --headless
```

Traction Teacher training and robust continuation scripts are under
`research_scripts/`:

```bash
bash research_scripts/finetune_robust_teacher.sh
bash research_scripts/finetune_robust_teacher_stability.sh
```

The final Teacher consumes `640` deployable observations plus one privileged
effective-friction value. The Student never receives ground-truth friction at
deployment.

Student distillation:

```bash
python research_scripts/distill_traction_student.py --help
python research_scripts/train_shared_magnetic_policy.py --help
python research_scripts/fine_tune_shared_magnetic_dagger.py --help
```

Large rollout datasets are not stored in Git. The collectors and exact dataset
schemas are included so they can be regenerated.

## Evaluation

Isaac Lab:

```bash
python scripts/rsl_rl/eval_friction_matrix.py --help
python research_scripts/evaluate_teacher.py --help
```

MuJoCo:

```bash
python research_scripts/mujoco_friction_speed_matrix.py --help
```

The evaluator forces DDS onto loopback for simulation tests and reports
friction × command speed, lateral drift, fall count and continuous
friction-switch response.

## Guarded two-surface deployment

The sensor-ablation workflow is documented in
[SENSORLESS_TWO_SURFACE_AUTO_20260730.md](docs/results/SENSORLESS_TWO_SURFACE_AUTO_20260730.md).

```bash
cd deploy/robots/g1_29dof

# collect the same 0.20 m/s condition on known LOW and HIGH floors
G1_REAL_TEST_ACK=YES ./collect_two_surface_proprio.sh low \
  --network <interface> --log
G1_REAL_TEST_ACK=YES ./collect_two_surface_proprio.sh high \
  --network <interface> --log
```

AUTO mode refuses to start unless a separately calibrated real-floor
classifier has been explicitly installed. Hardware tests require a load-rated
overhead harness, an independent emergency-stop operator and low-speed,
straight-line trials.

## Demo

- [MuJoCo friction comparison](assets/demo/mujoco_friction_model_comparison_cmd10.mp4)
- [Low-speed behavior](assets/demo/g1_slow_speed.mp4)
- [High-speed behavior](assets/demo/g1_fast_speed.mp4)
- [Training curves](assets/figures/00_dashboard.png)

## Citation

The paper citation will be added after publication. Until then, cite the
repository metadata in [CITATION.cff](CITATION.cff).

## License and attribution

This derivative repository retains the upstream Apache License 2.0 license and
Git history. See [LICENCE](LICENCE), [NOTICE](NOTICE) and the
[upstream README](docs/UPSTREAM_README.md). Unitree and G1 are trademarks of
their respective owners; this is not an official Unitree release.
