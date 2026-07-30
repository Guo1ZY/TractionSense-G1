# Shared Motion Student — DAgger Round 2

Date: 2026-07-29

## Decision

Round 2 is complete.

- **Isaac Lab selection:** Stage C
- **Cross-simulator selection:** none
- **Real-robot deployment:** rejected

Stage C fixes the catastrophic Round-1 covariate shift in Isaac Lab, but it
does not preserve high-friction speed in MuJoCo.  The separate deploy slot is
installed only for simulation validation and remains inactive.

## Exact on-policy DAgger collection

Round-1 Student:

`unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round1_distill/shared_magnetic_policy.pt`

The collector was corrected to save the exact 1864-D tensor consumed by the
Student.  In particular, observation channel 1862 now contains the deployable
lateral-velocity estimator output rather than simulator ground truth.

Each DAgger record contains:

- 1864-D Student input
- 641-D privileged Teacher input
- friction, command, seed and simulation step
- exact pre-fall flag
- 25-step post-reset recovery flag
- sample priority: normal 1×, recovery 4×, pre-fall 8×

| Split | Seed | Samples | Pre-fall | Recovery |
|---|---:|---:|---:|---:|
| Train | 112358 | 56,424 | 143 | 874 |
| Train | 173205 | 56,399 | 102 | 633 |
| Test | 223607 | 56,430 | 139 | 844 |

The aggregated Stage-A/B/C training corpus contains 199,223 samples.  It
combines the two Round-2 Student seeds with safe Round-1 data instead of
replacing the safe distribution.

Frozen Teacher:

`unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/traction_teacher_motion_balancedsym8900/exported/policy.onnx`

## Staged trust-region distillation

The rejected Round-1 weights were used only to generate difficult states.
Optimization was re-anchored at the safe `trust010` Student.

### Stage A — shared foot encoder only

- Trainable: one shared dual-foot spatial-temporal encoder
- Teacher residual: 0.05 low / 0.20 high
- Base-action anchor: 2.0
- Learning rate: `2e-5`, 12 epochs

This stage recovered the required adaptive behavior in Isaac Lab.

### Stage B — final action head only

- Parent: Stage A
- Trainable: final `128 -> 29` layer
- Teacher residual: 0.03 low / 0.10 high
- Base-action anchor: 3.0
- Learning rate: `1e-5`, 8 epochs

Offline gates passed, but lateral displacement was slightly worse.  Stage B
was not selected as the full-network parent.

### Stage C — full-network low-rate branch

- Parent: Stage A
- Trainable: complete Student
- Teacher residual: 0.02 low / 0.05 high
- Base-action anchor: 5.0
- Learning rate: `5e-6`, 5 epochs

Stage C reduced tail action errors and removed the rare Stage-A fall on seed
42.

## Isaac Lab closed-loop selection

Five friction values, two speeds, 64 environments, 300 measured steps:

| Stage | Seeds | Fall events | High-μ vx at cmd=1 | Low-μ vx at cmd=1 | Mean |vy| | Max lateral |
|---|---|---:|---:|---:|---:|---:|
| A | 223607, 42 | 1 | 0.913 | 0.271 | 0.0776 | 0.428 m |
| B | 223607 | 0 | 0.908 | 0.274 | 0.0777 | 0.431 m |
| C | 223607, 42 | **0** | **0.914** | **0.270** | **0.0775** | **0.427 m** |

Stage C therefore passes the Isaac safety and traction-adaptation conditions:

- 1,280 condition-environment rollouts without a fall
- low friction self-limits speed
- high friction tracks approximately 0.91 m/s

## MuJoCo cross-simulator result

Stage C was installed to the inactive slot:

`traction_shared_motion_student_round2_c`

The 5×2 MuJoCo matrix used:

- dual-foot magnetic bridge
- motion feedback
- deployable lateral-velocity estimator
- DDS loopback only

Results:

- 10/10 cells stable
- 0 falls
- low friction, 1.0 m/s command: 0.185 m/s
- high friction, 1.0 m/s command: 0.180 m/s
- high-friction target: at least 0.80 m/s
- overall: **NEEDS_WORK**

The Student is safe but its high-friction speed adaptation does not transfer
from Isaac Lab to MuJoCo.  This isolates the remaining problem to the
cross-simulator magnetic/contact/dynamics observation distribution rather than
basic locomotion stability.

MuJoCo produced 4,030 deploy observation records for the next DAgger round:

`unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_stage_c_mujoco/policy_obs.npz`

## Artifacts

- Round-2 data:
  `unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_data/`
- Stage A:
  `unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_stage_a_foot_encoder/`
- Stage B:
  `unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_stage_b_actor_head/`
- Stage C:
  `unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_stage_c_full_from_a/`
- Stage C MuJoCo:
  `unitree_rl_lab/logs/evaluations/traction_shared_motion_student/20260729_round2_stage_c_mujoco/`

## Required Round 3

1. Convert the 4,030 exact MuJoCo observations into the current motion-feedback
   Teacher schema.
2. Label them with frozen `model_8900`.
3. Compare the Hall-array, packet-period, valid-bit and motion-feature
   distributions between Isaac Lab and MuJoCo.
4. Mix MuJoCo samples with the safe Isaac corpus using simulator-balanced
   batches; do not fine-tune on MuJoCo alone.
5. Continue from Stage C with a small trust region and select only if both
   simulators pass.

## Safety state

The active real-robot configuration remains unchanged:

`policy_dir: config/policy/velocity/foot`

The Stage-C install manifest explicitly records `"activated": false`.
