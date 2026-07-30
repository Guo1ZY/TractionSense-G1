# Robust traction recovery pipeline result — 2026-07-22

## Outcome

The recovery branch has achieved the intended friction-dependent behavior in
MuJoCo:

- low friction actively limits forward speed even when the command is large;
- high friction preserves substantially more of the requested speed;
- the final standalone 640-D Student completed all 15 friction/speed cells
  without a fall;
- the previously observed Student failure at `mu=0.80, cmd=1.00 m/s` was
  corrected by a second closed-loop DAgger pass (`0.316 -> 0.859 m/s`).
- a candidate-only 0.5 m/s^2 command slew limiter removes the fixed seed-42
  high-friction transient while retaining the friction-dependent speed split;
  the strict full Isaac matrix still contains one separate low-friction fall.

This is a simulation candidate, not a real-robot release.  The unfiltered
randomized Isaac stress test contains a rare high-friction transient tail
(3/128 falls in the strict seed-42 rerun), and the filtered full matrix retains
one low-friction fall.  The stable 2026-07-21 deployment slots and the main
deploy config therefore remain unchanged.

## Selected recovery artifacts

- 641-D Oracle Teacher checkpoint:
  `unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust_shoulder_recovery/2026-07-21_18-39-21_high_mu_transient_recovery_from_7775/model_7890.pt`
- isolated Oracle Teacher slot:
  `unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/traction_teacher_recovery_7890`
- 640-D to friction DAgger estimator:
  `unitree_rl_lab/logs/evaluations/friction_estimator/20260722_model_7890_recovery_teacher/model_sim2sim_dagger1/friction_estimator.onnx`
- 640-D standalone Student checkpoint:
  `unitree_rl_lab/logs/evaluations/traction_student/20260722_model_7890_privileged_aux_dagger2/student_actor.pt`
- isolated Student slot:
  `unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/traction_student_7890_dagger2_candidate`

The Oracle Teacher consumes 640 deployable observations plus exact effective
friction as input 641.  The estimator uses only the 640 deployable observations.
The final Student directly maps those 640 observations to 29 joint actions and
does not receive exact friction.

## MuJoCo acceptance matrices

| System | Falls | low `mu=0.08`, cmd 1.5 | high `mu=0.80`, cmd 1.0 | high `mu=1.20`, cmd 1.5 | Result |
|---|---:|---:|---:|---:|:---:|
| Oracle Teacher | 0/15 cells | 0.169 m/s | 0.843 m/s | 1.133 m/s | PASS |
| Teacher + estimated friction | 0/15 cells | 0.173 m/s | 0.892 m/s | 1.139 m/s | PASS |
| standalone Student DAgger2 | 0/15 cells | 0.175 m/s | 0.859 m/s | 1.149 m/s | PASS |
| Student DAgger2 + command slew | 0/15 cells | 0.181 m/s | 0.797 m/s | 1.100 m/s | PASS |

Final Student full matrix:

| friction | cmd 0.5 | cmd 1.0 | cmd 1.5 |
|---:|---:|---:|---:|
| 0.08 | 0.164 | 0.189 | 0.175 |
| 0.20 | 0.252 | 0.243 | 0.225 |
| 0.40 | 0.388 | 0.380 | 0.326 |
| 0.80 | 0.496 | 0.859 | 0.970 |
| 1.20 | 0.539 | 0.954 | 1.149 |

All values are mean forward speed in m/s.  The Student's high/low-friction
speed separation at command 1.5 m/s is 0.975 m/s.  Its high-friction lateral
speed gate also passes (`|vy|=0.132 m/s <= 0.25 m/s`).  The complete report is
at
`unitree_rl_lab/logs/evaluations/mujoco_traction_student/model_7890_student_dagger2_full/summary.md`.

## Estimator and distillation gates

The final DAgger estimator passes the unseen Isaac gates:

- friction MAE: 0.0599;
- extreme-regime classification accuracy: 0.9975;
- low-friction predicted mean: 0.1706;
- high-friction predicted mean: 0.9954.

On the collected MuJoCo covariate-shift rollout its friction MAE is 0.0197.
The final Student's unseen Isaac action imitation MAE is 0.00803 and its 95th
percentile absolute error is 0.02205; both offline gates pass.

## Remaining acceptance gap

Model 7890's randomized Isaac high-friction test results were:

| seed | environments | `mu=1.20, cmd=1.50` falls |
|---:|---:|---:|
| 42 | 128 | 3 (strict rerun) |
| 43 | 128 | 0 |
| 44 | 128 | 0 |

The evaluator now counts terminations across the command-ramp/warm-up and
measurement phases and labels both phases in its fall diagnostics.  The strict
seed-42 rerun records three bad-orientation terminations.  They occur after
forward speed briefly reaches approximately 2.0--2.25 m/s.

The first 80-iteration guard fine-tune did not remove the fixed seed-42
failures.  A further isolated 120-iteration continuation was scanned at models
8000, 8020, 8040, 8060, 8080 and 8088; none reached zero falls (the best results
were 2/128).  Both guard branches were therefore rejected.  Model 7890 remains
the best cross-simulator candidate, but it is not approved for hardware until
the remaining randomized-domain tail is resolved and the complete pipeline is
revalidated.

## Command-transient mitigation

The three strict seed-42 failures share a command transient: forward speed
briefly overshoots to approximately 2.0--2.25 m/s before a forward-pitch
termination.  Repeating the same 128-environment Isaac test while changing
only the 0-to-1.5 m/s command ramp from one second to three seconds gives zero
falls and 1.225 m/s mean forward speed.

The corresponding full 5-friction by 3-speed Isaac matrix has zero falls in
14/15 cells.  Its only failure is 1/128 at `mu=0.08, cmd=1.50`; all three
`mu=1.20` cells are zero-fall.  Thus the slew limiter fixes the diagnosed
high-friction overshoot but does not yet satisfy the stricter global
randomized-domain zero-fall requirement.
The strict matrix and fall diagnostic are under
`unitree_rl_lab/logs/evaluations/traction_teacher_robust/shoulder_recovery_slew_full/model_7890_seed42_ramp150`.

A final 100-iteration low-friction recovery branch was also tested.  Its early
model 7910 reached zero falls over the complete seed-42 15-cell matrix, but
seed 43 and seed 44 each exposed 1/128 high-friction falls.  Models 7930,
7950, 7970 and 7989 all repeated the seed-43 high-friction failure.  This branch
therefore moved the tail between regimes instead of reducing cross-seed risk
and was rejected; no estimator or Student was retrained from it.

The isolated DAgger2 Student slot now carries a configurable linear command
slew rate of 0.5 m/s^2 (yaw 1.0 rad/s^2).  The C++ runtime applies the same
filter to gamepad and automated command-file inputs.  Slots without the YAML
field retain the legacy behavior.  A second full MuJoCo matrix with this
candidate interface passes all 15 cells with zero falls:

- `mu=0.08, cmd=1.50`: 0.181 m/s;
- `mu=0.80, cmd=1.00`: 0.797 m/s;
- `mu=1.20, cmd=1.00`: 0.920 m/s;
- `mu=1.20, cmd=1.50`: 1.100 m/s.

The normal gamepad gain remains 1.0, so full forward stick defaults to 1.0
m/s even though the stress-test envelope remains available up to 1.5 m/s.
The complete filtered-interface matrix is at
`unitree_rl_lab/logs/evaluations/mujoco_traction_student/model_7890_student_dagger2_slew_full/summary.md`.

## Deployment state

- stable rollback Teacher, estimator and Student from model 7750 are untouched;
- `deploy/robots/g1_29dof/config/config.yaml` was not switched to model 7890;
- the 7890 artifacts are installed only in explicitly named candidate slots;
- MuJoCo validation used loopback only and sent no command to the robot.

Re-run the candidate-only filtered-interface matrix with:

```bash
cd <workspace>
./scripts/validate_student_7890_slew_mujoco.sh
```
