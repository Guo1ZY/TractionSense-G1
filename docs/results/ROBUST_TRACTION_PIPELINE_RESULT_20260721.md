# Robust traction pipeline result — 2026-07-21

## Selected artifacts

- Oracle Teacher checkpoint: `unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_teacher_robust_stability/2026-07-21_15-20-09_transient_band_from_7700/model_7750.pt`
- DAgger friction estimator: `unitree_rl_lab/logs/evaluations/friction_estimator/20260721_model_7750_robust_teacher/model_sim2sim_dagger1/friction_estimator.onnx`
- Standalone 640-D Student: `unitree_rl_lab/logs/evaluations/traction_student/20260721_model_7750_privileged_aux_dagger2/student_actor.pt`
- Installed Student ONNX: `unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/traction_student/exported/policy.onnx`

The Teacher is 641-D and its final input is exact effective friction.  The
installed Student is 640-D and its deploy YAML contains no exact-friction term.

## Acceptance results

| System | Test | Falls | low μ=0.08, cmd=1.5 | high μ=1.20, cmd=1.5 | high-μ \|vy\| | Result |
|---|---|---:|---:|---:|---:|:---:|
| Oracle Teacher | Isaac full matrix, seed 42 | 0 | 0.184 m/s | 1.101 m/s | 0.097 m/s max matrix | PASS |
| Oracle Teacher | MuJoCo full matrix | 0 | 0.155 m/s | 0.853 m/s | 0.122 m/s | PASS |
| Teacher + estimated μ | MuJoCo full matrix | 0 | 0.154 m/s | 0.840 m/s | 0.124 m/s | PASS |
| Standalone 640-D Student | MuJoCo full matrix | 0 | 0.158 m/s | 0.840 m/s | 0.130 m/s | PASS |

Student unseen-Isaac action distillation metrics: MAE 0.00578, 95th-percentile
absolute error 0.01585.  The Student was accepted only after closed-loop MuJoCo
passed; the first two offline candidates are retained as diagnostics and are
not installed.

## Reproduction

```bash
cd <workspace>
./scripts/train_traction_student.sh
./scripts/validate_student_mujoco.sh --skip-build --strict
```

These results are simulation-only.  No command in this pipeline targets the
real robot network; MuJoCo validation uses loopback (`lo`).
