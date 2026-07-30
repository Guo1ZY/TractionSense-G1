# Curated checkpoints

Only representative artifacts needed to reproduce the paper pipeline are
included. Intermediate PPO checkpoints, TensorBoard logs, rollout datasets,
OBS1 binaries, rejected branches and machine-specific real-robot calibration
files are intentionally excluded.

| File | Role | Input → output | Hardware status |
|---|---|---|---|
| `baseline_model_49999.pt` | Original locomotion initialization | task checkpoint | simulation baseline |
| `oracle_teacher_model_8900.pt` | Privileged Oracle Teacher | 641 → 29 | simulation only |
| `student_dagger2_actor.pt` | Standalone distilled Student | 640 → 29 | simulation candidate |
| `student_dagger2_policy.onnx` | Deployable Student export | 640 → 29 | simulation candidate |
| `traction_magnetic_motion_8900.onnx` | Estimator-guided magnetic composite | 1864 → 29 | inactive candidate |
| `friction_estimator_1864.pt` | Causal traction estimator | 1864 → scalar | inactive candidate |
| `lateral_velocity_estimator.onnx` | Causal lateral estimator | history → scalar | inactive candidate |
| `proprio_traction_classifier_480.onnx` | Two-surface classifier | 480 → `p_low` | MuJoCo only |

## SHA-256

```text
c508af7910a69e2bc06111caaa677d5bea521bfb52fc654d82d38b499e2ae99b  baseline_model_49999.pt
19775305eacb8af50531f63da8de60a5035f49bce69ab05e4fd85fec995f8d47  friction_estimator_1864.pt
6277a684349e59b0c671692155e780987ef8d0e1d80021126a853a386007fd91  lateral_velocity_estimator.onnx
9004d60e306798d9377546a6f5f3de58dfd25b1d5ad4a57067d89f0ea112891d  oracle_teacher_model_8900.pt
ecb6bf60cf7e461c9aaf876853637b193e003793bff6e2552ddd279bcda61693  proprio_traction_classifier_480.onnx
bc3b43f8b7d30d9e92cf6cc695b77770789f1e0832c7457504e2cc59acb5a0a0  student_dagger2_actor.pt
46d41bf1ed18f24659392160ad583bd1e34f592865b4c4794f953427687a67c5  student_dagger2_policy.onnx
636871c055bab7e465928c728ecc25da2baa6b3ef12f4307fc36c2643a713917  traction_magnetic_motion_8900.onnx
```

The real-floor-calibrated 480-D classifier is deliberately not published:
it must be fitted on the exact physical robot, shoe sole and test surfaces.
