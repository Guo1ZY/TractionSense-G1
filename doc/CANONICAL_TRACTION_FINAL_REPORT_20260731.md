# Traction-Adaptive Humanoid Locomotion with Flexible Magnetic Tactile Feet

Date: 2026-07-31  
Branch: `feature/traction-adaptive-tactile-policy`  
Status: **software/training/Sim2Sim candidate, not real-robot validated**

This report only contains measured results from the current checkout and actual
run logs. It does not claim that the short training runs are converged, that the
ankle rigid-body velocity proxy is contact-point slip ground truth, or that a
Hall-to-force calibration exists.

## 1. Project overview

The current 29-DOF G1 asset, action order, default pose, PD gains, mass/inertia
and main collision model were preserved. The implemented path is:

```text
Isaac ground-filtered ContactSensor force
  -> signed left/right ankle-roll local force [6]
  -> Physics-consistent tactile observation randomization
  -> 15-frame tactile-proprioceptive history
  -> Temporal tactile-proprioceptive student encoder
  -> slip probability / traction margin / confidence
  -> Slip-aware traction-adaptive command governor
  -> preserved locomotion actor + gated traction residual
  -> 29 joint-position actions
```

The Teacher separately receives a privileged traction representation containing
current simulated friction, ideal force and contact/slip diagnostics. No Hall,
magnetic-field or flexible-material truth is fabricated in simulation.

## 2. Final method structure

- **Physics-consistent tactile observation randomization** maps ideal local
  force to deployment-domain force using per-axis scale, bias, drift,
  cross-axis coupling, installation rotation, delay, low-pass filtering,
  load-dependent noise, saturation, dropout, burst dropout, spikes, validity,
  age and optional simplified hysteresis.
- **Privileged traction representation** is a 135-D current-frame vector
  compressed to a configurable 8-D or 16-D Teacher latent.
- **Temporal tactile-proprioceptive student encoder** uses a shared left/right
  foot GRU or TCN, a proprioceptive GRU, and heads for latent, two slip
  probabilities, traction margin and confidence.
- **Slip-aware traction-adaptive command governor** uses only deployable
  Student outputs. It applies risk debounce, hysteresis, minimum hold,
  fast-down/slow-recovery filtering, acceleration/deceleration limits,
  lateral/yaw limiting, push-off scale and invalid-sensor fallback.
- A zero-initialized residual traction branch protects the pretrained
  locomotion actor at initialization.

## 3. Modified files

Main new or changed canonical files:

- `source/unitree_rl_lab/unitree_rl_lab/traction/{schema,sensor_layout,ble,history,diagnostics,tactile,networks,governor,rsl_models,isaac_observations,isaac_events,experiments,evaluation,deployment}.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/traction_canonical/__init__.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/traction_rsl_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_canonical_traction_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/utils/partial_checkpoint.py`
- `scripts/traction/{collect_traction_dataset,distill_traction_student,evaluate_traction,export_traction_policy,initialize_gated_baseline,replay_traction_policy}.py`
- `scripts/tests/{audit_baseline_rollout,regression_partial_checkpoint,smoke_canonical_traction_env,test_traction_core,test_traction_deployment,test_traction_evaluation,test_traction_networks_governor,test_traction_sensor_schema}.py`
- `doc/TRACTION_TACTILE_AUDIT_20260731.md`
- `doc/FOOT_SENSOR_LAYOUT_AND_WIRE_SCHEMA.md`
- MuJoCo:
  `simulate/src/traction/foot_force_bridge.{h,cc}`,
  `simulate_python/{traction_force_bridge,run_traction_sim2sim,run_traction_matrix,compare_traction_matrices}.py`
- Real single-foot utilities:
  `/home/mosense/guo_1/vola_sensor/record_raw_hall.py`, plus the legacy
  dashboards' region split and double-swap fixes.

Existing unrelated dirty files and untracked checkpoints/configurations were
not reset, deleted or bulk-staged.

## 4. Git branches and commits

`unitree_rl_lab`:

```text
9ff9662 feat: audit locomotion and tactile pipeline
0bcab42 feat: add canonical foot sensor layout
8f81ca5 feat: add observation-aware checkpoint migration
4df61ff feat: add tactile randomization and traction diagnostics
a1c3662 feat: add teacher student traction policy and governor
d898982 feat: add canonical traction teacher student training
0fbf806 feat: add traction curriculum and evaluation suite
b652a0d feat: add deployable traction policy package
3ea767f feat: export gated traction student policy
b2709c3 feat: add asymmetric friction transitions
db3aff8 feat: preserve locomotion warm start in traction training
b70057c feat: add explicit proprio baseline evaluation
7217070 feat: balance slip supervision in student distillation
dbc019e feat: supervise privileged traction margin
6e00330 feat: calibrate student confidence and traction governor
```

`unitree_mujoco`:

```text
3fdc15f feat: add canonical mujoco traction bridge and fixed-policy matrix
a1bd102 feat: add no-governor sim2sim ablation
5a56d0b feat: compare fixed-policy sim2sim matrices
f5d3fa0 test: label fixed-policy comparisons explicitly
a981b0a test: report sim2sim confidence intervals
```

## 5. 29-DOF joint/action order

| Index | Joint | Index | Joint |
|---:|---|---:|---|
| 0 | `left_hip_pitch_joint` | 15 | `left_shoulder_roll_joint` |
| 1 | `right_hip_pitch_joint` | 16 | `right_shoulder_roll_joint` |
| 2 | `waist_yaw_joint` | 17 | `left_ankle_roll_joint` |
| 3 | `left_hip_roll_joint` | 18 | `right_ankle_roll_joint` |
| 4 | `right_hip_roll_joint` | 19 | `left_shoulder_yaw_joint` |
| 5 | `waist_roll_joint` | 20 | `right_shoulder_yaw_joint` |
| 6 | `left_hip_yaw_joint` | 21 | `left_elbow_joint` |
| 7 | `right_hip_yaw_joint` | 22 | `right_elbow_joint` |
| 8 | `waist_pitch_joint` | 23 | `left_wrist_roll_joint` |
| 9 | `left_knee_joint` | 24 | `right_wrist_roll_joint` |
| 10 | `right_knee_joint` | 25 | `left_wrist_pitch_joint` |
| 11 | `left_shoulder_pitch_joint` | 26 | `right_wrist_pitch_joint` |
| 12 | `right_shoulder_pitch_joint` | 27 | `left_wrist_yaw_joint` |
| 13 | `left_ankle_pitch_joint` | 28 | `right_wrist_yaw_joint` |
| 14 | `right_ankle_pitch_joint` |  |  |

Action dimension is 29. Action scale is 0.25 rad per policy unit.

## 6. Actor and critic observations

Audited proprio baseline, five frames:

- Actor: `base_ang_vel 3 + projected_gravity 3 + command 3 +
  joint_pos_rel 29 + joint_vel 29 + previous_action 29 = 96/frame`;
  `96×5=480`.
- Critic prepends `base_lin_vel 3/frame`; `99×5=495`.
- Raw-force MLP compatibility mode appends six normalized force components:
  actor `102×5=510`, critic `105×5=525`.

Canonical Teacher:

- One frame is `96 current proprio + 3 adjusted/raw command +
  135 privileged traction = 234`.
- RSL input is five time-major frames: `234×5=1170`.
- The adapter reconstructs the exact old 480-D actor or 495-D critic
  term-major history, then appends the newest 16-D privileged latent.

Canonical Student:

- One deployable frame is 106-D.
- Fifteen frames (`0.30 s` at 50 Hz) give a 1590-D time-major input.
- Action output remains 29-D.

## 7. History order

- Legacy 480/495/510/525 schemas: term-major; within each term frames are
  oldest-to-newest.
- Canonical Student: time-major, oldest-to-newest; within each frame:
  `base_ang_vel[3], projected_gravity[3], joint_pos_rel[29],
  joint_vel[29], previous_action[29], raw_command[3],
  observed_force[6], valid[2], age[2]`.
- Reset clears manual history. Isaac-managed history was observed to seed its
  slots with the first post-reset sample and not leak a previous episode.
- Deployment suppresses governor decisions until 15 real frames have arrived;
  this avoids acting on reset-padding estimator transients.

## 8. Foot-force coordinates

- Bodies: `left_ankle_roll_link`, `right_ankle_roll_link`.
- Isaac ContactSensors are ground-filtered to
  `/World/ground/terrain/mesh` and track friction force.
- World force is the filtered normal plus friction vector, rotated in a GPU
  batch with inverse ankle-roll world quaternion.
- Frame: matching ankle-roll local frame.
- Axis meaning: `+x` toe, `+y` robot-left, `+z` up.
- Unit: N in logging; policy input is
  `F_local_N / (current_robot_mass_kg × 9.81)`, clipped to `[-2,2]`.
- No-contact output is zero apart from an intentionally randomized observation
  model's bias/noise.

## 9. Left/right force order

The only canonical order is:

```text
[L_Fx, L_Fy, L_Fz, R_Fx, R_Fy, R_Fz]
```

An Isaac rotation round-trip test measured maximum error
`1.373×10^-4 N`; the tested swing-foot force was exactly `0 N`.

## 10. Fifteen-point layout

Coordinates below are normalized topology coordinates, **not millimetres**.
Region 0/1/2 means forefoot/midfoot/heel.

| Point | `(x,y)` | Region | Point | `(x,y)` | Region |
|---|---|---:|---|---|---:|
| P0 | `(0.371,-0.016)` | 0 | P8 | `(-0.022,-0.241)` | 1 |
| P1 | `(0.299,0.201)` | 0 | P9 | `(-0.093,-0.004)` | 1 |
| P2 | `(0.297,-0.016)` | 0 | P10 | `(-0.251,-0.004)` | 2 |
| P3 | `(0.295,-0.248)` | 0 | P11 | `(-0.309,0.177)` | 2 |
| P4 | `(0.195,-0.018)` | 0 | P12 | `(-0.317,-0.004)` | 2 |
| P5 | `(0.065,0.013)` | 1 | P13 | `(-0.313,-0.208)` | 2 |
| P6 | `(-0.014,0.215)` | 1 | P14 | `(-0.382,-0.004)` | 2 |
| P7 | `(-0.017,0.002)` | 1 |  |  |  |

The source CAD establishes an insole envelope but contains no named 15-point
anchors. IDW display pixels are not promoted to physical sensor positions.

## 11. BLE channel to P0–P14 mapping

The canonical provisional mapping is explicit identity:

```text
BLE channel 0..14 -> P0..P14
```

This preserves the current firmware/UI order. It is marked provisional because
no independently verified PCB/CAD channel table was found. The canonical path
does not apply a hidden P12/P13 swap.

## 12. Hall-axis transforms

Per-channel provisional XY rotations, P0 through P14, are:

```text
[-90,-90,0,+90,0,-90,-90,180,+90,-90,0,-90,180,+90,0] degrees
```

Z is unchanged. The provisional right-foot mirror is
`diag(1,-1,1)`. These transforms document existing visualization orientation;
they are not claimed as force calibration. The legacy P12/P13 X correction
that was performed twice was removed from that path so a correction cannot
silently cancel itself.

## 13. Real interface status

Measured code protocol:

- BLE name `FootSensor15`;
- Notify UUID `0000ab01-0000-1000-8000-00805f9b34fb`;
- frame 125 bytes;
- header bytes `[0]=0x7D, [2]=0xF0, [3]=0x02`;
- payload `15×8` bytes;
- channel format big-endian `>hhhh`: `temperature_x10, Hall_X, Hall_Y, Hall_Z`;
- Hall fields are signed int16 with continuity unwrapping to bounded int32;
- timestamps and sequence are host-generated because the decoded payload has
  no known device timestamp/sequence;
- temperature is converted to degrees C.

No Hall `[15,3] -> net Fx,Fy,Fz` calibration parameters, regression model or
checkpoint were found. Therefore `CalibratedMagneticFootAdapter` requires an
externally supplied measured calibration and refuses construction without one.
No live BLE connection claim is made in this run.

## 14. Checkpoint migration

`old_to_new_flat_index()` maps by observation term and history index rather
than assuming new features are at the flattened tail.

For 480→510 actor and 495→525 critic:

- all old first-layer columns are copied to their semantic new indices;
- the thirty force-history columns are zero initialized;
- hidden/output layers, 29-D output and action standard deviation are reused.

Measured zero-force regression:

- action-mean maximum absolute error: `2.288818×10^-5`;
- value maximum absolute error: `1.220703×10^-4`;
- all 30 new actor input columns were zero.

## 15. Slip-label definition

Defaults:

- contact on/off: `Fn > 12 N` / retain while `Fn >= 6 N`;
- slip candidate on: contact and ankle planar speed `>0.12 m/s`;
- slip off: no contact or speed `<0.06 m/s`;
- minimum candidate duration: `0.04 s`;
- `Fn=abs(Fz)`, `Ft=sqrt(Fx²+Fy²)`,
  `utilization=Ft/(Fn+1e-6)`.

The current velocity is the ankle/foot rigid-body planar velocity and is named
`foot_slip_proxy` in code, logs and results. It is not presented as exact
contact-point relative velocity.

## 16. Privileged traction observation

The 135-D current privileged vector contains:

- current proprioception 96;
- left/right exact simulated friction 2;
- ideal local foot force 6;
- left/right Fn, Ft, utilization, contact, slip speed and slip label;
- four planar foot-velocity proxy components;
- base linear velocity 3;
- terrain/contact fields 4;
- available dynamics fields 8.

Unavailable randomized fields are explicit nominal zeros. Future ground
friction is not included.

## 17. Teacher network

- `PrivilegedTractionEncoder`: `135 -> 128 -> 64 -> latent(16)`, ELU.
- Warm-started locomotion actor input:
  reconstructed baseline 480 + latent 16.
- Warm-started critic input:
  reconstructed baseline 495 + latent 16.
- Actor/critic heads remain `512 -> 256 -> 128 -> 29/1`.

Actual 512-environment, 100-iteration run:

- 1,228,800 environment steps;
- final mean reward `39.40`;
- final mean episode length `938.91/1000`;
- timeout `0.9668`;
- bad-orientation termination `0.0312`;
- base-height termination `0.0020`;
- command curriculum level `0.3`;
- no logged NaN.

## 18. Student network

- Shared per-foot GRU or TCN input:
  `[Fx,Fy,Fz,valid,age]`; tested implementations share left/right weights.
- Default GRU hidden 32 per foot.
- Proprioceptive GRU hidden 64.
- Fusion `128 -> latent(16)`.
- Heads: two slip probabilities, traction margin and confidence.
- Locomotion branch: frozen audited 480-D baseline actor plus
  zero-initialized gated 29-D residual.
- Confidence combines a learned head with the causal validity/age confidence
  and is explicitly supervised during offline distillation.

An independent 512-environment, 100-iteration RSL Student PPO smoke/medium run
completed 1,228,800 environment steps with:

- final mean reward `40.07`;
- episode length `956.86/1000`;
- timeout `0.9746`;
- base-height termination `0.0078`;
- bad-orientation termination `0.0195`;
- no logged NaN.

This PPO run did not supervise auxiliary heads; the selected exported candidate
uses the later offline Teacher/DAgger auxiliary supervision.

## 19. TactileObservationModel

The default provisional ranges are centralized:

- scale `0.92..1.08`;
- episode bias `-4..4 N`, drift up to `0.6 N/sqrt(s)`, capped `6 N`;
- off-diagonal coupling `-0.04..0.04`;
- installation rotation `-3..3 deg`;
- delay `0..3` policy steps;
- low-pass time constant `0..0.04 s`;
- noise floor `0..0.8 N`, load fraction `0..0.012`;
- saturation `450..900 N`;
- sample dropout `0..0.01`;
- burst start `0..0.002`, length `2..8` steps;
- spike probability `0..0.001`, amplitude `10..50 N`;
- simplified hysteresis fraction `0..0.03`.

Stages 0–5 progressively enable ideal, scale/bias/noise, delay/low-pass,
coupling/rotation, drift/dropout/spike, and full saturation/hysteresis.
Parameters are explicitly provisional, not measured sensor statistics.

## 20. Command governor

Selected configuration:

- risk enter/exit `0.45/0.25`;
- risk debounce `0.06 s`, minimum hold `0.20 s`;
- persistent slip `0.12 s`;
- fast-down/slow-recovery constants `0.08/0.80 s`;
- normal maxima `vx=1.5 m/s`, `vy=0.6 m/s`, yaw `1.2 rad/s`;
- low/persistent minimum speed scale `0.22`;
- invalid-sensor scale `0.45`;
- traction-margin warning/critical `0.55/0.20`;
- acceleration normal/low `2.0/0.35`;
- deceleration normal/low `2.5/0.8`;
- lateral/yaw/push-off minimum scales `0.30/0.25/0.30`.

The governor does not consume ground-friction truth. A two-pass runtime first
estimates traction from raw-command history, adjusts the newest command, then
re-evaluates the fixed policy. The no-governor reference is explicit, not an
implicit setting.

## 21. Rewards

The canonical task retains the original velocity/yaw tracking, stability,
termination, impact, smoothness, energy and limit terms, and adds:

- slip-speed-proxy penalty, weight `-0.5`;
- normalized tangential-push penalty, weight `-0.08`;
- high-friction unsupported-slowdown penalty, weight `-0.2`.

The last term only activates when the minimum per-foot simulated friction is
above `0.7`; it prevents a policy from obtaining apparent safety by always
moving slowly.

## 22. Curriculum and scenarios

- Per-foot static and dynamic friction are equal and sampled continuously from
  `0.05..1.20`.
- Asymmetric left/right friction probability is `0.5`.
- Friction is resampled in interval events every `1.5..3.0 s`.
- The exact physics material and privileged label share one buffer; Isaac
  smoke measured material/label maximum error below `1e-6`.
- Tactile curriculum has six stages as described above.
- Experiment registry contains all 25 requested baselines/ablations, including
  history length, GRU/TCN, latent size and sensor perturbations.

## 23. Training commands

Representative reproducible commands:

```bash
cd /home/mosense/guo/unitree_rl_lab

/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-TractionCanonicalTeacher \
  --num_envs 512 --max_iterations 100 --seed 20260731 --headless \
  --partial_checkpoint model/rl/model_49999.pt \
  --run_name warmstart_medium_100iter_seed20260731

/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-TractionCanonicalStudent \
  --num_envs 512 --max_iterations 100 --seed 20260731 --headless \
  --partial_checkpoint model/rl/model_49999.pt \
  --run_name warmstart_medium_100iter_seed20260731

/home/mosense/miniconda3/envs/isaaclab-v2/bin/python \
  scripts/traction/distill_traction_student.py \
  --datasets \
    artifacts/canonical_traction_20260731/warmstart_teacher_transition_dataset_16000.npz \
    artifacts/canonical_traction_20260731/warmstart_dagger_transition_dataset_16000.npz \
  --baseline_checkpoint model/rl/model_49999.pt \
  --epochs 50 --batch_size 1024 --seed 20260731 --device cuda:0 \
  --output_dir \
    artifacts/canonical_traction_20260731/transition_balanced_confidence_teacher_dagger_distill_50epoch
```

The configured formal schedules remain Teacher 5000 and Student 12000
iterations. They were not run to completion in this execution.

## 24. Checkpoints

- Proprio baseline:
  `model/rl/model_49999.pt`, SHA-256
  `c508af7910a69e2bc06111caaa677d5bea521bfb52fc654d82d38b499e2ae99b`.
- Warm-start Teacher:
  `logs/rsl_rl/g1_29dof_canonical_traction_teacher/2026-07-31_17-57-28_warmstart_medium_100iter_seed20260731/model_99.pt`.
- Warm-start RSL Student:
  `logs/rsl_rl/g1_29dof_canonical_traction_student/2026-07-31_18-02-41_warmstart_medium_100iter_seed20260731/model_99.pt`.
- Selected confidence-supervised offline candidate:
  `artifacts/canonical_traction_20260731/transition_balanced_confidence_teacher_dagger_distill_50epoch/best.pt`.

The selected artifact is marked `trained_candidate`, not `validated`.

## 25. Training curves

- RSL TensorBoard event files are stored beside both 100-iteration
  checkpoints.
- Offline distillation curve:
  `artifacts/canonical_traction_20260731/transition_balanced_confidence_teacher_dagger_distill_50epoch/training_metrics.csv`.
- Isaac diagnostic trajectory plot:
  `artifacts/canonical_traction_20260731/evaluation_warmstart_teacher_dagger/trajectory.png`.

At epoch 49 on the 3,200-sample held-out simulation slice:

- action MSE `0.0177299713`;
- latent MSE `0.3037055290`;
- traction-margin MSE `0.0697288164`;
- confidence MSE `0.0276360364`;
- slip precision `0.6466381`;
- slip recall `0.9358178`;
- slip F1 `0.7648054`;
- slip AUC `0.9863096`.

These are dataset-slice metrics, not locomotion-validation metrics.

## 26. Isaac evaluation

Proprio baseline, four environments × 300 policy steps:

- actor/critic/action `480/495/29`;
- mean vx `0.7971276 m/s` for `0.8 m/s` command;
- actual/command ratio `0.9964094`;
- XY tracking error `0.0813951 m/s`;
- yaw tracking error `0.0777784 rad/s`;
- minimum root height `0.7364473 m`;
- terminations `0/4`;
- non-finite actor/critic/action values `0/0/0`.

Canonical environment smokes confirmed action 29, Teacher/Student dimensions,
finite observations/rewards, friction transitions, asymmetric friction and
exact material-label matching. The 100-iteration training results are in
sections 17–18.

The selected offline gated candidate has not yet undergone a separate long
fixed-policy Isaac scenario matrix with its deployment governor. The current
Isaac PPO task does not place that governor inside the rollout loop.

## 27. Ablations

Executed evidence includes:

- audited proprio baseline;
- ideal/raw-force dimension and migration regression;
- Teacher and randomized Student RSL training;
- Teacher-only and Student-rollout DAgger datasets;
- unbalanced vs class-balanced slip distillation;
- confidence-unsupervised vs confidence-supervised Student;
- governor-disabled baseline;
- RSL Student, unbalanced DAgger, balanced short DAgger and 50-epoch
  transition-aware candidates in MuJoCo;
- final dropout/bias/delay/cross-axis combined tactile stage and invalid-sensor
  scenarios.

The full 25-configuration multi-seed Isaac sweep is registered but was not
executed. No unrun row is reported as an experimental result.

## 28. MuJoCo evaluation

The bridge:

1. calls `mj_contactForce()`;
2. selects only contacts between a foot subtree and static ground;
3. converts contact-frame force to world using `contact.frame`;
4. applies the correct geom1/geom2 force sign;
5. rotates world force into the ankle-roll local frame;
6. returns the canonical signed six-axis vector.

All policies were fixed; MuJoCo performed no training or fine-tuning.
Final comparison used nine 3-second scenarios and three seeds:

| Scenario | Baseline/full falls | Velocity error baseline/full | Slip proxy rate baseline/full | Speed scale baseline/full |
|---|---:|---:|---:|---:|
| high friction | `0/0` | `0.1222/0.1324` | `0.0767/0.0933` | `1.0000/1.0000` |
| low friction | `1/0` | `0.8101/0.5086` | `0.2770/0.5500` | `1.0000/0.3295` |
| friction drop | `1/1` | `0.3056/0.2779` | `0.2188/0.2541` | `1.0000/0.7750` |
| friction recovery | `0/0` | `0.2661/0.2301` | `0.3400/0.3767` | `1.0000/0.4323` |
| asymmetric | `0/0` | `0.2216/0.1240` | `0.3800/0.2433` | `1.0000/0.4382` |
| turn | `0/0` | `0.0822/0.1135` | `0.0733/0.0667` | `1.0000/0.7992` |
| lateral | `0/0` | `0.1708/0.1597` | `0.0633/0.0567` | `1.0000/0.8684` |
| full tactile randomization | `0/0` | `0.0863/0.1051` | `0.0667/0.0822` | `1.0000/0.9320` |
| sensor invalid | `0/0` | `0.0863/0.1110` | `0.0667/0.0767` | `1.0000/0.5123` |

There were no non-finite values in 27 candidate runs. Low-friction survival
improved from `0/3` to `3/3`; high-friction scale remained exactly 1.0.
The extreme `0.9 -> 0.08` transition still fell `3/3`. Its trajectory shows
detection around 140 ms after the transition and scale reaching 0.22, but the
current actor does not recover from the already established 0.6 m/s motion.

Because baseline low-friction episodes terminate early, slip-rate comparisons
there have different exposure durations and must not be interpreted alone.
The paired CSV includes sample standard deviation and approximate 95% CI
half-width; three seeds are descriptive, not definitive.

## 29. Export commands

```bash
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python \
  scripts/traction/export_traction_policy.py \
  --checkpoint \
    artifacts/canonical_traction_20260731/transition_balanced_confidence_teacher_dagger_distill_50epoch/best.pt \
  --output_dir artifacts/canonical_traction_20260731/export_full_candidate_final \
  --sample_dataset \
    artifacts/canonical_traction_20260731/warmstart_dagger_transition_dataset_16000.npz \
  --training_status trained_candidate
```

Final hashes:

- ONNX:
  `ee1a4da35d1e16118115ff7b6fb433a458aca9304550b588d9bed77d1cad8d00`;
- TorchScript:
  `7394cd7d62ee1725afe55bd7247fbfd8743f88b8dc587a19fce46fa70c3d926a`.

Maximum export differences:

- ONNX action `4.768372×10^-7`;
- ONNX slip `5.960464×10^-8`;
- ONNX traction/confidence `0/0`;
- all TorchScript outputs `0`.

## 30. Deployment input/output schema

`DualFootForceInput`:

```text
timestamp
left_force_xyz[3], right_force_xyz[3]       signed N
left_valid, right_valid
left_age, right_age                         seconds
left_source, right_source
```

Policy input is float32 `[1,1590]`, representing 15×106. Policy outputs:

```text
action[1,29]
slip_probability[1,2]
traction_score[1,1]
sensor_confidence[1,1]
```

Runtime output also includes adjusted command, acceleration/deceleration
limits, yaw limit, speed scale, push-off scale, governor state, joint-position
target and safety flags.

The export directory contains `metadata.json`, `observation_schema.json`,
`tactile_randomization.json`, `command_governor.json`, ONNX, TorchScript and
README. Isaac, MuJoCo, recorded-force and future calibrated BLE adapters share
the same force/schema classes.

## 31. Single-foot offline replay

Raw Hall parser self-test:

```text
parsed_frames=1
hall_shape=[15,3]
temperature_first_c=25.0
sequence=1
calibrated_force_created=false
```

Final fixed-policy offline replay on one recorded simulation environment:

```text
mode=offline_only_no_robot_control
samples=250
nonfinite=0
maximum_action_abs=3.0799787
governor_states=[0]
```

The dual-foot aggregator marks a missing foot invalid with old age; it does not
copy the physical single-foot data. Real Hall-only recordings cannot enter the
force policy until a measured Hall-to-force calibration is supplied.

## 32. Known limitations

- The physical sensor layout and Hall-axis signs are provisional normalized
  topology, not metrology.
- There is no real Hall-to-net-force calibration.
- There is only one physical foot sensor.
- Foot slip uses an ankle rigid-body velocity proxy.
- Full formal Teacher/Student convergence was not run.
- The selected gated candidate was distilled offline; the independently run
  RSL PPO Student does not provide the selected auxiliary-head checkpoint.
- The deployment governor is not yet part of the Isaac PPO rollout loop, so
  joint actor/governor on-policy fine-tuning remains incomplete.
- The extreme abrupt friction-drop recovery failed in MuJoCo.
- Three-seed, three-second Sim2Sim is a short engineering matrix, not a
  publication-scale experiment.
- No actual G1 actuation was performed.

## 33. Uncompleted items

- Teacher 5000-iteration and Student 12000-iteration formal training.
- On-policy fine-tuning initialized from the confidence-supervised distilled
  gated Student with the governor active inside Isaac.
- Complete 25-configuration multi-seed Isaac ablation sweep.
- Long-horizon MuJoCo evaluation after the recovery-policy training above.
- Live dual-foot BLE acquisition and calibrated-force validation.
- Real G1 hardware-in-the-loop and robot control, intentionally out of scope
  here.

No values are supplied for these unrun experiments.

## 34. Hardware/data still required before G1 integration

1. A second foot sensor with confirmed left/right mounting.
2. PCB/CAD metrology for all P0–P14 positions and verified channel order.
3. Per-foot Hall XYZ axis/sign verification.
4. Temperature-spanning, multi-axis load-cell calibration data.
5. A versioned Hall `[15,3]`, temperature `[15]` to net-force `[3]` model with
   held-out error and saturation limits.
6. Measured sampling rate, delay, packet loss, bias, drift, cross-axis coupling
   and noise statistics to replace provisional randomization ranges.
7. Clock/latency characterization between BLE acquisition and the 50 Hz
   control loop.
8. Bench-only force comparison and timeout/dropout tests before any robot
   actuation.
9. Separate approval and safety procedure for eventual G1 control.

## 35. One-command reproduction sequence

Core software validation:

```bash
cd /home/mosense/guo/unitree_rl_lab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python -m pytest -q \
  scripts/tests/test_traction_core.py \
  scripts/tests/test_traction_deployment.py \
  scripts/tests/test_traction_evaluation.py \
  scripts/tests/test_traction_networks_governor.py \
  scripts/tests/test_traction_sensor_schema.py

cd /home/mosense/guo/unitree_mujoco
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python -m pytest -q \
  simulate_python/test/test_traction_force_bridge.py

/home/mosense/miniconda3/envs/isaaclab-v2/bin/python \
  simulate_python/run_traction_matrix.py \
  --policy \
    /home/mosense/guo/unitree_rl_lab/artifacts/canonical_traction_20260731/export_full_candidate_final/traction_student.ts \
  --duration_s 3.0 \
  --seeds 20260731 20260732 20260733 \
  --output_dir \
    /home/mosense/guo/unitree_rl_lab/artifacts/canonical_traction_20260731/reproduced_final_matrix
```

Actual final static checks in this run:

- canonical Python tests: `39 passed in 2.27 s`;
- MuJoCo bridge tests: `4 passed in 0.74 s`;
- C++ bridge compiled with
  `-std=c++17 -Wall -Wextra -Werror`;
- raw-Hall self-test passed;
- final offline replay: 250 samples, zero non-finite values;
- final MuJoCo candidate matrix: 27 runs, zero non-finite values.

This state is suitable for continued long Isaac recovery training and for
waiting on measured dual-foot calibration/hardware. It is not authorization
to control a real G1.
