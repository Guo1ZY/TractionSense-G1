# Traction-adaptive tactile policy: current-code audit

Date: 2026-07-31

Scope: read-only audit plus reproducible proprio-baseline rollout.
Branch: `feature/traction-adaptive-tactile-policy`

## Repository boundaries and preservation

- `/home/mosense/guo` is not a Git repository. Its nested repositories are
  `unitree_rl_lab`, `unitree_mujoco`, `rsl_rl`, and `unitree_sdk2`.
- The feature branch was created independently in `unitree_rl_lab` and
  `unitree_mujoco`.
- Both repositories already contained extensive tracked modifications and
  untracked checkpoints/configurations before this audit. They were retained
  in place; no reset, checkout, deletion, or bulk staging was performed.
- `/home/mosense/guo_1/vola_sensor` is not a Git repository. The only nested
  relevant Git worktree is the unrelated
  `FPC-UI-ALL/FPC_UI` chest-sensor application, which was inspected read-only.
- Generated rollout artifacts live below ignored `artifacts/`; existing
  checkpoints, exported models, logs, and videos were not overwritten.

## Preserved G1 asset and action contract

- Asset:
  `/home/mosense/Downloads/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`
- Runtime action dimension: **29**.
- Action type: joint-position target with default-pose offset.
- Action scale: **0.25 rad per unit action**.
- Runtime action/joint order:

| Index | Joint |
|---:|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `right_hip_pitch_joint` |
| 2 | `waist_yaw_joint` |
| 3 | `left_hip_roll_joint` |
| 4 | `right_hip_roll_joint` |
| 5 | `waist_roll_joint` |
| 6 | `left_hip_yaw_joint` |
| 7 | `right_hip_yaw_joint` |
| 8 | `waist_pitch_joint` |
| 9 | `left_knee_joint` |
| 10 | `right_knee_joint` |
| 11 | `left_shoulder_pitch_joint` |
| 12 | `right_shoulder_pitch_joint` |
| 13 | `left_ankle_pitch_joint` |
| 14 | `right_ankle_pitch_joint` |
| 15 | `left_shoulder_roll_joint` |
| 16 | `right_shoulder_roll_joint` |
| 17 | `left_ankle_roll_joint` |
| 18 | `right_ankle_roll_joint` |
| 19 | `left_shoulder_yaw_joint` |
| 20 | `right_shoulder_yaw_joint` |
| 21 | `left_elbow_joint` |
| 22 | `right_elbow_joint` |
| 23 | `left_wrist_roll_joint` |
| 24 | `right_wrist_roll_joint` |
| 25 | `left_wrist_pitch_joint` |
| 26 | `right_wrist_pitch_joint` |
| 27 | `left_wrist_yaw_joint` |
| 28 | `right_wrist_yaw_joint` |

The runtime order above is the Isaac articulation/action order. It is not the
same list order as `joint_sdk_names`; deployment must retain its explicit
runtime-to-SDK mapping.

## Default pose and PD

Non-zero default positions are:

- hip pitch `-0.1 rad` (left/right);
- knee `+0.3 rad` (left/right);
- ankle pitch `-0.2 rad` (left/right);
- shoulder pitch `+0.3 rad` (left/right);
- shoulder roll `+0.25/-0.25 rad` (left/right);
- elbow `+0.97 rad` (left/right);
- wrist roll `+0.15/-0.15 rad` (left/right).

Runtime stiffness/damping pairs are:

- hip pitch/roll/yaw: `100/2`;
- knee: `150/4`;
- ankle pitch/roll: `40/2`;
- waist yaw: `200/5`;
- waist roll/pitch: `40/5`;
- shoulder, elbow, and all wrist joints: `40/1`.

The USD mass/inertia and collision asset was not modified. With the existing
startup torso-mass randomization (`-1…+3 kg`), the seed-20260731 four-env
runtime total masses were `34.4453…36.2759 kg` (mean `35.2793 kg`).

## Baseline observation and timing

The runtime ObservationManager confirmed term-major concatenation. Every term
contains five samples ordered **oldest to newest**, then the next term follows.
After reset, Isaac Lab's circular buffer seeds all history slots with the first
post-reset sample; it does not leak the previous episode.

Actor:

| Flattened range | Term | Per-frame dim × history |
|---|---|---:|
| `[0:15)` | base angular velocity, scale `0.2` | `3 × 5` |
| `[15:30)` | projected gravity | `3 × 5` |
| `[30:45)` | velocity command `[vx,vy,yaw]` | `3 × 5` |
| `[45:190)` | joint position relative to default | `29 × 5` |
| `[190:335)` | joint velocity, scale `0.05` | `29 × 5` |
| `[335:480)` | previous action | `29 × 5` |

Actor total: **480**.

Critic prepends `base_lin_vel (3 × 5)` and otherwise uses the same ordered
terms, for a total of **495**.

- Physics timestep: **0.005 s**.
- Decimation: **4**.
- Policy/control timestep: **0.020 s (50 Hz)**.
- Episode duration: **20 s**.
- Actor/critic empirical normalization: disabled/identity in the checkpoint.
- Environment observation corruption is disabled for the measured play run.
- Raw-force task clipping is `[-2, 2]` after body-weight normalization.
- RSL-RL environment action clip uses the runner configuration; the policy
  Gaussian mean itself is not artificially clipped in the checkpoint.

## Checkpoint and entry points

- Baseline checkpoint:
  `/home/mosense/guo/unitree_rl_lab/model/rl/model_49999.pt`
- SHA-256:
  `c508af7910a69e2bc06111caaa677d5bea521bfb52fc654d82d38b499e2ae99b`
- Checkpoint iteration: `49999`; finite actor/critic tensors.
- Actor MLP: `480 → 512 → 256 → 128 → 29`, ELU.
- Critic MLP: `495 → 512 → 256 → 128 → 1`, ELU.
- Train: `scripts/rsl_rl/train.py`.
- Play and ONNX/TorchScript export: `scripts/rsl_rl/play.py`.
- Partial warm start:
  `unitree_rl_lab/utils/partial_checkpoint.py`.
- Friction matrix/Isaac evaluation:
  `scripts/rsl_rl/eval_friction_matrix.py`.
- Existing standalone distillation:
  `/home/mosense/guo/scripts/distill_traction_student.py`.
- Existing DAgger-like offline fine-tuning:
  `/home/mosense/guo/scripts/fine_tune_shared_magnetic_dagger.py`.
- C++ deploy:
  `deploy/robots/g1_29dof`.
- MuJoCo simulation:
  `/home/mosense/guo/unitree_mujoco` plus top-level research evaluators.

## Reproduced proprio baseline

Command:

```bash
python scripts/tests/audit_baseline_rollout.py \
  --checkpoint model/rl/model_49999.pt \
  --output-dir artifacts/baseline_20260731/metrics_seed20260731 \
  --num-envs 4 --steps 300 --seed 20260731 \
  --command-vx 0.8 --headless
```

Measured over four environments and 300 policy steps:

| Quantity | Actual result |
|---|---:|
| actor / critic / action dimensions | `480 / 495 / 29` |
| non-finite actor / critic / action values | `0 / 0 / 0` |
| terminations | `0` |
| environments with a termination | `0 / 4` |
| mean forward velocity | `0.7971276 m/s` |
| mean actual/command forward-velocity ratio | `0.9964094` |
| mean XY velocity tracking error | `0.0813951 m/s` |
| maximum sampled XY velocity tracking error | `0.5490065 m/s` |
| mean yaw tracking error | `0.0777784 rad/s` |
| mean projected-gravity XY norm | `0.0265500` |
| mean root height | `0.7628659 m` |
| minimum sampled root height | `0.7364473 m` |

Artifacts:

- `artifacts/baseline_20260731/metrics_seed20260731/baseline_metrics.json`
- `artifacts/baseline_20260731/metrics_seed20260731/baseline_trajectory.npz`
- `artifacts/baseline_20260731/metrics_seed20260731/env_cfg.yaml`
- `artifacts/baseline_20260731/checkpoint/videos/play/rl-video-step-0.mp4`

The video is 1280×720, 50 fps, 5.98 s. The original viewer camera is very
wide, so quantitative acceptance is based on the recorded tensors, not a
visual claim.

## Current local three-axis force path

`velocity_raw_foot_env_cfg.py` already provides a useful compatibility
milestone:

- dedicated left and right ContactSensors on
  `left_ankle_roll_link` and `right_ankle_roll_link`;
- ground-filtered normal plus friction force;
- batched world-to-link rotation using `quat_apply_inverse`;
- signed order
  `[L_Fx,L_Fy,L_Fz,R_Fx,R_Fy,R_Fz]`;
- body-weight normalization;
- actor `510`, critic `525`, action `29`;
- exact switch-off path back to `480/495`.

The code has a finite Isaac smoke entry point. The remaining checkpoint loader
still assumes prefix copying internally. Although that happens to match the
current term-major layout, the new work must generate and consume an explicit
old-to-new index map.

## Real single-foot sensor audit

Current BLE protocol in `/home/mosense/guo_1/vola_sensor/vis`:

- device name: `FootSensor15`;
- Notify characteristic:
  `0000ab01-0000-1000-8000-00805f9b34fb`;
- frame length: `125 B`;
- accepted header bytes: byte 0 `0x7D`, byte 2 `0xF0`, byte 3 `0x02`;
- payload: bytes `[4:124)`, `120 B = 15 × 8 B`;
- per channel: big-endian `>hhhh` = `T_x10, Hall_X, Hall_Y, Hall_Z`;
- Hall wire values: signed int16, then continuity unwrap to bounded int32;
- temperature: `T_x10 / 10 °C`, UI-valid range `[-40,125] °C`;
- timestamp: host `time.monotonic()` at decoded-frame arrival;
- no device timestamp or packet sequence is present in the decoded payload;
- sampling rate is measured from arrival intervals, not guaranteed by the
  protocol;
- baseline: multi-frame median in the new dashboard;
- filtering: axis-wise dead zone plus asymmetric EMA in the new dashboard;
- older super-resolution path also uses its external Kalman/filter stack.

Defects/gaps found:

- the old `ble_viz_superres_hot.py` swaps channel 12/13 X before filtering and
  swaps them again after filtering, cancelling the correction;
- no explicit BLE-channel-to-`P0…P14` configuration exists;
- no canonical per-channel 3×3 axis transform exists;
- the dashboard's region index split is `0:5`, `5:11`, `11:15`, which conflicts
  with the specified three five-point regions and must become `0:5`, `5:10`,
  `10:15`;
- there is no Hall-to-point-force or Hall-array-to-total-`Fx,Fy,Fz` calibrated
  model/checkpoint in `vola_sensor`;
- displayed “force” is currently a Hall-delta magnitude proxy, not Newtons.

CAD inspection found a 10 mm insole solid (`80.039 × 215.021 × 10.000 mm`) and
a multi-body 9.4 mm sole/base model, but neither contains named 15-point Hall
locations. Therefore the dashboard coordinates cannot be promoted to physical
millimetres. The canonical first layout must be explicitly marked provisional
and normalized.

## Current Teacher/Student/governor/Sim2Sim gaps

- The current “Teacher” appends one effective-friction scalar to a 640-D MLP;
  it is not the requested configurable privileged traction encoder.
- The current Student remains an MLP/offline distillation path. There is no
  canonical GRU/TCN force-history encoder producing per-foot slip probability,
  traction score, and confidence.
- The legacy “magnetic array proxy” fabricates Hall-like arrays from ideal
  force. It is unsuitable for the new force-only first version and must not be
  used as magnetic/Hall truth.
- Structured sensor noise currently covers gain, bias, low-pass, delay, and
  dropout on magnitude features, but not the complete signed-three-axis model
  (cross-axis matrix, installation rotation, drift, load noise, saturation,
  spike, independent validity/age, curriculum).
- Existing deploy governors are two-state/sensorless or direct speed caps.
  They do not implement the requested slip/traction/confidence-driven fast
  reduction and slow recovery interface.
- MuJoCo currently writes magnitude packets (`F0T1`) or a synthetic magnetic
  packet (`F0M1`). It calls `mj_contactForce`, but its canonical output is not
  yet the signed six-axis ankle-local force schema shared with Isaac.

## Test environment notes

- GPU: NVIDIA RTX 5070 Ti, 16 GB.
- Free disk at audit start: approximately 736 GB.
- Existing pure-software tests: **28 passed in 0.91 s**.
- Because the machine's ROS Jazzy pytest plugin is visible to the Python 3.11
  Isaac environment but its `lark` dependency is not, reproducible test
  commands set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. This is an environment
  isolation issue, not a project test failure.
