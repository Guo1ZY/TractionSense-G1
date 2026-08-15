# Hall cadence--stride friction course

## Purpose

This is an isolated alternative to the existing low-friction deceleration
curriculum.  Its hypothesis is

`forward speed ~= step cadence x step length`.

On the medium-friction patch, a feasible response may therefore be **higher
cadence with shorter steps** while still tracking the same requested velocity.
The curriculum does not prescribe either cadence, step length, or a gait
period.  It also never rewards body speed above the request.

The three intended causal phases are:

1. **HighStart:** settle the frozen fast-base gait near the 0.80 m/s request.
2. **Low:** infer the changed Hall time/space pattern, shorten the usable step
   and adjust cadence while minimizing actual contact-point slip and impacts.
   If an extreme sustained disturbance makes 0.80 m/s infeasible, the fall,
   posture, slip and impact constraints may naturally trade away speed; there
   is no hard-coded 0.24/0.32 m/s command or reward target.
3. **HighEnd:** with the request still unchanged, close unnecessary corrective
   authority and recover the nominal cadence/stride combination.

## Isolation and observation contract

Training task:

`Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride`

Long visualization task:

`Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo`

Both use the existing FastBase Hall actor and keep its actor input exactly
1864 dimensions and its privileged critic input at 570 dimensions.  Actor
observations contain Hall `Bx/By/Bz` history,
sample-health timing and proprioception.  They do **not** contain friction,
force, contact point, slip, course stage or a risk label.  True material/contact
quantities are used only by simulator rewards, the critic and the existing
LOW/HIGH auxiliary supervision.

Only the new cadence/stride environment sets
`hall_sensor_cfg.contact_distribution_mode="detailed"`.  It distributes each
raw normal/friction contact patch over the 15 Hall sites before the Scheme-A
TPU/magnetic forward model.  The default remains `aggregate`, so old training,
play and checkpoints do not silently change.

The new reward removes the inherited stage speed cap, spatial capture-speed
reward, fixed-period gait term, low-traction touchdown-count penalty and old
ankle-origin slip proxies.  It retains symmetric requested-velocity tracking
and constrains:

- rigid-foot contact-point tangential speed using `v_COM + omega x r`;
- fall, roll/pitch, base-height and base angular motion;
- lateral velocity and accumulated cross-track error;
- clipped contact-force rate (impact) and action slew;
- privileged friction-cone utilization during training only.

The contact-point metric currently assumes a static ground patch.  No-contact
rows are invalid/zero, malformed contact buffers fail closed, and no derived
force is exposed to the Hall actor.

## Runner and resume behavior

`FootTractionHallSpatialCadenceStridePPORunnerCfg` inherits the calibrated
FastBase runner: the validated speedboost teacher is frozen, while the Hall
gate/residual, critic and action standard deviation remain trainable.  It uses
64 steps per environment, 1000 iterations and saves every 25 iterations.

It deliberately does not inherit the `GateBceOnly` 12-update continuation
guard.  A fresh run starts from the config-owned frozen teacher and is a real
1000-iteration optimization, not a fully frozen 12-iteration placeholder.  A
strict resume should use only a schema-compatible checkpoint from this same
FastBase actor family; optimizer state is optional and should remain off after
an unstable run.

```bash
cd /home/mosense/guo/unitree_rl_lab
TERM=xterm PYTHONPATH=source/unitree_rl_lab \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride \
  --headless --device cuda:0 --num_envs 512 --seed 442 \
  --run_name cadence_stride_r1
```

Strict continuation, without optimizer state:

```bash
TERM=xterm PYTHONPATH=source/unitree_rl_lab \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMediumDenseCadenceStride \
  --headless --device cuda:0 --num_envs 512 --seed 442 \
  --resume_checkpoint /absolute/path/to/model_XXX.pt \
  --run_name cadence_stride_r2
```

## Long, clearly colored visualization

The ordinary play/evaluation geometry is unchanged.  The dedicated long demo
uses opaque blue/yellow/blue colliders:

- HighStart (blue): `x=[-6,0]`, `mu=0.90`;
- Low (yellow): `x=[0,6]`, `mu=0.28`;
- HighEnd (blue): `x=[6,18]`, `mu=0.90`.

Every patch is 3.2 m wide. It resets at `x=[-5.5,-5.1]`, succeeds at `x=17.5`,
uses four environments with 12 m spacing, and has a 65 s episode horizon.  The
longer horizon prevents a 0.4 m/s safety gait from timing out before the final
high-friction patch.
Training through the LongDemo registry ID still resolves to the short
MediumDense environment; only its `play_env_cfg_entry_point` uses long geometry.

```bash
cd /home/mosense/guo/unitree_rl_lab
TERM=xterm PYTHONPATH=source/unitree_rl_lab \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo \
  --checkpoint /absolute/path/to/model_XXX.pt \
  --num_envs 1 --device cuda:0 \
  --viewer_eye 11.0 -28.0 15.0 --viewer_lookat 6.0 0.0 0.4
```

For a roughly full-course recording (control step is 0.02 s):

```bash
TERM=xterm PYTHONPATH=source/unitree_rl_lab \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python3 \
  scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideLongDemo \
  --checkpoint /absolute/path/to/model_XXX.pt \
  --num_envs 1 --device cuda:0 --headless \
  --video --video_length 3500
```

## Formal objective and acceptance contract

The external command is held at **0.80 m/s in every High--Low--High and
Low--High--Low phase**.  Neither the material label nor a privileged estimate
of friction is allowed to alter that requested command.  The deployed actor
must infer the change from multi-frame Hall `Bx/By/Bz`, Hall sample-health
metadata and proprioceptive history.

The policy is free to change cadence, step/stride length, support timing and,
when the available traction makes tracking unsafe, its realized body speed.
There is deliberately no acceptance gate saying that Low speed must be below
a fixed value, or that cadence must always increase.  A shorter, faster gait is
one plausible learned response, not a hard-coded controller rule.

Candidate comparison against the original Unitree policy must use identical
requested commands, initial states, terrain/material sequence, episode length,
random seeds and dynamics/fault draws.  Report first-fall survival and exclude
post-reset samples from primary locomotion statistics.  A candidate advances
only when held-out multi-seed evaluation shows:

- no NaN and no regression in fall safety;
- strictly better survival/effective traversed distance on friction changes;
- HighStart preserves the original high-friction capability (target loss no
  greater than 5%);
- after returning to HighEnd, speed and step length recover to at least 90% of
  their own HighStart values without a privileged material trigger;
- Low behavior reduces corrected contact-point slip and keeps tilt, lateral
  drift, impact and action slew no worse than the baseline;
- nominal and Hall-fault tests are reported separately, including sensor
  dropout health-envelope intervention rather than hiding failures with reset
  averages.

Only a candidate that passes the matched Isaac tests proceeds to MuJoCo.  The
real-robot package remains inactive until sim-to-sim and interface/calibration
checks pass.

## What to measure after training

Do not judge this hypothesis from average speed alone.  Report each H--L--H
phase separately:

- requested-vx error and any overspeed;
- falls, roll/pitch, lateral drift and HighEnd recovery time;
- valid contact-point tangential slip and peak/integrated impact;
- touchdown-derived cadence and contact-to-contact step length;
- Hall gate/health traces without treating Hall as a force sensor.

The hypothesis is supported when the Low phase achieves a better
safety--mobility trade-off without increasing fall, slip, impact or HighEnd
residual authority.  Cadence and step length must be reported so that the
learned mechanism is visible, but no particular direction is required by the
gate.  If a cadence-up/step-length-down pattern emerges, it is an observed
strategy rather than a prescribed target.
