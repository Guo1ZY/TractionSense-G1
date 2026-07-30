# Two-surface fast demo

This branch targets the shortest path to one visible real-G1 result:

> With the same forward joystick input, the robot automatically caps itself at
> `0.20 m/s` on one known slippery floor and at `0.50 m/s` on one known
> high-traction floor.

It deliberately solves **binary recognition of the exact two demonstration
floors**, not general friction estimation. No Hall or plantar signal is used.
The input is the existing 480-D history of IMU, command, joint position, joint
velocity and previous action. A small classifier outputs `p_low`; a hysteretic
governor converts it into the two speed caps.

```text
same joystick command
        │
480-D proprioceptive history ──► p_low ──► hysteresis ──┬─ LOW  → 0.20 m/s
                                                        └─ HIGH → 0.50 m/s
```

## 0. Build

Build the existing G1 controller after checking out this branch:

```bash
cd deploy/robots/g1_29dof
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
cd ..
```

## 1. Confirm both fixed caps

Use a load-rated overhead harness, a separate emergency-stop operator, a clear
straight test lane and zero lateral/yaw command.

```bash
export G1_REAL_TEST_ACK=YES

# Stage A: conservative 0.20/0.35 m/s
G1_FAST_DEMO_PROFILE=safe ./two_surface_fast_demo.sh manual-low \
  --network <interface> --log
G1_FAST_DEMO_PROFILE=safe ./two_surface_fast_demo.sh manual-high \
  --network <interface> --log

# Stage B: visibly separated 0.20/0.50 m/s
G1_FAST_DEMO_PROFILE=clear ./two_surface_fast_demo.sh manual-low \
  --network <interface> --log
G1_FAST_DEMO_PROFILE=clear ./two_surface_fast_demo.sh manual-high \
  --network <interface> --log
```

Hold the same forward-stick position in both runs. Do not proceed until both
fixed modes are stable.

## 2. Collect the exact two floors

Use the same robot, shoes, payload, control policy, forward-stick input and
`0.20 m/s` cap. Record at least 20 seconds of steady walking per trial.

```bash
./two_surface_fast_demo.sh collect-low  --network <interface> --log
./two_surface_fast_demo.sh collect-high --network <interface> --log
```

One trial per floor is the minimum fast-demo path. Two or three separate trials
per floor are strongly preferred. All trials under
`logs/real/two_surface_fast_demo/` are used by the training command.

## 3. Fit and install the binary classifier

```bash
# Point this at an environment containing numpy, PyTorch and ONNX.
export G1_PYTHON=/path/to/isaaclab/environment/bin/python

./two_surface_fast_demo.sh train
./two_surface_fast_demo.sh status
```

Installation occurs only when the trainer reaches its current in-sample
acceptance criteria:

- balanced accuracy at least `0.90`;
- mean `p_low >= 0.65` on LOW;
- mean `p_low <= 0.35` on HIGH.

Because this fast branch intentionally learns the two test surfaces, these
numbers are a fit check—not evidence of generalization to unseen materials.

## 4. Automatic demonstration

```bash
# Explicitly repeat the safe automatic trial first.
G1_FAST_DEMO_PROFILE=safe ./two_surface_fast_demo.sh auto \
  --network <interface> --log

# Final visible comparison (clear is also the script default).
G1_FAST_DEMO_PROFILE=clear ./two_surface_fast_demo.sh auto \
  --network <interface> --log
```

AUTO starts conservatively at the LOW cap. While walking, the decision rules
are:

- enter LOW when `p_low >= 0.60` for `0.20 s`;
- enter HIGH when `p_low <= 0.40` for `0.80 s`;
- missing/invalid classifier feedback falls back to LOW;
- `v_y` and yaw commands remain locked to zero.

Keep one constant forward-stick input and run separate labeled trials on the
two floors. In the final `clear` profile, the terminal and governor CSV should
show `LOW`/`0.20` and `HIGH`/`0.50`, respectively.

## Tuning order

Only after the default demonstration succeeds:

1. repeat data collection if either floor flickers between states;
2. increase `G1_TRACTION_HIGH_HOLD` if LOW is falsely promoted;
3. lower the LOW cap before changing the classifier if the slippery trial is
   unstable;
4. do not exceed the `0.50 m/s` HIGH cap on this demonstration branch without
   a separate stability assessment.

The gamepad escape controls remain available:

- `RB + DOWN`: force LOW;
- `RB + UP`: force HIGH;
- `RB + LEFT`: return to AUTO/reset;
- `B`: Passive; the independent hardware emergency stop remains mandatory.
