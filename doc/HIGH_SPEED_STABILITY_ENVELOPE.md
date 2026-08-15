# High-speed stability envelope

`HighSpeedStabilityEnvelope` is an optional, default-off safety supervisor for
the 1864-D Hall locomotion actor. It reads only signals that are available at
deployment: five-frame command history, IMU angular velocity and projected
gravity, the actor's action history, and relative heading. It does not read the
terrain label, friction, contact state, or force.

Enable it only for an Isaac evaluation with:

```bash
python scripts/rsl_rl/eval_spatial_friction_course.py \
  --headless --checkpoint /path/to/model.pt --num_envs 16 --steps 400 \
  --high_speed_stability_envelope \
  --summary_json artifacts/stability_eval.json \
  --dataset_npz artifacts/stability_eval.npz
```

It may be enabled together with `--hall_health_envelope`. Packet health is
applied first and stability second; the stability state machine can only reduce
the magnitude of the upstream forward command. Every policy call sees all five
command-history frames rewritten to the command actually applied.

The states and default caps are:

- `NORMAL`: no additional cap.
- `WARN`: heading error above 0.40 rad for five frames; cap 0.55 m/s.
- `LIMIT`: heading above 0.45 rad for five frames, or 0.48 rad for three;
  cap 0.40 m/s.
- `EMERGENCY`: excessive tilt, action magnitude/saturation, or the configured
  heading/roll-pitch-rate condition; cap 0.25 m/s.

More conservative states take effect immediately after their stated trigger
persistence. Returning to `NORMAL` requires heading below 0.30 rad, projected
gravity tilt below 0.10, and roll/pitch rate below 0.8 rad/s for ten frames.

## Optional straight-heading correction

The v1 envelope only attenuates forward speed. A separately isolated A/B mode
adds bounded yaw correction after the state has reached `WARN`:

```bash
--high_speed_stability_envelope --stability_heading_correction
```

For a straight command it applies
`yaw_effective = clamp(yaw_upstream - 0.8 * relative_heading, -0.40, 0.40)`.
A positive measured heading error therefore produces a negative yaw command,
and vice versa. `NORMAL` is exactly transparent. The correction is also exactly
transparent during a commanded turn. Forward `vx` retains the original
only-attenuate invariant. The gain and bound may be changed for evaluation with
`--stability_heading_gain` and `--stability_heading_yaw_cap`; none of these
options modifies a deployment YAML.

The turn-intent detector removes the mean of a causal five-sample FIFO of yaw
corrections injected by the envelope itself. This FIFO must remain aligned with
the actor's five-frame command history. Subtracting only the most recent full
correction creates an unsafe one-frame-on/five-frames-off yaw pulse and resets
the WARN/LIMIT persistence counters.

## Optional frozen recovery actor

Command yaw cannot recover every high-speed instability: a robot may have
small heading error while lateral velocity, roll/pitch rate and tilt are already
diverging. For isolated Isaac A/B evaluation, an additional default-off handoff
can ramp from the baseline action into the frozen Stage7 recovery actor when the
stability state reaches `EMERGENCY`:

```bash
--high_speed_stability_envelope \
--stability_heading_correction \
--stability_recovery_checkpoint \
  logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_handoff_recovery/2026-08-10_13-31-15_stage7a_handoff_mild_mu018_026/model_6149.pt \
--stability_recovery_command 0.16 \
--stability_recovery_blend_in_s 0.20 \
--stability_recovery_blend_out_s 0.30
```

The expert receives a private copy of the same 1864-D deployable observation.
Only its term-major command columns `(30,33,36,39,42)`,
`(31,34,37,40,43)`, and `(32,35,38,41,44)` are changed to
`[0.16, 0, 0]`. Its newest action-history sample is the actual action returned
by the causal blend on the preceding step. No friction, contact, force or
course-stage truth enters the handoff. The deterministic MLP mean is loaded
strictly from `actor_state_dict`; Gaussian sampling variance is intentionally
unused during evaluation.

This is an evaluator-only candidate and remains disabled unless the checkpoint
flag is supplied. It does not modify the robot deployment YAML. Promote it to
the robot runtime only after multi-seed physics evaluation establishes that it
does not regress the command-only envelope.

## Straight-walking limitation

The current observation defines `relative_heading` against the yaw latched at
episode reset. That signal is correct for the present straight course, but it
is not a tracking error during an intentional turn. Whenever the absolute
five-frame mean yaw command exceeds 0.05 rad/s, heading-only WARN/LIMIT checks
are disabled. The supervisor retains tilt, roll/pitch-rate, and action
emergencies because these remain valid during a yaw turn.

Before enabling heading thresholds for general turning or on the robot, add a
commanded-heading reference by integrating the commanded yaw rate and use
`wrap(measured_yaw - commanded_heading)` as the error. Do not reinterpret the
reset-relative channel as that signal.

The evaluator records `stability_state`, `stability_reason_mask`, both upstream
and effective commands, trigger measurements, persistence counters, time, and
environment ID in the optional trace/dataset. No deployment YAML is changed by
the evaluator switch.
