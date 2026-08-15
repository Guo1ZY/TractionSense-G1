# Scheme-A detailed contact driver (Isaac Sim 5.1)

## Why this exists

The original Scheme-A path reduced each foot to one total force and one mean
contact point, then spread that load with a single Gaussian over P00--P14.  It
is stable and cheap, but toe/heel edge contacts and simultaneous contact
patches can collapse to nearly the same synthetic Hall pattern.

`HallFootSensorCfg.contact_distribution_mode` now selects:

- `aggregate` (default): unchanged total-force + mean-point path.  This keeps
  existing checkpoints and large-batch evaluation behavior unchanged.
- `detailed`: current PhysX normal contact patches and friction anchors are
  spatially assigned one by one to the 15 Hall sites.  The actor schema remains
  `[N,2,15,3]` per frame and receives only Hall `Bx/By/Bz` history.

## Current API contract

The installed runtime is Isaac Sim 5.1 / Isaac Lab 2.3.2 with
`omni.physics.tensors` 107.3.  Isaac Lab's own `ContactSensor` uses the same
current `RigidContactView` methods:

- `get_contact_data(dt)`: normal-force magnitude, point, normal, separation,
  pair count, pair start.  The point force is `magnitude * normal`, in N when
  `dt` is the physics step.
- `get_friction_data(dt)`: tangential force, friction anchor point, pair count,
  pair start, also in N.

Normal patches and friction anchors are independent ragged streams and are not
paired by index.  The adapter distributes each stream using its own positions,
then adds the resulting local Hall-site forces.

For every point `k`, stable Gaussian-softmax weights satisfy
`sum_s w[k,s] = 1`.  Therefore:

`sum_s F_local[env,foot,s] = R_world_from_foot^T * sum_k F_world[k]`.

The adapter verifies each raw stream against `force_matrix_w` or
`friction_forces_w`, rejects malformed count/start buffers, rejects referenced
NaN samples, and optionally rejects a completely full raw buffer.  It never
silently falls back to aggregate mode.

## Configuration and smoke test

```python
cfg.hall_sensor_cfg.contact_distribution_mode = "detailed"
sync_hall_sensor_cfg_to_policy_terms(cfg.observations, cfg.hall_sensor_cfg)
```

```bash
cd /home/mosense/guo/unitree_rl_lab
TERM=xterm PYTHONPATH=source/unitree_rl_lab \
  /home/mosense/miniconda3/envs/isaaclab-v2/bin/python3 \
  scripts/tests/smoke_hall_foot_sensor_env.py \
  --headless --device cuda:0 --num_envs 2 --steps 16 --seed 123 \
  --detailed_contact
```

The smoke test must report `policy_dim: 1864`, Hall shape `(N,2,15,3)`, and
`contact_distribution_mode: detailed`.

## Scope and cost

Detailed mode adds raw contact-buffer queries, ragged-buffer gathers and one
point-to-15-site softmax.  It remains much cheaper than creating 60 dynamic
magnet bodies per robot, but it is not free.  Keep `aggregate` for legacy
checkpoint comparison and first benchmark detailed throughput at the intended
environment count before a full PPO run.  Scheme A still approximates the TPU
with independent Kelvin--Voigt site states; detailed contact improves the
spatial mechanical driver but does not turn it into a deformable finite-element
model or a Hall-to-force estimator.

### Initial throughput check (2026-08-11)

On the installed RTX 5070 Ti, the same 512-environment, 256-policy-step
headless smoke measured:

- aggregate: 7.058 s, 18,570 environment-steps/s;
- detailed: 7.541 s, 17,381 environment-steps/s.

The measured detailed overhead was about 6.8%.  This smoke verifies the batch
path and provides an early cost bound; it is not the final walking-contact
benchmark.  A policy-driven H--L--H A/B is still required because its contact
buffer occupancy and Hall signal content are higher than the zero-action
smoke.
