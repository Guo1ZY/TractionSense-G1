# HIGH-only frozen-Teacher PPO anchor

This training path retains the validated `speedboost112` gait on the two
physical high-friction patches while leaving the low-friction capture phase
entirely reward-driven.

Data flow:

```text
Actor policy group (1864-D Hall + proprio, motion-feedback tail)
  |-- deployable Actor -> sampled action -> environment
  `-- private copy -> replace tail with real Hall packet ages
                   -> frozen speedboost112 Teacher (one call/step)
                   -> clamp [-3,3] -> delta cap -> 29-D rollout cache

private spatial stage: HIGH_START/HIGH_END -> anchor mask 1
                       LOW/other           -> anchor mask 0
```

Neither the Teacher target nor physical stage is inserted into the Actor
observation.  The custom algorithm also rejects any Actor observation-group
mapping other than exactly `actor: [policy]`.

## Stage-S1 command

```bash
/home/mosense/miniconda3/envs/isaaclab-v2/bin/python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionMildAnchored \
  --num_envs 512 \
  --seed 426 \
  --max_iterations 50 \
  --resume_checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_spatial_transition/2026-08-10_18-47-36_stage_s1_capture_lr1e5_from75/model_100.pt \
  --initial_actor_std 0.08 \
  --anchor_loss_coef 1.0 \
  --anchor_delta_cap 0.25 \
  --run_name stage_s1_frozen_speedboost_anchor
```

For Stage-S2, use the `SpatialFrictionMediumAnchored` task.  Its default
target delta cap is `0.30`; both lambda and cap remain CLI-overridable.

Each run writes `params/high_friction_anchor_manifest.json`, and every RSL
checkpoint embeds the same audit metadata under `high_friction_anchor`.  The
manifest includes the Teacher artifact hash, original ONNX hash, conversion
parity, RSL-RL version, action bounds, stage mask, learning-rate bounds, and
explicit no-truth-leakage declarations.
