#!/usr/bin/env bash
# Slope/stairs NEW model (separate from transition_retention_r5).
#
# - Task:   ...TransitionRetentionSlopeStairsV1 (ramps/stairs generator,
#           1864-D motion ABI with foot Hall + [body_vy, relative_heading])
# - Agent:  FootTraction...TransitionRetentionSlopeStairsPPORunnerCfg
#           (R5 composition: FastBase + capture gate/residual + stability
#            residual, frozen teacher; capture branches remain frozen)
# - Warm start: R5 rebalanced model_399 (actor+critic weights only; no
#           optimizer/iteration), exploration std reset for new terrain
set -euo pipefail

cd "$(dirname "$0")"

exec /home/mosense/miniconda3/envs/isaaclab-v2/bin/python train.py \
  --task Unitree-G1-29dof-Velocity-Foot-TractionMagneticMotionStudent-SpatialFrictionCadenceStrideTransitionRetentionSlopeStairsV1 \
  --headless \
  --num_envs 512 \
  --max_iterations 400 \
  --seed 800 \
  --partial_checkpoint /home/mosense/guo/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_foot_traction_hall_spatial_cadence_stride_transition_retention_r5/2026-08-13_20-43-49_transition_retention_r5_rebalanced/model_399.pt \
  --initial_actor_std 0.06
