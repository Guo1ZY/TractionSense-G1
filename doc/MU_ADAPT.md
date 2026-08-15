# Foot-MuAdapt（510 维 Fn+Ft + 结果导向奖励）

针对 MuJoCo 实测：ICE 上 `|v_xy|` 仍高，但是 **侧滑 vy**，前进 vx 差。

## 任务

`Unitree-G1-29dof-Velocity-Foot-MuAdapt`

| 项 | 设定 |
|----|------|
| Actor obs | 与 Foot-Full 相同 **510**（contact + Fn + Ft，history 5） |
| Critic | + ρ + slip_proxy |
| Track | **full**（无 slip-aware min_track 地板） |
| 新奖 | `stable_speed_bonus`、`slip_under_command`、`lateral_slip`、`track_lin_vel_x` |
| 指令 limit | vx ≤ **1.2**（在分布内） |
| 部署目录 | `deploy/.../velocity/foot_mu/`（不覆盖 foot/） |

## 训练

```bash
# smoke
/home/mosense/guo/scripts/finetune_g1_foot.sh --mu-adapt --smoke

# 正式（默认 strict resume 最新 MuAdapt 或 Full model_10900；不加载 optimizer）
/home/mosense/guo/scripts/finetune_g1_foot.sh --mu-adapt \
  --max-iterations 12000 --run-name foot_mu_adapt --num-envs 4096

# 若 std<0 / NaN 崩溃后，从最近 model_XXXX 续训（默认不带 optimizer）
/home/mosense/guo/scripts/finetune_g1_foot.sh --mu-adapt \
  --resume-checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot_mu_adapt/<run>/model_11500.pt \
  --max-iterations 8000 --run-name foot_mu_adapt_cont
```

### `std >= 0` 崩溃

`noise_std_type=scalar` 时 `std_param` 可被梯度推成负数/NaN。  
`train.py` 已加 **std guard**：每次 PPO update 前后 clamp 到 `[1e-3, 1.0]`。  
MuAdapt 另用更低 LR / entropy / grad clip。

日志：`logs/rsl_rl/unitree_g1_29dof_velocity_foot_mu_adapt/`

## 导出

```bash
cd /home/mosense/guo/unitree_rl_lab
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot-MuAdapt \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity_foot_mu_adapt/<run>/model_XXXX.pt \
  --num_envs 16 --headless
cp -v logs/.../exported/policy.onnx \
  deploy/robots/g1_29dof/config/policy/velocity/foot_mu/exported/policy.onnx
# config.yaml → policy_dir: config/policy/velocity/foot_mu
```

## MuJoCo 测协议

1. 重建 MuJoCo（写入 `/tmp/g1_base_vel.json`）：
```bash
cd /home/mosense/guo/unitree_mujoco/simulate/build && cmake --build . -j
```
2. 三终端：MuJoCo（`G1_MUJOCO_FOOT_BRIDGE=1`）/ g1_ctrl / logger
3. **满杆只推前后，左右摇杆回中**
4. Logger 标 `3` grip → 走 20s → 标 `1` ice → 走 20s
5. Summary 看 `|vx|` vs `|vy|`（不要只看 `|v|`）

```bash
export G1_MUJOCO_FOOT_BRIDGE=1
/home/mosense/guo/scripts/run_mujoco_friction.sh normal
# 另端
/home/mosense/guo/scripts/log_foot_mujoco.sh --tag mu_ab --hz 20 --mu-mode grip
```
