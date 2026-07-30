# Foot-Adaptive-V2：修复「改 μ 速度几乎不变」

## 问题根因（结合仓库实测结论）

当前 `Foot-Full` / slip-aware 路径在 MuJoCo 改摩擦后 **速度差不明显**，常见原因：

1. **`track_*_slip_aware` + `min_track_scale≈0.1`**
   一旦滑移，跟踪奖励被软掉 → 策略学会「中速保守步态」，在所有 μ 上都能存活，但 **高 μ 也没有加速动机**。

2. **Actor 看 `Ft`（切向力）≠ 地面 μ**
   稳态行走时 `Ft/Fn` 未顶到摩擦锥，高低 μ 的足底力观测量很像 → 网络难以用力通道调制速度。

3. **无「高 μ 且稳才给奖」的 outcome 项**
   只有跟踪 + 防滑惩罚时，最优常是全局中速。

4. **部署端若 foot bridge 为 0 / 超时全零**
   策略等价于无足底，只能靠本体感觉，适应更弱。

5. **速度包络**
   更高目标速度必须进入 **训练 command distribution**；deploy clamp 到 1.5 而训练只到 1.2 属于 OOD。

## V2 改动（不覆盖旧 foot 策略）

| 项 | V2 |
|----|----|
| Gym | `Unitree-G1-29dof-Velocity-Foot-Adaptive-V2` |
| Actor obs | contact, Fn, load_ratio, valid, age（**无 Ft**） |
| Critic | + ρ, slip_proxy, **ground_friction_μ** |
| Track | **full** `track_lin/ang`（去掉 slip 软地板） |
| 新奖 | `stable_speed_bonus`（跟得上且低滑才奖）+ `slip_under_command` |
| 指令 limit | vx ∈ [-0.5, **1.3**]（高速在分布内） |
| 部署目录 | `deploy/.../velocity/foot_v2/`（勿覆盖 `foot/`） |
| Schema | `obs_schema/foot_obs_v2.yaml`，actor **520** 维 |

## 训练

```bash
# smoke（~20 iter）
<repo>/research_scripts/finetune_g1_foot.sh --v2 --smoke

# 正式（partial 从 model_49999，不加载 optimizer）
<repo>/research_scripts/finetune_g1_foot.sh --v2 \
  --max-iterations 15000 --run-name foot_adaptive_v2 --num-envs 4096
```

日志：`logs/rsl_rl/unitree_g1_29dof_velocity_foot_adaptive_v2/`

## 评估

```bash
python scripts/rsl_rl/eval_friction_matrix.py \
  --task Unitree-G1-29dof-Velocity-Foot-Adaptive-V2 \
  --checkpoint logs/rsl_rl/.../model_XXXX.pt \
  --num_envs 64 --max_steps 250 --headless \
  --vx 0.4 0.8 1.1 --mu_bins 0.1 0.3 0.6 1.0
```

期望：高 μ 时 `mean_vxy` 更接近高 `vx`；低 μ 时 fall_rate / slip 更低、速度可明显低于指令。

## 导出 / MuJoCo

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Foot-Adaptive-V2 \
  --checkpoint <ckpt> --num_envs 16 --headless
# 安装到 foot_v2（不要覆盖 foot/policy.onnx）
cp -v logs/.../exported/policy.onnx \
  deploy/robots/g1_29dof/config/policy/velocity/foot_v2/exported/policy.onnx
# config.yaml: policy_dir: config/policy/velocity/foot_v2
```

MuJoCo：`G1_MUJOCO_FOOT_BRIDGE=1`，键 1/3 切换 μ，对比 `|v_xy|`。

## 与旧任务关系

- `Foot-Full` / `policy_full_10900.onnx` **保留** 作对照实验组。
- V2 的 `stable_speed_bonus` 是针对「μ 不变速」的主修复；旧 slip_aware 可作 ablation。
