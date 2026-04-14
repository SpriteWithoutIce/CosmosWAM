# Cosmos-WAM MOT + iMF 架构说明

本文档说明当前仓库中实现的 **Mixture-of-Tokens (MoT)** 联合注意力架构与 **Improved Mean-Flow (iMF)** 动作损失。

---

## 1. 架构概览

### 1.1 MoT（Mixture-of-Tokens）

与旧版 Cosmos-WAM（DiT 第 14–27 层通过 Cross-Attention 向独立的 14 层 ActionDiT 提供特征）不同，新版采用了 **MoT** 架构：

- **Video Expert**：28 层 `MiniTrainDIT`（加载 Cosmos-Predict2.5-2B 预训练权重）。
- **Action Expert**：28 层 `ActionExpert`（结构与 `MiniTrainDIT.Block` 相同，但参数独立）。
- **每一层都把 video tokens 和 action tokens 拼在一起做 joint self-attention**，然后用 asymmetric mask 控制各自的可见范围。

#### Attention Mask 规则

| 方向 | 可见范围 |
|------|----------|
| **Video → Video** | Fully bidirectional（所有 video tokens 互相可见） |
| **Video → Action** | Blocked（video 看不到 action） |
| **Action → Video** | 只能看到**条件帧**（第一帧 latent）的 tokens |
| **Action → Action** | Block-wise causal（当前 block 及之前的 action tokens 可见） |

这样设计的目的是：
- video 的生成逻辑不受影响（mask 没变）。
- action 能逐层读取对应层的 video 条件帧特征。
- action 和 video 通过**共享的梯度**间接耦合（参数独立，但梯度同时更新）。

### 1.2 Action Expert 的权重初始化

Action Expert 的 28 层 block 参数**不是随机初始化**，而是从 Video DiT 的预训练权重通过维度插值（`F.interpolate`）copy 过来的。具体逻辑在 `cosmos_wam/runtime.py` 中：

- 形状匹配的 tensor → 直接 copy。
- 形状不匹配的 tensor（如 hidden_dim 2048 → 1024）→ 逐维度 linear interpolate 后 copy。
- `action_encoder`、`time_embedding`、`pos_embedding`、`head` 仍随机初始化。

---

## 2. iMF（Improved Mean-Flow）损失

### 2.1 与旧版的区别

旧版 action 分支使用标准 Rectified Flow：模型预测 velocity `v_θ(z_t, t)`，loss = MSE(pred, `a - ε`)。

新版改用 **Improved Mean-Flow**，核心思想是模型直接预测一个"mean velocity field" `u_θ(z, r, t)`，并通过 JVP（Jacobian-Vector Product）构造更稳定的训练目标。

### 2.2 数学定义

**采样**（`t` 与 video 的 `t_video` **独立采样**）：
- `t ~ Uniform(0, 1)`
- `ε ~ N(0, I)`
- `z_t = (1 - t) · a + t · ε`
- `v_cond = ε - a`
- `r`：
  - 50% 概率 `r = t`（边界条件，退化为 IVC）
  - 50% 概率 `r ~ Uniform(0, t)`（完整 iMF）

**前向**：
```
v_pred = u_θ(z_t, r, t)
```

**JVP 计算**（仅对 `r < t` 的样本）：
```
u_pred, dudt = JVP(
    fn = u_θ(z, r, t),
    inputs = (z_t, r, t),
    tangents = (v_pred, 0, 1)
)
V_θ = u_pred + (t - r) · stop_grad(dudt)
```

**Loss**：
```
L_action = MSE(V_θ, v_cond)
```

当 `r = t` 时，`V_θ = u_θ(z_t, t, t)`，即普通的 velocity MSE。

### 2.3 时间 Embedding

Action Expert 接收双时间输入 `(r, t)`，但实际嵌入的是 **delta time**：
```python
emb = sinusoidal(t - r) + MLP
```

---

## 3. 推理方式

### 3.1 动作单步生成（`infer_action`）

与旧版的多步 Euler 去噪不同，iMF 支持**单步生成**：

```python
z_1 ~ N(0, I)
a_0 = z_1 - u_θ(z_1, r=0, t=1)
```

内部实现：
1. 用第一帧编码 video latent，prefill video K/V cache。
2. Action Expert 以 `z_1` 为输入、`r=0, t=1` 为时间，通过 MoT 的 cache 推理。
3. 输出 `a_0 = z_1 - pred`。

### 3.2 联合生成（`infer_joint`）

Video 仍按标准 RF 做多步 Euler 去噪。Action 每步用当前 `t` 作为 `r=t` 的边界条件：

```python
for t from 1 -> 0:
    pred_video, pred_action = mot_joint_forward(...)
    video = video - dt * pred_video
    action = action - dt * u_θ(action, r=t, t=t)
```

---

## 4. 配置参数说明

### 4.1 训练配置（`configs/train_cosmos_2b_*.yaml`）

```yaml
model:
  lambda_action: 1.0           # action loss 权重
  num_cond_frames: 1           # 条件帧数（固定为 1）
  mot_checkpoint_mixed_attn: true   # MoT 是否开启 gradient checkpointing

  dit_config:
    num_blocks: 28             # DiT 层数（必须与 action_head num_layers 一致）
    # ... 其他 DiT 参数不变

  action_head:
    action_dim: 7              # LIBERO 7-DOF / RoboTwin 16-DOF
    hidden_dim: 1024           # Action Expert hidden dim
    num_layers: 28             # 必须与 dit_config.num_blocks 相同
    num_heads: 16              # attention heads（必须与 DiT 相同）
    mlp_ratio: 4.0
    actions_per_latent: 8      # 每个 latent frame 对应的 action tokens 数
```

**注意**：
- `action_head.num_layers` 已统一改为 **28**（与 DiT 同层数）。
- `action_head.num_heads` 必须与 `dit_config.num_heads` 一致（当前为 16）。

### 4.2 评估配置（`configs/sim_*.yaml`）

评估配置中的 `model.action_head` 必须与训练时保持一致（特别是 `num_layers=28`）。

```yaml
EVALUATION:
  num_inference_steps: 4       # LIBERO 推荐 4-8；RoboTwin 推荐 20
  replan_steps: 5              # 每 N 步重新规划一次动作
  action_horizon: 32           # 预测的动作序列长度
```

---

## 5. 如何运行

### 5.1 训练

```bash
# LIBERO
python scripts/train.py --config-name train_cosmos_2b_libero

# RoboTwin
python scripts/train.py --config-name train_cosmos_2b_robotwin

# 多 GPU
torchrun --nproc_per_node=4 scripts/train.py --config-name train_cosmos_2b_libero
```

### 5.2 评估

**LIBERO 单任务**：
```bash
python experiments/libero/eval_libero_single.py \
    ckpt=/path/to/checkpoint.pt \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=0 \
    EVALUATION.num_trials=50 \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.text_embedding_cache_dir=./data/text_embeds_cache/libero
```

**RoboTwin 单任务**：
```bash
python experiments/robotwin/eval_robotwin_single.py \
    ckpt=/path/to/checkpoint.pt \
    EVALUATION.task_name=click_alarmclock \
    EVALUATION.dataset_stats_path=./dataset_stats.json \
    EVALUATION.text_embedding_cache_dir=./data/text_embeds_cache/robotwin
```

---

## 6. Checkpoint 兼容性

### 6.1 旧 checkpoint 无法直接 resume

由于参数路径发生了结构性变化：
- 旧版：`dit.blocks.*`、`action_head.blocks.*`
- 新版：`mot.mixtures.video.blocks.*`、`mot.mixtures.action.blocks.*`

**旧的训练 checkpoint 无法直接 `resume=true` 加载**。

### 6.2 迁移方案（Hot-start）

如果你有一个旧 checkpoint 的 video DiT 权重，可以：
1. 将旧 checkpoint 中的 `dit.*` 权重导出为新的 Cosmos DiT `.pt`。
2. 修改配置中的 `dit_checkpoint` 指向该文件。
3. 以 `resume=false` 启动训练，Action Expert 会自动从该 DiT 插值初始化。

或者，如果你有自定义脚本做 key remapping（把 `dit.` → `mot.mixtures.video.`、`action_head.` → `mot.mixtures.action.`），也可以手动转换旧 checkpoint。

---

## 7. 注意事项与故障排查

### 7.1 JVP 内存开销

`torch.autograd.functional.jvp` 会对 `r < t` 的样本（约 50%）重新 forward action branch，相当于这部分的 action 计算图翻倍。如果 batch size 较大导致 OOM，建议：
- 减小 `trainer.batch_size`
- 增大 `trainer.gradient_accumulation_steps`
- 暂时关闭 `mot_checkpoint_mixed_attn`（默认已开启，可节省内存）

### 7.2 Attention Mask 调试

如果训练时出现 attention mask shape mismatch 报错，检查：
- `video_size` 是否能被 `patch_spatial=2` 整除。
- `action_horizon` 是否能被 `actions_per_latent` 整除。
- `num_frames` 满足 `T % 4 == 1`（VAE 要求）。

### 7.3 单步 vs 多步推理

当前 `infer_action` 已经是**单步生成**。如果你在评估时发现动作质量不如预期，可以尝试：
- 检查训练时 `loss_action` 是否正常下降。
- 检查 `t - r` 的 embedding 是否稳定（因为 `t - r` 可能接近 0，sinusoidal embedding 在 0 附近变化较平缓）。

---

## 8. 关键文件索引

| 文件 | 说明 |
|------|------|
| `cosmos_wam/models/mot.py` | MoT 核心实现（joint attention、mask 构建、cache 推理） |
| `cosmos_wam/models/action_head_mot.py` | 28 层 ActionExpert |
| `cosmos_wam/models/cosmos_wam.py` | 主模型：iMF loss、训练/推理逻辑 |
| `cosmos_wam/models/dit_wrapper.py` | MiniTrainDIT（新增 `pre_dit`/`post_dit`、attn_mask 支持） |
| `cosmos_wam/runtime.py` | 训练入口（构建 MoT、Action Expert 插值初始化） |
| `cosmos_wam/trainer.py` | 训练循环（optimizer 参数分组） |
| `configs/train_cosmos_2b_libero.yaml` | LIBERO 训练配置 |
| `configs/train_cosmos_2b_robotwin.yaml` | RoboTwin 训练配置 |
| `experiments/libero/eval_libero_single.py` | LIBERO 评估入口 |
| `experiments/robotwin/cosmos_wam_policy/deploy_policy.py` | RoboTwin 评估入口 |
