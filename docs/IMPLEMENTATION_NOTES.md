# Cosmos-WAM MOT + iMF 实现详解

本文档详细记录了从旧版 Cosmos-WAM（Cross-Attention Action Head + 标准 Rectified Flow）迁移到新版 **MoT Joint Attention + iMF Loss** 的完整实现思路。

---

## 1. 整体架构变迁

### 1.1 旧架构
```
Video Latents ──► MiniTrainDIT (28层) ──► 第14-27层 hidden state
                                              │
                                              ▼
Noisy Action ──► ActionDiT (14层, Cross-Attn) ──► Predicted Velocity
```
- ActionDiT 只有 14 层，通过 **Cross-Attention** 读取 DiT 后半层的特征。
- Video 和 Action 的 self-attention 完全独立。
- Action loss 是标准 Rectified Flow：`MSE(v_pred, action - noise)`。

### 1.2 新架构 (MoT + iMF)
```
Video Latents ──► pre_dit ──► [video tokens]
                                    │
                                    ▼
                              MoT (28层 joint attention)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
        post_dit (video)                            post_dit (action)
              │                                           │
        Pred Velocity                              Pred Mean-Flow
```
- **Action Expert 改为 28 层**，与 DiT 同层数，参数独立。
- 每一层都把 video tokens 和 action tokens **concat 在一起做 mixed self-attention**。
- 用 **asymmetric attention mask** 控制双方的可见范围。
- Action loss 改为 **iMF**：通过 JVP 学习 mean velocity field `u_θ(z, r, t)`。

---

## 2. 逐文件实现详解

### 2.1 `cosmos_wam/models/dit_wrapper.py`

#### 改动点 A：Attention 支持 `attn_mask`
**原因**：MoT 需要在 mixed attention 里使用自定义 mask（video bidirectional，action 只能看条件帧等）。原版 `MiniTrainDIT` 的 `Attention` 完全没传 mask。

**实现**：
- `Attention.forward` 新增 `attn_mask: Optional[torch.Tensor] = None` 参数。
- `Attention.compute_attention` 将 mask 透传给 `self.attn_op(..., attn_mask=attn_mask)`。
- `MinimalA2AAttnOp` / `FlashAttnOp` 的 `forward` 从 `kwargs` 中读取 `attn_mask`。
- **fallback 机制**：如果 `flash_attn_func` 收到 mask，自动 fallback 到 `torch_attention_op`（因为 flash-attn 的 `flash_attn_func` 不支持任意 mask）。

#### 改动点 B：拆分 `pre_dit` / `post_dit`
**原因**：MoT 需要"手动"控制每一层的 self-attention（把 video 和 action 的 Q/K/V 拼起来），不能简单地调用 `Block.forward`（因为 `Block.forward` 内部已经封装了 self-attn + cross-attn + FFN 的完整流程）。

**实现**：
- **`pre_dit(x, timesteps, crossattn_emb)`**：
  1. `prepare_embedded_sequence`：patchify + RoPE。
  2. t-embedding：`t_embedder → t_embedding → t_embedding_norm`。
  3. cross-attn projection（如果需要）。
  4. 返回一个 dict：
     ```python
     {
         "tokens": x_flat,        # [B, T*H*W, D]
         "freqs": rope_emb,       # [S, 1, D]
         "t": t_emb,              # [B, T_tok, D]
         "t_mod": t_emb,          # 给 block 用的 modulation input
         "context": crossattn_emb,
         "meta": {"B": ..., "T": ..., "H": ..., "W": ..., "D": ...}
     }
     ```
- **`post_dit(tokens, pre_state)`**：
  1. 将 flat tokens reshape 回 5D `[B, T, H, W, D]`。
  2. `final_layer` + `unpatchify`。
- **`forward`** 重构为：
  ```python
  pre_state = self.pre_dit(...)
  for block in self.blocks:
      x = block(x, t_emb, crossattn_emb, rope_emb=...)
  output = self.post_dit(rearrange(x, ...), pre_state)
  ```

---

### 2.2 `cosmos_wam/models/action_head_mot.py` (新建)

这是全新的 action expert，用于替代旧版 `ActionDiT`。

#### 设计思路
- **层数**：必须和 video DiT 相同（28 层），因为 MoT 要求"逐层拼接"token。
- **Block 复用**：直接使用 `MiniTrainDIT` 的 `Block` 类，保证 attention backend、AdaLN、RoPE 等实现一致。
- **输入格式**：`Block` 要求 5D `[B, T, H, W, D]`，但 action 是 1D 序列。所以将 action reshape 为 `[B, S, 1, 1, D]`。

#### 关键组件
- **`action_encoder`**：`Linear(action_dim → hidden_dim → hidden_dim)` + SiLU。
- **`pos_embedding`**：`nn.Embedding(512, hidden_dim)`（learned positional embedding）。
  - **不用 RoPE**：video 用 RoPE，action 不用。这是通过 `Attention.compute_qkv` 中 `rope_emb is not None` 的判断自动实现的。
- **`time_embedding`**：
  - 输入是 `delta_t = t - r`。
  - `sinusoidal_embedding(delta_t, freq_dim=256) → MLP → RMSNorm`。
  - 输出 `[B, 1, hidden_dim]`，作为 `Block` 的 `emb_B_T_D`。
- **`text_embedding`**：将 context（Reason-1 7B 的 100352-dim embedding）投影到 `hidden_dim`。
- **`blocks`**：`nn.ModuleList([Block(...) for _ in range(28)])`。
- **`head`**：`nn.Linear(hidden_dim, action_dim)`，输出 predicted mean-flow。

#### `pre_dit` / `post_dit`
- 接口设计与 `MiniTrainDIT.pre_dit` 保持一致，方便 `MoT` 统一处理。
- `pre_dit` 返回 dict 中包含 `tokens_5d`（`[B, S, 1, 1, D]`），这是为了直接喂给 `Block`。

---

### 2.3 `cosmos_wam/models/mot.py` (新建)

这是整个改动的**核心文件**，实现了 joint attention 的 MoT 逻辑。

#### 初始化 (`__init__`)
- 接收 `mixtures = {"video": dit, "action": action_expert}`。
- 校验：两个 expert 必须有相同的 `num_layers`、`num_heads`、`head_dim`。

#### Mask 构建 (`_build_mot_attention_mask`)
**这是最关键的设计之一**。输出一个 `[S_total, S_total]` 的 bool mask（`True=attend`, `False=block`）。

```python
# video -> video: fully bidirectional
mask[:video_len, :video_len] = True

# action -> action: block-wise causal
for q_idx in range(action_seq_len):
    current_block = q_idx // actions_per_latent
    allowed_until = (current_block + 1) * actions_per_latent
    mask[video_len+q_idx, video_len:video_len+allowed_until] = True

# action -> video: only first-frame tokens
cond_tokens = video_tokens_per_frame * num_cond_frames
mask[video_len:, :cond_tokens] = True
```

#### 手动拆解 Block (`_build_expert_attention_io`)
**为什么需要手动拆解？**
因为 `Block.forward` 内部是"黑盒"：输入 5D tensor → 输出 5D tensor，中间已经完成了 self-attn + cross-attn + FFN。但 MoT 需要：
1. 从 video block 和 action block 分别提取 Q/K/V。
2. 把 Q/K/V concat 在一起做 mixed attention。
3. 把 mixed attention 的输出 split 开，再分别用各自的 block 做后半段（FFN、residual、cross-attn）。

**实现细节**（以 video expert 为例）：
```python
# 1. AdaLN for self-attn
shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(t_mod).chunk(3, dim=-1)
shift_sa = rearrange(shift_sa, "b t d -> b t 1 1 d")
...

# 2. Modulate + norm + flatten
norm_x = modulate(block.norm1(x), scale_sa, shift_sa)
norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")

# 3. Compute Q/K/V (with RoPE for video, without for action)
q, k, v = block.self_attn.compute_qkv(norm_x_flat, rope_emb=freqs)
# q,k,v: [B, S, H, head_dim]

# 4. 同时预计算 post-block 需要的其他 AdaLN 参数
shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(t_mod).chunk(3, dim=-1)
shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(t_mod).chunk(3, dim=-1)
```

#### Mixed Attention (`_mixed_attention`)
- 输入：`q_cat, k_cat, v_cat` 都是 `[B, S_total, H, head_dim]`。
- 将 mask `[S_total, S_total]` 扩展为 `[B, H, S_total, S_total]`。
- 调用 `F.scaled_dot_product_attention`（支持 bool mask 或 float additive mask）。
- 输出：`[B, S_total, H, head_dim]`。

#### Post-Block (`_apply_expert_post_block`)
对 split 后的 mixed attention 输出做剩余步骤：
1. `output_proj` + `output_dropout`：把 multi-head 结果 flatten 回 `[B, S, D]`。
2. Reshape 回 5D：`[B, T, H, W, D]`。
3. Residual：`x = x + gate_sa * attn_out`。
4. Cross-Attention to text context：`norm3 → cross_attn → residual`。
5. FFN：`norm2 → mlp['layer1'] → activation → mlp['layer2'] → residual`。

#### `forward` (训练用)
逐层循环 28 次：
1. 对每个 expert 调用 `_build_expert_attention_io`，收集 `q, k, v` 和 cache。
2. `torch.cat(q_chunks, dim=1)` 得到 `q_cat`。
3. `_mixed_attention(q_cat, k_cat, v_cat, mask)`。
4. 按 `seq_lens` split mixed output。
5. 对每个 expert 调用 `_apply_expert_post_block`。
6. **Gradient Checkpointing**：如果 `mot_checkpoint_mixed_attn=True` 且处于 training 模式，post-block 会用 `torch.utils.checkpoint.checkpoint` 包裹，节省显存。

#### `prefill_video_cache` (推理用)
- 只在 video branch 上做完整的 28 层 forward。
- 每层缓存该层的 `K` 和 `V`（`[B, S_video, H, head_dim]`）。
- 用于后续 action inference 时避免重复计算 video tokens。

#### `forward_action_with_video_cache` (推理用)
- 接收 action tokens 和 cached video K/V。
- 每层：
  1. 计算 action 的 `q_action, k_action, v_action`。
  2. `k_cat = [k_video_cache, k_action]`，`v_cat = [v_video_cache, v_action]`。
  3. `q_cat = q_action`（不需要拼 video 的 query，因为我们只取 action 的 attention 输出）。
  4. attention mask 使用 joint mask 的 action rows（`mask[video_seq_len:, :]`）。
  5. `_mixed_attention(q_action, k_cat, v_cat, action_mask)`。
  6. post-block。

---

### 2.4 `cosmos_wam/models/cosmos_wam.py` (重写)

这是整个训练和推理的** orchestration 层**。

#### 模型初始化
- 现在接收 `mot: MoT`。
- `self.dit` 和 `self.action_head` 改为 **property**，指向 `mot.mixtures["video"]` 和 `mot.mixtures["action"]`。
  - **为什么用 property？** 避免 `state_dict` 里同一个参数出现两次（`mot` 已经包含了两个 expert，如果再把它们作为 `CosmosWAM` 的直接属性，save checkpoint 时会存两份）。

#### `training_loss`

**Video 分支（标准 RF，不变）**：
```python
noise_v = torch.randn_like(latents)
t_video = torch.rand(B)
noisy_latents = (1 - t) * noise_v + t * latents
noisy_latents[:, :, :num_cond_frames] = latents[:, :, :num_cond_frames]  # keep clean

video_pre = self.dit.pre_dit(noisy_latents, t_video.unsqueeze(1), context)
```

**Action 分支（iMF）**：
```python
noise_a = torch.randn_like(action)
t_action = torch.rand(B)  # 独立采样，与 t_video 无关

# 50% r = t, 50% r ~ Uniform(0, t)
mask_r_eq_t = torch.rand(B) < 0.5
r = torch.where(mask_r_eq_t, t_action, torch.rand(B) * t_action)

noisy_action = (1 - t_action) * noise_a + t_action * action
target_v_cond = noise_a - action  # ε - a_1

action_pre = self.action_head.pre_dit(noisy_action, r, t_action, context)
```

**MoT Joint Forward**：
- 用 `self.mot._build_mot_attention_mask(...)` 构建 mask。
- 调用 `self.mot.forward(...)` 同时得到 `tokens_out["video"]` 和 `tokens_out["action"]`。
- Video 输出经 `post_dit` 得到 `pred_v`，计算 `loss_video`。

**`u_theta_fn` 的构造（iMF 核心）**：
```python
def u_theta_fn(z, rv, tv):
    ap = self.action_head.pre_dit(z, rv, tv, context)
    to = self.mot.forward(
        embeds_all={
            "video": video_tokens_const,   # .detach()
            "action": ap["tokens_5d"],
        },
        attention_mask=attention_mask,
        freqs_all={"video": video_freqs_const, "action": None},
        context_all={"video": video_context_const, "action": ap["context"]},
        t_mod_all={"video": video_t_mod_const, "action": ap["t_mod"]},
    )
    return self.action_head.post_dit(to["action"], ap)
```
- **video tokens 用 `.detach()`**：确保 JVP 的梯度不会回传到 video expert（我们只关心 action 参数的梯度）。

**JVP 计算**：
```python
v_pred = u_theta_fn(noisy_action, r, t_action)

if (r < t).any():
    u_pred, dudt = torch.autograd.functional.jvp(
        lambda z, rv, tv: u_theta_fn(z, rv, tv),
        (noisy_action, r, t_action),
        (v_pred, torch.zeros_like(r), torch.ones_like(t_action)),
    )
    V_theta = u_pred + (t - r).view(B, 1, 1) * dudt.detach()
else:
    V_theta = v_pred

loss_action = F.mse_loss(V_theta, target_v_cond)
```
- `dudt.detach()`：stop gradient on the JVP derivative，这是 iMF 的关键（防止高方差条件速度被 JVP 放大）。

**总 Loss**：
```python
loss = loss_video + lambda_action * loss_action
```

#### `infer_action` (单步生成)
1. 编码第一帧 → `first_frame_latent`。
2. `self.dit.pre_dit(first_frame_latent, t=0, context)`。
3. `self.mot.prefill_video_cache(...)` 缓存 28 层 K/V。
4. `z_1 = torch.randn(...)`。
5. `action_pre = self.action_head.pre_dit(z_1, r=0, t=1, context)`。
6. `action_tokens = self.mot.forward_action_with_video_cache(...)`。
7. `pred = self.action_head.post_dit(action_tokens, action_pre)`。
8. `a_0 = z_1 - pred`。

#### `infer_joint` (联合生成)
- **Stage 1**：单独 denoise video（标准 Rectified Flow，多步 Euler）。
  ```python
  for t from 1 -> 0:
      pred_v = self.dit(video_latents, t, context)
      video_latents = video_latents + dt * pred_v
      video_latents[:, :, :1] = first_frame_latent.clone()
  ```
- **Stage 2**：用最终 denoised video 做 prefill cache，然后 action **单步生成**：
  1. `video_pre = self.dit.pre_dit(denoised_video, t=0, context)`
  2. `video_kv_cache = self.mot.prefill_video_cache(...)`
  3. `z_1 = torch.randn(...)`
  4. `pred = u_θ(z_1, r=0, t=1)` via `forward_action_with_video_cache`
  5. `a_0 = z_1 - pred`

这与 `infer_action` 的区别仅在于 video cache 的来源：`infer_action` 只用第一帧，`infer_joint` 用完整生成的 video。

---

### 2.5 `cosmos_wam/runtime.py`

#### 改动：ActionExpert 的构建与初始化
1. **导入** `ActionExpert` 和 `MoT`。
2. **构建 `ActionExpert`**：
   - `num_layers = cfg.model.dit_config.num_blocks`（28），而不是从 `cfg.model.action_head.num_layers` 读取。
   - `text_dim = crossattn_proj_in_channels`（100352，对应 Reason-1 7B）。
3. **从 Video DiT 插值初始化 Action Expert**：
   ```python
   for key in action_state.keys():
       if key.startswith("blocks.") and key in dit_state:
           src = dit_state[key]
           target = action_state[key]
           if src.shape == target.shape:
               action_state[key] = src.clone()
           else:
               action_state[key] = _resize_tensor_to_shape(src, target.shape)
   action_head.load_state_dict(action_state, strict=False)
   ```
   - `_resize_tensor_to_shape` 会对每个维度用 `F.interpolate(..., mode="linear", align_corners=True)` 进行 resize。
4. **构建 MoT**：
   ```python
   mot = MoT(mixtures={"video": dit, "action": action_head})
   ```
5. **构建 CosmosWAM**：
   ```python
   model = CosmosWAM(mot=mot, vae=vae, ...)
   ```

---

### 2.6 `cosmos_wam/trainer.py`

#### 改动：optimizer 参数分组
- 旧版：`list(self.model.dit.parameters())`、`list(self.model.action_head.parameters())`
- 新版：`list(self.model.mot.mixtures["video"].parameters())`、`list(self.model.mot.mixtures["action"].parameters())`
- 其余训练逻辑（accelerator、scheduler、checkpoint save/load）完全不变。

---

### 2.7 配置文件 (`configs/*.yaml`)

所有配置文件中 `action_head.num_layers` 统一从 **14 改为 28**：
- `train_cosmos_2b_libero.yaml`
- `train_cosmos_2b_robotwin.yaml`
- `train_cosmos_2b.yaml`
- `sim_libero.yaml`
- `sim_robotwin.yaml`

---

### 2.8 评估脚本

#### `experiments/libero/eval_libero_single.py`
- 移除 `ActionDiT` 的导入。
- 改为导入 `ActionExpert` 和 `MoT`。
- 模型构建逻辑与 `runtime.py` 一致：
  1. Build `MiniTrainDIT`。
  2. Build `ActionExpert(num_layers=28)`。
  3. Build `MoT`。
  4. Build `CosmosWAM(mot=...)`。

#### `experiments/robotwin/cosmos_wam_policy/deploy_policy.py`
- 同样改为 `ActionExpert` + `MoT` + `CosmosWAM(mot=...)` 的构建链。

---

## 3. 关键设计决策与原因

### 3.1 为什么 Action Expert 的 self-attention 里不用 RoPE？
Video tokens 使用 3D RoPE（时-空-高-宽），而 action tokens 是 1D 序列，没有自然的 3D 空间结构。我们在 `MoT._build_expert_attention_io` 中对 video 传 `freqs=rope_emb`，对 action 传 `freqs=None`。`Attention.compute_qkv` 只在 `rope_emb is not None` 时应用 RoPE，因此 action 自动跳过 RoPE，改用 learned positional embedding。

### 3.2 为什么要手动拆解 `Block.forward`？
因为 `Block.forward` 是一个完整的"输入 5D → 输出 5D"的黑盒。MoT 需要在 self-attention 阶段"截胡"：把 video 和 action 的 Q/K/V 拼在一起做 joint attention，然后再 split 回去各自走 FFN。这要求我们必须**在外部复现 Block 内部的前半段逻辑**（AdaLN → norm → QKV projection → RoPE）。

### 3.3 为什么 `video_tokens_const` 要 `.detach()`？
在 `u_theta_fn` 中，video tokens 被当作常量。因为 JVP 会对 `u_theta_fn` 的输入 `(z, r, t)` 做梯度追踪，如果不 detach video tokens，它们也会被纳入计算图，导致 video expert 收到来自 action JVP 的额外梯度，这是不期望的。

### 3.4 为什么 `dudt` 要 `.detach()`？
这是 iMF 的核心改进。原始 Mean-Flow 的 JVP 会使用条件速度 `v_cond = ε - a` 作为切向量，但这会导致高方差。iMF 改用网络自身的预测 `v_pred` 作为切向量，并且对 `dudt` 做 `detach()`，防止 JVP 的高阶导数方差被反向传播放大。

### 3.5 为什么 `infer_action` 是单步？
在 Mean-Flow 框架下，velocity field `u_θ(z, r, t)` 满足：
```
a_0 = z_1 - u_θ(z_1, 0, 1)
```
因此可以直接从标准高斯噪声 `z_1` 单步 denoise 到干净动作 `a_0`，不需要像标准 Rectified Flow 那样走 20 步 Euler。

---

## 4. 数据流与梯度流

### 4.1 训练时的数据流
```
Dataset
  ├─► Video ──► VAE encode ──► noisy_latents ──► pre_dit ──► [video tokens]
  │                                                                │
  └─► Action ──► add noise ──► noisy_action ──► pre_dit ──► [action tokens]
                                                                  │
                                                              MoT.forward
                                                                  │
                                    ┌─────────────────────────────┴─────────────────────────────┐
                                    ▼                                                           ▼
                            post_dit (video)                                             post_dit (action)
                                    │                                                           │
                            loss_video = MSE(pred_v, target_v)                              u_theta_fn + JVP
                                                                                                  │
                                                                                            V_theta ──► loss_action
```

### 4.2 梯度流
- `loss_video` 的梯度 → `mot.mixtures["video"]`（28 层 video block 参数）。
- `loss_action` 的梯度 → `mot.mixtures["action"]`（28 层 action block 参数）。
- **没有共享参数**：两个 expert 的参数是独立的，但它们通过联合优化目标（同一个 `loss_total`）进行耦合。
- JVP 的梯度只流经 `u_theta_fn` 中的 action path，不会回传到 video path（因为 video tokens 被 detach）。

---

## 5. 已知的限制与风险

1. **旧 checkpoint 不兼容**：由于 `state_dict` 的 key 路径从 `dit.` / `action_head.` 变成了 `mot.mixtures.video.` / `mot.mixtures.action.`，旧的训练 checkpoint 无法直接 resume。
2. **JVP 显存开销**：约 50% 的样本会触发 JVP，相当于这部分 action forward 计算图翻倍。大 batch size 可能 OOM。
3. **Gradient Checkpointing 的兼容性**：原版 `MiniTrainDIT` 的 `enable_selective_checkpoint` 是通过 wrap `Block.forward` 实现的。在新架构中，`Block.forward` 不再被直接调用，因此 DiT 的 selective checkpointing **在 MoT forward 中不生效**。不过 `MoT` 自身实现了 post-block 的 checkpointing（通过 `mot_checkpoint_mixed_attn`），可以在一定程度上弥补。
4. **推理时 video prefill 的代价**：`infer_action` 每次都需要跑 28 层 video prefill 来缓存 K/V。如果用于高频重规划（如 replan every step），video prefill 会成为固定开销。
