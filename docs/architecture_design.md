# VLA 架构设计文档 v3

## 1. 整体架构概览

本架构分为三个核心模块：**World Model（Cosmos DiT）**、**Latent Query 模块**、**Action Head（iMF）**。

```
┌─────────────────────────────────────────────────────────────────┐
│                           输入                                   │
│  RGB视频 [B,3,33,224,224] + state + depth + eef_pos              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│ 时间4×下采样   │   │ LatentQuery   │   │ DepthEncoder      │
│ + VAE Encode  │   │ Encoder       │   │ (Perceiver)       │
└───────┬───────┘   └───────┬───────┘   └─────────┬─────────┘
        │                   │                     │
        ↓                   ↓                     │
┌───────────────────────────────────────┐         │
│         Cosmos DiT (28层)              │         │
│  Layer 1-24: [cond, noisy]            │         │
│  Layer 25-28: [cond, noisy, query]    │         │
│                                       │         │
│  输出:                                 │         │
│    cond_hidden [B, 392, 2048]         │         │
│    noisy_hidden [B, 784, 2048]        │         │
│    query_hidden [B, 32, 2048]         │         │
└───────────────┬───────────────────────┘         │
                │                                 │
    ┌───────────┼───────────────┐                 │
    ↓           ↓               ↓                 │
┌────────┐ ┌─────────┐ ┌─────────────────┐        │
│Video   │ │Traj     │ │ video_ctx       │        │
│Loss    │ │Loss     │ │ [B,424,2048]    │        │
│        │ │[B,32,2] │ │ (cond+query)    │        │
└────────┘ └─────────┘ └────────┬────────┘        │
                                │                 │
                    ┌───────────┴─────────────────┘
                    ↓
        ┌───────────────────────────────┐
        │     Action Head (16层 DiT)     │
        │  输入: [state, depth, action]  │
        │        [B, 49, 768]           │
        │  Cross-attn: video_ctx        │
        │  输出: action [B, 32, act_dim] │
        │  Loss: iMF                    │
        └───────────────────────────────┘
```

---

## 2. World Model（Cosmos DiT）

### 2.1 定位

Cosmos DiT 作为视觉 backbone，通过视频生成任务学习 future-relevant 的条件帧表示。

### 2.2 输入处理

```
RGB 视频 [B, 3, 33, 224, 224]
    ↓ 手动时间 4× 下采样
[B, 3, 9, 224, 224]
    ↓ Cosmos VAE Encoder（时间 4×，空间 8×）
video_latent [B, 16, 3, 14, 28]
    ↓ 分离
condition_latent [B, 16, 1, 14, 28]   # 第 1 帧
noisy_latent [B, 16, 2, 14, 28]       # 后 2 帧（训练时加噪）
```

### 2.3 Patchify

```
condition_latent [B, 16, 1, 14, 28]
    ↓ flatten + linear projection
condition_tokens [B, 392, 2048]       # 1 × 14 × 28 = 392

noisy_latent [B, 16, 2, 14, 28]
    ↓ flatten + linear projection
noisy_tokens [B, 784, 2048]           # 2 × 14 × 28 = 784
```

### 2.4 DiT 结构

**Layer 1-24**：标准双向 DiT

```
输入: concat([condition_tokens, noisy_tokens], dim=1) = [B, 1176, 2048]

每层:
    x = x + SelfAttention(AdaLN(x, t_emb))
    x = x + CrossAttention(AdaLN(x, t_emb), text_emb)
    x = x + FFN(AdaLN(x, t_emb))
```

**Layer 25-28**：插入 Latent Query

```
输入: concat([condition_tokens, noisy_tokens, latent_query], dim=1) = [B, 1208, 2048]

Attention Mask:
┌────────────────────────────────────────────┐
│              cond(392)  noisy(784)  query(32) │
│ cond(392)       ✓           ✓          ✗      │
│ noisy(784)      ✓           ✓          ✗      │
│ query(32)       ✓           ✗          ✓      │
└────────────────────────────────────────────┘

- latent query 能 attend 到 condition tokens + 自己
- latent query 不能 attend 到 noisy tokens（保护视频生成）
- noisy tokens 不能 attend 到 latent query
```

### 2.5 输出

```
DiT 最后一层输出 [B, 1208, 2048]
    ↓ split
cond_hidden [B, 392, 2048]      # → video_ctx
noisy_hidden [B, 784, 2048]     # → 视频生成 loss
query_hidden [B, 32, 2048]      # → video_ctx + 轨迹监督 loss
```

### 2.6 梯度设计

| Loss            | 更新模块                                |
| --------------- | --------------------------------------- |
| 视频生成 loss   | Cosmos DiT 全部                         |
| 轨迹监督 loss   | Cosmos DiT 后 4 层 + LatentQueryEncoder |
| iMF action loss | Action Head（梯度不回传进 DiT）         |

---

## 3. Latent Query 模块

### 3.1 定位

32 个 latent query token，每个对应未来一个时间步，从条件帧中提取末端执行器运动轨迹信息。

### 3.2 LatentQueryEncoder

```
eef_pos_3d [B, 3]
    ↓ 相机外参投影
pixel_uv [B, 2]
    ↓ 生成 2D 高斯 heatmap
heatmap [B, 1, 224, 224]
    ↓ PatchEmbed (Conv2d, kernel=14, stride=14)
patch_features [B, 256, 2048]       # 16 × 16 = 256 patches
    ↓ + spatial_pos_embed
    ↓ Perceiver Resampler (cross-attention)
    ↓ 256 queries → 32 queries
latent_query_init [B, 32, 2048]
    ↓ + temporal_pos_embed
latent_query_init [B, 32, 2048]
```

**代码实现**：

```python
class LatentQueryEncoder(nn.Module):
    def __init__(self, hidden_dim=2048, num_queries=32, sigma=8.0):
        super().__init__()
        self.sigma = sigma
        self.num_queries = num_queries

        # Patch embed: [1, 224, 224] → [2048, 16, 16]
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=14, stride=14)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        # Perceiver Resampler: 256 → 32
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # 时序位置编码
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # 相机参数（固定）
        self.register_buffer('intrinsic', torch.zeros(3, 3))
        self.register_buffer('extrinsic', torch.zeros(4, 4))

    def forward(self, eef_pos_3d):
        B = eef_pos_3d.shape[0]

        # 投影到 2D
        pixel_uv = self.project_to_2d(eef_pos_3d)

        # 生成高斯 heatmap
        heatmap = self.generate_gaussian(pixel_uv)

        # Patch embed
        patch_features = self.patch_embed(heatmap)
        patch_features = patch_features.flatten(2).permute(0, 2, 1)
        patch_features = patch_features + self.patch_pos_embed

        # Perceiver: 256 → 32
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, patch_features, patch_features)
        latent_query_init = self.norm(queries + attn_out)

        # 时序位置编码
        latent_query_init = latent_query_init + self.temporal_pos_embed

        return latent_query_init
```

### 3.3 轨迹监督

每个 query 对应一个时间步，监督目标是未来 32 帧末端执行器的 2D 像素坐标。

**监督目标准备**：

```
eef_positions_3d [B, 32, 3]     # 未来 32 帧的 3D 位置
    ↓ 相机投影
target_trajectory [B, 32, 2]    # 对应的 2D pixel 坐标 (u, v)
```

**TrajectoryHead**：

```python
class TrajectoryHead(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, latent_query_hidden, target_trajectory):
        pred = self.head(latent_query_hidden)  # [B, 32, 2]
        loss = F.smooth_l1_loss(pred, target_trajectory)
        return loss
```

---

## 4. Action Head（iMF）

### 4.1 概述

基于 GR00T 风格的 DiT，使用 improved Mean Flow (iMF) loss，实现一步采样。

### 4.2 输入

| 输入         | Shape               | 来源                      |
| ------------ | ------------------- | ------------------------- |
| video_ctx    | [B, 424, 2048]      | Cosmos DiT (cond + query) |
| state        | [B, state_dim]      | 当前帧机器人状态          |
| depth        | [B, 1, 224, 224]    | 腕部深度图                |
| noisy_action | [B, 32, action_dim] | 加噪后的 action           |
| r, t         | [B], [B]            | iMF 时间步                |

### 4.3 StateEncoder

```python
class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state):
        # state: [B, state_dim] → [B, 1, hidden_dim]
        return self.mlp(state).unsqueeze(1)
```

### 4.4 DepthEncoder

```python
class DepthEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_queries=16):
        super().__init__()
        # Patch embed
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=14, stride=14)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        # Perceiver Resampler: 256 → 16
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, depth):
        # depth: [B, 1, 224, 224] → [B, 16, hidden_dim]
        B = depth.shape[0]

        x = self.patch_embed(depth).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed

        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.norm(queries + attn_out)
        out = self.norm2(queries + self.ffn(queries))

        return out
```

### 4.5 ActionEncoder（iMF 版本）

关键：只编码 `t - r`（跳跃距离），不分别编码 `r` 和 `t`。

```python
class ActionEncoderIMF(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, hidden_dim) * 0.02)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, noisy_action, r, t):
        # noisy_action: [B, 32, action_dim]
        # r, t: [B]
        B, T, _ = noisy_action.shape

        action_emb = self.action_proj(noisy_action) + self.pos_embed

        delta_t = t - r
        time_emb = self.time_embed(delta_t).unsqueeze(1).expand(-1, T, -1)

        combined = torch.cat([action_emb, time_emb], dim=-1)
        return self.combine(combined)  # [B, 32, hidden_dim]
```

### 4.6 DiT 结构

16 层，交替 self-attention（奇数层）和 cross-attention（偶数层）。

```python
class ActionDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, cross_attention_dim=None, is_self_attn=False):
        super().__init__()
        self.is_self_attn = is_self_attn
        self.adaln = AdaLayerNorm(hidden_dim)

        if is_self_attn:
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads,
                kdim=cross_attention_dim, vdim=cross_attention_dim,
                batch_first=True
            )

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, video_ctx, time_emb):
        x_norm = self.adaln(x, time_emb)

        if self.is_self_attn:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        else:
            attn_out, _ = self.attn(x_norm, video_ctx, video_ctx)

        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x
```

### 4.7 AdaLayerNorm

```python
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x, time_emb):
        emb = F.silu(time_emb)
        scale, shift = self.linear(emb).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

### 4.8 完整 Action Head

```python
class ActionHeadIMF(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        action_dim=7,
        state_dim=14,
        num_layers=16,
        num_heads=12,
        cross_attention_dim=768,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = 32

        # Encoders
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.depth_encoder = DepthEncoder(hidden_dim, num_queries=16)
        self.action_encoder = ActionEncoderIMF(action_dim, hidden_dim)

        # Video context projection: 2048 → 768
        self.video_ctx_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Timestep embedding
        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT Blocks
        self.blocks = nn.ModuleList([
            ActionDiTBlock(
                hidden_dim, num_heads,
                cross_attention_dim=hidden_dim if i % 2 == 0 else None,
                is_self_attn=(i % 2 == 1),
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _forward_once(self, video_ctx, state, depth, noisy_action, r, t):
        state_tokens = self.state_encoder(state)          # [B, 1, D]
        depth_tokens = self.depth_encoder(depth)          # [B, 16, D]
        action_tokens = self.action_encoder(noisy_action, r, t)  # [B, 32, D]

        x = torch.cat([state_tokens, depth_tokens, action_tokens], dim=1)  # [B, 49, D]

        video_ctx_proj = self.video_ctx_proj(video_ctx)   # [B, 424, D]

        time_emb = self.time_proj(self.time_embed(t - r)) # [B, D]

        for block in self.blocks:
            x = block(x, video_ctx_proj, time_emb)

        x = self.norm_out(x)
        pred = self.action_decoder(x[:, -32:])            # [B, 32, action_dim]
        return pred

    def forward(self, video_ctx, state, depth, actions):
        """训练: iMF loss"""
        B = actions.shape[0]

        # 采样时间
        r, t = self.sample_time(B)

        # 构造 noisy trajectory
        noise = torch.randn_like(actions)
        z_t = (1 - t[:, None, None]) * actions + t[:, None, None] * noise

        # 第一次 forward: r=t，得到 v_theta
        v_theta = self._forward_once(video_ctx, state, depth, z_t, t, t)

        # 第二次 forward: JVP
        primals = (z_t, r, t)
        tangents = (v_theta.detach(), torch.zeros_like(r), torch.ones_like(t))

        def fn(z, r_in, t_in):
            return self._forward_once(video_ctx, state, depth, z, r_in, t_in)

        u, dudt = torch.func.jvp(fn, primals, tangents)

        # 复合函数 V
        V = u + (t - r)[:, None, None] * dudt.detach()

        # Loss
        target = noise - actions
        loss = F.mse_loss(V, target)
        return loss

    @torch.no_grad()
    def predict_action(self, video_ctx, state, depth):
        """推理: 一步采样"""
        B = state.shape[0]
        z_1 = torch.randn(B, 32, self.action_dim, device=state.device)

        r = torch.zeros(B, device=state.device)
        t = torch.ones(B, device=state.device)

        u = self._forward_once(video_ctx, state, depth, z_1, r, t)
        return z_1 - u
```

---

## 5. 训练流程

```python
class VLAModel(nn.Module):
    def forward(self, batch):
        # 1. Latent Query 初始化
        latent_query_init = self.latent_query_encoder(batch['eef_pos'])

        # 2. Cosmos DiT 前向
        cond_hidden, noisy_hidden, query_hidden = self.cosmos_dit(
            batch['video'], latent_query_init
        )

        # 3. 视频生成 Loss
        loss_video = self.compute_video_loss(noisy_hidden, batch['video'])

        # 4. 轨迹监督 Loss
        loss_traj = self.trajectory_head(query_hidden, batch['eef_trajectory_2d'])

        # 5. Action Loss (iMF)
        video_ctx = torch.cat([cond_hidden, query_hidden], dim=1)  # [B, 424, 2048]
        loss_action = self.action_head(
            video_ctx.detach(),  # stop gradient
            batch['state'],
            batch['depth'],
            batch['action'],
        )

        # 6. 总 Loss
        loss = loss_video + λ_traj * loss_traj + λ_action * loss_action
        return loss
```

---

## 6. Shape 汇总

| 模块                       | 输入                | 输出                |
| -------------------------- | ------------------- | ------------------- |
| **VAE Encoder**            | [B, 3, 9, 224, 224] | [B, 16, 3, 14, 28]  |
| **LatentQueryEncoder**     | eef_pos [B, 3]      | [B, 32, 2048]       |
| **Cosmos DiT Layer 1-24**  | [B, 1176, 2048]     | [B, 1176, 2048]     |
| **Cosmos DiT Layer 25-28** | [B, 1208, 2048]     | [B, 1208, 2048]     |
| **TrajectoryHead**         | [B, 32, 2048]       | pred [B, 32, 2]     |
| **video_ctx**              | cond + query        | [B, 424, 2048]      |
| **StateEncoder**           | [B, state_dim]      | [B, 1, 768]         |
| **DepthEncoder**           | [B, 1, 224, 224]    | [B, 16, 768]        |
| **ActionEncoderIMF**       | [B, 32, action_dim] | [B, 32, 768]        |
| **VideoCtxProjection**     | [B, 424, 2048]      | [B, 424, 768]       |
| **ActionDiTBlock (×16)**   | [B, 49, 768]        | [B, 49, 768]        |
| **ActionDecoder**          | [B, 32, 768]        | [B, 32, action_dim] |

---

## 7. 参数量估算

| 模块                   | 参数量    |
| ---------------------- | --------- |
| Cosmos DiT (28 层)     | ~2B       |
| LatentQueryEncoder     | ~10M      |
| TrajectoryHead         | ~2M       |
| StateEncoder           | ~1M       |
| DepthEncoder           | ~25M      |
| ActionEncoderIMF       | ~2M       |
| VideoCtxProjection     | ~1.5M     |
| ActionDiTBlock (16 层) | ~150M     |
| ActionDecoder          | ~0.5M     |
| **Action Head 总计**   | **~180M** |

---

## 8. 推理流程

```python
@torch.no_grad()
def inference(self, rgb_frame, wrist_depth, state, eef_pos):
    # 1. Latent Query 初始化（在线算，< 1ms）
    latent_query_init = self.latent_query_encoder(eef_pos)

    # 2. Cosmos DiT（只需要 condition 帧）
    cond_hidden, query_hidden = self.cosmos_dit.inference(
        rgb_frame, latent_query_init
    )
    video_ctx = torch.cat([cond_hidden, query_hidden], dim=1)

    # 3. Depth Encoder（可与 Cosmos DiT 并行）
    # 见并行方案

    # 4. Action Head（一步采样）
    action = self.action_head.predict_action(video_ctx, state, wrist_depth)

    return action  # [B, 32, action_dim]
```

---

## 9. 关键设计决策总结

| 设计点              | 决策                   | 理由                            |
| ------------------- | ---------------------- | ------------------------------- |
| Latent Query 数量   | 32                     | 对应 32 帧 action，时序一一对应 |
| Latent Query 初始化 | 2D 高斯 + Perceiver    | 利用当前帧末端执行器位置        |
| Latent Query 监督   | 2D 轨迹坐标 [B, 32, 2] | 直接、可解释、与 action 对齐    |
| Latent Query 插入层 | 后 4 层 (25-28)        | 平衡信息获取与视频生成保护      |
| Action Head Loss    | iMF                    | 一步采样，训练稳定              |
| Timestep 编码       | 只编码 t - r           | 论文推荐，简洁有效              |
| Depth 位置          | 输入序列（self-attn）  | 与 action 双向交互              |
| Latent Query 位置   | Cross-attn 条件        | 单向信息提供                    |
