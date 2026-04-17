import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


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


class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state):
        # state: [B, state_dim] -> [B, 1, hidden_dim]
        return self.mlp(state).unsqueeze(1)


class DepthEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_queries=16, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Patch embed
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=14, stride=14)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        # Perceiver Resampler layers
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=hidden_dim, num_heads=8, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

    def forward(self, depth):
        # depth: [B, 1, 224, 224] -> [B, 16, hidden_dim]
        B = depth.shape[0]

        x = self.patch_embed(depth).flatten(2).permute(0, 2, 1)  # [B, 256, D]
        x = x + self.pos_embed

        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](queries, x, x)
            queries = layer["norm1"](queries + attn_out)
            queries = layer["norm2"](queries + layer["ffn"](queries))

        return queries


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        original_dtype = t.dtype
        t = t.float()  # 强制 fp32 计算
        
        half = self.dim // 2
        device = t.device
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        return emb.to(original_dtype)  # 转回 bf16


class ActionEncoderIMF(nn.Module):
    def __init__(self, action_dim, hidden_dim, action_horizon=32):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, noisy_action, t):
        # noisy_action: [B, T, action_dim]
        # t: [B]
        B, T, _ = noisy_action.shape

        action_emb = self.action_proj(noisy_action) + self.pos_embed

        time_emb = self.time_embed(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, T, -1)

        combined = torch.cat([action_emb, time_emb], dim=-1)
        return self.combine(combined)  # [B, T, hidden_dim]


class ActionDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, cross_attention_dim=None, is_self_attn=False, dropout=0.0, final_dropout=False):
        super().__init__()
        self.is_self_attn = is_self_attn
        self.adaln = AdaLayerNorm(hidden_dim)

        if is_self_attn:
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        else:
            cross_dim = cross_attention_dim if cross_attention_dim is not None else hidden_dim
            self.attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                kdim=cross_dim,
                vdim=cross_dim,
                batch_first=True,
                dropout=dropout,
            )

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout) if final_dropout else nn.Identity(),
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


class ActionDiT(nn.Module):
    """Deprecated: old ActionDiT has been replaced by ActionHeadIMF."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ActionDiT has been removed. Please use ActionHeadIMF for the new architecture."
        )


class ActionHeadIMF(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        action_dim=7,
        state_dim=8,
        action_horizon=32,
        num_layers=16,
        num_heads=12,
        cross_attention_dim=768,
        video_ctx_dim=2048,
        dropout=0.1,
        final_dropout=True,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_layers = num_layers
        self.gradient_checkpointing = False
        self.num_timestep_buckets = num_timestep_buckets

        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.noise_s = noise_s

        # Encoders
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.depth_encoder = DepthEncoder(hidden_dim, num_queries=16, num_layers=2)
        self.action_encoder = ActionEncoderIMF(action_dim, hidden_dim, action_horizon=action_horizon)

        # Video context projection: 2048 -> 768
        self.video_ctx_proj = nn.Sequential(
            nn.Linear(video_ctx_dim, hidden_dim),
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
                hidden_dim,
                num_heads,
                cross_attention_dim=hidden_dim if i % 2 == 0 else None,
                is_self_attn=(i % 2 == 1),
                dropout=dropout,
                final_dropout=final_dropout,
            )
            for i in range(num_layers)
        ])

        # Output: align with GR00T DiT (ada-norm output + proj)
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.proj_out_2 = nn.Linear(hidden_dim, action_dim)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _forward_once(self, video_ctx, state, depth, noisy_action, t):
        device = next(self.parameters()).device
        video_ctx = video_ctx.to(device)
        state = state.to(device)
        depth = depth.to(device)
        noisy_action = noisy_action.to(device)
        t = t.to(device)

        state_tokens = self.state_encoder(state)          # [B, 1, D]
        depth_tokens = self.depth_encoder(depth)          # [B, 16, D]
        action_tokens = self.action_encoder(noisy_action, t)  # [B, T, D]

        x = torch.cat([state_tokens, depth_tokens, action_tokens], dim=1)  # [B, 49, D]

        video_ctx_proj = self.video_ctx_proj(video_ctx)   # [B, 424, D]

        time_emb = self.time_proj(self.time_embed(t))  # [B, D]

        for block in self.blocks:
            x = block(x, video_ctx_proj, time_emb)

        # GR00T-style adaptive output normalization
        shift, scale = self.proj_out_1(F.silu(time_emb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        pred = self.proj_out_2(x[:, -self.action_horizon:])  # [B, T, action_dim]
        return pred

    def forward(self, video_ctx, state, depth, actions):
        B = actions.shape[0]
        device = actions.device
        dtype = actions.dtype  # bf16

        t = self.sample_time(B, device=device, dtype=dtype)
        noise = torch.randn_like(actions)
        z_t = (1.0 - t[:, None, None]) * actions + t[:, None, None] * noise
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()
        pred = self._forward_once(video_ctx, state, depth, z_t, t_discretized)
        loss = F.mse_loss(pred, velocity)
        return loss

    @torch.no_grad()
    def predict_action(self, video_ctx, state, depth, num_steps=4):
        """Inference: multi-step Euler integration for flow matching."""
        B = state.shape[0]
        device = state.device
        target_dtype = next(self.parameters()).dtype

        video_ctx = video_ctx.to(dtype=target_dtype)
        state = state.to(dtype=target_dtype)
        depth = depth.to(dtype=target_dtype)

        actions = torch.randn(B, self.action_horizon, self.action_dim, device=device, dtype=target_dtype)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cont = i / num_steps
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((B,), t_discretized, device=device, dtype=torch.long)
            pred_velocity = self._forward_once(video_ctx, state, depth, actions, t_tensor)
            actions = actions + dt * pred_velocity.to(dtype=target_dtype)

        return actions
