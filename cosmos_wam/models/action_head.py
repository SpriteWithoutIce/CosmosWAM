from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim > 1:
        t = t.view(t.shape[0], -1)[:, 0]
    t = t.float()
    half = dim // 2
    device = t.device
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _sinusoidal_position_embedding(pos: torch.Tensor, dim: int) -> torch.Tensor:
    pos = pos.float()
    half = dim // 2
    device = pos.device
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
    args = pos[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _build_block_causal_self_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    block_size = max(int(block_size), 1)
    for q_idx in range(seq_len):
        current_block = q_idx // block_size
        allowed_until = min(seq_len, (current_block + 1) * block_size)
        mask[q_idx, :allowed_until] = 0.0
    return mask


class ActionBlock(nn.Module):
    """Self-Attn + Cross-Attn(video) + FFN with AdaLN."""

    def __init__(self, hidden_dim: int, num_heads: int, video_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            kdim=video_dim,
            vdim=video_dim,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio), bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim, bias=False),
        )

        # AdaLN modulation: 9 params (sa_shift, sa_scale, sa_gate,
        #                              ca_shift, ca_scale, ca_gate,
        #                              mlp_shift, mlp_scale, mlp_gate)
        self.modulation = nn.Linear(hidden_dim, 9 * hidden_dim, bias=False)
        nn.init.zeros_(self.modulation.weight)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        video_ctx: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure all inputs match the model dtype (for mixed precision)
        target_dtype = self.modulation.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        if t_emb.dtype != target_dtype:
            t_emb = t_emb.to(dtype=target_dtype)
        if video_ctx.dtype != target_dtype:
            video_ctx = video_ctx.to(dtype=target_dtype)
        
        # t_emb: [B, D] -> modulation: [B, 9*D]
        mods = self.modulation(t_emb).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods
        # Ensure all AdaLN params match input dtype
        target_dtype = x.dtype
        shift_sa = shift_sa.unsqueeze(1).to(dtype=target_dtype)
        scale_sa = scale_sa.unsqueeze(1).to(dtype=target_dtype)
        gate_sa = gate_sa.unsqueeze(1).to(dtype=target_dtype)
        shift_ca = shift_ca.unsqueeze(1).to(dtype=target_dtype)
        scale_ca = scale_ca.unsqueeze(1).to(dtype=target_dtype)
        gate_ca = gate_ca.unsqueeze(1).to(dtype=target_dtype)
        shift_mlp = shift_mlp.unsqueeze(1).to(dtype=target_dtype)
        scale_mlp = scale_mlp.unsqueeze(1).to(dtype=target_dtype)
        gate_mlp = gate_mlp.unsqueeze(1).to(dtype=target_dtype)

        # Self-attention with AdaLN
        norm_x = self.norm1(x) * (1 + scale_sa) + shift_sa
        # Ensure norm_x matches attention weight dtype
        attn_dtype = self.self_attn.in_proj_weight.dtype
        norm_x_attn = norm_x.to(dtype=attn_dtype)
        attn_out, _ = self.self_attn(norm_x_attn, norm_x_attn, norm_x_attn, attn_mask=self_mask)
        x = x + gate_sa * attn_out.to(dtype=x.dtype)

        # Cross-attention to video context with AdaLN
        norm_x = self.norm2(x) * (1 + scale_ca) + shift_ca
        cross_dtype = self.cross_attn.in_proj_weight.dtype
        norm_x_cross = norm_x.to(dtype=cross_dtype)
        video_ctx_cross = video_ctx.to(dtype=cross_dtype)
        cross_out, _ = self.cross_attn(norm_x_cross, video_ctx_cross, video_ctx_cross)
        x = x + gate_ca * cross_out.to(dtype=x.dtype)

        # FFN with AdaLN
        norm_x = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        mlp_dtype = self.mlp[0].weight.dtype
        mlp_out = self.mlp(norm_x.to(dtype=mlp_dtype))
        x = x + gate_mlp * mlp_out.to(dtype=x.dtype)
        return x


class ActionDiT(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 14,
        num_heads: int = 16,
        video_dim: int = 2048,
        mlp_ratio: float = 4.0,
        actions_per_latent: int = 8,
        timestep_buckets: int = 1000,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.video_dim = int(video_dim)
        self.actions_per_latent = int(actions_per_latent)
        self.timestep_buckets = int(timestep_buckets)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Timestep embedding: sinusoidal + MLP
        self.timestep_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Video context projection + positional encoding
        self.video_in = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_embedding = nn.Embedding(512, hidden_dim)  # max 512 actions

        self.blocks = nn.ModuleList([
            ActionBlock(hidden_dim, num_heads, hidden_dim, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.action_out = nn.Linear(hidden_dim, action_dim)

        self._cached_masks: dict = {}

    @property
    def requires_video_hidden_layers(self) -> bool:
        return True

    def _build_token_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        t = timestep.to(device=device)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.clamp(0.0, 1.0)
        temb = _sinusoidal_timestep_embedding(t * max(float(self.timestep_buckets - 1), 1.0), self.hidden_dim)
        # Ensure temb matches timestep_proj weight dtype for mixed precision
        temb = temb.to(dtype=self.timestep_proj[0].weight.dtype)
        temb = self.timestep_proj(temb).unsqueeze(1)  # [B, 1, D]
        return temb

    def _build_video_positional_encoding(
        self,
        batch_size: int,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        d_model = self.hidden_dim
        frame_idx = torch.arange(num_frames, device=device, dtype=torch.float32)
        frame_emb = _sinusoidal_position_embedding(frame_idx, d_model).unsqueeze(1)  # [F, 1, D]

        if spatial_hw is not None:
            h, w = spatial_hw
            y_idx = torch.arange(h, device=device, dtype=torch.float32)
            x_idx = torch.arange(w, device=device, dtype=torch.float32)
            y_emb = _sinusoidal_position_embedding(y_idx, d_model).unsqueeze(1).expand(h, w, d_model)
            x_emb = _sinusoidal_position_embedding(x_idx, d_model).unsqueeze(0).expand(h, w, d_model)
            spatial_emb = (y_emb + x_emb).reshape(1, h * w, d_model)
        else:
            token_idx = torch.arange(tokens_per_frame, device=device, dtype=torch.float32)
            spatial_emb = _sinusoidal_position_embedding(token_idx, d_model).unsqueeze(0)

        pos = frame_emb + spatial_emb  # [F, Tpf, D]
        pos = pos.reshape(1, num_frames * tokens_per_frame, d_model)
        return pos.expand(batch_size, -1, -1).to(dtype=dtype)

    def _get_self_mask(self, num_actions: int, device: torch.device) -> torch.Tensor:
        key = (num_actions, self.actions_per_latent, str(device))
        if key not in self._cached_masks:
            mask = _build_block_causal_self_mask(
                seq_len=num_actions,
                block_size=self.actions_per_latent,
                device=device,
            )
            self._cached_masks[key] = mask
        return self._cached_masks[key]

    def _encode_video_context(
        self,
        video_tokens: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """video_tokens: [B, K, H, W, D] or [B, K*H*W, D]"""
        if video_tokens.ndim == 5:
            bsz, num_frames, h, w, hidden_dim = video_tokens.shape
            tokens = video_tokens.view(bsz, num_frames * h * w, hidden_dim)
            spatial_hw = (h, w)
            tokens_per_frame = h * w
        elif video_tokens.ndim == 3:
            bsz, num_tokens, hidden_dim = video_tokens.shape
            tokens = video_tokens
            spatial_hw = None
            tokens_per_frame = 1
            num_frames = num_tokens
        else:
            raise ValueError(f"Unexpected video_tokens shape: {video_tokens.shape}")

        # Ensure tokens match video_in weight dtype
        target_dtype = self.video_in[0].weight.dtype
        video_ctx = self.video_in(tokens.to(dtype=target_dtype))
        video_ctx = video_ctx + self._build_video_positional_encoding(
            batch_size=bsz,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
            dtype=dtype,
            spatial_hw=spatial_hw,
        )
        return video_ctx

    def forward(
        self,
        z_action: torch.Tensor,
        video_tokens: List[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if z_action.ndim != 3:
            raise ValueError(f"Expected z_action [B, L, D], got {tuple(z_action.shape)}")
        if len(video_tokens) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} video layers, got {len(video_tokens)}")

        bsz, seq_len, _ = z_action.shape
        device = z_action.device
        # Get target dtype from model weights
        target_dtype = self.action_encoder[0].weight.dtype

        # Ensure timestep is float32 for sinusoidal embedding
        temb = self._build_token_timestep(timestep, batch_size=bsz, seq_len=seq_len, device=device)

        # Encode action
        x = self.action_encoder(z_action.to(dtype=target_dtype))
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0) + temb

        self_mask = self._get_self_mask(num_actions=seq_len, device=device)

        for block, layer_video_tokens in zip(self.blocks, video_tokens):
            video_ctx = self._encode_video_context(layer_video_tokens, dtype=target_dtype)
            x = block(x, temb.squeeze(1).to(dtype=target_dtype), video_ctx, self_mask=self_mask)

        x = self.out_norm(x)
        return self.action_out(x)
