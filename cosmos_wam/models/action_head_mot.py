from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .dit_wrapper import Attention, Timesteps, RMSNorm


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.float()
    half = dim // 2
    device = t.device
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ZeroCrossAttnModulation(nn.Module):
    """Dummy module that returns zeros for cross-attn modulation (when no cross-attn exists)."""
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, x):
        # Return zeros with shape [B, T, 3*D] to match expected chunk(3) output
        B, T, _ = x.shape
        return torch.zeros(B, T, 3 * self.output_dim, device=x.device, dtype=x.dtype)


class SelfAttentionBlock(nn.Module):
    """Simplified block with only Self-Attention + MLP, no Cross-Attention."""

    def __init__(
        self,
        x_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        backend: str = "minimal_a2a",
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.use_adaln_lora = adaln_lora_dim > 0
        head_dim = x_dim // num_heads
        mlp_hidden_dim = int(x_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            query_dim=x_dim,
            context_dim=None,  # Self-attention only
            n_heads=num_heads,
            head_dim=head_dim,
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.norm2 = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        # MLP layers
        self.mlp = nn.ModuleDict({
            'layer1': nn.Linear(x_dim, mlp_hidden_dim, bias=False),
            'layer2': nn.Linear(mlp_hidden_dim, x_dim, bias=False),
        })
        self.mlp_activation = nn.GELU(approximate="tanh")

        if self.use_adaln_lora:
            # Only need adaLN for self-attn and mlp (no cross-attn)
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False),
            )
            nn.init.zeros_(self.adaln_modulation_self_attn[-1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[-1].weight)

        # Dummy cross-attn modulation for MoT compatibility (returns zeros)
        self.adaln_modulation_cross_attn = ZeroCrossAttnModulation(x_dim)

        self.num_heads = num_heads

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H, W, D = x_B_T_H_W_D.shape

        if self.use_adaln_lora:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        shift_sa = rearrange(shift_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_sa = rearrange(scale_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_sa = rearrange(gate_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        # Self-Attention
        norm_x = modulate(self.norm1(x_B_T_H_W_D), scale_sa, shift_sa)
        norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")
        attn_out_flat = self.self_attn(norm_x_flat, rope_emb=rope_emb_L_1_1_D)
        attn_out = rearrange(attn_out_flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * attn_out

        # MLP
        norm_x = modulate(self.norm2(x_B_T_H_W_D), scale_mlp, shift_mlp)
        mlp_out = self.mlp['layer2'](self.mlp_activation(self.mlp['layer1'](norm_x)))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * mlp_out

        return x_B_T_H_W_D


class ActionExpert(nn.Module):
    """28-layer action expert using only Self-Attention, compatible with MoT."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 28,
        num_heads: int = 16,
        text_dim: int = 100352,
        mlp_ratio: float = 4.0,
        backend: str = "minimal_a2a",
        use_wan_fp32_strategy: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_embedding = nn.Embedding(512, hidden_dim)

        # Timestep embedding for delta_t = t - r
        self.freq_dim = 256
        self.timesteps = Timesteps(self.freq_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_norm = RMSNorm(hidden_dim, eps=1e-6)

        # No text_embedding projection needed - we don't use cross-attention
        # Text info should be injected via MoT joint attention instead

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                x_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                backend=backend,
                use_wan_fp32_strategy=use_wan_fp32_strategy,
                adaln_lora_dim=adaln_lora_dim,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def _build_time_embedding(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """r, t: [B]. Returns [B, 1, hidden_dim]."""
        # Get model dtype from time_mlp weights
        model_dtype = next(self.time_mlp.parameters()).dtype
        delta_t = (t - r).clamp(0.0, 1.0).to(dtype=model_dtype)
        emb = _sinusoidal_embedding(delta_t, self.freq_dim).to(dtype=model_dtype)
        emb = self.time_mlp(emb)
        emb = self.time_norm(emb)
        return emb.unsqueeze(1)  # [B, 1, hidden_dim]

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, object]:
        if action_tokens.ndim != 3:
            raise ValueError(f"action_tokens must be [B, S, D], got {tuple(action_tokens.shape)}")
        if r.ndim != 1 or t.ndim != 1:
            raise ValueError(f"r and t must be 1D, got {r.shape} and {t.shape}")

        bsz, seq_len, _ = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype

        x = self.action_encoder(action_tokens)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        x = x + self.pos_embedding(pos_ids).unsqueeze(0)

        t_emb = self._build_time_embedding(r, t)  # [B, 1, hidden_dim]

        # Reshape to 5D for Block compatibility: [B, S, 1, 1, hidden_dim]
        x_5d = x.unsqueeze(2).unsqueeze(3)

        return {
            "tokens_5d": x_5d,
            "tokens": x,  # flat for convenience
            "freqs": None,
            "t": t_emb,
            "t_mod": t_emb,
            "context": None,  # No cross-attention context needed
            "meta": {
                "batch_size": bsz,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, object]) -> torch.Tensor:
        """tokens can be 5D [B,S,1,1,D] or flat [B,S,D]."""
        if tokens.ndim == 5:
            tokens = tokens.squeeze(2).squeeze(2)  # [B, S, D]
        x = self.out_norm(tokens)
        return self.action_out(x)

    def forward(self, action_tokens: torch.Tensor, r: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        pre = self.pre_dit(action_tokens, r, t, context)
        x = pre["tokens_5d"]
        t_emb = pre["t_mod"]
        # No context passing to blocks (no cross-attention)
        for block in self.blocks:
            x = block(x, t_emb, rope_emb_L_1_1_D=None)
        return self.post_dit(x, pre)
