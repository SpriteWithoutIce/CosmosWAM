from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .dit_wrapper import Block, Timesteps, RMSNorm


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


class ActionExpert(nn.Module):
    """28-layer action expert using MiniTrainDIT Block, compatible with MoT."""

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

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            Block(
                x_dim=hidden_dim,
                context_dim=hidden_dim,
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
        delta_t = (t - r).clamp(0.0, 1.0)
        emb = _sinusoidal_embedding(delta_t, self.freq_dim)
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

        context_emb = self.text_embedding(context.to(dtype=dtype))

        # Reshape to 5D for Block compatibility: [B, S, 1, 1, hidden_dim]
        x_5d = x.unsqueeze(2).unsqueeze(3)

        return {
            "tokens_5d": x_5d,
            "tokens": x,  # flat for convenience
            "freqs": None,
            "t": t_emb,
            "t_mod": t_emb,
            "context": context_emb,
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
        ctx = pre["context"]
        for block in self.blocks:
            x = block(x, t_emb, ctx, rope_emb_L_1_1_D=None)
        return self.post_dit(x, pre)
