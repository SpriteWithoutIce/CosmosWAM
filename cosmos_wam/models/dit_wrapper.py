"""Stripped-down MiniTrainDIT from cosmos minimal_v4_dit.py.
Removes transformer_engine, megatron, and NATTEN dependencies.
"""

from __future__ import annotations
import math
from typing import Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# RMSNorm (replaces te.pytorch.RMSNorm)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to q and k.
    q,k: [B, S, H, D] (bshd format)
    freqs: [S, 1, D] or [1, S, 1, D]
    """
    cos = torch.cos(freqs).to(q.dtype)
    sin = torch.sin(freqs).to(q.dtype)
    if cos.ndim == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention backend (replaces transformer_engine / NATTEN)
# ---------------------------------------------------------------------------
def torch_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """q,k,v: [B, S, H, D]"""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
    )
    out = out.transpose(1, 2)
    if flatten_heads:
        out = rearrange(out, "b s h d -> b s (h d)")
    return out


class MinimalA2AAttnOp(nn.Module):
    def forward(self, q, k, v, **kwargs):
        return torch_attention_op(q, k, v, attn_mask=None)

    def set_context_parallel_group(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Attention layer
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "minimal_a2a",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        self.backend = backend
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "torch":
            self.attn_op = torch_attention_op
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if not hasattr(self.attn_op, "set_context_parallel_group"):
            def _dummy(*args, **kwargs):
                pass
            self.attn_op.set_context_parallel_group = _dummy

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.query_dim)
        nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.context_dim)
        nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.head_dim * self.n_heads)
        nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)
        for layer in (self.q_norm, self.k_norm, self.v_norm):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            orig_dtype = q.dtype
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            q, k = apply_rotary_pos_emb(q, k, rope_emb, tensor_format=self.qkv_format, fused=True)
            if self.use_wan_fp32_strategy:
                q = q.to(orig_dtype)
                k = k.to(orig_dtype)
        return q, k, v

    def compute_attention(self, q, k, v, **kwargs):
        result = self.attn_op(q, k, v, **kwargs)
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)

    def set_context_parallel_group(self, *args, **kwargs):
        self.attn_op.set_context_parallel_group(*args, **kwargs)


# ---------------------------------------------------------------------------
# PatchEmbed & FinalLayer
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Cosmos-style PatchEmbed using Linear instead of Conv3d."""
    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        # Cosmos uses a single Linear layer (not Conv3d)
        # Input: flattened patch
        patch_dim = in_channels * temporal_patch_size * spatial_patch_size * spatial_patch_size
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, out_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        B, C, T, H, W = x.shape
        p_t = self.temporal_patch_size
        p_s = self.spatial_patch_size
        
        # Ensure input matches weight dtype for mixed precision
        weight_dtype = self.proj[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        
        # Patchify: [B, C, T, H, W] -> [B, T//p_t, H//p_s, W//p_s, C*p_t*p_s*p_s]
        x = x.view(B, C, T // p_t, p_t, H // p_s, p_s, W // p_s, p_s)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, T // p_t, H // p_s, W // p_s, C * p_t * p_s * p_s)
        
        # Apply linear projection
        x = self.proj(x)
        
        # Output: [B, T//p_t, H//p_s, W//p_s, out_channels]
        return x


class FinalLayer(nn.Module):
    """Cosmos-style FinalLayer matching checkpoint structure."""
    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
        )
        
        # Cosmos checkpoint: adaln_modulation.1 (Linear to 256), adaln_modulation.2 (Linear to 4096)
        # Sequential structure: SiLU -> Linear(hidden, 256) -> Linear(256, 4096)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
        )
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[2].weight)

    def forward(self, x, emb, adaln_lora_B_T_3D=None):
        if self.use_wan_fp32_strategy:
            assert emb.dtype == torch.float32
        shift, scale = self.adaln_modulation(emb).chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x

    @property
    def hidden_size(self):
        return self.linear.in_features


# ---------------------------------------------------------------------------
# Timestep / MLP
# ---------------------------------------------------------------------------
class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    """Cosmos-style MLP with nested structure matching checkpoint."""
    def __init__(self, in_dim: int, out_dim: int, activation: str = "silu", use_adaln_lora: bool = False):
        super().__init__()
        # Cosmos checkpoint has: t_embedder.1.linear_1.weight, t_embedder.1.linear_2.weight
        # This is a nested ModuleList structure
        self.layer1 = nn.ModuleList([
            nn.Sequential(),  # Placeholder for any pre-processing
        ])
        self.layer1[0].add_module('linear_1', nn.Linear(in_dim, out_dim, bias=False))
        
        self.activation = nn.SiLU() if activation == "silu" else nn.GELU(approximate="tanh")
        
        # Second linear layer (projects to 3x out_dim for adaLN in Cosmos)
        self.layer2 = nn.Linear(out_dim, 3 * out_dim if use_adaln_lora else out_dim, bias=False)
        self.use_adaln_lora = use_adaln_lora
        
        std = 1.0 / math.sqrt(in_dim)
        nn.init.trunc_normal_(self.layer1[0].linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(out_dim)
        nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor):
        # Ensure input matches weight dtype for mixed precision
        weight_dtype = self.layer1[0].linear_1.weight.dtype
        if sample.dtype != weight_dtype:
            sample = sample.to(dtype=weight_dtype)
        emb = self.layer1[0].linear_1(sample)
        emb = self.activation(emb)
        emb = self.layer2(emb)
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


# ---------------------------------------------------------------------------
# 3D RoPE position embedding
# ---------------------------------------------------------------------------
def get_rotary_pos_emb(head_dim: int, max_length: int, theta: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_length, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb  # [max_length, head_dim]


class VideoPositionEmb(nn.Module):
    def __init__(self, pos_emb_cls: str = "sincos"):
        super().__init__()
        self.pos_emb_cls = pos_emb_cls

    def generate_embeddings(
        self,
        x_B_T_H_W_D: torch.Tensor,
        head_dim: int,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
    ) -> torch.Tensor:
        B, T, H, W, D = x_B_T_H_W_D.shape
        theta_t = 10000.0 / rope_t_extrapolation_ratio
        theta_h = 10000.0 / rope_h_extrapolation_ratio
        theta_w = 10000.0 / rope_w_extrapolation_ratio

        rope_t = get_rotary_pos_emb(head_dim, T, theta_t)
        rope_h = get_rotary_pos_emb(head_dim, H, theta_h)
        rope_w = get_rotary_pos_emb(head_dim, W, theta_w)

        # Broadcast and sum: [T, 1, 1, D] + [1, H, 1, D] + [1, 1, W, D]
        rope_t = rope_t.view(T, 1, 1, head_dim)
        rope_h = rope_h.view(1, H, 1, head_dim)
        rope_w = rope_w.view(1, 1, W, head_dim)

        rope = rope_t + rope_h + rope_w  # [T, H, W, D]
        rope = rope.view(T * H * W, 1, head_dim)
        return rope.to(x_B_T_H_W_D.device, dtype=x_B_T_H_W_D.dtype)

    def forward(self, x_B_T_H_W_D, **kwargs):
        return None  # For non-rope types (not used in our setup)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
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
            context_dim=None,
            n_heads=num_heads,
            head_dim=head_dim,
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.norm3 = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            query_dim=x_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=head_dim,
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.norm2 = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        # Cosmos uses named layers: layer1, layer2
        self.mlp = nn.ModuleDict({
            'layer1': nn.Linear(x_dim, mlp_hidden_dim, bias=False),
            'layer2': nn.Linear(mlp_hidden_dim, x_dim, bias=False),
        })
        self.mlp_activation = nn.GELU(approximate="tanh")

        if self.use_adaln_lora:
            # AdaLN with LoRA - matches Cosmos checkpoint format
            # Checkpoint uses: Sequential(SiLU, Linear(x_dim, 256), Linear(256, 3*x_dim))
            # Named as: adaln_modulation_*.1.weight (first Linear) and adaln_modulation_*.2.weight (second Linear)
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
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
            nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            # Standard AdaLN
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False),
            )
            nn.init.zeros_(self.adaln_modulation_self_attn[-1].weight)
            nn.init.zeros_(self.adaln_modulation_cross_attn[-1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[-1].weight)

        self.num_heads = num_heads

    def init_weights(self):
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        std = 1.0 / math.sqrt(self.mlp['layer1'].in_features)
        nn.init.trunc_normal_(self.mlp['layer1'].weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.mlp['layer2'].in_features)
        nn.init.trunc_normal_(self.mlp['layer2'].weight, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H, W, D = x_B_T_H_W_D.shape

        if self.use_adaln_lora:
            # Sequential(SiLU, Linear(x_dim, 256), Linear(256, 3*x_dim))
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        shift_sa = rearrange(shift_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_sa = rearrange(scale_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_sa = rearrange(gate_sa, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_ca = rearrange(shift_ca, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_ca = rearrange(scale_ca, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_ca = rearrange(gate_ca, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        norm_x = modulate(self.norm1(x_B_T_H_W_D), scale_sa, shift_sa)
        norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")
        attn_out_flat = self.self_attn(norm_x_flat, rope_emb=rope_emb_L_1_1_D)
        attn_out = rearrange(attn_out_flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * attn_out

        norm_x = modulate(self.norm3(x_B_T_H_W_D), scale_ca, shift_ca)
        norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")
        cross_out_flat = self.cross_attn(norm_x_flat, context=crossattn_emb)
        cross_out = rearrange(cross_out_flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca * cross_out

        norm_x = modulate(self.norm2(x_B_T_H_W_D), scale_mlp, shift_mlp)
        mlp_out = self.mlp['layer2'](self.mlp_activation(self.mlp['layer1'](norm_x)))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * mlp_out
        return x_B_T_H_W_D


# ---------------------------------------------------------------------------
# Selective activation checkpointing
# ---------------------------------------------------------------------------
class SACConfig:
    class CheckpointMode:
        NONE = "none"
        BLOCK_WISE = "block_wise"

    def __init__(self, mode: str = "none", every_n_blocks: int = 1):
        self.mode = mode
        self.every_n_blocks = every_n_blocks

    def get_context_fn(self):
        def context_fn():
            return torch.enable_grad()
        return context_fn


def enable_selective_checkpoint(model: MiniTrainDIT, sac_config: SACConfig, blocks: nn.ModuleList):
    if sac_config.mode == SACConfig.CheckpointMode.NONE:
        return
    for block_id, block in blocks.named_children():
        if int(block_id) % sac_config.every_n_blocks == 0:
            from torch.utils.checkpoint import checkpoint
            # Wrap block forward
            orig_forward = block.forward
            def make_forward(orig):
                def _forward(x, emb, crossattn_emb, rope_emb=None):
                    return torch.utils.checkpoint.checkpoint(
                        orig, x, emb, crossattn_emb, rope_emb,
                        use_reentrant=False,
                    )
                return _forward
            block.forward = make_forward(orig_forward)


# ---------------------------------------------------------------------------
# MiniTrainDIT
# ---------------------------------------------------------------------------
class MiniTrainDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = False,  # Disable for checkpoint compatibility
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "minimal_a2a",
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        use_wan_fp32_strategy: bool = False,
        adaln_lora_dim: int = 256,
        use_t_embedding_adaln_lora: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.concat_padding_mask = concat_padding_mask

        self.patch_embedding = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels + (1 if concat_padding_mask else 0),
            out_channels=model_channels,
        )

        self.pos_embedder = VideoPositionEmb(pos_emb_cls=pos_emb_cls)
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio

        self.t_embedder = Timesteps(model_channels)
        self.t_embedding = MLP(model_channels, model_channels, activation="silu", use_adaln_lora=use_t_embedding_adaln_lora)
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                backend=atten_backend,
                use_wan_fp32_strategy=use_wan_fp32_strategy,
                adaln_lora_dim=adaln_lora_dim,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            use_adaln_lora=adaln_lora_dim > 0,
            adaln_lora_dim=adaln_lora_dim,
        )

        self.use_crossattn_projection = use_crossattn_projection
        if use_crossattn_projection:
            # Cosmos checkpoint uses Sequential: crossattn_proj.0 (Linear), crossattn_proj.1 (activation)
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
            )
        else:
            self.crossattn_proj = None

    def prepare_embedded_sequence(self, x_B_C_T_H_W: torch.Tensor, fps=None, padding_mask=None):
        if self.concat_padding_mask and padding_mask is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)

        # PatchEmbed now returns [B, T, H, W, D] directly
        x_B_T_H_W_D = self.patch_embedding(x_B_C_T_H_W)

        rope_emb = self.pos_embedder.generate_embeddings(
            x_B_T_H_W_D,
            head_dim=self.model_channels // self.num_heads,
            rope_h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            rope_w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            rope_t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )

        return x_B_T_H_W_D, rope_emb, None

    def unpatchify(self, x_B_T_H_W_O: torch.Tensor) -> torch.Tensor:
        B, T, H, W, O = x_B_T_H_W_O.shape
        C = self.out_channels
        p_t = self.patch_temporal
        p_s = self.patch_spatial
        x = x_B_T_H_W_O.view(B, T, H, W, C, p_t, p_s, p_s)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, T * p_t, H * p_s, W * p_s)
        return x

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        intermediate_feature_ids: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x_B_T_H_W_D, rope_emb_L_1_1_D, _ = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, _ = self.t_embedding(self.t_embedder(timesteps_B_T))
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        B, T, H, W, D = x_B_T_H_W_D.shape
        intermediate_features = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                feat = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
                intermediate_features.append(feat)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        if intermediate_feature_ids:
            return x_B_C_Tt_Hp_Wp, intermediate_features
        return x_B_C_Tt_Hp_Wp
