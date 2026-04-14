from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit_wrapper import Block, modulate


class MoT(nn.Module):
    """Mixture-of-Tokens for joint video-action attention."""

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures or "video" not in mixtures or "action" not in mixtures:
            raise ValueError("mixtures must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.head_dim = int(first_expert.model_channels // self.num_heads)

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}")
            if int(expert.num_heads) != self.num_heads:
                raise ValueError(f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}")
            # head_dim derived from model_channels // num_heads; assume compatibility

    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        num_cond_frames: int = 1,
        actions_per_latent: int = 8,
    ) -> torch.Tensor:
        """Build asymmetric [S_total, S_total] bool mask.
        True = can attend, False = blocked.
        """
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video: fully bidirectional
        mask[:video_seq_len, :video_seq_len] = True

        # action -> action: block-wise causal
        action_start = video_seq_len
        for q_idx in range(action_seq_len):
            current_block = q_idx // actions_per_latent
            allowed_until = min(action_seq_len, (current_block + 1) * actions_per_latent)
            mask[action_start + q_idx, action_start:action_start + allowed_until] = True

        # action -> video: only conditional (first) frame tokens
        cond_tokens = min(video_tokens_per_frame * num_cond_frames, video_seq_len)
        mask[action_start:, :cond_tokens] = True

        return mask

    def _build_expert_attention_io(
        self,
        block: Block,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor],
        t_mod: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replicate Block's pre-self-attention logic.
        x: [B, T, H, W, D] (5D)
        freqs: [S, 1, D] or None
        t_mod: [B, T_tok, D] (typically [B, 1, D] broadcasted)
        Returns q, k, v [B, S, H, head_dim], residual_x [B, T, H, W, D], and modulation params.
        """
        B, T, H, W, D = x.shape
        use_adaln = block.use_adaln_lora

        # AdaLN for self-attn
        if use_adaln:
            shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(t_mod).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(t_mod).chunk(3, dim=-1)

        shift_sa = rearrange(shift_sa, "b t d -> b t 1 1 d").type_as(x)
        scale_sa = rearrange(scale_sa, "b t d -> b t 1 1 d").type_as(x)
        gate_sa = rearrange(gate_sa, "b t d -> b t 1 1 d").type_as(x)

        norm_x = modulate(block.norm1(x), scale_sa, shift_sa)
        norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")
        q, k, v = block.self_attn.compute_qkv(norm_x_flat, rope_emb=freqs)
        # q, k, v: [B, S, H, head_dim]

        # AdaLN for cross-attn and mlp (needed for post-block)
        if use_adaln:
            shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(t_mod).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(t_mod).chunk(3, dim=-1)
        else:
            shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(t_mod).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(t_mod).chunk(3, dim=-1)

        shift_ca = rearrange(shift_ca, "b t d -> b t 1 1 d").type_as(x)
        scale_ca = rearrange(scale_ca, "b t d -> b t 1 1 d").type_as(x)
        gate_ca = rearrange(gate_ca, "b t d -> b t 1 1 d").type_as(x)
        shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d").type_as(x)
        scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d").type_as(x)
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d").type_as(x)

        return (
            q, k, v,
            x,  # residual
            gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_mlp, scale_mlp, gate_mlp,
        )

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """q_cat, k_cat, v_cat: [B, S_total, H, head_dim]
        attention_mask: [S_total, S_total] bool -> expanded to [B, H, S_total, S_total]
        Returns: [B, S_total, H, head_dim]
        """
        B = q_cat.shape[0]
        H = self.num_heads
        # sdpa expects [B, H, S, D]
        q = q_cat.transpose(1, 2)
        k = k_cat.transpose(1, 2)
        v = v_cat.transpose(1, 2)

        # Expand mask
        if attention_mask.dtype == torch.bool:
            # bool mask -> use directly
            attn_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        else:
            # float/additive mask
            attn_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2)  # [B, S_total, H, head_dim]

    def _apply_expert_post_block(
        self,
        block: Block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_sa: torch.Tensor,
        shift_ca: torch.Tensor,
        scale_ca: torch.Tensor,
        gate_ca: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply output projection, residual, cross-attn, and MLP.
        mixed_attn_out: [B, S, H, head_dim] for this expert
        residual_x: [B, T, H, W, D]
        Returns updated x: [B, T, H, W, D]
        """
        B, T, H, W, D = residual_x.shape
        S = T * H * W

        # output projection
        mixed_flat = rearrange(mixed_attn_out, "b s h d -> b s (h d)")
        attn_out = block.self_attn.output_dropout(block.self_attn.output_proj(mixed_flat))
        attn_out = rearrange(attn_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x = residual_x + gate_sa * attn_out

        # cross-attn (only if block has cross_attn attribute, e.g., full Block)
        if hasattr(block, 'cross_attn'):
            norm_x = modulate(block.norm3(x), scale_ca, shift_ca)
            norm_x_flat = rearrange(norm_x, "b t h w d -> b (t h w) d")
            cross_out = block.cross_attn(norm_x_flat, context=context)
            cross_out = rearrange(cross_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            x = x + gate_ca * cross_out

        # mlp
        norm_x = modulate(block.norm2(x), scale_mlp, shift_mlp)
        mlp_out = block.mlp['layer2'](block.mlp_activation(block.mlp['layer1'](norm_x)))
        x = x + gate_mlp * mlp_out

        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, Optional[torch.Tensor]],
        context_all: Dict[str, Optional[torch.Tensor]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Joint forward pass through all layers."""
        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks: List[torch.Tensor] = []
            k_chunks: List[torch.Tensor] = []
            v_chunks: List[torch.Tensor] = []
            cached: List[dict] = []
            seq_lens: List[int] = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all.get(name)
                t_mod = t_mod_all[name]

                q, k, v, residual_x, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self._build_expert_attention_io(
                    block, x, freqs, t_mod
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                # S = T*H*W for 5D input; for action it's [B, S, 1, 1, D] so S = x.shape[1]
                seq_lens.append(x.shape[1] * x.shape[2] * x.shape[3])
                cached.append({
                    "block": block,
                    "residual_x": residual_x,
                    "gate_sa": gate_sa,
                    "shift_ca": shift_ca,
                    "scale_ca": scale_ca,
                    "gate_ca": gate_ca,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                })

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(f"Mask seq len mismatch: mask={attention_mask.shape[0]} vs tokens={total_seq}")

            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :, :]
                cache = cached[self.expert_order.index(name)]
                block = cache["block"]
                context = context_all.get(name)

                def _post_fn(
                    mixed_slc,
                    res_x,
                    g_sa,
                    s_ca, sc_ca, g_ca,
                    s_mlp, sc_mlp, g_mlp,
                    blk=block,
                    ctx=context,
                ):
                    return self._apply_expert_post_block(
                        blk, res_x, mixed_slc, g_sa, s_ca, sc_ca, g_ca, s_mlp, sc_mlp, g_mlp, ctx,
                    )

                if self.mot_checkpoint_mixed_attn and self.training:
                    updated = torch.utils.checkpoint.checkpoint(
                        _post_fn,
                        mixed_slice,
                        cache["residual_x"],
                        cache["gate_sa"],
                        cache["shift_ca"],
                        cache["scale_ca"],
                        cache["gate_ca"],
                        cache["shift_mlp"],
                        cache["scale_mlp"],
                        cache["gate_mlp"],
                        use_reentrant=False,
                    )
                else:
                    updated = _post_fn(
                        mixed_slice,
                        cache["residual_x"],
                        cache["gate_sa"],
                        cache["shift_ca"],
                        cache["scale_ca"],
                        cache["gate_ca"],
                        cache["shift_mlp"],
                        cache["scale_mlp"],
                        cache["gate_mlp"],
                    )

                tokens_all[name] = updated
                start = end

        return tokens_all

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: Optional[torch.Tensor],
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Prefill video branch and cache per-layer K/V for action inference."""
        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: List[Dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            q, k, v, residual_x, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self._build_expert_attention_io(
                block, x, video_freqs, video_t_mod
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = self._apply_expert_post_block(
                block, residual_x, mixed, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp, video_context,
            )
            kv_cache.append({"k": k, "v": v})

        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: Optional[torch.Tensor],
        video_kv_cache: List[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V."""
        expert = self.mixtures["action"]
        x = action_tokens
        total_seq_len = video_seq_len + x.shape[1] * x.shape[2] * x.shape[3]
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(f"Mask seq len mismatch: mask={attention_mask.shape[0]} vs expected={total_seq_len}")

        action_seq_len = x.shape[1] * x.shape[2] * x.shape[3]
        action_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            q_action, k_action, v_action, residual_x, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = self._build_expert_attention_io(
                block, x, None, action_t_mod
            )
            layer_cache = video_kv_cache[layer_idx]
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            q_cat = torch.cat([torch.empty_like(k_video), q_action], dim=1)  # dummy q for video part
            # Actually mixed_attention uses q_cat for all. But action_mask already blocks video->action,
            # and action->video is handled by the mask. We still need q for video in q_cat even though
            # its output will be ignored (since we only use the action slice).
            # Wait - in inference we don't need video output. But the mask ensures video tokens don't
            # attend to action. We DO need to run mixed attention with full q_cat because q_cat includes
            # video queries that attend to video keys (and we need this for the attention function to work).
            # However, in `forward_action_with_video_cache`, we only care about action output.
            # But if we don't update video tokens, the next layer's video K/V cache would be wrong.
            # Since we're using cached K/V, video tokens are not evolving. So q_cat for video can be
            # anything, but to be safe we should use the same q as prefill? No, in FastWAM's
            # `forward_action_with_video_cache`, they only compute action q/k/v and use cached video k/v.
            # They call mixed_attention with q_cat=q_action, k_cat=[k_video, k_action], v_cat=[v_video, v_action],
            # and attention_mask=action_attention_mask (which is action rows only).
            # That's more efficient! Let's do that.
            mixed_action = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )
            x = self._apply_expert_post_block(
                block, residual_x, mixed_action, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp, action_context,
            )

        return x
