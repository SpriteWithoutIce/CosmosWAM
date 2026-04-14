from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.autograd.functional as autograd_f

from .dit_wrapper import MiniTrainDIT
from .vae_wrapper import VideoTokenizerInterface
from .action_head_mot import ActionExpert
from .mot import MoT


class CosmosWAM(nn.Module):
    def __init__(
        self,
        mot: MoT,
        vae: VideoTokenizerInterface,
        lambda_action: float = 1.0,
        num_cond_frames: int = 1,
        actions_per_latent: int = 8,
    ):
        super().__init__()
        self.mot = mot
        self.vae = vae
        self.lambda_action = float(lambda_action)
        self.num_cond_frames = int(num_cond_frames)
        self.actions_per_latent = int(actions_per_latent)

    @property
    def dit(self):
        """Alias for video expert (trainer compatibility)."""
        return self.mot.mixtures["video"]

    @property
    def action_head(self):
        """Alias for action expert."""
        return self.mot.mixtures["action"]

    def build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video = sample["video"]              # [B, 3, T, H, W]
        action = sample["action"]            # [B, Ta, action_dim]
        context = sample["context"]          # [B, L, D]
        proprio = sample.get("proprio", None)

        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype

        action = action.to(device=dit_device, dtype=dit_dtype)
        context = context.to(device=dit_device, dtype=dit_dtype)
        if proprio is not None:
            proprio = proprio.to(device=dit_device, dtype=dit_dtype)

        with torch.no_grad():
            latents = self.vae.encode(video)
            latents = latents.to(device=dit_device, dtype=dit_dtype)

        if latents.shape[1] == 16:
            padding = torch.zeros(latents.shape[0], 2, *latents.shape[2:],
                                  device=dit_device, dtype=dit_dtype)
            latents = torch.cat([latents, padding], dim=1)

        return {
            "latents": latents,
            "action": action,
            "context": context,
            "proprio": proprio,
        }

    def training_loss(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        inputs = self.build_inputs(sample)
        latents = inputs["latents"]
        action = inputs["action"]
        context = inputs["context"]
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # -------- Video Rectified Flow --------
        noise_v = torch.randn_like(latents)
        t_video = torch.rand(B, device=device, dtype=torch.float32)
        t_video_bf16 = t_video.view(B, 1, 1, 1, 1).to(dtype=dtype)
        noisy_latents = (1.0 - t_video_bf16) * noise_v + t_video_bf16 * latents
        noisy_latents[:, :, : self.num_cond_frames] = latents[:, :, : self.num_cond_frames].clone()

        video_pre = self.dit.pre_dit(
            x_B_C_T_H_W=noisy_latents,
            timesteps_B_T=t_video.unsqueeze(1).to(dtype=dtype),
            crossattn_emb=context,
        )

        # -------- Action iMF --------
        noise_a = torch.randn_like(action)
        t_action = torch.rand(B, device=device, dtype=torch.float32)

        # 50% r = t (boundary), 50% r ~ Uniform(0, t)
        mask_r_eq_t = torch.rand(B, device=device) < 0.5
        r = torch.where(
            mask_r_eq_t,
            t_action,
            torch.rand(B, device=device) * t_action,
        )

        t_action_cast = t_action.view(B, 1, 1).to(dtype=action.dtype)
        noisy_action = (1.0 - t_action_cast) * noise_a + t_action_cast * action
        target_v_cond = noise_a - action  # e - a_1

        action_pre = self.action_head.pre_dit(
            action_tokens=noisy_action,
            r=r,
            t=t_action,
            context=context,
        )

        # Build MoT mask
        video_seq_len = video_pre["tokens"].shape[1]
        action_seq_len = action_pre["meta"]["seq_len"]
        H_lat, W_lat = latents.shape[3], latents.shape[4]
        video_tokens_per_frame = (H_lat // self.dit.patch_spatial) * (W_lat // self.dit.patch_spatial)

        attention_mask = self.mot._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_cond_frames=self.num_cond_frames,
            actions_per_latent=self.actions_per_latent,
        )

        # Single MoT forward for both branches
        tokens_out = self.mot.forward(
            embeds_all={
                "video": video_pre["tokens"].view(B, video_pre["meta"]["T"], video_pre["meta"]["H"], video_pre["meta"]["W"], video_pre["meta"]["D"]),
                "action": action_pre["tokens_5d"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": None,
            },
            context_all={
                "video": video_pre["context"],
                "action": action_pre["context"],
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_v = self.dit.post_dit(
            rearrange(tokens_out["video"], "b t h w d -> b (t h w) d"),
            video_pre,
        )

        target_v = latents - noise_v
        if self.num_cond_frames > 0:
            loss_video = F.mse_loss(pred_v[:, :, self.num_cond_frames:], target_v[:, :16, self.num_cond_frames:])
        else:
            loss_video = F.mse_loss(pred_v, target_v[:, :16])

        # Define u_theta for action branch (re-runs action pre_dit + mot action slice + post_dit)
        # Video input tokens are treated as constants (use original pre_dit tokens, not mot output)
        video_tokens_const = video_pre["tokens"].view(
            B, video_pre["meta"]["T"], video_pre["meta"]["H"], video_pre["meta"]["W"], video_pre["meta"]["D"]
        ).detach()
        video_t_mod_const = video_pre["t_mod"].detach()
        video_context_const = video_pre["context"].detach()
        video_freqs_const = video_pre["freqs"]
        action_context_const = action_pre["context"].detach() if action_pre["context"] is not None else None

        def u_theta_fn(z, rv, tv):
            ap = self.action_head.pre_dit(z, rv, tv, context)
            to = self.mot.forward(
                embeds_all={
                    "video": video_tokens_const,
                    "action": ap["tokens_5d"],
                },
                attention_mask=attention_mask,
                freqs_all={
                    "video": video_freqs_const,
                    "action": None,
                },
                context_all={
                    "video": video_context_const,
                    "action": ap["context"],
                },
                t_mod_all={
                    "video": video_t_mod_const,
                    "action": ap["t_mod"],
                },
            )
            return self.action_head.post_dit(to["action"], ap)

        # v_pred at (z_t, r, t)
        v_pred = u_theta_fn(noisy_action, r, t_action)

        # Samples where r < t: compute JVP
        r_lt_t_mask = (r < t_action).view(B, 1, 1)
        if r_lt_t_mask.any():
            u_pred, dudt = autograd_f.jvp(
                lambda z, rv, tv: u_theta_fn(z, rv, tv),
                (noisy_action, r, t_action),
                (v_pred, torch.zeros_like(r), torch.ones_like(t_action)),
            )
            V_theta = torch.where(
                r_lt_t_mask,
                u_pred + (t_action - r).view(B, 1, 1) * dudt.detach(),
                v_pred,
            )
        else:
            V_theta = v_pred

        loss_action = F.mse_loss(V_theta, target_v_cond)

        loss = loss_video + self.lambda_action * loss_action
        return loss, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "loss_total": loss.detach(),
        }

    @torch.no_grad()
    def infer_action(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        if first_frame_pixels.ndim == 4:
            first_frame_pixels = first_frame_pixels.unsqueeze(0)
        B = first_frame_pixels.shape[0]
        device = first_frame_pixels.device
        dit_dtype = next(self.dit.parameters()).dtype
        context = context.to(dtype=dit_dtype)

        latents = self.vae.encode(first_frame_pixels)
        first_frame_latent = latents[:, :, :1, :, :]

        if first_frame_latent.shape[1] == 16:
            padding = torch.zeros(
                first_frame_latent.shape[0], 2,
                *first_frame_latent.shape[2:],
                device=device, dtype=first_frame_latent.dtype
            )
            first_frame_latent = torch.cat([first_frame_latent, padding], dim=1)

        video_pre = self.dit.pre_dit(
            x_B_C_T_H_W=first_frame_latent,
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=latents.dtype),
            crossattn_emb=context,
        )

        video_seq_len = video_pre["tokens"].shape[1]
        H_lat, W_lat = first_frame_latent.shape[3], first_frame_latent.shape[4]
        video_tokens_per_frame = (H_lat // self.dit.patch_spatial) * (W_lat // self.dit.patch_spatial)

        full_mask = self.mot._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_cond_frames=self.num_cond_frames,
            actions_per_latent=self.actions_per_latent,
        )
        video_self_mask = full_mask[:video_seq_len, :video_seq_len]

        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"].view(B, video_pre["meta"]["T"], video_pre["meta"]["H"], video_pre["meta"]["W"], video_pre["meta"]["D"]),
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context=video_pre["context"],
            video_attention_mask=video_self_mask,
        )

        z_1 = torch.randn(B, action_horizon, self.action_head.action_dim, device=device, dtype=dit_dtype)
        action_pre = self.action_head.pre_dit(
            action_tokens=z_1,
            r=torch.zeros(B, device=device),
            t=torch.ones(B, device=device),
            context=context,
        )

        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens_5d"],
            action_t_mod=action_pre["t_mod"],
            action_context=action_pre["context"],
            video_kv_cache=video_kv_cache,
            attention_mask=full_mask,
            video_seq_len=video_seq_len,
        )
        pred = self.action_head.post_dit(action_tokens, action_pre)
        a_0 = z_1 - pred
        return a_0

    @torch.no_grad()
    def infer_joint(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> Dict[str, Any]:
        """Inference: jointly generate future video latents and actions.
        
        Video is denoised via standard multi-step Euler. Action is predicted
        using ONLY the clean conditional frame (first frame), matching the
        original architecture's behavior.
        """
        self.eval()
        if first_frame_pixels.ndim == 4:
            first_frame_pixels = first_frame_pixels.unsqueeze(0)
        B = first_frame_pixels.shape[0]
        device = first_frame_pixels.device
        dit_dtype = next(self.dit.parameters()).dtype
        context = context.to(dtype=dit_dtype)

        latents = self.vae.encode(first_frame_pixels)
        first_frame_latent = latents[:, :, :1, :, :]
        C, Tl, Hl, Wl = latents.shape[1:]

        if first_frame_latent.shape[1] == 16:
            padding = torch.zeros(B, 2, Tl, Hl, Wl, device=device, dtype=latents.dtype)
            first_frame_latent = torch.cat([first_frame_latent, padding], dim=1)

        if C == 16:
            video_latents = torch.randn(B, 18, Tl, Hl, Wl, device=device, dtype=dit_dtype)
            video_latents[:, :, :1] = first_frame_latent.clone()
        else:
            video_latents = torch.randn(B, C, Tl, Hl, Wl, device=device, dtype=dit_dtype)
            video_latents[:, :, :1] = first_frame_latent.clone()

        # Denoise video with standard multi-step Euler
        for i in range(num_inference_steps):
            t = torch.full((B,), 1.0 - i / num_inference_steps, device=device, dtype=torch.float32)
            dt = -1.0 / num_inference_steps

            pred_v = self.dit(
                x_B_C_T_H_W=video_latents,
                timesteps_B_T=t.unsqueeze(1),
                crossattn_emb=context,
            )
            video_latents = video_latents + dt * pred_v
            video_latents[:, :, :1] = first_frame_latent.clone()

        # Action: single-step from clean first-frame only (same as infer_action)
        video_pre = self.dit.pre_dit(
            x_B_C_T_H_W=first_frame_latent,
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=first_frame_latent.dtype),
            crossattn_emb=context,
        )

        video_seq_len = video_pre["tokens"].shape[1]
        H_lat, W_lat = first_frame_latent.shape[3], first_frame_latent.shape[4]
        video_tokens_per_frame = (H_lat // self.dit.patch_spatial) * (W_lat // self.dit.patch_spatial)

        full_mask = self.mot._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_cond_frames=self.num_cond_frames,
            actions_per_latent=self.actions_per_latent,
        )
        video_self_mask = full_mask[:video_seq_len, :video_seq_len]

        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"].view(B, video_pre["meta"]["T"], video_pre["meta"]["H"], video_pre["meta"]["W"], video_pre["meta"]["D"]),
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context=video_pre["context"],
            video_attention_mask=video_self_mask,
        )

        z_1 = torch.randn(B, action_horizon, self.action_head.action_dim, device=device, dtype=dit_dtype)
        action_pre = self.action_head.pre_dit(
            action_tokens=z_1,
            r=torch.zeros(B, device=device),
            t=torch.ones(B, device=device),
            context=context,
        )

        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens_5d"],
            action_t_mod=action_pre["t_mod"],
            action_context=action_pre["context"],
            video_kv_cache=video_kv_cache,
            attention_mask=full_mask,
            video_seq_len=video_seq_len,
        )
        pred = self.action_head.post_dit(action_tokens, action_pre)
        action = z_1 - pred

        video_pixels = self.vae.decode(video_latents)
        return {
            "video_pixels": video_pixels,
            "action": action,
        }
