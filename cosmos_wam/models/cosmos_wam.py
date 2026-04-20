from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit_wrapper import MiniTrainDIT
from .vae_wrapper import VideoTokenizerInterface
from .action_head import ActionDiT


class CosmosWAM(nn.Module):
    def __init__(
        self,
        dit: MiniTrainDIT,
        vae: VideoTokenizerInterface,
        action_head: ActionDiT,
        lambda_action: float = 1.0,
        lambda_video: float = 0.5,
        num_cond_frames: int = 1,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.action_head = action_head
        self.lambda_action = float(lambda_action)
        self.lambda_video = float(lambda_video)
        self.num_cond_frames = int(num_cond_frames)

        # Register forward hooks on blocks[14:28] to capture hidden states
        self._video_features: Dict[int, torch.Tensor] = {}
        for layer_idx in range(14, 28):
            self.dit.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output: [B, T, H, W, D]
            self._video_features[layer_idx] = output
        return hook

    def _extract_two_views(self, video: torch.Tensor) -> torch.Tensor:
        """Extract main and wrist views from video tensor.
        
        Supports two input formats:
          - [B, C, T, H, W] (single camera, temporal frames)
          - [B, C, num_cameras, T, H, W] (multi-camera from concat_multi_camera=None)
        
        Returns:
            two_view_video: [B, C, 2, H, W] where T=2 is [main_view, wrist_view]
        """
        if video.ndim == 5:
            main = video[:, :, 0:1, :, :]
            if video.shape[2] >= 2:
                wrist_idx = min(1, video.shape[2] - 1)
                wrist = video[:, :, wrist_idx:wrist_idx+1, :, :]
            else:
                wrist = main
            return torch.cat([main, wrist], dim=2)
        elif video.ndim == 6:
            main = video[:, :, 0, 0:1, :, :]
            if video.shape[2] >= 2:
                wrist = video[:, :, 1, 0:1, :, :]
            else:
                wrist = main
            return torch.cat([main, wrist], dim=2)
        else:
            raise ValueError(
                f"Unexpected video shape {tuple(video.shape)}. "
                "Expected [B, C, T, H, W] or [B, C, num_cameras, T, H, W]."
            )

    def build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video = sample["video"]              # [B, C, T, H, W] (already concatenated by dataset)
        action = sample["action"]            # [B, Ta, action_dim]
        context = sample["context"]          # [B, L, D]
        proprio = sample.get("proprio", None)

        # Get DiT's device and dtype (match model parameters for mixed precision)
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype
        
        # Move inputs to device, using model's dtype (supports fp32/bf16/fp16)
        action = action.to(device=dit_device, dtype=dit_dtype)
        context = context.to(device=dit_device, dtype=dit_dtype)
        if proprio is not None:
            proprio = proprio.to(device=dit_device, dtype=dit_dtype)

        with torch.no_grad():
            video_input = video.to(device=dit_device, dtype=torch.float32)
            # VAE temporal conv requires T >= kernel_size(3). Pad if needed.
            if video_input.shape[2] < 4:
                repeat_factor = (4 + video_input.shape[2] - 1) // video_input.shape[2]
                video_input = video_input.repeat(1, 1, repeat_factor, 1, 1)[:, :, :4, :, :]
            # VAE encode - VAE uses its own dtype
            latents = self.vae.encode(video_input)  # [B, C_latent, T_latent, H_latent, W_latent]
            # Move to DiT's device and convert to DiT's dtype
            latents = latents.to(device=dit_device, dtype=dit_dtype)
        
        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(latents.shape[0], 2, *latents.shape[2:], 
                                  device=dit_device, dtype=dit_dtype)
            latents = torch.cat([latents, padding], dim=1)  # [B, 18, T_latent, H_latent, W_latent]
        return {
            "latents": latents,
            "action": action,
            "context": context,
            "proprio": proprio,
        }

    def training_loss(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        inputs = self.build_inputs(sample)
        latents = inputs["latents"]          # [B, 18, T_latent, H_latent, W_latent]
        action = inputs["action"]            # [B, Ta, action_dim]
        context = inputs["context"]          # [B, L, D]
        proprio = inputs["proprio"]          # [B, num_obs_steps, state_dim] or None
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        T_lat = latents.shape[2]

        # -------- Video Flow Matching --------
        # Sample noise for latent channels (first 16 ch)
        noise_z = torch.randn(B, 16, T_lat, latents.shape[3], latents.shape[4],
                              device=device, dtype=torch.float32)
        
        # Target velocity: v = latent - noise
        target_v = latents[:, :16, ...].float() - noise_z  # [B, 16, T, H, W]
        
        # Sample timestep
        t_video = torch.rand(B, device=device, dtype=torch.float32)
        t_expanded = t_video.view(B, 1, 1, 1, 1)
        
        # Noisy latent: x_t = (1-t) * noise + t * latent
        noisy_latent = (1 - t_expanded) * noise_z + t_expanded * latents[:, :16, ...].float()
        
        # Keep condition frame clean (first latent frame)
        noisy_latent[:, :, 0:1, :, :] = latents[:, :16, 0:1, :, :].float()
        
        # Concat padding mask channels back
        noisy_input = torch.cat([noisy_latent, latents[:, 16:, ...]], dim=1).to(dtype=dtype)
        
        # DiT forward, collecting all intermediate hidden states
        pred, hidden_list = self.dit(
            x_B_C_T_H_W=noisy_input,
            timesteps_B_T=t_video,
            crossattn_emb=context,
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )
        
        # Video loss: only on generation frames (index >= 1)
        pred_v = pred[:, :16, ...]  # [B, 16, T, H, W]
        loss_video_raw = F.mse_loss(pred_v, target_v.to(dtype=dtype), reduction='none')
        video_mask = torch.zeros(B, 1, T_lat, 1, 1, device=device, dtype=dtype)
        video_mask[:, :, 1:, :, :] = 1.0  # mask out condition frame
        loss_video = (loss_video_raw * video_mask).mean()

        # -------- Extract hidden states for action head (only condition frame) --------
        video_cond_list: List[torch.Tensor] = []
        H_lat, W_lat = latents.shape[3], latents.shape[4]
        H_int = H_lat // self.dit.patch_spatial
        W_int = W_lat // self.dit.patch_spatial
        T_int = T_lat // self.dit.patch_temporal
        for action_layer_idx in range(self.action_head.num_layers):
            cosmos_layer_idx = 14 + action_layer_idx
            layer_hidden = hidden_list[cosmos_layer_idx]  # [B, T*H*W, D]
            B_actual, N, D_vid = layer_hidden.shape
            assert N == T_int * H_int * W_int, (
                f"Hidden state size mismatch: N={N}, expected {T_int}*{H_int}*{W_int}={T_int * H_int * W_int}"
            )
            layer_grid = layer_hidden.view(B_actual, T_int, H_int, W_int, D_vid)
            # Only take condition frame (first temporal position)
            cond_grid = layer_grid[:, 0:1, ...]  # [B, 1, H, W, D]
            video_cond_list.append(cond_grid)

        # -------- Action Rectified Flow --------
        noise_a = torch.randn_like(action)
        t_action = torch.rand(B, device=device, dtype=torch.float32)

        # Ensure t_action is same dtype as action for mixed precision
        t_action_cast = t_action.view(B, 1, 1).to(dtype=action.dtype)
        noisy_action = (1.0 - t_action_cast) * noise_a + t_action_cast * action
        target_a = action - noise_a

        # Prepare state for action head: use the first observation step
        state = None
        if proprio is not None and self.action_head.state_dim > 0:
            state = proprio[:, 0, :]  # [B, state_dim]

        pred_action = self.action_head(noisy_action, video_cond_list, t_action, state=state)
        loss_action = F.mse_loss(pred_action, target_a)

        loss_total = self.lambda_video * loss_video + self.lambda_action * loss_action
        return loss_total, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "loss_total": loss_total.detach(),
        }

    @torch.no_grad()
    def infer_action(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Inference: given condition frame(s), predict action sequence.
        
        Args:
            first_frame_pixels: 
                - [B, C, H, W] single concatenated image (horizontal concat of 2 cameras)
                - [B, C, 1, H, W] with temporal dim
        """
        self.eval()
        device = first_frame_pixels.device

        # Add temporal dim if needed
        if first_frame_pixels.ndim == 4:
            video_input = first_frame_pixels.unsqueeze(2)  # [B, C, 1, H, W]
        elif first_frame_pixels.ndim == 5:
            video_input = first_frame_pixels  # [B, C, T, H, W]
        else:
            raise ValueError(
                f"Unexpected first_frame_pixels ndim={first_frame_pixels.ndim}, shape {tuple(first_frame_pixels.shape)}"
            )

        B = video_input.shape[0]

        # VAE temporal conv requires T >= 4; replicate if needed
        if video_input.shape[2] < 4:
            repeat_factor = (4 + video_input.shape[2] - 1) // video_input.shape[2]
            video_input = video_input.repeat(1, 1, repeat_factor, 1, 1)[:, :, :4, :, :]
        
        # VAE encode
        latents = self.vae.encode(video_input)  # [B, C, T_latent, H, W]

        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(
                latents.shape[0], 2, 
                *latents.shape[2:], 
                device=device, dtype=latents.dtype
            )
            latents = torch.cat([latents, padding], dim=1)  # [B, 18, T_latent, H, W]

        # Run dit clean pass (t=0) to extract features from condition frame
        context_dtype = next(self.dit.parameters()).dtype
        _ = self.dit(
            x_B_C_T_H_W=latents,
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=latents.dtype),
            crossattn_emb=context.to(dtype=context_dtype),
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )
        # Only take condition frame (first temporal position) hidden states
        video_cond_cache = []
        for i in range(self.action_head.num_layers):
            feat = self._video_features[14 + i]  # [B, T_int, H_int, W_int, D]
            video_cond_cache.append(feat[:, 0:1, ...].detach().clone())
        
        # Action flow matching denoising
        action = torch.randn(B, action_horizon, self.action_head.action_dim, device=device)
        for i in range(num_inference_steps):
            t = torch.full((B,), i / num_inference_steps, device=device, dtype=torch.float32)
            pred = self.action_head(action, video_cond_cache, t, state=None)
            dt = 1.0 / num_inference_steps
            action = action + dt * pred
        return action

    @torch.no_grad()
    def infer_joint(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> Dict[str, Any]:
        """Inference: jointly generate future video latents and actions.
        NOTE: This method is kept for backward compatibility but now uses
        encoder-only DiT (t=0) for video conditioning.
        """
        self.eval()
        # For encoder-only mode, we just call infer_action and return dummy video
        action = self.infer_action(
            first_frame_pixels=first_frame_pixels,
            action_horizon=action_horizon,
            context=context,
            num_inference_steps=num_inference_steps,
        )
        return {
            "video_pixels": None,
            "action": action,
        }
