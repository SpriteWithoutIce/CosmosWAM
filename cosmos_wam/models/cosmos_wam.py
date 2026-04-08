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
        num_cond_frames: int = 1,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.action_head = action_head
        self.lambda_action = float(lambda_action)
        self.num_cond_frames = int(num_cond_frames)

        # Register forward hooks on blocks[14:28] to capture conditional-frame hidden states
        self._video_features: Dict[int, torch.Tensor] = {}
        for layer_idx in range(14, 28):
            self.dit.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output: [B, T, H, W, D]
            cond = output[:, : self.num_cond_frames, :, :, :]  # [B, K, H, W, D]
            self._video_features[layer_idx] = cond
        return hook

    def build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video = sample["video"]              # [B, 3, T, H, W]
        action = sample["action"]            # [B, Ta, action_dim]
        context = sample["context"]          # [B, L, D]
        proprio = sample.get("proprio", None)

        # Get DiT's device (where the model parameters are)
        dit_device = next(self.dit.parameters()).device
        
        # Ensure all inputs are float32 (for no mixed precision mode)
        action = action.to(device=dit_device, dtype=torch.float32)
        context = context.to(device=dit_device, dtype=torch.float32)
        if proprio is not None:
            proprio = proprio.to(device=dit_device, dtype=torch.float32)

        with torch.no_grad():
            # VAE encode - input needs to be on VAE's device and float32
            latents = self.vae.encode(video)  # [B, C_latent, T_latent, H_latent, W_latent]
            # Move to DiT's device and convert to float32
            latents = latents.to(device=dit_device, dtype=torch.float32)
        
        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(latents.shape[0], 2, *latents.shape[2:], 
                                  device=dit_device, dtype=torch.float32)
            latents = torch.cat([latents, padding], dim=1)  # [B, 18, T, H, W]
        return {
            "latents": latents,
            "action": action,
            "context": context,
            "proprio": proprio,
        }

    def training_loss(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        inputs = self.build_inputs(sample)
        latents = inputs["latents"]          # [B, C, T_latent, H_latent, W_latent]
        action = inputs["action"]            # [B, Ta, action_dim]
        context = inputs["context"]          # [B, L, D]
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # -------- Video Rectified Flow --------
        noise_v = torch.randn_like(latents)
        t_video = torch.rand(B, device=device, dtype=torch.float32)

        # Add noise: x_t = (1 - t) * noise + t * x0
        # Ensure t_video is same dtype as latents for mixed precision
        t_video_bf16 = t_video.view(B, 1, 1, 1, 1).to(dtype=dtype)
        noisy_latents = (1.0 - t_video_bf16) * noise_v + t_video_bf16 * latents
        # Replace conditional frames with clean latents
        noisy_latents[:, :, : self.num_cond_frames] = latents[:, :, : self.num_cond_frames].clone()

        # DiT forward, collecting all intermediate hidden states
        pred_v, hidden_list = self.dit(
            x_B_C_T_H_W=noisy_latents,
            timesteps_B_T=t_video.unsqueeze(1).to(dtype=dtype),
            crossattn_emb=context,
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )

        target_v = latents - noise_v
        # Video loss: skip conditional frames
        # pred_v is [B, 16, T, H, W], target_v is [B, 18, T, H, W] (with 2 padding channels)
        # Only compare the first 16 channels
        if self.num_cond_frames > 0:
            loss_video = F.mse_loss(pred_v[:, :, self.num_cond_frames :], target_v[:, :16, self.num_cond_frames :])
        else:
            loss_video = F.mse_loss(pred_v, target_v[:, :16])

        # -------- Action Rectified Flow --------
        noise_a = torch.randn_like(action)
        t_action = torch.rand(B, device=device, dtype=torch.float32)

        # Ensure t_action is same dtype as action for mixed precision
        t_action_cast = t_action.view(B, 1, 1).to(dtype=action.dtype)
        noisy_action = (1.0 - t_action_cast) * noise_a + t_action_cast * action
        target_a = action - noise_a

        # Extract conditional-frame hidden states from cosmos layers 14-27
        video_cond_list: List[torch.Tensor] = []
        for action_layer_idx in range(self.action_head.num_layers):
            cosmos_layer_idx = 14 + action_layer_idx
            layer_hidden = hidden_list[cosmos_layer_idx]  # [B, T*H*W, D]
            # Reshape back to [B, T, H, W, D] using latent shape
            T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
            # Note: after patchify, internal T/H/W may differ from latent T/H/W
            # We infer from layer_hidden token count
            B_actual, N, D_vid = layer_hidden.shape
            # Find internal T, H, W such that T*H*W == N
            # Since layer_hidden comes from x_B_T_H_W_D before final reshape,
            # N should equal T_int * H_int * W_int
            # We can recover T_int from output shape of dit, but easier:
            # The hook output was [B, T_int, H_int, W_int, D] and we flattened K*H*W.
            # So we need H_int and W_int. They are: latent_H//patch_spatial, latent_W//patch_spatial
            H_int = H_lat // self.dit.patch_spatial
            W_int = W_lat // self.dit.patch_spatial
            T_int = N // (H_int * W_int)
            layer_grid = layer_hidden.view(B_actual, T_int, H_int, W_int, D_vid)
            cond_grid = layer_grid[:, : self.num_cond_frames, :, :, :]  # [B, K, H_int, W_int, D]
            video_cond_list.append(cond_grid)

        pred_action = self.action_head(noisy_action, video_cond_list, t_action)
        loss_action = F.mse_loss(pred_action, target_a)

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
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Inference: given first frame, predict action sequence."""
        self.eval()
        if first_frame_pixels.ndim == 4:
            first_frame_pixels = first_frame_pixels.unsqueeze(0)
        B = first_frame_pixels.shape[0]
        device = first_frame_pixels.device

        # Encode first frame
        latents = self.vae.encode(first_frame_pixels)  # [B, C, T_latent, H, W]
        first_frame_latent = latents[:, :, :1, :, :]   # [B, C, 1, H, W]

        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if first_frame_latent.shape[1] == 16:
            padding = torch.zeros(
                first_frame_latent.shape[0], 2, 
                *first_frame_latent.shape[2:], 
                device=device, dtype=first_frame_latent.dtype
            )
            first_frame_latent = torch.cat([first_frame_latent, padding], dim=1)  # [B, 18, 1, H, W]

        # Build minimal latent with only conditional frame to trigger hooks
        dummy_latents = first_frame_latent

        # Run dit once to populate video features cache
        _ = self.dit(
            x_B_C_T_H_W=dummy_latents,
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=latents.dtype),
            crossattn_emb=context,
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )
        video_cond_cache = [self._video_features[14 + i].detach().clone() for i in range(self.action_head.num_layers)]
        
        # Action flow matching denoising
        action = torch.randn(B, action_horizon, self.action_head.action_dim, device=device)
        for i in range(num_inference_steps):
            # Use 1-t to match training: t=0 is noise, t=1 is action
            t = torch.full((B,), i / num_inference_steps, device=device, dtype=torch.float32)
            pred = self.action_head(action, video_cond_cache, t)
            # Flow from noise to action: dx/dt = pred
            dt = 1.0 / num_inference_steps
            action = action + dt * pred
            # DEBUG: Print every few steps
            if i % 4 == 0 or i == num_inference_steps - 1:
                print(f"[COSMOS_WAM] Step {i:2d}: t={t.item():.3f}, pred_std={pred.std().item():.4f}, action_std={action.std().item():.4f}", flush=True)
        
        # DEBUG: Final action stats
        print(f"[COSMOS_WAM] Final action: mean={action.mean().item():.4f}, std={action.std().item():.4f}, "
              f"min={action.min().item():.4f}, max={action.max().item():.4f}", flush=True)
        return action

    @torch.no_grad()
    def infer_joint(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> Dict[str, Any]:
        """Inference: jointly generate future video latents and actions."""
        self.eval()
        if first_frame_pixels.ndim == 4:
            first_frame_pixels = first_frame_pixels.unsqueeze(0)
        B = first_frame_pixels.shape[0]
        device = first_frame_pixels.device
        dtype = self.dit.model_channels  # not used as dtype, just for ref

        latents = self.vae.encode(first_frame_pixels)
        first_frame_latent = latents[:, :, :1, :, :]
        C, Tl, Hl, Wl = latents.shape[1:]

        # Initialize video latents with noise, keep first frame clean
        video_latents = torch.randn(B, C, Tl, Hl, Wl, device=device)
        video_latents[:, :, :1] = first_frame_latent.clone()

        # Action init
        action = torch.randn(B, action_horizon, self.action_head.action_dim, device=device)

        for i in range(num_inference_steps):
            t = torch.full((B,), 1.0 - i / num_inference_steps, device=device, dtype=torch.float32)

            # Video prediction
            pred_v, hidden_list = self.dit(
                x_B_C_T_H_W=video_latents,
                timesteps_B_T=t.unsqueeze(1),
                crossattn_emb=context,
                intermediate_feature_ids=list(range(self.dit.num_blocks)),
            )
            dt = -1.0 / num_inference_steps
            video_latents = video_latents + dt * pred_v
            video_latents[:, :, :1] = first_frame_latent.clone()

            # Action prediction using current step's hidden states
            video_cond_list = []
            for action_layer_idx in range(self.action_head.num_layers):
                cosmos_layer_idx = 14 + action_layer_idx
                layer_hidden = hidden_list[cosmos_layer_idx]
                B_actual, N, D_vid = layer_hidden.shape
                H_int = Hl // self.dit.patch_spatial
                W_int = Wl // self.dit.patch_spatial
                T_int = N // (H_int * W_int)
                layer_grid = layer_hidden.view(B_actual, T_int, H_int, W_int, D_vid)
                cond_grid = layer_grid[:, : self.num_cond_frames, :, :, :]
                video_cond_list.append(cond_grid)

            pred_a = self.action_head(action, video_cond_list, t)
            action = action + dt * pred_a

        # Decode video latents to pixels
        video_pixels = self.vae.decode(video_latents)
        return {
            "video_pixels": video_pixels,
            "action": action,
        }
