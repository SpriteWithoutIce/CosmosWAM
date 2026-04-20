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

    def _encode_single_view(self, frame_B_C_1_H_W: torch.Tensor) -> torch.Tensor:
        """Encode a single-view frame through VAE with temporal padding.
        
        Native WanVAE_ CausalConv3d requires T >= kernel_size(3).
        We replicate-pad T=1 to T=4, encode, and get latent T=1.
        """
        B, C, _, H, W = frame_B_C_1_H_W.shape
        # Replicate single frame to T=4 to satisfy VAE temporal conv
        frame_padded = frame_B_C_1_H_W.repeat(1, 1, 4, 1, 1)  # [B, C, 4, H, W]
        latent = self.vae.encode(frame_padded)  # [B, 16, 1, h, w]
        return latent

    def build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video = sample["video"]              # [B, C, T, H, W] or [B, C, num_cameras, T, H, W]
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
            # Extract two views
            two_view_video = self._extract_two_views(video.to(device=dit_device, dtype=torch.float32))
            # Encode each camera view separately, then stack along temporal dim.
            # This way latent T = num_cameras (e.g. 2), and each camera is a distinct condition frame.
            main_frame = two_view_video[:, :, 0:1, :, :]   # [B, C, 1, H, W]
            wrist_frame = two_view_video[:, :, 1:2, :, :]  # [B, C, 1, H, W]
            main_latent = self._encode_single_view(main_frame)   # [B, 16, 1, h, w]
            wrist_latent = self._encode_single_view(wrist_frame) # [B, 16, 1, h, w]
            latents = torch.cat([main_latent, wrist_latent], dim=2)  # [B, 16, 2, h, w]
            latents = latents.to(device=dit_device, dtype=dit_dtype)
        
        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(latents.shape[0], 2, *latents.shape[2:], 
                                  device=dit_device, dtype=dit_dtype)
            latents = torch.cat([latents, padding], dim=1)  # [B, 18, 2, h, w]
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
        proprio = inputs["proprio"]          # [B, num_obs_steps, state_dim] or None
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # -------- DiT Encoder-only forward (no video denoising) --------
        # Use timestep=0 for clean feature extraction
        timesteps = torch.zeros(B, 1, device=device, dtype=dtype)

        # DiT forward, collecting all intermediate hidden states
        _, hidden_list = self.dit(
            x_B_C_T_H_W=latents,
            timesteps_B_T=timesteps,
            crossattn_emb=context,
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )

        # Extract hidden states from blocks[14:28] for action head
        video_cond_list: List[torch.Tensor] = []
        T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
        H_int = H_lat // self.dit.patch_spatial
        W_int = W_lat // self.dit.patch_spatial
        T_int = T_lat // self.dit.patch_temporal
        for action_layer_idx in range(self.action_head.num_layers):
            cosmos_layer_idx = 14 + action_layer_idx
            layer_hidden = hidden_list[cosmos_layer_idx]  # [B, T*H*W, D]
            B_actual, N, D_vid = layer_hidden.shape
            assert N == B_actual * T_int * H_int * W_int, (
                f"Hidden state size mismatch: N={N}, expected {B_actual}*{T_int}*{H_int}*{W_int}"
            )
            layer_grid = layer_hidden.view(B_actual, T_int, H_int, W_int, D_vid)
            video_cond_list.append(layer_grid)

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

        return loss_action, {
            "loss_action": loss_action.detach(),
            "loss_total": loss_action.detach(),
        }

    @torch.no_grad()
    def infer_action(
        self,
        first_frame_pixels: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Inference: given first frame(s), predict action sequence.
        
        Args:
            first_frame_pixels: 
                - [B, C, H, W] single image
                - [B, C, 1, H, W] single view with temporal dim (from deploy_policy)
                - [B, num_cameras, C, H, W] multi-view images
        """
        self.eval()
        device = first_frame_pixels.device

        # Normalize input to two-view temporal format [B, C, 2, H, W]
        if first_frame_pixels.ndim == 4:
            # [B, C, H, W] -> add temporal dim and replicate
            two_view = first_frame_pixels.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B, C, 2, H, W]
        elif first_frame_pixels.ndim == 5:
            if first_frame_pixels.shape[2] == 1:
                # [B, C, 1, H, W] from deploy_policy -> replicate to T=2
                two_view = first_frame_pixels.repeat(1, 1, 2, 1, 1)  # [B, C, 2, H, W]
            elif first_frame_pixels.shape[1] in (2, 3):
                # [B, num_cameras, C, H, W] multi-view -> permute to temporal
                # Assume num_cameras is small (2 or 3), treat cameras as temporal frames
                two_view = first_frame_pixels.permute(0, 2, 1, 3, 4)  # [B, C, num_cameras, H, W]
            else:
                raise ValueError(
                    f"Unexpected 5D first_frame_pixels shape {tuple(first_frame_pixels.shape)}. "
                    "Expected [B, C, 1, H, W] or [B, num_cameras, C, H, W]."
                )
        else:
            raise ValueError(
                f"Unexpected first_frame_pixels ndim={first_frame_pixels.ndim}, shape {tuple(first_frame_pixels.shape)}"
            )

        B = two_view.shape[0]

        # Encode each view separately, then stack along temporal dim
        main_frame = two_view[:, :, 0:1, :, :]   # [B, C, 1, H, W]
        wrist_frame = two_view[:, :, 1:2, :, :]  # [B, C, 1, H, W]
        main_latent = self._encode_single_view(main_frame)   # [B, 16, 1, h, w]
        wrist_latent = self._encode_single_view(wrist_frame) # [B, 16, 1, h, w]
        latents = torch.cat([main_latent, wrist_latent], dim=2)  # [B, 16, 2, h, w]

        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(
                latents.shape[0], 2, 
                *latents.shape[2:], 
                device=device, dtype=latents.dtype
            )
            latents = torch.cat([latents, padding], dim=1)  # [B, 18, 2, h, w]

        # Run dit once to populate video features cache
        context_dtype = next(self.dit.parameters()).dtype
        _ = self.dit(
            x_B_C_T_H_W=latents,
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=latents.dtype),
            crossattn_emb=context.to(dtype=context_dtype),
            intermediate_feature_ids=list(range(self.dit.num_blocks)),
        )
        video_cond_cache = [self._video_features[14 + i].detach().clone() for i in range(self.action_head.num_layers)]
        
        # Action flow matching denoising
        action = torch.randn(B, action_horizon, self.action_head.action_dim, device=device)
        for i in range(num_inference_steps):
            # Use 1-t to match training: t=0 is noise, t=1 is action
            t = torch.full((B,), i / num_inference_steps, device=device, dtype=torch.float32)
            pred = self.action_head(action, video_cond_cache, t, state=None)
            # Flow from noise to action: dx/dt = pred
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
