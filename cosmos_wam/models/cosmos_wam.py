from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_wrapper import MiniTrainDIT
from .vae_wrapper import VideoTokenizerInterface
from .latent_query import LatentQueryEncoder, TrajectoryHead
from .action_head import ActionHeadIMF


class CosmosWAM(nn.Module):
    def __init__(
        self,
        dit: MiniTrainDIT,
        vae: VideoTokenizerInterface,
        latent_query_encoder: LatentQueryEncoder,
        trajectory_head: TrajectoryHead,
        action_head: ActionHeadIMF,
        lambda_action: float = 1.0,
        lambda_traj: float = 1.0,
        num_cond_frames: int = 1,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.latent_query_encoder = latent_query_encoder
        self.trajectory_head = trajectory_head
        self.action_head = action_head
        self.lambda_action = float(lambda_action)
        self.lambda_traj = float(lambda_traj)
        self.num_cond_frames = int(num_cond_frames)

    def build_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        video = sample["video"]              # [B, 3, T, H, W]
        action = sample["action"]            # [B, Ta, action_dim]
        context = sample["context"]          # [B, L, D]
        proprio = sample.get("proprio", None)
        state = sample.get("state", None)
        eef_pos = sample.get("eef_pos", None)
        depth_map = sample.get("depth_map", None)
        target_trajectory = sample.get("target_trajectory", None)

        # Get DiT's device and dtype (match model parameters for mixed precision)
        dit_device = next(self.dit.parameters()).device
        dit_dtype = next(self.dit.parameters()).dtype
        
        # Move inputs to device, using model's dtype
        action = action.to(device=dit_device, dtype=dit_dtype)
        context = context.to(device=dit_device, dtype=dit_dtype)
        if proprio is not None:
            proprio = proprio.to(device=dit_device, dtype=dit_dtype)
        if state is not None:
            state = state.to(device=dit_device, dtype=dit_dtype)
        if eef_pos is not None:
            eef_pos = eef_pos.to(device=dit_device, dtype=dit_dtype)
        if depth_map is not None:
            depth_map = depth_map.to(device=dit_device, dtype=dit_dtype)
        if target_trajectory is not None:
            target_trajectory = target_trajectory.to(device=dit_device, dtype=dit_dtype)

        with torch.no_grad():
            latents = self.vae.encode(video)
            latents = latents.to(device=dit_device, dtype=dit_dtype)
        
        # Pad latents from 16 to 18 channels to match Cosmos checkpoint
        if latents.shape[1] == 16:
            padding = torch.zeros(latents.shape[0], 2, *latents.shape[2:], 
                                  device=dit_device, dtype=dit_dtype)
            latents = torch.cat([latents, padding], dim=1)
        return {
            "latents": latents,
            "action": action,
            "context": context,
            "proprio": proprio,
            "state": state,
            "eef_pos": eef_pos,
            "depth_map": depth_map,
            "target_trajectory": target_trajectory,
        }

    def training_loss(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        inputs = self.build_inputs(sample)
        latents = inputs["latents"]
        action = inputs["action"]
        context = inputs["context"]
        state = inputs["state"]
        eef_pos = inputs["eef_pos"]
        depth_map = inputs["depth_map"]
        target_trajectory = inputs["target_trajectory"]
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # -------- Latent Query --------
        latent_query = self.latent_query_encoder(eef_pos)

        # -------- Video Rectified Flow --------
        noise_v = torch.randn_like(latents)
        t_video = torch.rand(B, device=device, dtype=torch.float32)

        t_video_bf16 = t_video.view(B, 1, 1, 1, 1).to(dtype=dtype)
        noisy_latents = (1.0 - t_video_bf16) * noise_v + t_video_bf16 * latents
        noisy_latents[:, :, : self.num_cond_frames] = latents[:, :, : self.num_cond_frames].clone()

        # DiT forward with latent query
        pred_v, cond_hidden, noisy_hidden, query_hidden = self.dit(
            x_B_C_T_H_W=noisy_latents,
            timesteps_B_T=t_video.unsqueeze(1).to(dtype=dtype),
            crossattn_emb=context,
            latent_query_tokens=latent_query,
            num_cond_frames=self.num_cond_frames,
        )

        target_v = latents - noise_v
        if self.num_cond_frames > 0:
            loss_video = F.mse_loss(pred_v, target_v[:, :16, self.num_cond_frames :])
        else:
            loss_video = F.mse_loss(pred_v, target_v[:, :16])

        # -------- Trajectory Supervision --------
        loss_traj = self.trajectory_head(query_hidden, target_trajectory)

        # -------- Action iMF Loss --------
        video_ctx = torch.cat([cond_hidden, query_hidden], dim=1).detach()
        loss_action = self.action_head(video_ctx, state, depth_map, action)

        loss = loss_video + self.lambda_traj * loss_traj + self.lambda_action * loss_action
        return loss, {
            "loss_video": loss_video,
            "loss_traj": loss_traj,
            "loss_action": loss_action,
            "loss_total": loss,
        }

    @torch.no_grad()
    def infer_action(
        self,
        first_frame_pixels: torch.Tensor,
        wrist_depth: torch.Tensor,
        state: torch.Tensor,
        eef_pos: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: given first frame, predict action sequence (one-step iMF)."""
        self.eval()
        if first_frame_pixels.ndim == 4:
            first_frame_pixels = first_frame_pixels.unsqueeze(0)
        if wrist_depth.ndim == 3:
            wrist_depth = wrist_depth.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if eef_pos.ndim == 1:
            eef_pos = eef_pos.unsqueeze(0)

        B = first_frame_pixels.shape[0]
        device = first_frame_pixels.device
        dtype = next(self.dit.parameters()).dtype

        # Encode first frame
        latents = self.vae.encode(first_frame_pixels)
        first_frame_latent = latents[:, :, :1, :, :]

        # Pad latents from 16 to 18 channels
        if first_frame_latent.shape[1] == 16:
            padding = torch.zeros(
                first_frame_latent.shape[0], 2,
                *first_frame_latent.shape[2:],
                device=device, dtype=first_frame_latent.dtype
            )
            first_frame_latent = torch.cat([first_frame_latent, padding], dim=1)

        # Latent query
        latent_query = self.latent_query_encoder(eef_pos.to(dtype=dtype))

        # DiT forward (condition only)
        context = context.to(dtype=dtype)
        _, cond_hidden, _, query_hidden = self.dit(
            x_B_C_T_H_W=first_frame_latent.to(dtype=dtype),
            timesteps_B_T=torch.zeros(B, 1, device=device, dtype=dtype),
            crossattn_emb=context,
            latent_query_tokens=latent_query,
            num_cond_frames=self.num_cond_frames,
        )

        video_ctx = torch.cat([cond_hidden, query_hidden], dim=1)

        # Action head one-step prediction (use main action_head, may be bf16 for inference)
        action = self.action_head.predict_action(
            video_ctx, state.to(dtype=dtype), wrist_depth.to(dtype=dtype)
        )
        return action
