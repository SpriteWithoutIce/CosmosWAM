#!/usr/bin/env python3
"""Inference script for Cosmos-WAM model."""

import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_wam.models.cosmos_wam import CosmosWAM
from cosmos_wam.models.dit_wrapper import MiniTrainDIT, SACConfig
from cosmos_wam.models.vae_wrapper import Wan2pt1VAEInterface
from cosmos_wam.models.action_head import ActionDiT
from cosmos_wam.models.ckpt_loader import load_dit_from_checkpoint


def load_model_from_checkpoint(ckpt_path: str, cfg) -> CosmosWAM:
    """Load trained model from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file (e.g., outputs/checkpoints/step_0001000.pt)
        cfg: Model configuration
    
    Returns:
        Loaded CosmosWAM model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Build VAE (frozen, same as training)
    vae = Wan2pt1VAEInterface(
        vae_pth=cfg.model.vae_checkpoint,
        temporal_window=cfg.model.get("vae_temporal_window", 16),
    )
    vae.eval()
    
    # 2. Build DiT
    dit = MiniTrainDIT(
        max_img_h=cfg.model.dit_config.max_img_h,
        max_img_w=cfg.model.dit_config.max_img_w,
        max_frames=cfg.model.dit_config.max_frames,
        in_channels=cfg.model.dit_config.in_channels,
        out_channels=cfg.model.dit_config.out_channels,
        patch_spatial=cfg.model.dit_config.patch_spatial,
        patch_temporal=cfg.model.dit_config.patch_temporal,
        model_channels=cfg.model.dit_config.model_channels,
        num_blocks=cfg.model.dit_config.num_blocks,
        num_heads=cfg.model.dit_config.num_heads,
        mlp_ratio=cfg.model.dit_config.get("mlp_ratio", 4.0),
        atten_backend=cfg.model.dit_config.get("atten_backend", "minimal_a2a"),
        crossattn_emb_channels=cfg.model.dit_config.crossattn_emb_channels,
        use_crossattn_projection=cfg.model.dit_config.get("use_crossattn_projection", False),
        crossattn_proj_in_channels=cfg.model.dit_config.get("crossattn_proj_in_channels", 1024),
        pos_emb_cls=cfg.model.dit_config.get("pos_emb_cls", "rope3d"),
        pos_emb_learnable=cfg.model.dit_config.get("pos_emb_learnable", False),
        rope_h_extrapolation_ratio=cfg.model.dit_config.get("rope_h_extrapolation_ratio", 1.0),
        rope_w_extrapolation_ratio=cfg.model.dit_config.get("rope_w_extrapolation_ratio", 1.0),
        rope_t_extrapolation_ratio=cfg.model.dit_config.get("rope_t_extrapolation_ratio", 1.0),
        use_wan_fp32_strategy=cfg.model.dit_config.get("use_wan_fp32_strategy", False),
        adaln_lora_dim=cfg.model.dit_config.get("adaln_lora_dim", 256),
        use_t_embedding_adaln_lora=cfg.model.dit_config.get("use_t_embedding_adaln_lora", True),
    )
    
    # 3. Build Action Head
    action_head = ActionDiT(
        action_dim=cfg.model.action_head.action_dim,
        hidden_dim=cfg.model.action_head.hidden_dim,
        num_layers=cfg.model.action_head.num_layers,
        num_heads=cfg.model.action_head.num_heads,
        video_dim=cfg.model.action_head.video_dim,
        mlp_ratio=cfg.model.action_head.get("mlp_ratio", 4.0),
        actions_per_latent=cfg.model.action_head.get("actions_per_latent", 8),
    )
    
    # 4. Build CosmosWAM
    model = CosmosWAM(
        dit=dit,
        vae=vae,
        action_head=action_head,
        lambda_action=cfg.model.get("lambda_action", 1.0),
        num_cond_frames=cfg.model.get("num_cond_frames", 1),
    )
    
    # 5. Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Load model weights
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"Warning: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    
    print(f"Loaded checkpoint from step {ckpt.get('step', 'unknown')}")
    
    model = model.to(device)
    model.eval()
    return model, ckpt.get("step", 0)


@torch.no_grad()
def infer_action(
    model: CosmosWAM,
    first_frame: torch.Tensor,  # [B, 3, H, W] or [3, H, W]
    text_embedding: torch.Tensor,  # [B, L, D] or [L, D]
    num_steps: int = 20,
    action_horizon: int = 32,
) -> torch.Tensor:
    """Inference: given first frame, predict action sequence.
    
    Args:
        model: Loaded CosmosWAM model
        first_frame: First frame image [B, 3, H, W] or [3, H, W]
        text_embedding: Text embedding for conditioning [B, L, D] or [L, D]
        num_steps: Number of diffusion steps for action denoising
        action_horizon: Length of action sequence to predict
    
    Returns:
        Predicted actions [B, action_horizon, action_dim]
    """
    device = next(model.parameters()).device
    
    # Add batch dimension if needed
    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if text_embedding.ndim == 2:
        text_embedding = text_embedding.unsqueeze(0)
    
    B = first_frame.shape[0]
    
    # Encode first frame through VAE to get latent
    with torch.no_grad():
        # VAE expects [B, 3, T, H, W], we have [B, 3, H, W]
        first_frame_5d = first_frame.unsqueeze(2)  # [B, 3, 1, H, W]
        latent = model.vae.encode(first_frame_5d)  # [B, C_latent, 1, H_latent, W_latent]
        # Pad to 18 channels
        if latent.shape[1] == 16:
            padding = torch.zeros_like(latent[:, :2])
            latent = torch.cat([latent, padding], dim=1)
    
    # Use model's inference method
    actions = model.infer_action(
        first_frame_pixels=first_frame,
        action_horizon=action_horizon,
        context=text_embedding,
        num_inference_steps=num_steps,
    )
    
    return actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_cosmos_2b_robotwin.yaml")
    parser.add_argument("--text", type=str, default="Pick up the bottle", help="Task instruction")
    parser.add_argument("--image", type=str, help="Path to first frame image")
    args = parser.parse_args()
    
    # Load config
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    GlobalHydra.instance().clear()
    config_dir = str(Path(args.config).parent.absolute())
    config_name = Path(args.config).stem
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    
    # Load model
    model, step = load_model_from_checkpoint(args.checkpoint, cfg)
    print(f"Model loaded (step {step}), ready for inference")
    
    # TODO: Load actual image and text embedding
    # For now, just print usage
    print("\nExample usage:")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  config: {args.config}")
    print(f"  text: {args.text}")


if __name__ == "__main__":
    main()
