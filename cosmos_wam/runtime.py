from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import torch

from .models.ckpt_loader import load_dit_from_checkpoint, load_vae_from_checkpoint
from .trainer import CosmosWAMTrainer


def run_training(cfg: DictConfig):
    from .models.cosmos_wam import CosmosWAM
    from .models.dit_wrapper import MiniTrainDIT, SACConfig
    from .models.vae_wrapper import Wan2pt1VAEInterface
    from .models.latent_query import LatentQueryEncoder, TrajectoryHead
    from .models.action_head import ActionHeadIMF
    from datetime import datetime

    # Add timestamp to output_dir for unique experiment identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = cfg.trainer.output_dir
    cfg.trainer.output_dir = f"{base_output_dir}_{timestamp}"
    
    # Detect local rank for multi-GPU setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # 1. Build VAE - each rank has its own VAE on its own GPU
    vae = Wan2pt1VAEInterface(
        vae_pth=cfg.model.vae_checkpoint,
        temporal_window=cfg.model.get("vae_temporal_window", 16),
        device=device,
    )
    vae.eval()  # VAE is always frozen

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
    # Load pre-trained DiT weights only if not resuming from a training checkpoint
    resume_ckpt_path = cfg.trainer.get("resume_from_checkpoint", None)
    if not resume_ckpt_path:
        load_dit_from_checkpoint(dit, cfg.model.dit_checkpoint, strict=False)
    else:
        print(f"[runtime] Skipping pre-trained DiT loading, will resume from {resume_ckpt_path}")

    # 3. Build Latent Query Encoder
    latent_query_encoder = LatentQueryEncoder(
        hidden_dim=cfg.model.latent_query.get("hidden_dim", 2048),
        num_queries=cfg.model.latent_query.get("num_queries", 32),
        sigma=cfg.model.latent_query.get("sigma", 8.0),
        camera_params_path=cfg.model.latent_query.get("camera_params_path", None),
        image_height=cfg.model.latent_query.get("image_height", 224),
        image_width=cfg.model.latent_query.get("image_width", 224),
    )

    # 4. Build Trajectory Head
    trajectory_head = TrajectoryHead(
        hidden_dim=cfg.model.trajectory_head.get("hidden_dim", 2048),
    )

    # 5. Build Action Head (iMF)
    action_head = ActionHeadIMF(
        hidden_dim=cfg.model.action_head.hidden_dim,
        action_dim=cfg.model.action_head.action_dim,
        state_dim=cfg.model.action_head.get("state_dim", 8),
        action_horizon=cfg.model.action_head.get("action_horizon", 32),
        num_layers=cfg.model.action_head.num_layers,
        num_heads=cfg.model.action_head.num_heads,
        cross_attention_dim=cfg.model.action_head.get("cross_attention_dim", 768),
        video_ctx_dim=cfg.model.action_head.get("video_ctx_dim", 2048),
        dropout=cfg.model.action_head.get("dropout", 0.1),
        final_dropout=cfg.model.action_head.get("final_dropout", True),
    )

    # 6. Build CosmosWAM
    model = CosmosWAM(
        dit=dit,
        vae=vae,
        latent_query_encoder=latent_query_encoder,
        trajectory_head=trajectory_head,
        action_head=action_head,
        lambda_action=cfg.model.get("lambda_action", 1.0),
        lambda_traj=cfg.model.get("lambda_traj", 1.0),
        num_cond_frames=cfg.model.get("num_cond_frames", 1),
    )

    # ActionHeadIMF uses torch.func.jvp which is fundamentally incompatible with bf16
    # in PyTorch functorch. Maintain a separate fp32 copy for training, outside DeepSpeed.
    action_head_train = ActionHeadIMF(
        hidden_dim=cfg.model.action_head.hidden_dim,
        action_dim=cfg.model.action_head.action_dim,
        state_dim=cfg.model.action_head.get("state_dim", 8),
        action_horizon=cfg.model.action_head.get("action_horizon", 32),
        num_layers=cfg.model.action_head.num_layers,
        num_heads=cfg.model.action_head.num_heads,
        cross_attention_dim=cfg.model.action_head.get("cross_attention_dim", 768),
        video_ctx_dim=cfg.model.action_head.get("video_ctx_dim", 2048),
        dropout=cfg.model.action_head.get("dropout", 0.1),
        final_dropout=cfg.model.action_head.get("final_dropout", True),
    ).float()
    action_head_train.load_state_dict(action_head.state_dict())
    print("[runtime] ActionHeadIMF fp32 training copy created (outside DeepSpeed)")

    # Enable gradient checkpointing if configured
    if cfg.model.get("enable_gradient_checkpointing", False):
        model.dit.gradient_checkpointing = True
        print("[runtime] Gradient checkpointing enabled for DiT")

    # 7. Build Datasets
    train_dataset = instantiate(cfg.data.train)
    val_dataset = instantiate(cfg.data.get("val", None)) if "val" in cfg.data else None

    # 8. Train
    trainer = CosmosWAMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        action_head_train=action_head_train,
        cfg=cfg,
    )
    trainer.train()
