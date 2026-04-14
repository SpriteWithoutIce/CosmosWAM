from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import torch
import torch.nn.functional as F

from .models.ckpt_loader import load_dit_from_checkpoint, load_vae_from_checkpoint
from .trainer import CosmosWAMTrainer


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce tensor rank: src={tuple(src.shape)}, target={target_shape}")
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()
    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(f"Resize failed: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}")
    return out.to(dtype=src.dtype)


def run_training(cfg: DictConfig):
    from .models.cosmos_wam import CosmosWAM
    from .models.dit_wrapper import MiniTrainDIT, SACConfig
    from .models.vae_wrapper import Wan2pt1VAEInterface
    from .models.action_head_mot import ActionExpert
    from .models.mot import MoT
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

    # Optional: enable gradient checkpointing on DiT (only used when running video standalone)
    if cfg.model.get("enable_gradient_checkpointing", True):
        from .models.dit_wrapper import enable_selective_checkpoint
        enable_selective_checkpoint(dit, SACConfig(mode="block_wise", every_n_blocks=1), dit.blocks)

    # 3. Build Action Expert
    action_head = ActionExpert(
        action_dim=cfg.model.action_head.action_dim,
        hidden_dim=cfg.model.action_head.hidden_dim,
        num_layers=cfg.model.dit_config.num_blocks,  # must match DiT layers (28)
        num_heads=cfg.model.action_head.num_heads,
        text_dim=cfg.model.dit_config.get("crossattn_proj_in_channels", 100352),
        mlp_ratio=cfg.model.action_head.get("mlp_ratio", 4.0),
        backend=cfg.model.dit_config.get("atten_backend", "minimal_a2a"),
        use_wan_fp32_strategy=cfg.model.dit_config.get("use_wan_fp32_strategy", False),
        adaln_lora_dim=cfg.model.dit_config.get("adaln_lora_dim", 256),
    )

    # Initialize action expert blocks from video DiT weights via interpolation
    if not resume_ckpt_path:
        dit_state = dit.state_dict()
        action_state = action_head.state_dict()
        copied = 0
        interpolated = 0
        for key in action_state.keys():
            if not key.startswith("blocks."):
                continue
            if key not in dit_state:
                continue
            src = dit_state[key]
            target = action_state[key]
            if tuple(src.shape) == tuple(target.shape):
                action_state[key] = src.clone()
                copied += 1
            else:
                try:
                    resized = _resize_tensor_to_shape(src, tuple(target.shape))
                    action_state[key] = resized
                    interpolated += 1
                except ValueError as e:
                    print(f"[runtime] Warning: could not resize {key} from {tuple(src.shape)} to {tuple(target.shape)}: {e}")
        action_head.load_state_dict(action_state, strict=False)
        print(f"[runtime] Initialized action expert from DiT: copied={copied}, interpolated={interpolated}")

    # 4. Build MoT
    mot = MoT(
        mixtures={"video": dit, "action": action_head},
        mot_checkpoint_mixed_attn=cfg.model.get("mot_checkpoint_mixed_attn", True),
    )

    # 5. Build CosmosWAM
    model = CosmosWAM(
        mot=mot,
        vae=vae,
        lambda_action=cfg.model.get("lambda_action", 1.0),
        num_cond_frames=cfg.model.get("num_cond_frames", 1),
        actions_per_latent=cfg.model.action_head.get("actions_per_latent", 8),
    )

    # 5.4 Print model size summary
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    def count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Parameter Summary")
    print("="*60)
    
    # Video model (DiT)
    dit_params = count_params(model.dit)
    dit_trainable = count_trainable(model.dit)
    print(f"Video Model (DiT):        {dit_params/1e9:>8.3f}B params  (trainable: {dit_trainable/1e9:>8.3f}B)")
    
    # Action head
    action_params = count_params(model.action_head)
    action_trainable = count_trainable(model.action_head)
    print(f"Action Head:              {action_params/1e9:>8.3f}B params  (trainable: {action_trainable/1e9:>8.3f}B)")
    
    # VAE (frozen)
    vae_params = count_params(model.vae)
    print(f"VAE (frozen):             {vae_params/1e9:>8.3f}B params")
    
    # Total
    total_params = count_params(model)
    total_trainable = count_trainable(model)
    print("-"*60)
    print(f"Total:                    {total_params/1e9:>8.3f}B params  (trainable: {total_trainable/1e9:>8.3f}B)")
    print("="*60 + "\n")
    
    # 5.5 Compile model with torch.compile for H100 optimization
    if cfg.model.get("compile", False):
        print("[runtime] Compiling model with torch.compile (mode='max-autotune')...")
        model.dit = torch.compile(model.dit, mode="max-autotune", fullgraph=False)
        model.action_head = torch.compile(model.action_head, mode="max-autotune", fullgraph=False)
        print("[runtime] Model compilation done")

    # 6. Build Datasets
    train_dataset = instantiate(cfg.data.train)
    val_dataset = instantiate(cfg.data.get("val", None)) if "val" in cfg.data else None

    # 7. Train
    trainer = CosmosWAMTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, cfg=cfg)
    trainer.train()
