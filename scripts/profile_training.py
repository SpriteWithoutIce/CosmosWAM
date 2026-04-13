#!/usr/bin/env python3
"""Profile training to find bottlenecks."""

import sys
sys.path.insert(0, '/home/jwhe/linyihan/LIBERO')

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import time
from tqdm import tqdm

@hydra.main(config_path="../configs", config_name="train_cosmos_2b_libero", version_base="1.3")
def main(cfg: DictConfig):
    # Limit to single GPU for profiling
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("Building model...")
    from cosmos_wam.models.cosmos_wam import CosmosWAM
    from cosmos_wam.models.dit_wrapper import MiniTrainDIT, SACConfig
    from cosmos_wam.models.vae_wrapper import Wan2pt1VAEInterface
    from cosmos_wam.models.action_head import ActionDiT
    
    device = torch.device("cuda:0")
    
    # Build VAE
    vae = Wan2pt1VAEInterface(
        vae_pth=cfg.model.vae_checkpoint,
        temporal_window=cfg.model.get("vae_temporal_window", 16),
        device=device,
    )
    vae.eval()
    
    # Build DiT
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
    
    # Build Action Head
    action_head = ActionDiT(
        action_dim=cfg.model.action_head.action_dim,
        hidden_dim=cfg.model.action_head.hidden_dim,
        num_layers=cfg.model.action_head.num_layers,
        num_heads=cfg.model.action_head.num_heads,
        video_dim=cfg.model.action_head.video_dim,
        mlp_ratio=cfg.model.action_head.get("mlp_ratio", 4.0),
        actions_per_latent=cfg.model.action_head.get("actions_per_latent", 8),
    )
    
    # Build CosmosWAM
    model = CosmosWAM(
        dit=dit,
        vae=vae,
        action_head=action_head,
        lambda_action=cfg.model.get("lambda_action", 1.0),
        num_cond_frames=cfg.model.get("num_cond_frames", 1),
    ).to(device)
    
    print("Loading dataset...")
    train_dataset = instantiate(cfg.data.train)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        betas=(0.9, 0.95),
    )
    
    print("\n" + "="*60)
    print("Profiling 10 steps...")
    print("="*60)
    
    model.train()
    times = {
        "data_loading": [],
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }
    
    iterator = iter(train_loader)
    
    # Warmup
    for _ in range(2):
        batch = next(iterator)
        loss, _ = model.training_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    for step in range(10):
        total_start = time.time()
        
        # Data loading time
        data_start = time.time()
        batch = next(iterator)
        torch.cuda.synchronize()
        data_time = time.time() - data_start
        times["data_loading"].append(data_time)
        
        # Forward time
        fwd_start = time.time()
        loss, loss_dict = model.training_loss(batch)
        torch.cuda.synchronize()
        fwd_time = time.time() - fwd_start
        times["forward"].append(fwd_time)
        
        # Backward time
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - bwd_start
        times["backward"].append(bwd_time)
        
        # Optimizer time
        opt_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_time = time.time() - opt_start
        times["optimizer"].append(opt_time)
        
        total_time = time.time() - total_start
        times["total"].append(total_time)
        
        print(f"Step {step+1}: total={total_time:.3f}s | "
              f"data={data_time:.3f}s | fwd={fwd_time:.3f}s | "
              f"bwd={bwd_time:.3f}s | opt={opt_time:.3f}s | "
              f"loss={loss_dict['loss_total']:.4f}")
    
    print("\n" + "="*60)
    print("Average times (excluding first step):")
    print("="*60)
    for key, vals in times.items():
        avg = sum(vals[1:]) / len(vals[1:])  # Exclude first step
        print(f"  {key:15s}: {avg:.3f}s ({avg/sum(times['total'][1:])*100:.1f}%)")
    
    print(f"\nThroughput: {cfg.trainer.batch_size / (sum(times['total'][1:])/len(times['total'][1:])):.2f} samples/sec")
    print(f"Steps/sec: {1 / (sum(times['total'][1:])/len(times['total'][1:])):.2f}")

if __name__ == "__main__":
    main()
