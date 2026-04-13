#!/usr/bin/env python3
"""Debug script to check action statistics in the dataset."""

import sys
sys.path.insert(0, '/home/jwhe/linyihan/LIBERO')

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import numpy as np
from tqdm import tqdm

@hydra.main(config_path="../configs", config_name="train_cosmos_2b_libero", version_base="1.3")
def main(cfg: DictConfig):
    print("Loading dataset...")
    train_dataset = instantiate(cfg.data.train)
    
    # Collect action statistics
    all_actions = []
    print(f"Dataset size: {len(train_dataset)}")
    
    # Sample 1000 batches to check
    num_samples = min(1000, len(train_dataset))
    for i in tqdm(range(num_samples)):
        sample = train_dataset[i]
        action = sample["action"]  # [T, action_dim]
        all_actions.append(action.numpy())
    
    all_actions = np.concatenate(all_actions, axis=0)  # [N*T, action_dim]
    
    print(f"\nAction shape: {all_actions.shape}")
    print(f"\nAction statistics per dimension:")
    for dim in range(all_actions.shape[1]):
        dim_values = all_actions[:, dim]
        print(f"  Dim {dim}: mean={dim_values.mean():.4f}, std={dim_values.std():.4f}, "
              f"min={dim_values.min():.4f}, max={dim_values.max():.4f}")
    
    # Check for outliers
    print(f"\nOutlier check (values beyond 3 std):")
    for dim in range(all_actions.shape[1]):
        dim_values = all_actions[:, dim]
        mean, std = dim_values.mean(), dim_values.std()
        outliers = np.abs(dim_values - mean) > 3 * std
        outlier_pct = outliers.mean() * 100
        print(f"  Dim {dim}: {outlier_pct:.2f}% outliers")

if __name__ == "__main__":
    main()
