#!/usr/bin/env python3
"""Test dataset loading."""

import sys
sys.path.insert(0, '/home/jwhe/linyihan/CosmosWAM')

from cosmos_wam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from omegaconf import OmegaConf

# Test config
shape_meta = {
    "images": [
        {"key": "cam_high", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]}
    ],
    "action": [
        {"key": "default", "raw_shape": 16, "shape": 16}
    ],
    "state": [
        {"key": "default", "raw_shape": 16, "shape": 16}
    ]
}

# Test dataset initialization
try:
    dataset = RobotVideoDataset(
        dataset_dirs=["/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50"],  # Replace with actual path
        shape_meta=shape_meta,
        num_frames=33,
        video_size=[240, 320],
        text_embedding_cache_dir="/home/jwhe/linyihan/datasets/text_embeds_cache",
        context_len=512,
        val_set_proportion=0.0,
        is_training_set=True,
        concat_multi_camera="horizontal",
    )
    print(f"Dataset initialized successfully! Length: {len(dataset)}")
    
    # Test getitem
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Context shape: {sample['context'].shape}")
    print(f"Context mask shape: {sample['context_mask'].shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
