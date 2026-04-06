#!/usr/bin/env python3
"""Test dataset loading."""

import sys
sys.path.insert(0, '/home/jwhe/linyihan/CosmosWAM')

import torch
from cosmos_wam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from cosmos_wam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from cosmos_wam.datasets.lerobot.transforms.image import ToTensor
from torchvision.transforms import Resize

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

# Create processor
processor = FastWAMProcessor(
    shape_meta=shape_meta,
    num_obs_steps=33,
    num_output_cameras=1,
    action_output_dim=16,
    proprio_output_dim=16,
    action_state_transforms=None,
    use_stepwise_action_norm=False,
    norm_default_mode="min/max",
    norm_exception_mode=None,
    action_state_merger=None,  # Will be set below
    train_transforms={
        "cam_high": [ToTensor(), Resize([240, 320])]
    },
    val_transforms={
        "cam_high": [ToTensor(), Resize([240, 320])]
    },
)

# Test dataset initialization
try:
    dataset = RobotVideoDataset(
        dataset_dirs=["./data/lerobot_robotwin_eef_clean_50/close_the_box"],  # Replace with actual path
        shape_meta=shape_meta,
        num_frames=33,
        video_size=[240, 320],
        text_embedding_cache_dir="./data/text_embeds_cache/robotwin_reason1",
        context_len=512,
        val_set_proportion=0.0,
        is_training_set=True,
        concat_multi_camera="horizontal",
        processor=processor,
    )
    print(f"Dataset initialized successfully! Length: {len(dataset)}")
    
    # Test getitem
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Proprio shape: {sample['proprio'].shape}")
    print(f"Context shape: {sample['context'].shape}")
    print(f"Context mask shape: {sample['context_mask'].shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
